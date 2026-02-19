#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path

# --- TF tuning: set BEFORE importing tensorflow ---
def set_tf_env(args):
    # CPU-only TF (you said TF can stay CPU)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force TF CPU (Torch can still use GPU in same container)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", str(args.tf_log_level))

    # oneDNN: ON by default usually faster on Intel, but keep switchable
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1" if args.onednn else "0")

    # Threading: Make it deterministic for docker
    os.environ.setdefault("OMP_NUM_THREADS", str(args.omp_threads))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(args.intra))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(args.inter))

    # Disable XLA JIT (often not helpful for this pipeline)
    if args.disable_xla:
        os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")


def sh(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"<failed: {e}>"


def log_system_info():
    info = {}
    info["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["uname"] = sh("uname -a")
    info["nproc"] = sh("nproc")
    info["cpuset"] = sh("cat /sys/fs/cgroup/cpuset.cpus 2>/dev/null || true")
    info["cpuset_effective"] = sh("cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true")
    info["cpu_max"] = sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null || true")
    info["governor"] = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true")
    info["cur_freq"] = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || true")
    info["max_freq"] = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || true")
    info["min_freq"] = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 2>/dev/null || true")
    info["intel_pstate"] = sh("cat /sys/devices/system/cpu/intel_pstate/status 2>/dev/null || true")
    info["df_shm"] = sh("df -h /dev/shm 2>/dev/null || true")
    return info


def timeit(label, fn):
    t0 = time.perf_counter()
    ret = fn()
    dt = time.perf_counter() - t0
    return ret, dt


def load_image(path, roi=None):
    import tifffile
    import numpy as np

    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()

    if roi is not None:
        x, y, w, h = roi
        arr = arr[y:y+h, x:x+w]

    # Ensure contiguous (NMS / TF likes it)
    arr = np.ascontiguousarray(arr)
    return arr


def parse_roi(s):
    # "x,y,w,h"
    if s is None:
        return None
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 'x,y,w,h'")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he_tiff", required=True, help="Path to HE_full_path (e.g., /app/results/P1_CRC/tmp/he.tiff)")
    ap.add_argument("--out_npz", required=True, help="Output labels npz path (e.g., /app/results/P1_CRC/tmp/he.npz)")
    ap.add_argument("--repeats", type=int, default=3, help="Number of full runs")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs (not counted)")
    ap.add_argument("--roi", type=str, default=None, help="Optional ROI as x,y,w,h (makes it faster but repeatable).")
    ap.add_argument("--model", default="2D_versatile_he")
    ap.add_argument("--prob", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=None, help="Override NMS threshold (optional)")
    ap.add_argument("--block", type=int, default=4096, help="block_size for predict_instances_big (pixels)")
    ap.add_argument("--overlap", type=int, default=128, help="min_overlap")
    ap.add_argument("--context", type=int, default=128, help="context")
    ap.add_argument("--keep_labels", action="store_true", help="Keep labels from each run (otherwise overwrite)")
    ap.add_argument("--no_io_cache", action="store_true",
                    help="Try to drop OS cache before each run (needs privileges; best-effort)")
    # TF env knobs
    ap.add_argument("--omp_threads", type=int, default=32)
    ap.add_argument("--intra", type=int, default=32)
    ap.add_argument("--inter", type=int, default=2)
    ap.add_argument("--onednn", action="store_true", help="Enable oneDNN ops (recommended on Intel)")
    ap.add_argument("--disable_xla", action="store_true", help="Disable TF XLA auto_jit")
    ap.add_argument("--tf_log_level", type=int, default=2, help="TF_CPP_MIN_LOG_LEVEL (0-3)")
    args = ap.parse_args()

    set_tf_env(args)

    # Now imports (after env)
    import numpy as np
    import tensorflow as tf
    import bin2cell as b2c

    roi = parse_roi(args.roi)

    print("=== System ===")
    sysinfo = log_system_info()
    print(json.dumps(sysinfo, indent=2, ensure_ascii=False))

    print("\n=== TensorFlow ===")
    print("tf.__version__:", tf.__version__)
    print("tf devices:", tf.config.list_physical_devices())
    print("build_info:", tf.sysconfig.get_build_info())
    print("intra:", tf.config.threading.get_intra_op_parallelism_threads(),
          "inter:", tf.config.threading.get_inter_op_parallelism_threads())
    print("env OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    print("env TF_NUM_INTRAOP_THREADS:", os.environ.get("TF_NUM_INTRAOP_THREADS"))
    print("env TF_NUM_INTEROP_THREADS:", os.environ.get("TF_NUM_INTEROP_THREADS"))
    print("env TF_ENABLE_ONEDNN_OPTS:", os.environ.get("TF_ENABLE_ONEDNN_OPTS"))
    print("env TF_XLA_FLAGS:", os.environ.get("TF_XLA_FLAGS"))

    he_path = Path(args.he_tiff)
    out_npz = Path(args.out_npz)
    assert he_path.exists(), f"HE TIFF not found: {he_path}"

    # Show image metadata once
    arr, dt_load = timeit("load_image", lambda: load_image(str(he_path), roi=roi))
    print("\n=== Image ===")
    print("loaded in sec:", round(dt_load, 4))
    print("shape:", arr.shape, "dtype:", arr.dtype, "min/max:", int(arr.min()), int(arr.max()))
    del arr

    def maybe_drop_cache():
        if not args.no_io_cache:
            return
        # best-effort: this needs privileged container / root with permission
        sh("sync")
        sh("echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true")

    # Warmup (model download / graph build / first tracing)
    print("\n=== Warmup ===")
    for i in range(args.warmup):
        maybe_drop_cache()
        t0 = time.perf_counter()
        b2c.stardist(
            image_path=str(he_path),
            labels_npz_path=str(out_npz),
            stardist_model=args.model,
            prob_thresh=args.prob,
            block_size=args.block,
            min_overlap=args.overlap,
            context=args.context,
            **({} if args.nms is None else {"nms_thresh": args.nms})
        )
        dt = time.perf_counter() - t0
        print(f"warmup {i+1}/{args.warmup} sec:", round(dt, 3))

    # Bench runs
    print("\n=== Benchmark ===")
    results = []
    for r in range(args.repeats):
        maybe_drop_cache()

        # overwrite or versioned output
        npz_path = out_npz
        if args.keep_labels:
            npz_path = out_npz.with_name(out_npz.stem + f".run{r+1}.npz")

        t0 = time.perf_counter()
        b2c.stardist(
            image_path=str(he_path),
            labels_npz_path=str(npz_path),
            stardist_model=args.model,
            prob_thresh=args.prob,
            block_size=args.block,
            min_overlap=args.overlap,
            context=args.context,
            **({} if args.nms is None else {"nms_thresh": args.nms})
        )
        dt = time.perf_counter() - t0

        size_mb = npz_path.stat().st_size / (1024 * 1024) if npz_path.exists() else None
        results.append({"run": r+1, "sec": dt, "npz_mb": size_mb})
        print(f"run {r+1}/{args.repeats} sec:", round(dt, 3), "npz_mb:", None if size_mb is None else round(size_mb, 2))

    print("\n=== Summary ===")
    secs = [x["sec"] for x in results]
    print("mean sec:", round(float(np.mean(secs)), 3))
    print("median sec:", round(float(np.median(secs)), 3))
    print("min/max sec:", round(float(np.min(secs)), 3), "/", round(float(np.max(secs)), 3))
    print("raw:", json.dumps(results, indent=2))

    print("\nTIP:")
    print("- block_size を 2048/4096/8192 で振って、どこで遅くなる or エラーになるか確認すると原因切り分けが速いです。")
    print("- Dockerだけ遅い場合は governor=powersave が大本命。ホスト側で performance にすると改善しやすいです。")


if __name__ == "__main__":
    main()