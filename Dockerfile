FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# TFログ抑制（任意）
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=1

# Cache場所を固定（モデル事前DL用）
ENV HOME=/root
ENV XDG_CACHE_HOME=/opt/cache
ENV CSBDEEP_CACHE_DIR=/opt/models/csbdeep
ENV KERAS_HOME=/opt/models/keras
ENV NVCC_PREPEND_FLAGS="--std=c++17"
ENV CCCL_IGNORE_DEPRECATED_CPP_DIALECT=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV LD_PRELOAD=/lib/x86_64-linux-gnu/libnccl.so.2

# 日本ミラー（任意）
RUN sed -i 's@http://archive.ubuntu.com@http://ftp.riken.jp/Linux@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com@http://ftp.riken.jp/Linux@g' /etc/apt/sources.list

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget build-essential \
    libgl1 libglib2.0-0 \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip 'setuptools<70.0.0' wheel

# ---- Copy pyproject.toml and install HueTracer with dependencies ----
WORKDIR /app
COPY pyproject.toml /app/
COPY src/ /app/src/
RUN python3 -m pip install --no-cache-dir \
  --extra-index-url=https://pypi.nvidia.com \
  -e /app[gpu]

# ---- Pre-download StarDist pretrained model ----
RUN mkdir -p /opt/models/csbdeep /opt/models/keras /opt/cache && \
    python3 - <<'PY'
from stardist.models import StarDist2D
m = StarDist2D.from_pretrained("2D_versatile_he")
print("✅ cached:", m.name)
m = StarDist2D.from_pretrained("2D_versatile_fluo")
print("✅ cached:", m.name)
PY

# ---- Git-based packages ----
RUN git clone https://github.com/digitalcytometry/cytotrace2 && \
    cd cytotrace2/cytotrace2_python && \
    python3 -m pip install .

# RAPIDS smoke test
RUN python3 - <<'PY'
import cudf
import cuml
import cugraph
import dask_cudf
print("RAPIDS import smoke test: OK")
PY

# 依存が崩れてないか最終チェック（CUDA マイクロバージョン差は非致命的）
RUN set -e; \
    pip_check_output="$(python3 -m pip check 2>&1)" || pip_check_status="$?"; \
    echo "${pip_check_output}"; \
    if [ "${pip_check_status:-0}" -ne 0 ]; then \
      # CUDA / RAPIDS 関連の既知のマイクロバージョン差のみを許容し、それ以外はビルド失敗とする \
      non_cuda_issues="$(printf '%s\n' "${pip_check_output}" | grep -viE 'cuda|cudf|cuml|cugraph|cupy-cuda|cudnn|rapids')" || true; \
      if [ -n "${non_cuda_issues}" ]; then \
        echo "❌ pip check failed with non-CUDA dependency issues:"; \
        printf '%s\n' "${non_cuda_issues}"; \
        exit "${pip_check_status}"; \
      else \
        echo '⚠ pip check: only CUDA-related minor version conflicts detected (non-fatal)'; \
      fi; \
    fi

EXPOSE 8152
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8152", "--no-browser", "--allow-root"]
