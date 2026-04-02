"""
Nucleus segmentation and label processing utilities.
Contains StarDist-based segmentation, label expansion, and GEX image generation.
"""

import os
from typing import Any, Optional

import numpy as np
import scipy.sparse


def segment_nuclei_stardist(
    image_path: str,
    output_npz_path: str,
    model_type: str = "he",
    prob_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    block_size: int = 4096,
    context: int = 128,
    norm_min: int = 3,
    norm_max: int = 99.8,
) -> tuple[np.ndarray, dict]:
    """
    Segment nuclei using StarDist2D model (HE or GEX mode).

    Handles: image normalization, StarDist prediction, sparse matrix save.
    """

    def normalize(img, pmin=3, pmax=99.8, axis=None):
        mi = np.percentile(img, pmin, axis=axis, keepdims=True)
        ma = np.percentile(img, pmax, axis=axis, keepdims=True)
        img = (img - mi) / (ma - mi)
        return np.clip(img, 0, 1)

    try:
        import tifffile
        from stardist.models import StarDist2D
    except ImportError as e:
        raise ImportError(f"Required packages missing: {e}")

    print(f"🚀 Loading image: {image_path}")
    img = tifffile.imread(image_path)
    print(f"   Shape: {img.shape}")

    print(f"🧪 Normalizing ({norm_min}% - {norm_max}%)...")
    if model_type == "he":
        img_norm = normalize(img, norm_min, norm_max, axis=(0, 1))
        axes = "YXC"
    else:
        img_norm = normalize(img, norm_min, norm_max, axis=None)
        axes = "YX"

    model_name = "2D_versatile_he" if model_type == "he" else "2D_versatile_fluo"
    print(f"🤖 Loading StarDist model: {model_name}")
    model = StarDist2D.from_pretrained(model_name)

    print(f"📊 Running prediction (block_size={block_size}, context={context})...")
    labels, details = model.predict_instances_big(
        img_norm.astype("float32"),
        axes=axes,
        block_size=block_size,
        min_overlap=context,
        context=context,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh if model_type == "he" else nms_thresh / 5,
        scale=1.0,
        nms_kwargs={"use_kdtree": True, "verbose": False},
    )

    n_detected = len(details.get("points", []))
    print(f"✅ Detection complete! {n_detected} objects detected.")

    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    if os.path.exists(output_npz_path):
        os.remove(output_npz_path)

    labels_sparse = scipy.sparse.csr_matrix(labels.astype("int32"))
    scipy.sparse.save_npz(output_npz_path, labels_sparse)
    print(f"💾 Saved: {output_npz_path}")

    return labels, details


def expand_nuclei_labels(
    adata: Any,
    labels_key: str = "labels_he",
    max_bin_distance: int = 2,
    expanded_labels_key: str = "labels_he_expanded",
    b2c_module: Optional[Any] = None,
) -> None:
    """Expand nuclei labels to neighboring bins."""
    if b2c_module is None:
        import bin2cell as b2c_module

    if labels_key not in adata.obs:
        raise ValueError(f"'{labels_key}' not found in adata.obs columns")

    print(f"🔄 Expanding labels from '{labels_key}' -> '{expanded_labels_key}'")
    print(f"   Max bin distance: {max_bin_distance}")

    b2c_module.expand_labels(
        adata,
        labels_key=labels_key,
        max_bin_distance=max_bin_distance,
        expanded_labels_key=expanded_labels_key,
    )

    expanded_count = np.sum(adata.obs[expanded_labels_key] > 0)
    print(f"✅ Expansion complete: {expanded_count} spots with labels")


def generate_gex_segment_image(
    adata: Any,
    value_key: str = "n_counts_adjusted",
    mpp: float = 0.5,
    sigma: float = 5,
    output_path: Optional[str] = None,
    b2c_module: Optional[Any] = None,
) -> str:
    """Generate synthetic grid image from gene expression data."""
    if b2c_module is None:
        import bin2cell as b2c_module

    if output_path is None:
        raise ValueError("output_path is required")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"🎨 Generating grid image from '{value_key}'...")
    print(f"   MPP: {mpp}, Sigma: {sigma}")

    b2c_module.grid_image(
        adata=adata,
        val=value_key,
        mpp=mpp,
        sigma=sigma,
        save_path=output_path,
    )

    print(f"✅ Grid image saved: {output_path}")
    return output_path
