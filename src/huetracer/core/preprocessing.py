"""
Bin2Cell preprocessing pipeline and spatial dataset utilities.
Contains crop-aware AnnData updates and Visium loading/preprocessing.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class CropRegion:
    """Crop region specification."""
    x: int
    y: int
    width: int
    height: int


def dummy_image_shape_check(_: Any, __: Any) -> None:
    """Dummy check to bypass bin2cell image dimension validation."""
    print("ℹ️ Skipping image dimension check (Custom Crop Mode)")


def install_bin2cell_safety_override(b2c_module: Optional[Any] = None) -> bool:
    """
    Install safety override to skip bin2cell image dimension checking.

    This is necessary when using cropped images where dimensions may not match
    the inferred dimensions from expression data.

    Args:
        b2c_module: bin2cell module (imported if None)

    Returns:
        True if override was successfully installed
    """
    if b2c_module is None:
        import bin2cell as b2c_module

    if hasattr(b2c_module, "bin2cell") and hasattr(b2c_module.bin2cell, "actual_vs_inferred_image_shape"):
        b2c_module.bin2cell.actual_vs_inferred_image_shape = dummy_image_shape_check
        return True

    if hasattr(b2c_module, "actual_vs_inferred_image_shape"):
        b2c_module.actual_vs_inferred_image_shape = dummy_image_shape_check
        return True

    return False


def update_adata_for_crop(
    adata: Any,
    crop: CropRegion,
    lib_id: Optional[str] = None,
) -> Any:
    """
    Update AnnData spatial information based on cropped image region.

    Performs:
    1. Coordinate shift (moves crop origin to [0, 0])
    2. Spot filtering (removes spots outside crop bounds)
    3. Reference image cropping (lowres/hires)

    Args:
        adata: scanpy AnnData object with spatial data
        crop: Crop region coordinates and dimensions
        lib_id: Library ID (auto-detected if None)

    Returns:
        Updated AnnData object
    """
    if lib_id is None:
        lib_id = list(adata.uns["spatial"].keys())[0]

    # 1. Coordinate shift
    adata.obsm["spatial"] = adata.obsm["spatial"] - np.array([crop.x, crop.y])

    # 2. Spot filtering
    subset_mask = (
        (adata.obsm["spatial"][:, 0] >= 0)
        & (adata.obsm["spatial"][:, 0] < crop.width)
        & (adata.obsm["spatial"][:, 1] >= 0)
        & (adata.obsm["spatial"][:, 1] < crop.height)
    )
    adata._inplace_subset_obs(subset_mask)
    print(f"✅ Subset spots: {np.sum(subset_mask)} spots remaining.")

    # 3. Reference image cropping
    scalefactors = adata.uns["spatial"][lib_id]["scalefactors"]
    images_dict = adata.uns["spatial"][lib_id]["images"]

    if "tissue_lowres_image" in images_dict:
        scale = scalefactors["tissue_lowres_scalef"]
        lx = int(crop.x * scale)
        ly = int(crop.y * scale)
        lw = int(crop.width * scale)
        lh = int(crop.height * scale)

        orig_img = images_dict["tissue_lowres_image"]
        max_h, max_w = orig_img.shape[:2]
        ly_end = min(ly + lh, max_h)
        lx_end = min(lx + lw, max_w)
        images_dict["tissue_lowres_image"] = orig_img[ly:ly_end, lx:lx_end]

    if "tissue_hires_image" in images_dict:
        scale = scalefactors["tissue_hires_scalef"]
        hx = int(crop.x * scale)
        hy = int(crop.y * scale)
        hw = int(crop.width * scale)
        hh = int(crop.height * scale)

        orig_img = images_dict["tissue_hires_image"]
        max_h, max_w = orig_img.shape[:2]
        hy_end = min(hy + hh, max_h)
        hx_end = min(hx + hw, max_w)
        images_dict["tissue_hires_image"] = orig_img[hy:hy_end, hx:hx_end]

    print(
        f"✅ AnnData updated for crop at ({crop.x}, {crop.y}) "
        f"with size {crop.width}x{crop.height}"
    )
    return adata


@dataclass
class Bin2CellLoadParams:
    """Parameters for bin2cell data loading and preprocessing."""
    mpp: float = 0.5
    prob_thresh_he: float = 0.05
    prob_thresh_gex: float = 0.7
    nms_thresh: float = 0.5
    max_bin_distance: int = 2
    block_size: int = 4096


@dataclass
class Bin2CellLoadResult:
    """Result of bin2cell data loading and preprocessing."""
    adata: Any
    lib_id: str
    he_full_path: str
    params: Bin2CellLoadParams

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "sp_adata": self.adata,
            "lib_id": self.lib_id,
            "mpp": self.params.mpp,
            "prob_thresh_HE": self.params.prob_thresh_he,
            "prob_thresh_GEX": self.params.prob_thresh_gex,
            "nms_thresh": self.params.nms_thresh,
            "max_bin_distance": self.params.max_bin_distance,
            "Block_size": self.params.block_size,
            "HE_full_path": self.he_full_path,
        }


def load_visium_and_preprocess(
    expression_path: str,
    source_image_path: str,
    crop: CropRegion,
    tmp_path: str,
    params: Bin2CellLoadParams,
    b2c_module: Optional[Any] = None,
    scanpy_module: Optional[Any] = None,
) -> Bin2CellLoadResult:
    """
    Load Visium data, crop coordinates, and preprocess.

    Combines: Visium reading -> crop adjustment -> HE image generation -> destriping.

    Args:
        expression_path: Path to 2um expression data
        source_image_path: Path to cropped H&E image
        crop: Crop region specification
        tmp_path: Temporary directory for HE outputs
        params: Bin2CellLoadParams
        b2c_module: bin2cell module (imported if None)
        scanpy_module: scanpy module (imported if None)

    Returns:
        Bin2CellLoadResult with AnnData and paths
    """
    if b2c_module is None:
        import bin2cell as b2c_module
    if scanpy_module is None:
        import scanpy as scanpy_module

    os.makedirs(tmp_path, exist_ok=True)
    he_full_path = os.path.join(tmp_path, "he.tif")
    if os.path.exists(he_full_path):
        os.remove(he_full_path)

    print(f"Reading Visium data from: {expression_path}")
    print(f"Using Cropped Image: {source_image_path}")

    adata = b2c_module.read_visium(expression_path, source_image_path=source_image_path)

    lib_id = list(adata.uns["spatial"].keys())[0]
    print(f"✅ Auto-detected lib_id: {lib_id}")

    adata = update_adata_for_crop(adata=adata, crop=crop, lib_id=lib_id)

    print("Preprocessing (Filter genes/cells)...")
    adata.var_names_make_unique()
    scanpy_module.pp.filter_genes(adata, min_cells=3)
    scanpy_module.pp.filter_cells(adata, min_counts=1)

    print(f"Generating Scaled HE Image (mpp={params.mpp})...")
    b2c_module.scaled_he_image(adata, mpp=params.mpp, save_path=he_full_path)
    print("Running Destriping...")
    b2c_module.destripe(adata, adjust_counts=True)

    result = Bin2CellLoadResult(
        adata=adata,
        lib_id=lib_id,
        he_full_path=he_full_path,
        params=params,
    )
    print(f"✅ sp_adata created in load_visium_and_preprocess (lib_id={lib_id}, n_obs={adata.n_obs})")
    return result
