"""
Spatial ROI computation and tissue boundary detection for Visium HD.
Handles automatic tissue detection, ROI calculation, and grid alignment.
"""

import os
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .image_ops import get_image_dimensions


def load_spatial_h5ad(results_path: str, sample_name: str, suffix: str = "_b2c.h5ad"):
    """Load spatial AnnData and return data with library id and file path."""
    import scanpy as sc

    h5ad_save_path = os.path.join(results_path, f"{sample_name}{suffix}")
    sp_adata_raw = sc.read_h5ad(h5ad_save_path)
    spatial = sp_adata_raw.uns.get("spatial", {})
    lib_id = list(spatial.keys())[0] if spatial else None
    return sp_adata_raw, lib_id, h5ad_save_path


def get_row_col_range(sp_adata_raw) -> Dict[str, int]:
    """Return min/max ranges for array_row and array_col."""
    row_vals = sp_adata_raw.obs["array_row"].to_numpy()
    col_vals = sp_adata_raw.obs["array_col"].to_numpy()
    return {
        "row_min": int(np.nanmin(row_vals)),
        "row_max": int(np.nanmax(row_vals)),
        "col_min": int(np.nanmin(col_vals)),
        "col_max": int(np.nanmax(col_vals)),
    }


def sanitize_bounds(x1: int, x2: int, y1: int, y2: int) -> Tuple[int, int, int, int]:
    """Normalize bounds so lower/upper order is guaranteed."""
    sx1, sx2 = sorted((x1, x2))
    sy1, sy2 = sorted((y1, y2))
    return sx1, sx2, sy1, sy2


def make_array_row_col_mask(sp_adata_raw, x1: int, x2: int, y1: int, y2: int):
    """Create boolean mask based on array_row/array_col bounds."""
    sx1, sx2, sy1, sy2 = sanitize_bounds(x1, x2, y1, y2)
    return (
        (sp_adata_raw.obs["array_row"] >= sx1)
        & (sp_adata_raw.obs["array_row"] <= sx2)
        & (sp_adata_raw.obs["array_col"] >= sy1)
        & (sp_adata_raw.obs["array_col"] <= sy2)
    )


def apply_array_row_col_mask(sp_adata_raw, x1: int, x2: int, y1: int, y2: int) -> Dict[str, Any]:
    """Apply row/col mask and return subset + selected bounds."""
    sx1, sx2, sy1, sy2 = sanitize_bounds(x1, x2, y1, y2)
    mask = make_array_row_col_mask(sp_adata_raw, sx1, sx2, sy1, sy2)
    n_selected = int(mask.sum())
    if n_selected == 0:
        return {
            "sp_adata": None,
            "n_selected": 0,
            "mask_x1_val": sx1,
            "mask_x2_val": sx2,
            "mask_y1_val": sy1,
            "mask_y2_val": sy2,
            "mask": mask,
        }
    return {
        "sp_adata": sp_adata_raw[mask].copy(),
        "n_selected": n_selected,
        "mask_x1_val": sx1,
        "mask_x2_val": sx2,
        "mask_y1_val": sy1,
        "mask_y2_val": sy2,
        "mask": mask,
    }


@dataclass
class CropConfig:
    """Configuration for image cropping / ROI parameters."""
    x: int
    y: int
    width: int
    height: int
    step: int = 512  # Grid alignment step size
    
    def snap_to_grid(self, full_width: int, full_height: int) -> "CropConfig":
        """
        Snap crop coordinates to grid alignment (STEP multiples).
        
        Args:
            full_width: Full image width
            full_height: Full image height
            
        Returns:
            New CropConfig with snapped coordinates
        """
        step_w = min(int(self.step), full_width)
        step_h = min(int(self.step), full_height)
        
        # Floor for top-left (don't move too much)
        x2 = (self.x // step_w) * step_w
        y2 = (self.y // step_h) * step_h
        
        # Ceil for width/height (don't make smaller)
        w2 = int(math.ceil(self.width / step_w) * step_w) if step_w > 0 else self.width
        h2 = int(math.ceil(self.height / step_h) * step_h) if step_h > 0 else self.height
        
        # Clamp to valid range
        x2 = max(0, min(x2, max(0, full_width - step_w)))
        y2 = max(0, min(y2, max(0, full_height - step_h)))
        
        # Adjust width/height to not exceed boundaries
        max_w = full_width - x2
        max_h = full_height - y2
        
        if step_w > 0:
            w2 = min(w2, (max_w // step_w) * step_w if max_w >= step_w else max_w)
        else:
            w2 = min(w2, max_w)
            
        if step_h > 0:
            h2 = min(h2, (max_h // step_h) * step_h if max_h >= step_h else max_h)
        else:
            h2 = min(h2, max_h)
        
        # Ensure at least one tile
        w2 = max(min(step_w, max_w), w2)
        h2 = max(min(step_h, max_h), h2)
        
        return CropConfig(x=x2, y=y2, width=w2, height=h2, step=self.step)


def auto_detect_tissue_bounds(
    expression_path_8um: str,
    source_image_path: str,
    margin: int = 200,
    step: Optional[int] = None
) -> CropConfig:
    """
    Automatically detect tissue boundary from Visium spatial data.
    
    Core algorithm: reads tissue spot coordinates, computes bounding box,
    applies margin, and optionally snaps to grid.
    
    Args:
        expression_path_8um: Path to 8um binned expression data directory
        source_image_path: Path to source BigTIFF image
        margin: Margin around tissue (pixels)
        step: Grid alignment step (if None, no snapping)
        
    Returns:
        CropConfig with detected bounds
        
    Raises:
        FileNotFoundError: If tissue_positions file not found
        ValueError: If no tissue spots detected
    """
    # Find tissue positions file
    parquet_path = os.path.join(expression_path_8um, "spatial", "tissue_positions.parquet")
    csv_path = os.path.join(expression_path_8um, "spatial", "tissue_positions.csv")
    
    df = None
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"tissue_positions file not found in {expression_path_8um}/spatial/")
    
    # Filter tissue spots
    tissue_df = df[df["in_tissue"] == 1]
    if tissue_df.empty:
        raise ValueError("No tissue spots found (in_tissue == 1)")
    
    # Calculate bounding box
    min_x = int(tissue_df["pxl_col_in_fullres"].min())
    max_x = int(tissue_df["pxl_col_in_fullres"].max())
    min_y = int(tissue_df["pxl_row_in_fullres"].min())
    max_y = int(tissue_df["pxl_row_in_fullres"].max())
    
    # Add margin
    x = max(0, min_x - margin)
    y = max(0, min_y - margin)
    width = max_x - min_x + (margin * 2)
    height = max_y - min_y + (margin * 2)
    
    config = CropConfig(x=x, y=y, width=width, height=height, step=step or 1)
    
    # Snap to grid if step provided
    if step and step > 1:
        full_w, full_h = get_image_dimensions(source_image_path)
        config = config.snap_to_grid(full_w, full_h)
    
    return config
