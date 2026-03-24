"""
Image processing operations for large spatial transcriptomics images.
Provides core functionality for reading, cropping, and saving BigTIFF images.
"""

import os
from typing import Tuple, TYPE_CHECKING

import numpy as np
import tifffile
import zarr

if TYPE_CHECKING:
    from .spatial_roi import CropConfig


def get_image_dimensions(source_image_path: str) -> Tuple[int, int]:
    """
    Get dimensions of BigTIFF image using Zarr.
    
    Args:
        source_image_path: Path to .btf or .tiff file
        
    Returns:
        (width, height) tuple
    """
    try:
        store = tifffile.imread(source_image_path, aszarr=True)
        z_img = zarr.open(store, mode='r')
        height, width = z_img.shape[0], z_img.shape[1]
        return width, height
    except Exception as e:
        raise RuntimeError(f"Failed to read image dimensions: {e}")


def save_cropped_image(
    source_image_path: str,
    output_path: str,
    crop_config: "CropConfig",
    compression: str = "lzw",
    overwrite: bool = False
) -> None:
    """
    Save cropped region from BigTIFF image.
    
    Args:
        source_image_path: Path to source .btf/.tiff file
        output_path: Path to save cropped image
        crop_config: Crop configuration (from spatial_roi module)
        compression: TIFF compression method
        overwrite: If False, skip if output exists
        
    Raises:
        FileExistsError: If output exists and overwrite=False
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read source via zarr
    store = tifffile.imread(source_image_path, aszarr=True)
    z_img = zarr.open(store, mode='r')
    
    # Extract crop region
    y1, y2 = crop_config.y, crop_config.y + crop_config.height
    x1, x2 = crop_config.x, crop_config.x + crop_config.width
    crop_data = np.asarray(z_img[y1:y2, x1:x2])
    
    # Save as BigTIFF
    tifffile.imwrite(output_path, crop_data, bigtiff=True, compression=compression)
    
    print(f"✅ Saved cropped image: {output_path}")
    print(f"   Region: ({crop_config.x}, {crop_config.y}) + {crop_config.width}x{crop_config.height}")
    print(f"   Shape: {crop_data.shape}")
