"""
Visium-specific data loaders for spatial transcriptomics.
Handles loading of Visium reference images and spatial metadata.
"""

import os
from typing import Dict

import PIL.Image
import tifffile


def load_reference_images(expression_path_8um: str) -> Dict[str, PIL.Image.Image]:
    """
    Load Visium reference images for preview and visualization.
    
    Args:
        expression_path_8um: Path to 8um expression directory
        
    Returns:
        Dictionary with 'lowres' and optionally 'cytassist' PIL Images
        
    Example:
        >>> ref_images = load_reference_images("/path/to/8um_data")
        >>> lowres_img = ref_images['lowres']
        >>> if 'cytassist' in ref_images:
        ...     cytassist_img = ref_images['cytassist']
    """
    result = {}
    
    # Low-res image
    lowres_path = os.path.join(expression_path_8um, "spatial", "tissue_lowres_image.png")
    if os.path.exists(lowres_path):
        result['lowres'] = PIL.Image.open(lowres_path)
    
    # CytAssist image
    cytassist_path = os.path.join(expression_path_8um, "spatial", "cytassist_image.tiff")
    if os.path.exists(cytassist_path):
        try:
            ca_data = tifffile.imread(cytassist_path)
            result['cytassist'] = PIL.Image.fromarray(ca_data)
        except Exception:
            pass
    
    return result
