"""AnnData saving utilities with optional memory cleanup."""

import gc
import os
import time
from typing import Any, Optional


def format_bytes(n: float) -> str:
    """Format byte size to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def save_anndata(
    adata: Any,
    output_path: str,
    remove_existing: bool = True,
    verbose: bool = True,
) -> float:
    """Save a single AnnData object to h5ad format.
    
    Args:
        adata: AnnData object to save
        output_path: Output file path (.h5ad)
        remove_existing: Remove existing file before saving
        verbose: Print progress messages
        
    Returns:
        File size in bytes
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if remove_existing and os.path.exists(output_path):
        os.remove(output_path)
        if verbose:
            print(f"🧹 Removed existing: {output_path}")
    
    if verbose:
        print(f"💾 Saving to {output_path} ...")
    
    adata.write_h5ad(output_path)
    file_size = os.path.getsize(output_path)
    
    if verbose:
        print(f"✅ Saved ({format_bytes(file_size)})")
    
    return file_size


def save_anndata_batch(
    adata_dict: dict[str, Any],
    output_paths: dict[str, str],
    remove_existing: bool = True,
    delete_after: bool = False,
    verbose: bool = True,
) -> dict[str, float]:
    """Save multiple AnnData objects with optional memory cleanup.
    
    Args:
        adata_dict: Dictionary of {name: adata_object}
        output_paths: Dictionary of {name: output_path}
        remove_existing: Remove existing files before saving
        delete_after: Delete objects from memory after saving
        verbose: Print progress messages
        
    Returns:
        Dictionary of {name: file_size_bytes}
    """
    if verbose:
        print("=== Saving AnnData Objects ===")
    
    t0 = time.time()
    sizes = {}
    
    for name, adata in adata_dict.items():
        if name not in output_paths:
            if verbose:
                print(f"⚠️ Skipping {name}: no output path specified")
            continue
        
        output_path = output_paths[name]
        sizes[name] = save_anndata(adata, output_path, remove_existing, verbose)
    
    if delete_after:
        for name in adata_dict.keys():
            del adata_dict[name]
        gc.collect()
        if verbose:
            print("🧠 Memory freed.")
    
    if verbose:
        print(f"\nDone. Total time: {time.time() - t0:.2f} sec")
    
    return sizes
