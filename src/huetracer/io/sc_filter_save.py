import os
import json
import re
import time
import gc

def fmt_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def parse_exclude(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def _normalize_cluster_name(value):
    s = str(value).strip()
    if re.fullmatch(r"\d+", s):
        return f"C{s}"
    if re.fullmatch(r"C\d+", s):
        return s
    return s

def save_filtered_sc_adata(
    ad,
    annotation_dict,
    sc_filtered_path,
    exclude_labels=None,
    write_celltype=True,
    show_plots=True,
    remove_existing=True,
    delete_after=False
):
    import scanpy as sc
    t0 = time.time()
    if remove_existing and os.path.exists(sc_filtered_path):
        os.remove(sc_filtered_path)
    # 1) cell_type mapping
    if write_celltype:
        mapped = []
        missing = set()
        for cl in ad.obs["leiden"].astype(str).tolist():
            normalized_cl = _normalize_cluster_name(cl)
            cell_type = None
            if annotation_dict is not None:
                cell_type = annotation_dict.get(cl)
                if cell_type is None:
                    cell_type = annotation_dict.get(normalized_cl)
            if cell_type is not None:
                mapped.append(cell_type)
            else:
                mapped.append("Other")
                missing.add(normalized_cl)
        ad.obs["cell_type"] = mapped
    # 2) filter
    exclude = exclude_labels or ["Doublet", "Other"]
    keep_mask = ~ad.obs["cell_type"].isin(exclude)
    filtered_sc_adata = ad[keep_mask].copy()
    # 3) plots
    if show_plots:
        sc.pl.umap(filtered_sc_adata, color=["leiden"], show=True)
        sc.pl.umap(filtered_sc_adata, color=["cell_type"], show=True)
    # 4) cell_type_annotation
    filtered_sc_adata.obs["cell_type_annotation"] = filtered_sc_adata.obs["leiden"].astype(str)
    # 5) save
    filtered_sc_adata.write_h5ad(sc_filtered_path)
    size = os.path.getsize(sc_filtered_path)
    n_kept = filtered_sc_adata.n_obs
    result = {
        "n_kept": n_kept,
        "n_total": ad.n_obs,
        "excluded": exclude,
        "missing_clusters": sorted(list(missing)) if write_celltype else [],
        "file_size": fmt_bytes(size),
        "elapsed": time.time()-t0
    }
    # 6) cleanup
    if delete_after:
        del filtered_sc_adata
        gc.collect()
    return result
