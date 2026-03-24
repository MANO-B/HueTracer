import os
import glob
import scanpy as sc
import anndata as ad
import gc
from typing import List, Dict, Any

def detect_dataset_type(path: str) -> str:
    if os.path.isdir(path):
        mtx = glob.glob(os.path.join(path, "matrix.mtx*"))
        feats = glob.glob(os.path.join(path, "features.tsv*")) + glob.glob(os.path.join(path, "genes.tsv*"))
        bars = glob.glob(os.path.join(path, "barcodes.tsv*"))
        if mtx and feats and bars:
            return "mtx"
        mtx2 = glob.glob(os.path.join(path, "**/matrix.mtx*"), recursive=True)
        if mtx2:
            return "mtx_parent"
        return "dir_unknown"
    low = path.lower()
    if low.endswith(".h5ad"):
        return "h5ad"
    if low.endswith(".h5"):
        return "h5"
    return "unknown"


def scan_sc_datasets(base_dir: str) -> List[str]:
    base_dir = os.path.expanduser(base_dir.strip())
    if not base_dir or not os.path.exists(base_dir):
        return []
    hits = []
    for mtx in glob.glob(os.path.join(base_dir, "**/matrix.mtx*"), recursive=True):
        hits.append(os.path.dirname(mtx))
    for h5 in glob.glob(os.path.join(base_dir, "**/*.h5"), recursive=True):
        hits.append(h5)
    for h5adf in glob.glob(os.path.join(base_dir, "**/*.h5ad"), recursive=True):
        hits.append(h5adf)
    hits = sorted(list(dict.fromkeys(hits)), key=lambda p: (p.count(os.sep), len(p), p))
    return hits


def read_one(path: str, mode: str = "auto") -> ad.AnnData:
    path = os.path.expanduser(path)
    if mode == "auto":
        typ = detect_dataset_type(path)
        if typ == "mtx":
            return sc.read_10x_mtx(path, var_names="gene_symbols", cache=True)
        if typ == "mtx_parent":
            mtx = glob.glob(os.path.join(path, "**/matrix.mtx*"), recursive=True)[0]
            folder = os.path.dirname(mtx)
            return sc.read_10x_mtx(folder, var_names="gene_symbols", cache=True)
        if typ == "h5":
            return sc.read_10x_h5(path)
        if typ == "h5ad":
            return sc.read_h5ad(path)
        raise ValueError(f"cannot auto-detect: {path}")
    if mode == "mtx":
        return sc.read_10x_mtx(path, var_names="gene_symbols", cache=True)
    if mode == "h5":
        return sc.read_10x_h5(path)
    if mode == "h5ad":
        return sc.read_h5ad(path)
    raise ValueError("unknown read_mode")


def qc_and_filter(adata: ad.AnnData, p: Dict[str, Any]) -> ad.AnnData:
    adata.var_names_make_unique()
    mt_prefix = p["mt_prefix"]
    adata.var["MT"] = adata.var_names.str.startswith(mt_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["MT"], percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(adata, min_genes=p["min_genes"])
    sc.pp.filter_genes(adata, min_cells=p["min_cells"])
    sc.pp.filter_cells(adata, min_counts=p["min_counts"])
    adata = adata[adata.obs["total_counts"] <= p["max_counts"]].copy()
    adata = adata[adata.obs["pct_counts_MT"] < p["max_pct_mt"]].copy()
    return adata
