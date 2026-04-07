import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from .reproducibility import set_global_seed, get_seed_from_env

def scanpy_qc_pipeline(sp_adata, params, seed=None, out=None):
    """
    Core logic for Scanpy QC / Filtering / Leiden pipeline.
    Args:
        sp_adata: AnnData object
        params: dict of parameters (same keys as widget)
        seed: random seed
        out: output widget (optional)
    Returns:
        sp_adata, sp_adata_sponly
    """
    effective_seed = get_seed_from_env(default=42) if seed is None else int(seed)
    set_global_seed(effective_seed)

    def _mt_hist(sp, xmax=10.0):
        vals = sp.obs["pct_counts_MT"].to_numpy()
        vals = vals[np.isfinite(vals)]
        vals = vals[vals <= xmax]
        plt.figure(figsize=(7, 4))
        plt.hist(vals, bins=50)
        plt.title(f"pct_counts_MT (<= {xmax})")
        plt.xlabel("pct_counts_MT")
        plt.ylabel("n")
        plt.show()

    sp_adata.var_names_make_unique()
    # bin_count filter
    if "bin_count" in sp_adata.obs.columns:
        before = sp_adata.n_obs
        sp_adata = sp_adata[sp_adata.obs["bin_count"] >= params["bin_min"]].copy()
        print(f"bin_count filter: {before:,} -> {sp_adata.n_obs:,} (>= {params['bin_min']})")
    else:
        print("⚠️ sp_adata.obs['bin_count'] not found. Skipping bin_count filter.")
    # Round counts if requested
    if params["round_counts"]:
        sp_adata.X = np.round(sp_adata.X).copy()
    sp_adata.raw = sp_adata.copy()
    # MT annotation + QC metrics
    prefix = params["prefix_mt"]
    sp_adata.var["MT"] = sp_adata.var_names.str.startswith(prefix)
    sc.pp.calculate_qc_metrics(sp_adata, qc_vars=["MT"], percent_top=None, log1p=False, inplace=True)
    print(f"Total cells (after early steps): {sp_adata.n_obs:,}")
    # QC plots (before filtering)
    if params["make_plots"]:
        sc.set_figure_params(fontsize=20, figsize=[7, 7])
        sc.pl.highest_expr_genes(sp_adata, n_top=params["qc_top_n"])
        sc.pl.violin(sp_adata, ["n_genes_by_counts", "total_counts", "pct_counts_MT"],
            jitter=0.4, multi_panel=True)
        sc.pl.scatter(sp_adata, "total_counts", "n_genes_by_counts",
            color="pct_counts_MT", size=40)
        sc.pl.scatter(sp_adata, x="total_counts", y="pct_counts_MT")
        _mt_hist(sp_adata, xmax=params["mt_hist_max"])
    # Filtering
    before = sp_adata.n_obs
    sc.pp.filter_cells(sp_adata, min_counts=int(params["min_counts"]))
    print(f"min_counts: {before:,} -> {sp_adata.n_obs:,} (>= {params['min_counts']})")
    before = sp_adata.n_obs
    if "total_counts" not in sp_adata.obs.columns:
        sc.pp.calculate_qc_metrics(sp_adata, qc_vars=["MT"], percent_top=None, log1p=False, inplace=True)
    sp_adata = sp_adata[sp_adata.obs["total_counts"] <= int(params["max_counts"])].copy()
    print(f"max_counts: {before:,} -> {sp_adata.n_obs:,} (<= {params['max_counts']})")
    before = sp_adata.n_obs
    sp_adata = sp_adata[sp_adata.obs["pct_counts_MT"] < float(params["mt_max"])].copy()
    print(f"MT% filter: {before:,} -> {sp_adata.n_obs:,} (< {params['mt_max']})")
    if params["make_plots"]:
        sc.set_figure_params(fontsize=20, figsize=[7, 7])
        sc.pl.highest_expr_genes(sp_adata, n_top=params["qc_top_n"])
        sc.pl.violin(sp_adata, ["n_genes_by_counts", "total_counts", "pct_counts_MT"],
            jitter=0.4, multi_panel=True)
        sc.pl.scatter(sp_adata, "total_counts", "n_genes_by_counts",
            color="pct_counts_MT", size=40)
        sc.pl.scatter(sp_adata, x="total_counts", y="pct_counts_MT")
        _mt_hist(sp_adata, xmax=params["mt_hist_max"])
    sp_adata.layers["counts"] = sp_adata.X.copy()
    sp_adata.raw = sp_adata
    sp_adata_sponly = sp_adata.copy()
    sp_adata_sponly.X = sp_adata_sponly.raw.X.copy()
    sp_adata_sponly.var = sp_adata_sponly.raw.var.copy()
    sp_adata_sponly.layers["counts"] = sp_adata_sponly.X.copy()
    sc.pp.normalize_total(sp_adata_sponly)
    sc.pp.log1p(sp_adata_sponly)
    sc.pp.highly_variable_genes(
        sp_adata_sponly,
        flavor="seurat_v3",
        n_top_genes=int(params["hvg_n"]),
        layer="counts",
        subset=False
    )
    print("Performing PCA...")
    sc.tl.pca(sp_adata_sponly, svd_solver="arpack",
        mask_var="highly_variable", n_comps=int(params["pca_n"]))
    sc.pp.neighbors(sp_adata_sponly, random_state=effective_seed)
    print("Performing UMAP...")
    sc.tl.umap(sp_adata_sponly, random_state=effective_seed)
    print("Performing Leiden...")
    sc.tl.leiden(
        sp_adata_sponly,
        resolution=float(params["leiden_res"]),
        flavor="igraph",
        n_iterations=int(params["leiden_iters"]),
        key_added="leiden",
        random_state=effective_seed
    )
    print("✅ Embeddings + Leiden done.")
    print("sp_adata_sponly.n_obs:", sp_adata_sponly.n_obs)
    sc.pl.umap(sp_adata_sponly, color=["leiden"], use_raw=False)
    sc.pl.umap(sp_adata_sponly, color=["total_counts"], use_raw=False)
    sp_adata.obs["leiden_nucleus"] = sp_adata_sponly.obs["leiden"].astype(str)
    sp_adata.obsm["X_PCA_nucleus"] = sp_adata_sponly.obsm["X_pca"].copy()
    sp_adata.obsm["X_umap_nucleus"] = sp_adata_sponly.obsm["X_umap"].copy()
    print("✅ Copied leiden + embeddings back to sp_adata.")
    print(f"Final Total number of cells: {sp_adata.n_obs:,}")
    print("Done! Go ahead.")
    return sp_adata, sp_adata_sponly
