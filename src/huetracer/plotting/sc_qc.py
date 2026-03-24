import scanpy as sc
import anndata as ad

def plot_qc(adata: ad.AnnData):
    print(f"📊 QC plots for sc_adata_merged: {adata.shape}")
    sc.pl.highest_expr_genes(adata, n_top=30)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_MT'],
                 jitter=0.4, multi_panel=True)
    sc.pl.scatter(adata, 'total_counts', 'n_genes_by_counts', color='pct_counts_MT', size=40)
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_MT')
    print("Done! Go ahead.")
