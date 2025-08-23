import pandas as pd
import numpy as np
import scipy
import scanpy as sc
import scipy.sparse as sparse
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def make_coexp_cc_df(ligand_adata, edge_df, role):
    sender = edge_df.cell1 if role == "sender" else edge_df.cell2
    receiver = edge_df.cell2 if role == "sender" else edge_df.cell1
    coexp_df = pd.DataFrame(
        ligand_adata[sender].X *
        ligand_adata[receiver].layers['activity'],
        columns=ligand_adata.var_names, index=edge_df.index
    )
    coexp_df['cell2_type'] = edge_df['cell2_type']
    coexp_df['cell1_type'] = edge_df['cell1_type']
    coexp_cc_df = coexp_df.groupby(['cell2_type', 'cell1_type']).sum()
    coexp_cc_df = coexp_cc_df.reset_index().melt(id_vars=['cell1_type', 'cell2_type'], var_name='ligand', value_name='coactivity')
    return coexp_cc_df

def make_non_zero_values(mat):
    top_mat = mat > 0
    return(top_mat)

def make_positive_values(mat):
    mat[mat < 0] = 0
    return(mat)
    
def make_top_values(mat, top_fraction = 0.1, axis=0):
    top_mat = mat > np.quantile(mat, 1 - top_fraction, axis=axis, keepdims=True)
    return(top_mat)

def safe_toarray(x):
    if type(x) != np.ndarray:
        return x.toarray()
    else:
        return x

def add_zscore_layers(sp_adata, top_fraction=0.01):
    """
    Function to add z-score layers to an AnnData object
    
    Parameters:
    -----------
    sp_adata : AnnData
        AnnData object of single-cell data
    top_fraction : float
        Fraction of top genes to keep (default: 0.01)
    """
    # Get data shape
    shape = sp_adata.shape
    
    # Get a dense array of X
    if sparse.issparse(sp_adata.X):
        X_dense = sp_adata.X.toarray()
    else:
        X_dense = sp_adata.X.copy()
    
    # Prepare zero matrices for results
    sp_adata.layers["zscore_by_celltype"] = np.zeros_like(X_dense)
    sp_adata.layers["zscore_all_celltype"] = np.zeros_like(X_dense)
    
    # Calculate global standard deviation using non-zero values (moved up for efficiency)
    std_all = np.array([
        np.std(gene_expr[gene_expr != 0]) if np.any(gene_expr != 0) else 1
        for gene_expr in X_dense.T
    ])
    std_all[std_all == 0] = 1  # Prevent division by zero
    
    # Calculate z-score for each cell type
    for ct in sp_adata.obs["celltype"].unique():
        idx = sp_adata.obs["celltype"] == ct
        X_sub = X_dense[idx]
        
        # Calculate mean within cell type
        mean = X_sub.mean(axis=0)
        
        # Calculate proper z-score: (value - celltype_mean) / global_nonzero_std
        z = (X_sub - mean) / std_all
        
        # Convert to positive values and store in layer
        sp_adata.layers["zscore_by_celltype"][idx] = make_positive_values(z)
    
    # Calculate overall z-score (high expression identification)
    z_all = X_dense
    zscore_all = make_positive_values(z_all)
    zscore_all = make_top_values(zscore_all, axis=0, top_fraction=top_fraction)
    
    sp_adata.layers["zscore_all_celltype"] = zscore_all

def construct_microenvironment_data(sp_adata, ligands, expr_up_by_ligands, neighbor_cell_numbers=19):
    n_cells = len(sp_adata)
    
    # Step 1: Vectorized metadata extraction
    cluster_values = sp_adata.obs['cluster'].values if 'cluster' in sp_adata.obs.columns else np.full(n_cells, 'unknown')
    celltype_values = sp_adata.obs['celltype'].values if 'celltype' in sp_adata.obs.columns else np.full(n_cells, 'unknown')
    
    # Step 2: Optimized coordinates and neighbors
    coords = sp_adata.obs[["array_row", "array_col"]].values.astype(np.float32)
    nbrs = NearestNeighbors(n_neighbors=neighbor_cell_numbers, algorithm='ball_tree', n_jobs=-1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    
    # Step 3: Expression data handling
    if "zscore_all_celltype" in sp_adata.layers:
        exp_data_raw = sp_adata.layers["zscore_all_celltype"]
    else:
        exp_data_raw = sp_adata.X
    
    if scipy.sparse.issparse(exp_data_raw):
        exp_data = exp_data_raw.toarray()
    else:
        exp_data = exp_data_raw
    
    # Step 4: Ultra-fast vectorized aggregation
    # Use advanced indexing for maximum speed
    neighbor_sums = np.zeros_like(exp_data)
    zscore_sums = np.zeros_like(expr_up_by_ligands)
    
    for i in range(neighbor_cell_numbers):
        neighbor_sums += exp_data[indices[:, i]]
        zscore_sums += expr_up_by_ligands[indices[:, i]]
    
    # Step 5: Ligand extraction
    gene_names = sp_adata.var_names
    ligand_mask = gene_names.isin(ligands)
    available_ligands = gene_names[ligand_mask].tolist()
    ligand_indices = np.where(ligand_mask)[0]
    
    exp_data_ligands = exp_data[:, ligand_indices]
    
    # Step 6: Center adata
    center_adata = sp_adata[:, available_ligands].copy()
    
    # Handle dimension matching
    if len(available_ligands) <= expr_up_by_ligands.shape[1]:
        center_adata.layers["expr_up"] = expr_up_by_ligands[:, :len(available_ligands)]
    else:
        # Pad with zeros if needed
        padded_expr = np.zeros((expr_up_by_ligands.shape[0], len(available_ligands)))
        padded_expr[:, :expr_up_by_ligands.shape[1]] = expr_up_by_ligands
        center_adata.layers["expr_up"] = padded_expr
    
    # Step 7: Ultra-fast edge creation using vectorized operations
    n_edges = n_cells * neighbor_cell_numbers
    
    # Pre-allocate arrays
    center_ids = np.repeat(np.arange(n_cells), neighbor_cell_numbers)
    neighbor_ids = indices.ravel()
    
    # Vectorized name mapping
    cell_names = center_adata.obs_names.values
    
    edge_df = pd.DataFrame({
        'edge': np.arange(n_edges),
        'cell1': cell_names[center_ids],
        'cell2': cell_names[neighbor_ids],
        'cell1_type': celltype_values[center_ids],
        'cell2_type': celltype_values[neighbor_ids],
        'cell1_cluster': cluster_values[center_ids],
        'cell2_cluster': cluster_values[neighbor_ids]
    })
    
    print(f"{len(edge_df)} edges, {center_adata.shape} center_adata")
    
    return edge_df, center_adata, exp_data_ligands

def prepare_microenv_data(sp_adata_raw, sp_adata_microenvironment, lt_df_raw, min_frac=0.001, n_top_genes=2000):
    print("Starting data preparation...")
    
    # Step 1: Common cells with proper matrix handling
    common_cells = sp_adata_microenvironment.obs_names.intersection(sp_adata_raw.obs_names)
    sp_adata = sp_adata_raw[common_cells].copy()
    
    # Step 2: Fix COO matrix and efficient normalization
    if scipy.sparse.issparse(sp_adata.X):
        # Convert COO to CSR if needed (AnnData compatibility)
        if isinstance(sp_adata.X, scipy.sparse.coo_matrix):
            sp_adata.X = sp_adata.X.tocsr()
        
        # Keep as sparse for memory efficiency during normalization
        bin_counts = sp_adata.obs['bin_count'].values
        # Create sparse diagonal matrix for efficient multiplication
        diag_matrix = scipy.sparse.diags(1 / bin_counts, format='csr')
        sp_adata.X = diag_matrix @ sp_adata.X
    else:
        bin_counts = sp_adata.obs['bin_count'].values
        sp_adata.X = sp_adata.X / bin_counts[:, np.newaxis]
    
    sp_adata.raw = None
    
    # Step 3: Metadata (vectorized)
    microenv_obs = sp_adata_microenvironment.obs.loc[common_cells]
    sp_adata.obs['cluster'] = microenv_obs['predicted_microenvironment'].values
    sp_adata.obs['celltype'] = microenv_obs['predicted_cell_type'].values
    sp_adata.obs_names_make_unique()
    
    # Step 4: Pre-filter genes efficiently
    min_cells = int(np.ceil(sp_adata.n_obs * min_frac))
    
    if scipy.sparse.issparse(sp_adata.X):
        gene_counts = np.asarray((sp_adata.X > 0).sum(axis=0)).flatten()
    else:
        gene_counts = (sp_adata.X > 0).sum(axis=0)
    
    valid_genes_mask = gene_counts >= min_cells
    sp_adata = sp_adata[:, valid_genes_mask].copy()
    
    # Step 5: Streamlined processing with proper matrix handling
    # Force materialization if view
    if sp_adata.is_view:
        sp_adata = sp_adata.copy()
    
    filtered_adata = sp_adata.copy()
    
    # Ensure proper matrix format
    if scipy.sparse.issparse(filtered_adata.X):
        if isinstance(filtered_adata.X, scipy.sparse.coo_matrix):
            filtered_adata.X = filtered_adata.X.tocsr()
    
    # Efficient normalization
    # sc.pp.normalize_total(filtered_adata, target_sum=1e4)
    # sc.pp.log1p(filtered_adata)
    #filtered_adata.layers["counts"] = filtered_adata.X.copy()
    
    # Step 6: Combined gene selection strategy    
    # HVG genes
    sc.pp.highly_variable_genes(filtered_adata, n_top_genes=n_top_genes)
    hvg_genes = set(filtered_adata.var[filtered_adata.var['highly_variable']].index)
    
    # Top expression genes
    if scipy.sparse.issparse(filtered_adata.X):
        mean_expr = np.asarray(filtered_adata.X.mean(axis=0)).flatten()
    else:
        mean_expr = filtered_adata.X.mean(axis=0)
    
    top_expr_indices = np.argpartition(mean_expr, -n_top_genes)[-n_top_genes:]
    top_expr_genes = set(filtered_adata.var_names[top_expr_indices])
    
    # Marker genes (simplified approach)
    sc.tl.rank_genes_groups(filtered_adata, groupby='celltype', method='logreg', n_genes=min(100, n_top_genes), max_iter=2000)
    marker_genes_df = pd.DataFrame(filtered_adata.uns['rank_genes_groups']['names'])
    marker_genes = set(marker_genes_df.values.flatten())
    marker_genes.discard(np.nan)  # Remove NaN values
    
    # Combine all gene sets
    all_selected_genes = hvg_genes | top_expr_genes | marker_genes | set(lt_df_raw.columns)
    # all_selected_genes = hvg_genes | marker_genes | set(lt_df_raw.columns)
    final_genes = list(all_selected_genes & set(sp_adata.var_names))
    
    # Final subsetting and LT processing
    sp_adata = sp_adata[:, final_genes].copy()
    
    # Optimized LT processing
    common_genes = list(set(lt_df_raw.index) & set(sp_adata.var_names))
    lt_df = lt_df_raw.loc[common_genes].copy()
    sp_adata = sp_adata[:, common_genes]
    
    common_columns = list(set(lt_df.columns) & set(sp_adata.var_names))
    lt_df = lt_df.loc[:, common_columns]
    
    # Efficient normalization
    column_sums = lt_df.sum(axis=0)
    column_sums = column_sums.replace(0, 1)
    lt_df = lt_df.div(column_sums, axis=1)
    
    print(f"sp_adata {sp_adata.shape}, lt_df {lt_df.shape}")
    
    return sp_adata, lt_df