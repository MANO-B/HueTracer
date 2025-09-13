import pandas as pd
import numpy as np
import scipy
import scanpy as sc
import scipy.sparse as sparse
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy import stats
from scipy.stats import beta, binom, chi2_contingency
from statsmodels.stats.multitest import multipletests
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    
    # Calculate global standard deviation (moved up for efficiency)
    # std_all = np.array([
    #     np.std(gene_expr[gene_expr != 0]) if np.any(gene_expr != 0) else 1
    #     for gene_expr in X_dense.T
    # ])
    std_all = np.array([
        np.mean(gene_expr[gene_expr != 0]) if np.any(gene_expr != 0) else 1
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
    z_all = X_dense# - X_dense.mean(axis=0)
    zscore_all = make_positive_values(z_all)
    sp_adata.layers["zscore_all_celltype_raw"] = zscore_all
    zscore_alls = make_top_values(zscore_all, axis=0, top_fraction=top_fraction)
    sp_adata.layers["zscore_all_celltype"] = zscore_alls
    
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
    # all_selected_genes = hvg_genes | top_expr_genes | marker_genes | set(lt_df_raw.columns)
    # all_selected_genes = hvg_genes | marker_genes | set(lt_df_raw.columns)
    all_selected_genes = marker_genes | set(lt_df_raw.columns)
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

def calculate_enhanced_coexpression_coactivity(edge_df, center_adata, exp_data, expr_up_by_ligands, 
                                               sp_adata, role="receiver", up_rate=1.25):
    """
    Fast enhanced co-expression analysis
    
    Optimization points:
    1. Vectorized calculations
    2. Pre-calculation and caching
    3. Memory-efficient processing
    4. Reduction of unnecessary statistical calculations
    """
    
    print("Fast enhanced co-expression calculation...")
    
    # Data preparation (optimized)
    center_adata.X = exp_data
    gene_names = center_adata.var_names.tolist()
    n_genes = len(gene_names)
    
    # Sender-receiver relationship for role='receiver'
    if role == "receiver":
        actual_sender = edge_df.cell2.values  # convert to numpy array
        actual_receiver = edge_df.cell1.values
        sender_type_col = 'cell2_type'
        receiver_type_col = 'cell1_type'
    else:
        actual_sender = edge_df.cell1.values
        actual_receiver = edge_df.cell2.values
        sender_type_col = 'cell1_type'
        receiver_type_col = 'cell2_type'
    
    # Pre-calculation of index mapping
    cell_to_idx = {cell: idx for idx, cell in enumerate(center_adata.obs_names)}
    sender_indices = np.array([cell_to_idx[cell] for cell in actual_sender])
    receiver_indices = np.array([cell_to_idx[cell] for cell in actual_receiver])
    
    # Get expression data (vectorized)
    sender_expr = exp_data[sender_indices]  # (n_edges, n_genes)
    receiver_expr = expr_up_by_ligands[receiver_indices]  # (n_edges, n_genes)
    
    if hasattr(sender_expr, 'toarray'):
        sender_expr = sender_expr.toarray()
    if hasattr(receiver_expr, 'toarray'):
        receiver_expr = receiver_expr.toarray()
    
    # Co-expression calculation (vectorized)
    coexp_matrix = sender_expr * receiver_expr
    
    # Cell type encoding (fast)
    sender_types = edge_df[sender_type_col].values
    receiver_types = edge_df[receiver_type_col].values
    
    unique_sender_types = np.unique(sender_types)
    unique_receiver_types = np.unique(receiver_types)
    
    sender_type_to_idx = {t: i for i, t in enumerate(unique_sender_types)}
    receiver_type_to_idx = {t: i for i, t in enumerate(unique_receiver_types)}
    
    sender_type_encoded = np.array([sender_type_to_idx[t] for t in sender_types])
    receiver_type_encoded = np.array([receiver_type_to_idx[t] for t in receiver_types])
    
    # Baseline calculation (pre-calculation/caching)
    print("Calculating baselines...")
    baseline_rates = fast_calculate_baseline_rates(sp_adata, expr_up_by_ligands, gene_names)
    
    # Large-scale contingency table calculation (vectorized)
    print("Computing contingency tables...")
    results_data = fast_compute_all_contingency_tables(
        sender_expr, receiver_expr, sender_type_encoded, receiver_type_encoded,
        unique_sender_types, unique_receiver_types, gene_names, baseline_rates
    )
    
    # Formatting results to existing format
    print("Formatting results...")
    coexp_cc_df = format_results_to_existing_format(
        results_data, unique_sender_types, unique_receiver_types, gene_names, 
        sender_type_col, receiver_type_col, role, up_rate
    )
    
    # Create bargraph_df (existing format)
    bargraph_data = {
        receiver_type_col: receiver_types,
        sender_type_col: sender_types
    }
    
    for i, gene in enumerate(gene_names):
        bargraph_data[gene] = coexp_matrix[:, i]
    
    bargraph_df = pd.DataFrame(bargraph_data, index=edge_df.index)
    
    # Result summary
    n_significant = np.sum(coexp_cc_df['is_significant'])
    n_enhanced_significant = np.sum(coexp_cc_df.get('enhanced_significant', False))
    
    print(f"Completed: {len(coexp_cc_df)} interactions")
    print(f"Traditional: {n_significant} significant")
    print(f"Enhanced: {n_enhanced_significant} significant with baseline consideration")
    
    return coexp_cc_df, bargraph_df

def fast_calculate_baseline_rates(sp_adata, expr_up_by_ligands, gene_names):
    """
    Fast calculation of baseline response rates
    """
    baseline_rates = {}
    
    # Per-cell-type processing (vectorized)
    cell_types = sp_adata.obs['celltype'].unique()
    celltype_values = sp_adata.obs['celltype'].values
    
    for cell_type in cell_types:
        cell_mask = celltype_values == cell_type
        
        if not np.any(cell_mask):
            continue
        
        # Expression data for that cell type (vectorized)
        cell_expr = expr_up_by_ligands[cell_mask]
        
        # Batch calculation of response rates for all ligands
        response_rates = np.mean(cell_expr > 0, axis=0)
        
        # Store in dictionary (only necessary part)
        baseline_rates[cell_type] = dict(zip(gene_names[:len(response_rates)], response_rates))
    
    return baseline_rates

def fast_compute_all_contingency_tables(sender_expr, receiver_expr, sender_type_encoded, 
                                      receiver_type_encoded, unique_sender_types, 
                                      unique_receiver_types, gene_names, baseline_rates):
    """
    Fast calculation of all contingency tables
    """
    n_sender_types = len(unique_sender_types)
    n_receiver_types = len(unique_receiver_types)
    n_genes = len(gene_names)
    
    # For storing results
    results_data = []
    
    # Process for each sender-receiver cell combination
    for s_idx, sender_type in enumerate(unique_sender_types):
        for r_idx, receiver_type in enumerate(unique_receiver_types):
            
            # Extract edges for this combination
            mask = (sender_type_encoded == s_idx) & (receiver_type_encoded == r_idx)
            
            if not np.any(mask):
                continue
            
            # Expression data for this combination
            s_expr_subset = sender_expr[mask]  # (n_edges_subset, n_genes)
            r_expr_subset = receiver_expr[mask]
            
            # Batch calculation of contingency tables for all genes (vectorized)
            contingency_stats = compute_vectorized_contingency_stats(
                s_expr_subset, r_expr_subset, gene_names, baseline_rates.get(receiver_type, {})
            )
            
            # Add to results
            for gene_idx, gene in enumerate(gene_names):
                stats_dict = {k: v[gene_idx] if hasattr(v, '__len__') else v for k, v in contingency_stats.items()}
                stats_dict.update({
                    'sender_type': sender_type,
                    'receiver_type': receiver_type,
                    'ligand': gene
                })
                results_data.append(stats_dict)
    
    return results_data

def compute_vectorized_contingency_stats(sender_expr, receiver_expr, gene_names, baseline_dict):
    """
    Calculation of vectorized contingency table statistics
    """
    n_edges, n_genes = sender_expr.shape
    
    # Binarization (vectorized)
    sender_binary = sender_expr > 0  # (n_edges, n_genes)
    receiver_binary = receiver_expr > 0
    
    # Batch calculation of four situations
    sender_pos_receiver_pos = np.sum(sender_binary & receiver_binary, axis=0)  # (n_genes,)
    sender_pos_receiver_neg = np.sum(sender_binary & ~receiver_binary, axis=0)
    sender_neg_receiver_pos = np.sum(~sender_binary & receiver_binary, axis=0)
    sender_neg_receiver_neg = np.sum(~sender_binary & ~receiver_binary, axis=0)
    
    # Basic statistics
    sender_positive_count = sender_pos_receiver_pos + sender_pos_receiver_neg
    sender_negative_count = sender_neg_receiver_pos + sender_neg_receiver_neg
    
    # Conditional probability (with zero-division protection)
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_prob_r_given_s = np.divide(sender_pos_receiver_pos, sender_positive_count, 
                                          out=np.zeros_like(sender_pos_receiver_pos, dtype=float),
                                          where=sender_positive_count>0)
        
        cond_prob_r_given_not_s = np.divide(sender_neg_receiver_pos, sender_negative_count,
                                              out=np.zeros_like(sender_neg_receiver_pos, dtype=float),
                                              where=sender_negative_count>0)
    
    interaction_enhancement = cond_prob_r_given_s - cond_prob_r_given_not_s
    
    # Baseline information
    baseline_rates = np.array([baseline_dict.get(gene, 0.0) for gene in gene_names])
    
    # Fast statistical test (approximation of Fisher's exact test)
    fisher_p_values, odds_ratios = fast_vectorized_fisher_test(
        sender_pos_receiver_pos, sender_pos_receiver_neg,
        sender_neg_receiver_pos, sender_neg_receiver_neg
    )
    
    # Binomial test against baseline (vectorized)
    binomial_p_values = fast_vectorized_binomial_test(
        sender_pos_receiver_pos, sender_positive_count, baseline_rates
    )
    
    return {
        'total_interactions': n_edges,
        'sender_positive': sender_positive_count,
        'interaction_positive': sender_pos_receiver_pos,
        'sender_pos_receiver_pos': sender_pos_receiver_pos,
        'sender_pos_receiver_neg': sender_pos_receiver_neg,
        'sender_neg_receiver_pos': sender_neg_receiver_pos,
        'sender_neg_receiver_neg': sender_neg_receiver_neg,
        'cond_prob_receiver_given_sender': cond_prob_r_given_s,
        'cond_prob_receiver_given_not_sender': cond_prob_r_given_not_s,
        'interaction_enhancement': interaction_enhancement,
        'baseline_response_rate': baseline_rates,
        'enhanced_fisher_p': fisher_p_values,
        'enhanced_odds_ratio': odds_ratios,
        'baseline_binomial_p': binomial_p_values
    }

def fast_vectorized_fisher_test(a, b, c, d):
    """
    Approximation of vectorized Fisher's exact test
    """
    n_genes = len(a)
    p_values = np.full(n_genes, np.nan)
    odds_ratios = np.full(n_genes, np.nan)
    
    # Mask for valid cases
    valid_mask = (a + b + c + d) > 0
    
    if np.any(valid_mask):
        a_valid = a[valid_mask]
        b_valid = b[valid_mask]
        c_valid = c[valid_mask]
        d_valid = d[valid_mask]
        
        # Odds ratio calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            or_values = np.divide(a_valid * d_valid, b_valid * c_valid,
                                    out=np.full_like(a_valid, np.inf, dtype=float),
                                    where=(b_valid * c_valid) > 0)
        
        # Chi-square approximation (large sample)
        n_total = a_valid + b_valid + c_valid + d_valid
        expected_a = (a_valid + b_valid) * (a_valid + c_valid) / n_total
        
        # Chi-square statistic
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2_stats = np.divide((a_valid - expected_a) ** 2, expected_a,
                                     out=np.zeros_like(expected_a),
                                     where=expected_a > 0)
            chi2_stats += np.divide((b_valid - (a_valid + b_valid - expected_a)) ** 2, 
                                      a_valid + b_valid - expected_a,
                                      out=np.zeros_like(expected_a),
                                      where=(a_valid + b_valid - expected_a) > 0)
        
        # p-value approximation (for large samples)
        p_approx = 1 - stats.chi2.cdf(chi2_stats, df=1)
        
        # Store results in the original array
        p_values[valid_mask] = p_approx
        odds_ratios[valid_mask] = or_values
    
    return p_values, odds_ratios

def fast_vectorized_binomial_test(successes, trials, baseline_rates, alpha=0.05):
    """
    Vectorized Binomial test
    """
    n_genes = len(successes)
    p_values = np.full(n_genes, np.nan)
    
    # Mask for valid cases
    valid_mask = (trials > 0) & (baseline_rates > 0) & (baseline_rates < 1)
    
    if np.any(valid_mask):
        # Use normal approximation (large sample)
        s_valid = successes[valid_mask]
        t_valid = trials[valid_mask] 
        r_valid = baseline_rates[valid_mask]
        
        # Expected value and standard deviation
        expected = t_valid * r_valid
        std_dev = np.sqrt(t_valid * r_valid * (1 - r_valid))
        
        # Z-statistic
        with np.errstate(divide='ignore', invalid='ignore'):
            z_stats = np.divide(s_valid - expected, std_dev,
                                  out=np.zeros_like(s_valid, dtype=float),
                                  where=std_dev > 0)
        
        # p-value for two-sided test
        p_approx = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
        
        p_values[valid_mask] = p_approx
    
    return p_values

def format_results_to_existing_format(results_data, unique_sender_types, unique_receiver_types,
                                      gene_names, sender_type_col, receiver_type_col, role, up_rate):
    """
    Format results to existing format
    """
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # ★★★ 修正点 ★★★
    # roleの値に関わらず、最終的なカラムの意味を統一する
    # ルール：'cell1_type' は常に受信細胞、'cell2_type' は常に送信細胞とする
    df['cell1_type'] = df['receiver_type'] 
    df['cell2_type'] = df['sender_type']
    
    # (ここから下の処理は変更なし)
    
    # Add basic statistics
    df['coactivity_per_sender_cell_expr_ligand'] = np.divide(
        df['interaction_positive'], df['sender_positive'],
        out=np.zeros_like(df['interaction_positive'], dtype=float),
        where=df['sender_positive'] > 0
    )
    
    # Traditional statistical test (simplified version)
    print("Computing traditional statistics...")
    df = add_traditional_statistics(df, up_rate)
    
    # Significance determination for enhanced statistics
    df['enhanced_significant'] = (df['enhanced_fisher_p'] < 0.05) & (df['enhanced_fisher_p'].notna())
    df['baseline_significant'] = (df['baseline_binomial_p'] < 0.05) & (df['baseline_binomial_p'].notna())
    
    # Multiple testing correction (fast version)
    print("Applying multiple testing correction...")
    df = add_fast_multiple_testing_correction(df)
    
    return df

def add_traditional_statistics(df, up_rate):
    """
    Fast addition of traditional statistics
    """
    
    # Population rate calculation (per ligand)
    ligand_stats = df.groupby('ligand').agg({
        'interaction_positive': 'sum',
        'sender_positive': 'sum'
    })
    
    population_rates = {}
    for ligand in ligand_stats.index:
        total_success = ligand_stats.loc[ligand, 'interaction_positive']
        total_trials = ligand_stats.loc[ligand, 'sender_positive']
        if total_trials > 0:
            population_rates[ligand] = total_success / total_trials
        else:
            population_rates[ligand] = 0.0
    
    # Map to each row
    df['population_mean_rate'] = df['ligand'].map(population_rates)
    expected_rates = up_rate * df['population_mean_rate']
    
    # Binomial test (vectorized)
    valid_mask = (df['sender_positive'] > 0) & (expected_rates <= 1.0) & (expected_rates > 0)
    
    p_values = np.full(len(df), np.nan)
    
    if np.any(valid_mask):
        # Use normal approximation
        successes = df.loc[valid_mask, 'interaction_positive'].values
        trials = df.loc[valid_mask, 'sender_positive'].values  
        rates = expected_rates.loc[valid_mask].values
        
        expected = trials * rates
        std_dev = np.sqrt(trials * rates * (1 - rates))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            z_stats = np.divide(successes - expected, std_dev,
                                  out=np.zeros_like(successes, dtype=float),
                                  where=std_dev > 0)
        
        p_approx = 1 - stats.norm.cdf(z_stats)  # one-sided test (right)
        p_values[valid_mask] = p_approx
    
    df['p_value'] = p_values
    df['is_significant'] = (p_values < 0.05) & ~np.isnan(p_values)
    
    # Beta confidence interval (vectorized)
    alpha_post = df['interaction_positive'] + 0.5
    beta_post = df['sender_positive'] - df['interaction_positive'] + 0.5
    
    alpha_post = np.maximum(alpha_post, 0.5)
    beta_post = np.maximum(beta_post, 0.5)
    
    df['ci_lower_beta'] = beta.ppf(0.025, alpha_post, beta_post)
    df['ci_upper_beta'] = beta.ppf(0.975, alpha_post, beta_post)
    
    return df

def add_fast_multiple_testing_correction(df):
    """
    Fast multiple testing correction
    """
    
    # Traditional p-values
    valid_p = df['p_value'].dropna()
    if len(valid_p) > 0:
        corrected = multipletests(valid_p, method='bonferroni')
        df.loc[df['p_value'].notna(), 'p_value_bonferroni'] = corrected[1]
        df.loc[df['p_value'].notna(), 'is_significant_bonferroni'] = corrected[0]
    else:
        df['p_value_bonferroni'] = np.nan
        df['is_significant_bonferroni'] = False
    
    # Enhanced p-values
    valid_enhanced_p = df['enhanced_fisher_p'].dropna()
    if len(valid_enhanced_p) > 0:
        corrected_enhanced = multipletests(valid_enhanced_p, method='bonferroni')
        df.loc[df['enhanced_fisher_p'].notna(), 'enhanced_fisher_p_bonferroni'] = corrected_enhanced[1]
        df.loc[df['enhanced_fisher_p'].notna(), 'enhanced_significant_bonferroni'] = corrected_enhanced[0]
    else:
        df['enhanced_fisher_p_bonferroni'] = np.nan
        df['enhanced_significant_bonferroni'] = False
    
    return df

def display_top_interactions_by_cell_type(coexp_cc_df, enhancement_threshold=1.25, top_n=10, 
                                          min_responses_with_sender=5, min_responses_without_sender=5):
    """
    Display top interactions by cell type
    
    Parameters:
    -----------
    coexp_cc_df : DataFrame
        DataFrame of analysis results
    enhancement_threshold : float
        Threshold for interaction enhancement effect (default: 1.25)
    top_n : int
        Number of top interactions to display for each cell type (default: 10)
    min_responses_with_sender : int
        Minimum number of responses with sender (default: 5)
    min_responses_without_sender : int
        Minimum number of responses without sender (default: 5)
    """
    
    # Extract significant interaction enhancements (with added condition for minimum responses)
    significant_interactions = coexp_cc_df[
        (coexp_cc_df['enhanced_significant'] == True) &
        (coexp_cc_df['enhanced_odds_ratio'] >= enhancement_threshold) &  # Odds ratio enhancement of 1.25 or more
        (coexp_cc_df.get('sender_pos_receiver_pos', 0) >= min_responses_with_sender) &  # Min responses with sender
        (coexp_cc_df.get('sender_neg_receiver_pos', 0) >= min_responses_without_sender)  # Min responses without sender
    ].copy()
    
    if len(significant_interactions) == 0:
        print(f"No significant interactions found with the specified criteria:")
        print(f"   - Enhancement threshold: >{enhancement_threshold}")
        print(f"   - Min responses with sender: >={min_responses_with_sender}")
        print(f"   - Min responses without sender: >={min_responses_without_sender}")
        return
    
    print(f"=== Top Enhanced Interactions (Odds Ratio >= {enhancement_threshold}) ===")
    print(f"Filter criteria:")
    print(f"   - Minimum responses with sender: >= {min_responses_with_sender}")
    print(f"   - Minimum responses without sender: >= {min_responses_without_sender}")
    print(f"Total significant interactions: {len(significant_interactions)}")
    print()
    
    # Group and process by receiver cell type
    receiver_groups = significant_interactions.groupby('cell1_type' if 'cell1_type' in significant_interactions.columns else 'receiver_cell_type')
    
    
    for receiver_type, group in receiver_groups:
        print(f"📱 Receiver Cell Type: {receiver_type}")
        print("-" * 60)
        
        # Sort by interaction enhancement effect and get top results
        top_interactions = group.nlargest(top_n, 'enhanced_odds_ratio')
        
        for i, (_, row) in enumerate(top_interactions.iterrows(), 1):
            sender_type = row['cell2_type'] if 'cell2_type' in row else row['sender_cell_type']
            receiver_type_display = row['cell1_type'] if 'cell1_type' in row else row['receiver_cell_type']
            ligand = row['ligand']
            
            print(f"   {i:2d}. {sender_type} -> {receiver_type_display} ({ligand})")
            print(f"         Response rate with sender: {row['cond_prob_receiver_given_sender']:.3f}")
            print(f"         Response rate without sender: {row['cond_prob_receiver_given_not_sender']:.3f}")
            print(f"         Enhancement: +{row['interaction_enhancement']:.3f} (Odds ratio: {row['enhanced_odds_ratio']:.2f})")
            
            # Additional statistical information
            if 'enhanced_fisher_p' in row and not pd.isna(row['enhanced_fisher_p']):
                print(f"         p-value: {row['enhanced_fisher_p']:.2e}")
            
            # Also display actual observed counts
            if 'sender_pos_receiver_pos' in row:
                total_with_sender = row.get('sender_positive', 'N/A')
                responded_with_sender = row.get('sender_pos_receiver_pos', 'N/A')
                responded_without_sender = row.get('sender_neg_receiver_pos', 'N/A')
                total_without_sender = row.get('total_interactions', 0) - row.get('sender_positive', 0) if 'total_interactions' in row else 'N/A'
                
                print(f"         Observed counts: With sender({responded_with_sender}/{total_with_sender}), Without sender({responded_without_sender}/{total_without_sender})")
            
            print()
        
        print()

def display_summary_statistics(coexp_cc_df, enhancement_threshold=1.25,
                               min_responses_with_sender=5, min_responses_without_sender=5):
    """
    Display summary statistics of interactions
    """
    
    significant_interactions = coexp_cc_df[
        (coexp_cc_df['enhanced_significant'] == True) &
        (coexp_cc_df['enhanced_odds_ratio'] >= enhancement_threshold) &
        (coexp_cc_df.get('sender_pos_receiver_pos', 0) >= min_responses_with_sender) &
        (coexp_cc_df.get('sender_neg_receiver_pos', 0) >= min_responses_without_sender)
    ]
    
    print("=== Summary Statistics ===")
    print(f"Total interactions: {len(coexp_cc_df)}")
    print(f"Significant enhanced interactions: {len(significant_interactions)} ({len(significant_interactions)/len(coexp_cc_df):.1%})")
    print()
    
    if len(significant_interactions) > 0:
        # Statistics by receiver cell type
        # ★★★ CORRECTED: Group by cell1_type for receivers ★★★
        # ★★★ 修正点: Receiverは cell1_type でグループ化 ★★★
        receiver_stats = significant_interactions.groupby('cell1_type' if 'cell1_type' in significant_interactions.columns else 'receiver_cell_type').agg({
            'enhanced_odds_ratio': ['count', 'mean', 'max'],
            'cond_prob_receiver_given_sender': 'mean',
            'cond_prob_receiver_given_not_sender': 'mean'
        }).round(3)
        
        print("Statistics by Receiver Cell Type:")
        print(receiver_stats)
        print()
        
        # Statistics by sender cell type
        # ★★★ CORRECTED: Group by cell2_type for senders ★★★
        # ★★★ 修正点: Senderは cell2_type でグループ化 ★★★
        sender_stats = significant_interactions.groupby('cell2_type' if 'cell2_type' in significant_interactions.columns else 'sender_cell_type').agg({
            'enhanced_odds_ratio': ['count', 'mean', 'max'],
            'cond_prob_receiver_given_sender': 'mean'
        }).round(3)
        
        print("Statistics by Sender Cell Type:")
        print(sender_stats)
        print()
        
        # Statistics by ligand
        ligand_stats = significant_interactions.groupby('ligand').agg({
            'enhanced_odds_ratio': ['count', 'mean', 'max']
        }).round(3)
        
        print("Top Ligands (by number of interactions):")
        print(ligand_stats.sort_values(('enhanced_odds_ratio', 'count'), ascending=False).head(10))
        
def display_high_spontaneous_responses(coexp_cc_df, spontaneous_threshold=0.1, 
                                       min_responses_without_sender=5):
    """
    Display combinations with high spontaneous responses
    """
    
    high_spontaneous = coexp_cc_df[
        (coexp_cc_df['cond_prob_receiver_given_not_sender'] > spontaneous_threshold) &
        (coexp_cc_df.get('sender_neg_receiver_pos', 0) >= min_responses_without_sender)  # Minimum response count
    ].copy()
    
    if len(high_spontaneous) > 0:
        high_spontaneous = high_spontaneous.sort_values('cond_prob_receiver_given_not_sender', ascending=False)
        
        print(f"=== High Spontaneous Responses (Response rate w/o sender > {spontaneous_threshold:.1%}) ===")
        
        for _, row in high_spontaneous.head(20).iterrows():
            receiver_type = row['cell2_type'] if 'cell2_type' in row else row['receiver_cell_type']
            ligand = row['ligand']
            spontaneous_rate = row['cond_prob_receiver_given_not_sender']
            
            print(f"{receiver_type} responds to {ligand}: {spontaneous_rate:.3f} ({spontaneous_rate:.1%}) without sender")
            
            if 'sender_neg_receiver_pos' in row:
                responded = row['sender_neg_receiver_pos']
                total_without = row.get('total_interactions', 0) - row.get('sender_positive', 0)
                print(f"     Observed counts: {responded}/{total_without}")
        
        print()

def display_inhibitory_effects(coexp_cc_df, inhibition_threshold=-0.05,
                               min_responses_with_sender=5, min_responses_without_sender=5):
    """
    Display interactions showing inhibitory effects
    """
    
    inhibitory_effects = coexp_cc_df[
        (coexp_cc_df['interaction_enhancement'] < inhibition_threshold) &
        (coexp_cc_df.get('enhanced_significant', False) == True) &
        (coexp_cc_df.get('sender_pos_receiver_pos', 0) >= min_responses_with_sender) &
        (coexp_cc_df.get('sender_neg_receiver_pos', 0) >= min_responses_without_sender)
    ].copy()
    
    if len(inhibitory_effects) > 0:
        inhibitory_effects = inhibitory_effects.sort_values('interaction_enhancement', ascending=True)
        
        print(f"=== Inhibitory Effects (Enhancement < {inhibition_threshold:.1%}) ===")
        
        for _, row in inhibitory_effects.head(10).iterrows():
            sender_type = row['cell1_type'] if 'cell1_type' in row else row['sender_cell_type']
            receiver_type = row['cell2_type'] if 'cell2_type' in row else row['receiver_cell_type']
            ligand = row['ligand']
            inhibition = row['interaction_enhancement']
            
            print(f"{sender_type} inhibits {receiver_type} response to {ligand}: {inhibition:.3f} ({inhibition:.1%})")
            print(f"     Response rate with sender: {row['cond_prob_receiver_given_sender']:.3f}")
            print(f"     Response rate without sender: {row['cond_prob_receiver_given_not_sender']:.3f}")
        
        print()

# Example usage
def comprehensive_interaction_analysis(coexp_cc_df, enhancement_threshold=0.02, spontaneous_threshold=0.1,
                                     inhibition_threshold=-0.05, min_responses_with_sender=5, min_responses_without_sender=5):
    """
    Run comprehensive interaction analysis
    """
    
    print("🔬 Comprehensive Cell-Cell Interaction Analysis")
    print("=" * 80)
    
    # 1. Summary statistics
    display_summary_statistics(coexp_cc_df, enhancement_threshold=1.25,
                               min_responses_with_sender=min_responses_with_sender,
                               min_responses_without_sender=min_responses_without_sender)
    
    # 2. Top interactions by cell type (Odds ratio >= enhancement_threshold, 5 each)
    display_top_interactions_by_cell_type(coexp_cc_df, enhancement_threshold=1.25, top_n=5,
                                          min_responses_with_sender=min_responses_with_sender,
                                          min_responses_without_sender=min_responses_without_sender)
    
    # 3. High spontaneous responses
    display_high_spontaneous_responses(coexp_cc_df, spontaneous_threshold=0.1,
                                       min_responses_without_sender=min_responses_without_sender)
    
    # 4. Inhibitory effects
    display_inhibitory_effects(coexp_cc_df, inhibition_threshold=-0.05,
                               min_responses_with_sender=min_responses_with_sender,
                               min_responses_without_sender=min_responses_without_sender)


def calculate_enhanced_coexpression_coactivity_cluster(edge_df, center_adata, exp_data, expr_up_by_ligands, 
                                                     sp_adata, cluster_label, role="receiver", up_rate=1.25):
    """
    Fast, cluster-specific enhanced co-expression analysis
    
    Parameters:
    -----------
    edge_df : DataFrame
        Edge information
    center_adata : AnnData
        Center cell data
    exp_data : ndarray
        Expression data
    expr_up_by_ligands : ndarray
        Ligand response data
    sp_adata : AnnData
        Spatial transcriptomics data (for baseline calculation)
    cluster_label : str or list
        Target cluster(s)
    role : str
        "receiver" or "sender"
    up_rate : float
        Multiplier for expected value
    """
    
    print(f"Fast enhanced cluster-specific analysis for cluster: {cluster_label}")
    
    # Data preparation (optimized)
    center_adata.X = exp_data
    gene_names = center_adata.var_names.tolist()
    n_genes = len(gene_names)
    
    # Cluster filtering (fast)
    if isinstance(cluster_label, (list, tuple)):
        cluster_set = set(str(c) for c in cluster_label)
    else:
        cluster_set = {str(cluster_label)}
    
    # Identification of valid cells (vectorized)
    cell_clusters = center_adata.obs['cluster'].astype(str)
    cluster_mask = cell_clusters.isin(cluster_set)
    valid_cell1_indices = set(center_adata.obs_names[cluster_mask])
    
    # Edge filtering
    edge_mask = edge_df['cell1'].isin(valid_cell1_indices)
    filtered_edge_df = edge_df[edge_mask].copy()
    
    if len(filtered_edge_df) == 0:
        print(f"Warning: No interactions found for cluster {cluster_label}")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Processing {len(filtered_edge_df)} edges for cluster analysis...")
    
    # Set role
    if role == "receiver":
        actual_sender = filtered_edge_df.cell2.values
        actual_receiver = filtered_edge_df.cell1.values
        sender_type_col = 'cell2_type'
        receiver_type_col = 'cell1_type'
    else:
        actual_sender = filtered_edge_df.cell1.values
        actual_receiver = filtered_edge_df.cell2.values
        sender_type_col = 'cell1_type'
        receiver_type_col = 'cell2_type'
    
    # Pre-calculation of index mapping
    cell_to_idx = {cell: idx for idx, cell in enumerate(center_adata.obs_names)}
    sender_indices = np.array([cell_to_idx[cell] for cell in actual_sender])
    receiver_indices = np.array([cell_to_idx[cell] for cell in actual_receiver])
    
    # Get expression data (vectorized)
    sender_expr = exp_data[sender_indices]
    receiver_expr = expr_up_by_ligands[receiver_indices]
    
    if hasattr(sender_expr, 'toarray'):
        sender_expr = sender_expr.toarray()
    if hasattr(receiver_expr, 'toarray'):
        receiver_expr = receiver_expr.toarray()
    
    # Co-expression calculation (vectorized)
    coexp_matrix = sender_expr * receiver_expr
    
    # Cell type encoding (fast)
    sender_types = filtered_edge_df[sender_type_col].values
    receiver_types = filtered_edge_df[receiver_type_col].values
    
    unique_sender_types = np.unique(sender_types)
    unique_receiver_types = np.unique(receiver_types)
    
    sender_type_to_idx = {t: i for i, t in enumerate(unique_sender_types)}
    receiver_type_to_idx = {t: i for i, t in enumerate(unique_receiver_types)}
    
    sender_type_encoded = np.array([sender_type_to_idx[t] for t in sender_types])
    receiver_type_encoded = np.array([receiver_type_to_idx[t] for t in receiver_types])
    
    # Baseline calculation (cluster-specific)
    print("Calculating cluster-specific baselines...")
    baseline_rates = fast_calculate_cluster_baseline_rates(sp_adata, expr_up_by_ligands, gene_names, cluster_label)
    
    # Large-scale contingency table calculation (vectorized)
    print("Computing contingency tables...")
    results_data = fast_compute_cluster_contingency_tables(
        sender_expr, receiver_expr, sender_type_encoded, receiver_type_encoded,
        unique_sender_types, unique_receiver_types, gene_names, baseline_rates
    )
    
    # Formatting results to existing format
    print("Formatting results...")
    coexp_cc_df = format_cluster_results_to_existing_format(
        results_data, unique_sender_types, unique_receiver_types, gene_names, 
        sender_type_col, receiver_type_col, role, up_rate
    )
    
    # Create bargraph_df (existing format)
    bargraph_data = {
        receiver_type_col: receiver_types,
        sender_type_col: sender_types
    }
    
    for i, gene in enumerate(gene_names):
        bargraph_data[gene] = coexp_matrix[:, i]
    
    bargraph_df = pd.DataFrame(bargraph_data, index=filtered_edge_df.index)
    
    # Result summary
    n_significant = np.sum(coexp_cc_df['is_significant'])
    n_enhanced_significant = np.sum(coexp_cc_df.get('enhanced_significant', False))
    n_baseline_significant = np.sum(coexp_cc_df.get('baseline_significant', False))
    
    print(f"Completed cluster analysis: {len(coexp_cc_df)} interactions")
    print(f"Traditional: {n_significant} significant")
    print(f"Enhanced: {n_enhanced_significant} significant with baseline consideration")
    print(f"Baseline: {n_baseline_significant} significant vs cluster baseline")
    
    return coexp_cc_df, bargraph_df

def fast_calculate_cluster_baseline_rates(sp_adata, expr_up_by_ligands, gene_names, cluster_label):
    """
    Fast calculation of cluster-specific baseline response rates
    """
    baseline_rates = {}
    
    # Calculate baseline only with cells from the specified cluster
    if isinstance(cluster_label, (list, tuple)):
        cluster_set = set(str(c) for c in cluster_label)
    else:
        cluster_set = {str(cluster_label)}
    
    # Create cluster mask
    cluster_mask = sp_adata.obs['cluster'].astype(str).isin(cluster_set)
    cluster_cells = sp_adata[cluster_mask]
    
    if len(cluster_cells) == 0:
        print(f"Warning: No cells found for cluster {cluster_label}")
        return {}
    
    # Per-cell-type processing within the cluster
    cell_types = cluster_cells.obs['celltype'].unique()
    celltype_values = cluster_cells.obs['celltype'].values
    
    # Get expression data within the cluster
    cluster_indices = np.where(cluster_mask)[0]
    cluster_expr_up = expr_up_by_ligands[cluster_indices]
    
    for cell_type in cell_types:
        cell_mask_in_cluster = celltype_values == cell_type
        
        if not np.any(cell_mask_in_cluster):
            continue
        
        # Expression data for that cell type (only within the cluster)
        cell_expr = cluster_expr_up[cell_mask_in_cluster]
        
        # Batch calculation of response rates for all ligands
        response_rates = np.mean(cell_expr > 0, axis=0)
        
        # Store in dictionary
        baseline_rates[cell_type] = dict(zip(gene_names[:len(response_rates)], response_rates))
    
    return baseline_rates

def fast_compute_cluster_contingency_tables(sender_expr, receiver_expr, sender_type_encoded, 
                                            receiver_type_encoded, unique_sender_types, 
                                            unique_receiver_types, gene_names, baseline_rates):
    """
    Fast calculation of cluster-specific contingency tables
    """
    n_sender_types = len(unique_sender_types)
    n_receiver_types = len(unique_receiver_types)
    n_genes = len(gene_names)
    
    # For storing results
    results_data = []
    
    # Process for each sender-receiver cell combination
    for s_idx, sender_type in enumerate(unique_sender_types):
        for r_idx, receiver_type in enumerate(unique_receiver_types):
            
            # Extract edges for this combination
            mask = (sender_type_encoded == s_idx) & (receiver_type_encoded == r_idx)
            
            if not np.any(mask):
                continue
            
            # Expression data for this combination
            s_expr_subset = sender_expr[mask]
            r_expr_subset = receiver_expr[mask]
            
            # Batch calculation of contingency tables for all genes (vectorized)
            contingency_stats = compute_vectorized_cluster_contingency_stats(
                s_expr_subset, r_expr_subset, gene_names, baseline_rates.get(receiver_type, {})
            )
            
            # Add to results
            for gene_idx, gene in enumerate(gene_names):
                stats_dict = {k: v[gene_idx] if hasattr(v, '__len__') else v for k, v in contingency_stats.items()}
                stats_dict.update({
                    'sender_type': sender_type,
                    'receiver_type': receiver_type,
                    'ligand': gene
                })
                results_data.append(stats_dict)
    
    return results_data

def compute_vectorized_cluster_contingency_stats(sender_expr, receiver_expr, gene_names, baseline_dict):
    """
    Vectorized contingency table statistics for clusters
    """
    n_edges, n_genes = sender_expr.shape
    
    # Binarization (vectorized)
    sender_binary = sender_expr > 0
    receiver_binary = receiver_expr > 0
    
    # Batch calculation of four situations
    sender_pos_receiver_pos = np.sum(sender_binary & receiver_binary, axis=0)
    sender_pos_receiver_neg = np.sum(sender_binary & ~receiver_binary, axis=0)
    sender_neg_receiver_pos = np.sum(~sender_binary & receiver_binary, axis=0)
    sender_neg_receiver_neg = np.sum(~sender_binary & ~receiver_binary, axis=0)
    
    # Basic statistics
    sender_positive_count = sender_pos_receiver_pos + sender_pos_receiver_neg
    sender_negative_count = sender_neg_receiver_pos + sender_neg_receiver_neg
    
    # Conditional probability (with zero-division protection)
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_prob_r_given_s = np.divide(sender_pos_receiver_pos, sender_positive_count, 
                                          out=np.zeros_like(sender_pos_receiver_pos, dtype=float),
                                          where=sender_positive_count>0)
        
        cond_prob_r_given_not_s = np.divide(sender_neg_receiver_pos, sender_negative_count,
                                              out=np.zeros_like(sender_neg_receiver_pos, dtype=float),
                                              where=sender_negative_count>0)
    
    interaction_enhancement = cond_prob_r_given_s - cond_prob_r_given_not_s
    
    # Baseline information
    baseline_rates = np.array([baseline_dict.get(gene, 0.0) for gene in gene_names])
    
    # Fast statistical test
    fisher_p_values, odds_ratios = fast_vectorized_fisher_test_cluster(
        sender_pos_receiver_pos, sender_pos_receiver_neg,
        sender_neg_receiver_pos, sender_neg_receiver_neg
    )
    
    # Binomial test against baseline (vectorized)
    binomial_p_values = fast_vectorized_binomial_test_cluster(
        sender_pos_receiver_pos, sender_positive_count, baseline_rates
    )
    
    return {
        'total_interactions': n_edges,
        'sender_positive': sender_positive_count,
        'interaction_positive': sender_pos_receiver_pos,
        'sender_pos_receiver_pos': sender_pos_receiver_pos,
        'sender_pos_receiver_neg': sender_pos_receiver_neg,
        'sender_neg_receiver_pos': sender_neg_receiver_pos,
        'sender_neg_receiver_neg': sender_neg_receiver_neg,
        'cond_prob_receiver_given_sender': cond_prob_r_given_s,
        'cond_prob_receiver_given_not_sender': cond_prob_r_given_not_s,
        'interaction_enhancement': interaction_enhancement,
        'baseline_response_rate': baseline_rates,
        'enhanced_fisher_p': fisher_p_values,
        'enhanced_odds_ratio': odds_ratios,
        'baseline_binomial_p': binomial_p_values
    }

def fast_vectorized_fisher_test_cluster(a, b, c, d):
    """
    Vectorized Fisher test for clusters
    """
    n_genes = len(a)
    p_values = np.full(n_genes, np.nan)
    odds_ratios = np.full(n_genes, np.nan)
    
    # Mask for valid cases
    valid_mask = (a + b + c + d) > 0
    
    if np.any(valid_mask):
        a_valid = a[valid_mask]
        b_valid = b[valid_mask]
        c_valid = c[valid_mask]
        d_valid = d[valid_mask]
        
        # Odds ratio calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            or_values = np.divide(a_valid * d_valid, b_valid * c_valid,
                                    out=np.full_like(a_valid, np.inf, dtype=float),
                                    where=(b_valid * c_valid) > 0)
        
        # Chi-square approximation
        n_total = a_valid + b_valid + c_valid + d_valid
        expected_a = (a_valid + b_valid) * (a_valid + c_valid) / n_total
        
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2_stats = np.divide((a_valid - expected_a) ** 2, expected_a,
                                     out=np.zeros_like(expected_a),
                                     where=expected_a > 0)
        
        # p-value approximation
        p_approx = 1 - stats.chi2.cdf(chi2_stats, df=1)
        
        p_values[valid_mask] = p_approx
        odds_ratios[valid_mask] = or_values
    
    return p_values, odds_ratios

def fast_vectorized_binomial_test_cluster(successes, trials, baseline_rates, alpha=0.05):
    """
    Vectorized Binomial test for clusters
    """
    n_genes = len(successes)
    p_values = np.full(n_genes, np.nan)
    
    # Mask for valid cases
    valid_mask = (trials > 0) & (baseline_rates > 0) & (baseline_rates < 1)
    
    if np.any(valid_mask):
        # Use normal approximation
        s_valid = successes[valid_mask]
        t_valid = trials[valid_mask] 
        r_valid = baseline_rates[valid_mask]
        
        # Expected value and standard deviation
        expected = t_valid * r_valid
        std_dev = np.sqrt(t_valid * r_valid * (1 - r_valid))
        
        # Z-statistic
        with np.errstate(divide='ignore', invalid='ignore'):
            z_stats = np.divide(s_valid - expected, std_dev,
                                  out=np.zeros_like(s_valid, dtype=float),
                                  where=std_dev > 0)
        
        # p-value for two-sided test
        p_approx = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
        
        p_values[valid_mask] = p_approx
    
    return p_values

def format_cluster_results_to_existing_format(results_data, unique_sender_types, unique_receiver_types, 
                                              gene_names, sender_type_col, receiver_type_col, role, up_rate):
    """
    Format cluster results to existing format
    """
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Adjust column names for existing format
    if role == "receiver":
        df['cell1_type'] = df['receiver_type'] 
        df['cell2_type'] = df['sender_type']
    else:
        df['cell1_type'] = df['sender_type']
        df['cell2_type'] = df['receiver_type']
    
    # Add basic statistics
    df['coactivity_per_sender_cell_expr_ligand'] = np.divide(
        df['interaction_positive'], df['sender_positive'],
        out=np.zeros_like(df['interaction_positive'], dtype=float),
        where=df['sender_positive'] > 0
    )
    
    # Traditional statistical test (simplified version)
    print("Computing traditional statistics...")
    df = add_traditional_statistics_cluster(df, up_rate)
    
    # Significance determination for enhanced statistics
    df['enhanced_significant'] = (df['enhanced_fisher_p'] < 0.05) & (df['enhanced_fisher_p'].notna())
    df['baseline_significant'] = (df['baseline_binomial_p'] < 0.05) & (df['baseline_binomial_p'].notna())
    
    # Multiple testing correction (fast version)
    print("Applying multiple testing correction...")
    df = add_fast_multiple_testing_correction_cluster(df)
    
    return df

def add_traditional_statistics_cluster(df, up_rate):
    """
    Fast addition of traditional statistics for clusters
    """
    
    # Population rate calculation (per ligand)
    ligand_stats = df.groupby('ligand').agg({
        'interaction_positive': 'sum',
        'sender_positive': 'sum'
    })
    
    population_rates = {}
    for ligand in ligand_stats.index:
        total_success = ligand_stats.loc[ligand, 'interaction_positive']
        total_trials = ligand_stats.loc[ligand, 'sender_positive']
        if total_trials > 0:
            population_rates[ligand] = total_success / total_trials
        else:
            population_rates[ligand] = 0.0
    
    # Map to each row
    df['population_mean_rate'] = df['ligand'].map(population_rates)
    expected_rates = up_rate * df['population_mean_rate']
    
    # Binomial test (vectorized)
    valid_mask = (df['sender_positive'] > 0) & (expected_rates <= 1.0) & (expected_rates > 0)
    
    p_values = np.full(len(df), np.nan)
    
    if np.any(valid_mask):
        # Use normal approximation
        successes = df.loc[valid_mask, 'interaction_positive'].values
        trials = df.loc[valid_mask, 'sender_positive'].values  
        rates = expected_rates.loc[valid_mask].values
        
        expected = trials * rates
        std_dev = np.sqrt(trials * rates * (1 - rates))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            z_stats = np.divide(successes - expected, std_dev,
                                  out=np.zeros_like(successes, dtype=float),
                                  where=std_dev > 0)
        
        p_approx = 1 - stats.norm.cdf(z_stats)
        p_values[valid_mask] = p_approx
    
    df['p_value'] = p_values
    df['is_significant'] = (p_values < 0.05) & ~np.isnan(p_values)
    
    # Beta confidence interval (vectorized)
    alpha_post = df['interaction_positive'] + 0.5
    beta_post = df['sender_positive'] - df['interaction_positive'] + 0.5
    
    alpha_post = np.maximum(alpha_post, 0.5)
    beta_post = np.maximum(beta_post, 0.5)
    
    df['ci_lower_beta'] = beta.ppf(0.025, alpha_post, beta_post)
    df['ci_upper_beta'] = beta.ppf(0.975, alpha_post, beta_post)
    
    return df

def add_fast_multiple_testing_correction_cluster(df):
    """
    Fast multiple testing correction for clusters
    """
    
    # Traditional p-values
    valid_p = df['p_value'].dropna()
    if len(valid_p) > 0:
        corrected = multipletests(valid_p, method='bonferroni')
        df.loc[df['p_value'].notna(), 'p_value_bonferroni'] = corrected[1]
        df.loc[df['p_value'].notna(), 'is_significant_bonferroni'] = corrected[0]
    else:
        df['p_value_bonferroni'] = np.nan
        df['is_significant_bonferroni'] = False
    
    # Enhanced p-values
    valid_enhanced_p = df['enhanced_fisher_p'].dropna()
    if len(valid_enhanced_p) > 0:
        corrected_enhanced = multipletests(valid_enhanced_p, method='bonferroni')
        df.loc[df['enhanced_fisher_p'].notna(), 'enhanced_fisher_p_bonferroni'] = corrected_enhanced[1]
        df.loc[df['enhanced_fisher_p'].notna(), 'enhanced_significant_bonferroni'] = corrected_enhanced[0]
    else:
        df['enhanced_fisher_p_bonferroni'] = np.nan
        df['enhanced_significant_bonferroni'] = False
    
    # Baseline p-values
    valid_baseline_p = df['baseline_binomial_p'].dropna()
    if len(valid_baseline_p) > 0:
        corrected_baseline = multipletests(valid_baseline_p, method='bonferroni')
        df.loc[df['baseline_binomial_p'].notna(), 'baseline_binomial_p_bonferroni'] = corrected_baseline[1]
        df.loc[df['baseline_binomial_p'].notna(), 'baseline_significant_bonferroni'] = corrected_baseline[0]
    else:
        df['baseline_binomial_p_bonferroni'] = np.nan
        df['baseline_significant_bonferroni'] = False
    
    return df

def compute_detailed_cluster_analysis(celltype_cluster_data):
    """
    Detailed analysis of cell type x cluster x ligand
    """
    
    detailed_stats = celltype_cluster_data.copy()
    
    # Calculation of conditional probability
    detailed_stats['response_rate_with_high_stimulation'] = np.divide(
        detailed_stats['interaction_positive'],
        detailed_stats['high_stimulation_environment'],
        out=np.zeros_like(detailed_stats['interaction_positive'], dtype=float),
        where=detailed_stats['high_stimulation_environment'] > 0
    )
    
    # Calculation of response rate in low stimulation environment
    detailed_stats['low_stimulation_responses'] = (
        detailed_stats['center_cell_response'] - detailed_stats['interaction_positive']
    )
    detailed_stats['low_stimulation_opportunities'] = (
        detailed_stats['total_observations'] - detailed_stats['high_stimulation_environment']
    )
    
    detailed_stats['response_rate_with_low_stimulation'] = np.divide(
        detailed_stats['low_stimulation_responses'],
        detailed_stats['low_stimulation_opportunities'],
        out=np.zeros_like(detailed_stats['low_stimulation_responses'], dtype=float),
        where=detailed_stats['low_stimulation_opportunities'] > 0
    )
    
    # Stimulation enhancement effect
    detailed_stats['stimulation_enhancement'] = (
        detailed_stats['response_rate_with_high_stimulation'] - 
        detailed_stats['response_rate_with_low_stimulation']
    )
    
    # Simplified statistical test (fast version for large data)
    detailed_stats['is_significant'] = (
        (detailed_stats['high_stimulation_environment'] >= 5) &
        (detailed_stats['low_stimulation_opportunities'] >= 5) &
        (detailed_stats['stimulation_enhancement'] > 0.01)  # 1% or more enhancement effect
    )
    
    return detailed_stats

def compute_detailed_cluster_analysis(celltype_cluster_data):
    """
    Detailed analysis of cell type x cluster x ligand
    """
    
    detailed_stats = celltype_cluster_data.copy()
    
    # Calculation of conditional probability
    detailed_stats['response_rate_with_high_stimulation'] = np.divide(
        detailed_stats['interaction_positive'],
        detailed_stats['high_stimulation_environment'],
        out=np.zeros_like(detailed_stats['interaction_positive'], dtype=float),
        where=detailed_stats['high_stimulation_environment'] > 0
    )
    
    # Calculation of response rate in low stimulation environment
    detailed_stats['low_stimulation_responses'] = (
        detailed_stats['center_cell_response'] - detailed_stats['interaction_positive']
    )
    detailed_stats['low_stimulation_opportunities'] = (
        detailed_stats['total_observations'] - detailed_stats['high_stimulation_environment']
    )
    
    detailed_stats['response_rate_with_low_stimulation'] = np.divide(
        detailed_stats['low_stimulation_responses'],
        detailed_stats['low_stimulation_opportunities'],
        out=np.zeros_like(detailed_stats['low_stimulation_responses'], dtype=float),
        where=detailed_stats['low_stimulation_opportunities'] > 0
    )
    
    # Stimulation enhancement effect
    detailed_stats['stimulation_enhancement'] = (
        detailed_stats['response_rate_with_high_stimulation'] - 
        detailed_stats['response_rate_with_low_stimulation']
    )
    
    # Simplified statistical test (fast version for large data)
    detailed_stats['is_significant'] = (
        (detailed_stats['high_stimulation_environment'] >= 5) &
        (detailed_stats['low_stimulation_opportunities'] >= 5) &
        (detailed_stats['stimulation_enhancement'] > 0.01)  # 1% or more enhancement effect
    )
    
    return detailed_stats

def calculate_cumulative_ligand_coexpression_analysis(edge_df, center_adata, exp_data, expr_up_by_ligands, 
                                                      sp_adata, neighbor_cell_numbers=19, 
                                                      top_percentile_threshold=1.0, role="receiver", 
                                                      up_rate=1.25):
    """
    Cell-cell interaction analysis by cumulative ligand stimulation
    """
    
    print(f"Cumulative ligand stimulation analysis (top {top_percentile_threshold}% threshold)")
    
    # Data preparation
    center_adata.X = exp_data
    gene_names = center_adata.var_names.tolist()
    n_genes = len(gene_names)
    
    # Reconstruct edges: group by center cell
    print("Reconstructing neighborhood relationships...")
    neighborhood_data = reconstruct_neighborhoods(edge_df, neighbor_cell_numbers)
    
    # Calculation of cumulative ligand expression
    print("Computing cumulative ligand expressions...")
    cumulative_ligand_expr = compute_cumulative_ligand_expression(
        neighborhood_data, center_adata, exp_data, gene_names
    )
    
    # Definition of high stimulation environment (Top percentile)
    print(f"Defining high stimulation environments (top {top_percentile_threshold}%)...")
    high_stimulation_mask = define_high_stimulation_environments(
        cumulative_ligand_expr, top_percentile_threshold
    )
    
    # Response data of center cells
    center_cell_responses = get_center_cell_responses(
        neighborhood_data, expr_up_by_ligands, gene_names
    )
    
    # Interaction analysis by cumulative stimulation
    print("Analyzing cumulative stimulation interactions...")
    interaction_results = analyze_cumulative_interactions(
        neighborhood_data, cumulative_ligand_expr, high_stimulation_mask,
        center_cell_responses, gene_names, top_percentile_threshold
    )
    
    # Comparative analysis by cell type and microenvironment cluster
    print("Performing cell type and microenvironment cluster analysis...")
    celltype_analysis = perform_celltype_cluster_analysis(
        interaction_results, neighborhood_data, sp_adata
    )
    
    # Statistical test and formatting
    print("Computing statistics and formatting results...")
    final_results, detailed_cluster_results = compute_cumulative_statistics_and_format(
        interaction_results, celltype_analysis, up_rate
    )
    
    # Baseline comparison
    baseline_results = compute_cumulative_baseline_comparison(
        final_results, sp_adata, expr_up_by_ligands, gene_names
    )
    
    # Integration of results (reset index and concatenate)
    final_results_clean = final_results.reset_index(drop=True)
    baseline_results_clean = baseline_results.reset_index(drop=True)
    coexp_cc_df = pd.concat([final_results_clean, baseline_results_clean], axis=1)
    
    # Creation of data equivalent to bargraph_df
    bargraph_df = create_cumulative_bargraph_data(
        neighborhood_data, cumulative_ligand_expr, center_cell_responses, gene_names
    )
    
    # Result summary
    print_cumulative_analysis_summary(coexp_cc_df, top_percentile_threshold)
    
    # Also return detailed cluster analysis results
    return coexp_cc_df, bargraph_df, detailed_cluster_results

def reconstruct_neighborhoods(edge_df, neighbor_cell_numbers):
    """
    Reconstruct neighborhood relationships from edge data
    """
    neighborhoods = {}
    
    # Group by center cell
    if 'cell1' in edge_df.columns and 'cell2' in edge_df.columns:
        # Assume cell1 is the center cell and cell2 is the neighbor cell
        grouped = edge_df.groupby('cell1')
        
        for center_cell, group in grouped:
            neighbor_cells = group['cell2'].tolist()
            neighborhoods[center_cell] = {
                'center_cell': center_cell,
                'neighbor_cells': neighbor_cells[:neighbor_cell_numbers],  # max 19
                'center_cell_type': group['cell1_type'].iloc[0],
                'neighbor_cell_types': group['cell2_type'].tolist()[:neighbor_cell_numbers],
                'microenvironment_cluster': group.get('cell1_cluster', pd.Series(['unknown'] * len(group))).iloc[0],  # use cell1_cluster
                'edge_indices': group.index.tolist()[:neighbor_cell_numbers]
            }
    
    neighborhood_df = pd.DataFrame.from_dict(neighborhoods, orient='index')
    
    print(f"Reconstructed {len(neighborhood_df)} neighborhoods")
    print(f"Microenvironment clusters found: {neighborhood_df['microenvironment_cluster'].nunique()}")
    
    return neighborhood_df

def compute_cumulative_ligand_expression(neighborhood_data, center_adata, exp_data, gene_names):
    """
    Calculate cumulative ligand expression from neighbors for each center cell
    exp_data is numerical (expression values)
    """
    n_centers = len(neighborhood_data)
    n_genes = len(gene_names)
    
    cumulative_expr = np.zeros((n_centers, n_genes), dtype=np.float32)
    
    # Mapping from cell name to index
    cell_to_idx = {cell: idx for idx, cell in enumerate(center_adata.obs_names)}
    
    print(f"Computing cumulative ligand expression for {n_centers} centers, {n_genes} genes")
    print(f"exp_data shape: {exp_data.shape}, dtype: {exp_data.dtype}")
    
    for i, (center_cell, row) in enumerate(neighborhood_data.iterrows()):
        neighbor_cells = row['neighbor_cells']
        
        # Get indices of neighbor cells
        neighbor_indices = []
        for neighbor_cell in neighbor_cells:
            if neighbor_cell in cell_to_idx:
                neighbor_indices.append(cell_to_idx[neighbor_cell])
        
        if neighbor_indices:
            # Get expression data of neighbor cells (exp_data is numerical)
            neighbor_expr = exp_data[neighbor_indices]
            if hasattr(neighbor_expr, 'toarray'):
                neighbor_expr = neighbor_expr.toarray()
            
            # Calculate sum of ligand expression
            cumulative_expr[i] = np.sum(neighbor_expr, axis=0)
            
        if i % 10000 == 0:  # Progress display
            print(f"Processed {i}/{n_centers} centers")
    
    print(f"Cumulative expression computed. Mean: {np.mean(cumulative_expr):.3f}, Max: {np.max(cumulative_expr):.3f}")
    
    return cumulative_expr

def define_high_stimulation_environments(cumulative_expr, top_percentile_threshold):
    """
    Define high stimulation environments for each ligand
    cumulative_expr: numerical cumulative expression
    """
    n_centers, n_genes = cumulative_expr.shape
    high_stimulation_mask = np.zeros((n_centers, n_genes), dtype=bool)
    
    # Calculate threshold for each ligand
    percentile_threshold = 100 - top_percentile_threshold
    
    print(f"Defining high stimulation environments (top {top_percentile_threshold}%)")
    
    for gene_idx in range(n_genes):
        gene_expr = cumulative_expr[:, gene_idx]
                
        if len(gene_expr) > 0:
            threshold = np.percentile(gene_expr, percentile_threshold)
            # threshold = 0
            high_stimulation_mask[:, gene_idx] = gene_expr > threshold
            
            # Debug info (only for the first few)
            if gene_idx < 5:
                n_high = np.sum(high_stimulation_mask[:, gene_idx])
                print(f"   Gene {gene_idx}: threshold={threshold:.3f}, high_stim_cells={n_high} ({n_high/n_centers*100:.1f}%)")
        else:
            # No high stimulation environment if all are zero
            high_stimulation_mask[:, gene_idx] = False
    
    total_high_stim = np.sum(high_stimulation_mask)
    total_possible = n_centers * n_genes
    print(f"Total high stimulation environments: {total_high_stim} / {total_possible} ({total_high_stim/total_possible*100:.1f}%)")
    
    return high_stimulation_mask

def get_center_cell_responses(neighborhood_data, expr_up_by_ligands, gene_names):
    """
    Get response data of center cells
    expr_up_by_ligands is already boolean (response/no response)
    """
    n_centers = len(neighborhood_data)
    n_genes = len(gene_names)
    
    center_responses = np.zeros((n_centers, n_genes), dtype=bool)
    
    # Get indices of center cells (in order of neighborhood_data index)
    center_cell_names = neighborhood_data.index.tolist()
    
    print(f"Processing responses for {len(center_cell_names)} center cells")
    print(f"expr_up_by_ligands shape: {expr_up_by_ligands.shape}, dtype: {expr_up_by_ligands.dtype}")
    
    # Use expr_up_by_ligands as is since it's already boolean
    for i, center_cell in enumerate(center_cell_names):
        if i < expr_up_by_ligands.shape[0]:
            if hasattr(expr_up_by_ligands, 'toarray'):
                # For sparse matrices
                response_data = expr_up_by_ligands[i, :n_genes].toarray().flatten()
                center_responses[i] = response_data.astype(bool)
            else:
                # For dense matrices
                center_responses[i] = expr_up_by_ligands[i, :n_genes].astype(bool)
    
    print(f"Center responses shape: {center_responses.shape}, dtype: {center_responses.dtype}")
    print(f"Response rate: {np.mean(center_responses):.3f}")
    
    return center_responses

def analyze_cumulative_interactions(neighborhood_data, cumulative_expr, high_stimulation_mask, 
                                  center_responses, gene_names, top_percentile_threshold):
    """
    Interaction analysis by cumulative stimulation
    center_responses: boolean array (response/no response)
    cumulative_expr: float array (cumulative expression)
    """
    interaction_results = []
    
    n_centers, n_genes = cumulative_expr.shape
    
    print(f"Analyzing {n_centers} centers × {n_genes} genes = {n_centers * n_genes} interactions")
    
    for center_idx, (center_cell, row) in enumerate(neighborhood_data.iterrows()):
        center_cell_type = row['center_cell_type']
        microenv_cluster = row['microenvironment_cluster']
        
        for gene_idx, gene in enumerate(gene_names):
            # Whether it's a high stimulation environment
            high_stimulation = high_stimulation_mask[center_idx, gene_idx]
            
            # Response of the center cell (already boolean)
            center_response = center_responses[center_idx, gene_idx]
            
            # Cumulative expression (numerical)
            cumulative_value = cumulative_expr[center_idx, gene_idx]
            
            interaction_results.append({
                'center_cell': center_cell,
                'center_cell_type': center_cell_type,
                'microenvironment_cluster': microenv_cluster,
                'ligand': gene,
                'cumulative_ligand_expression': cumulative_value,
                'high_stimulation_environment': high_stimulation,
                'center_cell_response': center_response,
                'interaction_positive': high_stimulation and center_response,  # if both are True
                'stimulation_positive': high_stimulation,
                'response_positive': center_response
            })
        
        if center_idx % 10000 == 0:  # Progress display
            print(f"Analyzed {center_idx}/{n_centers} centers")
    
    results_df = pd.DataFrame(interaction_results)
    
    # Summary of results
    print(f"Interaction analysis complete:")
    print(f"   Total interactions: {len(results_df)}")
    print(f"   High stimulation environments: {results_df['high_stimulation_environment'].sum()} ({results_df['high_stimulation_environment'].mean()*100:.1f}%)")
    print(f"   Center cell responses: {results_df['center_cell_response'].sum()} ({results_df['center_cell_response'].mean()*100:.1f}%)")
    print(f"   Interaction positive: {results_df['interaction_positive'].sum()} ({results_df['interaction_positive'].mean()*100:.1f}%)")
    
    return results_df

def perform_celltype_cluster_analysis(interaction_results, neighborhood_data, sp_adata):
    """
    Analysis by cell type and microenvironment cluster
    """
    
    # Aggregation by cell type (including microenvironment_cluster)
    celltype_analysis = interaction_results.groupby(['center_cell_type', 'microenvironment_cluster', 'ligand']).agg({
        'high_stimulation_environment': 'sum',
        'center_cell_response': 'sum', 
        'interaction_positive': 'sum',
        'center_cell': 'count'  # Total count
    }).rename(columns={'center_cell': 'total_observations'}).reset_index()
    
    # Also create aggregation for microenvironment cluster only
    cluster_only_analysis = interaction_results.groupby(['microenvironment_cluster', 'ligand']).agg({
        'high_stimulation_environment': 'sum',
        'center_cell_response': 'sum',
        'interaction_positive': 'sum', 
        'center_cell': 'count'
    }).rename(columns={'center_cell': 'total_observations'}).reset_index()
    
    # Aggregation for cell type only (same as traditional)
    celltype_only_analysis = interaction_results.groupby(['center_cell_type', 'ligand']).agg({
        'high_stimulation_environment': 'sum',
        'center_cell_response': 'sum',
        'interaction_positive': 'sum',
        'center_cell': 'count'
    }).rename(columns={'center_cell': 'total_observations'}).reset_index()
    
    return {
        'celltype_cluster_analysis': celltype_analysis,  # cell type x cluster x ligand
        'cluster_analysis': cluster_only_analysis,      # cluster x ligand
        'celltype_analysis': celltype_only_analysis      # cell type x ligand (traditional)
    }

def compute_cumulative_statistics_and_format(interaction_results, celltype_analysis, up_rate):
    """
    Statistical calculation and result formatting
    """
    
    # Traditional cell-type-only analysis
    celltype_stats = celltype_analysis['celltype_analysis'].copy()
    
    # Calculation of conditional probability
    celltype_stats['response_rate_with_high_stimulation'] = np.divide(
        celltype_stats['interaction_positive'],
        celltype_stats['high_stimulation_environment'],
        out=np.zeros_like(celltype_stats['interaction_positive'], dtype=float),
        where=celltype_stats['high_stimulation_environment'] > 0
    )
    
    # Calculation of response rate in low stimulation environment
    celltype_stats['low_stimulation_responses'] = (
        celltype_stats['center_cell_response'] - celltype_stats['interaction_positive']
    )
    celltype_stats['low_stimulation_opportunities'] = (
        celltype_stats['total_observations'] - celltype_stats['high_stimulation_environment']
    )
    
    celltype_stats['response_rate_with_low_stimulation'] = np.divide(
        celltype_stats['low_stimulation_responses'],
        celltype_stats['low_stimulation_opportunities'],
        out=np.zeros_like(celltype_stats['low_stimulation_responses'], dtype=float),
        where=celltype_stats['low_stimulation_opportunities'] > 0
    )
    
    # Stimulation enhancement effect
    celltype_stats['stimulation_enhancement'] = (
        celltype_stats['response_rate_with_high_stimulation'] - 
        celltype_stats['response_rate_with_low_stimulation']
    )
    
    # Fisher exact test
    celltype_stats['fisher_p_value'] = np.nan
    celltype_stats['odds_ratio'] = np.nan
    
    for idx, row in celltype_stats.iterrows():
        # 2x2 contingency table
        high_responded = row['interaction_positive']
        high_not_responded = row['high_stimulation_environment'] - high_responded
        low_responded = row['low_stimulation_responses']
        low_not_responded = row['low_stimulation_opportunities'] - low_responded
        
        if (high_responded + high_not_responded > 0) and (low_responded + low_not_responded > 0):
            try:
                contingency_table = [[high_responded, high_not_responded], 
                                     [low_responded, low_not_responded]]
                odds_ratio, p_value = stats.fisher_exact(contingency_table)
                celltype_stats.loc[idx, 'fisher_p_value'] = p_value
                celltype_stats.loc[idx, 'odds_ratio'] = odds_ratio
            except:
                pass
    
    # Significance determination
    celltype_stats['is_significant'] = (celltype_stats['odds_ratio'] > up_rate) & \
    (celltype_stats['fisher_p_value'] < 0.05) & (celltype_stats['fisher_p_value'].notna()) & \
    (celltype_stats['interaction_positive'] >= 5)
    
    # Also create detailed analysis for cell type x cluster x ligand
    detailed_stats = compute_detailed_cluster_analysis(celltype_analysis['celltype_cluster_analysis'])
    
    return celltype_stats, detailed_stats

def compute_cumulative_baseline_comparison(results_df, sp_adata, expr_up_by_ligands, gene_names):
    """
    Baseline comparison with cumulative stimulation
    """
    
    # Create a copy of the DataFrame and reset the index
    results_clean = results_df.reset_index(drop=True).copy()
    
    # Baseline response rates by cell type
    cell_types = sp_adata.obs['celltype'].unique()
    
    baseline_rates = {}
    for cell_type in cell_types:
        cell_mask = sp_adata.obs['celltype'] == cell_type
        if np.any(cell_mask):
            cell_expr = expr_up_by_ligands[cell_mask]
            if hasattr(cell_expr, 'toarray'):
                cell_expr = cell_expr.toarray()
            cell_baseline = np.mean(cell_expr > 0, axis=0)
            baseline_rates[cell_type] = dict(zip(gene_names[:len(cell_baseline)], cell_baseline))
    
    # Add baseline information to each row
    baseline_response_rates = []
    baseline_binomial_ps = []
    baseline_significants = []
    
    for idx, row in results_clean.iterrows():
        baseline_rate = baseline_rates.get(row['center_cell_type'], {}).get(row['ligand'], 0.0)
        baseline_response_rates.append(baseline_rate)
        
        # Binomial test against baseline
        if (row['high_stimulation_environment'] > 0) and (baseline_rate > 0) and (baseline_rate < 1):
            try:
                p_value = stats.binom_test(
                    int(row['interaction_positive']),
                    int(row['high_stimulation_environment']),
                    baseline_rate,
                    alternative='two-sided'
                )
                baseline_binomial_ps.append(p_value)
                baseline_significants.append(p_value < 0.05)
            except:
                baseline_binomial_ps.append(np.nan)
                baseline_significants.append(False)
        else:
            baseline_binomial_ps.append(np.nan)
            baseline_significants.append(False)
    
    # Create new DataFrame
    baseline_df = pd.DataFrame({
        'baseline_response_rate': baseline_response_rates,
        'baseline_binomial_p': baseline_binomial_ps,
        'baseline_significant': baseline_significants
    })
    
    return baseline_df

def create_cumulative_bargraph_data(neighborhood_data, cumulative_expr, center_responses, gene_names):
    """
    Create data equivalent to bargraph_df
    """
    
    bargraph_data = {
        'center_cell_type': [row['center_cell_type'] for _, row in neighborhood_data.iterrows()],
        'microenvironment_cluster': [row['microenvironment_cluster'] for _, row in neighborhood_data.iterrows()]
    }
    
    # Product of cumulative expression and center cell response for each ligand
    for i, gene in enumerate(gene_names):
        bargraph_data[f'cumulative_{gene}'] = cumulative_expr[:, i]
        bargraph_data[f'response_{gene}'] = center_responses[:, i]
        bargraph_data[f'interaction_{gene}'] = cumulative_expr[:, i] * center_responses[:, i]
    
    bargraph_df = pd.DataFrame(bargraph_data)
    
    return bargraph_df

def print_cumulative_analysis_summary(results_df, top_percentile_threshold):
    """
    Display summary of analysis results
    """
    
    total_combinations = len(results_df)
    significant_interactions = len(results_df[results_df['is_significant'] == True])
    
    # Check if baseline_significant exists
    if 'baseline_significant' in results_df.columns:
        baseline_significant = len(results_df[results_df['baseline_significant'] == True])
    else:
        baseline_significant = 0
    
    print(f"\n=== Cumulative Ligand Stimulation Analysis Summary ===")
    print(f"High stimulation threshold: Top {top_percentile_threshold}%")
    print(f"Total cell type-ligand combinations: {total_combinations}")
    print(f"Significant stimulation enhancements: {significant_interactions}")
    print(f"Baseline-significant interactions: {baseline_significant}")
    
    if significant_interactions > 0:
        # Reset index to avoid duplicates
        results_clean = results_df.reset_index(drop=True)
        significant_results = results_clean[results_clean['is_significant'] == True]
        
        if len(significant_results) > 0:
            top_enhancements = significant_results.nlargest(10, 'stimulation_enhancement')
            
            print(f"\nTop 10 Stimulation Enhancements:")
            for _, row in top_enhancements.iterrows():
                print(f"   {row['center_cell_type']} + {row['ligand']}: "
                      f"High stimulation response {row['response_rate_with_high_stimulation']:.3f}, "
                      f"Low stimulation response {row['response_rate_with_low_stimulation']:.3f}, "
                      f"Enhancement: +{row['stimulation_enhancement']:.3f} "
                      f"(p={row['fisher_p_value']:.2e})")

# Microenvironment cluster-specific analysis
def analyze_microenvironment_cluster_effects(coexp_cc_df, top_n_clusters=10):
    """
    Analysis of effects by microenvironment cluster
    """
    
    print(f"\n=== Microenvironment Cluster Analysis ===")
    
    # Clean up DataFrame
    df_clean = coexp_cc_df.reset_index(drop=True)
    
    # Check if microenvironment_cluster column exists
    if 'microenvironment_cluster' not in df_clean.columns:
        print("Warning: microenvironment_cluster column not found in results")
        print(f"Available columns: {list(df_clean.columns)}")
        return pd.DataFrame()
    
    # Check cluster values
    cluster_values = df_clean['microenvironment_cluster'].value_counts()
    print(f"Found {len(cluster_values)} unique microenvironment clusters:")
    print(cluster_values.head())
    
    # Statistics by cluster
    if len(cluster_values) > 0:
        cluster_stats = df_clean.groupby('microenvironment_cluster').agg({
            'is_significant': 'sum',
            'stimulation_enhancement': ['mean', 'max'],
            'center_cell_type': 'count'
        }).round(3)
        
        cluster_stats.columns = ['significant_interactions', 'mean_enhancement', 'max_enhancement', 'total_combinations']
        cluster_stats = cluster_stats.sort_values('significant_interactions', ascending=False)
        
        print(f"\nTop {top_n_clusters} clusters by significant interactions:")
        print(cluster_stats.head(top_n_clusters))
        
        return cluster_stats
    else:
        print("No cluster data found for analysis")
        return pd.DataFrame()

# Example usage
def run_cumulative_analysis_with_clusters(edge_df, center_adata, exp_data, expr_up_by_ligands, 
                                          sp_adata, top_percentile_threshold=1.0, up_rate=1.25):
    """
    Run cumulative stimulation analysis and microenvironment cluster analysis
    """
    
    # Main analysis (also get detailed cluster results)
    coexp_cc_df, bargraph_df, detailed_cluster_results = calculate_cumulative_ligand_coexpression_analysis(
        edge_df, center_adata, exp_data, expr_up_by_ligands, sp_adata,
        neighbor_cell_numbers=19, top_percentile_threshold=top_percentile_threshold,
        role="receiver", up_rate=up_rate
    )
    
    # Microenvironment cluster analysis (using detailed results)
    cluster_analysis = analyze_microenvironment_cluster_effects_detailed(detailed_cluster_results)
    
    return {
        'interaction_results': coexp_cc_df,          # cell type x ligand
        'detailed_cluster_results': detailed_cluster_results,  # cell type x cluster x ligand
        'bargraph_data': bargraph_df,
        'cluster_analysis': cluster_analysis
    }

def analyze_microenvironment_cluster_effects_detailed(detailed_cluster_results, top_n_clusters=10):
    """
    Microenvironment analysis using detailed cluster results
    """
    
    print(f"\n=== Microenvironment Cluster Analysis (Detailed) ===")
    
    if detailed_cluster_results is None or len(detailed_cluster_results) == 0:
        print("No detailed cluster results available")
        return pd.DataFrame()
    
    # Clean up DataFrame
    df_clean = detailed_cluster_results.reset_index(drop=True)
    
    print(f"Detailed results shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")
    
    if 'microenvironment_cluster' in df_clean.columns:
        # Check cluster values
        cluster_values = df_clean['microenvironment_cluster'].value_counts()
        print(f"Found {len(cluster_values)} unique microenvironment clusters in detailed results:")
        
        # Statistics by cluster
        cluster_stats = df_clean.groupby('microenvironment_cluster').agg({
            'is_significant': 'sum',
            'stimulation_enhancement': ['mean', 'max'],
            'center_cell_type': 'count'
        }).round(3)
        
        cluster_stats.columns = ['significant_interactions', 'mean_enhancement', 'max_enhancement', 'total_combinations']
        cluster_stats = cluster_stats.sort_values('significant_interactions', ascending=False)
        
        print(f"\nTop {top_n_clusters} clusters by significant interactions:")
        print(cluster_stats.head(top_n_clusters))
        
        # Also display top interactions for each cluster
        print(f"\nTop interactions per cluster:")
        for cluster in cluster_stats.head(5).index:  # Top 5 clusters
            cluster_data = df_clean[df_clean['microenvironment_cluster'] == cluster]
            if len(cluster_data) > 0:
                significant_data = cluster_data[cluster_data['is_significant'] == True]
                if len(significant_data) > 0:
                    top_in_cluster = significant_data.nlargest(3, 'stimulation_enhancement')
                    print(f"   Cluster {cluster}:")
                    for _, row in top_in_cluster.iterrows():
                        print(f"     {row['center_cell_type']} + {row['ligand']}: +{row['stimulation_enhancement']:.3f}")
        
        return cluster_stats
    else:
        print("microenvironment_cluster column not found in detailed results")
        return pd.DataFrame()
