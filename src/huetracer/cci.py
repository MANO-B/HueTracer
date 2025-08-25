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

# def add_zscore_layers(sp_adata, top_fraction=0.01):
#     min_cells_per_type=5
#     if sparse.issparse(sp_adata.X):
#         X_dense = sp_adata.X.toarray()
#     else:
#         X_dense = sp_adata.X.copy()
    
#     sp_adata.layers["zscore_by_celltype"] = np.zeros_like(X_dense)
#     sp_adata.layers["zscore_all_celltype"] = np.zeros_like(X_dense)
    
#     print("Calculating pooled within-celltype standard deviations...")
    
#     # Initialize for pooled std calculation
#     pooled_std = np.zeros(X_dense.shape[1])
    
#     for gene_idx in range(X_dense.shape[1]):
#         gene_expr_all = X_dense[:, gene_idx]
        
#         # Collect within-celltype variations for this gene
#         sum_squared_deviations = 0
#         total_df = 0  # degrees of freedom
        
#         for ct in sp_adata.obs["celltype"].unique():
#             idx = sp_adata.obs["celltype"] == ct
#             gene_expr_ct = gene_expr_all[idx]
            
#             # Use only non-zero values within this cell type
#             nonzero_expr = gene_expr_ct[gene_expr_ct != 0]
            
#             if len(nonzero_expr) > 1:  # Need at least 2 values for std
#                 mean_ct = np.mean(nonzero_expr)
#                 # Sum of squared deviations from cell-type mean
#                 sum_squared_deviations += np.sum((nonzero_expr - mean_ct) ** 2)
#                 total_df += len(nonzero_expr) - 1  # degrees of freedom
        
#         # Pooled standard deviation
#         if total_df > 0:
#             pooled_std[gene_idx] = np.sqrt(sum_squared_deviations / total_df)
#         else:
#             pooled_std[gene_idx] = 1  # Default for genes with insufficient data
    
#     # Prevent division by very small values
#     pooled_std[pooled_std < 1e-6] = 1e-6
    
#     print(f"Pooled within-celltype std range: {pooled_std.min():.4f} - {pooled_std.max():.4f}")
    
#     # Calculate z-scores using pooled within-celltype std
#     for ct in sp_adata.obs["celltype"].unique():
#         idx = sp_adata.obs["celltype"] == ct
#         X_sub = X_dense[idx]
#         n_cells = np.sum(idx)
        
#         if n_cells >= min_cells_per_type:
#             # Calculate mean within cell type
#             mean = X_sub.mean(axis=0)
            
#             # Z-score using pooled within-celltype std
#             z = (X_sub - mean) / pooled_std
            
#             sp_adata.layers["zscore_by_celltype"][idx] = make_positive_values(z)
#             print(f"  Processed {ct}: {n_cells} cells")
#         else:
#             print(f"  Skipped {ct}: only {n_cells} cells (< {min_cells_per_type})")
    
#     # High expression genes
#     z_all = X_dense
#     zscore_all = make_positive_values(z_all)
#     zscore_all = make_top_values(zscore_all, axis=0, top_fraction=top_fraction)
#     sp_adata.layers["zscore_all_celltype"] = zscore_all
    
#     print("Advanced within-celltype variation normalization completed!")
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
    高速化された改良共発現解析
    
    最適化ポイント:
    1. ベクトル化された計算
    2. 事前計算とキャッシュ
    3. メモリ効率的な処理
    4. 不要な統計計算の削減
    """
    
    print("Fast enhanced co-expression calculation...")
    
    # データ準備（最適化）
    center_adata.X = exp_data
    gene_names = center_adata.var_names.tolist()
    n_genes = len(gene_names)
    
    # role='receiver'の場合の送信・受信関係
    if role == "receiver":
        actual_sender = edge_df.cell2.values  # numpy配列に変換
        actual_receiver = edge_df.cell1.values
        sender_type_col = 'cell2_type'
        receiver_type_col = 'cell1_type'
    else:
        actual_sender = edge_df.cell1.values
        actual_receiver = edge_df.cell2.values
        sender_type_col = 'cell1_type'
        receiver_type_col = 'cell2_type'
    
    # インデックスマッピングの事前計算
    cell_to_idx = {cell: idx for idx, cell in enumerate(center_adata.obs_names)}
    sender_indices = np.array([cell_to_idx[cell] for cell in actual_sender])
    receiver_indices = np.array([cell_to_idx[cell] for cell in actual_receiver])
    
    # 発現データの取得（ベクトル化）
    sender_expr = exp_data[sender_indices]  # (n_edges, n_genes)
    receiver_expr = expr_up_by_ligands[receiver_indices]  # (n_edges, n_genes)
    
    if hasattr(sender_expr, 'toarray'):
        sender_expr = sender_expr.toarray()
    if hasattr(receiver_expr, 'toarray'):
        receiver_expr = receiver_expr.toarray()
    
    # 共発現計算（ベクトル化）
    coexp_matrix = sender_expr * receiver_expr
    
    # 細胞タイプのエンコーディング（高速化）
    sender_types = edge_df[sender_type_col].values
    receiver_types = edge_df[receiver_type_col].values
    
    unique_sender_types = np.unique(sender_types)
    unique_receiver_types = np.unique(receiver_types)
    
    sender_type_to_idx = {t: i for i, t in enumerate(unique_sender_types)}
    receiver_type_to_idx = {t: i for i, t in enumerate(unique_receiver_types)}
    
    sender_type_encoded = np.array([sender_type_to_idx[t] for t in sender_types])
    receiver_type_encoded = np.array([receiver_type_to_idx[t] for t in receiver_types])
    
    # ベースライン計算（事前計算・キャッシュ）
    print("Calculating baselines...")
    baseline_rates = fast_calculate_baseline_rates(sp_adata, expr_up_by_ligands, gene_names)
    
    # 大規模な分割表計算（ベクトル化）
    print("Computing contingency tables...")
    results_data = fast_compute_all_contingency_tables(
        sender_expr, receiver_expr, sender_type_encoded, receiver_type_encoded,
        unique_sender_types, unique_receiver_types, gene_names, baseline_rates
    )
    
    # 既存フォーマットでの結果整理
    print("Formatting results...")
    coexp_cc_df = format_results_to_existing_format(
        results_data, unique_sender_types, unique_receiver_types, gene_names, 
        sender_type_col, receiver_type_col, role, up_rate
    )
    
    # bargraph_df作成（既存フォーマット）
    bargraph_data = {
        receiver_type_col: receiver_types,
        sender_type_col: sender_types
    }
    
    for i, gene in enumerate(gene_names):
        bargraph_data[gene] = coexp_matrix[:, i]
    
    bargraph_df = pd.DataFrame(bargraph_data, index=edge_df.index)
    
    # 結果サマリー
    n_significant = np.sum(coexp_cc_df['is_significant'])
    n_enhanced_significant = np.sum(coexp_cc_df.get('enhanced_significant', False))
    
    print(f"Completed: {len(coexp_cc_df)} interactions")
    print(f"Traditional: {n_significant} significant")
    print(f"Enhanced: {n_enhanced_significant} significant with baseline consideration")
    
    return coexp_cc_df, bargraph_df

def fast_calculate_baseline_rates(sp_adata, expr_up_by_ligands, gene_names):
    """
    ベースライン反応率の高速計算
    """
    baseline_rates = {}
    
    # 細胞タイプごとの処理（ベクトル化）
    cell_types = sp_adata.obs['celltype'].unique()
    celltype_values = sp_adata.obs['celltype'].values
    
    for cell_type in cell_types:
        cell_mask = celltype_values == cell_type
        
        if not np.any(cell_mask):
            continue
        
        # その細胞タイプの発現データ（ベクトル化）
        cell_expr = expr_up_by_ligands[cell_mask]
        
        # 全リガンドの反応率を一括計算
        response_rates = np.mean(cell_expr > 0, axis=0)
        
        # 辞書に格納（必要な分のみ）
        baseline_rates[cell_type] = dict(zip(gene_names[:len(response_rates)], response_rates))
    
    return baseline_rates

def fast_compute_all_contingency_tables(sender_expr, receiver_expr, sender_type_encoded, 
                                      receiver_type_encoded, unique_sender_types, 
                                      unique_receiver_types, gene_names, baseline_rates):
    """
    全ての分割表を高速計算
    """
    n_sender_types = len(unique_sender_types)
    n_receiver_types = len(unique_receiver_types)
    n_genes = len(gene_names)
    
    # 結果格納用
    results_data = []
    
    # 送信細胞・受信細胞の組み合わせごとに処理
    for s_idx, sender_type in enumerate(unique_sender_types):
        for r_idx, receiver_type in enumerate(unique_receiver_types):
            
            # この組み合わせのエッジを抽出
            mask = (sender_type_encoded == s_idx) & (receiver_type_encoded == r_idx)
            
            if not np.any(mask):
                continue
            
            # この組み合わせの発現データ
            s_expr_subset = sender_expr[mask]  # (n_edges_subset, n_genes)
            r_expr_subset = receiver_expr[mask]
            
            # 全遺伝子の分割表を一括計算（ベクトル化）
            contingency_stats = compute_vectorized_contingency_stats(
                s_expr_subset, r_expr_subset, gene_names, baseline_rates.get(receiver_type, {})
            )
            
            # 結果に追加
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
    ベクトル化された分割表統計の計算
    """
    n_edges, n_genes = sender_expr.shape
    
    # 二値化（ベクトル化）
    sender_binary = sender_expr > 0  # (n_edges, n_genes)
    receiver_binary = receiver_expr > 0
    
    # 4つの状況を一括計算
    sender_pos_receiver_pos = np.sum(sender_binary & receiver_binary, axis=0)  # (n_genes,)
    sender_pos_receiver_neg = np.sum(sender_binary & ~receiver_binary, axis=0)
    sender_neg_receiver_pos = np.sum(~sender_binary & receiver_binary, axis=0)
    sender_neg_receiver_neg = np.sum(~sender_binary & ~receiver_binary, axis=0)
    
    # 基本統計
    sender_positive_count = sender_pos_receiver_pos + sender_pos_receiver_neg
    sender_negative_count = sender_neg_receiver_pos + sender_neg_receiver_neg
    
    # 条件付き確率（ゼロ除算対策）
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_prob_r_given_s = np.divide(sender_pos_receiver_pos, sender_positive_count, 
                                       out=np.zeros_like(sender_pos_receiver_pos, dtype=float),
                                       where=sender_positive_count>0)
        
        cond_prob_r_given_not_s = np.divide(sender_neg_receiver_pos, sender_negative_count,
                                           out=np.zeros_like(sender_neg_receiver_pos, dtype=float),
                                           where=sender_negative_count>0)
    
    interaction_enhancement = cond_prob_r_given_s - cond_prob_r_given_not_s
    
    # ベースライン情報
    baseline_rates = np.array([baseline_dict.get(gene, 0.0) for gene in gene_names])
    
    # 高速統計検定（Fisher正確検定の近似）
    fisher_p_values, odds_ratios = fast_vectorized_fisher_test(
        sender_pos_receiver_pos, sender_pos_receiver_neg,
        sender_neg_receiver_pos, sender_neg_receiver_neg
    )
    
    # Binomial test against baseline（ベクトル化）
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
    ベクトル化されたFisher正確検定の近似
    """
    n_genes = len(a)
    p_values = np.full(n_genes, np.nan)
    odds_ratios = np.full(n_genes, np.nan)
    
    # 有効なケースのマスク
    valid_mask = (a + b + c + d) > 0
    
    if np.any(valid_mask):
        a_valid = a[valid_mask]
        b_valid = b[valid_mask]
        c_valid = c[valid_mask]
        d_valid = d[valid_mask]
        
        # Odds ratio計算
        with np.errstate(divide='ignore', invalid='ignore'):
            or_values = np.divide(a_valid * d_valid, b_valid * c_valid,
                                 out=np.full_like(a_valid, np.inf, dtype=float),
                                 where=(b_valid * c_valid) > 0)
        
        # Chi-square近似（大標本）
        n_total = a_valid + b_valid + c_valid + d_valid
        expected_a = (a_valid + b_valid) * (a_valid + c_valid) / n_total
        
        # Chi-square統計量
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2_stats = np.divide((a_valid - expected_a) ** 2, expected_a,
                                  out=np.zeros_like(expected_a),
                                  where=expected_a > 0)
            chi2_stats += np.divide((b_valid - (a_valid + b_valid - expected_a)) ** 2, 
                                   a_valid + b_valid - expected_a,
                                   out=np.zeros_like(expected_a),
                                   where=(a_valid + b_valid - expected_a) > 0)
        
        # p値近似（大標本の場合）
        p_approx = 1 - stats.chi2.cdf(chi2_stats, df=1)
        
        # 結果を元の配列に格納
        p_values[valid_mask] = p_approx
        odds_ratios[valid_mask] = or_values
    
    return p_values, odds_ratios

def fast_vectorized_binomial_test(successes, trials, baseline_rates, alpha=0.05):
    """
    ベクトル化されたBinomial検定
    """
    n_genes = len(successes)
    p_values = np.full(n_genes, np.nan)
    
    # 有効なケースのマスク
    valid_mask = (trials > 0) & (baseline_rates > 0) & (baseline_rates < 1)
    
    if np.any(valid_mask):
        # 正規近似を使用（大標本）
        s_valid = successes[valid_mask]
        t_valid = trials[valid_mask] 
        r_valid = baseline_rates[valid_mask]
        
        # 期待値と標準偏差
        expected = t_valid * r_valid
        std_dev = np.sqrt(t_valid * r_valid * (1 - r_valid))
        
        # Z統計量
        with np.errstate(divide='ignore', invalid='ignore'):
            z_stats = np.divide(s_valid - expected, std_dev,
                               out=np.zeros_like(s_valid, dtype=float),
                               where=std_dev > 0)
        
        # 両側検定のp値
        p_approx = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
        
        p_values[valid_mask] = p_approx
    
    return p_values

def format_results_to_existing_format(results_data, unique_sender_types, unique_receiver_types, 
                                    gene_names, sender_type_col, receiver_type_col, role, up_rate):
    """
    既存フォーマットへの結果整理
    """
    
    # データフレーム作成
    df = pd.DataFrame(results_data)
    
    # 既存フォーマット用の列名調整
    if role == "receiver":
        df['cell1_type'] = df['receiver_type'] 
        df['cell2_type'] = df['sender_type']
    else:
        df['cell1_type'] = df['sender_type']
        df['cell2_type'] = df['receiver_type']
    
    # 基本統計の追加
    df['coactivity_per_sender_cell_expr_ligand'] = np.divide(
        df['interaction_positive'], df['sender_positive'],
        out=np.zeros_like(df['interaction_positive'], dtype=float),
        where=df['sender_positive'] > 0
    )
    
    # 従来の統計検定（簡易版）
    print("Computing traditional statistics...")
    df = add_traditional_statistics(df, up_rate)
    
    # 強化された統計の有意性判定
    df['enhanced_significant'] = (df['enhanced_fisher_p'] < 0.05) & (df['enhanced_fisher_p'].notna())
    df['baseline_significant'] = (df['baseline_binomial_p'] < 0.05) & (df['baseline_binomial_p'].notna())
    
    # Multiple testing correction（高速版）
    print("Applying multiple testing correction...")
    df = add_fast_multiple_testing_correction(df)
    
    return df

def add_traditional_statistics(df, up_rate):
    """
    従来統計の高速追加
    """
    
    # 母集団レート計算（リガンド別）
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
    
    # 各行にマップ
    df['population_mean_rate'] = df['ligand'].map(population_rates)
    expected_rates = up_rate * df['population_mean_rate']
    
    # Binomial test（ベクトル化）
    valid_mask = (df['sender_positive'] > 0) & (expected_rates <= 1.0) & (expected_rates > 0)
    
    p_values = np.full(len(df), np.nan)
    
    if np.any(valid_mask):
        # 正規近似を使用
        successes = df.loc[valid_mask, 'interaction_positive'].values
        trials = df.loc[valid_mask, 'sender_positive'].values  
        rates = expected_rates.loc[valid_mask].values
        
        expected = trials * rates
        std_dev = np.sqrt(trials * rates * (1 - rates))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            z_stats = np.divide(successes - expected, std_dev,
                               out=np.zeros_like(successes, dtype=float),
                               where=std_dev > 0)
        
        p_approx = 1 - stats.norm.cdf(z_stats)  # 右側検定
        p_values[valid_mask] = p_approx
    
    df['p_value'] = p_values
    df['is_significant'] = (p_values < 0.05) & ~np.isnan(p_values)
    
    # Beta信頼区間（ベクトル化）
    alpha_post = df['interaction_positive'] + 0.5
    beta_post = df['sender_positive'] - df['interaction_positive'] + 0.5
    
    alpha_post = np.maximum(alpha_post, 0.5)
    beta_post = np.maximum(beta_post, 0.5)
    
    df['ci_lower_beta'] = beta.ppf(0.025, alpha_post, beta_post)
    df['ci_upper_beta'] = beta.ppf(0.975, alpha_post, beta_post)
    
    return df

def add_fast_multiple_testing_correction(df):
    """
    高速多重検定補正
    """
    
    # 従来のp値
    valid_p = df['p_value'].dropna()
    if len(valid_p) > 0:
        corrected = multipletests(valid_p, method='bonferroni')
        df.loc[df['p_value'].notna(), 'p_value_bonferroni'] = corrected[1]
        df.loc[df['p_value'].notna(), 'is_significant_bonferroni'] = corrected[0]
    else:
        df['p_value_bonferroni'] = np.nan
        df['is_significant_bonferroni'] = False
    
    # 強化されたp値
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
    細胞種ごとの上位相互作用を表示
    
    Parameters:
    -----------
    coexp_cc_df : DataFrame
        解析結果のデータフレーム
    enhancement_threshold : float
        相互作用強化効果の閾値（デフォルト: 1.25）
    top_n : int
        各細胞種で表示する上位の数（デフォルト: 10）
    min_responses_with_sender : int
        送信細胞ありでの最低反応回数（デフォルト: 5）
    min_responses_without_sender : int
        送信細胞なしでの最低反応回数（デフォルト: 5）
    """
    
    # 有意な相互作用強化を抽出（最低反応回数の条件を追加）
    significant_interactions = coexp_cc_df[
        (coexp_cc_df['enhanced_significant'] == True) &
        (coexp_cc_df['enhanced_odds_ratio'] >= enhancement_threshold) &  # オッズ比1．25以上の強化効果
        (coexp_cc_df.get('sender_pos_receiver_pos', 0) >= min_responses_with_sender) &  # 送信ありでの最低反応数
        (coexp_cc_df.get('sender_neg_receiver_pos', 0) >= min_responses_without_sender)  # 送信なしでの最低反応数
    ].copy()
    
    if len(significant_interactions) == 0:
        print(f"No significant interactions found with the specified criteria:")
        print(f"  - Enhancement threshold: >{enhancement_threshold}")
        print(f"  - Min responses with sender: ≥{min_responses_with_sender}")
        print(f"  - Min responses without sender: ≥{min_responses_without_sender}")
        return
    
    print(f"=== 真の相互作用強化（強化効果 ≥ {enhancement_threshold}）===")
    print(f"フィルター条件:")
    print(f"  - 送信細胞ありでの最低反応数: ≥{min_responses_with_sender}")
    print(f"  - 送信細胞なしでの最低反応数: ≥{min_responses_without_sender}")
    print(f"Total significant interactions: {len(significant_interactions)}")
    print()
    
    # 受信細胞タイプ別にグループ化して処理
    receiver_groups = significant_interactions.groupby('cell2_type' if 'cell2_type' in significant_interactions.columns else 'receiver_cell_type')
    
    for receiver_type, group in receiver_groups:
        print(f"📱 受信細胞タイプ: {receiver_type}")
        print("-" * 60)
        
        # 相互作用強化効果でソートして上位を取得
        top_interactions = group.nlargest(top_n, 'enhanced_odds_ratio')
        
        for i, (_, row) in enumerate(top_interactions.iterrows(), 1):
            sender_type = row['cell1_type'] if 'cell1_type' in row else row['sender_cell_type']
            receiver_type_display = row['cell2_type'] if 'cell2_type' in row else row['receiver_cell_type']
            ligand = row['ligand']
            
            print(f"  {i:2d}. {sender_type} → {receiver_type_display} ({ligand})")
            print(f"      送信細胞ありの反応率: {row['cond_prob_receiver_given_sender']:.3f}")
            print(f"      送信細胞なしの反応率: {row['cond_prob_receiver_given_not_sender']:.3f}")
            print(f"      強化効果: +{row['interaction_enhancement']:.3f} (Odds ratio: {row['enhanced_odds_ratio']:.2f})")
            
            # 追加統計情報
            if 'enhanced_fisher_p' in row and not pd.isna(row['enhanced_fisher_p']):
                print(f"      p値: {row['enhanced_fisher_p']:.2e}")
            
            # 実際の観測数も表示
            if 'sender_pos_receiver_pos' in row:
                total_with_sender = row.get('sender_positive', 'N/A')
                responded_with_sender = row.get('sender_pos_receiver_pos', 'N/A')
                responded_without_sender = row.get('sender_neg_receiver_pos', 'N/A')
                total_without_sender = row.get('total_interactions', 0) - row.get('sender_positive', 0) if 'total_interactions' in row else 'N/A'
                
                print(f"      観測数: 送信あり({responded_with_sender}/{total_with_sender}), 送信なし({responded_without_sender}/{total_without_sender})")
            
            print()
        
        print()

def display_summary_statistics(coexp_cc_df, enhancement_threshold=1.25, 
                             min_responses_with_sender=5, min_responses_without_sender=5):
    """
    相互作用の要約統計を表示
    """
    
    significant_interactions = coexp_cc_df[
        (coexp_cc_df['enhanced_significant'] == True) &
        (coexp_cc_df['enhanced_odds_ratio'] >= enhancement_threshold) &
        (coexp_cc_df.get('sender_pos_receiver_pos', 0) >= min_responses_with_sender) &
        (coexp_cc_df.get('sender_neg_receiver_pos', 0) >= min_responses_without_sender)
    ]
    
    print("=== 要約統計 ===")
    print(f"全相互作用数: {len(coexp_cc_df)}")
    print(f"有意な相互作用強化: {len(significant_interactions)} ({len(significant_interactions)/len(coexp_cc_df):.1%})")
    print()
    
    if len(significant_interactions) > 0:
        # 受信細胞タイプ別の統計
        receiver_stats = significant_interactions.groupby('cell2_type' if 'cell2_type' in significant_interactions.columns else 'receiver_cell_type').agg({
            'enhanced_odds_ratio': ['count', 'mean', 'max'],
            'cond_prob_receiver_given_sender': 'mean',
            'cond_prob_receiver_given_not_sender': 'mean'
        }).round(3)
        
        print("受信細胞タイプ別統計:")
        print(receiver_stats)
        print()
        
        # 送信細胞タイプ別の統計
        sender_stats = significant_interactions.groupby('cell1_type' if 'cell1_type' in significant_interactions.columns else 'sender_cell_type').agg({
            'enhanced_odds_ratio': ['count', 'mean', 'max'],
            'cond_prob_receiver_given_sender': 'mean'
        }).round(3)
        
        print("送信細胞タイプ別統計:")
        print(sender_stats)
        print()
        
        # リガンド別の統計
        ligand_stats = significant_interactions.groupby('ligand').agg({
            'enhanced_odds_ratio': ['count', 'mean', 'max']
        }).round(3)
        
        print("上位リガンド（相互作用数順）:")
        print(ligand_stats.sort_values(('enhanced_odds_ratio', 'count'), ascending=False).head(10))

def display_high_spontaneous_responses(coexp_cc_df, spontaneous_threshold=0.1, 
                                     min_responses_without_sender=5):
    """
    高い自発的反応を示す組み合わせを表示
    """
    
    high_spontaneous = coexp_cc_df[
        (coexp_cc_df['cond_prob_receiver_given_not_sender'] > spontaneous_threshold) &
        (coexp_cc_df.get('sender_neg_receiver_pos', 0) >= min_responses_without_sender)  # 最低反応回数
    ].copy()
    
    if len(high_spontaneous) > 0:
        high_spontaneous = high_spontaneous.sort_values('cond_prob_receiver_given_not_sender', ascending=False)
        
        print(f"=== 高い自発的反応（送信細胞なしでの反応率 > {spontaneous_threshold:.1%}）===")
        
        for _, row in high_spontaneous.head(20).iterrows():
            receiver_type = row['cell2_type'] if 'cell2_type' in row else row['receiver_cell_type']
            ligand = row['ligand']
            spontaneous_rate = row['cond_prob_receiver_given_not_sender']
            
            print(f"{receiver_type} responds to {ligand}: {spontaneous_rate:.3f} ({spontaneous_rate:.1%}) without sender")
            
            if 'sender_neg_receiver_pos' in row:
                responded = row['sender_neg_receiver_pos']
                total_without = row.get('total_interactions', 0) - row.get('sender_positive', 0)
                print(f"    観測数: {responded}/{total_without}")
        
        print()

def display_inhibitory_effects(coexp_cc_df, inhibition_threshold=-0.05,
                             min_responses_with_sender=5, min_responses_without_sender=5):
    """
    阻害効果を示す相互作用を表示
    """
    
    inhibitory_effects = coexp_cc_df[
        (coexp_cc_df['interaction_enhancement'] < inhibition_threshold) &
        (coexp_cc_df.get('enhanced_significant', False) == True) &
        (coexp_cc_df.get('sender_pos_receiver_pos', 0) >= min_responses_with_sender) &
        (coexp_cc_df.get('sender_neg_receiver_pos', 0) >= min_responses_without_sender)
    ].copy()
    
    if len(inhibitory_effects) > 0:
        inhibitory_effects = inhibitory_effects.sort_values('interaction_enhancement', ascending=True)
        
        print(f"=== 阻害効果（強化効果 < {inhibition_threshold:.1%}）===")
        
        for _, row in inhibitory_effects.head(10).iterrows():
            sender_type = row['cell1_type'] if 'cell1_type' in row else row['sender_cell_type']
            receiver_type = row['cell2_type'] if 'cell2_type' in row else row['receiver_cell_type']
            ligand = row['ligand']
            inhibition = row['interaction_enhancement']
            
            print(f"{sender_type} inhibits {receiver_type} response to {ligand}: {inhibition:.3f} ({inhibition:.1%})")
            print(f"    送信ありの反応率: {row['cond_prob_receiver_given_sender']:.3f}")
            print(f"    送信なしの反応率: {row['cond_prob_receiver_given_not_sender']:.3f}")
        
        print()

# 使用例
def comprehensive_interaction_analysis(coexp_cc_df, enhancement_threshold=0.02, spontaneous_threshold=0.1,
                                       inhibition_threshold=-0.05, min_responses_with_sender=5, min_responses_without_sender=5):
    """
    包括的な相互作用解析の実行
    """
    
    print("🔬 包括的細胞間相互作用解析")
    print("=" * 80)
    
    # 1. 要約統計
    display_summary_statistics(coexp_cc_df, enhancement_threshold=1.25,
                             min_responses_with_sender=min_responses_with_sender,
                             min_responses_without_sender=min_responses_without_sender)
    
    # 2. 細胞種別の上位相互作用（Odds比>=enhancement_threshold、各5個）
    display_top_interactions_by_cell_type(coexp_cc_df, enhancement_threshold=1.25, top_n=5,
                                        min_responses_with_sender=min_responses_with_sender,
                                        min_responses_without_sender=min_responses_without_sender)
    
    # 3. 高い自発的反応
    display_high_spontaneous_responses(coexp_cc_df, spontaneous_threshold=0.1,
                                     min_responses_without_sender=min_responses_without_sender)
    
    # 4. 阻害効果
    display_inhibitory_effects(coexp_cc_df, inhibition_threshold=-0.05,
                             min_responses_with_sender=min_responses_with_sender,
                             min_responses_without_sender=min_responses_without_sender)


def calculate_enhanced_coexpression_coactivity_cluster(edge_df, center_adata, exp_data, expr_up_by_ligands, 
                                                          sp_adata, cluster_label, role="receiver", up_rate=1.25):
    """
    高速化されたクラスタ特異的改良共発現解析
    
    Parameters:
    -----------
    edge_df : DataFrame
        エッジ情報
    center_adata : AnnData
        中心細胞データ
    exp_data : ndarray
        発現データ
    expr_up_by_ligands : ndarray
        リガンド反応データ
    sp_adata : AnnData
        空間転写データ（ベースライン計算用）
    cluster_label : str or list
        対象クラスタ
    role : str
        "receiver" または "sender"
    up_rate : float
        期待値の倍率
    """
    
    print(f"Fast enhanced cluster-specific analysis for cluster: {cluster_label}")
    
    # データ準備（最適化）
    center_adata.X = exp_data
    gene_names = center_adata.var_names.tolist()
    n_genes = len(gene_names)
    
    # クラスタフィルタリング（高速化）
    if isinstance(cluster_label, (list, tuple)):
        cluster_set = set(str(c) for c in cluster_label)
    else:
        cluster_set = {str(cluster_label)}
    
    # 有効細胞の特定（ベクトル化）
    cell_clusters = center_adata.obs['cluster'].astype(str)
    cluster_mask = cell_clusters.isin(cluster_set)
    valid_cell1_indices = set(center_adata.obs_names[cluster_mask])
    
    # エッジフィルタリング
    edge_mask = edge_df['cell1'].isin(valid_cell1_indices)
    filtered_edge_df = edge_df[edge_mask].copy()
    
    if len(filtered_edge_df) == 0:
        print(f"Warning: No interactions found for cluster {cluster_label}")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Processing {len(filtered_edge_df)} edges for cluster analysis...")
    
    # role設定
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
    
    # インデックスマッピングの事前計算
    cell_to_idx = {cell: idx for idx, cell in enumerate(center_adata.obs_names)}
    sender_indices = np.array([cell_to_idx[cell] for cell in actual_sender])
    receiver_indices = np.array([cell_to_idx[cell] for cell in actual_receiver])
    
    # 発現データの取得（ベクトル化）
    sender_expr = exp_data[sender_indices]
    receiver_expr = expr_up_by_ligands[receiver_indices]
    
    if hasattr(sender_expr, 'toarray'):
        sender_expr = sender_expr.toarray()
    if hasattr(receiver_expr, 'toarray'):
        receiver_expr = receiver_expr.toarray()
    
    # 共発現計算（ベクトル化）
    coexp_matrix = sender_expr * receiver_expr
    
    # 細胞タイプのエンコーディング（高速化）
    sender_types = filtered_edge_df[sender_type_col].values
    receiver_types = filtered_edge_df[receiver_type_col].values
    
    unique_sender_types = np.unique(sender_types)
    unique_receiver_types = np.unique(receiver_types)
    
    sender_type_to_idx = {t: i for i, t in enumerate(unique_sender_types)}
    receiver_type_to_idx = {t: i for i, t in enumerate(unique_receiver_types)}
    
    sender_type_encoded = np.array([sender_type_to_idx[t] for t in sender_types])
    receiver_type_encoded = np.array([receiver_type_to_idx[t] for t in receiver_types])
    
    # ベースライン計算（クラスタ特異的）
    print("Calculating cluster-specific baselines...")
    baseline_rates = fast_calculate_cluster_baseline_rates(sp_adata, expr_up_by_ligands, gene_names, cluster_label)
    
    # 大規模な分割表計算（ベクトル化）
    print("Computing contingency tables...")
    results_data = fast_compute_cluster_contingency_tables(
        sender_expr, receiver_expr, sender_type_encoded, receiver_type_encoded,
        unique_sender_types, unique_receiver_types, gene_names, baseline_rates
    )
    
    # 既存フォーマットでの結果整理
    print("Formatting results...")
    coexp_cc_df = format_cluster_results_to_existing_format(
        results_data, unique_sender_types, unique_receiver_types, gene_names, 
        sender_type_col, receiver_type_col, role, up_rate
    )
    
    # bargraph_df作成（既存フォーマット）
    bargraph_data = {
        receiver_type_col: receiver_types,
        sender_type_col: sender_types
    }
    
    for i, gene in enumerate(gene_names):
        bargraph_data[gene] = coexp_matrix[:, i]
    
    bargraph_df = pd.DataFrame(bargraph_data, index=filtered_edge_df.index)
    
    # 結果サマリー
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
    クラスタ特異的ベースライン反応率の高速計算
    """
    baseline_rates = {}
    
    # 指定クラスタの細胞のみでベースラインを計算
    if isinstance(cluster_label, (list, tuple)):
        cluster_set = set(str(c) for c in cluster_label)
    else:
        cluster_set = {str(cluster_label)}
    
    # クラスタマスクの作成
    cluster_mask = sp_adata.obs['cluster'].astype(str).isin(cluster_set)
    cluster_cells = sp_adata[cluster_mask]
    
    if len(cluster_cells) == 0:
        print(f"Warning: No cells found for cluster {cluster_label}")
        return {}
    
    # クラスタ内の細胞タイプごとの処理
    cell_types = cluster_cells.obs['celltype'].unique()
    celltype_values = cluster_cells.obs['celltype'].values
    
    # クラスタ内の発現データを取得
    cluster_indices = np.where(cluster_mask)[0]
    cluster_expr_up = expr_up_by_ligands[cluster_indices]
    
    for cell_type in cell_types:
        cell_mask_in_cluster = celltype_values == cell_type
        
        if not np.any(cell_mask_in_cluster):
            continue
        
        # その細胞タイプの発現データ（クラスタ内のみ）
        cell_expr = cluster_expr_up[cell_mask_in_cluster]
        
        # 全リガンドの反応率を一括計算
        response_rates = np.mean(cell_expr > 0, axis=0)
        
        # 辞書に格納
        baseline_rates[cell_type] = dict(zip(gene_names[:len(response_rates)], response_rates))
    
    return baseline_rates

def fast_compute_cluster_contingency_tables(sender_expr, receiver_expr, sender_type_encoded, 
                                          receiver_type_encoded, unique_sender_types, 
                                          unique_receiver_types, gene_names, baseline_rates):
    """
    クラスタ特異的分割表の高速計算
    """
    n_sender_types = len(unique_sender_types)
    n_receiver_types = len(unique_receiver_types)
    n_genes = len(gene_names)
    
    # 結果格納用
    results_data = []
    
    # 送信細胞・受信細胞の組み合わせごとに処理
    for s_idx, sender_type in enumerate(unique_sender_types):
        for r_idx, receiver_type in enumerate(unique_receiver_types):
            
            # この組み合わせのエッジを抽出
            mask = (sender_type_encoded == s_idx) & (receiver_type_encoded == r_idx)
            
            if not np.any(mask):
                continue
            
            # この組み合わせの発現データ
            s_expr_subset = sender_expr[mask]
            r_expr_subset = receiver_expr[mask]
            
            # 全遺伝子の分割表を一括計算（ベクトル化）
            contingency_stats = compute_vectorized_cluster_contingency_stats(
                s_expr_subset, r_expr_subset, gene_names, baseline_rates.get(receiver_type, {})
            )
            
            # 結果に追加
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
    クラスタ用ベクトル化分割表統計
    """
    n_edges, n_genes = sender_expr.shape
    
    # 二値化（ベクトル化）
    sender_binary = sender_expr > 0
    receiver_binary = receiver_expr > 0
    
    # 4つの状況を一括計算
    sender_pos_receiver_pos = np.sum(sender_binary & receiver_binary, axis=0)
    sender_pos_receiver_neg = np.sum(sender_binary & ~receiver_binary, axis=0)
    sender_neg_receiver_pos = np.sum(~sender_binary & receiver_binary, axis=0)
    sender_neg_receiver_neg = np.sum(~sender_binary & ~receiver_binary, axis=0)
    
    # 基本統計
    sender_positive_count = sender_pos_receiver_pos + sender_pos_receiver_neg
    sender_negative_count = sender_neg_receiver_pos + sender_neg_receiver_neg
    
    # 条件付き確率（ゼロ除算対策）
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_prob_r_given_s = np.divide(sender_pos_receiver_pos, sender_positive_count, 
                                       out=np.zeros_like(sender_pos_receiver_pos, dtype=float),
                                       where=sender_positive_count>0)
        
        cond_prob_r_given_not_s = np.divide(sender_neg_receiver_pos, sender_negative_count,
                                           out=np.zeros_like(sender_neg_receiver_pos, dtype=float),
                                           where=sender_negative_count>0)
    
    interaction_enhancement = cond_prob_r_given_s - cond_prob_r_given_not_s
    
    # ベースライン情報
    baseline_rates = np.array([baseline_dict.get(gene, 0.0) for gene in gene_names])
    
    # 高速統計検定
    fisher_p_values, odds_ratios = fast_vectorized_fisher_test_cluster(
        sender_pos_receiver_pos, sender_pos_receiver_neg,
        sender_neg_receiver_pos, sender_neg_receiver_neg
    )
    
    # Binomial test against baseline（ベクトル化）
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
    クラスタ用ベクトル化Fisher検定
    """
    n_genes = len(a)
    p_values = np.full(n_genes, np.nan)
    odds_ratios = np.full(n_genes, np.nan)
    
    # 有効なケースのマスク
    valid_mask = (a + b + c + d) > 0
    
    if np.any(valid_mask):
        a_valid = a[valid_mask]
        b_valid = b[valid_mask]
        c_valid = c[valid_mask]
        d_valid = d[valid_mask]
        
        # Odds ratio計算
        with np.errstate(divide='ignore', invalid='ignore'):
            or_values = np.divide(a_valid * d_valid, b_valid * c_valid,
                                 out=np.full_like(a_valid, np.inf, dtype=float),
                                 where=(b_valid * c_valid) > 0)
        
        # Chi-square近似
        n_total = a_valid + b_valid + c_valid + d_valid
        expected_a = (a_valid + b_valid) * (a_valid + c_valid) / n_total
        
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2_stats = np.divide((a_valid - expected_a) ** 2, expected_a,
                                  out=np.zeros_like(expected_a),
                                  where=expected_a > 0)
        
        # p値近似
        p_approx = 1 - stats.chi2.cdf(chi2_stats, df=1)
        
        p_values[valid_mask] = p_approx
        odds_ratios[valid_mask] = or_values
    
    return p_values, odds_ratios

def fast_vectorized_binomial_test_cluster(successes, trials, baseline_rates, alpha=0.05):
    """
    クラスタ用ベクトル化Binomial検定
    """
    n_genes = len(successes)
    p_values = np.full(n_genes, np.nan)
    
    # 有効なケースのマスク
    valid_mask = (trials > 0) & (baseline_rates > 0) & (baseline_rates < 1)
    
    if np.any(valid_mask):
        # 正規近似を使用
        s_valid = successes[valid_mask]
        t_valid = trials[valid_mask] 
        r_valid = baseline_rates[valid_mask]
        
        # 期待値と標準偏差
        expected = t_valid * r_valid
        std_dev = np.sqrt(t_valid * r_valid * (1 - r_valid))
        
        # Z統計量
        with np.errstate(divide='ignore', invalid='ignore'):
            z_stats = np.divide(s_valid - expected, std_dev,
                               out=np.zeros_like(s_valid, dtype=float),
                               where=std_dev > 0)
        
        # 両側検定のp値
        p_approx = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
        
        p_values[valid_mask] = p_approx
    
    return p_values

def format_cluster_results_to_existing_format(results_data, unique_sender_types, unique_receiver_types, 
                                            gene_names, sender_type_col, receiver_type_col, role, up_rate):
    """
    クラスタ結果の既存フォーマット整理
    """
    
    # データフレーム作成
    df = pd.DataFrame(results_data)
    
    # 既存フォーマット用の列名調整
    if role == "receiver":
        df['cell1_type'] = df['receiver_type'] 
        df['cell2_type'] = df['sender_type']
    else:
        df['cell1_type'] = df['sender_type']
        df['cell2_type'] = df['receiver_type']
    
    # 基本統計の追加
    df['coactivity_per_sender_cell_expr_ligand'] = np.divide(
        df['interaction_positive'], df['sender_positive'],
        out=np.zeros_like(df['interaction_positive'], dtype=float),
        where=df['sender_positive'] > 0
    )
    
    # 従来の統計検定（簡易版）
    print("Computing traditional statistics...")
    df = add_traditional_statistics_cluster(df, up_rate)
    
    # 強化された統計の有意性判定
    df['enhanced_significant'] = (df['enhanced_fisher_p'] < 0.05) & (df['enhanced_fisher_p'].notna())
    df['baseline_significant'] = (df['baseline_binomial_p'] < 0.05) & (df['baseline_binomial_p'].notna())
    
    # Multiple testing correction（高速版）
    print("Applying multiple testing correction...")
    df = add_fast_multiple_testing_correction_cluster(df)
    
    return df

def add_traditional_statistics_cluster(df, up_rate):
    """
    クラスタ用従来統計の高速追加
    """
    
    # 母集団レート計算（リガンド別）
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
    
    # 各行にマップ
    df['population_mean_rate'] = df['ligand'].map(population_rates)
    expected_rates = up_rate * df['population_mean_rate']
    
    # Binomial test（ベクトル化）
    valid_mask = (df['sender_positive'] > 0) & (expected_rates <= 1.0) & (expected_rates > 0)
    
    p_values = np.full(len(df), np.nan)
    
    if np.any(valid_mask):
        # 正規近似を使用
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
    
    # Beta信頼区間（ベクトル化）
    alpha_post = df['interaction_positive'] + 0.5
    beta_post = df['sender_positive'] - df['interaction_positive'] + 0.5
    
    alpha_post = np.maximum(alpha_post, 0.5)
    beta_post = np.maximum(beta_post, 0.5)
    
    df['ci_lower_beta'] = beta.ppf(0.025, alpha_post, beta_post)
    df['ci_upper_beta'] = beta.ppf(0.975, alpha_post, beta_post)
    
    return df

def add_fast_multiple_testing_correction_cluster(df):
    """
    クラスタ用高速多重検定補正
    """
    
    # 従来のp値
    valid_p = df['p_value'].dropna()
    if len(valid_p) > 0:
        corrected = multipletests(valid_p, method='bonferroni')
        df.loc[df['p_value'].notna(), 'p_value_bonferroni'] = corrected[1]
        df.loc[df['p_value'].notna(), 'is_significant_bonferroni'] = corrected[0]
    else:
        df['p_value_bonferroni'] = np.nan
        df['is_significant_bonferroni'] = False
    
    # 強化されたp値
    valid_enhanced_p = df['enhanced_fisher_p'].dropna()
    if len(valid_enhanced_p) > 0:
        corrected_enhanced = multipletests(valid_enhanced_p, method='bonferroni')
        df.loc[df['enhanced_fisher_p'].notna(), 'enhanced_fisher_p_bonferroni'] = corrected_enhanced[1]
        df.loc[df['enhanced_fisher_p'].notna(), 'enhanced_significant_bonferroni'] = corrected_enhanced[0]
    else:
        df['enhanced_fisher_p_bonferroni'] = np.nan
        df['enhanced_significant_bonferroni'] = False
    
    # ベースラインp値
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
    細胞種×クラスタ×リガンドの詳細解析
    """
    
    detailed_stats = celltype_cluster_data.copy()
    
    # 条件付き確率の計算
    detailed_stats['response_rate_with_high_stimulation'] = np.divide(
        detailed_stats['interaction_positive'],
        detailed_stats['high_stimulation_environment'],
        out=np.zeros_like(detailed_stats['interaction_positive'], dtype=float),
        where=detailed_stats['high_stimulation_environment'] > 0
    )
    
    # 低刺激環境での反応率計算
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
    
    # 刺激強化効果
    detailed_stats['stimulation_enhancement'] = (
        detailed_stats['response_rate_with_high_stimulation'] - 
        detailed_stats['response_rate_with_low_stimulation']
    )
    
    # 簡易統計検定（大量のデータなので高速版）
    detailed_stats['is_significant'] = (
        (detailed_stats['high_stimulation_environment'] >= 5) &
        (detailed_stats['low_stimulation_opportunities'] >= 5) &
        (detailed_stats['stimulation_enhancement'] > 0.01)  # 1%以上の強化効果
    )
    
    return detailed_stats

def compute_detailed_cluster_analysis(celltype_cluster_data):
    """
    細胞種×クラスタ×リガンドの詳細解析
    """
    
    detailed_stats = celltype_cluster_data.copy()
    
    # 条件付き確率の計算
    detailed_stats['response_rate_with_high_stimulation'] = np.divide(
        detailed_stats['interaction_positive'],
        detailed_stats['high_stimulation_environment'],
        out=np.zeros_like(detailed_stats['interaction_positive'], dtype=float),
        where=detailed_stats['high_stimulation_environment'] > 0
    )
    
    # 低刺激環境での反応率計算
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
    
    # 刺激強化効果
    detailed_stats['stimulation_enhancement'] = (
        detailed_stats['response_rate_with_high_stimulation'] - 
        detailed_stats['response_rate_with_low_stimulation']
    )
    
    # 簡易統計検定（大量のデータなので高速版）
    detailed_stats['is_significant'] = (
        (detailed_stats['high_stimulation_environment'] >= 5) &
        (detailed_stats['low_stimulation_opportunities'] >= 5) &
        (detailed_stats['stimulation_enhancement'] > 0.01)  # 1%以上の強化効果
    )
    
    return detailed_stats

def calculate_cumulative_ligand_coexpression_analysis(edge_df, center_adata, exp_data, expr_up_by_ligands, 
                                                    sp_adata, neighbor_cell_numbers=19, 
                                                    top_percentile_threshold=1.0, role="receiver", 
                                                    up_rate=1.25):
    """
    累積リガンド刺激による細胞間相互作用解析
    """
    
    print(f"Cumulative ligand stimulation analysis (top {top_percentile_threshold}% threshold)")
    
    # データ準備
    center_adata.X = exp_data
    gene_names = center_adata.var_names.tolist()
    n_genes = len(gene_names)
    
    # エッジの再構築：中心細胞ごとにグループ化
    print("Reconstructing neighborhood relationships...")
    neighborhood_data = reconstruct_neighborhoods(edge_df, neighbor_cell_numbers)
    
    # 累積リガンド発現量の計算
    print("Computing cumulative ligand expressions...")
    cumulative_ligand_expr = compute_cumulative_ligand_expression(
        neighborhood_data, center_adata, exp_data, gene_names
    )
    
    # 高刺激環境の定義（Top percentile）
    print(f"Defining high stimulation environments (top {top_percentile_threshold}%)...")
    high_stimulation_mask = define_high_stimulation_environments(
        cumulative_ligand_expr, top_percentile_threshold
    )
    
    # 中心細胞の反応データ
    center_cell_responses = get_center_cell_responses(
        neighborhood_data, expr_up_by_ligands, gene_names
    )
    
    # 累積刺激による相互作用解析
    print("Analyzing cumulative stimulation interactions...")
    interaction_results = analyze_cumulative_interactions(
        neighborhood_data, cumulative_ligand_expr, high_stimulation_mask,
        center_cell_responses, gene_names, top_percentile_threshold
    )
    
    # 細胞種別・微小環境クラスタ別の比較解析
    print("Performing cell type and microenvironment cluster analysis...")
    celltype_analysis = perform_celltype_cluster_analysis(
        interaction_results, neighborhood_data, sp_adata
    )
    
    # 統計検定とフォーマット
    print("Computing statistics and formatting results...")
    final_results, detailed_cluster_results = compute_cumulative_statistics_and_format(
        interaction_results, celltype_analysis, up_rate
    )
    
    # ベースライン比較
    baseline_results = compute_cumulative_baseline_comparison(
        final_results, sp_adata, expr_up_by_ligands, gene_names
    )
    
    # 結果の統合（インデックスをリセットして結合）
    final_results_clean = final_results.reset_index(drop=True)
    baseline_results_clean = baseline_results.reset_index(drop=True)
    coexp_cc_df = pd.concat([final_results_clean, baseline_results_clean], axis=1)
    
    # bargraph_df相当のデータ作成
    bargraph_df = create_cumulative_bargraph_data(
        neighborhood_data, cumulative_ligand_expr, center_cell_responses, gene_names
    )
    
    # 結果サマリー
    print_cumulative_analysis_summary(coexp_cc_df, top_percentile_threshold)
    
    # 詳細なクラスタ解析結果も返す
    return coexp_cc_df, bargraph_df, detailed_cluster_results

def reconstruct_neighborhoods(edge_df, neighbor_cell_numbers):
    """
    エッジデータから近傍関係を再構築
    """
    neighborhoods = {}
    
    # 中心細胞ごとにグループ化
    if 'cell1' in edge_df.columns and 'cell2' in edge_df.columns:
        # cell1を中心細胞、cell2を近傍細胞と仮定
        grouped = edge_df.groupby('cell1')
        
        for center_cell, group in grouped:
            neighbor_cells = group['cell2'].tolist()
            neighborhoods[center_cell] = {
                'center_cell': center_cell,
                'neighbor_cells': neighbor_cells[:neighbor_cell_numbers],  # 最大19個
                'center_cell_type': group['cell1_type'].iloc[0],
                'neighbor_cell_types': group['cell2_type'].tolist()[:neighbor_cell_numbers],
                'microenvironment_cluster': group.get('cell1_cluster', pd.Series(['unknown'] * len(group))).iloc[0],  # cell1_clusterを使用
                'edge_indices': group.index.tolist()[:neighbor_cell_numbers]
            }
    
    neighborhood_df = pd.DataFrame.from_dict(neighborhoods, orient='index')
    
    print(f"Reconstructed {len(neighborhood_df)} neighborhoods")
    print(f"Microenvironment clusters found: {neighborhood_df['microenvironment_cluster'].nunique()}")
    
    return neighborhood_df

def compute_cumulative_ligand_expression(neighborhood_data, center_adata, exp_data, gene_names):
    """
    各中心細胞の近傍からの累積リガンド発現量を計算
    exp_data は数値（発現量）
    """
    n_centers = len(neighborhood_data)
    n_genes = len(gene_names)
    
    cumulative_expr = np.zeros((n_centers, n_genes), dtype=np.float32)
    
    # 細胞名からインデックスへのマッピング
    cell_to_idx = {cell: idx for idx, cell in enumerate(center_adata.obs_names)}
    
    print(f"Computing cumulative ligand expression for {n_centers} centers, {n_genes} genes")
    print(f"exp_data shape: {exp_data.shape}, dtype: {exp_data.dtype}")
    
    for i, (center_cell, row) in enumerate(neighborhood_data.iterrows()):
        neighbor_cells = row['neighbor_cells']
        
        # 近傍細胞のインデックスを取得
        neighbor_indices = []
        for neighbor_cell in neighbor_cells:
            if neighbor_cell in cell_to_idx:
                neighbor_indices.append(cell_to_idx[neighbor_cell])
        
        if neighbor_indices:
            # 近傍細胞の発現データを取得（exp_dataは数値）
            neighbor_expr = exp_data[neighbor_indices]
            if hasattr(neighbor_expr, 'toarray'):
                neighbor_expr = neighbor_expr.toarray()
            
            # リガンド発現量の総和を計算
            cumulative_expr[i] = np.sum(neighbor_expr, axis=0)
            
        if i % 10000 == 0:  # 進捗表示
            print(f"Processed {i}/{n_centers} centers")
    
    print(f"Cumulative expression computed. Mean: {np.mean(cumulative_expr):.3f}, Max: {np.max(cumulative_expr):.3f}")
    
    return cumulative_expr

def define_high_stimulation_environments(cumulative_expr, top_percentile_threshold):
    """
    各リガンドについて高刺激環境を定義
    cumulative_expr: 数値の累積発現量
    """
    n_centers, n_genes = cumulative_expr.shape
    high_stimulation_mask = np.zeros((n_centers, n_genes), dtype=bool)
    
    # リガンドごとに閾値を計算
    percentile_threshold = 100 - top_percentile_threshold
    
    print(f"Defining high stimulation environments (top {top_percentile_threshold}%)")
    
    for gene_idx in range(n_genes):
        gene_expr = cumulative_expr[:, gene_idx]
                
        if len(gene_expr) > 0:
            threshold = np.percentile(gene_expr, percentile_threshold)
            # threshold = 0
            high_stimulation_mask[:, gene_idx] = gene_expr > threshold
            
            # デバッグ情報（最初の数個のみ）
            if gene_idx < 5:
                n_high = np.sum(high_stimulation_mask[:, gene_idx])
                print(f"  Gene {gene_idx}: threshold={threshold:.3f}, high_stim_cells={n_high} ({n_high/n_centers*100:.1f}%)")
        else:
            # 全て0の場合は高刺激環境なし
            high_stimulation_mask[:, gene_idx] = False
    
    total_high_stim = np.sum(high_stimulation_mask)
    total_possible = n_centers * n_genes
    print(f"Total high stimulation environments: {total_high_stim} / {total_possible} ({total_high_stim/total_possible*100:.1f}%)")
    
    return high_stimulation_mask

def get_center_cell_responses(neighborhood_data, expr_up_by_ligands, gene_names):
    """
    中心細胞の反応データを取得
    expr_up_by_ligands は既にboolean（反応あり/なし）
    """
    n_centers = len(neighborhood_data)
    n_genes = len(gene_names)
    
    center_responses = np.zeros((n_centers, n_genes), dtype=bool)
    
    # 中心細胞のインデックスを取得（neighborhood_dataのインデックス順）
    center_cell_names = neighborhood_data.index.tolist()
    
    print(f"Processing responses for {len(center_cell_names)} center cells")
    print(f"expr_up_by_ligands shape: {expr_up_by_ligands.shape}, dtype: {expr_up_by_ligands.dtype}")
    
    # expr_up_by_ligandsは既にbooleanなのでそのまま使用
    for i, center_cell in enumerate(center_cell_names):
        if i < expr_up_by_ligands.shape[0]:
            if hasattr(expr_up_by_ligands, 'toarray'):
                # スパース行列の場合
                response_data = expr_up_by_ligands[i, :n_genes].toarray().flatten()
                center_responses[i] = response_data.astype(bool)
            else:
                # 密行列の場合
                center_responses[i] = expr_up_by_ligands[i, :n_genes].astype(bool)
    
    print(f"Center responses shape: {center_responses.shape}, dtype: {center_responses.dtype}")
    print(f"Response rate: {np.mean(center_responses):.3f}")
    
    return center_responses

def analyze_cumulative_interactions(neighborhood_data, cumulative_expr, high_stimulation_mask, 
                                  center_responses, gene_names, top_percentile_threshold):
    """
    累積刺激による相互作用解析
    center_responses: boolean array (反応あり/なし)
    cumulative_expr: float array (累積発現量)
    """
    interaction_results = []
    
    n_centers, n_genes = cumulative_expr.shape
    
    print(f"Analyzing {n_centers} centers × {n_genes} genes = {n_centers * n_genes} interactions")
    
    for center_idx, (center_cell, row) in enumerate(neighborhood_data.iterrows()):
        center_cell_type = row['center_cell_type']
        microenv_cluster = row['microenvironment_cluster']
        
        for gene_idx, gene in enumerate(gene_names):
            # 高刺激環境かどうか
            high_stimulation = high_stimulation_mask[center_idx, gene_idx]
            
            # 中心細胞の反応（既にboolean）
            center_response = center_responses[center_idx, gene_idx]
            
            # 累積発現量（数値）
            cumulative_value = cumulative_expr[center_idx, gene_idx]
            
            interaction_results.append({
                'center_cell': center_cell,
                'center_cell_type': center_cell_type,
                'microenvironment_cluster': microenv_cluster,
                'ligand': gene,
                'cumulative_ligand_expression': cumulative_value,
                'high_stimulation_environment': high_stimulation,
                'center_cell_response': center_response,
                'interaction_positive': high_stimulation and center_response,  # 両方がTrueの場合
                'stimulation_positive': high_stimulation,
                'response_positive': center_response
            })
        
        if center_idx % 10000 == 0:  # 進捗表示
            print(f"Analyzed {center_idx}/{n_centers} centers")
    
    results_df = pd.DataFrame(interaction_results)
    
    # 結果の要約
    print(f"Interaction analysis complete:")
    print(f"  Total interactions: {len(results_df)}")
    print(f"  High stimulation environments: {results_df['high_stimulation_environment'].sum()} ({results_df['high_stimulation_environment'].mean()*100:.1f}%)")
    print(f"  Center cell responses: {results_df['center_cell_response'].sum()} ({results_df['center_cell_response'].mean()*100:.1f}%)")
    print(f"  Interaction positive: {results_df['interaction_positive'].sum()} ({results_df['interaction_positive'].mean()*100:.1f}%)")
    
    return results_df

def perform_celltype_cluster_analysis(interaction_results, neighborhood_data, sp_adata):
    """
    細胞種・微小環境クラスタ別の解析
    """
    
    # 細胞種別の集約（microenvironment_clusterを含める）
    celltype_analysis = interaction_results.groupby(['center_cell_type', 'microenvironment_cluster', 'ligand']).agg({
        'high_stimulation_environment': 'sum',
        'center_cell_response': 'sum', 
        'interaction_positive': 'sum',
        'center_cell': 'count'  # 総数
    }).rename(columns={'center_cell': 'total_observations'}).reset_index()
    
    # 微小環境クラスタのみの集約も作成
    cluster_only_analysis = interaction_results.groupby(['microenvironment_cluster', 'ligand']).agg({
        'high_stimulation_environment': 'sum',
        'center_cell_response': 'sum',
        'interaction_positive': 'sum', 
        'center_cell': 'count'
    }).rename(columns={'center_cell': 'total_observations'}).reset_index()
    
    # 細胞種のみの集約（従来と同じ）
    celltype_only_analysis = interaction_results.groupby(['center_cell_type', 'ligand']).agg({
        'high_stimulation_environment': 'sum',
        'center_cell_response': 'sum',
        'interaction_positive': 'sum',
        'center_cell': 'count'
    }).rename(columns={'center_cell': 'total_observations'}).reset_index()
    
    return {
        'celltype_cluster_analysis': celltype_analysis,  # 細胞種 × クラスタ × リガンド
        'cluster_analysis': cluster_only_analysis,       # クラスタ × リガンド
        'celltype_analysis': celltype_only_analysis      # 細胞種 × リガンド（従来）
    }

def compute_cumulative_statistics_and_format(interaction_results, celltype_analysis, up_rate):
    """
    統計計算と結果フォーマット
    """
    
    # 従来の細胞種のみの解析
    celltype_stats = celltype_analysis['celltype_analysis'].copy()
    
    # 条件付き確率の計算
    celltype_stats['response_rate_with_high_stimulation'] = np.divide(
        celltype_stats['interaction_positive'],
        celltype_stats['high_stimulation_environment'],
        out=np.zeros_like(celltype_stats['interaction_positive'], dtype=float),
        where=celltype_stats['high_stimulation_environment'] > 0
    )
    
    # 低刺激環境での反応率計算
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
    
    # 刺激強化効果
    celltype_stats['stimulation_enhancement'] = (
        celltype_stats['response_rate_with_high_stimulation'] - 
        celltype_stats['response_rate_with_low_stimulation']
    )
    
    # Fisher exact test
    celltype_stats['fisher_p_value'] = np.nan
    celltype_stats['odds_ratio'] = np.nan
    
    for idx, row in celltype_stats.iterrows():
        # 2x2分割表
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
    
    # 有意性判定
    celltype_stats['is_significant'] = (celltype_stats['odds_ratio'] > up_rate) & \
    (celltype_stats['fisher_p_value'] < 0.05) & (celltype_stats['fisher_p_value'].notna()) & \
    (celltype_stats['interaction_positive'] >= 5)
    
    # 細胞種×クラスタ×リガンドの詳細解析も作成
    detailed_stats = compute_detailed_cluster_analysis(celltype_analysis['celltype_cluster_analysis'])
    
    return celltype_stats, detailed_stats

def compute_cumulative_baseline_comparison(results_df, sp_adata, expr_up_by_ligands, gene_names):
    """
    累積刺激でのベースライン比較
    """
    
    # DataFrameのコピーを作成してインデックスをリセット
    results_clean = results_df.reset_index(drop=True).copy()
    
    # 細胞タイプ別ベースライン反応率
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
    
    # 各行にベースライン情報を追加
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
    
    # 新しいDataFrameを作成
    baseline_df = pd.DataFrame({
        'baseline_response_rate': baseline_response_rates,
        'baseline_binomial_p': baseline_binomial_ps,
        'baseline_significant': baseline_significants
    })
    
    return baseline_df

def create_cumulative_bargraph_data(neighborhood_data, cumulative_expr, center_responses, gene_names):
    """
    bargraph_df相当のデータ作成
    """
    
    bargraph_data = {
        'center_cell_type': [row['center_cell_type'] for _, row in neighborhood_data.iterrows()],
        'microenvironment_cluster': [row['microenvironment_cluster'] for _, row in neighborhood_data.iterrows()]
    }
    
    # 各リガンドの累積発現量と中心細胞反応の積
    for i, gene in enumerate(gene_names):
        bargraph_data[f'cumulative_{gene}'] = cumulative_expr[:, i]
        bargraph_data[f'response_{gene}'] = center_responses[:, i]
        bargraph_data[f'interaction_{gene}'] = cumulative_expr[:, i] * center_responses[:, i]
    
    bargraph_df = pd.DataFrame(bargraph_data)
    
    return bargraph_df

def print_cumulative_analysis_summary(results_df, top_percentile_threshold):
    """
    解析結果のサマリー表示
    """
    
    total_combinations = len(results_df)
    significant_interactions = len(results_df[results_df['is_significant'] == True])
    
    # baseline_significantが存在するかチェック
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
        # インデックスを重複しないようにリセット
        results_clean = results_df.reset_index(drop=True)
        significant_results = results_clean[results_clean['is_significant'] == True]
        
        if len(significant_results) > 0:
            top_enhancements = significant_results.nlargest(10, 'stimulation_enhancement')
            
            print(f"\nTop 10 Stimulation Enhancements:")
            for _, row in top_enhancements.iterrows():
                print(f"  {row['center_cell_type']} + {row['ligand']}: "
                      f"High stimulation response {row['response_rate_with_high_stimulation']:.3f}, "
                      f"Low stimulation response {row['response_rate_with_low_stimulation']:.3f}, "
                      f"Enhancement: +{row['stimulation_enhancement']:.3f} "
                      f"(p={row['fisher_p_value']:.2e})")

# 微小環境クラスタ特異的解析
def analyze_microenvironment_cluster_effects(coexp_cc_df, top_n_clusters=10):
    """
    微小環境クラスタ別の効果解析
    """
    
    print(f"\n=== Microenvironment Cluster Analysis ===")
    
    # DataFrameのクリーンアップ
    df_clean = coexp_cc_df.reset_index(drop=True)
    
    # microenvironment_clusterカラムが存在するかチェック
    if 'microenvironment_cluster' not in df_clean.columns:
        print("Warning: microenvironment_cluster column not found in results")
        print(f"Available columns: {list(df_clean.columns)}")
        return pd.DataFrame()
    
    # クラスタ値の確認
    cluster_values = df_clean['microenvironment_cluster'].value_counts()
    print(f"Found {len(cluster_values)} unique microenvironment clusters:")
    print(cluster_values.head())
    
    # クラスタ別の統計
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

# 使用例
def run_cumulative_analysis_with_clusters(edge_df, center_adata, exp_data, expr_up_by_ligands, 
                                        sp_adata, top_percentile_threshold=1.0, up_rate=1.25):
    """
    累積刺激解析と微小環境クラスタ解析の実行
    """
    
    # メイン解析（詳細クラスタ結果も取得）
    coexp_cc_df, bargraph_df, detailed_cluster_results = calculate_cumulative_ligand_coexpression_analysis(
        edge_df, center_adata, exp_data, expr_up_by_ligands, sp_adata,
        neighbor_cell_numbers=19, top_percentile_threshold=top_percentile_threshold,
        role="receiver", up_rate=up_rate
    )
    
    # 微小環境クラスタ解析（詳細結果を使用）
    cluster_analysis = analyze_microenvironment_cluster_effects_detailed(detailed_cluster_results)
    
    return {
        'interaction_results': coexp_cc_df,              # 細胞種 × リガンド
        'detailed_cluster_results': detailed_cluster_results,  # 細胞種 × クラスタ × リガンド
        'bargraph_data': bargraph_df,
        'cluster_analysis': cluster_analysis
    }

def analyze_microenvironment_cluster_effects_detailed(detailed_cluster_results, top_n_clusters=10):
    """
    詳細クラスタ結果を使った微小環境解析
    """
    
    print(f"\n=== Microenvironment Cluster Analysis (Detailed) ===")
    
    if detailed_cluster_results is None or len(detailed_cluster_results) == 0:
        print("No detailed cluster results available")
        return pd.DataFrame()
    
    # DataFrameのクリーンアップ
    df_clean = detailed_cluster_results.reset_index(drop=True)
    
    print(f"Detailed results shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")
    
    if 'microenvironment_cluster' in df_clean.columns:
        # クラスタ値の確認
        cluster_values = df_clean['microenvironment_cluster'].value_counts()
        print(f"Found {len(cluster_values)} unique microenvironment clusters in detailed results:")
        
        # クラスタ別の統計
        cluster_stats = df_clean.groupby('microenvironment_cluster').agg({
            'is_significant': 'sum',
            'stimulation_enhancement': ['mean', 'max'],
            'center_cell_type': 'count'
        }).round(3)
        
        cluster_stats.columns = ['significant_interactions', 'mean_enhancement', 'max_enhancement', 'total_combinations']
        cluster_stats = cluster_stats.sort_values('significant_interactions', ascending=False)
        
        print(f"\nTop {top_n_clusters} clusters by significant interactions:")
        print(cluster_stats.head(top_n_clusters))
        
        # 各クラスタの上位相互作用も表示
        print(f"\nTop interactions per cluster:")
        for cluster in cluster_stats.head(5).index:  # 上位5クラスタ
            cluster_data = df_clean[df_clean['microenvironment_cluster'] == cluster]
            if len(cluster_data) > 0:
                significant_data = cluster_data[cluster_data['is_significant'] == True]
                if len(significant_data) > 0:
                    top_in_cluster = significant_data.nlargest(3, 'stimulation_enhancement')
                    print(f"  Cluster {cluster}:")
                    for _, row in top_in_cluster.iterrows():
                        print(f"    {row['center_cell_type']} + {row['ligand']}: +{row['stimulation_enhancement']:.3f}")
        
        return cluster_stats
    else:
        print("microenvironment_cluster column not found in detailed results")
        return pd.DataFrame()