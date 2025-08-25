from scipy import stats
from scipy.stats import beta, binom, norm
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
import warnings

def wilson_score_interval_vectorized(successes, trials, alpha=0.05):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        
        z = stats.norm.ppf(1 - alpha/2)
        n = trials.astype(float)
        successes = successes.astype(float)
        
        # Zero division and invalid data protection
        mask = (n > 0) & np.isfinite(n) & np.isfinite(successes) & (successes >= 0) & (successes <= n)
        
        ci_lower = np.full_like(successes, 0.0, dtype=float)
        ci_upper = np.full_like(successes, 1.0, dtype=float)
        
        if np.any(mask):
            # 有効なデータのみで計算
            valid_n = n[mask]
            valid_successes = successes[mask]
            valid_p_hat = valid_successes / valid_n
            
            # Wilson interval calculation
            denominator = 1 + z**2 / valid_n
            center = (valid_p_hat + z**2 / (2 * valid_n)) / denominator
            margin = z * np.sqrt((valid_p_hat * (1 - valid_p_hat) + z**2 / (4 * valid_n)) / valid_n) / denominator
            
            # Ensure bounds are within [0, 1]
            ci_lower[mask] = np.maximum(0, center - margin)
            ci_upper[mask] = np.minimum(1, center + margin)
        
        return ci_lower, ci_upper

def compute_population_rates_vectorized(coexp_cc_df):
    n_rows = len(coexp_cc_df)
    population_rates = np.full(n_rows, np.nan)
    
    # Group by ligand for efficiency
    for ligand in coexp_cc_df['ligand'].unique():
        ligand_mask = coexp_cc_df['ligand'] == ligand
        ligand_data = coexp_cc_df[ligand_mask]
        
        # Calculate population totals for this ligand
        ligand_total_success = ligand_data['interaction_positive'].sum()
        ligand_total_trials = ligand_data['sender_positive'].sum()
        
        for idx, row in ligand_data.iterrows():
            # Exclude current observation
            current_success = row['interaction_positive']
            current_trials = row['sender_positive']
            
            other_success = ligand_total_success - current_success
            other_trials = ligand_total_trials - current_trials
            
            # Division by zero protection with explicit check
            if other_trials > 0 and not np.isnan(other_trials) and not np.isnan(other_success):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rate = other_success / other_trials
                    # Additional sanity check
                    if np.isfinite(rate) and 0 <= rate <= 1:
                        population_rates[idx] = rate
                    else:
                        population_rates[idx] = 0.0
            else:
                population_rates[idx] = 0.0
    
    return population_rates

def beta_binomial_test_vectorized_no_numba(coexp_cc_df, alpha=0.05, up_rate=1.5):
    # データの準備
    n_rows = len(coexp_cc_df)
    interaction_positive = coexp_cc_df['interaction_positive'].values
    sender_positive = coexp_cc_df['sender_positive'].values
    
    # 母集団レート計算
    population_rates = compute_population_rates_vectorized(coexp_cc_df)
    # ベクトル化されたp値計算
    expected_rates = up_rate * population_rates
    
    # Handle edge cases - より厳密な条件チェック
    valid_mask = (
        (sender_positive > 0) & 
        ~np.isnan(population_rates) & 
        (expected_rates <= 1.0) & 
        (expected_rates >= 0.0) &
        (interaction_positive >= 0) &
        (interaction_positive <= sender_positive)
    )
    
    p_values = np.full(n_rows, np.nan)
    
    # Binomial CDF calculation (vectorized) with better error handling
    if np.any(valid_mask):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valid_interactions = interaction_positive[valid_mask]
            valid_trials = sender_positive[valid_mask]
            valid_rates = expected_rates[valid_mask]
            
            # Additional safety check
            safe_mask = (valid_rates > 0) & (valid_rates < 1)
            if np.any(safe_mask):
                p_values_temp = 1 - binom.cdf(
                    valid_interactions[safe_mask] - 1, 
                    valid_trials[safe_mask], 
                    valid_rates[safe_mask]
                )
                
                # Create full-size array and assign
                temp_p_values = np.full(np.sum(valid_mask), np.nan)
                temp_p_values[safe_mask] = p_values_temp
                p_values[valid_mask] = temp_p_values
    # Beta confidence intervals (vectorized) with division by zero protection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        
        alpha_post = interaction_positive + 0.5
        beta_post = sender_positive - interaction_positive + 0.5
        
        # Ensure positive parameters
        alpha_post = np.maximum(alpha_post, 0.5)
        beta_post = np.maximum(beta_post, 0.5)
        
        ci_lower_beta = beta.ppf(alpha/2, alpha_post, beta_post)
        ci_upper_beta = beta.ppf(1 - alpha/2, alpha_post, beta_post)
    
    # Wilson confidence intervals (vectorized)
    ci_lower_wilson, ci_upper_wilson = wilson_score_interval_vectorized(
        interaction_positive, sender_positive, alpha
    )
    
    # Results compilation - use pd.concat instead of DataFrame construction
    ci_width_beta = ci_upper_beta - ci_lower_beta
    ci_width_wilson = ci_upper_wilson - ci_lower_wilson
    is_significant = (p_values < alpha) & ~np.isnan(p_values)
    # Create all columns at once to avoid fragmentation
    result_data = {
        'p_value': p_values,
        'ci_lower_beta': ci_lower_beta,
        'ci_upper_beta': ci_upper_beta,
        'ci_lower_wilson': ci_lower_wilson,
        'ci_upper_wilson': ci_upper_wilson,
        'ci_width_beta': ci_width_beta,
        'ci_width_wilson': ci_width_wilson,
        'is_significant': is_significant,
        'population_mean_rate': population_rates
    }
    
    results_df = pd.DataFrame(result_data, index=coexp_cc_df.index)
    
    return results_df

def calculate_coexpression_coactivity(edge_df, center_adata, exp_data, expr_up_by_ligands, role="sender", up_rate=1.25):
    
    print("Whole data co-expression calculation...")
    # Prepare data
    center_adata.X = exp_data
    center_adata_receiver = center_adata.copy()
    center_adata_receiver.X = expr_up_by_ligands
    center_adata_receiver.layers['expr_up'] = center_adata_receiver.X.copy()

    # Get expression matrices
    sender = edge_df.cell1 if role == "sender" else edge_df.cell2
    receiver = edge_df.cell2 if role == "sender" else edge_df.cell1

    sender_expr = center_adata[sender].X
    receiver_expr = center_adata_receiver[receiver].X
    
    if hasattr(sender_expr, 'toarray'):
        sender_expr = sender_expr.toarray()
    if hasattr(receiver_expr, 'toarray'):
        receiver_expr = receiver_expr.toarray()

    # Optimized co-expression calculation
    coexp_matrix = sender_expr * receiver_expr
    
    # Create DataFrame efficiently - avoid fragmentation
    gene_names = center_adata.var_names.tolist()
    cell_types = edge_df[['cell1_type', 'cell2_type']].values
    
    # Prepare all data at once to avoid fragmentation
    all_data = {
        'cell1_type': cell_types[:, 0],
        'cell2_type': cell_types[:, 1]
    }
    
    # Add all sender columns at once
    sender_cols = {}
    coexp_cols = {}
    
    for i, gene in enumerate(gene_names):
        sender_cols[f'sender_{gene}'] = sender_expr[:, i]
        coexp_cols[f'coexp_{gene}'] = coexp_matrix[:, i]
    
    # Combine all dictionaries
    all_data.update(sender_cols)
    all_data.update(coexp_cols)
    
    # Create DataFrame in one go
    temp_df = pd.DataFrame(all_data)
    
    # Efficient aggregation
    groupby_cols = ['cell1_type', 'cell2_type']
    
    # Get column lists for aggregation
    sender_gene_cols = [f'sender_{gene}' for gene in gene_names]
    coexp_gene_cols = [f'coexp_{gene}' for gene in gene_names]
    
    # Aggregate all at once
    sender_agg = temp_df.groupby(groupby_cols, observed=False)[sender_gene_cols].sum()
    coexp_agg = temp_df.groupby(groupby_cols, observed=False)[coexp_gene_cols].sum()
    
    # Reshape to long format efficiently using vectorized operations
    
    # Prepare index for results
    group_indices = sender_agg.index
    n_groups = len(group_indices)
    n_genes = len(gene_names)
    
    # Pre-allocate arrays for results
    cell1_types_long = []
    cell2_types_long = []
    ligands_long = []
    sender_pos_long = []
    inter_pos_long = []
    
    # Vectorized approach for long format conversion
    for i, (cell1_type, cell2_type) in enumerate(group_indices):
        # Extract values for this group
        sender_values = sender_agg.iloc[i].values  # All sender gene values
        coexp_values = coexp_agg.iloc[i].values    # All coexp gene values
        
        # Extend lists (vectorized)
        cell1_types_long.extend([cell1_type] * n_genes)
        cell2_types_long.extend([cell2_type] * n_genes)
        ligands_long.extend(gene_names)
        sender_pos_long.extend(sender_values)
        inter_pos_long.extend(coexp_values)
    
    # Create result DataFrame efficiently
    coexp_cc_df = pd.DataFrame({
        'cell1_type': cell1_types_long,
        'cell2_type': cell2_types_long,
        'ligand': ligands_long,
        'sender_positive': sender_pos_long,
        'interaction_positive': inter_pos_long
    })
    
    # Vectorized coactivity calculation
    mask = coexp_cc_df['sender_positive'] > 0
    coexp_cc_df['coactivity_per_sender_cell_expr_ligand'] = 0.0
    coexp_cc_df.loc[mask, 'coactivity_per_sender_cell_expr_ligand'] = (
        coexp_cc_df.loc[mask, 'interaction_positive'] / coexp_cc_df.loc[mask, 'sender_positive']
    )
    
    # Fast statistical testing
    results_df = beta_binomial_test_vectorized_no_numba(coexp_cc_df, alpha=0.05, up_rate=up_rate)
    coexp_cc_df = pd.concat([coexp_cc_df, results_df], axis=1)
    
    # Multiple testing correction
    valid_pvals = coexp_cc_df['p_value'].dropna()
    if len(valid_pvals) > 0:
        corrected_pvals = multipletests(valid_pvals, method='bonferroni')[1]
        
        # Efficient column addition
        bonferroni_cols = pd.DataFrame({
            'p_value_bonferroni': np.nan,
            'is_significant_bonferroni': False
        }, index=coexp_cc_df.index)
        
        bonferroni_cols.loc[coexp_cc_df['p_value'].notna(), 'p_value_bonferroni'] = corrected_pvals
        bonferroni_cols.loc[coexp_cc_df['p_value'].notna(), 'is_significant_bonferroni'] = corrected_pvals < 0.05
        
        coexp_cc_df = pd.concat([coexp_cc_df, bonferroni_cols], axis=1)
    else:
        # Add empty columns efficiently
        empty_cols = pd.DataFrame({
            'p_value_bonferroni': np.nan,
            'is_significant_bonferroni': False
        }, index=coexp_cc_df.index)
        coexp_cc_df = pd.concat([coexp_cc_df, empty_cols], axis=1)
    
    # Create bargraph_df efficiently
    bargraph_data = {
        'cell2_type': edge_df['cell2_type'].values,
        'cell1_type': edge_df['cell1_type'].values
    }
    
    # Add all gene columns at once
    gene_data = {}
    for i, gene in enumerate(gene_names):
        gene_data[gene] = coexp_matrix[:, i]
    
    n_significant = len(coexp_cc_df[coexp_cc_df['is_significant'] == True])
    n_significant_bonf = len(coexp_cc_df[coexp_cc_df['is_significant_bonferroni'] == True])
    bargraph_data.update(gene_data)
    bargraph_df = pd.DataFrame(bargraph_data, index=edge_df.index)
    
    print(f"Completion all cluster data: {len(coexp_cc_df)} interactions")
    print(f"Found {n_significant} significant interactions (uncorrected)")
    print(f"Found {n_significant_bonf} significant interactions (Bonferroni corrected)")

    return coexp_cc_df, bargraph_df
    

def calculate_coexpression_coactivity_cluster(edge_df, center_adata, exp_data, expr_up_by_ligands, 
                                    cluster_label, role="receiver", up_rate=1.25):
    print("Cluster-specific co-expression calculation...")
    
    # Prepare data
    center_adata.X = exp_data
    center_adata_receiver = center_adata.copy()
    center_adata_receiver.X = expr_up_by_ligands

    # Efficient cluster filtering
    if isinstance(cluster_label, (list, tuple)):
        cluster_set = set(str(c) for c in cluster_label)
    else:
        cluster_set = {str(cluster_label)}
    
    # Get valid cells efficiently
    cell_clusters = center_adata.obs['cluster'].astype(str)
    cluster_mask = cell_clusters.isin(cluster_set)
    valid_cell1_indices = set(center_adata.obs_names[cluster_mask])
    
    # Filter interactions
    edge_mask = edge_df['cell1'].isin(valid_cell1_indices)
    filtered_edge_df = edge_df[edge_mask]
    
    if len(filtered_edge_df) == 0:
        print(f"Warning: No interactions found for cluster {cluster_label}")
        return pd.DataFrame(), pd.DataFrame()

    # Get expression data
    sender_ids = filtered_edge_df.cell2.values
    receiver_ids = filtered_edge_df.cell1.values
    
    sender_expr = center_adata[sender_ids].X
    receiver_expr = center_adata_receiver[receiver_ids].X
    
    if hasattr(sender_expr, 'toarray'):
        sender_expr = sender_expr.toarray()
    if hasattr(receiver_expr, 'toarray'):
        receiver_expr = receiver_expr.toarray()

    # Vectorized calculations
    coexp_matrix = sender_expr * receiver_expr
    gene_names = center_adata.var_names.tolist()
    
    # Fix for object dtype issue - use pandas groupby instead of numpy unique
    print("Efficient aggregation using pandas groupby...")
    
    # Create temporary DataFrame for aggregation
    temp_data = {
        'cell1_type': filtered_edge_df['cell1_type'].values,
        'cell2_type': filtered_edge_df['cell2_type'].values
    }
    
    # Add expression data efficiently
    for i, gene in enumerate(gene_names):
        temp_data[f'sender_{gene}'] = sender_expr[:, i]
        temp_data[f'coexp_{gene}'] = coexp_matrix[:, i]
    
    temp_df = pd.DataFrame(temp_data)
    
    # Efficient pandas aggregation
    groupby_cols = ['cell1_type', 'cell2_type']
    sender_cols = [f'sender_{gene}' for gene in gene_names]
    coexp_cols = [f'coexp_{gene}' for gene in gene_names]
    
    # Aggregate using pandas (handles object dtypes correctly)
    sender_agg = temp_df.groupby(groupby_cols, observed=False)[sender_cols].sum()
    coexp_agg = temp_df.groupby(groupby_cols, observed=False)[coexp_cols].sum()
    
    # Convert to long format efficiently    
    results_list = []
    for (cell1_type, cell2_type), sender_row in sender_agg.iterrows():
        coexp_row = coexp_agg.loc[(cell1_type, cell2_type)]
        
        for i, gene in enumerate(gene_names):
            sender_pos = sender_row.iloc[i]  # Use iloc for position-based access
            inter_pos = coexp_row.iloc[i]
            
            results_list.append({
                'cell1_type': cell1_type,
                'cell2_type': cell2_type,
                'ligand': gene,
                'sender_positive': sender_pos,
                'interaction_positive': inter_pos,
                'coactivity_per_sender_cell_expr_ligand': (
                    inter_pos / sender_pos if sender_pos > 0 else 0.0
                )
            })
    
    filtered_coexp_df = pd.DataFrame(results_list)
    
    # Fast statistical testing
    print("Executing statistical tests...")
    results_df = beta_binomial_test_vectorized_no_numba(filtered_coexp_df, alpha=0.05, up_rate=up_rate)
    final_coexp_df = pd.concat([filtered_coexp_df, results_df], axis=1)
    
    # Multiple testing correction
    print("Applying multiple testing correction...")
    valid_pvals = final_coexp_df['p_value'].dropna()
    if len(valid_pvals) > 0:
        corrected_pvals = multipletests(valid_pvals, method='bonferroni')[1]
        bonferroni_cols = pd.DataFrame({
            'p_value_bonferroni': np.nan,
            'is_significant_bonferroni': False
        }, index=final_coexp_df.index)
        bonferroni_cols.loc[final_coexp_df['p_value'].notna(), 'p_value_bonferroni'] = corrected_pvals
        bonferroni_cols.loc[final_coexp_df['p_value'].notna(), 'is_significant_bonferroni'] = corrected_pvals < 0.05
        final_coexp_df = pd.concat([final_coexp_df, bonferroni_cols], axis=1)
    else:
        empty_cols = pd.DataFrame({
            'p_value_bonferroni': np.nan,
            'is_significant_bonferroni': False
        }, index=final_coexp_df.index)
        final_coexp_df = pd.concat([final_coexp_df, empty_cols], axis=1)
    
    # Create bargraph DataFrame efficiently
    print("Creating bargraph DataFrame...")
    bargraph_data = {
        'cell2_type': filtered_edge_df['cell2_type'].values,
        'cell1_type': filtered_edge_df['cell1_type'].values
    }
    gene_data = {gene: coexp_matrix[:, i] for i, gene in enumerate(gene_names)}
    bargraph_data.update(gene_data)
    bargraph_df = pd.DataFrame(bargraph_data, index=filtered_edge_df.index)
    
    n_significant = len(final_coexp_df[final_coexp_df['is_significant'] == True])
    n_significant_bonf = len(final_coexp_df[final_coexp_df['is_significant_bonferroni'] == True])
    
    print(f"Completion: {len(final_coexp_df)} interactions")
    print(f"Found {n_significant} significant interactions (uncorrected)")
    print(f"Found {n_significant_bonf} significant interactions (Bonferroni corrected)")
    
    return final_coexp_df, bargraph_df
