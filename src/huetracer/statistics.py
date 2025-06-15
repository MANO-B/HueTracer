from scipy import stats
from scipy.stats import beta, binom
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np


def beta_binomial_test_vs_population(success, trials, population_df, alpha=0.05, up_rate = 1.5):
    """
    ベータ二項分布を用いて個々の相互作用が母集団の平均のup_rate倍より有意に高いかを検定
    
    Parameters:
    success: interaction_positive (成功数)
    trials: sender_positive (試行数)
    population_df: 母集団のデータフレーム
    alpha: 有意水準
    up_rate: 母集団の平均よりもどれだけ相互作用が増加しているかの閾値
    
    Returns:
    p_value: P値
    ci_lower, ci_upper: 95%信頼区間
    is_significant: 有意かどうか
    population_mean: 母集団平均
    """
    
    # 母集団の平均成功率を計算（現在の観測を除く）
    other_interactions = population_df[
        ~((population_df['interaction_positive'] == success) & 
          (population_df['sender_positive'] == trials))
    ]
    
    if len(other_interactions) == 0:
        # 他のデータがない場合は検定不可
        return np.nan, np.nan, np.nan, False, np.nan
    
    # 重み付き平均（試行数で重み付け）
    total_success = other_interactions['interaction_positive'].sum()
    total_trials = other_interactions['sender_positive'].sum()
    population_rate = up_rate * total_success / total_trials if total_trials > 0 else 0
    
    # 二項検定: 観測値が母集団平均よりup_rate以上有意に高いか
    p_value = 1 - binom.cdf(success - 1, trials, population_rate)
    
    # ベータ分布による信頼区間計算 (Jeffreys prior: Beta(0.5, 0.5))
    # より保守的な信頼区間
    alpha_post = success + 0.5
    beta_post = trials - success + 0.5
    
    ci_lower = beta.ppf(alpha/2, alpha_post, beta_post)
    ci_upper = beta.ppf(1 - alpha/2, alpha_post, beta_post)
    
    is_significant = p_value < alpha
    
    return p_value, ci_lower, ci_upper, is_significant, population_rate

def wilson_score_interval(success, trials, alpha=0.05):
    """
    Wilson score interval (より正確な信頼区間)
    """
    if trials == 0:
        return 0, 1
    
    z = stats.norm.ppf(1 - alpha/2)
    p = success / trials
    
    denominator = 1 + (z**2 / trials)
    centre = (p + (z**2 / (2 * trials))) / denominator
    half_width = z * np.sqrt((p * (1 - p) / trials) + (z**2 / (4 * trials**2))) / denominator
    
    return max(0, centre - half_width), min(1, centre + half_width)

def calculate_coexpression_coactivity(edge_df, center_adata, exp_data, expr_up_by_ligands, role="sender", up_rate = 1.5):
    # Xとexpr_upを更新するコピーを作成
    center_adata.X = exp_data
    center_adata_receiver = center_adata.copy()
    center_adata_receiver.X = expr_up_by_ligands
    center_adata_receiver.layers['expr_up'] = center_adata_receiver.X.copy()

    # 送信者と受信者のID
    sender = edge_df.cell1 if role == "sender" else edge_df.cell2
    receiver = edge_df.cell2 if role == "sender" else edge_df.cell1

    # 共発現を計算
    coexp_cc_df = pd.DataFrame(
        center_adata[sender].X.toarray() * center_adata_receiver[receiver].X.toarray(),
        columns=center_adata.var_names,
        index=edge_df.index
    )
    coexp_cc_df['cell2_type'] = edge_df['cell2_type']
    coexp_cc_df['cell1_type'] = edge_df['cell1_type']
    bargraph_df = coexp_cc_df.copy()
    # sender 側の発現行列
    coexp_cc_df_sender = coexp_cc_df.copy()
    coexp_cc_df_sender.iloc[:, :-2] = center_adata[sender].X.toarray()

    # リガンド列の抽出
    ligand_cols = [col for col in coexp_cc_df.columns if col not in ['cell1_type', 'cell2_type']]

    # 各細胞ペアごとのカウント
    sender_pos_count = (
        coexp_cc_df_sender.groupby(['cell2_type', 'cell1_type'])[ligand_cols]
        .sum().reset_index()
        .rename(columns={lig: lig + '_sender_pos' for lig in ligand_cols})
    )
    inter_pos_count = (
        coexp_cc_df.groupby(['cell2_type', 'cell1_type'])[ligand_cols]
        .sum().reset_index()
        .rename(columns={lig: lig + '_inter_pos' for lig in ligand_cols})
    )

    # long形式に変換
    sender_long = sender_pos_count.melt(id_vars=['cell1_type', 'cell2_type'], 
                                        var_name='ligand', value_name='sender_positive')
    sender_long['ligand'] = sender_long['ligand'].str.replace('_sender_pos', '', regex=False)

    inter_long = inter_pos_count.melt(id_vars=['cell1_type', 'cell2_type'], 
                                      var_name='ligand', value_name='interaction_positive')
    inter_long['ligand'] = inter_long['ligand'].str.replace('_inter_pos', '', regex=False)

    # 結合
    coexp_cc_df = sender_long.merge(inter_long, on=['cell1_type', 'cell2_type', 'ligand'], how='left')

    # 1陽性細胞あたりの共発現
    coexp_cc_df['coactivity_per_sender_cell_expr_ligand'] = (
        coexp_cc_df['interaction_positive'] / coexp_cc_df['sender_positive']
    )
    coexp_cc_df.loc[coexp_cc_df['sender_positive'] == 0, 'coactivity_per_sender_cell_expr_ligand'] = 0

    # 検定実行
    results = []
    for _, row in coexp_cc_df.iterrows():
        success = row['interaction_positive']
        trials = row['sender_positive']

        p_val, ci_low_beta, ci_high_beta, is_sig, pop_mean = beta_binomial_test_vs_population(
            success, trials, coexp_cc_df, alpha=0.05
        )
        ci_low_wilson, ci_high_wilson = wilson_score_interval(success, trials, alpha=0.05, up_rate = up_rate)

        ci_width_beta = ci_high_beta - ci_low_beta if not np.isnan(ci_high_beta) else np.nan
        ci_width_wilson = ci_high_wilson - ci_low_wilson

        results.append({
            'p_value': p_val,
            'ci_lower_beta': ci_low_beta,
            'ci_upper_beta': ci_high_beta,
            'ci_lower_wilson': ci_low_wilson,
            'ci_upper_wilson': ci_high_wilson,
            'ci_width_beta': ci_width_beta,
            'ci_width_wilson': ci_width_wilson,
            'is_significant': is_sig,
            'population_mean_rate': pop_mean
        })

    results_df = pd.DataFrame(results)
    coexp_cc_df = pd.concat([coexp_cc_df, results_df], axis=1)

    # 多重検定補正（Bonferroni）
    valid_pvals = coexp_cc_df['p_value'].dropna()
    if len(valid_pvals) > 0:
        corrected_pvals = multipletests(valid_pvals, method='bonferroni')[1]
        coexp_cc_df['p_value_bonferroni'] = np.nan
        coexp_cc_df.loc[coexp_cc_df['p_value'].notna(), 'p_value_bonferroni'] = corrected_pvals
        coexp_cc_df['is_significant_bonferroni'] = coexp_cc_df['p_value_bonferroni'] < 0.05

    return coexp_cc_df, bargraph_df
