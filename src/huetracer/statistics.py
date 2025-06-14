from scipy import stats
from scipy.stats import beta, binom

def beta_binomial_test_vs_population(success, trials, population_df, alpha=0.05, up_rate = 0.25):
    """
    ベータ二項分布を用いて個々の相互作用が母集団より有意に高いかを検定
    
    Parameters:
    success: interaction_positive (成功数)
    trials: sender_positive (試行数)
    population_df: 母集団のデータフレーム
    alpha: 有意水準
    up_rate: 平均よりもどれだけ相互作用が増加しているかの閾値
    
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
    population_rate = (1 + up_rate) * total_success / total_trials if total_trials > 0 else 0
    
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
