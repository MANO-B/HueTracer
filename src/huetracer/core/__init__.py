"""HueTracer Core Analysis Modules"""

from .cci import (
    make_coexp_cc_df, make_non_zero_values, make_positive_values, make_top_values, 
    safe_toarray, add_zscore_layers, construct_microenvironment_data, prepare_microenv_data,
    calculate_enhanced_coexpression_coactivity, calculate_enhanced_coexpression_coactivity_cluster,
    comprehensive_interaction_analysis, calculate_cumulative_ligand_coexpression_analysis,
    run_cumulative_analysis_with_clusters
)

from .harmony import run_harmony, Harmony, safe_entropy, moe_correct_ridge

from .statistics import (
    wilson_score_interval_vectorized, compute_population_rates_vectorized,
    beta_binomial_test_vectorized_no_numba, calculate_coexpression_coactivity,
    calculate_coexpression_coactivity_cluster
)

from .transfer import (
    SCVILabelTransfer, run_scvi_label_transfer, analyze_predictions,
    create_confusion_matrix, create_spatial_plot, create_hires_overlay_plot
)

from .vae import (
    SpatialMicroenvironmentAnalyzer, MicroenvironmentVAE, vae_loss,
    pick_torch_device, format_time
)

__all__ = [
    # cci
    'make_coexp_cc_df', 'make_non_zero_values', 'make_positive_values', 'make_top_values',
    'safe_toarray', 'add_zscore_layers', 'construct_microenvironment_data', 
    'prepare_microenv_data', 'calculate_enhanced_coexpression_coactivity',
    'calculate_enhanced_coexpression_coactivity_cluster', 'comprehensive_interaction_analysis',
    'calculate_cumulative_ligand_coexpression_analysis', 'run_cumulative_analysis_with_clusters',
    
    # harmony
    'run_harmony', 'Harmony', 'safe_entropy', 'moe_correct_ridge',
    
    # statistics
    'wilson_score_interval_vectorized', 'compute_population_rates_vectorized',
    'beta_binomial_test_vectorized_no_numba', 'calculate_coexpression_coactivity',
    'calculate_coexpression_coactivity_cluster',
    
    # transfer
    'SCVILabelTransfer', 'run_scvi_label_transfer', 'analyze_predictions',
    'create_confusion_matrix', 'create_spatial_plot', 'create_hires_overlay_plot',
    
    # vae
    'SpatialMicroenvironmentAnalyzer', 'MicroenvironmentVAE', 'vae_loss',
    'pick_torch_device', 'format_time',
]
