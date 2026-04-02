"""HueTracer Core Analysis Modules"""

from .preprocessing import (
    Bin2CellLoadParams,
    Bin2CellLoadResult,
    CropRegion,
    dummy_image_shape_check,
    install_bin2cell_safety_override,
    update_adata_for_crop,
    load_visium_and_preprocess,
)

from .nucleus_segmentation import (
    segment_nuclei_stardist,
    expand_nuclei_labels,
    generate_gex_segment_image,
)

from .cci import (
    make_coexp_cc_df, make_non_zero_values, make_positive_values, make_top_values, 
    safe_toarray, add_zscore_layers, construct_microenvironment_data, prepare_microenv_data,
    calculate_enhanced_coexpression_coactivity, calculate_enhanced_coexpression_coactivity_cluster,
    comprehensive_interaction_analysis, calculate_cumulative_ligand_coexpression_analysis,
    run_cumulative_analysis_with_clusters
)

from .scanpy_qc_pipeline import scanpy_qc_pipeline

from .spatial_roi import (
    CropConfig,
    auto_detect_tissue_bounds,
    load_spatial_h5ad,
    get_row_col_range,
    sanitize_bounds,
    make_array_row_col_mask,
    apply_array_row_col_mask,
)

from .image_ops import (
    get_image_dimensions,
    save_cropped_image,
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

from .sc_dataset import (
    detect_dataset_type, scan_sc_datasets, read_one, qc_and_filter
)

__all__ = [
    # spatial_dataset + preprocessing
    'Bin2CellLoadParams',
    'Bin2CellLoadResult',
    'CropRegion',
    'dummy_image_shape_check',
    'install_bin2cell_safety_override',
    'update_adata_for_crop',
    'load_visium_and_preprocess',
    
    # nucleus_segmentation
    'segment_nuclei_stardist',
    'expand_nuclei_labels',
    'generate_gex_segment_image',
    
    # cropper
    'CropConfig',
    'auto_detect_tissue_bounds',
    'load_spatial_h5ad',
    'get_row_col_range',
    'sanitize_bounds',
    'make_array_row_col_mask',
    'apply_array_row_col_mask',
    
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

    # sc_dataset
    'detect_dataset_type', 'scan_sc_datasets', 'read_one', 'qc_and_filter',

    # scanpy_qc_pipeline
    'scanpy_qc_pipeline',
]
