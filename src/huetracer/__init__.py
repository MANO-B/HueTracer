"""
HueTracer - Spatial transcriptomics analysis toolkit

This package provides tools for:
- Cell-cell interaction analysis
- Batch effect correction and integration
- Single-cell label transfer
- Spatial microenvironment analysis using VAE
- Gene expression visualization and analysis
"""

from .core.preprocessing import (
    Bin2CellLoadParams,
    Bin2CellLoadResult,
    CropRegion,
    install_bin2cell_safety_override,
    update_adata_for_crop,
    load_visium_and_preprocess,
)

from .core.nucleus_segmentation import (
    segment_nuclei_stardist,
    expand_nuclei_labels,
    generate_gex_segment_image,
)

from .core.spatial_roi import (
    CropConfig,
    auto_detect_tissue_bounds,
    load_spatial_h5ad,
    get_row_col_range,
    sanitize_bounds,
    make_array_row_col_mask,
    apply_array_row_col_mask,
)

from .core.image_ops import (
    get_image_dimensions,
    save_cropped_image,
)

from .core.statistics import (
    wilson_score_interval_vectorized,
    compute_population_rates_vectorized,
    beta_binomial_test_vectorized_no_numba,
    calculate_coexpression_coactivity,
    calculate_coexpression_coactivity_cluster,
)

from .core.vae import SpatialMicroenvironmentAnalyzer

from .core.harmony import run_harmony

from .core.cci import (
    make_coexp_cc_df,
    make_non_zero_values,
    make_positive_values,
    make_top_values,
    safe_toarray,
    add_zscore_layers,
    construct_microenvironment_data,
    prepare_microenv_data,
    MicroenvironmentCCIWorkflow,
    make_celltype_microenv_group_label,
    split_celltype_microenv_group_label,
    prepare_grouped_nichenet_cci_data,
    compute_receiver_group_degs,
    build_spatial_sender_group_profiles,
    calculate_grouped_spatial_nichenet_scores,
    GroupedNicheNetCCIWorkflow,
    calculate_enhanced_coexpression_coactivity,
    calculate_enhanced_coexpression_coactivity_cluster,
    comprehensive_interaction_analysis,
    calculate_cumulative_ligand_coexpression_analysis,
    run_cumulative_analysis_with_clusters,
)

from .core.transfer import (
    SCVILabelTransfer,
    run_scvi_label_transfer,
    analyze_predictions,
    create_confusion_matrix,
    create_spatial_plot,
    create_hires_overlay_plot,
)

from .core.sc_dataset import (
    detect_dataset_type, 
    scan_sc_datasets, 
    read_one, 
    qc_and_filter
)

# I/O Modules
from .io.config import (
    PathConfig,
    build_path_config,
    save_path_config_json,
    validate_path_config,
    scan_config_files,
    load_config_file,
    as_dict,
)

from .io.visium import (
    load_reference_images,
)

from .io.saver import (
    save_anndata,
    save_anndata_batch,
    format_bytes,
)

from .io.sc_filter_save import (
    fmt_bytes,
    parse_exclude,
    save_filtered_sc_adata,
)

# Plotting and Visualization Modules
from .plotting.plot import (
    plot_all_cell_type_highlights,
    plot_all_clusters_highlights,
    plot_gene_cci_and_sankey,
    analyze_cell_proximity,
    plot_spatial_plotly_fast,
)

# Backward compatibility: legacy tutorials use huetracer.plot.*
from .plotting import plot as plot

from .plotting.sc_qc import (
    plot_qc,
)

# Interactive Widgets and UI Modules
from .widgets.selection import (
    LassoCellSelectorMicroenvironment,
    lasso_selection_microenvironment,
    LassoCellSelectorCellType,
    lasso_selection_cell_type,
    DistanceMicroenvironmentSelector,
    distance_selection_microenvironment,
)

from .widgets.widget import (
    SpatialGeneExpressionViewer,
    VolcanoPlotter,
)

from .widgets.config import (
    PathSetupWidget,
    create_path_setup_widget,
    ConfigSelector,
)

from .widgets.bin2cell import (
    Bin2CellLoaderWidget,
    create_bin2cell_loader_widget,
    Bin2CellConverterWidget,
    create_bin2cell_converter_widget,
)

from .widgets.roi_selector import (
    ImageCropperWidget,
    ROISelectorWidget,
    create_roi_selector_widget,
)

from .widgets.export import (
    SaveAnnDataWidget,
    create_save_anndata_widget,
)

from .widgets.sc_widget import (
    SCFilterWidget,
    SCPlotWidget,
    show_sc_filter_save_widget,
)

from .widgets.cluster_annotation_helper import (
    ClusterAnnotationHelper,
)

from .widgets.cci import (
    NicheNetDownloaderWidget,
    create_nichenet_downloader_widget,
)

# Explicit all variable for cleaner star imports
__all__ = [
    # Core - Statistics
    "wilson_score_interval_vectorized",
    "compute_population_rates_vectorized",
    "beta_binomial_test_vectorized_no_numba",
    "calculate_coexpression_coactivity",
    "calculate_coexpression_coactivity_cluster",
    # Core - VAE
    "SpatialMicroenvironmentAnalyzer",
    # Core - Harmony
    "run_harmony",
    # Core - CCI
    "make_coexp_cc_df",
    "make_non_zero_values",
    "make_positive_values",
    "make_top_values",
    "safe_toarray",
    "add_zscore_layers",
    "construct_microenvironment_data",
    "prepare_microenv_data",
    "MicroenvironmentCCIWorkflow",
    "make_celltype_microenv_group_label",
    "split_celltype_microenv_group_label",
    "prepare_grouped_nichenet_cci_data",
    "compute_receiver_group_degs",
    "build_spatial_sender_group_profiles",
    "calculate_grouped_spatial_nichenet_scores",
    "GroupedNicheNetCCIWorkflow",
    "calculate_enhanced_coexpression_coactivity",
    "calculate_enhanced_coexpression_coactivity_cluster",
    "comprehensive_interaction_analysis",
    "calculate_cumulative_ligand_coexpression_analysis",
    "run_cumulative_analysis_with_clusters",
    # Core - Transfer
    "SCVILabelTransfer",
    "run_scvi_label_transfer",
    "analyze_predictions",
    "create_confusion_matrix",
    "create_spatial_plot",
    "create_hires_overlay_plot",
    # Core - Bin2Cell Loader
    "Bin2CellLoadParams",
    "Bin2CellLoadResult",
    "CropRegion",
    "install_bin2cell_safety_override",
    "update_adata_for_crop",
    "load_visium_and_preprocess",
    "segment_nuclei_stardist",
    "expand_nuclei_labels",
    "generate_gex_segment_image",
    # Core - SC Dataset
    "detect_dataset_type",
    "scan_sc_datasets",
    "read_one",
    "qc_and_filter",
    # I/O
    "PathConfig",
    "build_path_config",
    "save_path_config_json",
    "validate_path_config",
    "scan_config_files",
    "load_config_file",
    "as_dict",
    "CropConfig",
    "auto_detect_tissue_bounds",
    "load_spatial_h5ad",
    "get_row_col_range",
    "sanitize_bounds",
    "make_array_row_col_mask",
    "apply_array_row_col_mask",
    "save_cropped_image",
    "save_anndata",
    "save_anndata_batch",
    "format_bytes",
    "fmt_bytes",
    "parse_exclude",
    "save_filtered_sc_adata",
    # Plotting
    "plot_all_cell_type_highlights",
    "plot_all_clusters_highlights",
    "plot_gene_cci_and_sankey",
    "analyze_cell_proximity",
    "plot_spatial_plotly_fast",
    "plot_qc",
    "plot",
    # Widgets
    "LassoCellSelectorMicroenvironment",
    "lasso_selection_microenvironment",
    "LassoCellSelectorCellType",
    "lasso_selection_cell_type",
    "DistanceMicroenvironmentSelector",
    "distance_selection_microenvironment",
    "SpatialGeneExpressionViewer",
    "VolcanoPlotter",
    "PathSetupWidget",
    "create_path_setup_widget",
    "ConfigSelector",
    "ImageCropperWidget",
    "Bin2CellLoaderWidget",
    "create_bin2cell_loader_widget",
    "Bin2CellConverterWidget",
    "create_bin2cell_converter_widget",
    "ROISelectorWidget",
    "create_roi_selector_widget",
    "SaveAnnDataWidget",
    "create_save_anndata_widget",
    "SCFilterWidget",
    "SCPlotWidget",
    "show_sc_filter_save_widget",
    "NicheNetDownloaderWidget",
    "create_nichenet_downloader_widget",
]

__version__ = "0.1.0"