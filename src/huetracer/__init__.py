"""
HueTracer - Spatial transcriptomics analysis toolkit

This package provides tools for:
- Cell-cell interaction analysis
- Batch effect correction and integration
- Single-cell label transfer
- Spatial microenvironment analysis using VAE
- Gene expression visualization and analysis
"""

# Core Analysis Modules
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

# Plotting and Visualization Modules
from .plotting.plot import (
    plot_all_cell_type_highlights,
    plot_all_clusters_highlights,
    plot_gene_cci_and_sankey,
    analyze_cell_proximity,
)

# Interactive Widgets and UI Modules
from .widgets.selection import (
    LassoCellSelectorMicroenvironment,
    lasso_selection_microenvironment,
    LassoCellSelectorCellType,
    lasso_selection_cell_type,
)

from .widgets.widget import (
    SpatialGeneExpressionViewer,
    VolcanoPlotter,
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
    # Plotting
    "plot_all_cell_type_highlights",
    "plot_all_clusters_highlights",
    "plot_gene_cci_and_sankey",
    "analyze_cell_proximity",
    # Widgets
    "LassoCellSelectorMicroenvironment",
    "lasso_selection_microenvironment",
    "LassoCellSelectorCellType",
    "lasso_selection_cell_type",
    "SpatialGeneExpressionViewer",
    "VolcanoPlotter",
]

__version__ = "0.1.0"