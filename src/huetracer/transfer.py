import os
import pandas as pd
import numpy as np
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import warnings
from typing import Optional, Dict, Tuple, Any
import logging

# Log settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SCVILabelTransfer:
    """Class for cell type transfer using scANVI"""
    
    def __init__(self, device = "auto"):
        """
        Parameters:
        -----------
        device : str or torch.device
            Computational device ("auto", "cpu", "cuda", "mps", or a torch.device object)
        """
        self.device_str = self._setup_device(device)
        self.device = self.device_str  # For backward compatibility
        self._configure_pytorch_settings()
        logger.info(f"Using device: {self.device_str}")
        
    def _configure_pytorch_settings(self):
        """Basic settings for PyTorch and scvi-tools"""
        try:
            import torch
            # Set default num_workers
            if hasattr(torch.utils.data, '_utils'):
                # Suppress PyTorch DataLoader warnings
                import warnings
                warnings.filterwarnings("ignore", ".*does not have many workers.*")
        except ImportError:
            pass
        
    def _setup_device(self, device) -> str:
        """Device setup - supports torch.device object or string"""
        try:
            import torch
            
            # If it's a torch.device object
            if isinstance(device, torch.device):
                return str(device.type)
            
            # If it's a string
            if device == "auto":
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            else:
                return str(device)
                
        except ImportError:
            logger.warning("PyTorch not found. Using CPU.")
            return "cpu"
    
    def prepare_data(self, 
                     sc_adata: sc.AnnData, 
                     sp_adata: sc.AnnData,
                     cell_type_key: str = "cell_type_annotation") -> sc.AnnData:
        """
        Combines reference and query data and prepares it for scANVI
        
        Parameters:
        -----------
        sc_adata : AnnData
            Reference single-cell data
        sp_adata : AnnData  
            Query spatial data
        cell_type_key : str
            Key for cell type annotation
            
        Returns:
        --------
        adata_combined : AnnData
            Combined data
        """
        logger.info("Starting data preparation...")
        
        # Prepare reference data
        adata_ref = sc_adata.copy()
        #adata_ref.obs['batch'] = 'reference'
        if adata_ref.raw is not None:
            adata_ref.X = adata_ref.raw.X.copy()
            adata_ref.var = adata_ref.raw.var.copy()
            adata_ref.raw = None
            
        # Prepare query data  
        adata_query = sp_adata.copy()
        adata_query.obs['batch'] = 'query'
        if adata_query.raw is not None:
            adata_query.X = adata_query.raw.X.copy()
            adata_query.var = adata_query.raw.var.copy()
            adata_query.raw = None
            
        # Combine data
        adata_combined = adata_ref.concatenate(adata_query, batch_key="dataset")
        adata_combined.obs['batch'] = pd.Categorical(adata_combined.obs['batch'])
        
        logger.info(f"Combined data size: {adata_combined.shape}")
        return adata_combined
    
    def _process_cell_type_labels(self, 
                                  adata: sc.AnnData, 
                                  cell_type_key: str) -> sc.AnnData:
        """Pre-process cell type labels"""
        
        if cell_type_key not in adata.obs.columns:
            raise ValueError(f"Key '{cell_type_key}' not found in adata.obs")
            
        cat_col = adata.obs[cell_type_key].copy()
        
        # Handle categorical type
        if isinstance(cat_col.dtype, pd.CategoricalDtype):
            if "Unknown" not in cat_col.cat.categories:
                cat_col = cat_col.cat.add_categories(["Unknown"])
        
        # Replace missing values with Unknown
        cat_col = cat_col.fillna("Unknown")
        
        # Re-set categories
        categories = cat_col.unique().tolist()
        if "Unknown" not in categories:
            categories.append("Unknown")
            
        adata.obs[cell_type_key] = pd.Categorical(cat_col, categories=categories)
        
        logger.info(f"Number of cell type categories: {len(categories)}")
        return adata
    
    def train_scvi_scanvi(self, 
                          adata_combined: sc.AnnData,
                          cell_type_key: str = "cell_type_annotation",
                          max_epochs: int = 50,
                          early_stopping: bool = True,
                          num_workers: int = 4,
                          batch_size: int = 128) -> Tuple[Any, Any]:
        """
        Train scVI/scANVI models
        
        Parameters:
        -----------
        adata_combined : AnnData
            Combined data
        cell_type_key : str
            Cell type key
        max_epochs : int
            Maximum number of epochs
        early_stopping : bool
            Whether to use early stopping
        num_workers : int
            Number of DataLoader workers (default: 4)
        batch_size : int
            Batch size
            
        Returns:
        --------
        scvi_model, scanvi_model : tuple
            Trained models
        """
        logger.info("Starting scVI model training...")
        
        # Optimize DataLoader settings
        import os
        if num_workers == "auto":
            # Auto-configure based on CPU cores (max 16)
            num_workers = min(os.cpu_count(), 16)
        elif num_workers is None:
            num_workers = 0  # Disable multiprocessing
            
        logger.info(f"DataLoader settings: num_workers={num_workers}, batch_size={batch_size}")
        
        # Settings for scvi-tools
        if num_workers > 0:
            # Set via environment variable (most reliable way)
            os.environ['SCVI_NUM_WORKERS'] = str(num_workers)
            
            # Set via scvi settings (if available)
            try:
                import scvi.settings as settings
                if hasattr(settings, 'num_threads'):
                    settings.num_threads = num_workers
                logger.info(f"scvi settings: num_workers={num_workers}")
            except (ImportError, AttributeError):
                logger.info("Skipping scvi.settings configuration")
        
        # scVI setup and training
        scvi.model.SCVI.setup_anndata(adata_combined,
                                      batch_key="dataset",
                                      categorical_covariate_keys=["batch"])
        scvi_model = scvi.model.SCVI(adata_combined,
                                     n_hidden=256,
                                     n_latent=20,
                                     n_layers=2,
                                     dropout_rate=0.2,
                                     dispersion="gene-batch",
                                     gene_likelihood="zinb",
                                     encode_covariates=True)
        
        # Set device and move model
        if self.device_str in ["cuda", "mps"]:
            scvi_model.module.to(self.device_str)
        
        # Set accelerator (scvi only supports "gpu"/"cpu")
        accelerator = "gpu" if self.device_str in ["cuda", "mps"] else "cpu"
        
        # Prepare training arguments
        train_kwargs = {
            "accelerator": accelerator,
            "max_epochs": max_epochs,
            "early_stopping": early_stopping,
            "check_val_every_n_epoch": 1,
            "train_size": 0.9,
            "validation_size": 0.1,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.00
        }
        
        # Set batch size (depends on scvi-tools version)
        try:
            # Set batch size in newer versions
            train_kwargs["batch_size"] = batch_size
        except Exception:
            logger.warning("Direct setting of batch_size parameter is not supported")
        
        # Try to set DataLoader configuration (can also be set via env var)
        if num_workers > 0:
            # Reliably set via environment variable
            logger.info(f"Setting environment variable SCVI_NUM_WORKERS to {num_workers}")
        
        scvi_model.train(**train_kwargs)
        
        # Process cell type annotations
        adata_combined = self._process_cell_type_labels(adata_combined, cell_type_key)

        logger.info("Starting scANVI model training...")
        
        # scANVI setup and training
        scvi.model.SCANVI.setup_anndata(
            adata_combined, 
            labels_key=cell_type_key, 
            unlabeled_category="Unknown",
            batch_key="dataset",  # SCVIと同じ
            categorical_covariate_keys=["batch"]
        )
        
        scanvi_model = scvi.model.SCANVI.from_scvi_model(
            scvi_model, 
            labels_key=cell_type_key, 
            unlabeled_category="Unknown"
        )
        
        # Set device and move model
        if self.device_str in ["cuda", "mps"]:
            scanvi_model.module.to(self.device_str)
        
        # scANVI training settings
        scanvi_train_kwargs = {
            "accelerator": accelerator,
            "max_epochs": max_epochs//5,  # scANVI usually trains for fewer epochs
            "early_stopping": early_stopping,
            "check_val_every_n_epoch": 10,
        }
        
        # Set batch size
        try:
            scanvi_train_kwargs["batch_size"] = batch_size
        except Exception:
            logger.warning("Direct setting of batch_size parameter is not supported")
            
        scanvi_model.train(**scanvi_train_kwargs)
        
        logger.info("Model training complete")
        return scvi_model, scanvi_model
    
    def predict_labels(self, 
                     adata_combined: sc.AnnData,
                     scanvi_model: Any,
                     cell_type_key: str = "cell_type_annotation") -> sc.AnnData:
        """Execute label prediction"""
        
        logger.info("Executing label prediction...")
        
        # Run prediction
        predicted_labels = scanvi_model.predict(adata_combined)
        adata_combined.obs["scvi_predicted_labels"] = predicted_labels
        
        # Get prediction probabilities as well (optional)
        predictions = scanvi_model.predict(adata_combined, soft=True)
        prediction_df = pd.DataFrame(
            predictions, 
            index=adata_combined.obs.index,
            columns=scanvi_model.adata.obs[cell_type_key].cat.categories
        )
        
        # Save max probability as confidence
        adata_combined.obs["prediction_confidence"] = prediction_df.max(axis=1)
        
        logger.info("Prediction complete")
        return adata_combined
    
    def transfer_labels_to_spatial(self, 
                                 adata_combined: sc.AnnData,
                                 sp_adata: sc.AnnData,
                                 annotation_dict: Optional[Dict] = None) -> sc.AnnData:
        """Transfer predicted labels to spatial data"""
        
        logger.info("Transferring labels to spatial data...")
        
        # Get prediction results for Unknown labels
        unknown_mask = adata_combined.obs['cell_type_annotation'] == "Unknown"
        label_series = adata_combined.obs.loc[unknown_mask, 'scvi_predicted_labels']
        confidence_series = adata_combined.obs.loc[unknown_mask, 'prediction_confidence']
        
        # Process index (remove -1 suffix)
        base_index = label_series.index.str.replace(r'-1$', '', regex=True)
        label_series.index = base_index
        confidence_series.index = base_index
        
        # Find common indices
        common_idx = sp_adata.obs.index.intersection(label_series.index)
        
        # Handle categorical columns - add new categories if needed
        if 'scvi_predicted_labels' in sp_adata.obs.columns:
            if hasattr(sp_adata.obs['scvi_predicted_labels'], 'cat'):
                # Get unique values from prediction
                new_categories = set(label_series.loc[common_idx].unique())
                existing_categories = set(sp_adata.obs['scvi_predicted_labels'].cat.categories)
                missing_categories = new_categories - existing_categories
                
                if missing_categories:
                    sp_adata.obs['scvi_predicted_labels'] = sp_adata.obs['scvi_predicted_labels'].cat.add_categories(list(missing_categories))
        else:
            # Create new column as object type first
            sp_adata.obs['scvi_predicted_labels'] = pd.Series(index=sp_adata.obs.index, dtype=object)
        
        # Similar handling for confidence column
        if 'prediction_confidence' not in sp_adata.obs.columns:
            sp_adata.obs['prediction_confidence'] = pd.Series(index=sp_adata.obs.index, dtype=float)
        
        # Transfer to spatial data
        sp_adata.obs.loc[common_idx, 'scvi_predicted_labels'] = label_series.loc[common_idx]
        sp_adata.obs.loc[common_idx, 'prediction_confidence'] = confidence_series.loc[common_idx]
        
        # Map with annotation dictionary (if provided)
        if annotation_dict:
            # Create predicted_cell_type column
            predicted_mapped = sp_adata.obs['scvi_predicted_labels'].map(annotation_dict)
            sp_adata.obs['predicted_cell_type'] = predicted_mapped
        else:
            sp_adata.obs['predicted_cell_type'] = sp_adata.obs['scvi_predicted_labels'].copy()
        
        # Convert to categorical
        sp_adata.obs['predicted_cell_type'] = sp_adata.obs['predicted_cell_type'].astype('category')
        
        logger.info(f"Transfer complete: {len(common_idx)} cells")
        return sp_adata

def analyze_predictions(sp_adata: sc.AnnData, 
                        cluster_key: str = "leiden_nucleus",
                        prediction_key: str = "predicted_cell_type") -> None:
    """Analysis and visualization of prediction results"""
    
    # Pre-process data
    sp_adata_predicted = sp_adata.copy()
    sc.pp.normalize_total(sp_adata_predicted, target_sum=1e4)
    sc.pp.log1p(sp_adata_predicted)
    
    # Dendrogram and differential gene expression analysis
    sc.tl.dendrogram(sp_adata_predicted, groupby=prediction_key)
    sc.tl.rank_genes_groups(sp_adata_predicted, prediction_key, method='wilcoxon', use_raw=False)
    
    # Display heatmap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pl.rank_genes_groups_heatmap(sp_adata_predicted, show_gene_labels=True, use_raw=False)
    
    # Create and display confusion matrix
    create_confusion_matrix(sp_adata, cluster_key, prediction_key)
    
    return sp_adata_predicted

def create_confusion_matrix(sp_adata: sc.AnnData, 
                            cluster_key: str,
                            prediction_key: str) -> None:
    """Create and visualize confusion matrix"""
    
    conf_matrix = pd.crosstab(sp_adata.obs[cluster_key], sp_adata.obs[prediction_key])
    conf_matrix_pct = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix_pct, annot=True, fmt=".2f", cmap="viridis", linewidths=0,
            rasterized=True, cbar=True, cbar_kws={'label': 'Proportion'})
    plt.grid(False)
    plt.xlabel("Predicted Cell Type")
    plt.ylabel("Leiden Cluster")
    plt.title("Confusion Matrix (Row-normalized)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def create_spatial_plot(sp_adata: sc.AnnData,
                        lib_id: str,
                        sample_name: str,
                        save_path: str,
                        color_key: str = "predicted_cell_type") -> None:
    """Improved spatial plot creation"""
    
    # Basic spatial plot
    sc.pl.spatial(sp_adata, color=color_key,
                  title='Cell Type Annotation (scANVI)', 
                  size=20, img_key='hires', legend_fontsize=8,
                  spot_size=1, frameon=False)
    
    # High-resolution overlay plot
    create_hires_overlay_plot(sp_adata, lib_id, sample_name, save_path, color_key)

def create_hires_overlay_plot(sp_adata: sc.AnnData,
                              lib_id: str, 
                              sample_name: str,
                              save_path: str,
                              color_key: str = "predicted_cell_type") -> None:
    """Create high-resolution overlay plot"""
    
    # Coordinate calculation
    sf_hires = sp_adata.uns["spatial"][lib_id]["scalefactors"]["tissue_hires_scalef"]
    xy = (pd.DataFrame(sp_adata.obsm["spatial"] * sf_hires, 
                       columns=["x", "y"], 
                       index=sp_adata.obs_names)
          .join(sp_adata.obs["object_id"])
          .reset_index()
          .rename(columns={"index": "cell_id"}))
    
    merged = xy.merge(sp_adata.obs, on="object_id", how="inner")
    merged["group"] = merged[color_key].astype(str).str.strip()
    
    # Set colors and markers
    group_order = sorted(merged["group"].dropna().unique())
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X', 'P', 'H', '8', 'd', '|']
    
    # Select color palette (based on number of groups)
    if len(group_order) <= 10:
        palette = sns.color_palette("tab10", n_colors=len(group_order))
    elif len(group_order) <= 20:
        palette = sns.color_palette("tab20", n_colors=len(group_order))
    else:
        palette = sns.color_palette("husl", n_colors=len(group_order))
    
    color_map = dict(zip(group_order, palette))
    marker_cycle = cycle(markers)
    marker_map = {group: next(marker_cycle) for group in group_order}
    
    # Create plot
    hires_img = sp_adata.uns["spatial"][lib_id]["images"]["hires"]
    h, w = hires_img.shape[:2]
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.imshow(hires_img, extent=[0, w, h, 0])
    
    # Plot each group individually
    for group in group_order:
        data_sub = merged[merged["group"] == group]
        ax.scatter(data_sub["x"], data_sub["y"],
                   c=[color_map[group]], marker=marker_map[group],
                   s=1.0, alpha=0.7, label=group,
                   linewidths=0, rasterized=True)
    
    ax.invert_yaxis()
    ax.set_axis_off()
    
    # Improve legend
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                       title="Cell Type", markerscale=10, 
                       frameon=True, fancybox=True, shadow=True,
                       fontsize=8, title_fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Save
    filename = f"{sample_name}_spatial_overlay_scANVI.pdf"
    out_pdf = os.path.join(save_path, filename)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"High-resolution plot saved: {out_pdf}")

# Main execution example
def run_scvi_label_transfer(filtered_sc_adata: sc.AnnData,
                            sp_adata: sc.AnnData,
                            annotation_dict: Dict,
                            lib_id: str,
                            sample_name: str,
                            save_path_for_today: str,
                            h5ad_predicted_full_save_path: str,
                            device = "auto",
                            num_workers: int = "auto",
                            batch_size: int = 128,
                            max_epochs: int = 400) -> sc.AnnData:
    """
    Complete scANVI cell type transfer pipeline
    
    Parameters:
    -----------
    filtered_sc_adata : AnnData
        Filtered reference data
    sp_adata : AnnData
        Spatial data
    annotation_dict : dict
        Cell type annotation dictionary
    lib_id : str
        Library ID
    sample_name : str
        Sample name
    save_path_for_today : str
        Save path
    h5ad_predicted_full_save_path : str
        Path for the predicted H5AD file
    device : str or torch.device
        Computational device
    num_workers : int or "auto"
        Number of DataLoader workers ("auto" for automatic setting, recommended 4-16)
    batch_size : int
        Batch size (default: 128)
    max_epochs : int
        Maximum number of epochs (default: 400)
        
    Returns:
    --------
    sp_adata_predicted : AnnData
        Spatial data with prediction results
    """
    
    # Initialize scANVI label transfer class
    label_transfer = SCVILabelTransfer(device=device)
    
    try:
        # 1. Prepare data
        adata_combined = label_transfer.prepare_data(filtered_sc_adata, sp_adata)
        
        # 2. Train models
        scvi_model, scanvi_model = label_transfer.train_scvi_scanvi(
            adata_combined, 
            num_workers=num_workers,
            batch_size=batch_size,
            max_epochs=max_epochs
        )
        
        # 3. Run prediction
        adata_combined = label_transfer.predict_labels(adata_combined, scanvi_model)
        
        # 4. Transfer to spatial data
        sp_adata = label_transfer.transfer_labels_to_spatial(
            adata_combined, sp_adata, annotation_dict
        )
        
        # 5. Analyze results
        sp_adata_predicted = analyze_predictions(sp_adata)
        
        # 6. Visualize
        create_spatial_plot(sp_adata, lib_id, sample_name, save_path_for_today)
        
        # 7. Save results
        sp_adata_predicted.write_h5ad(h5ad_predicted_full_save_path)
        logger.info(f"Results saved: {h5ad_predicted_full_save_path}")
        
        # 8. Quality assessment report
        generate_quality_report(sp_adata_predicted, save_path_for_today, sample_name)
        
        return sp_adata_predicted
        
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        raise

def generate_quality_report(sp_adata: sc.AnnData, 
                            save_path: str, 
                            sample_name: str) -> None:
    """Generate a prediction quality report"""
    
    report_path = os.path.join(save_path, f"{sample_name}_quality_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== scANVI Cell Type Prediction Quality Report ===\n\n")
        
        # Basic statistics
        f.write("1. Basic Statistics\n")
        f.write(f"   - Total cells: {sp_adata.n_obs:,}\n")
        f.write(f"   - Number of predicted cell types: {sp_adata.obs['predicted_cell_type'].nunique()}\n")
        
        # Prediction confidence statistics
        if 'prediction_confidence' in sp_adata.obs.columns:
            conf_stats = sp_adata.obs['prediction_confidence'].describe()
            f.write(f"\n2. Prediction Confidence Statistics\n")
            f.write(f"   - Mean: {conf_stats['mean']:.3f}\n")
            f.write(f"   - Median: {conf_stats['50%']:.3f}\n")
            f.write(f"   - Minimum: {conf_stats['min']:.3f}\n")
            f.write(f"   - Maximum: {conf_stats['max']:.3f}\n")
            
            # Ratio of low-confidence cells
            low_conf_ratio = (sp_adata.obs['prediction_confidence'] < 0.5).sum() / sp_adata.n_obs
            f.write(f"   - Ratio of low-confidence cells (<0.5): {low_conf_ratio:.1%}\n")
        
        # Cell type-specific statistics
        f.write(f"\n3. Cell Type Distribution\n")
        type_counts = sp_adata.obs['predicted_cell_type'].value_counts()
        for cell_type, count in type_counts.items():
            ratio = count / sp_adata.n_obs
            f.write(f"   - {cell_type}: {count:,} ({ratio:.1%})\n")
    
    logger.info(f"Quality report saved: {report_path}")