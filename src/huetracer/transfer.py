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

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SCVILabelTransfer:
    """scANVIを使用した細胞タイプ転送のクラス"""
    
    def __init__(self, device = "auto"):
        """
        Parameters:
        -----------
        device : str or torch.device
            計算デバイス ("auto", "cpu", "cuda", "mps", またはtorch.deviceオブジェクト)
        """
        self.device_str = self._setup_device(device)
        self.device = self.device_str  # 後方互換性のため
        self._configure_pytorch_settings()
        logger.info(f"Using device: {self.device_str}")
        
    def _configure_pytorch_settings(self):
        """PyTorchとscvi-toolsの基本設定"""
        try:
            import torch
            # デフォルトのnum_workersを設定
            if hasattr(torch.utils.data, '_utils'):
                # PyTorchのDataLoaderの警告を抑制
                import warnings
                warnings.filterwarnings("ignore", ".*does not have many workers.*")
        except ImportError:
            pass
        
    def _setup_device(self, device) -> str:
        """デバイス設定 - torch.deviceオブジェクトまたは文字列に対応"""
        try:
            import torch
            
            # torch.deviceオブジェクトの場合
            if isinstance(device, torch.device):
                return str(device.type)
            
            # 文字列の場合
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
            logger.warning("PyTorchが見つかりません。CPUを使用します。")
            return "cpu"
    
    def prepare_data(self, 
                    sc_adata: sc.AnnData, 
                    sp_adata: sc.AnnData,
                    cell_type_key: str = "cell_type_annotation") -> sc.AnnData:
        """
        参照データとクエリデータを結合し、scANVI用に準備
        
        Parameters:
        -----------
        sc_adata : AnnData
            参照シングルセルデータ
        sp_adata : AnnData  
            クエリ空間データ
        cell_type_key : str
            細胞タイプアノテーションのキー
            
        Returns:
        --------
        adata_combined : AnnData
            結合されたデータ
        """
        logger.info("データの準備を開始...")
        
        # 参照データの準備
        adata_ref = sc_adata.copy()
        adata_ref.obs['batch'] = 'reference'
        if adata_ref.raw is not None:
            adata_ref.X = adata_ref.raw.X.copy()
            adata_ref.var = adata_ref.raw.var.copy()
            adata_ref.raw = None
            
        # クエリデータの準備  
        adata_query = sp_adata.copy()
        adata_query.obs['batch'] = 'query'
        if adata_query.raw is not None:
            adata_query.X = adata_query.raw.X.copy()
            adata_query.var = adata_query.raw.var.copy()
            adata_query.raw = None
            
        # データ結合
        adata_combined = adata_ref.concatenate(adata_query, batch_key="dataset")
        
        # 細胞タイプアノテーションの処理
        adata_combined = self._process_cell_type_labels(adata_combined, cell_type_key)
        
        logger.info(f"結合データサイズ: {adata_combined.shape}")
        return adata_combined
    
    def _process_cell_type_labels(self, 
                                 adata: sc.AnnData, 
                                 cell_type_key: str) -> sc.AnnData:
        """細胞タイプラベルの前処理"""
        
        if cell_type_key not in adata.obs.columns:
            raise ValueError(f"Key '{cell_type_key}' not found in adata.obs")
            
        cat_col = adata.obs[cell_type_key].copy()
        
        # カテゴリ型の処理
        if isinstance(cat_col.dtype, pd.CategoricalDtype):
            if "Unknown" not in cat_col.cat.categories:
                cat_col = cat_col.cat.add_categories(["Unknown"])
        
        # 欠損値をUnknownに置換
        cat_col = cat_col.fillna("Unknown")
        
        # カテゴリの再設定
        categories = cat_col.unique().tolist()
        if "Unknown" not in categories:
            categories.append("Unknown")
            
        adata.obs[cell_type_key] = pd.Categorical(cat_col, categories=categories)
        
        logger.info(f"細胞タイプカテゴリ数: {len(categories)}")
        return adata
    
    def train_scvi_scanvi(self, 
                         adata_combined: sc.AnnData,
                         cell_type_key: str = "cell_type_annotation",
                         max_epochs: int = 400,
                         early_stopping: bool = True,
                         num_workers: int = 4,
                         batch_size: int = 128) -> Tuple[Any, Any]:
        """
        scVI/scANVIモデルの訓練
        
        Parameters:
        -----------
        adata_combined : AnnData
            結合データ
        cell_type_key : str
            細胞タイプキー
        max_epochs : int
            最大エポック数
        early_stopping : bool
            早期停止を使用するか
        num_workers : int
            DataLoaderのワーカー数（デフォルト: 4）
        batch_size : int
            バッチサイズ
            
        Returns:
        --------
        scvi_model, scanvi_model : tuple
            訓練済みモデル
        """
        logger.info("scVIモデルの訓練を開始...")
        
        # DataLoaderの設定を最適化
        import os
        if num_workers == "auto":
            # CPUコア数に基づいて自動設定（最大16）
            num_workers = min(os.cpu_count(), 16)
        elif num_workers is None:
            num_workers = 0  # マルチプロセシングを無効
            
        logger.info(f"DataLoader設定: num_workers={num_workers}, batch_size={batch_size}")
        
        # scvi-tools用の設定
        if num_workers > 0:
            # 環境変数で設定（最も確実な方法）
            os.environ['SCVI_NUM_WORKERS'] = str(num_workers)
            
            # scvi settingsでの設定（利用可能な場合）
            try:
                import scvi.settings as settings
                if hasattr(settings, 'num_threads'):
                    settings.num_threads = num_workers
                logger.info(f"scvi設定: num_workers={num_workers}")
            except (ImportError, AttributeError):
                logger.info("scvi.settingsでの設定はスキップ")
        
        # PyTorchのDataLoaderデフォルト設定
        try:
            import torch
            torch.utils.data.DataLoader.__init__.__defaults__ = (
                None, 1, False, None, None, None, None, num_workers, False, False, None, 0.0, None, None, None, None
            )
        except Exception:
            logger.info("PyTorch DataLoaderのデフォルト設定変更はスキップ")
        
        # scVIセットアップと訓練
        scvi.model.SCVI.setup_anndata(adata_combined, batch_key="dataset")
        scvi_model = scvi.model.SCVI(adata_combined)
        
        # デバイスの設定とモデル移動
        if self.device_str in ["cuda", "mps"]:
            scvi_model.module.to(self.device_str)
        
        # acceleratorの設定 (scviのacceleratorは"gpu"/"cpu"のみサポート)
        accelerator = "gpu" if self.device_str in ["cuda", "mps"] else "cpu"
        
        # 訓練設定の準備
        train_kwargs = {
            "accelerator": accelerator,
            "max_epochs": max_epochs,
            "early_stopping": early_stopping,
            "check_val_every_n_epoch": 10,
        }
        
        # バッチサイズの設定（scvi-toolsバージョンによる）
        try:
            # 新しいバージョンでのバッチサイズ設定
            train_kwargs["batch_size"] = batch_size
        except Exception:
            logger.warning("batch_sizeパラメータの直接設定はサポートされていません")
        
        # DataLoaderの設定を試行（環境変数での設定も可能）
        if num_workers > 0:
            # 確実に環境変数で設定
            logger.info(f"環境変数SCVI_NUM_WORKERSを{num_workers}に設定")
        
        scvi_model.train(**train_kwargs)
        
        logger.info("scANVIモデルの訓練を開始...")
        
        # scANVIセットアップと訓練
        scvi.model.SCANVI.setup_anndata(
            adata_combined, 
            labels_key=cell_type_key, 
            unlabeled_category="Unknown"
        )
        
        scanvi_model = scvi.model.SCANVI.from_scvi_model(
            scvi_model, 
            labels_key=cell_type_key, 
            unlabeled_category="Unknown"
        )
        
        # デバイスの設定とモデル移動
        if self.device_str in ["cuda", "mps"]:
            scanvi_model.module.to(self.device_str)
        
        # scANVIの訓練設定
        scanvi_train_kwargs = {
            "accelerator": accelerator,
            "max_epochs": max_epochs//2,  # scANVIは通常短めに訓練
            "early_stopping": early_stopping,
            "check_val_every_n_epoch": 10,
        }
        
        # バッチサイズの設定
        try:
            scanvi_train_kwargs["batch_size"] = batch_size
        except Exception:
            logger.warning("batch_sizeパラメータの直接設定はサポートされていません")
            
        scanvi_model.train(**scanvi_train_kwargs)
        
        logger.info("モデル訓練完了")
        return scvi_model, scanvi_model
    
    def predict_labels(self, 
                      adata_combined: sc.AnnData,
                      scanvi_model: Any,
                      cell_type_key: str = "cell_type_annotation") -> sc.AnnData:
        """ラベル予測の実行"""
        
        logger.info("ラベル予測を実行...")
        
        # 予測実行
        predicted_labels = scanvi_model.predict(adata_combined)
        adata_combined.obs["scvi_predicted_labels"] = predicted_labels
        
        # 予測確率も取得（オプション）
        predictions = scanvi_model.predict(adata_combined, soft=True)
        prediction_df = pd.DataFrame(
            predictions, 
            index=adata_combined.obs.index,
            columns=scanvi_model.adata.obs[cell_type_key].cat.categories
        )
        
        # 最大確率を信頼度として保存
        adata_combined.obs["prediction_confidence"] = prediction_df.max(axis=1)
        
        logger.info("予測完了")
        return adata_combined
    
    def transfer_labels_to_spatial(self, 
                                  adata_combined: sc.AnnData,
                                  sp_adata: sc.AnnData,
                                  annotation_dict: Optional[Dict] = None) -> sc.AnnData:
        """空間データに予測ラベルを転送"""
        
        logger.info("空間データへのラベル転送...")
        
        # Unknownラベルの予測結果を取得
        unknown_mask = adata_combined.obs['cell_type_annotation'] == "Unknown"
        label_series = adata_combined.obs.loc[unknown_mask, 'scvi_predicted_labels']
        confidence_series = adata_combined.obs.loc[unknown_mask, 'prediction_confidence']
        
        # インデックス処理（-1サフィックス除去）
        base_index = label_series.index.str.replace(r'-1$', '', regex=True)
        label_series.index = base_index
        confidence_series.index = base_index
        
        # 空間データに転送
        common_idx = sp_adata.obs.index.intersection(label_series.index)
        sp_adata.obs.loc[common_idx, 'scvi_predicted_labels'] = label_series.loc[common_idx]
        sp_adata.obs.loc[common_idx, 'prediction_confidence'] = confidence_series.loc[common_idx]
        
        # アノテーション辞書でマッピング（提供されている場合）
        if annotation_dict:
            sp_adata.obs['predicted_cell_type'] = sp_adata.obs['scvi_predicted_labels'].map(annotation_dict)
        else:
            sp_adata.obs['predicted_cell_type'] = sp_adata.obs['scvi_predicted_labels']
            
        sp_adata.obs['predicted_cell_type'] = sp_adata.obs['predicted_cell_type'].astype('category')
        
        logger.info(f"転送完了: {len(common_idx)}個の細胞")
        return sp_adata

def analyze_predictions(sp_adata: sc.AnnData, 
                       cluster_key: str = "leiden_nucleus",
                       prediction_key: str = "predicted_cell_type") -> None:
    """予測結果の解析と可視化"""
    
    # データの前処理
    sp_adata_predicted = sp_adata.copy()
    sc.pp.normalize_total(sp_adata_predicted, target_sum=1e4)
    sc.pp.log1p(sp_adata_predicted)
    
    # 樹状図と差次元発現解析
    sc.tl.dendrogram(sp_adata_predicted, groupby=prediction_key)
    sc.tl.rank_genes_groups(sp_adata_predicted, prediction_key, method='wilcoxon', use_raw=False)
    
    # ヒートマップ表示
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pl.rank_genes_groups_heatmap(sp_adata_predicted, show_gene_labels=True, use_raw=False)
    
    # 混同行列の作成と表示
    create_confusion_matrix(sp_adata, cluster_key, prediction_key)
    
    return sp_adata_predicted

def create_confusion_matrix(sp_adata: sc.AnnData, 
                           cluster_key: str,
                           prediction_key: str) -> None:
    """混同行列の作成と可視化"""
    
    conf_matrix = pd.crosstab(sp_adata.obs[cluster_key], sp_adata.obs[prediction_key])
    conf_matrix_pct = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix_pct, annot=True, fmt=".2f", cmap="viridis", 
                cbar=True, cbar_kws={'label': 'Proportion'})
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
    """改善された空間プロット作成"""
    
    # 基本的な空間プロット
    sc.pl.spatial(sp_adata, color=color_key,
                  title='Cell Type Annotation (scANVI)', 
                  size=20, img_key='hires', legend_fontsize=8,
                  spot_size=1, frameon=False)
    
    # 高解像度オーバーレイプロット
    create_hires_overlay_plot(sp_adata, lib_id, sample_name, save_path, color_key)

def create_hires_overlay_plot(sp_adata: sc.AnnData,
                             lib_id: str, 
                             sample_name: str,
                             save_path: str,
                             color_key: str = "predicted_cell_type") -> None:
    """高解像度オーバーレイプロットの作成"""
    
    # 座標計算
    sf_hires = sp_adata.uns["spatial"][lib_id]["scalefactors"]["tissue_hires_scalef"]
    xy = (pd.DataFrame(sp_adata.obsm["spatial"] * sf_hires, 
                      columns=["x", "y"], 
                      index=sp_adata.obs_names)
          .join(sp_adata.obs["object_id"])
          .reset_index()
          .rename(columns={"index": "cell_id"}))
    
    merged = xy.merge(sp_adata.obs, on="object_id", how="inner")
    merged["group"] = merged[color_key].astype(str).str.strip()
    
    # 色とマーカーの設定
    group_order = sorted(merged["group"].dropna().unique())
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X', 'P', 'H', '8', 'd', '|']
    
    # カラーパレットの選択（グループ数に応じて）
    if len(group_order) <= 10:
        palette = sns.color_palette("tab10", n_colors=len(group_order))
    elif len(group_order) <= 20:
        palette = sns.color_palette("tab20", n_colors=len(group_order))
    else:
        palette = sns.color_palette("husl", n_colors=len(group_order))
    
    color_map = dict(zip(group_order, palette))
    marker_cycle = cycle(markers)
    marker_map = {group: next(marker_cycle) for group in group_order}
    
    # プロット作成
    hires_img = sp_adata.uns["spatial"][lib_id]["images"]["hires"]
    h, w = hires_img.shape[:2]
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.imshow(hires_img, extent=[0, w, h, 0])
    
    # 各グループを個別にプロット
    for group in group_order:
        data_sub = merged[merged["group"] == group]
        ax.scatter(data_sub["x"], data_sub["y"],
                  c=[color_map[group]], marker=marker_map[group],
                  s=1.0, alpha=0.7, label=group,
                  linewidths=0, rasterized=True)
    
    ax.invert_yaxis()
    ax.set_axis_off()
    
    # 凡例の改善
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                      title="Cell Type", markerscale=10, 
                      frameon=True, fancybox=True, shadow=True,
                      fontsize=8, title_fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # 保存
    filename = f"{sample_name}_spatial_overlay_scANVI.pdf"
    out_pdf = os.path.join(save_path, filename)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"高解像度プロットを保存: {out_pdf}")

# メイン実行例
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
    完全なscANVI細胞タイプ転送パイプライン
    
    Parameters:
    -----------
    filtered_sc_adata : AnnData
        フィルタリング済み参照データ
    sp_adata : AnnData
        空間データ
    annotation_dict : dict
        細胞タイプアノテーション辞書
    lib_id : str
        ライブラリID
    sample_name : str
        サンプル名
    save_path_for_today : str
        保存パス
    h5ad_predicted_full_save_path : str
        予測結果H5ADファイルパス
    device : str or torch.device
        計算デバイス
    num_workers : int or "auto"
        DataLoaderのワーカー数（"auto"で自動設定、推奨値は4-16）
    batch_size : int
        バッチサイズ（デフォルト: 128）
    max_epochs : int
        最大エポック数（デフォルト: 400）
        
    Returns:
    --------
    sp_adata_predicted : AnnData
        予測結果付き空間データ
    """
    
    # scANVIラベル転送クラスの初期化
    label_transfer = SCVILabelTransfer(device=device)
    
    try:
        # 1. データ準備
        adata_combined = label_transfer.prepare_data(filtered_sc_adata, sp_adata)
        
        # 2. モデル訓練
        scvi_model, scanvi_model = label_transfer.train_scvi_scanvi(
            adata_combined, 
            num_workers=num_workers,
            batch_size=batch_size,
            max_epochs=max_epochs
        )
        
        # 3. 予測実行
        adata_combined = label_transfer.predict_labels(adata_combined, scanvi_model)
        
        # 4. 空間データに転送
        sp_adata = label_transfer.transfer_labels_to_spatial(
            adata_combined, sp_adata, annotation_dict
        )
        
        # 5. 結果解析
        sp_adata_predicted = analyze_predictions(sp_adata)
        
        # 6. 可視化
        create_spatial_plot(sp_adata, lib_id, sample_name, save_path_for_today)
        
        # 7. 結果保存
        sp_adata_predicted.write_h5ad(h5ad_predicted_full_save_path)
        logger.info(f"結果を保存: {h5ad_predicted_full_save_path}")
        
        # 8. 品質評価レポート
        generate_quality_report(sp_adata_predicted, save_path_for_today, sample_name)
        
        return sp_adata_predicted
        
    except Exception as e:
        logger.error(f"処理中にエラーが発生: {str(e)}")
        raise

def generate_quality_report(sp_adata: sc.AnnData, 
                           save_path: str, 
                           sample_name: str) -> None:
    """予測品質レポートの生成"""
    
    report_path = os.path.join(save_path, f"{sample_name}_quality_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== scANVI細胞タイプ予測品質レポート ===\n\n")
        
        # 基本統計
        f.write("1. 基本統計\n")
        f.write(f"   - 総細胞数: {sp_adata.n_obs:,}\n")
        f.write(f"   - 予測細胞タイプ数: {sp_adata.obs['predicted_cell_type'].nunique()}\n")
        
        # 予測信頼度統計
        if 'prediction_confidence' in sp_adata.obs.columns:
            conf_stats = sp_adata.obs['prediction_confidence'].describe()
            f.write(f"\n2. 予測信頼度統計\n")
            f.write(f"   - 平均: {conf_stats['mean']:.3f}\n")
            f.write(f"   - 中央値: {conf_stats['50%']:.3f}\n")
            f.write(f"   - 最小値: {conf_stats['min']:.3f}\n")
            f.write(f"   - 最大値: {conf_stats['max']:.3f}\n")
            
            # 低信頼度細胞の割合
            low_conf_ratio = (sp_adata.obs['prediction_confidence'] < 0.5).sum() / sp_adata.n_obs
            f.write(f"   - 低信頼度細胞割合 (<0.5): {low_conf_ratio:.1%}\n")
        
        # 細胞タイプ別統計
        f.write(f"\n3. 細胞タイプ別分布\n")
        type_counts = sp_adata.obs['predicted_cell_type'].value_counts()
        for cell_type, count in type_counts.items():
            ratio = count / sp_adata.n_obs
            f.write(f"   - {cell_type}: {count:,} ({ratio:.1%})\n")
    
    logger.info(f"品質レポートを保存: {report_path}")
    