import matplotlib.pyplot as plt
import numpy as np

def plot_all_clusters_highlights(analyzer):
    """全Leidenクラスタのハイライトプロット"""
    
    # クラスタIDを取得
    cluster_ids = sorted(analyzer.adata.obs['leiden'].astype(str).unique())
    print(f"クラスタ数: {len(cluster_ids)}")
    
    # プロットの配置を計算
    num_clusters = len(cluster_ids)
    cols_per_row = 4
    rows = int(np.ceil(num_clusters / cols_per_row))
    
    # フィギュアを作成
    fig, axes = plt.subplots(rows, cols_per_row, 
                            figsize=(4 * cols_per_row, 4 * rows))
    
    # 1行の場合の処理
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # UMAP座標を取得
    umap_coords = analyzer.adata.obsm['X_umap']
    
    # 各クラスタについてプロット
    for i, cluster_id in enumerate(cluster_ids):
        ax = axes[i]
        
        # クラスタマスクを作成
        is_target_cluster = (analyzer.adata.obs['leiden'].astype(str) == cluster_id)
        target_count = is_target_cluster.sum()
        
        # 背景のセル（グレー）
        background_coords = umap_coords[~is_target_cluster]
        if len(background_coords) > 0:
            ax.scatter(background_coords[:, 0], background_coords[:, 1], 
                      c='lightgrey', s=0.5, alpha=0.3, rasterized=True)
        
        # ターゲットクラスタ（赤）
        target_coords = umap_coords[is_target_cluster]
        if len(target_coords) > 0:
            ax.scatter(target_coords[:, 0], target_coords[:, 1], 
                      c='red', s=0.5, alpha=0.5, rasterized=True)
        
        # タイトルとラベル
        ax.set_title(f'Cluster {cluster_id}\n(n={target_count})', fontsize=12)
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        
        # 軸の範囲を設定
        ax.set_xlim(umap_coords[:, 0].min() - 1, umap_coords[:, 0].max() + 1)
        ax.set_ylim(umap_coords[:, 1].min() - 1, umap_coords[:, 1].max() + 1)
        
        # グリッドを追加
        ax.grid(True, alpha=0.2)
        
        # 軸のラベルサイズを調整
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    # 空のサブプロットを削除
    for j in range(len(cluster_ids), len(axes)):
        fig.delaxes(axes[j])
    
    # レイアウト調整
    plt.tight_layout()
    plt.suptitle('Leiden Clusters Highlighted', y=1.02, fontsize=16)
    plt.show()
    
    return fig


def plot_all_cell_type_highlights(analyzer):
    """全cell_typeクラスタのハイライトプロット"""
    
    # クラスタIDを取得
    cluster_ids = sorted(analyzer.adata.obs['cell_type'].astype(str).unique())
    print(f"Cell type数: {len(cluster_ids)}")
    
    # プロットの配置を計算
    num_clusters = len(cluster_ids)
    cols_per_row = 4
    rows = int(np.ceil(num_clusters / cols_per_row))
    
    # フィギュアを作成
    fig, axes = plt.subplots(rows, cols_per_row, 
                            figsize=(4 * cols_per_row, 4 * rows))
    
    # 1行の場合の処理
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # UMAP座標を取得
    umap_coords = analyzer.adata.obsm['X_umap']
    
    # 各クラスタについてプロット
    for i, cluster_id in enumerate(cluster_ids):
        ax = axes[i]
        
        # クラスタマスクを作成
        is_target_cluster = (analyzer.adata.obs['cell_type'].astype(str) == cluster_id)
        target_count = is_target_cluster.sum()
        
        # 背景のセル（グレー）
        background_coords = umap_coords[~is_target_cluster]
        if len(background_coords) > 0:
            ax.scatter(background_coords[:, 0], background_coords[:, 1], 
                      c='lightgrey', s=0.5, alpha=0.3, rasterized=True)
        
        # ターゲットクラスタ（赤）
        target_coords = umap_coords[is_target_cluster]
        if len(target_coords) > 0:
            ax.scatter(target_coords[:, 0], target_coords[:, 1], 
                      c='red', s=0.5, alpha=0.5, rasterized=True)
        
        # タイトルとラベル
        ax.set_title(f'{cluster_id}\n(n={target_count})', fontsize=12)
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        
        # 軸の範囲を設定
        ax.set_xlim(umap_coords[:, 0].min() - 1, umap_coords[:, 0].max() + 1)
        ax.set_ylim(umap_coords[:, 1].min() - 1, umap_coords[:, 1].max() + 1)
        
        # グリッドを追加
        ax.grid(True, alpha=0.2)
        
        # 軸のラベルサイズを調整
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    # 空のサブプロットを削除
    for j in range(len(cluster_ids), len(axes)):
        fig.delaxes(axes[j])
    
    # レイアウト調整
    plt.tight_layout()
    plt.suptitle('Cell Type Highlighted', y=1.02, fontsize=16)
    plt.show()
    
    return fig
