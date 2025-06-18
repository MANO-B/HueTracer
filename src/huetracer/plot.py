import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import os
import plotly.graph_objects as go

def plot_gene_cci_and_sankey(target_cell_type, Gene_to_analyze, each_display_num,
                              bargraph_df, edge_df, cluster_cells, coexp_cc_df,
                              lib_id, role="receiver", save=False,
                              SAMPLE_NAME=None, save_path_for_today=None):

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import plotly.graph_objects as go

    # --- 全細胞タイプでの棒グラフ ---
    bargraph_df["cell1"] = edge_df["cell1"].values
    gene_counts = bargraph_df.groupby("cell1")[Gene_to_analyze].sum()
    result_series = pd.Series(0, index=cluster_cells.obs_names, dtype=int)
    result_series.loc[gene_counts.index] = gene_counts.astype(int)
    cluster_cells.obs['Gene_CCI'] = result_series
    mean_gene_cci = cluster_cells.obs.groupby('cluster')['Gene_CCI'].mean()

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    mean_gene_cci.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Average ' + Gene_to_analyze + ' CCI')
    ax1.set_title('Mean ' + Gene_to_analyze + '-activated cells per TME cluster (all cell types)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        filename = f"{SAMPLE_NAME}_{Gene_to_analyze}-activated_{target_cell_type}_barplot_all_clusters.pdf"
        out_pdf = os.path.join(save_path_for_today, filename)
        fig1.savefig(out_pdf, format="pdf", dpi=100, bbox_inches="tight")
    plt.close(fig1)

    # --- 対象細胞タイプでの棒グラフ ---
    giant_mask = cluster_cells.obs["celltype"] == target_cell_type
    giant_indices = giant_mask[giant_mask].index
    filtered_df = bargraph_df[bargraph_df['cell1_type'] == target_cell_type]
    gene_counts = filtered_df.groupby("cell1")[Gene_to_analyze].sum()
    result_series = pd.Series(0, index=giant_indices, dtype=int)
    result_series.loc[gene_counts.index] = gene_counts.astype(int)
    cluster_cells.obs.loc[giant_mask, 'Gene_CCI'] = result_series
    target_adata = cluster_cells[giant_mask].copy()
    sum_gene_cci = target_adata.obs.groupby('cluster')['Gene_CCI'].sum()
    cluster_counts = target_adata.obs['cluster'].value_counts().sort_index()
    mean_gene_cci_per_cell = sum_gene_cci / cluster_counts

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    mean_gene_cci_per_cell.plot(kind='bar', color='skyblue', ax=ax2)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Average ' + Gene_to_analyze + ' CCI')
    ax2.set_title('Mean ' + Gene_to_analyze + '-activation rate of ' + target_cell_type + ' per TME cluster')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        filename = f"{SAMPLE_NAME}_{Gene_to_analyze}-activated_{target_cell_type}_barplot_target_celltype.pdf"
        out_pdf = os.path.join(save_path_for_today, filename)
        fig2.savefig(out_pdf, format="pdf", dpi=100, bbox_inches="tight")
    plt.close(fig2)

    # --- 空間図の描画 ---
    hires_img = cluster_cells.uns["spatial"][lib_id]["images"]["hires"]
    h, w = hires_img.shape[:2]
    scale = cluster_cells.uns["spatial"][lib_id]["scalefactors"]["tissue_hires_scalef"]
    coords = cluster_cells.obsm["spatial_cropped_150_buffer"].copy()
    fig3 = plt.figure(figsize=(6, 6), dpi=100)
    ax3 = fig3.add_axes([0, 0, 1, 1])
    ax3.imshow(hires_img, extent=[0, w, h, 0], alpha=0.2)
    ax3.set_xlim(0, w)
    ax3.set_ylim(h, 0)
    ax3.axis('off')
    gene_cci_values = cluster_cells.obs["Gene_CCI"].copy()
    gene_cci_values[cluster_cells.obs['celltype'] != target_cell_type] = 0
    alphas = gene_cci_values.copy()
    alphas[gene_cci_values == 0] = 0
    alphas[gene_cci_values != 0] = 1
    coords[:, 0] = cluster_cells.obsm["spatial"][:, 0] * scale
    coords[:, 1] = cluster_cells.obsm["spatial"][:, 1] * scale
    scatter = ax3.scatter(
        coords[:, 0], coords[:, 1],
        c=gene_cci_values,
        cmap='jet',
        s=1,
        alpha=alphas,
        edgecolors='none'
    )
    ax3.set_title(Gene_to_analyze + '-activated ' + target_cell_type, fontsize=8)
    cb = fig3.colorbar(scatter, ax=ax3, shrink=0.4, aspect=40, pad=0.02)
    cb.set_label("CCI count", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    plt.tight_layout()
    if save:
        filename = f"{SAMPLE_NAME}_{Gene_to_analyze}-activated_{target_cell_type}_spatialmap.pdf"
        out_pdf = os.path.join(save_path_for_today, filename)
        fig3.savefig(out_pdf, format="pdf", dpi=1000, bbox_inches="tight")
    plt.close(fig3)

    # --- Sankey図の描画 ---
    sub_coexp_cc_df = coexp_cc_df.query(f"cell1_type == '{target_cell_type}'")
    sub_coexp_cc_df = sub_coexp_cc_df[sub_coexp_cc_df.is_significant]
    sub_coexp_cc_df = sub_coexp_cc_df.sort_values(
        'coactivity_per_sender_cell_expr_ligand', ascending=False
    ).groupby('cell2_type', as_index=False).head(n=each_display_num)

    cell1types = np.unique(sub_coexp_cc_df["cell1_type"])
    cell2types = np.unique(sub_coexp_cc_df["cell2_type"])
    tot_list = (
        list(sub_coexp_cc_df.ligand.unique()) +
        list(cell2types) +
        list(cell1types)
    )
    ligand_pos_dict = pd.Series({
        ligand: i for i, ligand in enumerate(sub_coexp_cc_df.ligand.unique())
    })
    celltype_pos_dict = pd.Series({
        celltype: i + len(ligand_pos_dict) for i, celltype in enumerate(cell2types)
    })
    receiver_dict = pd.Series({
        celltype: i + len(ligand_pos_dict) + len(cell2types)
        for i, celltype in enumerate(cell1types)
    })

    senders = (sub_coexp_cc_df.cell1_type.values
               if role == "sender" else sub_coexp_cc_df.cell2_type.values)
    receivers = (sub_coexp_cc_df.cell2_type.values
                 if role == "sender" else sub_coexp_cc_df.cell1_type.values)
    sources = pd.concat([
        ligand_pos_dict.loc[sub_coexp_cc_df.ligand.values],
        celltype_pos_dict.loc[senders]
    ])
    targets = pd.concat([
        receiver_dict.loc[receivers],
        ligand_pos_dict.loc[sub_coexp_cc_df.ligand.values]
    ])
    values = pd.concat([
        sub_coexp_cc_df['interaction_positive'],
        sub_coexp_cc_df['interaction_positive']
    ])
    labels = pd.concat([
        sub_coexp_cc_df['cell1_type'],
        sub_coexp_cc_df['cell2_type']
    ])
    unique_labels = labels.unique()
    palette = sns.color_palette("tab10", n_colors=len(unique_labels)).as_hex()
    target_color_dict = dict(zip(unique_labels, palette))
    colors = pd.Series(target_color_dict)[labels]
    fig4 = go.Figure(data=[go.Sankey(
        node=dict(label=tot_list),
        link=dict(source=sources, target=targets, value=values, color=colors, label=labels)
    )])
    fig4.update_layout(
        font_family="Courier New",
        width=600,
        height=1000,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    if save:
        filename = f"{SAMPLE_NAME}_{Gene_to_analyze}-activated_{target_cell_type}_sankey.pdf"
        out_pdf = os.path.join(save_path_for_today, filename)
        fig4.write_image(out_pdf, format="pdf", width=600, height=1000)

    fig4.show()
                                  
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
