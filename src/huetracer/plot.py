import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import anndata as ad
import os
import plotly.express as px
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display, clear_output
from PIL import Image
import base64
from io import BytesIO
from typing import Optional, Dict, List, Union, Tuple
from collections import defaultdict, Counter
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import warnings
import re
from itertools import cycle


def plot_gene_cci_and_sankey(target_cell_type, sender_cell_type, Gene_to_analyze, each_display_num,
                             bargraph_df, edge_df, cluster_cells, coexp_cc_df,
                             lib_id, role="receiver", save=False,
                             SAMPLE_NAME=None, save_path_for_today=None,
                             target_clusters=[0],
                             coexp_cc_df_cluster=None, bargraph_df_cluster=None,
                             display_column="interaction_positive",
                             significant_column='is_significant_bonferroni',
                             minimum_interaction=10,
                             save_format="html"):
    # Convert target_clusters to string if they're not already
    target_clusters_str = [str(c) for c in target_clusters]
    
    # Jupyter環境での表示設定
    plt.ion()  # インタラクティブモードを有効化
    
    # 必要な場合のみコピー作成
    if cluster_cells.is_view:
        cluster_cells = cluster_cells.copy()
    
    # データを配列として取得
    gene_col_data = bargraph_df[Gene_to_analyze].values
    cell1_data = edge_df["cell1"].values
    
    # pandasのgroupbyより高速なNumPy操作
    unique_cells, inverse_indices = np.unique(cell1_data, return_inverse=True)
    gene_counts_array = np.bincount(inverse_indices, weights=gene_col_data)
    gene_counts = pd.Series(gene_counts_array, index=unique_cells, name=Gene_to_analyze)
    
    # intersection計算の高速化（setを使用）
    cluster_obs_set = set(cluster_cells.obs_names)
    valid_mask = np.array([cell in cluster_obs_set for cell in gene_counts.index])
    valid_indices = gene_counts.index[valid_mask]
    gene_counts_filtered = gene_counts.iloc[valid_mask]
    
    print(f"Debug: Total gene_counts: {len(gene_counts)}, Valid: {len(gene_counts_filtered)}")
    
    # Series作成の高速化（辞書マッピング使用）
    result_array = np.zeros(len(cluster_cells.obs_names), dtype=int)
    obs_name_to_idx = {name: i for i, name in enumerate(cluster_cells.obs_names)}
    
    # vectorized assignment
    valid_idx_array = np.array([obs_name_to_idx[cell_id] for cell_id in valid_indices if cell_id in obs_name_to_idx])
    valid_counts = gene_counts_filtered.loc[[cell_id for cell_id in valid_indices if cell_id in obs_name_to_idx]].values
    result_array[valid_idx_array] = valid_counts.astype(int)
    
    cluster_cells.obs['Gene_CCI'] = result_array
    
    # groupby操作の高速化（NumPy使用）
    cluster_labels = cluster_cells.obs['cluster'].values
    gene_cci_values = cluster_cells.obs['Gene_CCI'].values
    
    # unique + boolean maskingで高速化
    unique_clusters = np.unique(cluster_labels)
    mean_gene_cci_list = []
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        mean_gene_cci_list.append(np.mean(gene_cci_values[mask]))
    
    mean_gene_cci = pd.Series(mean_gene_cci_list, index=unique_clusters)

    # --- Bar plot for all cell types ---
    print("Creating bar plot 1...")
    
    # Calculate proportion of cells that received ligand stimulation at least once per cluster
    stimulated_counts_list = []
    total_counts_list = []
    
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_gene_cci = gene_cci_values[cluster_mask]
        
        # Count cells that received stimulation (CCI > 0) at least once
        stimulated_count = np.sum(cluster_gene_cci > 0)
        total_count = np.sum(cluster_mask)
        
        stimulated_counts_list.append(stimulated_count)
        total_counts_list.append(total_count)
    
    stimulated_counts = pd.Series(stimulated_counts_list, index=unique_clusters)
    total_counts = pd.Series(total_counts_list, index=unique_clusters)
    
    # Calculate proportion (percentage)
    stimulation_proportion = np.divide(stimulated_counts.values, total_counts.values, 
                                     out=np.zeros_like(stimulated_counts.values, dtype=float), 
                                     where=total_counts.values!=0) * 100
    stimulation_proportion = pd.Series(stimulation_proportion, index=unique_clusters)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    stimulation_proportion.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('% of cells with ' + Gene_to_analyze + ' stimulation')
    ax1.set_title('Proportion of cells receiving ' + Gene_to_analyze + '-stimulation per TME cluster (all cell types)')
    #ax1.set_ylim(0, 100)  # Set y-axis to percentage scale
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 表示してから保存
    plt.show()
    
    if save:
        filename = f"{SAMPLE_NAME}_{Gene_to_analyze}-stimulated_{target_cell_type}_proportion_all_clusters.pdf"
        out_pdf = os.path.join(save_path_for_today, filename)
        fig1.savefig(out_pdf, format="pdf", dpi=100, bbox_inches="tight")
        print(f"Saved: {filename}")
    
    plt.close(fig1)

    # --- Bar plot for the target cell type ---
    print("Processing target cell type data...")
    
    # boolean indexingの高速化
    celltype_values = cluster_cells.obs["celltype"].values
    giant_mask = celltype_values == target_cell_type
    giant_indices = cluster_cells.obs_names[giant_mask]
    
    if not np.any(giant_mask):
        print(f"Warning: No cells found for target_cell_type '{target_cell_type}'")
        return
    
    # DataFrame filteringの高速化
    cell1_type_values = bargraph_df['cell1_type'].values
    target_mask = cell1_type_values == target_cell_type
    
    if not np.any(target_mask):
        print(f"Warning: No data found for target_cell_type '{target_cell_type}' in bargraph_df")
        return
    
    # NumPy配列で直接フィルタリング
    filtered_gene_data = gene_col_data[target_mask]
    filtered_cell1_data = cell1_data[target_mask]
    
    # 高速なgroupby代替（NumPy）
    unique_target_cells, inverse_indices = np.unique(filtered_cell1_data, return_inverse=True)
    target_gene_counts_array = np.bincount(inverse_indices, weights=filtered_gene_data)
    target_gene_counts = pd.Series(target_gene_counts_array, index=unique_target_cells)
    
    # intersection計算の高速化
    giant_indices_set = set(giant_indices)
    valid_target_mask = np.array([cell in giant_indices_set for cell in target_gene_counts.index])
    valid_target_indices = target_gene_counts.index[valid_target_mask]
    gene_counts_giant_filtered = target_gene_counts.iloc[valid_target_mask]
    
    print(f"Debug: Target celltype gene_counts: {len(target_gene_counts)}, Valid: {len(gene_counts_giant_filtered)}")
    
    # 結果の設定（vectorized操作）
    target_result_array = np.zeros(len(giant_indices), dtype=int)
    giant_name_to_idx = {name: i for i, name in enumerate(giant_indices)}
    
    # vectorized assignment
    valid_giant_idx_array = np.array([giant_name_to_idx[cell_id] for cell_id in valid_target_indices if cell_id in giant_name_to_idx])
    valid_giant_counts = gene_counts_giant_filtered.loc[[cell_id for cell_id in valid_target_indices if cell_id in giant_name_to_idx]].values
    if len(valid_giant_idx_array) > 0:
        target_result_array[valid_giant_idx_array] = valid_giant_counts.astype(int)
    
    # cluster_cells.obsの更新
    cluster_cells.obs.loc[giant_mask, 'Gene_CCI'] = target_result_array
    
    # target計算の高速化（NumPy）
    target_cluster_labels = cluster_labels[giant_mask]
    target_gene_cci = gene_cci_values[giant_mask]
    
    unique_target_clusters = np.unique(target_cluster_labels)
    sum_gene_cci_list = []
    cluster_counts_list = []
    
    for cluster in unique_target_clusters:
        cluster_mask = target_cluster_labels == cluster
        sum_gene_cci_list.append(np.sum(target_gene_cci[cluster_mask]))
        cluster_counts_list.append(np.sum(cluster_mask))
    
    sum_gene_cci = pd.Series(sum_gene_cci_list, index=unique_target_clusters)
    cluster_counts = pd.Series(cluster_counts_list, index=unique_target_clusters)
    
    # ゼロ除算回避
    mean_gene_cci_per_cell = np.divide(sum_gene_cci.values, cluster_counts.values, 
                                       out=np.zeros_like(sum_gene_cci.values, dtype=float), 
                                       where=cluster_counts.values!=0)
    mean_gene_cci_per_cell = pd.Series(mean_gene_cci_per_cell, index=unique_target_clusters)

    print("Creating bar plot 2...")
    
    # Calculate proportion of target cell type that received ligand stimulation per cluster
    target_stimulated_counts_list = []
    target_total_counts_list = []
    
    for cluster in unique_target_clusters:
        cluster_mask = target_cluster_labels == cluster
        cluster_target_gene_cci = target_gene_cci[cluster_mask]
        
        # Count target cells that received stimulation (CCI > 0) at least once
        target_stimulated_count = np.sum(cluster_target_gene_cci > 0)
        target_total_count = np.sum(cluster_mask)
        
        target_stimulated_counts_list.append(target_stimulated_count)
        target_total_counts_list.append(target_total_count)
    
    target_stimulated_counts = pd.Series(target_stimulated_counts_list, index=unique_target_clusters)
    target_total_counts = pd.Series(target_total_counts_list, index=unique_target_clusters)
    
    # Calculate proportion (percentage) for target cell type
    target_stimulation_proportion = np.divide(target_stimulated_counts.values, target_total_counts.values, 
                                            out=np.zeros_like(target_stimulated_counts.values, dtype=float), 
                                            where=target_total_counts.values!=0) * 100
    target_stimulation_proportion = pd.Series(target_stimulation_proportion, index=unique_target_clusters)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    target_stimulation_proportion.plot(kind='bar', color='skyblue', ax=ax2)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('% of ' + target_cell_type + ' with ' + Gene_to_analyze + ' stimulation')
    ax2.set_title('Proportion of ' + target_cell_type + ' receiving ' + Gene_to_analyze + '-stimulation per TME cluster')
    #ax2.set_ylim(0, 100)  # Set y-axis to percentage scale
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 表示してから保存
    plt.show()
    
    if save:
        filename = f"{SAMPLE_NAME}_{Gene_to_analyze}-stimulated_{target_cell_type}_proportion_target_celltype.pdf"
        out_pdf = os.path.join(save_path_for_today, filename)
        fig2.savefig(out_pdf, format="pdf", dpi=100, bbox_inches="tight")
        print(f"Saved: {filename}")
    
    plt.close(fig2)

    # --- Plot spatial map ---
    # 画像とデータの事前準備
    hires_img = cluster_cells.uns["spatial"][lib_id]["images"]["hires"]
    h, w = hires_img.shape[:2]
    scale = cluster_cells.uns["spatial"][lib_id]["scalefactors"]["tissue_hires_scalef"]
    
    # spatial座標の高速処理（vectorized操作）
    spatial_coords = cluster_cells.obsm["spatial"] * scale
    
    fig3, ax3 = plt.subplots(figsize=(6, 6), dpi=100)
    ax3.imshow(hires_img, extent=[0, w, h, 0], alpha=0.2)
    ax3.set_xlim(0, w)
    ax3.set_ylim(h, 0)
    ax3.axis('off')
    
    # boolean操作の高速化（NumPy）
    gene_cci_plot_values = gene_cci_values.copy()
    non_target_mask = celltype_values != target_cell_type
    gene_cci_plot_values[non_target_mask] = 0
    
    # alpha値の計算（vectorized）
    alphas = (gene_cci_plot_values != 0).astype(float)
    
    scatter = ax3.scatter(
        spatial_coords[:, 0], spatial_coords[:, 1],
        c=gene_cci_plot_values,
        cmap='jet',
        s=1,
        alpha=alphas,
        edgecolors='none'
    )
    ax3.set_title(Gene_to_analyze + '-activated ' + target_cell_type, fontsize=8)
    
    cax = fig3.add_axes([0.85, 0.2, 0.03, 0.6])
    cb = fig3.colorbar(scatter, cax=cax)
    cb.set_label("CCI count", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    plt.subplots_adjust(left=0.05, right=0.82, top=0.95, bottom=0.05)
    
    # 表示してから保存
    plt.show()
    
    if save:
        filename = f"{SAMPLE_NAME}_{Gene_to_analyze}-activated_{target_cell_type}_spatialmap.pdf"
        out_pdf = os.path.join(save_path_for_today, filename)
        fig3.savefig(out_pdf, format="pdf", dpi=1000, bbox_inches="tight")
        print(f"Saved: {filename}")
    
    plt.close(fig3)

    # --- Plot Sankey diagram 1: All clusters ---
    # query操作の高速化（NumPy boolean indexing）
    cell1_type_coexp = coexp_cc_df['cell1_type'].values
    target_coexp_mask = cell1_type_coexp == target_cell_type
    sub_coexp_cc_df_all = coexp_cc_df[target_coexp_mask].copy()
    
    if significant_column in sub_coexp_cc_df_all.columns:
        sig_mask = sub_coexp_cc_df_all[significant_column] == True
        sub_coexp_cc_df_all = sub_coexp_cc_df_all[sig_mask]
    
    if len(sub_coexp_cc_df_all) == 0:
        print(f"Warning: No significant interactions found for {target_cell_type}")
        return
    
    if 'interaction_positive' in sub_coexp_cc_df_all.columns:
        interaction_filter = sub_coexp_cc_df_all['interaction_positive'] >= minimum_interaction
        sub_coexp_cc_df_all = sub_coexp_cc_df_all[interaction_filter]
    
        interaction_filter = sub_coexp_cc_df_all['cell2_type'].isin(sender_cell_type)
        sub_coexp_cc_df_all = sub_coexp_cc_df_all[interaction_filter]

    # sort_values + groupby.head の処理
    sub_coexp_cc_df_all = sub_coexp_cc_df_all.sort_values(
        display_column, ascending=False
    ).groupby('cell2_type', as_index=False).head(n=each_display_num)

    # Sankeyダイアグラムの作成（全クラスター）
    cell1types_all = np.unique(sub_coexp_cc_df_all["cell1_type"])
    cell2types_all = np.unique(sub_coexp_cc_df_all["cell2_type"])
    tot_list_all = (
        list(sub_coexp_cc_df_all.ligand.unique()) +
        list(cell2types_all) +
        list(cell1types_all)
    )
    
    ligand_pos_dict_all = pd.Series({
        ligand: i for i, ligand in enumerate(sub_coexp_cc_df_all.ligand.unique())
    })
    celltype_pos_dict_all = pd.Series({
        celltype: i + len(ligand_pos_dict_all) for i, celltype in enumerate(cell2types_all)
    })
    receiver_dict_all = pd.Series({
        celltype: i + len(ligand_pos_dict_all) + len(cell2types_all)
        for i, celltype in enumerate(cell1types_all)
    })

    senders_all = (sub_coexp_cc_df_all.cell1_type.values
                   if role == "sender" else sub_coexp_cc_df_all.cell2_type.values)
    receivers_all = (sub_coexp_cc_df_all.cell2_type.values
                     if role == "sender" else sub_coexp_cc_df_all.cell1_type.values)
    
    sources_all = pd.concat([
        ligand_pos_dict_all.loc[sub_coexp_cc_df_all.ligand.values],
        celltype_pos_dict_all.loc[senders_all]
    ])
    targets_all = pd.concat([
        receiver_dict_all.loc[receivers_all],
        ligand_pos_dict_all.loc[sub_coexp_cc_df_all.ligand.values]
    ])
    values_all = pd.concat([
        sub_coexp_cc_df_all[display_column],
        sub_coexp_cc_df_all[display_column]
    ])
    labels_all = pd.concat([
        sub_coexp_cc_df_all['cell1_type'],
        sub_coexp_cc_df_all['cell2_type']
    ])
    
    unique_labels_all = labels_all.unique()
    palette_all = sns.color_palette("tab10", n_colors=len(unique_labels_all)).as_hex()
    target_color_dict_all = dict(zip(unique_labels_all, palette_all))
    colors_all = pd.Series(target_color_dict_all)[labels_all]
    
    fig4 = go.Figure(data=[go.Sankey(
        node=dict(label=tot_list_all),
        link=dict(source=sources_all, target=targets_all, value=values_all, color=colors_all, label=labels_all)
    )])
    fig4.update_layout(
        title=f"{target_cell_type}<br><sub>Only ≥{minimum_interaction} interactions</sub>",
        font_family="Courier New",
        width=1000,
        height=1000,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    if save:
        # Choose save format based on parameter
        if save_format == "html":
            filename = f"{SAMPLE_NAME}_{target_cell_type}_sankey_all_clusters.html"
            out_file = os.path.join(save_path_for_today, filename)
            fig4.write_html(out_file)
            print(f"Saved HTML: {filename}")
        elif save_format == "png":
            filename = f"{SAMPLE_NAME}_{target_cell_type}_sankey_all_clusters.png"
            out_file = os.path.join(save_path_for_today, filename)
            fig4.write_image(out_file, format="png", width=600, height=1000, scale=2)
            print(f"Saved PNG: {filename}")
        elif save_format == "both":
            # HTML (fast)
            filename_html = f"{SAMPLE_NAME}_{target_cell_type}_sankey_all_clusters.html"
            out_html = os.path.join(save_path_for_today, filename_html)
            fig4.write_html(out_html)
            print(f"Saved HTML: {filename_html}")
            
            # PNG (medium speed)
            try:
                filename_png = f"{SAMPLE_NAME}_{target_cell_type}_sankey_all_clusters.png"
                out_png = os.path.join(save_path_for_today, filename_png)
                fig4.write_image(out_png, format="png", width=600, height=1000, scale=2)
                print(f"Saved PNG: {filename_png}")
            except Exception as e:
                print(f"PNG save failed: {e}")
        else:  # pdf (slow - not recommended)
            filename = f"{SAMPLE_NAME}_{target_cell_type}_sankey_all_clusters.pdf"
            out_file = os.path.join(save_path_for_today, filename)
            print(f"Warning: PDF save is slow. Consider using save_format='html' or 'png'")
            fig4.write_image(out_file, format="pdf", width=600, height=1000)
            print(f"Saved PDF: {filename}")

    fig4.show()

    # --- Plot Sankey diagram 2: Target clusters only ---
    print("\n--- Creating Sankey plot 2 (Target clusters only) ---")
    
    # Use the cluster-specific dataframe if available, otherwise fall back to the general one
    # クラスタ特異的なデータフレームがあれば使用し、なければ汎用データフレームを使用
    coexp_data_for_plot = coexp_cc_df_cluster if coexp_cc_df_cluster is not None else coexp_cc_df

    # Step 1: Find the actual cell IDs that match the user's selection
    # ステップ1: ユーザーの選択に合致する実際の細胞IDを特定
    target_cell_mask = cluster_cells.obs['celltype'] == target_cell_type
    target_cluster_mask = cluster_cells.obs['cluster'].astype(str).isin(target_clusters_str)
    
    # Get the names of cells that are the correct type AND in the correct cluster
    # 正しい細胞タイプかつ正しいクラスターに属する細胞の名前を取得
    cells_in_target_clusters = cluster_cells.obs_names[target_cell_mask & target_cluster_mask]
    
    if cells_in_target_clusters.empty:
        print(f"Warning: No '{target_cell_type}' cells found in the selected clusters: {target_clusters_str}.")
        # To avoid stopping the entire function, we just skip this plot
        # 関数全体を停止させないため、このプロットだけをスキップ
        sub_coexp_cc_df_target = pd.DataFrame()
    else:
        # Step 2: Find all edges originating from these specific cells
        # ステップ2: これらの特定の細胞から発生するすべてのエッジを特定
        edges_in_clusters = edge_df[edge_df['cell1'].isin(cells_in_target_clusters)]

        if edges_in_clusters.empty:
            print(f"Warning: No interactions found originating from '{target_cell_type}' in clusters {target_clusters_str}.")
            sub_coexp_cc_df_target = pd.DataFrame()
        else:
            # Step 3: Use the information from the actual edges to filter the summary dataframe
            # ステップ3: 実際のエッジ情報を使って集計データフレームをフィルタリング
            
            # Find the unique sender cell types that are actually interacting with our target cells in these clusters
            # これらのクラスター内でターゲット細胞と実際に相互作用している送信側細胞タイプのユニークなリストを取得
            actual_senders_in_clusters = edges_in_clusters['cell2_type'].unique()
            
            # Filter the summary data:
            # - Receiver must be the target_cell_type
            # - Sender must be one of the cell types we found in the step above
            # 集計データをフィルタリング：
            # - 受信側は target_cell_type であること
            # - 送信側は上記ステップで見つけた細胞タイプのいずれかであること
            sub_coexp_cc_df_target = coexp_data_for_plot[
                (coexp_data_for_plot['cell1_type'] == target_cell_type) &
                (coexp_data_for_plot['cell2_type'].isin(actual_senders_in_clusters))
            ].copy()

            # Apply the other filters from the widgets
            # ウィジェットからの他のフィルターを適用
            if significant_column in sub_coexp_cc_df_target.columns:
                sub_coexp_cc_df_target = sub_coexp_cc_df_target[sub_coexp_cc_df_target[significant_column] == True]
            
            if 'interaction_positive' in sub_coexp_cc_df_target.columns:
                sub_coexp_cc_df_target = sub_coexp_cc_df_target[sub_coexp_cc_df_target['interaction_positive'] >= minimum_interaction]
            
            # Filter by the selected sender cell types in the widget
            # ウィジェットで選択された送信側細胞タイプでフィルタリング
            sub_coexp_cc_df_target = sub_coexp_cc_df_target[sub_coexp_cc_df_target['cell2_type'].isin(sender_cell_type)]

    # --- Proceed to plotting only if we have data ---
    # データがある場合のみプロット処理に進む
    if sub_coexp_cc_df_target.empty:
        print(f"Final check: No data available to plot Sankey for target clusters {target_clusters_str} with the current filters.")
    else:
        print(f"Found {len(sub_coexp_cc_df_target)} significant interactions for target clusters. Proceeding with Sankey plot...")
        
        # Select top interactions
        # 上位の相互作用を選択
        sub_coexp_cc_df_target = sub_coexp_cc_df_target.sort_values(
            display_column, ascending=False
        ).groupby('cell2_type', as_index=False).head(n=each_display_num)
        
        # (The rest of the plotting code for fig5 remains the same)
        # (ここから下の fig5 のプロットコードは変更なし)
        cell1types_target = np.unique(sub_coexp_cc_df_target["cell1_type"])
        cell2types_target = np.unique(sub_coexp_cc_df_target["cell2_type"])
        tot_list_target = (
            list(sub_coexp_cc_df_target.ligand.unique()) +
            list(cell2types_target) +
            list(cell1types_target)
        )
        
        ligand_pos_dict_target = pd.Series({
            ligand: i for i, ligand in enumerate(sub_coexp_cc_df_target.ligand.unique())
        })
        celltype_pos_dict_target = pd.Series({
            celltype: i + len(ligand_pos_dict_target) for i, celltype in enumerate(cell2types_target)
        })
        receiver_dict_target = pd.Series({
            celltype: i + len(ligand_pos_dict_target) + len(cell2types_target)
            for i, celltype in enumerate(cell1types_target)
        })

        senders_target = (sub_coexp_cc_df_target.cell1_type.values
                          if role == "sender" else sub_coexp_cc_df_target.cell2_type.values)
        receivers_target = (sub_coexp_cc_df_target.cell2_type.values
                            if role == "sender" else sub_coexp_cc_df_target.cell1_type.values)
        
        sources_target = pd.concat([
            ligand_pos_dict_target.loc[sub_coexp_cc_df_target.ligand.values],
            celltype_pos_dict_target.loc[senders_target]
        ])
        targets_target = pd.concat([
            receiver_dict_target.loc[receivers_target],
            ligand_pos_dict_target.loc[sub_coexp_cc_df_target.ligand.values]
        ])
        values_target = pd.concat([
            sub_coexp_cc_df_target[display_column],
            sub_coexp_cc_df_target[display_column]
        ])
        labels_target = pd.concat([
            sub_coexp_cc_df_target['cell1_type'],
            sub_coexp_cc_df_target['cell2_type']
        ])
        
        unique_labels_target = labels_target.unique()
        palette_target = sns.color_palette("Set2", n_colors=len(unique_labels_target)).as_hex()
        target_color_dict_target = dict(zip(unique_labels_target, palette_target))
        colors_target = pd.Series(target_color_dict_target)[labels_target]
        
        fig5 = go.Figure(data=[go.Sankey(
            node=dict(label=tot_list_target),
            link=dict(source=sources_target, target=targets_target, value=values_target, color=colors_target, label=labels_target)
        )])
        fig5.update_layout(
            title=f"{target_cell_type} - Clusters {target_clusters}<br><sub>Only ≥{minimum_interaction} interactions</sub>",
            font_family="Courier New",
            width=1000,
            height=1000,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        if save:
            # (Save logic remains the same)
            # (保存ロジックは変更なし)
            if save_format == "html":
                filename = f"{SAMPLE_NAME}_{target_cell_type}_sankey_target_clusters.html"
                out_file = os.path.join(save_path_for_today, filename)
                fig5.write_html(out_file)
                print(f"Saved HTML: {filename}")
            # ... (add other save formats if needed) ...

        fig5.show()
    
    print(f"Successfully generated NumPy-optimized plots for {target_cell_type} - {Gene_to_analyze}")
    print("Two Sankey diagrams created:")
    print("1. All clusters")
    print(f"2. Target clusters: {target_clusters}")
    
    # Get all gene columns from bargraph_df
    gene_columns = [col for col in bargraph_df.columns if col not in ['cell1_type', 'cell2_type']]
    
    # Calculate total ligand response for each cell (sum across all genes)
    total_ligand_data = bargraph_df[gene_columns].values
    total_ligand_response_per_cell = np.sum(total_ligand_data, axis=1)
    
    # Create mapping from cell1 to total response
    cell1_to_total_response = dict(zip(edge_df["cell1"].values, total_ligand_response_per_cell))
    
    # Map to cluster_cells
    total_response_array = np.zeros(len(cluster_cells.obs_names), dtype=int)
    for i, cell_name in enumerate(cluster_cells.obs_names):
        if cell_name in cell1_to_total_response:
            total_response_array[i] = int(cell1_to_total_response[cell_name])
    
    cluster_cells.obs['Total_Ligand_Response'] = total_response_array
    
    # --- Fig6: Bar plot for all cell types (total ligand response) ---
    
    total_response_values = cluster_cells.obs['Total_Ligand_Response'].values
    
    # Calculate proportion of cells that received any ligand stimulation per cluster
    total_stimulated_counts_list = []
    total_counts_list = []
    
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_total_response = total_response_values[cluster_mask]
        
        # Count cells that received any stimulation (total response > 0)
        total_stimulated_count = np.sum(cluster_total_response > 0)
        total_count = np.sum(cluster_mask)
        
        total_stimulated_counts_list.append(total_stimulated_count)
        total_counts_list.append(total_count)
    
    total_stimulated_counts = pd.Series(total_stimulated_counts_list, index=unique_clusters)
    total_counts = pd.Series(total_counts_list, index=unique_clusters)
    
    # Calculate proportion (percentage)
    total_stimulation_proportion = np.divide(total_stimulated_counts.values, total_counts.values, 
                                           out=np.zeros_like(total_stimulated_counts.values, dtype=float), 
                                           where=total_counts.values!=0) * 100
    total_stimulation_proportion = pd.Series(total_stimulation_proportion, index=unique_clusters)
    
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    total_stimulation_proportion.plot(kind='bar', color='lightcoral', ax=ax6)
    ax6.set_xlabel('Cluster')
    ax6.set_ylabel('% of cells with any ligand stimulation')
    ax6.set_title('Proportion of cells receiving any ligand stimulation per TME cluster (all cell types)')
    #ax6.set_ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()
    
    if save:
        filename = f"{SAMPLE_NAME}_total-ligand-stimulated_{target_cell_type}_proportion_all_clusters.pdf"
        out_pdf = os.path.join(save_path_for_today, filename)
        fig6.savefig(out_pdf, format="pdf", dpi=100, bbox_inches="tight")
        print(f"Saved: {filename}")
    
    plt.close(fig6)
    
    # --- Fig7: Bar plot for target cell type (total ligand response) ---
    
    # Filter for target cell type and calculate total response
    target_total_response_values = total_response_values[giant_mask]
    
    # Calculate proportion of target cells that received any ligand stimulation per cluster
    target_total_stimulated_counts_list = []
    target_total_counts_list = []
    
    for cluster in unique_target_clusters:
        cluster_mask = target_cluster_labels == cluster
        cluster_target_total_response = target_total_response_values[cluster_mask]
        
        # Count target cells that received any stimulation
        target_total_stimulated_count = np.sum(cluster_target_total_response > 0)
        target_total_count = np.sum(cluster_mask)
        
        target_total_stimulated_counts_list.append(target_total_stimulated_count)
        target_total_counts_list.append(target_total_count)
    
    target_total_stimulated_counts = pd.Series(target_total_stimulated_counts_list, index=unique_target_clusters)
    target_total_counts = pd.Series(target_total_counts_list, index=unique_target_clusters)
    
    # Calculate proportion (percentage) for target cell type
    target_total_stimulation_proportion = np.divide(target_total_stimulated_counts.values, target_total_counts.values, 
                                                  out=np.zeros_like(target_total_stimulated_counts.values, dtype=float), 
                                                  where=target_total_counts.values!=0) * 100
    target_total_stimulation_proportion = pd.Series(target_total_stimulation_proportion, index=unique_target_clusters)

    fig7, ax7 = plt.subplots(figsize=(10, 6))
    target_total_stimulation_proportion.plot(kind='bar', color='lightcoral', ax=ax7)
    ax7.set_xlabel('Cluster')
    ax7.set_ylabel('% of ' + target_cell_type + ' with any ligand stimulation')
    ax7.set_title('Proportion of ' + target_cell_type + ' receiving any ligand stimulation per TME cluster')
    #ax7.set_ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()
    
    if save:
        filename = f"{SAMPLE_NAME}_total-ligand-stimulated_{target_cell_type}_proportion_target_celltype.pdf"
        out_pdf = os.path.join(save_path_for_today, filename)
        fig7.savefig(out_pdf, format="pdf", dpi=100, bbox_inches="tight")
        print(f"Saved: {filename}")
    
    plt.close(fig7)
    
    # --- Fig8: Spatial map (total ligand response) ---
    
    # 画像とデータの事前準備
    hires_img = cluster_cells.uns["spatial"][lib_id]["images"]["hires"]
    h, w = hires_img.shape[:2]
    scale = cluster_cells.uns["spatial"][lib_id]["scalefactors"]["tissue_hires_scalef"]
    
    # spatial座標の高速処理（vectorized操作）
    spatial_coords = cluster_cells.obsm["spatial"] * scale
    
    fig8, ax8 = plt.subplots(figsize=(6, 6), dpi=100)
    ax8.imshow(hires_img, extent=[0, w, h, 0], alpha=0.2)
    ax8.set_xlim(0, w)
    ax8.set_ylim(h, 0)
    ax8.axis('off')
    
    # boolean操作の高速化（NumPy）
    total_response_plot_values = total_response_values.copy()
    non_target_mask = celltype_values != target_cell_type
    total_response_plot_values[non_target_mask] = 0
    
    # alpha値の計算（vectorized）
    alphas = (total_response_plot_values != 0).astype(float)
    
    scatter = ax8.scatter(
        spatial_coords[:, 0], spatial_coords[:, 1],
        c=total_response_plot_values,
        cmap='Reds',  # Different colormap for total response
        s=1,
        alpha=alphas,
        edgecolors='none'
    )
    ax8.set_title('Total ligand-stimulated ' + target_cell_type, fontsize=8)
    
    cax = fig8.add_axes([0.85, 0.2, 0.03, 0.6])
    cb = fig8.colorbar(scatter, cax=cax)
    cb.set_label("Total CCI count", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    plt.subplots_adjust(left=0.05, right=0.82, top=0.95, bottom=0.05)
    
    plt.show()
    
    if save:
        filename = f"{SAMPLE_NAME}_total-ligand-stimulated_{target_cell_type}_spatialmap_cropped.pdf"
        out_pdf = os.path.join(save_path_for_today, filename)
        fig8.savefig(out_pdf, format="pdf", dpi=1000, bbox_inches="tight")
        print(f"Saved: {filename}")
    
    plt.close(fig8)
    # 表示確認
    print("All plots (including total ligand response) should now be displayed above.")
    print("Generated plots:")
    print("Fig1: Single ligand stimulation proportion (all cell types)")
    print("Fig2: Single ligand stimulation proportion (target cell type)")
    print("Fig3: Single ligand spatial map")
    print("Fig4: Single ligand Sankey (all clusters)")
    print("Fig5: Single ligand Sankey (target clusters)")
    print("Fig6: Total ligand stimulation proportion (all cell types)")
    print("Fig7: Total ligand stimulation proportion (target cell type)")
    print("Fig8: Total ligand spatial map")

def plot_all_clusters_highlights(analyzer):
    """全Leidenクラスタのハイライトプロット"""
    
    # クラスタIDを取得
    cluster_ids = sorted(analyzer.adata.obs['leiden'].astype(str).unique())
    print(f"Clusters: {len(cluster_ids)}")
    
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
    print(f"Cell types: {len(cluster_ids)}")
    
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

# English Font Settings
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

def calculate_observed_proximities(df):
    """
    Calculate actually observed proximity counts
    """
    proximities = defaultdict(int)
    
    # For each cell1, count the types of its neighbors
    for _, row in df.iterrows():
        cell1_type = row['cell1_type']
        cell2_type = row['cell2_type']
        
        # Count proximity using cell type pairs as keys
        pair = tuple(sorted([cell1_type, cell2_type]))
        proximities[pair] += 1
    
    return dict(proximities)

def get_cell_info(df):
    """
    Get information about each cell and neighbor relationships
    """
    cell_info = {}
    
    # Collect basic information for each cell
    for _, row in df.iterrows():
        cell1_id = row['cell1']
        cell2_id = row['cell2']
        
        # Record cell1 information
        if cell1_id not in cell_info:
            cell_info[cell1_id] = {
                'type': row['cell1_type'],
                'cluster': row['cell1_cluster'],
                'neighbors': []
            }
        
        # Record cell2 information  
        if cell2_id not in cell_info:
            cell_info[cell2_id] = {
                'type': row['cell2_type'], 
                'cluster': row['cell2_cluster'],
                'neighbors': []
            }
        
        # Record neighbor relationships
        cell_info[cell1_id]['neighbors'].append(cell2_id)
    
    return cell_info

def permutation_test_optimized(df, n_permutations=1000, random_seed=42):
    """
    Optimized permutation test to evaluate statistical significance
    """
    np.random.seed(random_seed)
    
    # Calculate actual observations
    observed_proximities = calculate_observed_proximities(df)
    
    # Get cell information
    cell_info = get_cell_info(df)
    
    # Lists of all cell IDs and types
    all_cell_ids = np.array(list(cell_info.keys()))
    all_cell_types = np.array([cell_info[cell_id]['type'] for cell_id in all_cell_ids])
    
    print(f"Total number of cells: {len(all_cell_ids)}")
    print(f"Cell type varieties: {set(all_cell_types)}")
    print(f"Observed proximity patterns: {observed_proximities}")
    
    # Pre-compute neighbor arrays for faster access
    neighbor_arrays = {}
    for cell_id in all_cell_ids:
        neighbor_indices = [np.where(all_cell_ids == neighbor_id)[0][0] 
                          for neighbor_id in cell_info[cell_id]['neighbors']]
        neighbor_arrays[cell_id] = np.array(neighbor_indices)
    
    # Store permutation results
    permuted_proximities = {pair: np.zeros(n_permutations) for pair in observed_proximities.keys()}
    
    print(f"\nRunning permutation test ({n_permutations} iterations)...")
    
    for perm in range(n_permutations):
        if (perm + 1) % 100 == 0:
            print(f"  Progress: {perm + 1}/{n_permutations}")
        
        # Randomly shuffle cell types
        shuffled_types = np.random.permutation(all_cell_types)
        
        # Calculate proximity counts with shuffled data
        perm_proximities = defaultdict(int)
        
        for i, cell1_id in enumerate(all_cell_ids):
            cell1_type = shuffled_types[i]
            
            # Get neighbor indices and their types
            neighbor_indices = neighbor_arrays[cell1_id]
            for neighbor_idx in neighbor_indices:
                cell2_type = shuffled_types[neighbor_idx]
                pair = tuple(sorted([cell1_type, cell2_type]))
                perm_proximities[pair] += 1
        
        # Record results for each pair
        for pair in observed_proximities.keys():
            permuted_proximities[pair][perm] = perm_proximities.get(pair, 0)
    
    return observed_proximities, permuted_proximities

def calculate_statistics(observed_proximities, permuted_proximities):
    """
    Calculate statistical values (log fold change, p-value)
    """
    results = []
    
    for pair, observed_count in observed_proximities.items():
        perm_counts = permuted_proximities[pair]
        
        # Expected value
        expected_count = np.mean(perm_counts)
        
        # Log fold change calculation (add small value to avoid division by zero)
        log_fc = np.log2((observed_count + 1) / (expected_count + 1))
        
        # p-value calculation (two-tailed test)
        if log_fc >= 0:
            # Probability of getting values >= observed
            p_value = np.sum(perm_counts >= observed_count) / len(perm_counts)
        else:
            # Probability of getting values <= observed  
            p_value = np.sum(perm_counts <= observed_count) / len(perm_counts)
        
        # Two-tailed test, so multiply by 2
        p_value = min(2 * p_value, 1.0)
        
        # Determine if cell types are the same
        is_same_type = pair[0] == pair[1]
        
        results.append({
            'cell_type_pair': f"{pair[0]} - {pair[1]}",
            'type1': pair[0],
            'type2': pair[1],
            'is_same_type': is_same_type,
            'observed_count': observed_count,
            'expected_count': expected_count,
            'log_fold_change': log_fc,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    return pd.DataFrame(results)

def create_horizontal_barplot(results_df, save_path=None, sample_name="sample"):
    """
    Create horizontal bar plots (separated by same type vs different type)
    Figure height proportional to number of combinations
    """
    # Separate data into same type and different type
    same_type = results_df[results_df['is_same_type'] == True].copy()
    diff_type = results_df[results_df['is_same_type'] == False].copy()
    
    # Sort by log fold change
    same_type = same_type.sort_values('log_fold_change')
    diff_type = diff_type.sort_values('log_fold_change')
    
    # Calculate dynamic figure height based on number of combinations
    base_height = 3
    height_per_item = 0.5
    same_height = max(base_height, len(same_type) * height_per_item)
    diff_height = max(base_height, len(diff_type) * height_per_item)
    total_height = same_height + diff_height + 2  # Add space for titles
    
    fig, axes = plt.subplots(2, 1, figsize=(12, total_height), 
                            gridspec_kw={'height_ratios': [same_height, diff_height]}, 
                            sharex=True)
    
    # Same cell type pairs
    if len(same_type) > 0:
        colors_same = ['red' if p < 0.05 else 'lightcoral' for p in same_type['p_value']]
        y_pos_same = np.arange(len(same_type))
        
        bars1 = axes[0].barh(y_pos_same, same_type['log_fold_change'], color=colors_same, alpha=0.8)
        axes[0].set_yticks(y_pos_same)
        axes[0].set_yticklabels(same_type['cell_type_pair'])
        axes[0].set_title('Same Cell Type Pairs (Homotypic Proximity)', fontsize=14, pad=20)
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=1)
        axes[0].grid(True, alpha=0.3)
        
        # Add significance annotations
        for i, (idx, row) in enumerate(same_type.iterrows()):
            x_pos = row['log_fold_change']
            ha_align = 'left' if x_pos > 0 else 'right'
            x_offset = 0.05 if x_pos > 0 else -0.05
            
            if row['p_value'] < 0.001:
                axes[0].text(x_pos + x_offset, i, '***', ha=ha_align, va='center', fontweight='bold')
            elif row['p_value'] < 0.01:
                axes[0].text(x_pos + x_offset, i, '**', ha=ha_align, va='center', fontweight='bold')
            elif row['p_value'] < 0.05:
                axes[0].text(x_pos + x_offset, i, '*', ha=ha_align, va='center', fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'No same type pairs found', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Same Cell Type Pairs (Homotypic Proximity)', fontsize=14, pad=20)
    
    # Different cell type pairs
    if len(diff_type) > 0:
        colors_diff = ['blue' if p < 0.05 else 'lightblue' for p in diff_type['p_value']]
        y_pos_diff = np.arange(len(diff_type))
        
        bars2 = axes[1].barh(y_pos_diff, diff_type['log_fold_change'], color=colors_diff, alpha=0.8)
        axes[1].set_yticks(y_pos_diff)
        axes[1].set_yticklabels(diff_type['cell_type_pair'])
        axes[1].set_title('Different Cell Type Pairs (Heterotypic Proximity)', fontsize=14, pad=20)
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1)
        axes[1].grid(True, alpha=0.3)
        
        # Add significance annotations
        for i, (idx, row) in enumerate(diff_type.iterrows()):
            x_pos = row['log_fold_change']
            ha_align = 'left' if x_pos > 0 else 'right'
            x_offset = 0.05 if x_pos > 0 else -0.05
            
            if row['p_value'] < 0.001:
                axes[1].text(x_pos + x_offset, i, '***', ha=ha_align, va='center', fontweight='bold')
            elif row['p_value'] < 0.01:
                axes[1].text(x_pos + x_offset, i, '**', ha=ha_align, va='center', fontweight='bold')
            elif row['p_value'] < 0.05:
                axes[1].text(x_pos + x_offset, i, '*', ha=ha_align, va='center', fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No different type pairs found', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Different Cell Type Pairs (Heterotypic Proximity)', fontsize=14, pad=20)
    
    # X-axis label
    axes[1].set_xlabel('Log2 Fold Change\n← Segregation Tendency    Proximity Tendency →', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        filename = f"{sample_name}_cell_type_neighbor_proximity_barplot.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filepath}")
    
    plt.show()
    
    return fig

def display_results_table(results_df, save_path=None, sample_name="sample"):
    """
    Display and save results table
    """
    # Format results for display
    display_df = results_df.copy()
    display_df['log_fold_change'] = display_df['log_fold_change'].round(3)
    display_df['p_value'] = display_df['p_value'].round(4)
    display_df['expected_count'] = display_df['expected_count'].round(1)
    
    print("\n=== Statistical Analysis Results ===")
    print(display_df[['cell_type_pair', 'observed_count', 'expected_count', 
                     'log_fold_change', 'p_value', 'significant']].to_string(index=False))
    
    significant_count = sum(display_df['significant'])
    proximal_count = sum((display_df['log_fold_change'] > 0) & display_df['significant'])
    segregated_count = sum((display_df['log_fold_change'] < 0) & display_df['significant'])
    
    print(f"\nSignificant proximity patterns (p < 0.05): {significant_count}")
    print(f"Significantly proximal pairs: {proximal_count}")
    print(f"Significantly segregated pairs: {segregated_count}")
    
    # Save table if path provided
    if save_path:
        filename = f"{sample_name}_cell_type_neighbor_proximity_results.csv"
        filepath = os.path.join(save_path, filename)
        results_df.to_csv(filepath, index=False)
        print(f"Results table saved: {filepath}")
    
    return display_df

def analyze_cell_proximity(df, n_permutations=1000, random_seed=42, 
                         save_path=None, sample_name="sample", exclude_self=True):
    print("Starting cell proximity statistical analysis...")
    print(f"Original data shape: {df.shape}")
    
    # Filter out self-proximity if requested
    if exclude_self:
        original_count = len(df)
        df_filtered = df[df['cell1'] != df['cell2']].copy()
        excluded_count = original_count - len(df_filtered)
        print(f"Excluded {excluded_count} self-proximity entries")
        print(f"Filtered data shape: {df_filtered.shape}")
        df = df_filtered
    
    # Run optimized permutation test
    observed_proximities, permuted_proximities = permutation_test_optimized(df, n_permutations, random_seed)
    
    # Calculate statistics
    results_df = calculate_statistics(observed_proximities, permuted_proximities)
    
    # Display and save results
    display_df = display_results_table(results_df, save_path, sample_name)
    
    # Create and save visualization
    fig = create_horizontal_barplot(results_df, save_path, sample_name)
    
    if save_path:
        print(f"\nAll files saved to: {save_path}")
    
    return results_df, fig


class ClusterNamer:
    """
    An interactive widget for naming spatial clusters in Jupyter environments.

    This tool displays a spatial plot of clustered cells over a histology image
    and provides a user interface to assign meaningful names to each cluster ID.
    The annotations can then be saved back to the AnnData object.
    """
    def __init__(self, adata, cluster_key='leiden', image_key='hires', save_key='predicted_cell_type'):
        """
        Initializes the naming widget.

        Parameters
        ----------
        adata : AnnData
            The annotated data object containing spatial information.
        cluster_key : str
            The column in `adata.obs` that contains the cluster labels (e.g., 'leiden').
        image_key : str
            The key for the histology image in `adata.uns['spatial'][lib_id]['images']`.
        save_key : str
            The new column name in `adata.obs` where the annotations will be saved.
        """
        # --- 1. Data Initialization ---
        self.adata_ref = adata
        self.cluster_key = cluster_key
        self.save_key = save_key
        self.pan_offset = [0, 0] # [x_offset, y_offset]

        # --- 2. Load Spatial Data ---
        try:
            lib_id = list(self.adata_ref.uns['spatial'].keys())[0]
            self.image = self.adata_ref.uns['spatial'][lib_id]['images'][image_key]
            scale_factor = self.adata_ref.uns['spatial'][lib_id]['scalefactors'][f'tissue_{image_key}_scalef']
            self.coords = self.adata_ref.obsm['spatial'] * scale_factor
        except Exception as e:
            print(f"🛑 Error loading spatial data: {e}")
            return

        # --- 3. Cluster & Color Setup ---
        try:
            cluster_ids_int = self.adata_ref.obs[self.cluster_key].astype(str).unique().astype(int)
            self.clusters = np.array(sorted(cluster_ids_int)).astype(str)
        except ValueError:
            self.clusters = sorted(self.adata_ref.obs[self.cluster_key].astype(str).unique())
        
        palette = sns.color_palette("tab20", n_colors=max(20, len(self.clusters)))
        self.color_map = {cluster: palette[i % len(palette)] for i, cluster in enumerate(self.clusters)}

        # --- 4. Plotting Elements ---
        self.fig, self.ax = None, None
        self.scatter_plots = {}
        self.output = widgets.Output()

        # --- 5. UI Initialization ---
        self.create_ui()

    def create_ui(self):
        """Creates all the ipywidgets UI components."""
        existing_annotations = {}
        if self.save_key in self.adata_ref.obs:
            mapping_df = self.adata_ref.obs[[self.cluster_key, self.save_key]].drop_duplicates()
            mapping_df[self.cluster_key] = mapping_df[self.cluster_key].astype(str)
            existing_annotations = pd.Series(mapping_df[self.save_key].values, index=mapping_df[self.cluster_key]).to_dict()

        self.text_inputs = {}
        for c in self.clusters:
            default_value = existing_annotations.get(c, "")
            if default_value == 'Unannotated':
                default_value = ""
            self.text_inputs[c] = widgets.Text(value=default_value, description=f"Cluster {c}:", placeholder="e.g., Tumor cells", layout=widgets.Layout(width='flex'))
        
        self.visibility_checkboxes = {}
        for c in self.clusters:
            cb = widgets.Checkbox(value=True, description="", indent=False, layout=widgets.Layout(width='auto'))
            cb.observe(self.on_visibility_change, names='value')
            self.visibility_checkboxes[c] = cb
        
        # Buttons
        self.save_button = widgets.Button(description="Save Annotations", button_style='primary', icon='save')
        self.reset_button = widgets.Button(description="Reset All", button_style='warning', icon='refresh')
        self.select_all_button = widgets.Button(description="Select All", layout=widgets.Layout(width='120px'))
        self.deselect_all_button = widgets.Button(description="Deselect All", layout=widgets.Layout(width='120px'))
        
        # Pan buttons
        self.pan_up_button = widgets.Button(icon='arrow-up', layout=widgets.Layout(width='50px'))
        self.pan_down_button = widgets.Button(icon='arrow-down', layout=widgets.Layout(width='50px'))
        self.pan_left_button = widgets.Button(icon='arrow-left', layout=widgets.Layout(width='50px'))
        self.pan_right_button = widgets.Button(icon='arrow-right', layout=widgets.Layout(width='50px'))
        
        # Callbacks
        self.save_button.on_click(self.on_save_clicked)
        self.reset_button.on_click(self.on_reset_clicked)
        self.select_all_button.on_click(self.on_select_all_clicked)
        self.deselect_all_button.on_click(self.on_deselect_all_clicked)
        self.pan_up_button.on_click(self.on_pan)
        self.pan_down_button.on_click(self.on_pan)
        self.pan_left_button.on_click(self.on_pan)
        self.pan_right_button.on_click(self.on_pan)

        self.status_label = widgets.HTML(value="<b>Status:</b> Ready")
        
        # Sliders
        self.opacity_slider = widgets.FloatSlider(value=0.8, min=0.0, max=1.0, step=0.1, description='Image Opacity:', continuous_update=False)
        self.point_opacity_slider = widgets.FloatSlider(value=0.8, min=0.0, max=1.0, step=0.1, description='Point Opacity:', continuous_update=False)
        self.point_size_slider = widgets.FloatSlider(value=10.0, min=1.0, max=50.0, step=1.0, description='Point Size:', continuous_update=False)
        self.zoom_slider = widgets.FloatSlider(value=1.0, min=1.0, max=10.0, step=0.5, description='Zoom:', continuous_update=False)

        self.opacity_slider.observe(self.on_interactive_update, names='value')
        self.point_opacity_slider.observe(self.on_interactive_update, names='value')
        self.point_size_slider.observe(self.on_interactive_update, names='value')
        self.zoom_slider.observe(self.on_interactive_update, names='value')

    def create_plot(self):
        """Creates and renders the matplotlib plot."""
        with self.output:
            clear_output(wait=True)
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.img_artist = self.ax.imshow(self.image, alpha=self.opacity_slider.value)

            for cluster in self.clusters:
                is_visible = self.visibility_checkboxes[cluster].value
                mask = self.adata_ref.obs[self.cluster_key].astype(str) == cluster
                cluster_coords = self.coords[mask]
                
                # ✅ FIX: Use column 0 for X, 1 for Y to fix rotation.
                scatter = self.ax.scatter(
                    cluster_coords[:, 0],  # X coordinate
                    cluster_coords[:, 1],  # Y coordinate
                    s=self.point_size_slider.value,
                    color=self.color_map[cluster],
                    label=f"Cluster {cluster}",
                    rasterized=True, visible=is_visible, alpha=self.point_opacity_slider.value
                )
                self.scatter_plots[cluster] = scatter
            
            self.ax.set_title(f"Spatial Plot of '{self.cluster_key}'")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.invert_yaxis() # Invert y-axis to match image coordinate system
            self.update_view()
            plt.tight_layout()
            plt.show()
    
    def update_view(self):
        """Updates the plot's zoom and pan."""
        if not self.ax: return
        h, w, _ = self.image.shape
        zoom = self.zoom_slider.value
        
        view_w, view_h = w / zoom, h / zoom
        center_x, center_y = w / 2 + self.pan_offset[0], h / 2 + self.pan_offset[1]
        
        self.ax.set_xlim(center_x - view_w / 2, center_x + view_w / 2)
        # Y-limits are inverted because the axis is inverted
        self.ax.set_ylim(center_y + view_h / 2, center_y - view_h / 2)
        
        if self.fig: self.fig.canvas.draw_idle()

    def on_save_clicked(self, b):
        """Callback function for the save button."""
        annotation_map = {c: name.value for c, name in self.text_inputs.items() if name.value.strip()}
        if not annotation_map:
            self.status_label.value = "<b>Status:</b> ⚠️ No names entered."
            return

        series = self.adata_ref.obs[self.cluster_key].astype(str).map(annotation_map).fillna('Unannotated').astype('category')
        self.adata_ref.obs[self.save_key] = series
        self.adata_ref.obs['scvi_predicted_labels'] = series
        
        self.status_label.value = f"<b>Status:</b> ✅ Saved to `obs['{self.save_key}']` & `obs['scvi_predicted_labels']`"
        print("--- Annotation Summary ---\n", self.adata_ref.obs[self.save_key].value_counts())
        
    def on_reset_clicked(self, b):
        """Resets names, view, and redraws the plot."""
        for text_input in self.text_inputs.values(): text_input.value = ""
        self.status_label.value = "<b>Status:</b> Cleared names and reset view."
        self.zoom_slider.value = 1.0
        self.pan_offset = [0, 0]
        self.create_plot()

    def on_visibility_change(self, change):
        """Redraws the plot when a visibility checkbox changes."""
        self.create_plot()
        self.status_label.value = "<b>Status:</b> Plot updated."
    
    def on_select_all_clicked(self, b):
        """Callback for the 'Select All' button."""
        # ✅ FIX: Unobserve to prevent multiple redraws
        for cb in self.visibility_checkboxes.values():
            cb.unobserve(self.on_visibility_change, names='value')
        
        for cb in self.visibility_checkboxes.values():
            cb.value = True
        
        # Re-observe
        for cb in self.visibility_checkboxes.values():
            cb.observe(self.on_visibility_change, names='value')
        
        self.create_plot()

    def on_deselect_all_clicked(self, b):
        """Callback for the 'Deselect All' button."""
        # ✅ FIX: Unobserve to prevent multiple redraws
        for cb in self.visibility_checkboxes.values():
            cb.unobserve(self.on_visibility_change, names='value')
        
        for cb in self.visibility_checkboxes.values():
            cb.value = False
        
        # Re-observe
        for cb in self.visibility_checkboxes.values():
            cb.observe(self.on_visibility_change, names='value')

        self.create_plot()

    def on_interactive_update(self, change):
        """Redraws the plot for any interactive widget change."""
        self.create_plot()
        
    def on_pan(self, b):
        """Callback for panning buttons."""
        h, w, _ = self.image.shape
        pan_step = (w / self.zoom_slider.value) * 0.2
        
        if b.icon == 'arrow-up': self.pan_offset[1] -= pan_step
        elif b.icon == 'arrow-down': self.pan_offset[1] += pan_step
        elif b.icon == 'arrow-left': self.pan_offset[0] -= pan_step
        elif b.icon == 'arrow-right': self.pan_offset[0] += pan_step
        self.create_plot() # Redraw to apply pan

    def run(self):
        """Assembles the UI and plot, then displays the widget."""
        naming_entries = []
        for c in self.clusters:
            color_hex = to_hex(self.color_map[c])
            color_swatch = widgets.HTML(f"<div style='width:15px; height:15px; background-color:{color_hex}; border:1px solid black;'></div>")
            entry = widgets.HBox([self.visibility_checkboxes[c], color_swatch, self.text_inputs[c]])
            naming_entries.append(entry)
        
        cluster_list_box = widgets.VBox(naming_entries, layout=widgets.Layout(overflow_y='auto', max_height='600px'))
        
        pan_controls = widgets.VBox([
            self.pan_up_button,
            widgets.HBox([self.pan_left_button, self.pan_right_button]),
            self.pan_down_button
        ], layout=widgets.Layout(align_items='center'))
        
        control_panel = widgets.VBox([
            widgets.HTML("<h3>Cluster Naming Tool</h3>"),
            widgets.HTML("<p>Enter a name and check/uncheck to show/hide clusters.</p>"),
            cluster_list_box,
            widgets.HBox([self.select_all_button, self.deselect_all_button]),
            widgets.HTML("<hr>"), widgets.HTML("<b>Display Controls:</b>"),
            self.opacity_slider, self.point_opacity_slider, self.point_size_slider, self.zoom_slider,
            pan_controls,
            widgets.HTML("<hr>"),
            widgets.HBox([self.save_button, self.reset_button]), self.status_label
        ], layout=widgets.Layout(width='400px', margin='0 10px 0 0'))

        display(widgets.HBox([control_panel, self.output]))
        self.create_plot()

def name_clusters_interactively(adata, cluster_key='leiden', image_key='hires', save_key='predicted_cell_type'):
    """
    Launches the interactive cluster naming widget for a spatial AnnData object.
    """
    widget = ClusterNamer(adata=adata, cluster_key=cluster_key, image_key=image_key, save_key=save_key)
    widget.run()
    return widget

class SpatialExpressionVisualizer:
    """
    Interactive visualization tool for spatial gene expression analysis
    """
    
    def __init__(self, 
                 sp_adata_raw, 
                 sp_adata_microenvironment,
                 mask_coords: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize the visualizer
        
        Parameters:
        -----------
        sp_adata_raw : AnnData
            Raw spatial transcriptomics data
        sp_adata_microenvironment : AnnData  
            Microenvironment annotated data
        mask_coords : tuple, optional
            Mask coordinates (x1, x2, y1, y2) for spatial filtering
        """
        self.sp_adata_raw = sp_adata_raw
        self.sp_adata_microenvironment = sp_adata_microenvironment
        self.mask_coords = mask_coords
        self.processed_data = None
        self._is_updating_genes = False # Flag to prevent recursive updates
        
        # Validate data compatibility
        self._validate_data()
    
    def _validate_data(self):
        """Validate input data compatibility"""
        if 'predicted_cell_type' not in self.sp_adata_microenvironment.obs:
            raise ValueError("sp_adata_microenvironment must have 'predicted_cell_type' in obs")
        if 'predicted_microenvironment' not in self.sp_adata_microenvironment.obs:
            raise ValueError("sp_adata_microenvironment must have 'predicted_microenvironment' in obs")
            
    def _prepare_data(self):
        """Prepare and preprocess spatial data"""
        # Apply spatial mask if provided
        if self.mask_coords:
            mask_large_x1, mask_large_x2, mask_large_y1, mask_large_y2 = self.mask_coords
            cell_mask = ((self.sp_adata_raw.obs['array_row'] >= mask_large_x1) & 
                         (self.sp_adata_raw.obs['array_row'] <= mask_large_x2) & 
                         (self.sp_adata_raw.obs['array_col'] >= mask_large_y1) & 
                         (self.sp_adata_raw.obs['array_col'] <= mask_large_y2))
            sp_adata = self.sp_adata_raw.copy()[cell_mask]
        else:
            sp_adata = self.sp_adata_raw.copy()
            
        # Find common cells
        common_cells = self.sp_adata_microenvironment.obs_names.intersection(sp_adata.obs_names)
        sp_adata = self.sp_adata_raw[common_cells].copy()
        
        # Add annotations
        sp_adata.obs["predicted_cell_type"] = self.sp_adata_microenvironment.obs.loc[common_cells, "predicted_cell_type"]
        sp_adata.obs["predicted_microenvironment"] = self.sp_adata_microenvironment.obs.loc[common_cells, "predicted_microenvironment"]
        
        # Normalize
        X = sp_adata.X.toarray() if hasattr(sp_adata.X, "toarray") else sp_adata.X  # shape: (n_cells, n_genes)
        bin_counts = sp_adata.obs['bin_count'].values.astype(float)
        bin_counts[bin_counts == 0] = 1e-9 
        X = X / bin_counts.reshape(-1, 1)
        sp_adata.layers['binned_normalized'] = X
        # sc.pp.normalize_total(sp_adata, target_sum=1e4)
        
        self.processed_data = sp_adata
        return sp_adata
    
    def plot_expression_bars_combined(self, 
                                      cell_types: Union[str, List[str]], 
                                      genes: Union[str, List[str]],
                                      figsize: Tuple[int, int] = (12, 8),
                                      max_cols: int = 4,
                                      palette: str = 'tab10', # Default palette is now tab10
                                      show_std: bool = False,
                                      sum_genes: bool = False) -> None:
        """
        Plot combined gene expression bar charts (OR condition)
        """
        # Ensure inputs are lists
        if isinstance(cell_types, str):
            cell_types = [cell_types]
        if isinstance(genes, str):
            genes = [genes]
            
        # Prepare data if not already done
        if self.processed_data is None:
            self._prepare_data()
            
        sp_adata = self.processed_data
        
        # Validate selections
        available_cell_types = sp_adata.obs['predicted_cell_type'].unique()
        valid_cell_types = [ct for ct in cell_types if ct in available_cell_types]
        
        available_genes = sp_adata.var.index.tolist()
        valid_genes = [g for g in genes if g in available_genes]
        
        if not valid_cell_types:
            print(f"Error: None of the selected cell types found: {cell_types}")
            return
        if not valid_genes:
            print(f"Error: None of the selected genes found: {genes}")
            return
            
        # Filter data for selected cell types (OR condition)
        cell_mask = sp_adata.obs['predicted_cell_type'].isin(valid_cell_types)
        filtered_adata = sp_adata[cell_mask].copy()
        
        # Add expression data for valid genes
        gene_expression_df = pd.DataFrame(
            filtered_adata[:, valid_genes].layers['binned_normalized'].toarray(),
            index=filtered_adata.obs.index,
            columns=valid_genes
        )
        filtered_adata.obs = pd.concat([filtered_adata.obs, gene_expression_df], axis=1)

        # Handle summing genes
        genes_to_plot = valid_genes
        gene_titles = {g: g for g in valid_genes}

        if sum_genes and len(valid_genes) > 1:
            # Create a new column with the summed expression
            sum_col_name = 'summed_expression'
            filtered_adata.obs[sum_col_name] = gene_expression_df[valid_genes].sum(axis=1)
            
            # Update plot variables to handle a single plot for the sum
            genes_to_plot = [sum_col_name]
            gene_titles = {sum_col_name: f"{' + '.join(valid_genes)}\n(Summed Expression)"}

        # Calculate layout
        n_plots = len(genes_to_plot)
        ncols = min(max_cols, n_plots)
        nrows = (n_plots + ncols - 1) // ncols
        
        plt.close('all')
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=False, sharey=False)
        
        # Handle single plot case
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Generate plots for each gene
        for i, gene in enumerate(genes_to_plot):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Calculate statistics per microenvironment for current gene
            # 1. 現在の遺伝子名（gene）に対応する列のインデックス番号を取得します
            gene_idx = filtered_adata.var.index.get_loc(gene)
    
            # 2. .layers['binned_normalized'] から、その遺伝子の発現データを全セル分取得します
            expression_data = filtered_adata.layers['binned_normalized'][:, gene_idx]
            
            # 3. (もしデータが疎行列なら、計算のために密な配列に変換します)
            if hasattr(expression_data, 'toarray'):
                expression_data = expression_data.toarray().flatten()
    
            # 4. 集計用に、クラスタ名と発現データを結合した一時的なDataFrameを作成します
            temp_df = pd.DataFrame({
                'predicted_microenvironment': filtered_adata.obs['predicted_microenvironment'],
                'expression': expression_data
            })
    
            # 5. 作成したDataFrame上で、これまで通りの集計処理を行います
            plot_data = temp_df.groupby('predicted_microenvironment', observed=False)['expression'].agg(['mean', 'std', 'count']).reset_index()
            plot_data.rename(columns={'mean': 'mean_expression', 'std': 'std_expression', 'count': 'n_cells'}, inplace=True)
            
            plot_data['std_expression'] = plot_data['std_expression'].fillna(0)
            plot_data = plot_data[plot_data['n_cells'] > 0]
            
            if len(plot_data) == 0:
                ax.text(0.5, 0.5, f'No data for {gene_titles[gene]}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(gene_titles[gene], fontsize=12, fontweight='bold')
                continue
            
            # Step 1: どんな値からでも確実に数値部分を抽出する関数を定義
            def get_numeric_part(value):
                match = re.search(r'\d+', str(value))
                return int(match.group()) if match else float('inf')
            
            # Step 2: まず、DataFrameに存在するユニークなクラスタ名を取得
            unique_clusters = plot_data['predicted_microenvironment'].unique()
            
            # Step 3: それらのユニークなクラスタ名を「数値として」正しく並べ替える
            # これにより、ソートの基準となるリストが確定します (例: ['1', '3', '11', 'Tumor'])
            sorted_unique_clusters = sorted(unique_clusters, key=get_numeric_part)
            
            # Step 4: 確定した基準リストを元に、最終的なプロット用ラベルの「正しい順序」を作成
            # (例: ['ME 1', 'ME 3', 'ME 11', 'Tumor'])
            final_category_order = []
            for cluster in sorted_unique_clusters:
                num = get_numeric_part(cluster)
                if num != float('inf'):
                    final_category_order.append(f"ME {num}")
                else:
                    final_category_order.append(str(cluster)) # 数字でないものはそのまま
            
            # Step 5: 元の値を新しいラベルに変換し、同時に順序付きカテゴリ型に変換して順序を強制
            plot_data['predicted_microenvironment'] = pd.Categorical(
                plot_data['predicted_microenvironment'].apply(lambda x: f"ME {get_numeric_part(x)}" if get_numeric_part(x) != float('inf') else str(x)),
                categories=final_category_order,
                ordered=True
            )
            
            # Step 6: 最後に、定義したカテゴリ順でDataFrame全体を並べ替える
            plot_data = plot_data.sort_values(by="predicted_microenvironment")
            # Create bar plot
            try:
                bars = ax.bar(
                    x=plot_data['predicted_microenvironment'],
                    height=plot_data['mean_expression'],
                    yerr=plot_data['std_expression'] if show_std else None,
                    capsize=5 if show_std else None,
                    color=sns.color_palette(palette, len(plot_data)),
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5
                )

                if not plot_data.empty:
                    max_y_val = (plot_data['mean_expression'] + (plot_data['std_expression'] if show_std else 0)).max()
                    ax.set_ylim(top=max_y_val * 1.20)

                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

            except Exception as e:
                print(f"Error plotting {gene}: {e}")
                ax.text(0.5, 0.5, f'Error: {gene}', ha='center', va='center', transform=ax.transAxes)
            
            # Styling
            ax.set_title(gene_titles[gene], fontsize=12, fontweight='bold')
            ax.set_xlabel('Microenvironment' if i >= (nrows - 1) * ncols else '')
            ax.set_ylabel('Mean Expression')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.tick_params(axis='x', labelrotation=45, labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
        
        for j in range(n_plots, len(axes)):
            fig.delaxes(axes[j])
            
        plt.suptitle(f"Expression across Cell Types: {', '.join(valid_cell_types)}", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
        
        print(f"Analysis complete: {len(valid_cell_types)} cell types, {len(valid_genes)} genes")
        print(f"Value: average expression count per 2x2 µm bin")
        print(f"Total cells analyzed: {filtered_adata.n_obs}")
    
    def create_interactive_widget(self):
        """Create interactive widget interface"""
        
        if self.processed_data is None:
            self._prepare_data()
            
        sp_adata = self.processed_data
        available_cell_types = np.unique(sp_adata.obs["predicted_cell_type"])
        available_genes = sp_adata.var.index.tolist()
        
        style = {'description_width': 'initial'}
        
        cell_type_widget = widgets.SelectMultiple(
            options=available_cell_types,
            value=[available_cell_types[0]] if len(available_cell_types) > 0 else [],
            description='Cell Types:',
            style=style,
            rows=10,
            layout=widgets.Layout(width='250px')
        )
        
        gene_search = widgets.Text(
            value='',
            placeholder='Type to search genes...',
            description='Gene Search:',
            style=style,
            layout=widgets.Layout(width='250px')
        )

        gene_widget = widgets.SelectMultiple(
            options=available_genes[:100],
            value=[],
            description='Search Results:',
            style=style,
            rows=8,
            layout=widgets.Layout(width='250px')
        )

        saved_genes_widget = widgets.SelectMultiple(
            options=[],
            value=[],
            description='Saved Genes:',
            style=style,
            rows=8,
            layout=widgets.Layout(width='250px')
        )

        # *** MODIFIED: Button descriptions are now in English ***
        save_button = widgets.Button(description='▶ Save', tooltip='Add selected genes to the saved list', layout=widgets.Layout(width='90px'))
        remove_button = widgets.Button(description='◀ Remove', tooltip='Remove genes from the saved list', layout=widgets.Layout(width='90px'))
        
        show_std_widget = widgets.Checkbox(value=False, description='Show error bars', style=style)
        sum_genes_widget = widgets.Checkbox(value=False, description='Sum genes into one plot', style=style)
        
        update_button = widgets.Button(
            description='📊 Generate Plot',
            button_style='success',
            layout=widgets.Layout(width='180px', height='40px'),
            icon='bar-chart'
        )
        
        output_area = widgets.Output()
        
        def update_gene_options(change):
            if self._is_updating_genes:
                return
            try:
                self._is_updating_genes = True
                search_term = change['new'].upper()
                current_selection = set(gene_widget.value)
                
                if len(search_term) >= 2:
                    # Filter out genes that are already in the saved list
                    filtered_genes = {g for g in available_genes if search_term in g.upper() and g not in saved_genes_widget.options}
                else:
                    # Show initial list, excluding already saved genes
                    filtered_genes = {g for g in available_genes[:100] if g not in saved_genes_widget.options}
                
                combined_options = sorted(list(filtered_genes | current_selection))
                
                gene_widget.options = combined_options
                gene_widget.value = list(current_selection)
            finally:
                self._is_updating_genes = False

        gene_search.observe(update_gene_options, names='value')

        def on_save_clicked(b):
            selected_to_save = set(gene_widget.value)
            current_saved = set(saved_genes_widget.options)
            new_saved = sorted(list(current_saved | selected_to_save))
            
            saved_genes_widget.options = new_saved
            # *** MODIFIED: Automatically select all genes in the saved list ***
            saved_genes_widget.value = new_saved
            
            gene_widget.value = [] 
            update_gene_options({'new': gene_search.value})

        def on_remove_clicked(b):
            selected_to_remove = set(saved_genes_widget.value)
            current_saved = set(saved_genes_widget.options)
            new_saved = sorted(list(current_saved - selected_to_remove))

            saved_genes_widget.options = new_saved
            # *** MODIFIED: Automatically select all remaining genes ***
            saved_genes_widget.value = new_saved
            
            update_gene_options({'new': gene_search.value})
        
        save_button.on_click(on_save_clicked)
        remove_button.on_click(on_remove_clicked)

        def on_update_clicked(b):
            update_button.disabled = True
            update_button.description = '⏳ Generating...'
            
            with output_area:
                clear_output(wait=True)
                selected_cell_types = list(cell_type_widget.value)
                selected_genes = list(saved_genes_widget.value)
                
                if not selected_cell_types:
                    print("⚠️ Please select at least one cell type.")
                elif not selected_genes:
                    print("⚠️ Please select at least one gene and add it to the 'Saved Genes' list.")
                else:
                    print(f"🔍 Analyzing {len(selected_cell_types)} cell types × {len(selected_genes)} genes")
                    try:
                        is_summing = sum_genes_widget.value
                        num_plots = 1 if is_summing and len(selected_genes) > 1 else len(selected_genes)
                        fig_width = min(15, 5 * num_plots)

                        self.plot_expression_bars_combined(
                            cell_types=selected_cell_types,
                            genes=selected_genes,
                            palette='tab10',
                            show_std=show_std_widget.value,
                            sum_genes=is_summing,
                            figsize=(fig_width, 6)
                        )
                    except Exception as e:
                        print(f"❌ Error generating plot: {e}")
            
            update_button.disabled = False
            update_button.description = '📊 Generate Plot'
        
        update_button.on_click(on_update_clicked)
        
        # Layout definition
        gene_buttons = widgets.VBox([save_button, remove_button], layout=widgets.Layout(justify_content='center'))
        gene_selector = widgets.HBox([
            widgets.VBox([gene_search, gene_widget]),
            gene_buttons,
            saved_genes_widget
        ])

        controls = widgets.VBox([
            widgets.HTML("<h3>🎯 Spatial Gene Expression Analyzer</h3>"),
            widgets.HBox([
                widgets.VBox([widgets.HTML("<b><font size='3'>1. Select Cell Types</font></b>"), cell_type_widget]),
                widgets.VBox([widgets.HTML("<b><font size='3'>2. Find & Save Genes</font></b>"), gene_selector])
            ]),
            widgets.HTML("<b><font size='3'>3. Plot Options</font></b>", layout=widgets.Layout(margin='10px 0 0 0')),
            widgets.HBox([show_std_widget, sum_genes_widget]),
            widgets.Box([update_button], layout=widgets.Layout(display='flex', justify_content='center', margin='15px 0 0 0'))
        ], layout=widgets.Layout(border='1px solid #ccc', padding='15px', border_radius='5px'))
        
        display(controls)
        display(output_area)
        
        return controls, output_area

# Convenience functions (no changes needed)
def plot_spatial_expression(sp_adata_raw, 
                            sp_adata_microenvironment,
                            cell_types: Union[str, List[str]], 
                            genes: Union[str, List[str]],
                            mask_coords: Optional[Tuple[int, int, int, int]] = None,
                            **kwargs):
    visualizer = SpatialExpressionVisualizer(sp_adata_raw, sp_adata_microenvironment, mask_coords)
    return visualizer.plot_expression_bars_combined(cell_types, genes, **kwargs)

def create_spatial_widget(sp_adata_raw, 
                          sp_adata_microenvironment,
                          mask_coords: Optional[Tuple[int, int, int, int]] = None):
    visualizer = SpatialExpressionVisualizer(sp_adata_raw, sp_adata_microenvironment, mask_coords)
    visualizer.create_interactive_widget()
    return None

def plot_deg_by_microenvironment(
    sp_adata_raw: ad.AnnData,
    sp_adata_microenvironment: ad.AnnData,
    target_cell_type: str,
    mask_coords: Optional[Tuple[int, int, int, int]] = None,
    min_cells_per_group: int = 10,
    n_genes: int = 10,
    save: bool = True,
    save_path_for_today: str = None
) -> Optional[pd.DataFrame]:
    """
    指定された細胞タイプについて、微小環境間の発現差遺伝子(DEG)を計算し可視化する。

    この関数は、bin countによる正規化、対数変換、DEG計算（Wilcoxon検定）を行い、
    結果をヒートマップとドットプロットで表示します。

    Parameters
    ----------
    sp_adata_raw : ad.AnnData
        生の空間的遺伝子発現データ。`obs`に'bin_count'を含む必要があります。
    sp_adata_microenvironment : ad.AnnData
        細胞タイプと微小環境のアノテーションが付与されたデータ。
    target_cell_type : str
        解析対象とする細胞タイプの名前。
    mask_coords : Optional[Tuple[int, int, int, int]], optional
        空間的なフィルタリングを行うための座標 (x1, x2, y1, y2), by default None
    min_cells_per_group : int, optional
        DEG解析に含めるための微小環境グループあたりの最小細胞数, by default 10
    n_genes : int, optional
        プロットに表示するトップ遺伝子の数, by default 10
    save : bool, optional
        プロットをPDFファイルとして保存するかどうか, by default True

    Returns
    -------
    Optional[pd.DataFrame]
        DEG解析結果のトップ遺伝子をまとめたDataFrame。エラー時はNoneを返します。
    """
    plt.close('all')
    print(f"--- Starting DEG analysis for cell type: {target_cell_type} ---")

    # --- 1. データの前処理とフィルタリング ---
    sp_adata = sp_adata_raw.copy()
    if mask_coords:
        print("Applying spatial mask...")
        x1, x2, y1, y2 = mask_coords
        cell_mask = ((sp_adata.obs['array_row'] >= x1) & (sp_adata.obs['array_row'] <= x2) &
                     (sp_adata.obs['array_col'] >= y1) & (sp_adata.obs['array_col'] <= y2))
        sp_adata = sp_adata[cell_mask, :].copy()

    # bin_countによる正規化
    print("Normalizing by bin count...")
    X = sp_adata.X.toarray() if hasattr(sp_adata.X, "toarray") else sp_adata.X
    bin_counts = sp_adata.obs['bin_count'].values.astype(float)
    bin_counts[bin_counts == 0] = 1e-9
    sp_adata.layers['binned_normalized'] = X / bin_counts.reshape(-1, 1)

    # アノテーション情報のマージ
    common_cells = sp_adata_microenvironment.obs_names.intersection(sp_adata.obs_names)
    sp_adata = sp_adata[common_cells, :].copy()
    sp_adata.obs["predicted_cell_type"] = sp_adata_microenvironment.obs.loc[common_cells, "predicted_cell_type"]
    sp_adata.obs["predicted_microenvironment"] = sp_adata_microenvironment.obs.loc[common_cells, "predicted_microenvironment"]

    # 対象細胞タイプでフィルタリング
    sp_adata_filtered = sp_adata[sp_adata.obs['predicted_cell_type'] == target_cell_type].copy()

    if sp_adata_filtered.n_obs == 0:
        print(f"❌ Error: No cells remain after filtering by '{target_cell_type}'.")
        return None
    
    print(f"Found {sp_adata_filtered.n_obs} cells for '{target_cell_type}'.")
    sc.pp.log1p(sp_adata_filtered) # 対数変換

    # 細胞数が少ない微小環境グループを除外
    group_counts = sp_adata_filtered.obs['predicted_microenvironment'].value_counts()
    valid_groups = group_counts[group_counts >= min_cells_per_group].index.tolist()

    if len(valid_groups) < 2:
        print(f"❌ Error: Less than 2 microenvironment groups with at least {min_cells_per_group} cells.")
        print(f"Valid groups found: {valid_groups}")
        return None

    sp_adata_filtered = sp_adata_filtered[sp_adata_filtered.obs['predicted_microenvironment'].isin(valid_groups)].copy()
    print(f"Analyzing valid microenvironments: {valid_groups}")

    # --- 2. 発現差遺伝子（DEG）の計算 ---
    print("Running rank_genes_groups for differential expression analysis...")
    key_added = f'rank_genes_groups_microenvironment_{target_cell_type}'
    sc.tl.rank_genes_groups(
        sp_adata_filtered,
        groupby='predicted_microenvironment',
        method='wilcoxon',
        use_raw=False,
        layer='binned_normalized',
        key_added=key_added
    )

    # --- 3. 結果の抽出と可視化 ---
    print("Extracting results and generating plots...")
    
    # 結果をDataFrameにまとめる
    result = sp_adata_filtered.uns[key_added]
    all_gene_data = []
    for group in result['names'].dtype.names:
        for i in range(n_genes):
            all_gene_data.append({
                'microenvironment': group,
                'gene_name': result['names'][group][i],
                'score': result['scores'][group][i],
                'pvals_adj': result['pvals_adj'][group][i]
            })
    gene_rank_df = pd.DataFrame(all_gene_data)

    # ヒートマップ
    sc.settings.figdir = save_path_for_today
    print("Generating heatmap...")
    heatmap_savename = f'_{target_cell_type}_microenvironment_genes_heatmap.pdf' if save else None
    sc.pl.rank_genes_groups_heatmap(
        sp_adata_filtered,
        groupby='predicted_microenvironment',
        key=key_added,
        n_genes=n_genes,
        min_logfoldchange=0.5,
        show_gene_labels=True,
        use_raw=False,
        cmap='viridis',
        save=heatmap_savename,
        show=True # show()は最後にまとめて呼び出す
    )

    # ドットプロット
    print("Generating dotplot...")
    dotplot_savename = f'_{target_cell_type}_microenvironment_genes_dotplot.pdf' if save else None
    sc.pl.rank_genes_groups_dotplot(
        sp_adata_filtered,
        groupby='predicted_microenvironment',
        key=key_added,
        n_genes=max(5, n_genes // 2), # ドットプロットは少し遺伝子を減らす
        min_logfoldchange=0.5,
        standard_scale='var',
        save=dotplot_savename,
        show=True
    )
    
    plt.show()
    print("\n✅ Analysis complete.")
    
    return gene_rank_df

# --- 関数定義 ---
def create_report_plots(
    merged: pd.DataFrame,
    sp_adata_microenvironment: ad.AnnData,
    hires_img: np.ndarray,
    w: int,
    h: int,
    sample_name: str,
    save_path: str
) -> None:
    """
    提供されたデータから空間散布図と構成比率ヒートマップを作成し、保存します。

    Parameters
    ----------
    merged : pd.DataFrame
        'x', 'y', 'predicted_microenvironment'列を含むDataFrame。
    sp_adata_microenvironment : ad.AnnData
        `.obs`に'predicted_microenvironment'と'predicted_cell_type'を含むAnnDataオブジェクト。
    hires_img : np.ndarray
        散布図の背景となる高解像度画像。
    w : int
        画像の幅。
    h : int
        画像の高さ。
    sample_name : str
        保存ファイル名に使用するサンプル名。
    save_path : str
        プロットを保存するディレクトリパス。
    """
    print("--- Generating plots... ---")
    
    # === 1. 空間散布図の作成 ===
    group_order = sorted(merged["predicted_microenvironment"].dropna().unique())

    # マーカー形状リスト
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X', 'P', 'H', '8', 'd', '|']
    # カラーパレット
    palette = sns.color_palette("tab20", n_colors=len(group_order))

    # 色とマーカーの辞書作成
    color_map = dict(zip(group_order, palette))
    marker_cycle = cycle(markers)
    marker_map = {group: next(marker_cycle) for group in group_order}

    # 描画
    fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=300)
    ax1.imshow(hires_img, extent=[0, w, h, 0])

    for group in group_order:
        data_sub = merged[merged["predicted_microenvironment"] == group]
        ax1.scatter(
            data_sub["x"], data_sub["y"],
            c=[color_map[group]], marker=marker_map[group],
            s=0.5, alpha=0.5, label=group,
            linewidths=0, rasterized=True
        )

    ax1.invert_yaxis()
    ax1.set_axis_off()

    ax1.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", title="Microenvironment clustering",
        markerscale=8, frameon=False, fontsize=6
    )

    # 保存
    filename1 = sample_name + "_overlay_hires_by_microenvironment.pdf"
    out_pdf1 = os.path.join(save_path, filename1)
    fig1.savefig(out_pdf1, format="pdf", dpi=1000, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved scatter plot to {out_pdf1}")

    # === 2. 構成比率ヒートマップの作成 ===
    df = sp_adata_microenvironment.obs

    # クロス集計して割合を計算
    cross_tab = pd.crosstab(df['predicted_microenvironment'], df['predicted_cell_type'])
    proportions = cross_tab.div(cross_tab.sum(axis=1), axis=0)  # 行ごとに正規化

    # 描画
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    im = ax2.imshow(proportions.values, aspect='auto', cmap='viridis')

    # 軸ラベル
    ax2.set_xticks(np.arange(proportions.shape[1]))
    ax2.set_xticklabels(proportions.columns, rotation=45, ha='right')
    ax2.set_yticks(np.arange(proportions.shape[0]))
    ax2.set_yticklabels(proportions.index)

    # カラーバー
    cbar = ax2.figure.colorbar(im, ax=ax2)
    cbar.set_label('Fraction of cell type')

    # セル中央に%表示
    for i in range(proportions.shape[0]):
        for j in range(proportions.shape[1]):
            text = f"{proportions.values[i, j]*100:.1f}%"
            ax2.text(j, i, text, ha='center', va='center', color='white' if proportions.values[i, j] < 0.5 else 'black', fontsize=8)
    
    plt.grid(False)
    ax2.set_xlabel('Predicted Cell Type')
    ax2.set_ylabel('Predicted Microenvironment')
    ax2.set_title('Fraction of Predicted Cell Type per Microenvironment')
    plt.tight_layout()

    # 保存
    filename2 = sample_name + "_celltype_and_microenvironment.pdf"
    out_pdf2 = os.path.join(save_path, filename2)
    fig2.savefig(out_pdf2, format="pdf", dpi=1000, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved heatmap to {out_pdf2}")
    print("--- Plotting complete. ---")

def interactive_gene_histogram(
    adata: ad.AnnData,
    gene_list: List[str]
) -> None:
    """
    遺伝子を検索して選択し、その発現量のヒストグラムをインタラクティブに表示するウィジェットを作成します。
    (%matplotlib widget バックエンドに対応)
    """
    
    # 1. 最初にプロットの「枠」（FigureとAxes）を一度だけ作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # 2. ウィジェットを定義
    search_box = widgets.Text(
        value='',
        placeholder='Type characters to search...',
        description='Search Gene:',
        style={'description_width': 'initial'},
        layout={'width': '400px'}
    )
    
    # 改善：最初に候補を表示しておく
    initial_options = gene_list[:100]
    results_list = widgets.Select(
        options=initial_options,
        value=initial_options[0] if initial_options else None,
        description='Results:',
        rows=8,
        style={'description_width': 'initial'},
        layout={'width': '400px'}
    )
    
    # 3. プロットの中身を「更新」するための関数
    def _update_plot(gene: str):
        ax1.clear()
        ax2.clear()
        
        if not gene or gene not in adata.var_names:
            fig.suptitle("Select a gene from the list", y=1.05)
            ax1.axis("off")
            ax2.axis("off")
        else:
            fig.suptitle('')
            idx = adata.var_names.get_loc(gene)
            expr_all = adata.X[:, idx].toarray().flatten() if hasattr(adata.X, "tocsc") else adata.X[:, idx].flatten()
            
            # (左側のプロット)
            mean_val_all, std_val_all = np.mean(expr_all), np.std(expr_all)
            ax1.hist(expr_all, bins=100, log=True, color="steelblue", edgecolor="black")
            ax1.set_title(f"All Expression for {gene}\nMean={mean_val_all:.2f}, SD={std_val_all:.2f}")
            ax1.set_xlabel("Expression Level")
            ax1.set_ylabel("Cell Count (log scale)")
            ax1.grid(True, which="both", linestyle="--", alpha=0.4)

            # (右側のプロット)
            expr_nonzero = expr_all[expr_all > 0]
            if len(expr_nonzero) == 0:
                ax2.set_title(f"{gene}\nNo non-zero values")
                ax2.axis("off")
            else:
                mean_val_nonzero, std_val_nonzero = np.mean(expr_nonzero), np.std(expr_nonzero)
                ax2.hist(expr_nonzero, bins=100, log=True, color="coral", edgecolor="black")
                ax2.set_title(f"Non-Zero Expression for {gene}\nMean={mean_val_nonzero:.2f}, SD={std_val_nonzero:.2f}")
                ax2.set_xlabel("Expression Level (>0)")
                ax2.set_ylabel("Cell Count (log scale)")
                ax2.grid(True, which="both", linestyle="--", alpha=0.4)
        
        # ▼▼▼【重要】この行でプロットの再描画を指示します ▼▼▼
        fig.canvas.draw_idle()

    # 4. ウィジェット間の連携ロジック
    def on_search_change(change):
        search_term = change['new'].lower()
        if len(search_term) >= 1:
            filtered_genes = [g for g in gene_list if search_term in g.lower()][:100]
            results_list.options = filtered_genes
        elif len(search_term) == 0:
            results_list.options = gene_list[:100] # 空になったら先頭候補に戻す
        else:
            results_list.options = []

    def on_selection_change(change):
        if change['new']:
            _update_plot(change['new'])

    search_box.observe(on_search_change, names='value')
    results_list.observe(on_selection_change, names='value')
    
    ui = widgets.VBox([search_box, results_list])
    
    # ウィジェットとプロットを表示
    display(ui)
    
    # 最初のプロットを描画
    if results_list.value:
        _update_plot(results_list.value)

    
def interactive_cci_sankey(
    coexp_cc_df,
    edge_df,
    bargraph_df,
    sp_adata,
    lib_id,
    SAMPLE_NAME,
    save_path_for_today,
    coexp_cc_df_cluster,
    bargraph_df_cluster,
    save=True,
    save_format="html",
    # Default values for the new widgets
    # 新しいウィジェットのデフォルト値
    initial_significant_column="is_significant",
    initial_minimum_interaction=10,
    initial_display_column="interaction_positive"
):
    """
    Generates an interactive widget to visualize CCI and Sankey plots.
    CCIとサンキープロットを可視化するための対話型ウィジェットを生成します。
    """

    # --- Widget Setup ---
    # Prepare the lists for widget options
    # ウィジェットの選択肢リストを準備
    try:
        gene_list = sorted(coexp_cc_df['ligand'].unique())
        sender_cell_type_list = sorted(coexp_cc_df['cell1_type'].unique())
        target_cell_type_list = sorted(coexp_cc_df['cell2_type'].unique())
        cluster_list = sorted(edge_df['cell1_cluster'].astype(str).unique())

        initial_gene = coexp_cc_df['ligand'].mode()[0]
        initial_target_cell = coexp_cc_df['cell2_type'].mode()[0]
        initial_senders = sender_cell_type_list
        initial_cluster = [str(edge_df['cell1_cluster'].mode()[0])]

    except (KeyError, IndexError) as e:
        print(f"Error initializing widgets: Could not find required columns or data is empty. Details: {e}")
        return

    # Create the dictionary of widgets with updated labels and new additions
    # 更新されたラベルと新しいウィジェットを含む辞書を作成
    w = {
        'gene_to_analyze': widgets.Dropdown(
            options=gene_list, value=initial_gene, description='Ligand:',
            style={'description_width': 'initial'}
        ),
        'target_cell_type': widgets.Dropdown(
            options=target_cell_type_list, value=initial_target_cell, description='Target Cell:',
            style={'description_width': 'initial'}
        ),
        'sender_cell_type': widgets.SelectMultiple(
            options=sender_cell_type_list, value=initial_senders, description='Sender Cells:',
            rows=8, style={'description_width': 'initial'}
        ),
        'target_clusters': widgets.SelectMultiple(
            options=cluster_list, value=initial_cluster, description='Target Clusters:',
            rows=8, style={'description_width': 'initial'}
        ),
        'each_display_num': widgets.IntSlider(
            min=1, max=30, step=1, value=5, description='Ligands in sankey plot:',
            style={'description_width': 'initial'}
        ),
        'significant_column': widgets.Dropdown(
            options=["is_significant", "is_significant_bonferroni"], value=initial_significant_column,
            description='Significance:', style={'description_width': 'initial'}
        ),
        'minimum_interaction': widgets.IntSlider(
            min=1, max=100, step=1, value=initial_minimum_interaction,
            description='Min. Interactions:', style={'description_width': 'initial'}
        ),
        'display_column': widgets.Dropdown(
            options=[
                ('Positive interaction count', 'interaction_positive'),
                ('Positive interaction per positive ligand signal', 'coactivity_per_sender_cell_expr_ligand')
            ],
            value=initial_display_column, description='Display Value:',
            style={'description_width': 'initial'}, layout={'width': 'max-content'}
        ),
    }

    # --- Plotting Function ---
    def plot_ui(
        gene_to_analyze, target_cell_type, sender_cell_type, target_clusters,
        each_display_num, significant_column, minimum_interaction, display_column
    ):
        if not target_cell_type or not gene_to_analyze or not sender_cell_type or not target_clusters:
            print("Please select at least one sender cell type and one target cluster.")
            return
        
        try:
            print("--- Generating plot with new parameters ---")
            
            # Call the original plotting function with interactively selected values
            # 対話的に選択された値で元のプロット関数を呼び出す
            plot_gene_cci_and_sankey(
                target_cell_type=target_cell_type,
                sender_cell_type=list(sender_cell_type),
                Gene_to_analyze=gene_to_analyze,
                each_display_num=each_display_num,
                bargraph_df=bargraph_df,
                edge_df=edge_df,
                cluster_cells=sp_adata,
                coexp_cc_df=coexp_cc_df,
                lib_id=lib_id,
                save=save,
                SAMPLE_NAME=SAMPLE_NAME,
                save_path_for_today=save_path_for_today,
                target_clusters=list(target_clusters),
                coexp_cc_df_cluster=coexp_cc_df_cluster,
                bargraph_df_cluster=bargraph_df_cluster,
                save_format=save_format,
                significant_column=significant_column,
                minimum_interaction=minimum_interaction,
                display_column=display_column
            )
        except Exception as e:
            print(f"An error occurred during plotting: {e}")

    # --- Layout and Display ---
    out = widgets.interactive_output(plot_ui, w)

    # Arrange controls for a better layout
    # より良いレイアウトのためにコントロールを配置
    controls_selection = widgets.HBox([w['sender_cell_type'], w['target_clusters']], layout=widgets.Layout(justify_content='space-around'))
    controls_settings1 = widgets.HBox([w['gene_to_analyze'], w['target_cell_type']], layout=widgets.Layout(justify_content='space-around'))
    controls_settings2 = widgets.VBox([
        w['each_display_num'], w['minimum_interaction'], w['significant_column'], w['display_column']
    ])
    
    # Display the final widget interface
    # 最終的なウィジェットインターフェースを表示
    display(widgets.VBox([
        controls_settings1,
        controls_selection,
        controls_settings2
    ]), out)
