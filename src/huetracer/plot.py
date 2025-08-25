import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import os
import plotly.graph_objects as go
from collections import defaultdict, Counter

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
    # Use cluster-specific data if provided
    if coexp_cc_df_cluster is not None and bargraph_df_cluster is not None:
        coexp_data_for_cluster = coexp_cc_df_cluster
        bargraph_data_for_cluster = bargraph_df_cluster
    else:
        coexp_data_for_cluster = coexp_cc_df
        bargraph_data_for_cluster = bargraph_df
    
    # FIXED: 正しいクラスターフィルタリングロジック
    target_cell_mask = cluster_cells.obs['celltype'] == target_cell_type
    target_cluster_mask = cluster_cells.obs['cluster'].astype(str).isin(target_clusters_str)
    combined_mask = target_cell_mask & target_cluster_mask
    target_cluster_cells = cluster_cells.obs_names[combined_mask]
    
    if len(target_cluster_cells) == 0:
        print(f"Warning: No {target_cell_type} cells found in target clusters {target_clusters}")
        return
    
    # FIXED: edge_dfの正しいフィルタリング
    # edge_dfにcell1_clusterカラムがない可能性が高いため、直接cell1でフィルタ
    target_edge_mask = edge_df['cell1'].isin(target_cluster_cells)
    filtered_edge_df_target = edge_df[target_edge_mask]
    filtered_bargraph_df_target = bargraph_data_for_cluster[target_edge_mask]
    
    if len(filtered_edge_df_target) == 0:
        print(f"Warning: No interactions found for {target_cell_type} in target clusters")
        return
    
    # FIXED: target cluster用のcoexp_cc_dfを再計算
    # 元のcoexp_cc_dfから対象細胞タイプのデータをフィルタして、
    # 実際にtarget clusterに存在する相互作用のみを抽出
    
    # まず、target clusterの細胞が関与する相互作用を特定
    target_cell_interactions = set()
    
    # filtered_edge_df_targetから実際に存在するcell2_typeを取得
    actual_cell2_types = set(filtered_edge_df_target['cell2_type'].unique())
    
    # クラスター特有のcoexp_cc_dfから、target_cell_typeがcell1_typeで、
    # かつcell2_typeが実際にtarget clusterに存在するもののみを抽出
    target_coexp_mask = (
        (coexp_data_for_cluster['cell1_type'] == target_cell_type) &
        (coexp_data_for_cluster['cell2_type'].isin(actual_cell2_types))
    )
    
    sub_coexp_cc_df_target = coexp_data_for_cluster[target_coexp_mask].copy()
    
    if significant_column in sub_coexp_cc_df_target.columns:
        sig_mask = sub_coexp_cc_df_target[significant_column] == True
        sub_coexp_cc_df_target = sub_coexp_cc_df_target[sig_mask]
    
    if len(sub_coexp_cc_df_target) == 0:
        print(f"Warning: No significant interactions found for {target_cell_type} in target clusters")
        return

    if 'interaction_positive' in sub_coexp_cc_df_target.columns:
        interaction_filter = sub_coexp_cc_df_target['interaction_positive'] >= minimum_interaction
        sub_coexp_cc_df_target = sub_coexp_cc_df_target[interaction_filter]
        print(f"interaction_positive >= {minimum_interaction} filtering: {len(sub_coexp_cc_df_target)} interactions remain")

        interaction_filter = sub_coexp_cc_df_target['cell2_type'].isin(sender_cell_type)
        sub_coexp_cc_df_target = sub_coexp_cc_df_target[interaction_filter]
        print(f"sender filtering: {len(sub_coexp_cc_df_target)} interactions remain")

    
    # 上位相互作用を選択
    sub_coexp_cc_df_target = sub_coexp_cc_df_target.sort_values(
        display_column, ascending=False
    ).groupby('cell2_type', as_index=False).head(n=each_display_num)
    
    # Sankeyダイアグラムの作成（ターゲットクラスター）
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
        # Choose save format based on parameter
        if save_format == "html":
            filename = f"{SAMPLE_NAME}_{target_cell_type}_sankey_target_clusters.html"
            out_file = os.path.join(save_path_for_today, filename)
            fig5.write_html(out_file)
            print(f"Saved HTML: {filename}")
        elif save_format == "png":
            filename = f"{SAMPLE_NAME}_{target_cell_type}_sankey_target_clusters.png"
            out_file = os.path.join(save_path_for_today, filename)
            fig5.write_image(out_file, format="png", width=600, height=1000, scale=2)
            print(f"Saved PNG: {filename}")
        elif save_format == "both":
            # HTML (fast)
            filename_html = f"{SAMPLE_NAME}_{target_cell_type}_sankey_target_clusters.html"
            out_html = os.path.join(save_path_for_today, filename_html)
            fig5.write_html(out_html)
            print(f"Saved HTML: {filename_html}")
            
            # PNG (medium speed)
            try:
                filename_png = f"{SAMPLE_NAME}_{target_cell_type}_sankey_target_clusters.png"
                out_png = os.path.join(save_path_for_today, filename_png)
                fig5.write_image(out_png, format="png", width=600, height=1000, scale=2)
                print(f"Saved PNG: {filename_png}")
            except Exception as e:
                print(f"PNG save failed: {e}")
        else:  # pdf (slow - not recommended)
            filename = f"{SAMPLE_NAME}_{target_cell_type}_sankey_target_clusters.pdf"
            out_file = os.path.join(save_path_for_today, filename)
            print(f"Warning: PDF save is slow. Consider using save_format='html' or 'png'")
            fig5.write_image(out_file, format="pdf", width=600, height=1000)
            print(f"Saved PDF: {filename}")

    fig5.show()
    
    print(f"Successfully generated NumPy-optimized plots for {target_cell_type} - {Gene_to_analyze}")
    print("Two Sankey diagrams created:")
    print("1. All clusters")
    print(f"2. Target clusters: {target_clusters}")
    print("All plots should now be displayed above.")

    
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
