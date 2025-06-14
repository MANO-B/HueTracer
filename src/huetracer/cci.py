import pandas as pd
import numpy as np
import scipy
import scanpy as sc


def make_coexp_cc_df(ligand_adata, edge_df, role):
    sender = edge_df.cell1 if role == "sender" else edge_df.cell2
    receiver = edge_df.cell2 if role == "sender" else edge_df.cell1
    coexp_df = pd.DataFrame(
        ligand_adata[sender].X *
        ligand_adata[receiver].layers['activity'],
        columns=ligand_adata.var_names, index=edge_df.index
    )
    coexp_df['cell2_type'] = edge_df['cell2_type']
    coexp_df['cell1_type'] = edge_df['cell1_type']
    coexp_cc_df = coexp_df.groupby(['cell2_type', 'cell1_type']).sum()
    coexp_cc_df = coexp_cc_df.reset_index().melt(id_vars=['cell1_type', 'cell2_type'], var_name='ligand', value_name='coactivity')
    return coexp_cc_df

def make_non_zero_values(mat):
    top_mat = mat > 0
    return(top_mat)

def make_positive_values(mat):
    mat[mat < 0] = 0
    return(mat)
    
def make_top_values(mat, top_fraction = 0.1, axis=0):
    top_mat = mat > np.quantile(mat, 1 - top_fraction, axis=axis, keepdims=True)
    return(top_mat)

def safe_toarray(x):
    if type(x) != np.ndarray:
        return x.toarray()
    else:
        return x

def add_zscore_layers(sp_adata, top_fraction=0.1):
    # 密なゼロ行列を準備
    shape = sp_adata.shape
    zero_matrix = np.zeros(shape)

    sp_adata.layers["zscore_by_celltype"] = zero_matrix.copy()
    sp_adata.layers["zscore_all_celltype"] = zero_matrix.copy()
    sp_adata.X = sp_adata.X.tocsr()
    if scipy.sparse.issparse(sp_adata.X):
        X_dense = sp_adata.X.toarray()
    else:
        X_dense = sp_adata.X.copy()
    result = np.zeros_like(X_dense)

    # celltypeごとの zscore（make_positive_values の後で std で割る）
    for ct in sp_adata.obs["celltype"].unique():
        idx = sp_adata.obs["celltype"] == ct
        X_sub = X_dense[idx,:]
        mean = X_sub.mean(axis=0)
        std = np.array([
            np.std(gene_expr[gene_expr != 0]) if np.any(gene_expr != 0) else 0
            for gene_expr in X_sub.T
        ])
        std[std == 0] = 1

        result[idx, :] = X_sub - mean
        sp_adata.layers["zscore_by_celltype"][idx] = make_positive_values(result)

    # 全体の zscore（z は定義済みの X をそのまま）
    X = X_dense
    mean = X.mean(axis=0)
    std = np.array([
        np.std(gene_expr[gene_expr != 0]) if np.any(gene_expr != 0) else 0
        for gene_expr in X.T
    ])
    std[std == 0] = 1

    z = X - mean
    zscore_all = make_positive_values(z)
    zscore_all = make_top_values(zscore_all, axis=0, top_fraction=top_fraction)
    sp_adata.layers["zscore_all_celltype"] = zscore_all

    # zscore_by_celltype に std を適用（遅延適用して正規化）
    sp_adata.layers["zscore_by_celltype"] = sp_adata.layers["zscore_by_celltype"] / std

def prepare_microenv_data(sp_adata_raw, sp_adata_microenvironment, lt_df_raw, min_frac=0.001, n_top_genes=2000):
    # 共通細胞の抽出
    common_cells = sp_adata_microenvironment.obs_names.intersection(sp_adata_raw.obs_names)
    sp_adata = sp_adata_raw[common_cells].copy()
    if scipy.sparse.issparse(sp_adata.X):
        sp_adata.X = sp_adata.X.toarray()

    # ノーマライズ：bin_countで割る（整数カウント前提）
    sp_adata.X = np.round(sp_adata.X).copy() / sp_adata.obs['bin_count'].values[:, None]
    if isinstance(sp_adata.X, scipy.sparse.coo_matrix):
        sp_adata.X = sp_adata.X.tocsr()
    sp_adata.raw = None

    # メタデータの追加
    sp_adata.obs['cluster'] = sp_adata_microenvironment.obs.loc[common_cells, 'predicted_microenvironment']
    sp_adata.obs['celltype'] = sp_adata_microenvironment.obs.loc[common_cells, 'predicted_cell_type']
    sp_adata.obs_names_make_unique()

    # 0.1%未満の細胞でしか発現しない遺伝子を除外
    min_cells = int(np.ceil(sp_adata.n_obs * min_frac))
    sc.pp.filter_genes(sp_adata, min_cells=min_cells)

    # コピーして処理（フィルタ後データを使用）
    filtered_adata = sp_adata.copy()
    filtered_adata.layers["counts"] = filtered_adata.X.copy()
    filtered_adata.raw = filtered_adata.copy()

    sc.pp.normalize_total(filtered_adata, target_sum=1e4)
    sc.pp.log1p(filtered_adata)

    # logreg による細胞型ごとの特徴遺伝子抽出
    sc.tl.rank_genes_groups(filtered_adata, groupby='celltype', method='logreg', n_genes=n_top_genes, use_raw=True)
    top_genes_df = pd.DataFrame(filtered_adata.uns['rank_genes_groups']['names'])
    top_genes_list = top_genes_df.values.flatten().tolist()
    top_genes_list = [g for g in top_genes_list if pd.notnull(g)]
    common_hvg = [g for g in set(top_genes_list) if g in filtered_adata.var_names]

    # Seurat方式による変動遺伝子
    sc.pp.highly_variable_genes(filtered_adata, n_top_genes=n_top_genes, flavor='seurat_v3', layer="counts")
    ref_hvg = filtered_adata.var[filtered_adata.var['highly_variable']].index.tolist()

    # 平均発現での上位遺伝子も抽出
    mean_expression = np.asarray(filtered_adata.layers["counts"].mean(axis=0)).flatten()
    gene_names = filtered_adata.var_names
    top_expr_indices = np.argsort(mean_expression)[-n_top_genes:]
    top_expr_genes = gene_names[top_expr_indices]

    # 遺伝子リストを統合
    all_genes = set(common_hvg) | set(ref_hvg) | set(top_expr_genes)
    final_gene_list = [g for g in all_genes if g in filtered_adata.var_names]

    # アノテーション済みの遺伝子でサブセット
    sp_adata = sp_adata[:, final_gene_list]

    # リガンド・ターゲットの共通遺伝子に限定
    common_genes = pd.Index(np.intersect1d(lt_df_raw.index, sp_adata.var_names))
    lt_df = lt_df_raw.loc[common_genes].copy()
    sp_adata = sp_adata[:, common_genes]
    lt_df = lt_df.loc[:, lt_df.columns.intersection(sp_adata.var_names)]

    # 各リガンド列ごとに正規化（列和が1になるように）
    lt_df = lt_df.div(lt_df.sum(axis=0), axis=1)

    return sp_adata, lt_df
