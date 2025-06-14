import pandas as pd
import numpy as np

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
