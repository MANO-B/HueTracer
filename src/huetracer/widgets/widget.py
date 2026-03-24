import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
from adjustText import adjust_text as at
import os


class SpatialGeneExpressionViewer:
    def __init__(self, sp_adata, lib_id, save_path=None):
        self.sp_adata = sp_adata.copy()
        self.lib_id = lib_id
        self.fig = None 
        self.save_path = save_path

        self.sp_adata.obs['predicted_cell_type'] = self.sp_adata.obs['predicted_cell_type'].astype(str)
        self.sp_adata.obs['predicted_microenvironment'] = self.sp_adata.obs['predicted_microenvironment'].astype(str)
        
        self._init_ui_components()
        
    def _init_ui_components(self):
        """UIウィジェットを作成"""
        cell_types = sorted(self.sp_adata.obs['predicted_cell_type'].unique())
        microenvs = sorted(self.sp_adata.obs['predicted_microenvironment'].unique())
        self.all_genes = sorted(self.sp_adata.var_names.tolist())

        self.cell_type_selector = widgets.SelectMultiple(options=cell_types, description='Cell Types:', layout=widgets.Layout(height='100px', width='auto'))
        self.microenv_selector = widgets.SelectMultiple(options=microenvs, description='Microenvs:', layout=widgets.Layout(height='100px', width='auto'))
        
        self.gene_search = widgets.Text(value='', placeholder='Type to search genes...', description='Search Genes:', style={'description_width': 'initial'})
        self.gene_search_results = widgets.SelectMultiple(options=self.all_genes[:200], description='Search Results:', layout=widgets.Layout(height='150px'))
        self.add_gene_button = widgets.Button(description='Add >', button_style='info', layout=widgets.Layout(width='80px'))
        self.remove_gene_button = widgets.Button(description='< Remove', button_style='warning', layout=widgets.Layout(width='80px'))
        self.gene_selected_list = widgets.SelectMultiple(options=[], description='Selected Genes:', layout=widgets.Layout(height='150px'))

        self.point_size_slider = widgets.FloatSlider(value=15.0, min=0.1, max=30.0, step=0.1, description='Point Size:')
        self.bg_alpha_slider = widgets.FloatSlider(value=0.2, min=0.0, max=1.0, step=0.05, description='Background Alpha:')
        
        self.plot_button = widgets.Button(description='Update Plot', button_style='success', icon='paint-brush')
        self.status_label = widgets.HTML(value="<b>Status:</b> Ready")
        self.output = widgets.Output()

        self.save_format = widgets.Dropdown(options=['png', 'pdf'], value='png', description='Format:', layout=widgets.Layout(width='180px'))
        self.save_button = widgets.Button(description='Save Plot', button_style='primary', icon='save')

        # イベントハンドラ
        self.gene_search.observe(self._on_gene_search_change, names='value')
        self.add_gene_button.on_click(self._on_add_gene_clicked)
        self.remove_gene_button.on_click(self._on_remove_gene_clicked)
        self.plot_button.on_click(self._on_plot_button_clicked)
        self.save_button.on_click(self._on_save_button_clicked)

    def _on_gene_search_change(self, change):
        search_term = change['new'].lower()
        filtered_genes = [g for g in self.all_genes if search_term in g.lower()] if search_term else self.all_genes[:200]
        self.gene_search_results.options = filtered_genes

    def _on_add_gene_clicked(self, b):
        current_genes = set(self.gene_selected_list.options)
        current_genes.update(self.gene_search_results.value)
        self.gene_selected_list.options = sorted(list(current_genes))
    
    def _on_remove_gene_clicked(self, b):
        current_genes = set(self.gene_selected_list.options)
        new_genes = current_genes - set(self.gene_selected_list.value)
        self.gene_selected_list.options = sorted(list(new_genes))

    def _on_plot_button_clicked(self, b):
        with self.output:
            clear_output(wait=True)
            self.status_label.value = "<b>Status:</b> ⏳ Plotting..."
            
            genes = list(self.gene_selected_list.options)
            selected_cts = self.cell_type_selector.value
            selected_mes = self.microenv_selector.value
            
            if not all([selected_cts, selected_mes, genes]):
                self.status_label.value = "<b>Status:</b> ⚠️ Input missing!"; print("⚠️ Please select at least one cell type, microenvironment, and gene.")
                return

            adata_filtered = self.sp_adata[self.sp_adata.obs['predicted_cell_type'].isin(selected_cts) & self.sp_adata.obs['predicted_microenvironment'].isin(selected_mes)].copy()

            if adata_filtered.n_obs == 0:
                self.status_label.value = "<b>Status:</b> No data to plot."; print("No cells found for the selected criteria.")
                return

            gene_expression = self.sp_adata.raw[adata_filtered.obs_names, genes].X.toarray().sum(axis=1) if self.sp_adata.raw else adata_filtered[:, genes].X.toarray().sum(axis=1)
            adata_filtered.obs['gene_expression_sum'] = gene_expression

            # --- 👇 表示遺伝子数を5個に、フォントサイズを10に修正 ---
            if len(genes) > 5:
                title_genes = ", ".join(genes[:5]) + ", ..."
            else:
                title_genes = ", ".join(genes)
            full_title = f"Total Expression of: {title_genes}"
            
            self.fig, ax = plt.subplots(figsize=(8, 8))
            
            sc.pl.spatial(adata_filtered, library_id=self.lib_id, color='gene_expression_sum', title="", size=self.point_size_slider.value / 10, alpha_img=self.bg_alpha_slider.value, ax=ax, show=False, cmap='viridis')
            
            # --- 👇 カラーバーとタイトルの手動調整 ---
            # 1. 既存のカラーバーを見つけて、その位置とサイズを調整
            if len(self.fig.axes) > 1: # カラーバーの軸が存在するか確認
                cax = self.fig.axes[-1]
                pos = cax.get_position()
                # [left, bottom, width, height] bottomを少し上げ、heightを80%に
                cax.set_position([pos.x0, pos.y0 + pos.height * 0.1, pos.width, pos.height * 0.8])

            # 2. タイトルを小さいフォントサイズで設定
            ax.set_title(full_title, fontsize=10)
            # --- 👆 ここまでが修正箇所 ---

            plt.show()
            self.status_label.value = "<b>Status:</b> ✅ Plot updated."

    def _on_save_button_clicked(self, b):
        if self.fig:
            filename = f"spatial_plot.{self.save_format.value}"
            if self.save_path:
                os.makedirs(self.save_path, exist_ok=True)
                full_path = os.path.join(self.save_path, filename)
            else:
                full_path = filename
            
            self.fig.savefig(full_path, dpi=300, bbox_inches='tight')
            self.status_label.value = f"<b>Status:</b> ✅ Plot saved to {full_path}"
        else:
            self.status_label.value = "<b>Status:</b> ⚠️ No plot to save. Please generate a plot first."
    
    def display(self):
        button_layout = widgets.Layout(display='flex', flex_flow='column', align_items='center', justify_content='center')
        buttons = widgets.VBox([self.add_gene_button, self.remove_gene_button], layout=button_layout)
        
        search_panel = widgets.VBox([self.gene_search, self.gene_search_results], layout=widgets.Layout(width='40%'))
        selected_panel = widgets.VBox([self.gene_selected_list], layout=widgets.Layout(width='40%'))
        gene_selection_ui = widgets.HBox([search_panel, buttons, selected_panel])

        save_box = widgets.HBox([self.save_format, self.save_button])

        controls = widgets.VBox([
            widgets.HTML("<h3>Spatial Gene Expression Viewer</h3>"),
            self.cell_type_selector, self.microenv_selector,
            widgets.HTML("<hr><b>Gene Selection</b>"), gene_selection_ui,
            widgets.HTML("<hr><b>Plot Settings</b>"), self.point_size_slider, self.bg_alpha_slider,
            self.plot_button, widgets.HTML("<hr><b>Export</b>"), save_box, self.status_label
        ], layout=widgets.Layout(width='650px'))
        
        main_ui = widgets.HBox([controls, self.output])
        display(main_ui)

class VolcanoPlotter:
    # 👇 save_path引数を追加
    def __init__(self, sp_adata, save_path=None):
        self.sp_adata = sp_adata.copy()
        self.fig = None
        self.save_path = save_path # 保存パスをインスタンス変数として保持
        
        self.sp_adata.obs['predicted_cell_type'] = self.sp_adata.obs['predicted_cell_type'].astype(str)
        self.sp_adata.obs['predicted_microenvironment'] = self.sp_adata.obs['predicted_microenvironment'].astype(str)

        self._init_ui_components()

    def _init_ui_components(self):
        cell_types = sorted(self.sp_adata.obs['predicted_cell_type'].unique())
        microenvs = sorted(self.sp_adata.obs['predicted_microenvironment'].unique())

        self.g1_ct = widgets.SelectMultiple(options=cell_types, description='Cell Types (G1):')
        self.g1_me = widgets.SelectMultiple(options=microenvs, description='Microenvs (G1):')
        self.g2_ct = widgets.SelectMultiple(options=cell_types, description='Cell Types (G2):')
        self.g2_me = widgets.SelectMultiple(options=microenvs, description='Microenvs (G2):')

        common_style = {'description_width': 'initial'}
        self.min_cells_filter = widgets.IntText(value=10, description='Min Cells (Filter):', style=common_style)
        self.log2fc_thresh = widgets.FloatText(value=0.5, description='Log2FC Threshold:', style=common_style)
        self.pval_thresh = widgets.FloatText(value=0.05, description='Adj. P-val Threshold:', style=common_style)
        self.top_n_genes = widgets.IntText(value=20, description='Genes to Label:', style=common_style)
        
        self.run_button = widgets.Button(description='Run DGE Analysis', button_style='primary', icon='cogs')
        self.status_label = widgets.HTML(value="<b>Status:</b> Ready")
        self.output = widgets.Output()

        self.save_format = widgets.Dropdown(options=['png', 'pdf'], value='png', description='Format:', layout=widgets.Layout(width='180px'))
        self.save_button = widgets.Button(description='Save Plot', button_style='primary', icon='save')

        self.run_button.on_click(self._on_run_button_clicked)
        self.save_button.on_click(self._on_save_button_clicked)

    def _on_run_button_clicked(self, b):
        with self.output:
            clear_output(wait=True)
            self.status_label.value = "<b>Status:</b> ⏳ Running analysis..."
            
            try:
                g1_mask = self.sp_adata.obs['predicted_cell_type'].isin(self.g1_ct.value) & self.sp_adata.obs['predicted_microenvironment'].isin(self.g1_me.value)
                g2_mask = self.sp_adata.obs['predicted_cell_type'].isin(self.g2_ct.value) & self.sp_adata.obs['predicted_microenvironment'].isin(self.g2_me.value)

                if not g1_mask.any() or not g2_mask.any():
                    self.status_label.value = "<b>Status:</b> ❌ Error: Empty group(s)."; print("Error: One or both groups have 0 cells.")
                    return

                adata_volcano = self.sp_adata[g1_mask | g2_mask].copy()
                
                adata_volcano.obs['volcano_group'] = pd.Series(index=adata_volcano.obs.index, dtype='object')
                adata_volcano.obs.loc[g1_mask[g1_mask | g2_mask], 'volcano_group'] = 'group1'
                adata_volcano.obs.loc[g2_mask[g1_mask | g2_mask], 'volcano_group'] = 'group2'

                print(f"Group 1 cells: {adata_volcano.obs['volcano_group'].value_counts().get('group1', 0)}")
                print(f"Group 2 cells: {adata_volcano.obs['volcano_group'].value_counts().get('group2', 0)}")

                sc.pp.filter_genes(adata_volcano, min_cells=self.min_cells_filter.value)
                if adata_volcano.n_vars == 0:
                    self.status_label.value = "<b>Status:</b> ❌ Error: No genes left."; print("Error: No genes left after filtering.")
                    return
                print(f"Genes after filtering: {adata_volcano.n_vars}")
                
                sc.pp.normalize_total(adata_volcano, target_sum=1e4); sc.pp.log1p(adata_volcano)
                sc.tl.rank_genes_groups(adata_volcano, groupby='volcano_group', reference='group2', method='wilcoxon', use_raw=False, key_added='dge')

                result = adata_volcano.uns['dge']
                df = pd.DataFrame({'gene': result['names']['group1'], 'log2fc': result['logfoldchanges']['group1'], 'pvals_adj': result['pvals_adj']['group1']})
                
                df['pvals_adj'] = df['pvals_adj'].replace(0, np.finfo(float).eps)
                df['-log10_pvals_adj'] = -np.log10(df['pvals_adj'])
                df['significant'] = (df['pvals_adj'] < self.pval_thresh.value) & (np.abs(df['log2fc']) > self.log2fc_thresh.value)

                self.fig = plt.figure(figsize=(10, 8))
                ax = self.fig.add_subplot(111)
                sns.scatterplot(data=df, x='log2fc', y='-log10_pvals_adj', hue='significant', palette={True: 'red', False: 'grey'}, s=20, alpha=0.7, ax=ax)
                ax.axvline(self.log2fc_thresh.value, color='blue', linestyle='--', lw=1)
                ax.axvline(-self.log2fc_thresh.value, color='blue', linestyle='--', lw=1)
                ax.axhline(-np.log10(self.pval_thresh.value), color='green', linestyle='--', lw=1)
                
                top_genes = df[df['significant']].copy()
                top_genes['abs_log2fc'] = np.abs(top_genes['log2fc'])
                top_genes = top_genes.sort_values('abs_log2fc', ascending=False).head(self.top_n_genes.value)

                texts = [ax.text(row.log2fc, row['-log10_pvals_adj'], row.gene, fontsize=9) for _, row in top_genes.iterrows()]
                if texts: at(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

                ax.set_title('Volcano Plot: Group 1 vs Group 2'); ax.set_xlabel('Log2 Fold Change'); ax.set_ylabel('-Log10 (Adjusted p-value)'); ax.grid(True, linestyle=':', alpha=0.6)
                plt.show()
                self.status_label.value = "<b>Status:</b> ✅ Analysis complete."
            except Exception as e:
                self.status_label.value = f"<b>Status:</b> ❌ Error: {e}"; print(f"An error occurred: {e}")

    def _on_save_button_clicked(self, b):
        if self.fig:
            filename = f"volcano_plot.{self.save_format.value}"
            # --- 👇 指定されたパスに保存するロジック ---
            if self.save_path:
                os.makedirs(self.save_path, exist_ok=True)
                full_path = os.path.join(self.save_path, filename)
            else:
                full_path = filename

            self.fig.savefig(full_path, dpi=300, bbox_inches='tight')
            self.status_label.value = f"<b>Status:</b> ✅ Plot saved to {full_path}"
        else:
            self.status_label.value = "<b>Status:</b> ⚠️ No plot to save. Please run the analysis first."

    def display(self):
        g1_box = widgets.VBox([widgets.HTML("<h4>Group 1</h4>"), self.g1_ct, self.g1_me])
        g2_box = widgets.VBox([widgets.HTML("<h4>Group 2</h4>"), self.g2_ct, self.g2_me])
        param_box = widgets.VBox([widgets.HTML("<h4>Parameters</h4>"), self.min_cells_filter, self.log2fc_thresh, self.pval_thresh, self.top_n_genes])
        save_box = widgets.HBox([self.save_format, self.save_button])
        
        controls = widgets.VBox([
            widgets.HTML("<h3>Interactive Volcano Plot</h3>"), widgets.HBox([g1_box, g2_box]),
            param_box, self.run_button,
            widgets.HTML("<hr><b>Export</b>"), save_box, self.status_label
        ], layout=widgets.Layout(width='auto'))

        main_ui = widgets.VBox([controls, self.output])
        display(main_ui)
        