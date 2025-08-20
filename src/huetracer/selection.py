import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import LassoSelector, RectangleSelector, Button
from matplotlib.path import Path
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import cycle
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.patches import Polygon

class LassoCellSelectorMicroenvironment:
    def __init__(self, sp_adata, merged_df, lib_id, clusters):
        self.sp_adata = sp_adata
        self.sp_adata_ref = sp_adata
        self.merged = merged_df.copy()
        self.merged_original = merged_df
        self.lib_id = lib_id
        self.original_clusters = clusters.copy()
        self.current_clusters = clusters.copy()
        
        # 画像データ
        self.hires_img = sp_adata.uns["spatial"][lib_id]["images"]["hires"]
        self.h, self.w = self.hires_img.shape[:2]
        
        # グループ情報
        self.group_order = self.merged["group"].dropna().unique()
        
        # 座標範囲
        self.x_min_data = self.merged["x"].min()
        self.x_max_data = self.merged["x"].max()
        self.y_min_data = self.merged["y"].min()
        self.y_max_data = self.merged["y"].max()
        
        # 選択関連
        self.lasso_selector = None
        self.selected_path = None
        self.selected_indices = []
        self.current_selection_polygon = None
        
        # 表示設定
        self.displayed_groups = set(str(g) for g in self.group_order)
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        
        # プロット要素
        self.fig = None
        self.ax = None
        self.scatter_plots = {}
        
        print(f"=== Data Info ===")
        print(f"Total cells: {len(self.merged)}")
        print(f"Groups: {len(self.group_order)}")
        
        # 色設定
        self.setup_colors()
        
        # UI作成
        self.create_ui()
        
    def cleanup_selectors(self):
        """セレクターを適切にクリーンアップ"""
        if hasattr(self, 'lasso_selector') and self.lasso_selector is not None:
            try:
                self.lasso_selector.disconnect_events()
            except:
                pass
            self.lasso_selector = None
            
        if hasattr(self, 'rect_selector') and self.rect_selector is not None:
            try:
                self.rect_selector.set_active(False)
            except:
                pass
            self.rect_selector = None
    
    def setup_colors(self):
        """色とマーカーの設定"""
        palette = sns.color_palette("tab20", n_colors=max(20, len(self.group_order)))
        
        self.color_map = {}
        for i, group in enumerate(self.group_order):
            color = palette[i % len(palette)]
            self.color_map[group] = color
            self.color_map[str(group)] = color
    
    def create_ui(self):
        """UIコンポーネントの作成"""
        
        # 選択モード
        self.selection_mode = widgets.RadioButtons(
            options=['Lasso', 'Rectangle'],
            value='Lasso',
            #description='Mode:',
            style={'description_width': 'initial'}
        )
        self.selection_mode.observe(self.on_mode_change, names='value')
        
        # グループ表示選択（複数選択可能）
        self.group_selector = widgets.SelectMultiple(
            options=[str(g) for g in self.group_order],
            value=[str(g) for g in self.group_order][:min(1, len(self.group_order))],
            #description='Display Groups:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(height='150px')
        )
        self.group_selector.observe(self.on_group_display_change, names='value')
        
        # 新しいラベル入力
        self.new_label_input = widgets.Text(
            value='selected',
            placeholder='Enter new label',
            description='New Label:',
            style={'description_width': 'initial'}
        )

        # ズームコントロール
        self.zoom_slider = widgets.FloatSlider(
            value=1.0, min=0.5, max=10.0, step=0.1,
            description='Zoom:',
            readout_format='.1f'
        )
        self.zoom_slider.observe(self.on_zoom_change, names='value')
        
        # ボタン
        self.apply_btn = widgets.Button(
            description='Apply Selection',
            button_style='success',
            icon='check'
        )
        self.clear_selection_btn = widgets.Button(
            description='Clear Selection',
            button_style='warning',
            icon='times'
        )
        self.reset_btn = widgets.Button(
            description='Reset All',
            button_style='danger',
            icon='refresh'
        )
        self.fit_view_btn = widgets.Button(
            description='Fit to View',
            button_style='info',
            icon='expand'
        )
        self.update_anndata_btn = widgets.Button(
            description='Update AnnData',
            button_style='primary',
            icon='save',
            tooltip='Update the original AnnData object with current changes'
        )
        self.export_btn = widgets.Button(
            description='Export Data',
            button_style='info',
            icon='download',
            tooltip='Export updated merged DataFrame and clusters'
        )
        
        self.apply_btn.on_click(self.apply_selection)
        self.clear_selection_btn.on_click(self.clear_selection)
        self.reset_btn.on_click(self.reset_all)
        self.fit_view_btn.on_click(self.fit_to_view)
        self.update_anndata_btn.on_click(self.update_anndata)
        self.export_btn.on_click(self.export_data)
        
        # ステータス
        self.status = widgets.HTML(value="<b>Status:</b> Ready")
        self.selection_info = widgets.HTML(value="<b>Selected:</b> 0 cells")
        
        # 出力エリア
        self.output = widgets.Output()
        
        # 点のサイズ調整
        self.point_size_slider = widgets.FloatSlider(
            value=3.0, min=0.1, max=10.0, step=0.1,
            description='Point Size:',
            readout_format='.1f'
        )
        self.point_size_slider.observe(self.on_point_size_change, names='value')
        
        # 透明度調整
        self.alpha_slider = widgets.FloatSlider(
            value=0.8, min=0.1, max=1.0, step=0.1,
            description='Opacity:',
            readout_format='.1f'
        )
        self.alpha_slider.observe(self.on_alpha_change, names='value')
    
    def create_plot(self):
        """メインプロットを作成"""
        with self.output:
            clear_output(wait=True)
            
            # セレクターを無効化
            self.cleanup_selectors()
            
            # 既存のfigureがあれば適切に閉じる
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
                self.ax = None
            
            # 余分なfigureを閉じる
            plt.close('all')
            
             # 新しいfigureを作成
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            
            # 背景画像
            self.ax.imshow(self.hires_img, extent=[0, self.w, self.h, 0], alpha=0.8)
            
            # 表示するグループのみプロット
            self.scatter_plots = {}
            displayed_groups = list(self.group_selector.value)
            
            for group in displayed_groups:
                # str型とオリジナル型の両方をチェック
                group_data = self.merged[
                    (self.merged["group"] == group) | 
                    (self.merged["group"].astype(str) == str(group))
                ]
                
                if len(group_data) > 0:
                    scatter = self.ax.scatter(
                        group_data["x"],
                        group_data["y"],
                        c=[self.color_map.get(group, 'gray')],
                        s=self.point_size_slider.value,
                        alpha=self.alpha_slider.value,
                        label=str(group),
                        picker=True,
                        rasterized=True
                    )
                    self.scatter_plots[group] = scatter

            special_groups = self.new_label_input.value.strip()
            if not special_groups:
                for special_group in special_groups:
                    if special_group in self.merged["group"].values or str(special_group) in self.merged["group"].astype(str).values:
                        if special_group == -1 or special_group == "-1":
                            group_data = self.merged[(self.merged["group"] == -1) | (self.merged["group"] == "-1")]
                        else:
                            group_data = self.merged[self.merged["group"] == special_group]
                        
                        if len(group_data) > 0:
                            scatter = self.ax.scatter(
                                group_data["x"],
                                group_data["y"],
                                c=self.color_map.get(special_group, 'red'),
                                s=self.point_size_slider.value * 2,
                                alpha=0.9,
                                label=str(special_group),
                                marker='x',
                                rasterized=True
                            )
                            self.scatter_plots[special_group] = scatter
            
            # 軸設定
            self.update_view()
            
            # 凡例
            if len(self.scatter_plots) <= 20:
                self.ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
            
            self.ax.set_aspect('equal')
            self.ax.axis('off')
            
            # セレクター設定
            self.setup_selector()
            
            plt.tight_layout()
            plt.show()
    
    def setup_selector(self):
        """選択ツールの設定"""
        if self.selection_mode.value == 'Lasso':
            self.lasso_selector = LassoSelector(
                self.ax,
                onselect=self.on_lasso_select,
                useblit=True,
                button=[1],  # 左クリック
            )
        elif self.selection_mode.value == 'Rectangle':
            self.rect_selector = RectangleSelector(
                self.ax,
                self.on_rect_select,
                useblit=True,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords='pixels',
                interactive=True
            )
    
    def on_lasso_select(self, verts):
        """投げ縄選択時のコールバック"""
        # パスを作成
        self.selected_path = Path(verts)
        
        # 現在表示されているグループのデータのみ対象
        displayed_groups = list(self.group_selector.value)
        mask_display = self.merged["group"].isin(displayed_groups) | \
                       self.merged["group"].astype(str).isin(displayed_groups)
        
        displayed_data = self.merged[mask_display]
        
        # パス内の点を判定
        if len(displayed_data) > 0:
            points = displayed_data[['x', 'y']].values
            inside = self.selected_path.contains_points(points)
            self.selected_indices = displayed_data[inside].index.tolist()
        else:
            self.selected_indices = []
        
        # 選択領域を可視化
        if self.current_selection_polygon:
            self.current_selection_polygon.remove()
        
        self.current_selection_polygon = Polygon(
            verts, fill=False, edgecolor='yellow',
            linewidth=2, linestyle='--', alpha=0.8
        )
        self.ax.add_patch(self.current_selection_polygon)
        
        # 選択された点をハイライト
        if len(self.selected_indices) > 0:
            selected_data = self.merged.loc[self.selected_indices]
            self.ax.scatter(
                selected_data["x"],
                selected_data["y"],
                c='yellow',
                s=self.point_size_slider.value * 3,
                alpha=1.0,
                marker='o',
                edgecolors='red',
                linewidths=0.5
            )
        
        self.fig.canvas.draw_idle()
        
        # ステータス更新
        self.selection_info.value = f"<b>Selected:</b> {len(self.selected_indices)} cells"
    
    def on_rect_select(self, eclick, erelease):
        """矩形選択時のコールバック"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if None in [x1, y1, x2, y2]:
            return
        
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # 現在表示されているグループのデータのみ対象
        displayed_groups = list(self.group_selector.value)
        mask = (
            (self.merged["x"] >= x_min) & 
            (self.merged["x"] <= x_max) &
            (self.merged["y"] >= y_min) & 
            (self.merged["y"] <= y_max)
        )
        
        mask_display = self.merged["group"].isin(displayed_groups) | \
                      self.merged["group"].astype(str).isin(displayed_groups)
        mask = mask & mask_display
        
        self.selected_indices = self.merged[mask].index.tolist()
        
        # ステータス更新
        self.selection_info.value = f"<b>Selected:</b> {len(self.selected_indices)} cells"
    
    def on_mode_change(self, change):
        """選択モードが変更されたとき"""
        self.create_plot()
    
    def on_group_display_change(self, change):
        """表示グループが変更されたとき"""
        self.displayed_groups = set(change['new'])
        self.create_plot()
    
    def on_zoom_change(self, change):
        """ズーム変更時"""
        self.zoom_level = change['new']
        self.update_view()
    
    def on_point_size_change(self, change):
        """点のサイズ変更時"""
        for scatter in self.scatter_plots.values():
            scatter.set_sizes([change['new']])
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def on_alpha_change(self, change):
        """透明度変更時"""
        for scatter in self.scatter_plots.values():
            scatter.set_alpha(change['new'])
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def update_view(self):
        """ビューを更新（ズーム/パン）"""
        if self.ax is None:
            return
        
        # ズームに応じた表示範囲を計算
        x_center = (self.x_min_data + self.x_max_data) / 2 + self.pan_offset[0]
        y_center = (self.y_min_data + self.y_max_data) / 2 + self.pan_offset[1]
        x_range = (self.x_max_data - self.x_min_data) / self.zoom_level
        y_range = (self.y_max_data - self.y_min_data) / self.zoom_level
        
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center + y_range/2, y_center - y_range/2)  # Y軸は反転
        
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def fit_to_view(self, b):
        """ビューを全体に合わせる"""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.zoom_slider.value = 1.0
        self.update_view()
    
    def apply_selection(self, b):
        """選択を適用"""
        if len(self.selected_indices) > 0:
            # 新しいラベルを取得
            new_label = self.new_label_input.value.strip()
            if not new_label:
                self.status.value = "<b>Status:</b> ⚠️ Please enter a valid label"
                return
            
            try:
                new_label_value = new_label
            except:
                new_label_value = int(new_label)
            
            # 更新
            self.merged.loc[self.selected_indices, "predicted_microenvironment"] = new_label_value
            self.merged.loc[self.selected_indices, "group"] = new_label_value

            # clustersも更新
            for idx in self.selected_indices:
                if idx < len(self.current_clusters):
                    self.current_clusters[idx] = new_label_value
            
            # 色マップに新しいラベルを追加（まだない場合）
            if new_label_value not in self.color_map:
                # 新しい色を割り当て
                import matplotlib.pyplot as plt
                colors = plt.cm.tab20(np.linspace(0, 1, 20))
                new_color_idx = len(self.color_map) % 20
                self.color_map[new_label_value] = colors[new_color_idx]
                self.color_map[str(new_label_value)] = colors[new_color_idx]
            
            self.status.value = f"<b>Status:</b> Applied - {len(self.selected_indices)} cells changed to '{new_label_value}'"

            # プロットを更新
            self.create_plot()
            
            # 選択をクリア
            self.selected_indices = []
            self.selection_info.value = "<b>Selected:</b> 0 cells"
    
    def clear_selection(self, b):
        """現在の選択をクリア"""
        self.selected_indices = []
        self.selection_info.value = "<b>Selected:</b> 0 cells"
        
        if self.current_selection_polygon:
            self.current_selection_polygon.remove()
            self.current_selection_polygon = None
        
        self.create_plot()
        self.status.value = "<b>Status:</b> Selection cleared"
    
    def reset_all(self, b):
        """全てリセット"""
        self.merged["predicted_microenvironment"] = self.original_clusters
        self.merged["group"] = self.merged["predicted_microenvironment"]
        self.current_clusters = self.original_clusters.copy()
        
        self.selected_indices = []
        self.selection_info.value = "<b>Selected:</b> 0 cells"
        
        self.status.value = "<b>Status:</b> Reset complete"
        self.create_plot()
    
    def run(self):
        """UIを起動"""
        # レイアウト
        selection_box = widgets.VBox([
            widgets.HTML("<h3>Cell Selector: modify microenvironment</h3>"),
            widgets.HTML("<b>Selection methods:</b>"),
            self.selection_mode,
            widgets.HTML("<b>Displayed microenvironments:</b>"),
            self.new_label_input,  # 新しいラベル入力を追加
            self.group_selector,
            self.zoom_slider,
            self.point_size_slider,
            self.alpha_slider
        ])
        
        button_box = widgets.VBox([
            widgets.HTML("<b>Actions:</b>"),
            widgets.HBox([self.apply_btn, self.clear_selection_btn]),
            widgets.HBox([self.reset_btn, self.fit_view_btn]),
            widgets.HTML("<b>Data Management:</b>"),
            widgets.HBox([self.update_anndata_btn, self.export_btn]),
            self.selection_info,
            self.status
        ])
        
        control_panel = widgets.VBox([
            selection_box,
            button_box
        ], layout=widgets.Layout(width='350px'))
        
        # 全体表示
        display(widgets.HBox([
            control_panel,
            self.output
        ]))
        
        # 初期プロット
        self.create_plot()
        
        return self
    
    def update_anndata(self, b):
        """AnnDataオブジェクトを更新"""
        try:
            # AnnDataのobsを更新
            self.sp_adata_ref.obs["predicted_microenvironment"] = self.merged['predicted_microenvironment'].values
            # 元のmergedデータフレームも更新（参照渡しの場合）
            if hasattr(self, 'merged_original'):
                for col in ['predicted_microenvironment', 'group']:
                    if col in self.merged_original.columns:
                        self.merged_original[col] = self.merged[col].values
            
            # 成功メッセージ
            self.status.value = "<b>Status:</b> ✅ AnnData successfully updated!"
            
            # 統計情報を表示
            with self.output:
                print("\n" + "="*50)
                print("✅ AnnData Update Complete!")
                print("="*50)
                group_counts = self.merged["group"].value_counts()
                print("\nCurrent group distribution:")
                for group, count in group_counts.items():
                    print(f"  {group}: {count} cells")
                
                if -1 in group_counts.index:
                    percentage = (group_counts[-1] / len(self.merged)) * 100
                    print(f"\n📊 Microenvironment '-1' cells: {group_counts[-1]} ({percentage:.1f}%)")
                
                print("\n💾 Changes have been saved to:")
                print(f"  - sp_adata_microenvironment.obs['predicted_microenvironment']")
                print("="*50)
                
        except Exception as e:
            self.status.value = f"<b>Status:</b> ❌ Error updating AnnData: {str(e)}"
            with self.output:
                print(f"\n❌ Error: {e}")
    
    def export_data(self, b):
        """データをエクスポート（変数として返す）"""
        try:
            # グローバル変数に保存（Jupyter環境で使いやすいように）
            import __main__ as main
            main.updated_merged_export = self.merged.copy()
            main.updated_clusters_export = self.current_clusters.copy()
            
            self.status.value = "<b>Status:</b> 📦 Data exported to variables!"
            
            with self.output:
                print("\n" + "="*50)
                print("📦 Data Export Complete!")
                print("="*50)
                print("\nExported variables:")
                print("  - updated_merged_export: Updated merged DataFrame")
                print("  - updated_clusters_export: Updated clusters array")
                print("\nYou can now use these variables in your notebook:")
                print("  merged = updated_merged_export")
                print("  clusters = updated_clusters_export")
                print("="*50)
                
        except Exception as e:
            self.status.value = f"<b>Status:</b> ❌ Error exporting data: {str(e)}"
            with self.output:
                print(f"\n❌ Error: {e}")


# 使用関数
def lasso_selection_microenvironment(sp_adata, merged, lib_id, clusters):
    selector = LassoCellSelectorMicroenvironment(
        sp_adata,
        merged,
        lib_id,
        clusters
    )
    return selector.run()

    
class LassoCellSelectorCellType:
    def __init__(self, sp_adata, merged_df, lib_id, clusters):
        self.sp_adata = sp_adata
        self.sp_adata_ref = sp_adata
        self.merged = merged_df.copy()
        self.merged_original = merged_df
        self.lib_id = lib_id
        self.original_clusters = clusters.copy()
        self.current_clusters = clusters.copy()
        
        # matplotlib figure管理の警告を抑制
        plt.rcParams['figure.max_open_warning'] = 50
        
        # 画像データ
        self.hires_img = sp_adata.uns["spatial"][lib_id]["images"]["hires"]
        self.h, self.w = self.hires_img.shape[:2]
        
        # Cell typeとMicroenvironmentの情報を取得
        self.cell_type_order = self.merged["predicted_cell_type"].dropna().unique()
        self.microenv_order = self.merged["predicted_microenvironment"].dropna().unique()
        
        # 座標範囲
        self.x_min_data = self.merged["x"].min()
        self.x_max_data = self.merged["x"].max()
        self.y_min_data = self.merged["y"].min()
        self.y_max_data = self.merged["y"].max()
        
        # 選択関連
        self.lasso_selector = None
        self.rect_selector = None
        self.selected_path = None
        self.selected_indices = []
        self.current_selection_polygon = None
        self.selection_highlight = None  # 選択ハイライト用
        
        # 表示設定
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        
        # プロット要素
        self.fig = None
        self.ax = None
        self.scatter_plots = {}
        
        print(f"=== Data Info ===")
        print(f"Total cells: {len(self.merged)}")
        print(f"Cell types: {len(self.cell_type_order)}")
        print(f"Microenvironments: {len(self.microenv_order)}")
        
        # 色設定
        self.setup_colors()
        
        # UI作成
        self.create_ui()
        
    def setup_colors(self):
        """色とマーカーの設定"""
        palette = sns.color_palette("tab20", n_colors=max(20, len(self.microenv_order)))
        
        self.color_map = {}
        for i, microenv in enumerate(self.microenv_order):
            color = palette[i % len(palette)]
            self.color_map[microenv] = color
            self.color_map[str(microenv)] = color
    
    def create_ui(self):
        """UIコンポーネントの作成"""
        
        # 選択モード
        self.selection_mode = widgets.RadioButtons(
            options=['Lasso', 'Rectangle'],
            value='Lasso',
            style={'description_width': 'initial'}
        )
        self.selection_mode.observe(self.on_mode_change, names='value')
        
        # Cell type選択（単一選択）
        self.cell_type_selector = widgets.Dropdown(
            options=[str(ct) for ct in self.cell_type_order],
            value=str(self.cell_type_order[0]) if len(self.cell_type_order) > 0 else None,
            description='Target Cell Type:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Microenvironment表示選択（複数選択可能）
        self.microenv_selector = widgets.SelectMultiple(
            options=[str(me) for me in self.microenv_order],
            value=[str(me) for me in self.microenv_order][:min(3, len(self.microenv_order))],
            description='Display Microenvs:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(height='150px', width='300px')
        )
        self.microenv_selector.observe(self.on_microenv_display_change, names='value')
        
        # 新しいcell type名入力
        self.new_cell_type_input = widgets.Text(
            value='selected_cell_type',
            placeholder='Enter new cell type name',
            description='New Cell Type:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )

        # ズームコントロール
        self.zoom_slider = widgets.FloatSlider(
            value=1.0, min=0.5, max=10.0, step=0.1,
            description='Zoom:',
            readout_format='.1f'
        )
        self.zoom_slider.observe(self.on_zoom_change, names='value')
        
        # ボタン
        self.apply_btn = widgets.Button(
            description='Apply Selection',
            button_style='success',
            icon='check'
        )
        self.clear_selection_btn = widgets.Button(
            description='Clear Selection',
            button_style='warning',
            icon='times'
        )
        self.reset_btn = widgets.Button(
            description='Reset All',
            button_style='danger',
            icon='refresh'
        )
        self.fit_view_btn = widgets.Button(
            description='Fit to View',
            button_style='info',
            icon='expand'
        )
        self.update_anndata_btn = widgets.Button(
            description='Update AnnData',
            button_style='primary',
            icon='save',
            tooltip='Update the original AnnData object with current changes'
        )
        self.export_btn = widgets.Button(
            description='Export Data',
            button_style='info',
            icon='download',
            tooltip='Export updated merged DataFrame and clusters'
        )
        
        self.apply_btn.on_click(self.apply_selection)
        self.clear_selection_btn.on_click(self.clear_selection)
        self.reset_btn.on_click(self.reset_all)
        self.fit_view_btn.on_click(self.fit_to_view)
        self.update_anndata_btn.on_click(self.update_anndata)
        self.export_btn.on_click(self.export_data)
        
        # ステータス
        self.status = widgets.HTML(value="<b>Status:</b> Ready")
        self.selection_info = widgets.HTML(value="<b>Selected:</b> 0 cells")
        
        # 出力エリア
        self.output = widgets.Output()
        
        # 点のサイズ調整
        self.point_size_slider = widgets.FloatSlider(
            value=3.0, min=0.1, max=10.0, step=0.1,
            description='Point Size:',
            readout_format='.1f'
        )
        self.point_size_slider.observe(self.on_point_size_change, names='value')
        
        # 透明度調整
        self.alpha_slider = widgets.FloatSlider(
            value=0.8, min=0.1, max=1.0, step=0.1,
            description='Opacity:',
            readout_format='.1f'
        )
        self.alpha_slider.observe(self.on_alpha_change, names='value')
    
    def create_plot(self):
        """メインプロットを作成"""
        with self.output:
            clear_output(wait=True)
            
            # セレクターを無効化
            self.cleanup_selectors()
            
            # 既存のfigureがあれば適切に閉じる
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
                self.ax = None
            
            # 余分なfigureを閉じる
            plt.close('all')
            
            # 新しいfigureを作成 - サイズを調整
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            
            # 背景画像
            self.ax.imshow(self.hires_img, extent=[0, self.w, self.h, 0], alpha=0.8)
            
            # 表示するmicroenvironmentと選択されたcell typeの両方の条件を満たす細胞のみプロット
            self.scatter_plots = {}
            displayed_microenvs = list(self.microenv_selector.value)
            selected_cell_type = self.cell_type_selector.value
            
            # 選択されたcell typeかつ表示されたmicroenvironmentの細胞のみ表示
            if selected_cell_type and displayed_microenvs:
                for microenv in displayed_microenvs:
                    # 条件: 指定されたmicroenvironmentかつ指定されたcell type
                    microenv_data = self.merged[
                        ((self.merged["predicted_microenvironment"] == microenv) | 
                         (self.merged["predicted_microenvironment"].astype(str) == str(microenv))) &
                        ((self.merged["predicted_cell_type"] == selected_cell_type) | 
                         (self.merged["predicted_cell_type"].astype(str) == str(selected_cell_type)))
                    ]
                    
                    if len(microenv_data) > 0:
                        scatter = self.ax.scatter(
                            microenv_data["x"],
                            microenv_data["y"],
                            c=[self.color_map.get(microenv, 'gray')],
                            s=self.point_size_slider.value,
                            alpha=self.alpha_slider.value,
                            label=f'ME: {str(microenv)}',
                            picker=True,
                            rasterized=True
                        )
                        self.scatter_plots[f'{microenv}_{selected_cell_type}'] = scatter
            
            # 軸設定
            self.update_view()
            
            # 凡例
            if len(self.scatter_plots) <= 20:
                self.ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
            
            self.ax.set_aspect('equal')
            self.ax.axis('off')
            self.ax.set_title(f'Cell Type Selector - Target: {selected_cell_type}', fontsize=14, pad=20)
            
            # セレクター設定
            self.setup_selector()
            
            plt.tight_layout()
            plt.show()
    
    def cleanup_selectors(self):
        """セレクターを適切にクリーンアップ"""
        if hasattr(self, 'lasso_selector') and self.lasso_selector is not None:
            try:
                self.lasso_selector.disconnect_events()
            except:
                pass
            self.lasso_selector = None
            
        if hasattr(self, 'rect_selector') and self.rect_selector is not None:
            try:
                self.rect_selector.set_active(False)
            except:
                pass
            self.rect_selector = None
    
    def setup_selector(self):
        """選択ツールの設定"""
        if self.selection_mode.value == 'Lasso':
            self.lasso_selector = LassoSelector(
                self.ax,
                onselect=self.on_lasso_select,
                useblit=True,
                button=[1],  # 左クリック
            )
        elif self.selection_mode.value == 'Rectangle':
            self.rect_selector = RectangleSelector(
                self.ax,
                self.on_rect_select,
                useblit=True,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords='pixels',
                interactive=True
            )
    
    def on_lasso_select(self, verts):
        """投げ縄選択時のコールバック"""
        # パスを作成
        self.selected_path = Path(verts)
        
        # 現在表示されている細胞のみ対象（選択されたcell typeかつ表示されたmicroenvironment）
        displayed_microenvs = list(self.microenv_selector.value)
        selected_cell_type = self.cell_type_selector.value
        
        if not selected_cell_type or not displayed_microenvs:
            self.selected_indices = []
            self.selection_info.value = "<b>Selected:</b> 0 cells (No cell type or microenvironment selected)"
            return
        
        mask_display = (
            (self.merged["predicted_microenvironment"].isin(displayed_microenvs) | 
             self.merged["predicted_microenvironment"].astype(str).isin(displayed_microenvs)) &
            ((self.merged["predicted_cell_type"] == selected_cell_type) | 
             (self.merged["predicted_cell_type"].astype(str) == str(selected_cell_type)))
        )
        
        displayed_data = self.merged[mask_display]
        
        # パス内の点を判定
        if len(displayed_data) > 0:
            points = displayed_data[['x', 'y']].values
            inside = self.selected_path.contains_points(points)
            self.selected_indices = displayed_data[inside].index.tolist()
        else:
            self.selected_indices = []
        
        # 選択領域を可視化
        if self.current_selection_polygon:
            self.current_selection_polygon.remove()
        
        self.current_selection_polygon = Polygon(
            verts, fill=False, edgecolor='yellow',
            linewidth=2, linestyle='--', alpha=0.8
        )
        self.ax.add_patch(self.current_selection_polygon)
        
        # 選択された点をハイライト
        if len(self.selected_indices) > 0:
            selected_data = self.merged.loc[self.selected_indices]
            # 既存のハイライトを削除（もしあれば）
            if hasattr(self, 'selection_highlight') and self.selection_highlight:
                try:
                    self.selection_highlight.remove()
                except:
                    pass
            
            # 新しいハイライトを追加
            self.selection_highlight = self.ax.scatter(
                selected_data["x"],
                selected_data["y"],
                c='yellow',
                s=self.point_size_slider.value * 3,
                alpha=1.0,
                marker='o',
                edgecolors='red',
                linewidths=1.0,
                zorder=1000  # 最前面に表示
            )
        
        self.fig.canvas.draw_idle()
        
        # ステータス更新
        self.selection_info.value = f"<b>Selected:</b> {len(self.selected_indices)} cells (CT: {selected_cell_type})"
    
    def on_rect_select(self, eclick, erelease):
        """矩形選択時のコールバック"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if None in [x1, y1, x2, y2]:
            return
        
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # 現在表示されている細胞のみ対象（選択されたcell typeかつ表示されたmicroenvironment）
        displayed_microenvs = list(self.microenv_selector.value)
        selected_cell_type = self.cell_type_selector.value
        
        if not selected_cell_type or not displayed_microenvs:
            self.selected_indices = []
            self.selection_info.value = "<b>Selected:</b> 0 cells (No cell type or microenvironment selected)"
            return
        
        # 座標範囲内の点を選択
        mask = (
            (self.merged["x"] >= x_min) & 
            (self.merged["x"] <= x_max) &
            (self.merged["y"] >= y_min) & 
            (self.merged["y"] <= y_max)
        )
        
        # 表示条件と組み合わせ
        mask_display = (
            (self.merged["predicted_microenvironment"].isin(displayed_microenvs) | 
             self.merged["predicted_microenvironment"].astype(str).isin(displayed_microenvs)) &
            ((self.merged["predicted_cell_type"] == selected_cell_type) | 
             (self.merged["predicted_cell_type"].astype(str) == str(selected_cell_type)))
        )
        mask = mask & mask_display
        
        self.selected_indices = self.merged[mask].index.tolist()
        
        # 選択された点をハイライト
        if len(self.selected_indices) > 0:
            selected_data = self.merged.loc[self.selected_indices]
            # 既存のハイライトを削除（もしあれば）
            if hasattr(self, 'selection_highlight') and self.selection_highlight:
                try:
                    self.selection_highlight.remove()
                except:
                    pass
            
            # 新しいハイライトを追加
            self.selection_highlight = self.ax.scatter(
                selected_data["x"],
                selected_data["y"],
                c='yellow',
                s=self.point_size_slider.value * 3,
                alpha=1.0,
                marker='o',
                edgecolors='red',
                linewidths=1.0,
                zorder=1000  # 最前面に表示
            )
            
            self.fig.canvas.draw_idle()
        
        # ステータス更新
        self.selection_info.value = f"<b>Selected:</b> {len(self.selected_indices)} cells (CT: {selected_cell_type})"
    
    def on_mode_change(self, change):
        """選択モードが変更されたとき"""
        import time
        time.sleep(0.1)  # 少し待つ
        self.create_plot()
    
    def on_microenv_display_change(self, change):
        """表示microenvironmentが変更されたとき"""
        import time
        time.sleep(0.1)  # 少し待つ
        self.create_plot()
    
    def on_zoom_change(self, change):
        """ズーム変更時"""
        self.zoom_level = change['new']
        self.update_view()
    
    def on_point_size_change(self, change):
        """点のサイズ変更時"""
        for scatter in self.scatter_plots.values():
            scatter.set_sizes([change['new']])
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def on_alpha_change(self, change):
        """透明度変更時"""
        for scatter in self.scatter_plots.values():
            scatter.set_alpha(change['new'])
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def update_view(self):
        """ビューを更新（ズーム/パン）"""
        if self.ax is None:
            return
        
        # ズームに応じた表示範囲を計算
        x_center = (self.x_min_data + self.x_max_data) / 2 + self.pan_offset[0]
        y_center = (self.y_min_data + self.y_max_data) / 2 + self.pan_offset[1]
        x_range = (self.x_max_data - self.x_min_data) / self.zoom_level
        y_range = (self.y_max_data - self.y_min_data) / self.zoom_level
        
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center + y_range/2, y_center - y_range/2)  # Y軸は反転
        
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def fit_to_view(self, b):
        """ビューを全体に合わせる"""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.zoom_slider.value = 1.0
        self.update_view()
    
    def apply_selection(self, b):
        """選択を適用"""
        if len(self.selected_indices) > 0:
            # 新しいcell type名を取得
            new_cell_type = self.new_cell_type_input.value.strip()
            if not new_cell_type:
                self.status.value = "<b>Status:</b> ⚠️ Please enter a valid cell type name"
                return
            
            # 更新
            self.merged.loc[self.selected_indices, "predicted_cell_type"] = new_cell_type
            
            # clustersも更新（cell type情報を含める場合）
            for idx in self.selected_indices:
                if idx < len(self.current_clusters):
                    self.current_clusters[idx] = new_cell_type
            
            # cell type orderを更新（新しいcell typeが追加された場合）
            if new_cell_type not in self.cell_type_order:
                self.cell_type_order = np.append(self.cell_type_order, new_cell_type)
                # ドロップダウンオプションを更新
                self.cell_type_selector.options = [str(ct) for ct in self.cell_type_order]
            
            self.status.value = f"<b>Status:</b> Applied - {len(self.selected_indices)} cells changed to '{new_cell_type}'"

            # 選択をクリア
            self.selected_indices = []
            self.selection_info.value = "<b>Selected:</b> 0 cells"
            
            # プロットを更新
            self.create_plot()
    
    def clear_selection(self, b):
        """現在の選択をクリア"""
        self.selected_indices = []
        self.selection_info.value = "<b>Selected:</b> 0 cells"
        
        # 選択領域のポリゴンを削除
        if self.current_selection_polygon:
            try:
                self.current_selection_polygon.remove()
            except:
                pass
            self.current_selection_polygon = None
        
        # 選択ハイライトを削除
        if hasattr(self, 'selection_highlight') and self.selection_highlight:
            try:
                self.selection_highlight.remove()
            except:
                pass
            self.selection_highlight = None
        
        # プロットを再描画するのではなく、選択部分のみ更新
        if self.fig:
            self.fig.canvas.draw_idle()
        
        self.status.value = "<b>Status:</b> Selection cleared"
    
    def reset_all(self, b):
        """全てリセット"""
        # 元のclustersからcell type情報を復元
        if 'predicted_cell_type' in self.merged_original.columns:
            self.merged["predicted_cell_type"] = self.merged_original["predicted_cell_type"].copy()
        
        self.current_clusters = self.original_clusters.copy()
        
        self.selected_indices = []
        self.selection_info.value = "<b>Selected:</b> 0 cells"
        
        self.status.value = "<b>Status:</b> Reset complete"
        self.create_plot()
    
    def run(self):
        """UIを起動"""        
        # レイアウト
        selection_box = widgets.VBox([
            widgets.HTML("<h3>Cell Type Selector</h3>"),
            widgets.HTML("<b>Selection methods:</b>"),
            self.selection_mode,
            widgets.HTML("<b>Target Cell Type:</b>"),
            self.cell_type_selector,
            widgets.HTML("<b>New Cell Type Name:</b>"),
            self.new_cell_type_input,
            widgets.HTML("<b>Display Microenvironments:</b>"),
            self.microenv_selector,
            self.zoom_slider,
            self.point_size_slider,
            self.alpha_slider
        ])
        
        button_box = widgets.VBox([
            widgets.HTML("<b>Actions:</b>"),
            widgets.HBox([self.apply_btn, self.clear_selection_btn]),
            widgets.HBox([self.reset_btn, self.fit_view_btn]),
            widgets.HTML("<b>Data Management:</b>"),
            widgets.HBox([self.update_anndata_btn, self.export_btn]),
            self.selection_info,
            self.status
        ])
        
        control_panel = widgets.VBox([
            selection_box,
            button_box
        ], layout=widgets.Layout(width='350px'))
        
        # 全体表示
        main_ui = widgets.HBox([
            control_panel,
            self.output
        ])
        
        display(main_ui)
        
        # 少し待ってからプロットを作成
        import time
        time.sleep(0.1)
        
        # 初期プロット
        self.create_plot()
        
        return self
    
    def update_anndata(self, b):
        """AnnDataオブジェクトを更新"""
        try:
            # AnnDataのobsを更新
            self.sp_adata_ref.obs["predicted_cell_type"] = self.merged['predicted_cell_type'].values            
            
            # 元のmergedデータフレームも更新（参照渡しの場合）
            if hasattr(self, 'merged_original'):
                if 'predicted_cell_type' in self.merged_original.columns:
                    self.merged_original['predicted_cell_type'] = self.merged['predicted_cell_type'].values
            
            # 成功メッセージ
            self.status.value = "<b>Status:</b> ✅ AnnData successfully updated!"
            
            # 統計情報を表示
            with self.output:
                print("\n" + "="*50)
                print("✅ AnnData Update Complete!")
                print("="*50)
                cell_type_counts = self.merged["predicted_cell_type"].value_counts()
                print("\nCurrent cell type distribution:")
                for cell_type, count in cell_type_counts.items():
                    print(f"  {cell_type}: {count} cells")
                
                print(f"\n📊 Total unique cell types: {len(cell_type_counts)}")
                
                print("\n💾 Changes have been saved to:")
                print(f"  - sp_adata.obs['predicted_cell_type']")
                print("="*50)
                
        except Exception as e:
            self.status.value = f"<b>Status:</b> ❌ Error updating AnnData: {str(e)}"
            with self.output:
                print(f"\n❌ Error: {e}")
    
    def export_data(self, b):
        """データをエクスポート（変数として返す）"""
        try:
            # グローバル変数に保存（Jupyter環境で使いやすいように）
            import __main__ as main
            main.updated_merged_celltype_export = self.merged.copy()
            main.updated_clusters_celltype_export = self.current_clusters.copy()
            
            self.status.value = "<b>Status:</b> 📦 Data exported to variables!"
            
            with self.output:
                print("\n" + "="*50)
                print("📦 Data Export Complete!")
                print("="*50)
                print("\nExported variables:")
                print("  - updated_merged_celltype_export: Updated merged DataFrame")
                print("  - updated_clusters_celltype_export: Updated clusters array")
                print("\nYou can now use these variables in your notebook:")
                print("  merged = updated_merged_celltype_export")
                print("  clusters = updated_clusters_celltype_export")
                print("="*50)
                
        except Exception as e:
            self.status.value = f"<b>Status:</b> ❌ Error exporting data: {str(e)}"
            with self.output:
                print(f"\n❌ Error: {e}")


# 使用関数
def lasso_selection_cell_type(sp_adata, merged, lib_id, clusters):
    """
    Cell Type変更用のLasso選択ウィジェットを起動
    
    Parameters:
    - sp_adata: AnnDataオブジェクト
    - merged: マージされたDataFrame (predicted_cell_type, predicted_microenvironment, x, y列を含む)
    - lib_id: ライブラリID
    - clusters: クラスター情報
    
    Returns:
    - LassoCellSelectorCellType: セレクターオブジェクト
    """
    selector = LassoCellSelectorCellType(
        sp_adata,
        merged,
        lib_id,
        clusters
    )
    return selector.run()