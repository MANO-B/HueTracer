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
        self.sp_adata = sp_adata.copy()
        self.sp_adata_ref = sp_adata
        self.merged = merged_df.copy()
        self.merged_original = merged_df
        self.lib_id = lib_id
        self.original_clusters = clusters.copy()
        self.current_clusters = clusters.copy()
        
        # Image data
        self.hires_img = sp_adata.uns["spatial"][lib_id]["images"]["hires"]
        self.h, self.w = self.hires_img.shape[:2]
        
        # Group information
        self.group_order = self.merged["predicted_microenvironment"].dropna().unique()
        
        # Coordinate range
        self.x_min_data = self.merged["x"].min()
        self.x_max_data = self.merged["x"].max()
        self.y_min_data = self.merged["y"].min()
        self.y_max_data = self.merged["y"].max()
        
        # Selection-related variables
        self.lasso_selector = None
        self.selected_path = None
        self.selected_indices = []
        self.current_selection_polygon = None
        
        # Display settings
        self.displayed_groups = set(str(g) for g in self.group_order)
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        
        # Plot elements
        self.fig = None
        self.ax = None
        self.scatter_plots = {}
        
        print(f"=== Data Info ===")
        print(f"Total cells: {len(self.merged)}")
        print(f"Groups: {len(self.group_order)}")
        
        # Color settings
        self.setup_colors()
        
        # Create UI
        self.create_ui()
        
    def cleanup_selectors(self):
        """Properly clean up selectors"""
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
        """Set up colors and markers"""
        palette = sns.color_palette("tab20", n_colors=max(20, len(self.group_order)))
        
        self.color_map = {}
        for i, group in enumerate(self.group_order):
            color = palette[i % len(palette)]
            self.color_map[group] = color
            self.color_map[str(group)] = color
    
    def create_ui(self):
        """Create UI components"""
        
        # Selection mode
        self.selection_mode = widgets.RadioButtons(
            options=['Lasso', 'Rectangle'],
            value='Lasso',
            #description='Mode:',
            style={'description_width': 'initial'}
        )
        self.selection_mode.observe(self.on_mode_change, names='value')
        
        # Group display selection (multi-select)
        self.group_selector = widgets.SelectMultiple(
            options=[str(g) for g in self.group_order],
            value=[str(g) for g in self.group_order][:min(1, len(self.group_order))],
            #description='Display Groups:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(height='150px')
        )
        self.group_selector.observe(self.on_group_display_change, names='value')
        
        # New label input
        self.new_label_input = widgets.Text(
            value='selected',
            placeholder='Enter new label',
            description='New Label:',
            style={'description_width': 'initial'}
        )

        # Zoom controls
        self.zoom_slider = widgets.FloatSlider(
            value=1.0, min=0.5, max=10.0, step=0.1,
            description='Zoom:',
            readout_format='.1f'
        )
        self.zoom_slider.observe(self.on_zoom_change, names='value')
        
        # Buttons
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
        
        # Status
        self.status = widgets.HTML(value="<b>Status:</b> Ready")
        self.selection_info = widgets.HTML(value="<b>Selected:</b> 0 cells")
        
        # Output area
        self.output = widgets.Output()
        
        # Point size adjustment
        self.point_size_slider = widgets.FloatSlider(
            value=3.0, min=0.1, max=10.0, step=0.1,
            description='Point Size:',
            readout_format='.1f'
        )
        self.point_size_slider.observe(self.on_point_size_change, names='value')
        
        # Opacity adjustment
        self.alpha_slider = widgets.FloatSlider(
            value=0.8, min=0.1, max=1.0, step=0.1,
            description='Opacity:',
            readout_format='.1f'
        )
        self.alpha_slider.observe(self.on_alpha_change, names='value')
    
    def create_plot(self):
        """Create the main plot"""
        with self.output:
            clear_output(wait=True)
            
            # Disable selectors
            self.cleanup_selectors()
            
            # Close existing figure if any
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
                self.ax = None
            
            # Close extra figures
            plt.close('all')
            
             # Create new figure
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            
            # Background image
            self.ax.imshow(self.hires_img, extent=[0, self.w, self.h, 0], alpha=0.8)
            
            # Plot only the displayed groups
            self.scatter_plots = {}
            displayed_groups = list(self.group_selector.value)
            
            for group in displayed_groups:
                # Check for both string and original type
                group_data = self.merged[
                    (self.merged["predicted_microenvironment"] == group) | 
                    (self.merged["predicted_microenvironment"].astype(str) == str(group))
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
                    if special_group in self.merged["predicted_microenvironment"].values or str(special_group) in self.merged["predicted_microenvironment"].astype(str).values:
                        group_data = self.merged[self.merged["predicted_microenvironment"] == special_group]
                        
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
            
            # Axis settings
            self.update_view()
            
            # Legend
            if len(self.scatter_plots) <= 20:
                self.ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
            
            self.ax.set_aspect('equal')
            self.ax.axis('off')
            
            # Set up selectors
            self.setup_selector()
            
            plt.tight_layout()
            plt.show()
    
    def setup_selector(self):
        """Set up selection tools"""
        if self.selection_mode.value == 'Lasso':
            self.lasso_selector = LassoSelector(
                self.ax,
                onselect=self.on_lasso_select,
                useblit=True,
                button=[1],  # Left click
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
        """Callback for lasso selection"""
        # Create a path
        self.selected_path = Path(verts)
        
        # Consider only data for currently displayed groups
        displayed_groups = list(self.group_selector.value)
        mask_display = self.merged["predicted_microenvironment"].isin(displayed_groups) | \
                       self.merged["predicted_microenvironment"].astype(str).isin(displayed_groups)
        
        displayed_data = self.merged[mask_display]
        
        # Determine points inside the path
        if len(displayed_data) > 0:
            points = displayed_data[['x', 'y']].values
            inside = self.selected_path.contains_points(points)
            self.selected_indices = displayed_data[inside].index.tolist()
        else:
            self.selected_indices = []
        
        # Visualize the selected area
        if self.current_selection_polygon:
            self.current_selection_polygon.remove()
        
        self.current_selection_polygon = Polygon(
            verts, fill=False, edgecolor='yellow',
            linewidth=2, linestyle='--', alpha=0.8
        )
        self.ax.add_patch(self.current_selection_polygon)
        
        # Highlight selected points
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
        
        # Update status
        self.selection_info.value = f"<b>Selected:</b> {len(self.selected_indices)} cells"
    
    def on_rect_select(self, eclick, erelease):
        """Callback for rectangle selection"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if None in [x1, y1, x2, y2]:
            return
        
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Consider only data for currently displayed groups
        displayed_groups = list(self.group_selector.value)
        mask = (
            (self.merged["x"] >= x_min) & 
            (self.merged["x"] <= x_max) &
            (self.merged["y"] >= y_min) & 
            (self.merged["y"] <= y_max)
        )
        
        mask_display = self.merged["predicted_microenvironment"].isin(displayed_groups) | \
                       self.merged["predicted_microenvironment"].astype(str).isin(displayed_groups)
        mask = mask & mask_display
        
        self.selected_indices = self.merged[mask].index.tolist()
        
        # Update status
        self.selection_info.value = f"<b>Selected:</b> {len(self.selected_indices)} cells"
    
    def on_mode_change(self, change):
        """When selection mode is changed"""
        self.create_plot()
    
    def on_group_display_change(self, change):
        """When displayed groups are changed"""
        self.displayed_groups = set(change['new'])
        self.create_plot()
    
    def on_zoom_change(self, change):
        """When zoom is changed"""
        self.zoom_level = change['new']
        self.update_view()
    
    def on_point_size_change(self, change):
        """When point size is changed"""
        for scatter in self.scatter_plots.values():
            scatter.set_sizes([change['new']])
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def on_alpha_change(self, change):
        """When opacity is changed"""
        for scatter in self.scatter_plots.values():
            scatter.set_alpha(change['new'])
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def update_view(self):
        """Update the view (zoom/pan)"""
        if self.ax is None:
            return
        
        # Calculate the display range based on zoom
        x_center = (self.x_min_data + self.x_max_data) / 2 + self.pan_offset[0]
        y_center = (self.y_min_data + self.y_max_data) / 2 + self.pan_offset[1]
        x_range = (self.x_max_data - self.x_min_data) / self.zoom_level
        y_range = (self.y_max_data - self.y_min_data) / self.zoom_level
        
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center + y_range/2, y_center - y_range/2)  # Y-axis is inverted
        
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def fit_to_view(self, b):
        """Fit the view to the entire data"""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.zoom_slider.value = 1.0
        self.update_view()
    
    def apply_selection(self, b):
        """Apply the selection"""
        if len(self.selected_indices) > 0:
            # Get the new label
            new_label = self.new_label_input.value.strip()
            if not new_label:
                self.status.value = "<b>Status:</b> ⚠️ Please enter a valid label"
                return
                        
            cat_col = "predicted_microenvironment"
            
            # Update cell type order (if a new cell type was added)
            if new_label not in self.merged[cat_col].cat.categories:
                self.merged[cat_col] = self.merged[cat_col].cat.add_categories([new_label])
            # Update
            self.merged.loc[self.selected_indices, cat_col] = new_label
 
            # Add new label to color map (if not already there)
            if new_label not in self.color_map:
                # Assign a new color
                import matplotlib.pyplot as plt
                colors = plt.cm.tab20(np.linspace(0, 1, 20))
                new_color_idx = len(self.color_map) % 20
                self.color_map[new_label] = colors[new_color_idx]
                self.color_map[str(new_label)] = colors[new_color_idx]
            
            self.status.value = f"<b>Status:</b> Applied - {len(self.selected_indices)} cells changed to '{new_label}'"

            # Update plot
            self.create_plot()
            
            # Clear selection
            self.selected_indices = []
            self.selection_info.value = "<b>Selected:</b> 0 cells"
    
    def clear_selection(self, b):
        """Clear the current selection"""
        self.selected_indices = []
        self.selection_info.value = "<b>Selected:</b> 0 cells"
        
        if self.current_selection_polygon:
            self.current_selection_polygon.remove()
            self.current_selection_polygon = None
        
        self.create_plot()
        self.status.value = "<b>Status:</b> Selection cleared"
    
    def reset_all(self, b):
        """Reset everything"""
        self.merged["predicted_microenvironment"] = self.original_clusters
        self.current_clusters = self.original_clusters.copy()
        
        self.selected_indices = []
        self.selection_info.value = "<b>Selected:</b> 0 cells"
        
        self.status.value = "<b>Status:</b> Reset complete"
        self.create_plot()
    
    def run(self):
        """Launch the UI"""
        # Layout
        selection_box = widgets.VBox([
            widgets.HTML("<h3>Cell Selector: modify microenvironment</h3>"),
            widgets.HTML("<b>Selection methods:</b>"),
            self.selection_mode,
            widgets.HTML("<b>Displayed microenvironments:</b>"),
            self.new_label_input,  # Added new label input
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
        
        # Display everything
        display(widgets.HBox([
            control_panel,
            self.output
        ]))
        
        # Initial plot
        self.create_plot()
        
        return self
    
    def update_anndata(self, b):
        """Update the AnnData object"""
        try:
            # Update AnnData obs
            self.sp_adata_ref.obs["predicted_microenvironment"] = self.merged['predicted_microenvironment'].values
            self.merged_original['predicted_microenvironment'] = self.merged['predicted_microenvironment'].values
            
            # Success message
            self.status.value = "<b>Status:</b> ✅ AnnData successfully updated!"
            
            # Display statistics
            with self.output:
                print("\n" + "="*50)
                print("✅ AnnData Update Complete!")
                print("="*50)
                group_counts = self.merged["predicted_microenvironment"].value_counts()
                print("\nCurrent group distribution:")
                for group, count in group_counts.items():
                    print(f"   {group}: {count} cells")
                
                if -1 in group_counts.index:
                    percentage = (group_counts[-1] / len(self.merged)) * 100
                    print(f"\n📊 Microenvironment '-1' cells: {group_counts[-1]} ({percentage:.1f}%)")
                
                print("\n💾 Changes have been saved to:")
                print(f"   - sp_adata_microenvironment.obs['predicted_microenvironment']")
                print("="*50)
                
        except Exception as e:
            self.status.value = f"<b>Status:</b> ❌ Error updating AnnData: {str(e)}"
            with self.output:
                print(f"\n❌ Error: {e}")
    
    def export_data(self, b):
        """Export data (return as variables)"""
        try:
            # Save to global variables (for easy access in Jupyter)
            import __main__ as main
            main.updated_merged_export = self.merged.copy()
            main.updated_clusters_export = self.current_clusters.copy()
            
            self.status.value = "<b>Status:</b> 📦 Data exported to variables!"
            
            with self.output:
                print("\n" + "="*50)
                print("📦 Data Export Complete!")
                print("="*50)
                print("\nExported variables:")
                print("   - updated_merged_export: Updated merged DataFrame")
                print("   - updated_clusters_export: Updated clusters array")
                print("\nYou can now use these variables in your notebook:")
                print("   merged = updated_merged_export")
                print("   clusters = updated_clusters_export")
                print("="*50)
                
        except Exception as e:
            self.status.value = f"<b>Status:</b> ❌ Error exporting data: {str(e)}"
            with self.output:
                print(f"\n❌ Error: {e}")


# Usage function
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
        self.sp_adata = sp_adata.copy()
        self.sp_adata_ref = sp_adata
        self.merged = merged_df.copy()
        self.merged_original = merged_df
        self.lib_id = lib_id
        self.original_clusters = clusters.copy()
        self.current_clusters = clusters.copy()
        
        # Suppress matplotlib figure management warnings
        plt.rcParams['figure.max_open_warning'] = 50
        
        # Image data
        self.hires_img = sp_adata.uns["spatial"][lib_id]["images"]["hires"]
        self.h, self.w = self.hires_img.shape[:2]
        
        # Get Cell type and Microenvironment information
        self.cell_type_order = self.merged["predicted_cell_type"].dropna().unique()
        self.microenv_order = self.merged["predicted_microenvironment"].dropna().unique()
        
        # Coordinate range
        self.x_min_data = self.merged["x"].min()
        self.x_max_data = self.merged["x"].max()
        self.y_min_data = self.merged["y"].min()
        self.y_max_data = self.merged["y"].max()
        
        # Selection-related variables
        self.lasso_selector = None
        self.rect_selector = None
        self.selected_path = None
        self.selected_indices = []
        self.current_selection_polygon = None
        self.selection_highlight = None  # For selection highlighting
        
        # Display settings
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        
        # Plot elements
        self.fig = None
        self.ax = None
        self.scatter_plots = {}
        
        print(f"=== Data Info ===")
        print(f"Total cells: {len(self.merged)}")
        print(f"Cell types: {len(self.cell_type_order)}")
        print(f"Microenvironments: {len(self.microenv_order)}")
        
        # Color settings
        self.setup_colors()
        
        # Create UI
        self.create_ui()
        
    def setup_colors(self):
        """Set up colors and markers"""
        palette = sns.color_palette("tab20", n_colors=max(20, len(self.microenv_order)))
        
        self.color_map = {}
        for i, microenv in enumerate(self.microenv_order):
            color = palette[i % len(palette)]
            self.color_map[microenv] = color
            self.color_map[str(microenv)] = color
    
    def create_ui(self):
        """Create UI components"""
        
        # Selection mode
        self.selection_mode = widgets.RadioButtons(
            options=['Lasso', 'Rectangle'],
            value='Lasso',
            style={'description_width': 'initial'}
        )
        self.selection_mode.observe(self.on_mode_change, names='value')
        
        # Cell type selection (single-select)
        self.cell_type_selector = widgets.Dropdown(
            options=[str(ct) for ct in self.cell_type_order],
            value=str(self.cell_type_order[0]) if len(self.cell_type_order) > 0 else None,
            description='Target Cell Type:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Microenvironment display selection (multi-select)
        self.microenv_selector = widgets.SelectMultiple(
            options=[str(me) for me in self.microenv_order],
            value=[str(me) for me in self.microenv_order][:min(3, len(self.microenv_order))],
            description='Display Microenvs:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(height='150px', width='300px')
        )
        self.microenv_selector.observe(self.on_microenv_display_change, names='value')
        
        # New cell type name input
        self.new_cell_type_input = widgets.Text(
            value='selected_cell_type',
            placeholder='Enter new cell type name',
            description='New Cell Type:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )

        # Zoom controls
        self.zoom_slider = widgets.FloatSlider(
            value=1.0, min=0.5, max=10.0, step=0.1,
            description='Zoom:',
            readout_format='.1f'
        )
        self.zoom_slider.observe(self.on_zoom_change, names='value')
        
        # Buttons
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
        
        # Status
        self.status = widgets.HTML(value="<b>Status:</b> Ready")
        self.selection_info = widgets.HTML(value="<b>Selected:</b> 0 cells")
        
        # Output area
        self.output = widgets.Output()
        
        # Point size adjustment
        self.point_size_slider = widgets.FloatSlider(
            value=3.0, min=0.1, max=10.0, step=0.1,
            description='Point Size:',
            readout_format='.1f'
        )
        self.point_size_slider.observe(self.on_point_size_change, names='value')
        
        # Opacity adjustment
        self.alpha_slider = widgets.FloatSlider(
            value=0.8, min=0.1, max=1.0, step=0.1,
            description='Opacity:',
            readout_format='.1f'
        )
        self.alpha_slider.observe(self.on_alpha_change, names='value')
    
    def create_plot(self):
        """Create the main plot"""
        with self.output:
            clear_output(wait=True)
            
            # Disable selectors
            self.cleanup_selectors()
            
            # Close existing figure if any
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
                self.ax = None
            
            # Close extra figures
            plt.close('all')
            
            # Create a new figure - adjust size
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            
            # Background image
            self.ax.imshow(self.hires_img, extent=[0, self.w, self.h, 0], alpha=0.8)
            
            # Plot only the cells that meet the conditions for both displayed microenvironment and selected cell type
            self.scatter_plots = {}
            displayed_microenvs = list(self.microenv_selector.value)
            selected_cell_type = self.cell_type_selector.value
            
            # Display only cells of the selected cell type within the displayed microenvironments
            if selected_cell_type and displayed_microenvs:
                for microenv in displayed_microenvs:
                    # Condition: specified microenvironment AND specified cell type
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
            
            # Axis settings
            self.update_view()
            
            # Legend
            if len(self.scatter_plots) <= 20:
                self.ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
            
            self.ax.set_aspect('equal')
            self.ax.axis('off')
            self.ax.set_title(f'Cell Type Selector - Target: {selected_cell_type}', fontsize=14, pad=20)
            
            # Set up selectors
            self.setup_selector()
            
            plt.tight_layout()
            plt.show()
    
    def cleanup_selectors(self):
        """Properly clean up selectors"""
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
        """Set up selection tools"""
        if self.selection_mode.value == 'Lasso':
            self.lasso_selector = LassoSelector(
                self.ax,
                onselect=self.on_lasso_select,
                useblit=True,
                button=[1],  # Left click
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
        """Callback for lasso selection"""
        # Create a path
        self.selected_path = Path(verts)
        
        # Consider only data for currently displayed groups
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
        
        # Determine points inside the path
        if len(displayed_data) > 0:
            points = displayed_data[['x', 'y']].values
            inside = self.selected_path.contains_points(points)
            self.selected_indices = displayed_data[inside].index.tolist()
        else:
            self.selected_indices = []
        
        # Visualize the selected area
        if self.current_selection_polygon:
            self.current_selection_polygon.remove()
        
        self.current_selection_polygon = Polygon(
            verts, fill=False, edgecolor='yellow',
            linewidth=2, linestyle='--', alpha=0.8
        )
        self.ax.add_patch(self.current_selection_polygon)
        
        # Highlight selected points
        if len(self.selected_indices) > 0:
            selected_data = self.merged.loc[self.selected_indices]
            # Remove existing highlight if any
            if hasattr(self, 'selection_highlight') and self.selection_highlight:
                try:
                    self.selection_highlight.remove()
                except:
                    pass
            
            # Add new highlight
            self.selection_highlight = self.ax.scatter(
                selected_data["x"],
                selected_data["y"],
                c='yellow',
                s=self.point_size_slider.value * 3,
                alpha=1.0,
                marker='o',
                edgecolors='red',
                linewidths=1.0,
                zorder=1000  # Display on top
            )
        
        self.fig.canvas.draw_idle()
        
        # Update status
        self.selection_info.value = f"<b>Selected:</b> {len(self.selected_indices)} cells (CT: {selected_cell_type})"
    
    def on_rect_select(self, eclick, erelease):
        """Callback for rectangle selection"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if None in [x1, y1, x2, y2]:
            return
        
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Consider only data for currently displayed groups
        displayed_microenvs = list(self.microenv_selector.value)
        selected_cell_type = self.cell_type_selector.value
        
        if not selected_cell_type or not displayed_microenvs:
            self.selected_indices = []
            self.selection_info.value = "<b>Selected:</b> 0 cells (No cell type or microenvironment selected)"
            return
        
        # Select points within the coordinate range
        mask = (
            (self.merged["x"] >= x_min) & 
            (self.merged["x"] <= x_max) &
            (self.merged["y"] >= y_min) & 
            (self.merged["y"] <= y_max)
        )
        
        # Combine with display conditions
        mask_display = (
            (self.merged["predicted_microenvironment"].isin(displayed_microenvs) | 
             self.merged["predicted_microenvironment"].astype(str).isin(displayed_microenvs)) &
            ((self.merged["predicted_cell_type"] == selected_cell_type) | 
             (self.merged["predicted_cell_type"].astype(str) == str(selected_cell_type)))
        )
        mask = mask & mask_display
        
        self.selected_indices = self.merged[mask].index.tolist()
        
        # Highlight selected points
        if len(self.selected_indices) > 0:
            selected_data = self.merged.loc[self.selected_indices]
            # Remove existing highlight if any
            if hasattr(self, 'selection_highlight') and self.selection_highlight:
                try:
                    self.selection_highlight.remove()
                except:
                    pass
            
            # Add new highlight
            self.selection_highlight = self.ax.scatter(
                selected_data["x"],
                selected_data["y"],
                c='yellow',
                s=self.point_size_slider.value * 3,
                alpha=1.0,
                marker='o',
                edgecolors='red',
                linewidths=1.0,
                zorder=1000  # Display on top
            )
            
            self.fig.canvas.draw_idle()
        
        # Update status
        self.selection_info.value = f"<b>Selected:</b> {len(self.selected_indices)} cells (CT: {selected_cell_type})"
    
    def on_mode_change(self, change):
        """When selection mode is changed"""
        import time
        time.sleep(0.1)  # Wait a bit
        self.create_plot()
    
    def on_microenv_display_change(self, change):
        """When displayed microenvironments are changed"""
        import time
        time.sleep(0.1)  # Wait a bit
        self.create_plot()
    
    def on_zoom_change(self, change):
        """When zoom is changed"""
        self.zoom_level = change['new']
        self.update_view()
    
    def on_point_size_change(self, change):
        """When point size is changed"""
        for scatter in self.scatter_plots.values():
            scatter.set_sizes([change['new']])
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def on_alpha_change(self, change):
        """When opacity is changed"""
        for scatter in self.scatter_plots.values():
            scatter.set_alpha(change['new'])
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def update_view(self):
        """Update the view (zoom/pan)"""
        if self.ax is None:
            return
        
        # Calculate the display range based on zoom
        x_center = (self.x_min_data + self.x_max_data) / 2 + self.pan_offset[0]
        y_center = (self.y_min_data + self.y_max_data) / 2 + self.pan_offset[1]
        x_range = (self.x_max_data - self.x_min_data) / self.zoom_level
        y_range = (self.y_max_data - self.y_min_data) / self.zoom_level
        
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center + y_range/2, y_center - y_range/2)  # Y-axis is inverted
        
        if self.fig:
            self.fig.canvas.draw_idle()
    
    def fit_to_view(self, b):
        """Fit the view to the entire data"""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.zoom_slider.value = 1.0
        self.update_view()
    
    def apply_selection(self, b):
        """Apply the selection"""
        if len(self.selected_indices) > 0:
            # Get the new cell type name
            new_cell_type = self.new_cell_type_input.value.strip()
            if not new_cell_type:
                self.status.value = "<b>Status:</b> ⚠️ Please enter a valid cell type name"
                return
            
            # Update cell type order (if a new cell type was added)
            if new_cell_type not in self.cell_type_order:
                self.cell_type_order = np.append(self.cell_type_order, new_cell_type)
                # Update dropdown options
                self.cell_type_selector.options = [str(ct) for ct in self.cell_type_order]

            cat_col = "predicted_cell_type"
            if new_cell_type not in self.merged[cat_col].cat.categories:
                self.merged[cat_col] = self.merged[cat_col].cat.add_categories([new_cell_type])
            
            self.merged.loc[self.selected_indices, cat_col] = new_cell_type
                        
            self.status.value = f"<b>Status:</b> Applied - {len(self.selected_indices)} cells changed to '{new_cell_type}'"

            # Clear selection
            self.selected_indices = []
            self.selection_info.value = "<b>Selected:</b> 0 cells"
            
            # Update plot
            self.create_plot()
    
    def clear_selection(self, b):
        """Clear the current selection"""
        self.selected_indices = []
        self.selection_info.value = "<b>Selected:</b> 0 cells"
        
        # Remove selection polygon
        if self.current_selection_polygon:
            try:
                self.current_selection_polygon.remove()
            except:
                pass
            self.current_selection_polygon = None
        
        # Remove selection highlight
        if hasattr(self, 'selection_highlight') and self.selection_highlight:
            try:
                self.selection_highlight.remove()
            except:
                pass
            self.selection_highlight = None
        
        # Redraw the plot, but only update the selection part
        if self.fig:
            self.fig.canvas.draw_idle()
        
        self.status.value = "<b>Status:</b> Selection cleared"
    
    def reset_all(self, b):
        """Reset everything"""
        # Restore cell type info from original clusters
        if 'predicted_cell_type' in self.merged_original.columns:
            self.merged["predicted_cell_type"] = self.merged_original["predicted_cell_type"].copy()
        
        self.current_clusters = self.original_clusters.copy()
        
        self.selected_indices = []
        self.selection_info.value = "<b>Selected:</b> 0 cells"
        
        self.status.value = "<b>Status:</b> Reset complete"
        self.create_plot()
    
    def run(self):
        """Launch the UI"""
        # Layout
        selection_box = widgets.VBox([
            widgets.HTML("<h3>Selector for cell rename</h3>"),
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
        
        # Display everything
        main_ui = widgets.HBox([
            control_panel,
            self.output
        ])
        
        display(main_ui)
        
        # Wait a bit before creating the plot
        import time
        time.sleep(0.1)
        
        # Initial plot
        self.create_plot()
        
        return self
    
    def update_anndata(self, b):
        """Update the AnnData object"""
        try:
            # Update AnnData obs
            self.sp_adata_ref.obs["predicted_cell_type"] = self.merged['predicted_cell_type'].values            
            # Update the original merged DataFrame (if passed by reference)
            self.merged_original['predicted_cell_type'] = self.merged['predicted_cell_type'].values
            
            # Success message
            self.status.value = "<b>Status:</b> ✅ AnnData successfully updated!"
            
            # Display statistics
            with self.output:
                print("\n" + "="*50)
                print("✅ AnnData Update Complete!")
                print("="*50)
                cell_type_counts = self.merged["predicted_cell_type"].value_counts()
                print("\nCurrent cell type distribution:")
                for cell_type, count in cell_type_counts.items():
                    print(f"   {cell_type}: {count} cells")
                
                print(f"\n📊 Total unique cell types: {len(cell_type_counts)}")
                
                print("\n💾 Changes have been saved to:")
                print(f"   - sp_adata_microenvironment.obs['predicted_cell_type']")
                print("="*50)
                
        except Exception as e:
            self.status.value = f"<b>Status:</b> ❌ Error updating AnnData: {str(e)}"
            with self.output:
                print(f"\n❌ Error: {e}")
    
    def export_data(self, b):
        """Export data (return as variables)"""
        try:
            # Save to global variables (for easy access in Jupyter)
            import __main__ as main
            main.updated_merged_celltype_export = self.merged.copy()
            main.updated_clusters_celltype_export = self.current_clusters.copy()
            
            self.status.value = "<b>Status:</b> 📦 Data exported to variables!"
            
            with self.output:
                print("\n" + "="*50)
                print("📦 Data Export Complete!")
                print("="*50)
                print("\nExported variables:")
                print("   - updated_merged_celltype_export: Updated merged DataFrame")
                print("   - updated_clusters_celltype_export: Updated clusters array")
                print("\nYou can now use these variables in your notebook:")
                print("   merged = updated_merged_celltype_export")
                print("   clusters = updated_clusters_celltype_export")
                print("="*50)
                
        except Exception as e:
            self.status.value = f"<b>Status:</b> ❌ Error exporting data: {str(e)}"
            with self.output:
                print(f"\n❌ Error: {e}")


# Usage function
def lasso_selection_cell_type(sp_adata, merged, lib_id, clusters):
    """
    Launch a lasso selection widget for cell type modification.
    
    Parameters:
    - sp_adata: AnnData object
    - merged: Merged DataFrame (including predicted_cell_type, predicted_microenvironment, x, y columns)
    - lib_id: Library ID
    - clusters: Cluster information
    
    Returns:
    - LassoCellSelectorCellType: The selector object
    """
    selector = LassoCellSelectorCellType(
        sp_adata,
        merged,
        lib_id,
        clusters
    )
    return selector.run()