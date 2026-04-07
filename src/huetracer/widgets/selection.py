import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image as PILImage
import plotly.graph_objects as go
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def _build_figure_widget_or_raise():
    """Create FigureWidget or raise a clear ImportError.

    Plotly 6+ may require anywidget in notebook environments.
    """
    try:
        return go.FigureWidget()
    except Exception as exc:
        raise ImportError(
            "Plotly FigureWidget is unavailable in this environment. "
            "Install anywidget (pip install anywidget) or use fallback selector."
        ) from exc


def _downsample_image(img, factor):
    """Downsample an image array for display only."""
    if factor >= 1.0:
        return img

    h, w = img.shape[:2]
    new_h = max(1, int(h * factor))
    new_w = max(1, int(w * factor))

    if img.dtype in (np.float32, np.float64):
        arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        out = np.array(PILImage.fromarray(arr).resize((new_w, new_h), PILImage.LANCZOS))
        return (out / 255.0).astype(img.dtype)

    return np.array(PILImage.fromarray(img.astype(np.uint8)).resize((new_w, new_h), PILImage.LANCZOS))


def _limit_image_pixels(img, max_pixels=220000):
    """Cap display image size to keep plotly interaction responsive."""
    h, w = img.shape[:2]
    pixels = h * w
    if pixels <= max_pixels:
        return img

    factor = (max_pixels / float(pixels)) ** 0.5
    return _downsample_image(img, factor)


def _to_pil_image(arr):
    """Convert numpy image to PIL image for plotly layout images."""
    if arr.dtype in (np.float32, np.float64):
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return PILImage.fromarray(arr)


def _to_plotly_image_array(arr):
    """Convert image array to uint8 format accepted by go.Image."""
    if arr.dtype in (np.float32, np.float64):
        out = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        out = arr.astype(np.uint8)
    else:
        out = arr
    return out


def _as_aligned_series(values, index, name):
    """Normalize labels to a pandas Series aligned to `index`."""
    if isinstance(values, pd.Series):
        out = values.reindex(index)
    else:
        out = pd.Series(values, index=index)
    out.name = name
    return out


def _make_color_map(labels):
    """Create a stable color map for categorical labels."""
    palette = sns.color_palette("tab20", n_colors=max(20, len(labels)))
    cmap = {}
    for i, label in enumerate(labels):
        r, g, b = palette[i % len(palette)]
        cmap[str(label)] = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    return cmap


class _BasePlotlySelector:
    """Shared high-performance plotly selector behavior."""

    def __init__(self, sp_adata, merged_df, lib_id, downsample_factor=0.25):
        self.sp_adata_ref = sp_adata
        self.merged = merged_df.copy()
        self.merged_original = merged_df
        self.lib_id = lib_id
        self.downsample_factor = downsample_factor

        raw_img = sp_adata.uns["spatial"][lib_id]["images"]["0.5_mpp_150_buffer"]
        self.h, self.w = raw_img.shape[:2]
        self.bg_img = _limit_image_pixels(_downsample_image(raw_img, downsample_factor))
        self.bg_pil = _to_pil_image(self.bg_img)
        self.bg_plot = _to_plotly_image_array(self.bg_img)
        del raw_img

        self.index_values = self.merged.index.to_numpy()
        self.x = self.merged["x"].to_numpy()
        self.y = self.merged["y"].to_numpy()
        self.index_to_pos = {idx: i for i, idx in enumerate(self.index_values)}

        self.zoom_level = 1.0
        self.max_highlight_points = 20000
        self.max_render_points = 180000
        self.selected_global_indices = np.array([], dtype=self.index_values.dtype)
        self.visible_positions = np.array([], dtype=np.int64)

        self.output = widgets.Output(layout=widgets.Layout(width="100%", min_width="760px", height="760px"))
        self.status = widgets.HTML(value="<b>Status:</b> Ready")
        self.selection_info = widgets.HTML(value="<b>Selected:</b> 0 cells")
        self.point_size_slider = widgets.FloatSlider(
            value=2.0,
            min=0.2,
            max=10.0,
            step=0.1,
            description="Point Size:",
            readout_format=".1f",
        )
        self.opacity_slider = widgets.FloatSlider(
            value=0.35,
            min=0.05,
            max=1.0,
            step=0.05,
            description="Opacity:",
            readout_format=".2f",
        )
        self.zoom_slider = widgets.FloatSlider(
            value=1.0,
            min=0.5,
            max=12.0,
            step=0.1,
            description="Zoom:",
            readout_format=".1f",
        )
        self.selection_mode = widgets.ToggleButtons(
            options=[("Lasso", "lasso"), ("Rectangle", "select"), ("Pan", "pan")],
            value="lasso",
            description="Mode:",
        )

        self.point_size_slider.observe(self._on_style_change, names="value")
        self.opacity_slider.observe(self._on_style_change, names="value")
        self.zoom_slider.observe(self._on_zoom_change, names="value")
        self.selection_mode.observe(self._on_mode_change, names="value")

        self.fig = _build_figure_widget_or_raise()
        self.image_trace_idx = 0
        self.cells_trace_idx = 1
        self.selected_trace_idx = 2
        self._build_figure()

    def _build_figure(self):
        img_h, img_w = self.bg_plot.shape[:2]
        self.fig.add_trace(
            go.Image(
                z=self.bg_plot,
                x0=0,
                y0=0,
                dx=self.w / max(1, img_w),
                dy=self.h / max(1, img_h),
                name="HE",
                hoverinfo="skip",
                opacity=1.0,
            )
        )

        self.fig.add_trace(
            go.Scattergl(
                x=[],
                y=[],
                mode="markers",
                marker={"size": self.point_size_slider.value, "opacity": self.opacity_slider.value, "color": []},
                customdata=[],
                hoverinfo="skip",
                name="cells",
                showlegend=False,
            )
        )
        self.fig.add_trace(
            go.Scattergl(
                x=[],
                y=[],
                mode="markers",
                marker={"size": self.point_size_slider.value * 2.5, "opacity": 1.0, "color": "yellow"},
                hoverinfo="skip",
                name="selected",
                showlegend=False,
            )
        )

        self.fig.update_layout(
            dragmode=self.selection_mode.value,
            width=900,
            height=740,
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
        )
        self.fig.update_xaxes(showgrid=False, visible=False, range=[0, self.w])
        self.fig.update_yaxes(showgrid=False, visible=False, range=[self.h, 0], scaleanchor="x", scaleratio=1)

        self.fig.data[self.cells_trace_idx].on_selection(self._on_plot_selection)

    def _on_mode_change(self, change):
        self.fig.update_layout(dragmode=change["new"])

    def _on_zoom_change(self, change):
        self.zoom_level = change["new"]
        x_center = (self.x.min() + self.x.max()) / 2.0
        y_center = (self.y.min() + self.y.max()) / 2.0
        x_range = (self.x.max() - self.x.min()) / self.zoom_level
        y_range = (self.y.max() - self.y.min()) / self.zoom_level
        self.fig.update_xaxes(range=[x_center - x_range / 2, x_center + x_range / 2])
        self.fig.update_yaxes(range=[y_center + y_range / 2, y_center - y_range / 2])

    def _on_style_change(self, _):
        self.fig.data[self.cells_trace_idx].marker.size = self.point_size_slider.value
        self.fig.data[self.cells_trace_idx].marker.opacity = self.opacity_slider.value
        self.fig.data[self.selected_trace_idx].marker.size = self.point_size_slider.value * 2.5

    def _sample_render_positions(self, positions):
        if len(positions) <= self.max_render_points:
            return positions
        keep = np.random.default_rng(0).choice(positions, size=self.max_render_points, replace=False)
        keep.sort()
        return keep

    def _sample_highlight_positions(self, positions):
        if len(positions) <= self.max_highlight_points:
            return positions
        keep = np.random.default_rng(0).choice(positions, size=self.max_highlight_points, replace=False)
        keep.sort()
        return keep

    def _refresh_selected_overlay(self):
        if len(self.selected_global_indices) == 0:
            self.fig.data[self.selected_trace_idx].x = []
            self.fig.data[self.selected_trace_idx].y = []
            self.selection_info.value = "<b>Selected:</b> 0 cells"
            return

        pos = self.merged.index.get_indexer(self.selected_global_indices)
        pos = pos[pos >= 0]
        pos = self._sample_highlight_positions(pos)

        self.fig.data[self.selected_trace_idx].x = self.x[pos]
        self.fig.data[self.selected_trace_idx].y = self.y[pos]

        extra = ""
        if len(self.selected_global_indices) > self.max_highlight_points:
            extra = f" (showing {self.max_highlight_points})"
        self.selection_info.value = f"<b>Selected:</b> {len(self.selected_global_indices)} cells{extra}"

    def _on_plot_selection(self, trace, points, selector):
        if points is None or len(points.point_inds) == 0:
            self.selected_global_indices = np.array([], dtype=self.index_values.dtype)
            self._refresh_selected_overlay()
            return

        cd = np.asarray(trace.customdata)
        if cd.size == 0:
            self.selected_global_indices = np.array([], dtype=self.index_values.dtype)
            self._refresh_selected_overlay()
            return

        selected = cd[np.asarray(points.point_inds, dtype=np.int64)]
        self.selected_global_indices = np.asarray(selected, dtype=self.index_values.dtype)
        self._refresh_selected_overlay()

    def _refresh_main_trace(self, visible_positions, color_by):
        visible_positions = self._sample_render_positions(visible_positions)
        self.visible_positions = visible_positions

        colors = [color_by[pos] for pos in visible_positions]

        self.fig.data[self.cells_trace_idx].x = self.x[visible_positions]
        self.fig.data[self.cells_trace_idx].y = self.y[visible_positions]
        self.fig.data[self.cells_trace_idx].customdata = self.index_values[visible_positions]
        self.fig.data[self.cells_trace_idx].marker.color = colors
        self.fig.data[self.cells_trace_idx].marker.size = self.point_size_slider.value
        self.fig.data[self.cells_trace_idx].marker.opacity = self.opacity_slider.value

    def _mount_figure(self):
        with self.output:
            clear_output(wait=True)
            display(self.fig)

    def _rebuild_figure(self, keep_selection=True):
        # Rebuilding the widget is the most reliable way to clear Plotly selection artifacts.
        selected = self.selected_global_indices.copy() if keep_selection else np.array([], dtype=self.index_values.dtype)

        self.fig = _build_figure_widget_or_raise()
        self._build_figure()

        if hasattr(self, "_refresh_plot"):
            self._refresh_plot()

        self.selected_global_indices = selected
        self._refresh_selected_overlay()
        self._mount_figure()

    def _clear_selection(self, _=None):
        self.selected_global_indices = np.array([], dtype=self.index_values.dtype)
        self._refresh_selected_overlay()
        self._clear_selection_frame_only(keep_selection=False)
        self.status.value = "<b>Status:</b> Selection cleared"

    def _clear_selection_frame_only(self, _=None, keep_selection=True):
        self._rebuild_figure(keep_selection=keep_selection)
        self.status.value = "<b>Status:</b> Selection frame cleared"

    def _export_data(self, _=None):
        raise NotImplementedError

    def _update_anndata(self, _=None):
        raise NotImplementedError

    def _apply_selection(self, _=None):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class LassoCellSelectorMicroenvironment(_BasePlotlySelector):
    """High-performance microenvironment relabeling widget using Plotly ScatterGL."""

    def __init__(self, sp_adata, merged_df, lib_id, clusters, downsample_factor=0.25):
        self.original_clusters = _as_aligned_series(clusters, merged_df.index, "predicted_microenvironment")
        self.current_clusters = self.original_clusters.copy()

        super().__init__(sp_adata, merged_df, lib_id, downsample_factor=downsample_factor)

        self.labels = self.merged["predicted_microenvironment"].astype(str).to_numpy()
        self.group_order = [str(v) for v in self.merged["predicted_microenvironment"].dropna().unique()]
        self.displayed_groups = set(self.group_order)
        self.color_map = _make_color_map(self.group_order)

        self.group_selector = widgets.SelectMultiple(
            options=self.group_order,
            value=tuple(self.group_order),
            description="Groups:",
            layout=widgets.Layout(height="180px"),
        )
        self.new_label_input = widgets.Text(value="selected", description="New Label:")

        self.apply_btn = widgets.Button(description="Apply Selection", button_style="success")
        self.clear_btn = widgets.Button(description="Clear Selection", button_style="warning")
        self.reset_btn = widgets.Button(description="Reset Selected Groups", button_style="danger")
        self.update_btn = widgets.Button(description="Update AnnData", button_style="primary")
        self.export_btn = widgets.Button(description="Export", button_style="info")

        self.group_selector.observe(self._on_group_change, names="value")
        self.apply_btn.on_click(self._apply_selection)
        self.clear_btn.on_click(self._clear_selection)
        self.reset_btn.on_click(self._reset_selected_groups)
        self.update_btn.on_click(self._update_anndata)
        self.export_btn.on_click(self._export_data)

        self._refresh_plot()
        self._set_status_header()

    def _set_status_header(self):
        self.status.value = (
            f"<b>Status:</b> Ready | total cells={len(self.merged)}, groups={len(self.group_order)}, "
            f"image={self.w}x{self.h}, downsample={self.downsample_factor}"
        )

    def _on_group_change(self, change):
        self.displayed_groups = set(change["new"])
        self._clear_selection()
        self._refresh_plot()

    def _refresh_plot(self):
        if len(self.displayed_groups) == 0:
            self._refresh_main_trace(np.array([], dtype=np.int64), np.array([], dtype=object))
            self.status.value = "<b>Status:</b> No displayed groups selected"
            return

        mask = np.isin(self.labels, np.array(list(self.displayed_groups), dtype=object))
        pos = np.where(mask)[0]
        color_by = np.array([self.color_map.get(lbl, "rgb(120,120,120)") for lbl in self.labels], dtype=object)
        self._refresh_main_trace(pos, color_by)
        self.status.value = f"<b>Status:</b> Rendering {len(self.visible_positions)} / {mask.sum()} visible points"

    def _apply_selection(self, _=None):
        if len(self.selected_global_indices) == 0:
            self.status.value = "<b>Status:</b> No selected cells"
            return

        new_label = self.new_label_input.value.strip()
        if not new_label:
            self.status.value = "<b>Status:</b> Enter a valid label"
            return

        cat_col = "predicted_microenvironment"
        if pd.api.types.is_categorical_dtype(self.merged[cat_col]) and new_label not in self.merged[cat_col].cat.categories:
            self.merged[cat_col] = self.merged[cat_col].cat.add_categories([new_label])

        self.merged.loc[self.selected_global_indices, cat_col] = new_label

        # Keep local numpy labels in sync.
        selected_pos = self.merged.index.get_indexer(self.selected_global_indices)
        selected_pos = selected_pos[selected_pos >= 0]
        self.labels[selected_pos] = new_label

        if new_label not in self.color_map:
            extended = self.group_order + [new_label]
            self.color_map = _make_color_map(extended)

        if new_label not in self.group_order:
            self.group_order.append(new_label)
            self.group_selector.options = self.group_order

        changed = len(self.selected_global_indices)
        self._clear_selection()
        self._refresh_plot()
        self.status.value = f"<b>Status:</b> Applied new label '{new_label}' to {changed} cells"

    def _reset_selected_groups(self, _=None):
        try:
            selected_groups = [str(v) for v in self.group_selector.value]
            if len(selected_groups) == 0:
                self.status.value = "<b>Status:</b> Select groups to reset"
                return

            cat_col = "predicted_microenvironment"
            current_labels = self.merged[cat_col].astype(str)
            mask = current_labels.isin(selected_groups)
            target_idx = self.merged.index[mask]

            if len(target_idx) == 0:
                self.status.value = "<b>Status:</b> No cells found for selected groups"
                return

            # Revert selected-group cells back to original labels.
            self.merged.loc[target_idx, cat_col] = self.original_clusters.loc[target_idx].values
            self.current_clusters = self.merged[cat_col].copy()
            self.labels = self.merged[cat_col].astype(str).to_numpy()

            # Rebuild group list from current data to drop labels no longer used.
            self.group_order = [str(v) for v in self.merged[cat_col].dropna().unique()]
            self.color_map = _make_color_map(self.group_order)
            self.group_selector.options = self.group_order
            self.group_selector.value = tuple(self.group_order)
            self.displayed_groups = set(self.group_order)

            changed = len(target_idx)
            self._clear_selection()
            self._refresh_plot()
            self.status.value = f"<b>Status:</b> Reset selected groups for {changed} cells"
        except Exception as exc:
            self.status.value = f"<b>Status:</b> Reset failed: {exc}"

    def _update_anndata(self, _=None):
        self.sp_adata_ref.obs["predicted_microenvironment"] = self.merged["predicted_microenvironment"].values
        self.merged_original["predicted_microenvironment"] = self.merged["predicted_microenvironment"].values
        self.status.value = "<b>Status:</b> AnnData updated"

    def _export_data(self, _=None):
        import __main__ as main

        main.updated_merged_export = self.merged.copy()
        main.updated_clusters_export = self.current_clusters.copy()
        self.status.value = "<b>Status:</b> Exported to updated_merged_export / updated_clusters_export"

    def run(self):
        controls = widgets.VBox(
            [
                widgets.HTML("<h3>Microenvironment selector</h3>"),
                self.selection_mode,
                self.group_selector,
                self.new_label_input,
                self.zoom_slider,
                self.point_size_slider,
                self.opacity_slider,
                widgets.HBox([self.apply_btn, self.clear_btn]),
                widgets.HBox([self.reset_btn, self.update_btn, self.export_btn]),
                self.selection_info,
                self.status,
            ],
            layout=widgets.Layout(width="360px", min_width="360px"),
        )

        self._mount_figure()

        ui = widgets.HBox([controls, self.output], layout=widgets.Layout(width="100%", align_items="flex-start"))
        display(ui)
        return self


class LassoCellSelectorCellType(_BasePlotlySelector):
    """High-performance cell-type relabeling widget using Plotly ScatterGL."""

    def __init__(self, sp_adata, merged_df, lib_id, clusters, downsample_factor=0.25):
        self.original_clusters = _as_aligned_series(clusters, merged_df.index, "predicted_microenvironment")
        self.current_clusters = self.original_clusters.copy()

        super().__init__(sp_adata, merged_df, lib_id, downsample_factor=downsample_factor)

        self.microenv_labels = self.merged["predicted_microenvironment"].astype(str).to_numpy()
        self.celltype_labels = self.merged["predicted_cell_type"].astype(str).to_numpy()

        self.microenv_order = [str(v) for v in self.merged["predicted_microenvironment"].dropna().unique()]
        self.cell_type_order = [str(v) for v in self.merged["predicted_cell_type"].dropna().unique()]

        self.color_map = _make_color_map(self.microenv_order)

        default_ct = self.cell_type_order[0] if self.cell_type_order else None
        default_me = tuple(self.microenv_order[: min(3, len(self.microenv_order))])

        self.cell_type_selector = widgets.Dropdown(
            options=self.cell_type_order,
            value=default_ct,
            description="Target CT:",
            layout=widgets.Layout(width="320px"),
        )
        self.microenv_selector = widgets.SelectMultiple(
            options=self.microenv_order,
            value=default_me,
            description="Microenv:",
            layout=widgets.Layout(height="170px"),
        )
        self.new_cell_type_input = widgets.Text(value="selected_cell_type", description="New CT:")

        self.apply_btn = widgets.Button(description="Apply Selection", button_style="success")
        self.clear_btn = widgets.Button(description="Clear Selection", button_style="warning")
        self.reset_btn = widgets.Button(description="Reset Original Data", button_style="danger")
        self.update_btn = widgets.Button(description="Update AnnData", button_style="primary")
        self.export_btn = widgets.Button(description="Export", button_style="info")

        self.cell_type_selector.observe(self._on_filter_change, names="value")
        self.microenv_selector.observe(self._on_filter_change, names="value")
        self.apply_btn.on_click(self._apply_selection)
        self.clear_btn.on_click(self._clear_selection)
        self.reset_btn.on_click(self._reset_all)
        self.update_btn.on_click(self._update_anndata)
        self.export_btn.on_click(self._export_data)

        self._refresh_plot()
        self._set_status_header()

    def _set_status_header(self):
        self.status.value = (
            f"<b>Status:</b> Ready | total cells={len(self.merged)}, cell types={len(self.cell_type_order)}, "
            f"microenv={len(self.microenv_order)}, image={self.w}x{self.h}, downsample={self.downsample_factor}"
        )

    def _on_filter_change(self, _):
        self._clear_selection()
        self._refresh_plot()

    def _refresh_plot(self):
        target_ct = self.cell_type_selector.value
        microenvs = list(self.microenv_selector.value)

        if (target_ct is None) or (len(microenvs) == 0):
            self._refresh_main_trace(np.array([], dtype=np.int64), np.array([], dtype=object))
            self.status.value = "<b>Status:</b> Select target cell type and at least one microenvironment"
            return

        mask = (self.celltype_labels == str(target_ct)) & np.isin(self.microenv_labels, np.array(microenvs, dtype=object))
        pos = np.where(mask)[0]
        color_by = np.array([self.color_map.get(lbl, "rgb(120,120,120)") for lbl in self.microenv_labels], dtype=object)
        self._refresh_main_trace(pos, color_by)
        self.status.value = f"<b>Status:</b> Rendering {len(self.visible_positions)} / {mask.sum()} visible points"

    def _apply_selection(self, _=None):
        if len(self.selected_global_indices) == 0:
            self.status.value = "<b>Status:</b> No selected cells"
            return

        new_cell_type = self.new_cell_type_input.value.strip()
        if not new_cell_type:
            self.status.value = "<b>Status:</b> Enter a valid cell type"
            return

        cat_col = "predicted_cell_type"
        if pd.api.types.is_categorical_dtype(self.merged[cat_col]) and new_cell_type not in self.merged[cat_col].cat.categories:
            self.merged[cat_col] = self.merged[cat_col].cat.add_categories([new_cell_type])

        self.merged.loc[self.selected_global_indices, cat_col] = new_cell_type

        selected_pos = self.merged.index.get_indexer(self.selected_global_indices)
        selected_pos = selected_pos[selected_pos >= 0]
        self.celltype_labels[selected_pos] = new_cell_type

        if new_cell_type not in self.cell_type_order:
            self.cell_type_order.append(new_cell_type)
            self.cell_type_selector.options = self.cell_type_order

        changed = len(self.selected_global_indices)
        self._clear_selection()
        self._refresh_plot()
        self.status.value = f"<b>Status:</b> Applied new cell type '{new_cell_type}' to {changed} cells"

    def _reset_all(self, _=None):
        if "predicted_cell_type" in self.merged_original.columns:
            self.merged["predicted_cell_type"] = self.merged_original["predicted_cell_type"].copy()
            self.celltype_labels = self.merged["predicted_cell_type"].astype(str).to_numpy()
        
        # Reset to original clusters/microenvironments
        self.merged["predicted_microenvironment"] = self.original_clusters
        self.microenv_labels = self.merged["predicted_microenvironment"].astype(str).to_numpy()
        
        # Reset cell type and microenvironment order to original
        self.cell_type_order = [str(v) for v in self.merged["predicted_cell_type"].dropna().unique()]
        self.microenv_order = [str(v) for v in self.original_clusters.dropna().unique()]
        self.color_map = _make_color_map(self.microenv_order)
        
        self.cell_type_selector.options = self.cell_type_order
        self.microenv_selector.options = self.microenv_order
        
        self.current_clusters = self.original_clusters.copy()
        self._clear_selection()
        self._refresh_plot()
        self.status.value = "<b>Status:</b> Reset complete - all labels and groups restored"

    def _update_anndata(self, _=None):
        self.sp_adata_ref.obs["predicted_cell_type"] = self.merged["predicted_cell_type"].values
        self.merged_original["predicted_cell_type"] = self.merged["predicted_cell_type"].values
        self.status.value = "<b>Status:</b> AnnData updated"

    def _export_data(self, _=None):
        import __main__ as main

        main.updated_merged_celltype_export = self.merged.copy()
        main.updated_clusters_celltype_export = self.current_clusters.copy()
        self.status.value = "<b>Status:</b> Exported to updated_merged_celltype_export / updated_clusters_celltype_export"

    def run(self):
        controls = widgets.VBox(
            [
                widgets.HTML("<h3>Cell type selector</h3>"),
                self.selection_mode,
                self.cell_type_selector,
                self.new_cell_type_input,
                self.microenv_selector,
                self.zoom_slider,
                self.point_size_slider,
                self.opacity_slider,
                widgets.HBox([self.apply_btn, self.clear_btn]),
                widgets.HBox([self.reset_btn, self.update_btn, self.export_btn]),
                self.selection_info,
                self.status,
            ],
            layout=widgets.Layout(width="360px", min_width="360px"),
        )

        self._mount_figure()

        ui = widgets.HBox([controls, self.output], layout=widgets.Layout(width="100%", align_items="flex-start"))
        display(ui)
        return self


class DistanceMicroenvironmentSelector(_BasePlotlySelector):
    """Distance-based microenvironment relabeling widget.

    Supports nearest, centroid, kNN mean, and distance-transform style distances.
    """

    def __init__(self, sp_adata, merged_df, lib_id, clusters, downsample_factor=0.25):
        self.original_clusters = _as_aligned_series(clusters, merged_df.index, "predicted_microenvironment")
        self.current_clusters = self.original_clusters.copy()

        super().__init__(sp_adata, merged_df, lib_id, downsample_factor=downsample_factor)

        # Distance mode does not use lasso/box selection; keep navigation-focused controls.
        self.selection_mode.options = [("Pan", "pan")]
        self.selection_mode.value = "pan"
        self.selection_mode.disabled = True
        self.selection_mode.layout = widgets.Layout(display="none")
        self.fig.update_layout(
            dragmode="pan",
            modebar_remove=["lasso2d", "select2d"],
            modebar_add=["pan2d"],
        )

        self.cell_type_labels = self.merged["predicted_cell_type"].astype(str).to_numpy()
        self.microenv_labels = self.merged["predicted_microenvironment"].astype(str).to_numpy()
        self.cell_type_order = [str(v) for v in self.merged["predicted_cell_type"].dropna().unique()]
        self.microenv_order = [str(v) for v in self.merged["predicted_microenvironment"].dropna().unique()]

        self.method_selector = widgets.Dropdown(
            options=[
                ("Nearest distance", "nearest"),
                ("Centroid distance", "centroid"),
                ("kNN mean distance", "knn_mean"),
                ("Distance transform", "distance_transform"),
            ],
            value="nearest",
            description="Method:",
            layout=widgets.Layout(width="340px"),
        )
        self.reference_selector = widgets.SelectMultiple(
            options=self.cell_type_order,
            value=tuple(),
            description="Reference CT:",
            layout=widgets.Layout(height="170px"),
        )
        self.target_microenv_selector = widgets.SelectMultiple(
            options=self.microenv_order,
            value=tuple(),
            description="Target ME:",
            layout=widgets.Layout(height="170px"),
        )
        self.distance_unit = widgets.Dropdown(
            options=[("um", "um"), ("pixel", "px")],
            value="um",
            description="Dist Unit:",
            layout=widgets.Layout(width="200px"),
        )
        self.k_value = widgets.BoundedIntText(value=5, min=1, max=5000, description="k:", layout=widgets.Layout(width="180px"))
        self.threshold_count = widgets.Dropdown(options=[1, 2, 3], value=3, description="#Thresholds:", layout=widgets.Layout(width="220px"))
        self.threshold_1 = widgets.FloatText(value=50.0, description="Thr 1:", layout=widgets.Layout(width="220px"))
        self.threshold_2 = widgets.FloatText(value=100.0, description="Thr 2:", layout=widgets.Layout(width="220px"))
        self.threshold_3 = widgets.FloatText(value=150.0, description="Thr 3:", layout=widgets.Layout(width="220px"))
        self.relabel_target = widgets.Dropdown(
            options=[("all", "all"), ("th1", "th1")],
            value="all",
            description="Relabel Target:",
            layout=widgets.Layout(width="260px"),
        )
        self.new_label_input = widgets.Text(value="selected_microenv", description="New Label:", layout=widgets.Layout(width="320px"))
        self.label_prefix = widgets.Text(value="microenv_dist", description="Label Prefix:", layout=widgets.Layout(width="320px"))

        self.compute_btn = widgets.Button(description="Compute Distances", button_style="info")
        self.apply_btn = widgets.Button(description="Apply Labels", button_style="success")
        self.clear_btn = widgets.Button(description="Clear Preview", button_style="warning")
        self.update_btn = widgets.Button(description="Update AnnData", button_style="primary")
        self.export_btn = widgets.Button(description="Export", button_style="info")

        self.distance_summary = widgets.HTML(value="<b>Distance summary:</b> not computed")
        self.assignment_summary = widgets.HTML(value="<b>Assignment:</b> not computed")

        self._distance_cache = {}
        self._distance_values = None
        self._threshold_masks = {}

        # Keep reference CT color distinct from threshold palette (up to 3 colors).
        self.reference_color = "rgb(106,61,154)"
        self.target_color = "rgb(140,140,140)"
        self.threshold_colors = ["rgb(230,85,13)", "rgb(49,130,189)", "rgb(49,163,84)"]

        # Distance calculations run in image pixel space (x,y). Convert to/from physical um for UI.
        self.pixel_size_um = self._infer_pixel_size_um(default=0.5)

        self.selection_info.value = "<b>Selected:</b> N/A (distance mode)"

        self.threshold_count.observe(self._on_threshold_count_change, names="value")
        self.distance_unit.observe(self._on_distance_unit_change, names="value")
        self.reference_selector.observe(self._on_filter_change, names="value")
        self.target_microenv_selector.observe(self._on_filter_change, names="value")
        self.compute_btn.on_click(self._compute_and_preview)
        self.apply_btn.on_click(self._apply_selection)
        self.clear_btn.on_click(self._clear_selection)
        self.update_btn.on_click(self._update_anndata)
        self.export_btn.on_click(self._export_data)

        self._on_threshold_count_change(None)
        self._refresh_plot()
        self._set_status_header()

    def _set_status_header(self):
        self.status.value = (
            f"<b>Status:</b> Ready | total cells={len(self.merged)}, cell types={len(self.cell_type_order)}, "
            f"microenv={len(self.microenv_order)}, image={self.w}x{self.h}, downsample={self.downsample_factor}"
        )

    def _on_plot_selection(self, trace, points, selector):
        # Intentionally disabled in distance mode.
        return

    def _infer_pixel_size_um(self, default=0.5):
        # Prefer parsing mpp from image key naming convention: "{mpp}_mpp_..."
        image_keys = self.sp_adata_ref.uns.get("spatial", {}).get(self.lib_id, {}).get("images", {}).keys()
        for key in image_keys:
            key_str = str(key)
            if "_mpp" in key_str:
                try:
                    return float(key_str.split("_mpp", 1)[0])
                except Exception:
                    continue
        return float(default)

    def _distance_to_display_units(self, arr):
        if self.distance_unit.value == "um":
            return np.asarray(arr, dtype=np.float64) * float(self.pixel_size_um)
        return np.asarray(arr, dtype=np.float64)

    def _display_to_distance_units(self, value):
        if self.distance_unit.value == "um":
            return float(value) / float(self.pixel_size_um)
        return float(value)

    def _distance_unit_label(self):
        return "um" if self.distance_unit.value == "um" else "px"

    def _on_threshold_count_change(self, _):
        n = int(self.threshold_count.value)
        self.threshold_2.disabled = n < 2
        self.threshold_3.disabled = n < 3
        options = [("all", "all")] + [(f"th{i}", f"th{i}") for i in range(1, n + 1)]
        current = self.relabel_target.value
        self.relabel_target.options = options
        if current not in {v for _, v in options}:
            self.relabel_target.value = "all"
        self._update_threshold_labels()
        self._reset_preview_state()
        self._refresh_plot()

    def _on_distance_unit_change(self, _):
        self._update_threshold_labels()
        self._reset_preview_state()
        self._refresh_plot()

    def _update_threshold_labels(self):
        unit = self._distance_unit_label()
        self.threshold_1.description = f"Thr 1 ({unit}):"
        self.threshold_2.description = f"Thr 2 ({unit}):"
        self.threshold_3.description = f"Thr 3 ({unit}):"

    def _on_filter_change(self, _):
        self._reset_preview_state()
        self._refresh_plot()

    def _reset_preview_state(self):
        self._distance_values = None
        self._threshold_masks = {}
        self.distance_summary.value = "<b>Distance summary:</b> not computed"
        self.assignment_summary.value = "<b>Assignment:</b> not computed"

    def _get_thresholds(self):
        n = int(self.threshold_count.value)
        raw = [self.threshold_1.value, self.threshold_2.value, self.threshold_3.value][:n]
        thresholds = [float(v) for v in raw]
        if any(v < 0 for v in thresholds):
            raise ValueError("Thresholds must be non-negative")
        if any(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1)):
            raise ValueError("Thresholds must be strictly increasing")
        return thresholds

    def _coords_for_distance_transform(self):
        if ("array_col" in self.merged.columns) and ("array_row" in self.merged.columns):
            x = self.merged["array_col"].to_numpy(dtype=np.float64)
            y = self.merged["array_row"].to_numpy(dtype=np.float64)
            return x, y
        return self.x.astype(np.float64), self.y.astype(np.float64)

    def _distance_key(self, method, refs, k):
        return (method, tuple(sorted(refs)), int(k))

    def _compute_distances(self, method, refs, k):
        key = self._distance_key(method, refs, k)
        if key in self._distance_cache:
            return self._distance_cache[key]

        ref_mask = np.isin(self.cell_type_labels, np.array(refs, dtype=object))
        ref_idx = np.where(ref_mask)[0]
        if len(ref_idx) == 0:
            raise ValueError("No cells found for selected reference cell types")

        pts = np.column_stack([self.x, self.y])
        ref_pts = pts[ref_idx]

        if method == "nearest":
            tree = cKDTree(ref_pts)
            dists, _ = tree.query(pts, k=1)
            out = dists.astype(np.float64)
        elif method == "centroid":
            centroid = np.mean(ref_pts, axis=0)
            out = np.linalg.norm(pts - centroid[None, :], axis=1).astype(np.float64)
        elif method == "knn_mean":
            kk = max(1, min(int(k), len(ref_idx)))
            tree = cKDTree(ref_pts)
            dists, _ = tree.query(pts, k=kk)
            if kk == 1:
                out = dists.astype(np.float64)
            else:
                out = np.mean(dists, axis=1).astype(np.float64)
        elif method == "distance_transform":
            gx, gy = self._coords_for_distance_transform()
            gx = np.round(gx - np.min(gx)).astype(np.int64)
            gy = np.round(gy - np.min(gy)).astype(np.int64)
            h = int(np.max(gy)) + 1
            w = int(np.max(gx)) + 1
            h = max(h, 2)
            w = max(w, 2)

            mask = np.zeros((h, w), dtype=bool)
            mask[gy[ref_idx], gx[ref_idx]] = True
            dt = distance_transform_edt(~mask)
            out = dt[gy, gx].astype(np.float64)
        else:
            raise ValueError(f"Unknown method: {method}")

        self._distance_cache[key] = out
        return out

    def _make_threshold_masks(self, dists, thresholds, target_mask):
        masks = {}
        assigned = np.zeros(len(dists), dtype=bool)
        for i, thr in enumerate(thresholds, start=1):
            m = target_mask & (~assigned) & (dists <= thr)
            assigned[m] = True
            masks[f"th{i}"] = m
        unassigned = target_mask & (~assigned)
        return masks, unassigned

    def _format_distance_stats(self, d):
        n = len(d)
        if n == 0:
            return "count=0"
        p10 = float(np.percentile(d, 10))
        p50 = float(np.percentile(d, 50))
        p90 = float(np.percentile(d, 90))
        unit = self._distance_unit_label()
        return (
            f"count={n}, min={float(np.min(d)):.3f} {unit}, p10={p10:.3f} {unit}, "
            f"median={p50:.3f} {unit}, p90={p90:.3f} {unit}, max={float(np.max(d)):.3f} {unit}"
        )

    def _format_distance_summary(self, dists, target_mask=None, target_labels=None):
        if target_mask is None:
            target_mask = np.ones(len(dists), dtype=bool)

        d_target = self._distance_to_display_units(np.asarray(dists)[target_mask])
        target_stats = self._format_distance_stats(d_target)
        if target_labels:
            target_name = "+".join(target_labels)
        else:
            target_name = "selected_target"

        return f"<b>Distance summary (target ME: {target_name}):</b> {target_stats}"

    def _format_assignment_summary(self, dists, threshold_masks, unassigned_mask, prefix):
        if not threshold_masks:
            return "<b>Assignment:</b> none"

        unit = self._distance_unit_label()
        d = np.asarray(dists)
        parts = []
        # 色リスト（最大3つまで）
        colors = self.threshold_colors
        out_of_range_color = "rgb(180,180,180)"  # グレー

        for i in range(1, int(self.threshold_count.value) + 1):
            key = f"th{i}"
            mask = threshold_masks.get(key)
            color = colors[i-1] if i-1 < len(colors) else "rgb(120,120,120)"
            color_box = f'<span style="display:inline-block;width:1em;height:1em;background:{color};border:1px solid #888;margin-right:0.3em;"></span>'
            if mask is None:
                continue
            cnt = int(np.sum(mask))
            if cnt > 0:
                med = float(np.median(self._distance_to_display_units(d[mask])))
                parts.append(f"{color_box}{prefix}_{i}: {cnt} (median={med:.3f} {unit})")
            else:
                parts.append(f"{color_box}{prefix}_{i}: 0 (median=n/a)")

        out_cnt = int(np.sum(unassigned_mask))
        color_box = f'<span style="display:inline-block;width:1em;height:1em;background:{out_of_range_color};border:1px solid #888;margin-right:0.3em;"></span>'
        if out_cnt > 0:
            out_med = float(np.median(self._distance_to_display_units(d[unassigned_mask])))
            parts.append(f"{color_box}out_of_range: {out_cnt} (median={out_med:.3f} {unit})")
        else:
            parts.append(f"{color_box}out_of_range: 0 (median=n/a)")

        return "<b>Assignment:</b> " + " | ".join(parts)

    def _compute_and_preview(self, _=None):
        try:
            refs = list(self.reference_selector.value)
            if len(refs) == 0:
                raise ValueError("Select at least one reference cell type")

            target_me = list(self.target_microenv_selector.value)
            if len(target_me) == 0:
                raise ValueError("Select at least one target microenvironment")

            method = self.method_selector.value
            k = int(self.k_value.value)
            thresholds_display = self._get_thresholds()
            thresholds = [self._display_to_distance_units(v) for v in thresholds_display]
            prefix = self.label_prefix.value.strip()
            if not prefix:
                raise ValueError("Enter a non-empty label prefix")

            dists = self._compute_distances(method, refs, k)
            target_mask = np.isin(self.microenv_labels, np.array(target_me, dtype=object))
            threshold_masks, unassigned = self._make_threshold_masks(dists, thresholds, target_mask)

            self._distance_values = dists
            self._threshold_masks = threshold_masks
            self.distance_summary.value = self._format_distance_summary(
                dists,
                target_mask=target_mask,
                target_labels=target_me,
            )
            self.assignment_summary.value = self._format_assignment_summary(
                dists=dists,
                threshold_masks=threshold_masks,
                unassigned_mask=unassigned,
                prefix=prefix,
            )

            self._refresh_plot()
            self.status.value = "<b>Status:</b> Preview updated"
        except Exception as exc:
            self.status.value = f"<b>Status:</b> Compute failed: {exc}"

    def _refresh_plot(self):
        refs = list(self.reference_selector.value)
        target_me = list(self.target_microenv_selector.value)

        ref_mask = np.isin(self.cell_type_labels, np.array(refs, dtype=object)) if refs else np.zeros(len(self.merged), dtype=bool)
        target_mask = np.isin(self.microenv_labels, np.array(target_me, dtype=object)) if target_me else np.zeros(len(self.merged), dtype=bool)
        visible_mask = ref_mask | target_mask

        if not np.any(visible_mask):
            self._refresh_main_trace(np.array([], dtype=np.int64), np.array([], dtype=object))
            self.status.value = "<b>Status:</b> Select reference CT and/or target microenvironment"
            return

        color_by = np.full(len(self.merged), self.reference_color, dtype=object)
        color_by[target_mask] = self.target_color

        if self._distance_values is not None and self._threshold_masks:
            for i in range(1, int(self.threshold_count.value) + 1):
                key = f"th{i}"
                mask = self._threshold_masks.get(key)
                if mask is not None:
                    color_by[mask] = self.threshold_colors[i - 1]

        pos = np.where(visible_mask)[0]
        self._refresh_main_trace(pos, color_by)
        self.status.value = f"<b>Status:</b> Rendering {len(self.visible_positions)} / {int(np.sum(visible_mask))} visible points"

    def _clear_selection(self, _=None):
        # Return to initial view: no reference CT / no target microenvironment selected.
        with self.reference_selector.hold_trait_notifications(), self.target_microenv_selector.hold_trait_notifications():
            self.reference_selector.value = tuple()
            self.target_microenv_selector.value = tuple()
        self._reset_preview_state()
        self._refresh_plot()
        self.status.value = "<b>Status:</b> Cleared to initial view (no dots)"

    def _apply_selection(self, _=None):
        if (self._distance_values is None) or (not self._threshold_masks):
            self.status.value = "<b>Status:</b> Run 'Compute Distances' first"
            return

        new_label = self.new_label_input.value.strip()
        if not new_label:
            self.status.value = "<b>Status:</b> Enter a valid new label"
            return

        target_key = self.relabel_target.value
        if target_key == "all":
            relabel_mask = np.zeros(len(self.merged), dtype=bool)
            for mask in self._threshold_masks.values():
                relabel_mask |= mask
        else:
            relabel_mask = self._threshold_masks.get(target_key)
            if relabel_mask is None:
                self.status.value = f"<b>Status:</b> {target_key} is not available for current threshold setting"
                return

        target_idx = self.merged.index[relabel_mask]
        if len(target_idx) == 0:
            self.status.value = "<b>Status:</b> No cells matched selected relabel target"
            return

        cat_col = "predicted_microenvironment"
        if pd.api.types.is_categorical_dtype(self.merged[cat_col]) and new_label not in self.merged[cat_col].cat.categories:
            self.merged[cat_col] = self.merged[cat_col].cat.add_categories([new_label])

        self.merged.loc[target_idx, cat_col] = new_label
        self.current_clusters = self.merged[cat_col].copy()
        self.microenv_labels = self.merged[cat_col].astype(str).to_numpy()
        self.microenv_order = [str(v) for v in self.merged[cat_col].dropna().unique()]
        self.target_microenv_selector.options = self.microenv_order

        changed = len(target_idx)
        self._reset_preview_state()
        self._refresh_plot()
        self.status.value = f"<b>Status:</b> Applied '{new_label}' to {changed} cells ({target_key})"

    def _update_anndata(self, _=None):
        self.sp_adata_ref.obs["predicted_microenvironment"] = self.merged["predicted_microenvironment"].values
        self.merged_original["predicted_microenvironment"] = self.merged["predicted_microenvironment"].values
        self.status.value = "<b>Status:</b> AnnData updated"

    def _export_data(self, _=None):
        import __main__ as main

        main.updated_merged_distance_export = self.merged.copy()
        main.updated_clusters_distance_export = self.current_clusters.copy()
        self.status.value = "<b>Status:</b> Exported to updated_merged_distance_export / updated_clusters_distance_export"

    def run(self):
        controls = widgets.VBox(
            [
                widgets.HTML("<h3>Distance microenvironment selector</h3>"),
                self.method_selector,
                self.reference_selector,
                self.target_microenv_selector,
                widgets.HBox([self.k_value, self.threshold_count, self.distance_unit]),
                widgets.HBox([self.threshold_1, self.threshold_2, self.threshold_3]),
                self.label_prefix,
                self.relabel_target,
                self.new_label_input,
                self.zoom_slider,
                self.point_size_slider,
                self.opacity_slider,
                widgets.HBox([self.compute_btn, self.apply_btn, self.clear_btn]),
                widgets.HBox([self.update_btn, self.export_btn]),
                self.distance_summary,
                self.assignment_summary,
                self.status,
            ],
            layout=widgets.Layout(width="380px", min_width="380px"),
        )

        self._mount_figure()
        ui = widgets.HBox([controls, self.output], layout=widgets.Layout(width="100%", align_items="flex-start"))
        display(ui)
        return self


def lasso_selection_microenvironment(sp_adata, merged, lib_id, clusters, downsample_factor=0.25):
    """Launch high-performance microenvironment selector."""
    selector = LassoCellSelectorMicroenvironment(
        sp_adata=sp_adata,
        merged_df=merged,
        lib_id=lib_id,
        clusters=clusters,
        downsample_factor=downsample_factor,
    )
    return selector.run()


def lasso_selection_cell_type(sp_adata, merged, lib_id, clusters, downsample_factor=0.25):
    """Launch high-performance cell type selector."""
    selector = LassoCellSelectorCellType(
        sp_adata=sp_adata,
        merged_df=merged,
        lib_id=lib_id,
        clusters=clusters,
        downsample_factor=downsample_factor,
    )
    return selector.run()


def distance_selection_microenvironment(sp_adata, merged, lib_id, clusters, downsample_factor=0.25):
    """Launch distance-based microenvironment selector."""
    selector = DistanceMicroenvironmentSelector(
        sp_adata=sp_adata,
        merged_df=merged,
        lib_id=lib_id,
        clusters=clusters,
        downsample_factor=downsample_factor,
    )
    return selector.run()
