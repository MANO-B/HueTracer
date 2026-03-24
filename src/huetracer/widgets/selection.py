import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image as PILImage
import plotly.graph_objects as go


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
