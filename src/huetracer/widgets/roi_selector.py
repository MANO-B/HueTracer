# src/huetracer/widgets/roi_selector.py

"""Interactive ROI (Region of Interest) selector for spatial transcriptomics data."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scanpy as sc
import tifffile
import zarr
from IPython.display import clear_output, display

from huetracer.core.image_ops import get_image_dimensions, save_cropped_image
from huetracer.core.spatial_roi import (
    CropConfig,
    apply_array_row_col_mask,
    auto_detect_tissue_bounds,
    get_row_col_range,
    make_array_row_col_mask,
    sanitize_bounds,
)
from huetracer.io.visium import load_reference_images


class ROISelectorWidget:
    """Interactive ROI (Region of Interest) selector for AnnData spatial data.
    
    Allows users to select a rectangular region of interest (ROI) from spatial
    transcriptomics data using interactive range sliders. Displays the selected
    region overlaid on the spatial plot.
    
    Attributes:
        adata: Input AnnData object with spatial coordinates (array_row, array_col)
        img_key: Key for image data in adata.uns['spatial']. Auto-detected if None.
        basis: Basis key for spatial plotting (default: 'spatial_cropped_150_buffer')
        roi_x1, roi_x2: Selected row range (Y-axis)
        roi_y1, roi_y2: Selected column range (X-axis)
    
    Example:
        >>> roi_widget = ROISelectorWidget(adata)
        >>> roi_widget.display()
        >>> # After clicking "Confirm & Save ROI"
        >>> x1, x2, y1, y2 = roi_widget.get_roi()
    """
    
    def __init__(
        self,
        adata: Any,
        img_key: Optional[str] = None,
        basis: str = "spatial_cropped_150_buffer",
    ) -> None:
        """Initialize ROI selector widget.
        
        Args:
            adata: AnnData object with spatial data (obs columns: array_row, array_col)
            img_key: Key for image in adata.uns['spatial']. Auto-detected if None.
            basis: Basis key for spatial plot (default: 'spatial_cropped_150_buffer')
            
        Raises:
            ValueError: If adata is None or missing required spatial coordinates.
        """
        if adata is None:
            raise ValueError("AnnData object is required")
        
        if "array_row" not in adata.obs or "array_col" not in adata.obs:
            raise ValueError("AnnData must have 'array_row' and 'array_col' in obs")
        
        self.adata = adata
        self.img_key = img_key
        self.basis = basis
        
        # Auto-detect img_key if not provided
        if self.img_key is None:
            img_keys = [k for k in adata.uns.keys() if "mpp" in k and "buffer" in k]
            self.img_key = img_keys[0] if img_keys else None
        
        # Get data ranges
        self.min_row = int(adata.obs["array_row"].min())
        self.max_row = int(adata.obs["array_row"].max())
        self.min_col = int(adata.obs["array_col"].min())
        self.max_col = int(adata.obs["array_col"].max())
        
        print(f"📊 Data Range -- Row(Y): {self.min_row}-{self.max_row}, Col(X): {self.min_col}-{self.max_col}")
        
        # Selected ROI values
        self.roi_x1: Optional[int] = None
        self.roi_x2: Optional[int] = None
        self.roi_y1: Optional[int] = None
        self.roi_y2: Optional[int] = None
        
        self._init_ui_components()
        self.ui = self._build_ui()
    
    def _init_ui_components(self) -> None:
        style = {"description_width": "initial"}
        layout = widgets.Layout(width="600px")
        
        self.x_range_slider = widgets.IntRangeSlider(
            value=[self.min_col, min(self.min_col + 256, self.max_col)],
            min=self.min_col,
            max=self.max_col,
            step=256,
            description="X Range (array_col):",
            style=style,
            layout=layout,
        )
        
        self.y_range_slider = widgets.IntRangeSlider(
            value=[self.min_row, min(self.min_row + 256, self.max_row)],
            min=self.min_row,
            max=self.max_row,
            step=256,
            description="Y Range (array_row):",
            style=style,
            layout=layout,
        )
        
        self.btn_update = widgets.Button(
            description="Update Plot",
            button_style="info",
            icon="refresh",
        )
        self.btn_confirm = widgets.Button(
            description="Confirm & Save ROI",
            button_style="success",
            icon="check",
        )
        
        self.output_area = widgets.Output()
        
        self.btn_update.on_click(self._on_update_plot)
        self.btn_confirm.on_click(self._on_confirm_roi)
    
    def _build_ui(self) -> widgets.Widget:
        return widgets.VBox([
            widgets.HTML("<h3>🔍 Select Preview ROI</h3>"),
            self.x_range_slider,
            self.y_range_slider,
            widgets.HBox([self.btn_update, self.btn_confirm]),
            self.output_area,
        ])
    
    def _on_update_plot(self, b):
        with self.output_area:
            clear_output(wait=True)
            plt.close("all")
            
            x1, x2 = self.x_range_slider.value
            y1, y2 = self.y_range_slider.value
            
            print(f"Filtering: Row {y1}-{y2}, Col {x1}-{x2} ...")
            
            mask = (
                (self.adata.obs["array_row"] >= y1) &
                (self.adata.obs["array_row"] <= y2) &
                (self.adata.obs["array_col"] >= x1) &
                (self.adata.obs["array_col"] <= x2)
            )
            
            if np.sum(mask) == 0:
                print("⚠️ No spots in selected range.")
                return
            
            subset = self.adata[mask].copy()
            
            fig, ax = plt.subplots(figsize=(7, 7))
            if self.img_key:
                sc.pl.spatial(
                    subset,
                    color="n_counts",
                    img_key=self.img_key,
                    basis=self.basis,
                    cmap="Reds",
                    show=False,
                    ax=ax,
                    title=f"ROI: X[{x1}:{x2}], Y[{y1}:{y2}] ({np.sum(mask)} spots)",
                )
            else:
                sc.pl.spatial(
                    subset,
                    color="n_counts",
                    cmap="Reds",
                    show=False,
                    ax=ax,
                    title=f"ROI: X[{x1}:{x2}], Y[{y1}:{y2}] ({np.sum(mask)} spots)",
                )
            plt.show()
            del subset
    
    def _on_confirm_roi(self, b):
        with self.output_area:
            # Y Range Slider (Row) -> x1, x2
            # X Range Slider (Col) -> y1, y2
            self.roi_x1, self.roi_x2 = self.y_range_slider.value
            self.roi_y1, self.roi_y2 = self.x_range_slider.value
            
            print("-" * 30)
            print("✅ ROI Variables Saved!")
            print(f"roi_x1 = {self.roi_x1}")
            print(f"roi_x2 = {self.roi_x2}")
            print(f"roi_y1 = {self.roi_y1}")
            print(f"roi_y2 = {self.roi_y2}")
            print("-" * 30)
            print("Done! Go ahead.")
    
    def get_roi(self) -> Tuple[int, int, int, int]:
        """Return selected ROI coordinates (x1, x2, y1, y2).
        
        Returns:
            Tuple[int, int, int, int]: (roi_x1, roi_x2, roi_y1, roi_y2)
            
        Raises:
            ValueError: If ROI has not been confirmed yet.
        """
        if None in (self.roi_x1, self.roi_x2, self.roi_y1, self.roi_y2):
            raise ValueError("ROI not yet confirmed. Please click 'Confirm & Save ROI' first.")
        return (self.roi_x1, self.roi_x2, self.roi_y1, self.roi_y2)
    
    def display(self) -> None:
        display(self.ui)


def create_roi_selector_widget(
    adata: Any,
    img_key: Optional[str] = None,
) -> ROISelectorWidget:
    """Create and return a ROI selector widget instance.
    
    Args:
        adata: AnnData object with spatial data
        img_key: Optional image key. Auto-detected if None.
        
    Returns:
        ROISelectorWidget: Initialized widget instance
        
    Example:
        >>> roi_widget = create_roi_selector_widget(sp_adata)
        >>> roi_widget.display()
    """
    return ROISelectorWidget(adata, img_key=img_key)


class SpatialMaskSelectorWidget:
    """ROI selector focused on array_row/array_col filtering workflow.

    After clicking Apply, you can access:
    - sp_adata
    - mask_x1_val, mask_x2_val, mask_y1_val, mask_y2_val
    """

    def __init__(
        self,
        sp_adata_raw: Any,
        img_key: str = "0.5_mpp_150_buffer",
        basis: str = "spatial_cropped_150_buffer",
        color_key: str = "bin_count",
    ) -> None:
        if sp_adata_raw is None:
            raise ValueError("sp_adata_raw is required")
        if "array_row" not in sp_adata_raw.obs or "array_col" not in sp_adata_raw.obs:
            raise ValueError("AnnData must have 'array_row' and 'array_col' in obs")

        self.sp_adata_raw = sp_adata_raw
        self.img_key = img_key
        self.basis = basis
        self.color_key = color_key

        ranges = get_row_col_range(sp_adata_raw)
        self.row_min = ranges["row_min"]
        self.row_max = ranges["row_max"]
        self.col_min = ranges["col_min"]
        self.col_max = ranges["col_max"]

        self.sp_adata = None
        self.mask_x1_val = None
        self.mask_x2_val = None
        self.mask_y1_val = None
        self.mask_y2_val = None

        self._init_ui_components()
        self.ui = self._build_ui()

    def _init_ui_components(self) -> None:
        style = {"description_width": "initial"}
        layout_slider = widgets.Layout(width="420px")

        self.mask_small_x1 = widgets.IntSlider(
            value=self.row_min,
            min=self.row_min,
            max=self.row_max,
            step=1,
            description="min_x1 (row ≥)",
            style=style,
            layout=layout_slider,
        )
        self.mask_small_x2 = widgets.IntSlider(
            value=self.row_max,
            min=self.row_min,
            max=self.row_max,
            step=1,
            description="max_x2 (row ≤)",
            style=style,
            layout=layout_slider,
        )
        self.mask_small_y1 = widgets.IntSlider(
            value=self.col_min,
            min=self.col_min,
            max=self.col_max,
            step=1,
            description="min_y1 (col ≥)",
            style=style,
            layout=layout_slider,
        )
        self.mask_small_y2 = widgets.IntSlider(
            value=self.col_max,
            min=self.col_min,
            max=self.col_max,
            step=1,
            description="max_y2 (col ≤)",
            style=style,
            layout=layout_slider,
        )

        self.btn_preview = widgets.Button(description="👀 Preview (bin_count)", button_style="info")
        self.btn_apply = widgets.Button(description="✅ Apply & create sp_adata", button_style="success")
        self.btn_full = widgets.Button(description="↔️ Full range", button_style="")
        self.out = widgets.Output()

        self.btn_preview.on_click(self.preview)
        self.btn_apply.on_click(self.apply)
        self.btn_full.on_click(self.full_range)

    def _build_ui(self) -> widgets.Widget:
        return widgets.VBox(
            [
                widgets.HTML("<h3>🧭 ROI selector (array_row/array_col) with Scanpy spatial preview</h3>"),
                widgets.HBox([self.btn_full, self.btn_preview, self.btn_apply]),
                widgets.HTML("<hr>"),
                self.mask_small_x1,
                self.mask_small_x2,
                self.mask_small_y1,
                self.mask_small_y2,
                widgets.HTML("<hr>"),
                self.out,
            ]
        )

    def _current_bounds(self):
        return sanitize_bounds(
            self.mask_small_x1.value,
            self.mask_small_x2.value,
            self.mask_small_y1.value,
            self.mask_small_y2.value,
        )

    def _make_mask(self):
        x1, x2, y1, y2 = self._current_bounds()
        if self.mask_small_x1.value != x1:
            self.mask_small_x1.value = x1
        if self.mask_small_x2.value != x2:
            self.mask_small_x2.value = x2
        if self.mask_small_y1.value != y1:
            self.mask_small_y1.value = y1
        if self.mask_small_y2.value != y2:
            self.mask_small_y2.value = y2
        return make_array_row_col_mask(self.sp_adata_raw, x1, x2, y1, y2)

    def preview(self, _=None) -> None:
        with self.out:
            clear_output(wait=True)
            progress_label = widgets.Label('Drawing...')
            display(progress_label)
            plt.close('all')
            sc.set_figure_params(fontsize=20, figsize=[7, 7])

            mask = self._make_mask()
            n = int(mask.sum())
            total = self.sp_adata_raw.n_obs
            print(f"Selected: {n:,}/{total:,} ({n/total*100:.2f}%)")
            print(
                f"row: [{self.mask_small_x1.value}, {self.mask_small_x2.value}] | "
                f"col: [{self.mask_small_y1.value}, {self.mask_small_y2.value}]"
            )

            if n == 0:
                print("⚠️ No cells selected.")
                progress_label.value = ''
                return

            dtmp = self.sp_adata_raw[mask].copy()
            try:
                fig = sc.pl.spatial(
                    dtmp,
                    color=[self.color_key],
                    img_key=self.img_key,
                    basis=self.basis,
                    s=4,
                    save=False,
                    show=False,
                    return_fig=True,
                )
            except TypeError:
                ax = sc.pl.spatial(
                    dtmp,
                    color=[self.color_key],
                    img_key=self.img_key,
                    basis=self.basis,
                    s=4,
                    save=False,
                    show=False,
                )
                fig = ax.figure if hasattr(ax, "figure") else plt.gcf()

            display(fig)
            progress_label.value = ''
            plt.close(fig)
            del dtmp

    def apply(self, _=None) -> None:
        with self.out:
            clear_output(wait=True)

            result = apply_array_row_col_mask(
                self.sp_adata_raw,
                self.mask_small_x1.value,
                self.mask_small_x2.value,
                self.mask_small_y1.value,
                self.mask_small_y2.value,
            )
            if result["n_selected"] == 0 or result["sp_adata"] is None:
                print("❌ No cells selected. sp_adata not created.")
                return

            self.sp_adata = result["sp_adata"]
            self.mask_x1_val = result["mask_x1_val"]
            self.mask_x2_val = result["mask_x2_val"]
            self.mask_y1_val = result["mask_y1_val"]
            self.mask_y2_val = result["mask_y2_val"]

            print("✅ sp_adata created.")
            print(f"sp_adata.n_obs = {self.sp_adata.n_obs:,}")
            print(
                "mask_small_x1..y2 = "
                f"{self.mask_x1_val}, {self.mask_x2_val}, {self.mask_y1_val}, {self.mask_y2_val}"
            )
            print("Done! Go ahead.")

    def full_range(self, _=None) -> None:
        self.mask_small_x1.value = self.row_min
        self.mask_small_x2.value = self.row_max
        self.mask_small_y1.value = self.col_min
        self.mask_small_y2.value = self.col_max
        self.preview()

    def get_selected(self):
        """Return selected data and bounds for downstream analysis."""
        if self.sp_adata is None:
            raise ValueError("sp_adata is not created yet. Click '✅ Apply & create sp_adata' first.")
        return (
            self.sp_adata,
            self.mask_x1_val,
            self.mask_x2_val,
            self.mask_y1_val,
            self.mask_y2_val,
        )

    def display(self) -> None:
        display(self.ui)
        self.preview()


def create_spatial_mask_selector_widget(
    sp_adata_raw: Any,
    img_key: str = "0.5_mpp_150_buffer",
    basis: str = "spatial_cropped_150_buffer",
    color_key: str = "bin_count",
) -> SpatialMaskSelectorWidget:
    """Factory helper for SpatialMaskSelectorWidget."""
    return SpatialMaskSelectorWidget(
        sp_adata_raw=sp_adata_raw,
        img_key=img_key,
        basis=basis,
        color_key=color_key,
    )


class ImageCropperWidget:
    """Interactive widget for cropping large Visium HD images in notebooks."""

    def __init__(
        self,
        source_image_path: str,
        expression_path_8um: str,
        output_path: str,
        step: int = 512,
        default_width: int = 4000,
        default_height: int = 4000,
    ):
        """Initialize cropper UI and load preview metadata."""
        self.source_image_path = source_image_path
        self.expression_path_8um = expression_path_8um
        self.output_path = output_path
        self.step = int(step)

        self.full_width, self.full_height = get_image_dimensions(source_image_path)
        self.ref_images = load_reference_images(expression_path_8um)

        if "lowres" in self.ref_images:
            lowres = self.ref_images["lowres"]
            self.scale_x = lowres.width / self.full_width
            self.scale_y = lowres.height / self.full_height
        else:
            self.scale_x = self.scale_y = 1.0

        self._init_ui_components(default_width, default_height)

    def _init_ui_components(self, default_width: int, default_height: int):
        """Initialize all UI widgets."""
        style = {"description_width": "initial"}
        layout_slider = widgets.Layout(width="400px")
        layout_text = widgets.Layout(width="100px")

        self.x_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.full_width,
            step=1,
            description="",
            layout=layout_slider,
        )
        self.x_text = widgets.IntText(
            value=0,
            description="Start X:",
            style=style,
            layout=layout_text,
        )
        widgets.jslink((self.x_slider, "value"), (self.x_text, "value"))

        self.y_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.full_height,
            step=1,
            description="",
            layout=layout_slider,
        )
        self.y_text = widgets.IntText(
            value=0,
            description="Start Y:",
            style=style,
            layout=layout_text,
        )
        widgets.jslink((self.y_slider, "value"), (self.y_text, "value"))

        self.w_slider = widgets.IntSlider(
            value=default_width,
            min=100,
            max=self.full_width,
            step=1,
            description="",
            layout=layout_slider,
        )
        self.w_text = widgets.IntText(
            value=default_width,
            description="Width:",
            style=style,
            layout=layout_text,
        )
        widgets.jslink((self.w_slider, "value"), (self.w_text, "value"))

        self.h_slider = widgets.IntSlider(
            value=default_height,
            min=100,
            max=self.full_height,
            step=1,
            description="",
            layout=layout_slider,
        )
        self.h_text = widgets.IntText(
            value=default_height,
            description="Height:",
            style=style,
            layout=layout_text,
        )
        widgets.jslink((self.h_slider, "value"), (self.h_text, "value"))

        self.btn_auto = widgets.Button(
            description="Auto Detect Tissue",
            button_style="primary",
        )
        self.btn_preview = widgets.Button(
            description="Preview Location",
            button_style="info",
        )
        self.btn_check_hires = widgets.Button(
            description="Check Hi-Res (Detail)",
            button_style="warning",
        )
        self.btn_save = widgets.Button(
            description="Save Cropped TIFF",
            button_style="success",
            icon="save",
        )
        self.btn_delete = widgets.Button(
            description="Delete Saved Crop",
            button_style="danger",
            icon="trash",
        )

        self.output_area = widgets.Output()

        self.btn_auto.on_click(self._on_auto_detect)
        self.btn_preview.on_click(self._on_preview)
        self.btn_check_hires.on_click(self._on_check_hires)
        self.btn_save.on_click(self._on_save)
        self.btn_delete.on_click(self._on_delete)

    def _get_current_config(self) -> CropConfig:
        """Get current crop configuration from UI."""
        return CropConfig(
            x=self.x_text.value,
            y=self.y_text.value,
            width=self.w_text.value,
            height=self.h_text.value,
            step=self.step,
        )

    def _set_config(self, config: CropConfig):
        """Update UI with new configuration."""
        self.x_text.value = config.x
        self.y_text.value = config.y
        self.w_text.value = config.width
        self.h_text.value = config.height

    def _on_auto_detect(self, _):
        """Auto-detect tissue boundary."""
        with self.output_area:
            clear_output(wait=True)
            print("Detecting tissue boundary...")
            try:
                config = auto_detect_tissue_bounds(
                    self.expression_path_8um,
                    self.source_image_path,
                    margin=200,
                    step=self.step,
                )
                self._set_config(config)
                print(f"Auto-detected: ({config.x}, {config.y}) + {config.width}x{config.height}")
                self._on_preview(None)
            except Exception as e:
                print(f"Error: {e}")

    def _on_preview(self, b):
        """Show preview of crop location."""
        with self.output_area:
            if b is not None:
                clear_output(wait=True)
            plt.close("all")

            config = self._get_current_config()

            ncols = 2 if "cytassist" in self.ref_images else 1
            fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
            if ncols == 1:
                axes = [axes]

            if "lowres" in self.ref_images:
                ax = axes[0]
                ax.imshow(self.ref_images["lowres"])

                box_x = config.x * self.scale_x
                box_y = config.y * self.scale_y
                box_w = config.width * self.scale_x
                box_h = config.height * self.scale_y

                rect = patches.Rectangle(
                    (box_x, box_y),
                    box_w,
                    box_h,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.set_title("Low-Res Image (ROI)")

            if "cytassist" in self.ref_images:
                ax2 = axes[1]
                ax2.imshow(self.ref_images["cytassist"])
                ax2.set_title("CytAssist Image (Ref)")

            plt.show()

    def _on_check_hires(self, _):
        """Show high-resolution center detail (1000x1000)."""
        with self.output_area:
            clear_output(wait=True)
            plt.close("all")

            config = self._get_current_config()
            center_x = config.x + config.width // 2
            center_y = config.y + config.height // 2
            print(f"Checking center detail at x={center_x}, y={center_y}")

            try:
                store = tifffile.imread(self.source_image_path, aszarr=True)
                z_img = zarr.open(store, mode="r")
                sample = np.asarray(z_img[center_y : center_y + 1000, center_x : center_x + 1000])

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(sample)
                ax.set_title("Hi-Res Center Detail (1000x1000)")
                plt.show()
            except Exception as e:
                print(f"Error: {e}")

    def _on_save(self, _):
        """Save cropped image."""
        with self.output_area:
            clear_output(wait=True)

            if os.path.exists(self.output_path):
                print(f"File already exists: {self.output_path}")
                print("Using existing file. Delete first to overwrite.")
                return

            print("Saving... Please wait.")

            try:
                config = self._get_current_config()
                config = config.snap_to_grid(self.full_width, self.full_height)
                self._set_config(config)

                save_cropped_image(
                    self.source_image_path,
                    self.output_path,
                    config,
                    overwrite=False,
                )

            except Exception as e:
                print(f"Error: {e}")

    def _on_delete(self, _):
        """Delete saved crop file."""
        with self.output_area:
            clear_output(wait=True)
            if os.path.exists(self.output_path):
                try:
                    os.remove(self.output_path)
                    print(f"Deleted: {self.output_path}")
                    print("You can now save a new crop.")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("No file found to delete.")

    def display(self):
        """Display the widget UI."""
        display(
            widgets.VBox(
                [
                    widgets.HTML("<h3>Visium HD Large Image Cropper</h3>"),
                    self.btn_auto,
                    widgets.HTML("<hr>"),
                    widgets.HBox([self.x_text, self.x_slider]),
                    widgets.HBox([self.y_text, self.y_slider]),
                    widgets.HBox([self.w_text, self.w_slider]),
                    widgets.HBox([self.h_text, self.h_slider]),
                    widgets.HBox([self.btn_preview, self.btn_check_hires]),
                    widgets.HBox([self.btn_save, self.btn_delete]),
                    self.output_area,
                ]
            )
        )