from __future__ import annotations

import traceback
from typing import Any, Optional

import ipywidgets as widgets
import scanpy as sc
from IPython.display import clear_output, display

from ..core.preprocessing import (
    Bin2CellLoadParams,
    Bin2CellLoadResult,
    CropRegion,
    install_bin2cell_safety_override,
    load_visium_and_preprocess,
)


class Bin2CellLoaderWidget:
    def __init__(
        self,
        expression_path: str,
        source_image_path: str,
        tmp_path: str,
        crop_x: int,
        crop_y: int,
        crop_width: int,
        crop_height: int,
        step: int = 4096,
        runtime_namespace: Optional[dict[str, Any]] = None,
    ):
        self.expression_path = expression_path
        self.source_image_path = source_image_path
        self.tmp_path = tmp_path
        self.crop = CropRegion(
            x=int(crop_x),
            y=int(crop_y),
            width=int(crop_width),
            height=int(crop_height),
        )
        self.step = int(step)
        self.runtime_namespace = runtime_namespace

        self.result: Optional[Bin2CellLoadResult] = None
        self.runtime_values: dict[str, Any] = {}

        self._init_ui_components()
        self.ui = self._build_ui()

    def _init_ui_components(self) -> None:
        style = {"description_width": "150px"}
        layout = widgets.Layout(width="500px")

        self.mpp_w = widgets.FloatSlider(
            value=0.5,
            min=0.1,
            max=2.0,
            step=0.1,
            description="MPP (Microns Per Pixel):",
            style=style,
            layout=layout,
        )
        self.prob_he_w = widgets.FloatSlider(
            value=0.05,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Prob Thresh (HE):",
            style=style,
            layout=layout,
        )
        self.prob_gex_w = widgets.FloatSlider(
            value=0.7,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Prob Thresh (GEX):",
            style=style,
            layout=layout,
        )
        self.nms_w = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
            description="NMS Threshold:",
            style=style,
            layout=layout,
        )
        self.max_dist_w = widgets.IntSlider(
            value=2,
            min=1,
            max=10,
            description="Max Bin Distance:",
            style=style,
            layout=layout,
        )
        self.step_w = widgets.IntSlider(
            value=self.step,
            min=256,
            max=8192,
            step=256,
            description="Block size for bin2cell:",
            style=style,
            layout=layout,
        )

        self.btn_load = widgets.Button(
            description="Load Data & Preprocess",
            button_style="primary",
            icon="download",
            layout=widgets.Layout(width="300px"),
        )
        self.btn_load.on_click(self._on_load)
        self.output_log = widgets.Output()

    def _build_ui(self) -> widgets.Widget:
        return widgets.VBox(
            [
                widgets.HTML("<h3>⚙️ Data Loading & Parameters</h3>"),
                widgets.HBox([self.mpp_w, self.step_w]),
                widgets.HBox([self.prob_he_w, self.prob_gex_w]),
                widgets.HBox([self.nms_w, self.max_dist_w]),
                widgets.HTML("<br>"),
                self.btn_load,
                self.output_log,
            ]
        )

    def _current_params(self) -> Bin2CellLoadParams:
        return Bin2CellLoadParams(
            mpp=float(self.mpp_w.value),
            prob_thresh_he=float(self.prob_he_w.value),
            prob_thresh_gex=float(self.prob_gex_w.value),
            nms_thresh=float(self.nms_w.value),
            max_bin_distance=int(self.max_dist_w.value),
            block_size=int(self.step_w.value),
        )

    def _on_load(self, _: Any) -> None:
        with self.output_log:
            clear_output(wait=True)
            print("🚀 Starting Data Loading...")

            install_bin2cell_safety_override()
            params = self._current_params()

            try:
                self.result = load_visium_and_preprocess(
                    expression_path=self.expression_path,
                    source_image_path=self.source_image_path,
                    crop=self.crop,
                    tmp_path=self.tmp_path,
                    params=params,
                )

                self.runtime_values = self.result.to_runtime_dict()
                if self.runtime_namespace is not None:
                    self.runtime_namespace.update(self.runtime_values)

                print("✅ Data Loading & Preprocessing Complete!")
                print(f"Final AnnData Shape: {self.result.adata.shape}")
                print("Done! Go ahead.")
            except Exception as error:
                print(f"❌ Error: {error}")
                traceback.print_exc()

    def display(self) -> None:
        display(self.ui)


def create_bin2cell_loader_widget(
    expression_path: str,
    source_image_path: str,
    tmp_path: str,
    crop_x: int,
    crop_y: int,
    crop_width: int,
    crop_height: int,
    step: int = 4096,
    runtime_namespace: Optional[dict[str, Any]] = None,
) -> Bin2CellLoaderWidget:
    return Bin2CellLoaderWidget(
        expression_path=expression_path,
        source_image_path=source_image_path,
        tmp_path=tmp_path,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_width=crop_width,
        crop_height=crop_height,
        step=step,
        runtime_namespace=runtime_namespace,
    )


class Bin2CellConverterWidget:
    """Interactive widget to run bin_to_cell with automatic label-key switching.

    Supports two modes:
    - with_gex: uses `labels_joint`
    - no_gex: uses `labels_he_expanded`
    """

    def __init__(
        self,
        adata: Any,
        roi_x1: int,
        roi_x2: int,
        roi_y1: int,
        roi_y2: int,
        mpp: float = 0.5,
        basis: str = "spatial_cropped_150_buffer",
        b2c_module: Optional[Any] = None,
        runtime_namespace: Optional[dict[str, Any]] = None,
    ) -> None:
        self.adata = adata
        self.roi_x1 = int(roi_x1)
        self.roi_x2 = int(roi_x2)
        self.roi_y1 = int(roi_y1)
        self.roi_y2 = int(roi_y2)
        self.mpp = float(mpp)
        self.basis = basis
        self.runtime_namespace = runtime_namespace

        if b2c_module is None:
            import bin2cell as b2c_module
        self.b2c = b2c_module

        self.result = None

        self._init_ui_components()
        self.ui = self._build_ui()

    def _init_ui_components(self) -> None:
        self.html_note = widgets.HTML(
            """
<b>Important selection:</b> nuclei segmentation with or without gene expression data.<br>
Noise is removed downstream (minimum 2um bin area).<br>
There are often no major problems with the use of gene expression data.
"""
        )

        self.mode_w = widgets.ToggleButtons(
            options=[
                ("With gene expression", "with_gex"),
                ("Without gene expression", "no_gex"),
            ],
            value="with_gex",
            description="Mode:",
        )

        self.btn_run = widgets.Button(
            description="Run bin_to_cell & Plot",
            button_style="success",
            icon="play",
        )
        self.output = widgets.Output()
        self.btn_run.on_click(self._on_run)

    def _build_ui(self) -> widgets.Widget:
        return widgets.VBox(
            [
                widgets.HTML("<h3>🧬 Bin2Cell (Simple Mode)</h3>"),
                self.html_note,
                self.mode_w,
                self.btn_run,
                self.output,
            ]
        )

    def _get_labels_key(self) -> str:
        labels_key = "labels_joint" if self.mode_w.value == "with_gex" else "labels_he_expanded"
        if labels_key not in self.adata.obs.columns:
            raise ValueError(f"'{labels_key}' not found in adata.obs")
        return labels_key

    def _on_run(self, _: Any) -> None:
        with self.output:
            clear_output(wait=True)
            try:
                labels_key = self._get_labels_key()

                print("🚀 Running bin_to_cell")
                print("   Mode:", self.mode_w.value)
                print("   labels_key:", labels_key)
                print("Running...")

                self.result = self.b2c.bin_to_cell(
                    self.adata,
                    labels_key=labels_key,
                    spatial_keys=["spatial", self.basis],
                )

                if self.runtime_namespace is not None:
                    self.runtime_namespace["cdata"] = self.result

                print("Done!")
                print("✅ bin_to_cell complete")
                print("   Cells:", self.result.n_obs)

                cell_mask = (
                    (self.result.obs["array_row"] >= self.roi_x1)
                    & (self.result.obs["array_row"] <= self.roi_x2)
                    & (self.result.obs["array_col"] >= self.roi_y1)
                    & (self.result.obs["array_col"] <= self.roi_y2)
                )

                ddata = self.result[cell_mask]
                sc.set_figure_params(fontsize=20, figsize=[7, 7])
                sc.pl.spatial(
                    ddata,
                    color=["bin_count"],
                    img_key=f"{self.mpp}_mpp_150_buffer",
                    basis=self.basis,
                    s=4,
                    save=False,
                )
                print("Go ahead.")
                del ddata
            except Exception as error:
                print("❌ Error:", error)

    def get_result(self) -> Any:
        if self.result is None:
            raise ValueError("No result yet. Please click 'Run bin_to_cell & Plot' first.")
        return self.result

    def display(self) -> None:
        display(self.ui)


def create_bin2cell_converter_widget(
    adata: Any,
    roi_x1: int,
    roi_x2: int,
    roi_y1: int,
    roi_y2: int,
    mpp: float = 0.5,
    basis: str = "spatial_cropped_150_buffer",
    b2c_module: Optional[Any] = None,
    runtime_namespace: Optional[dict[str, Any]] = None,
) -> Bin2CellConverterWidget:
    return Bin2CellConverterWidget(
        adata=adata,
        roi_x1=roi_x1,
        roi_x2=roi_x2,
        roi_y1=roi_y1,
        roi_y2=roi_y2,
        mpp=mpp,
        basis=basis,
        b2c_module=b2c_module,
        runtime_namespace=runtime_namespace,
    )
