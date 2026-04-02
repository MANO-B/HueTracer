from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

import ipywidgets as widgets
from IPython.display import clear_output, display

from huetracer.io.config import (
    PathConfig,
    build_path_config,
    load_config_file,
    save_path_config_json,
    scan_config_files,
    update_config_file,
    validate_path_config,
)


class PathSetupWidget:
    """Interactive path setup widget for Visium HD tutorials."""

    def __init__(self, defaults: Optional[Dict[str, str]] = None):
        self.defaults = self._build_defaults(defaults)
        self.config: Optional[PathConfig] = None
        self.runtime_config: Optional[Dict[str, str]] = None

        self._init_ui_components()
        self.ui = self._build_ui()

    def _build_defaults(self, defaults: Optional[Dict[str, str]]) -> Dict[str, str]:
        base = {
            "sample_name": "P2_CRC",
            "base_dir": "/app/data/input",
            "source_img": "P2_CRC/Visium_HD_Human_Colon_Cancer_P2_tissue_image.btf",
            "exp_path_2um": "P2_CRC/binned_outputs/square_002um",
            "exp_path_8um": "P2_CRC/binned_outputs/square_008um",
            "date": datetime.now().strftime("%y%m%d"),
            "results_dir": "/app/results/P2_CRC",
        }
        if defaults:
            base.update(defaults)
        return base

    def _init_ui_components(self) -> None:
        style = {"description_width": "initial"}
        layout = widgets.Layout(width="95%")

        self.sample_name_w = widgets.Text(
            value=self.defaults["sample_name"],
            description="Sample Name:",
            style=style,
            layout=layout,
        )
        self.base_dir_w = widgets.Text(
            value=self.defaults["base_dir"],
            description="Base Directory:",
            placeholder="Path to your project root",
            style=style,
            layout=layout,
        )
        self.source_img_w = widgets.Text(
            value=self.defaults["source_img"],
            description="Microscope Image (.btf/.tiff):",
            style=style,
            layout=layout,
        )
        self.exp_2um_w = widgets.Text(
            value=self.defaults["exp_path_2um"],
            description="Expression Path (2um):",
            style=style,
            layout=layout,
        )
        self.exp_8um_w = widgets.Text(
            value=self.defaults["exp_path_8um"],
            description="Expression Path (8um):",
            style=style,
            layout=layout,
        )
        self.results_dir_w = widgets.Text(
            value=self.defaults["results_dir"],
            description="Results Directory:",
            style=style,
            layout=layout,
        )
        self.date_w = widgets.Text(
            value=self.defaults["date"],
            description="Date in results directory:",
            style=style,
            layout=layout,
        )

        self.run_button = widgets.Button(
            description="Set Paths",
            button_style="success",
            icon="check",
        )
        self.run_button.on_click(self._on_button_clicked)
        self.output = widgets.Output()

    def _build_ui(self) -> widgets.Widget:
        return widgets.VBox(
            [
                self.sample_name_w,
                self.base_dir_w,
                widgets.HTML(value="<b>Files from SpaceRanger outputs:</b>"),
                self.source_img_w,
                self.exp_2um_w,
                self.exp_8um_w,
                self.results_dir_w,
                self.date_w,
                self.run_button,
                self.output,
            ]
        )

    def _on_button_clicked(self, _: Any) -> None:
        with self.output:
            clear_output(wait=True)

            self.config = build_path_config(
                sample_name=self.sample_name_w.value,
                base_dir=self.base_dir_w.value,
                source_img=self.source_img_w.value,
                exp_path_2um=self.exp_2um_w.value,
                exp_path_8um=self.exp_8um_w.value,
                results_dir=self.results_dir_w.value,
                date=self.date_w.value,
            )

            config_path = save_path_config_json(self.config)
            path_status = validate_path_config(self.config)
            self.runtime_config = self.config.to_runtime_dict()

            print("✅ Configuration saved!")
            print(f"Saved to: {config_path}")
            print("✅ Parameters updated and directories created!")
            print(f"Results Path: {self.config.results_path}")
            print(f"Expression Path: {self.config.expression_path}")
            print(
                "Checking 2um path: "
                f"{'Exists' if path_status['expression_2um_exists'] else '⚠️ Not Found'}"
            )
            print("Done! Go ahead.")

    def display(self) -> None:
        print("📦 HueTracer: Visium HD Analysis Setup")
        display(self.ui)


def create_path_setup_widget(defaults: Optional[Dict[str, str]] = None) -> PathSetupWidget:
    """Factory helper to create a path setup widget instance."""
    return PathSetupWidget(defaults=defaults)


class ConfigSelector:
    """Config file selector widget for VisiumHD."""

    def __init__(self, base_dir_default="/app/data/input"):
        style = {"description_width": "140px"}
        w_full = widgets.Layout(width="800px")

        self.base_dir_w = widgets.Text(
            value=base_dir_default,
            description="Base dir:",
            style=style,
            layout=w_full,
        )
        self.refresh_btn = widgets.Button(description="Scan", button_style="info", icon="search")
        self.config_dd = widgets.Dropdown(options=[], description="Config file:", style=style, layout=w_full)
        self.load_btn = widgets.Button(description="Load", button_style="success", icon="download")
        self.out = widgets.Output()
        self.last_config: Optional[Dict[str, Any]] = None

        self.refresh_btn.on_click(self.do_scan)
        self.load_btn.on_click(self.do_load)

        self.ui = widgets.VBox([
            widgets.HTML("<h3>📁 VisiumHD config file selector</h3>"),
            widgets.HBox([self.base_dir_w, self.refresh_btn]),
            widgets.HBox([self.config_dd, self.load_btn]),
            self.out
        ])

    def display(self) -> None:
        display(self.ui)

    def do_scan(self, _: Any = None) -> None:
        with self.out:
            clear_output()
            base_dir = os.path.expanduser(self.base_dir_w.value.strip())
            config_dir = os.path.join(base_dir, "config")
            print("🔍 Searching in:", config_dir)
            files = scan_config_files(config_dir)
            if not files:
                self.config_dd.options = []
                print("⚠️ No config files found.")
                return
            opts = [(os.path.basename(f), f) for f in files]
            self.config_dd.options = opts
            self.config_dd.value = opts[0][1]
            print(f"✅ Found {len(files)} config(s).")

    def do_load(self, _: Any = None) -> None:
        with self.out:
            clear_output()
            if not self.config_dd.options:
                print("⚠️ Click Scan first.")
                return
            cfg_path = self.config_dd.value
            cfg = load_config_file(cfg_path)
            if cfg is None:
                print("❌ Failed to load config:", cfg_path)
                return
            self.last_config = cfg
            print("✅ Loaded config:", cfg_path)
            print("Keys:", list(cfg.keys()))
