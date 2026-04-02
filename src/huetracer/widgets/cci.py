"""Interactive widgets for CCI-related file preparation."""

from __future__ import annotations

import os
import zipfile
from typing import Any, Optional

import gdown
import ipywidgets as widgets
from IPython.display import clear_output, display


class NicheNetDownloaderWidget:
    """Download and prepare NicheNet ligand-target CSV only when needed.

    This widget checks whether the target CSV already exists, downloads a zip
    from Google Drive when missing, and optionally extracts it.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        url: str = "https://drive.google.com/uc?export=download&id=1WpHzP_ticvL1T_aIufHI1ISx3VQMnUhd",
        zip_filename: str = "ligand_target_df.csv.zip",
        extract_dir: str = "ligand_target_df",
        target_csv: str = "ligand_target_df.csv",
        runtime_namespace: Optional[dict[str, Any]] = None,
        runtime_var_name: str = "file_nichenet",
    ) -> None:
        self.runtime_namespace = runtime_namespace
        self.runtime_var_name = runtime_var_name
        self.file_nichenet: Optional[str] = None

        self._init_ui_components(
            base_dir=base_dir,
            url=url,
            zip_filename=zip_filename,
            extract_dir=extract_dir,
            target_csv=target_csv,
        )
        self.ui = self._build_ui()

    def _init_ui_components(
        self,
        base_dir: Optional[str],
        url: str,
        zip_filename: str,
        extract_dir: str,
        target_csv: str,
    ) -> None:
        style = {"description_width": "initial"}

        self.base_dir_w = widgets.Text(
            value=os.path.abspath(base_dir or os.getcwd()),
            description="BASE_DIR",
            style=style,
            layout=widgets.Layout(width="700px"),
        )
        self.url_w = widgets.Text(
            value=url,
            description="GoogleDrive URL",
            style=style,
            layout=widgets.Layout(width="700px"),
        )
        self.zip_w = widgets.Text(
            value=zip_filename,
            description="Zip filename",
            style=style,
            layout=widgets.Layout(width="420px"),
        )
        self.extract_w = widgets.Text(
            value=extract_dir,
            description="Extract dir",
            style=style,
            layout=widgets.Layout(width="420px"),
        )
        self.target_w = widgets.Text(
            value=target_csv,
            description="Target CSV",
            style=style,
            layout=widgets.Layout(width="420px"),
        )

        self.quiet_w = widgets.Checkbox(value=False, description="quiet download")
        self.force_w = widgets.Checkbox(value=False, description="force re-download")
        self.unzip_w = widgets.Checkbox(value=True, description="unzip after download")

        self.btn_check = widgets.Button(description="Check", button_style="info", icon="search")
        self.btn_get = widgets.Button(
            description="Download if missing", button_style="success", icon="download"
        )
        self.btn_clear = widgets.Button(description="Clear", icon="eraser")
        self.output = widgets.Output()

        self.btn_check.on_click(self._on_check)
        self.btn_get.on_click(self._on_download_if_missing)
        self.btn_clear.on_click(self._on_clear)

    def _build_ui(self) -> widgets.Widget:
        return widgets.VBox(
            [
                widgets.HTML(
                    "<h3>NicheNet ligand-target file downloader (download only if missing)</h3>"
                ),
                widgets.HBox([self.btn_get, self.btn_check, self.btn_clear]),
                self.base_dir_w,
                self.url_w,
                widgets.HBox([self.zip_w, self.extract_w, self.target_w]),
                widgets.HBox([self.unzip_w, self.quiet_w, self.force_w]),
                self.output,
            ]
        )

    def _paths(self) -> tuple[str, str, str, str]:
        base = os.path.abspath(self.base_dir_w.value.strip())
        zip_path = os.path.join(base, self.zip_w.value.strip())
        extract_name = self.extract_w.value.strip()
        extract_dir = os.path.join(base, extract_name) if extract_name else base
        target_path = os.path.join(extract_dir, self.target_w.value.strip())
        return base, zip_path, extract_dir, target_path

    def _set_runtime_file_path(self, target_path: str) -> None:
        self.file_nichenet = target_path
        if self.runtime_namespace is not None:
            self.runtime_namespace[self.runtime_var_name] = target_path

    def _on_check(self, _: Any = None) -> None:
        with self.output:
            clear_output(wait=True)
            base, zip_path, extract_dir, target_path = self._paths()
            print("BASE_DIR:", base)
            print("Target:", target_path)
            print("Exists target?", os.path.exists(target_path))
            print("Zip:", zip_path)
            print("Exists zip?", os.path.exists(zip_path))
            print("Extract dir:", extract_dir)
            print("Exists extract dir?", os.path.exists(extract_dir))
            if os.path.exists(target_path):
                self._set_runtime_file_path(target_path)
                print(f"Ready. {self.runtime_var_name} =", target_path)

    def _on_download_if_missing(self, _: Any = None) -> None:
        with self.output:
            clear_output(wait=True)
            base, zip_path, extract_dir, target_path = self._paths()
            os.makedirs(base, exist_ok=True)

            if os.path.exists(target_path) and not self.force_w.value:
                self._set_runtime_file_path(target_path)
                print("Target already exists. Skipping download.")
                print(f"{self.runtime_var_name} =", target_path)
                return

            print("Target does not exist.")
            if os.path.exists(zip_path) and not self.force_w.value:
                print("Zipped file already downloaded.")
            else:
                print("Downloading from Google Drive...")
                print("URL:", self.url_w.value)
                print("->", zip_path)
                try:
                    gdown.download(self.url_w.value, zip_path, quiet=self.quiet_w.value)
                except PermissionError:
                    # Some environments fail on os.utime after write; continue if file exists.
                    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
                        print("PermissionError on utime; file seems downloaded. Continuing...")
                    else:
                        raise

            if self.unzip_w.value:
                try:
                    print("Extracting zip...")
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(zip_path, "r") as zip_handle:
                        zip_handle.extractall(extract_dir)
                    print("Extracted to:", extract_dir)
                except Exception as error:
                    print("Unzip failed:", repr(error))

            if not os.path.exists(target_path):
                candidate = os.path.join(extract_dir, os.path.basename(target_path))
                if os.path.exists(candidate):
                    target_path = candidate

            if os.path.exists(target_path):
                self._set_runtime_file_path(target_path)
                print("Ready.")
                print(f"{self.runtime_var_name} =", target_path)
            else:
                print("Download/unzip finished, but target CSV was not found:")
                print(" -", target_path)
                print("Check zip contents or adjust Target CSV / Extract dir.")
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_handle:
                        print("\nZip contents (first 30):")
                        for name in zip_handle.namelist()[:30]:
                            print(" ", name)
                except Exception:
                    pass

    def _on_clear(self, _: Any = None) -> None:
        with self.output:
            clear_output(wait=True)

    def display(self) -> None:
        display(self.ui)


def create_nichenet_downloader_widget(
    base_dir: Optional[str] = None,
    url: str = "https://drive.google.com/uc?export=download&id=1WpHzP_ticvL1T_aIufHI1ISx3VQMnUhd",
    zip_filename: str = "ligand_target_df.csv.zip",
    extract_dir: str = "ligand_target_df",
    target_csv: str = "ligand_target_df.csv",
    runtime_namespace: Optional[dict[str, Any]] = None,
    runtime_var_name: str = "file_nichenet",
) -> NicheNetDownloaderWidget:
    """Create a NicheNetDownloaderWidget instance."""
    return NicheNetDownloaderWidget(
        base_dir=base_dir,
        url=url,
        zip_filename=zip_filename,
        extract_dir=extract_dir,
        target_csv=target_csv,
        runtime_namespace=runtime_namespace,
        runtime_var_name=runtime_var_name,
    )