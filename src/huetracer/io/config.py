from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PathConfig:
    sample_name: str
    base_dir: str
    source_img: str
    exp_path_2um: str
    exp_path_8um: str
    results_dir: str
    date: str
    tmp_path: str = "/tmp"

    @property
    def source_image_path(self) -> str:
        return os.path.join(self.base_dir, self.source_img)

    @property
    def expression_path(self) -> str:
        return os.path.join(self.base_dir, self.exp_path_2um)

    @property
    def expression_path_8um(self) -> str:
        return os.path.join(self.base_dir, self.exp_path_8um)

    @property
    def results_path(self) -> str:
        return os.path.join(self.results_dir, self.date)

    @property
    def config_dir(self) -> str:
        return os.path.join(self.base_dir, "config")

    @property
    def b2c_config_save_path(self) -> str:
        return os.path.join(self.config_dir, f"{self.sample_name}_{self.date}_b2c_config.json")

    @property
    def source_image_in_spaceranger(self) -> str:
        return os.path.join(self.expression_path, "Visium_HD_tissue_image_full_res.tiff")

    def to_runtime_dict(self) -> Dict[str, str]:
        return {
            "SAMPLE_NAME": self.sample_name,
            "BASE_DIR": self.base_dir,
            "SOURCE_IMAGE_PATH": self.source_image_path,
            "EXPRESSION_PATH": self.expression_path,
            "EXPRESSION_PATH_8UM": self.expression_path_8um,
            "RESULTS_PATH": self.results_path,
            "DATE": self.date,
            "TMP_PATH": self.tmp_path,
            "B2C_CONFIG_SAVE_PATH": self.b2c_config_save_path,
            "source_image_path": self.source_image_in_spaceranger,
        }


def build_path_config(
    sample_name: str,
    base_dir: str,
    source_img: str,
    exp_path_2um: str,
    exp_path_8um: str,
    results_dir: str,
    date: str | None = None,
    tmp_path: str = "/tmp",
) -> PathConfig:
    if date is None:
        date = datetime.now().strftime("%y%m%d")
    return PathConfig(
        sample_name=sample_name,
        base_dir=base_dir,
        source_img=source_img,
        exp_path_2um=exp_path_2um,
        exp_path_8um=exp_path_8um,
        results_dir=results_dir,
        date=date,
        tmp_path=tmp_path,
    )


def validate_path_config(config: PathConfig) -> Dict[str, bool]:
    return {
        "source_image_exists": os.path.exists(config.source_image_path),
        "expression_2um_exists": os.path.exists(config.expression_path),
        "expression_8um_exists": os.path.exists(config.expression_path_8um),
    }


def save_path_config_json(config: PathConfig, ensure_dirs: bool = True) -> str:
    if ensure_dirs:
        os.makedirs(config.results_path, exist_ok=True)
        os.makedirs(config.tmp_path, exist_ok=True)
        os.makedirs(config.config_dir, exist_ok=True)

    with open(config.b2c_config_save_path, "w", encoding="utf-8") as file_handle:
        json.dump(config.to_runtime_dict(), file_handle, indent=4)

    return config.b2c_config_save_path


def as_dict(config: PathConfig) -> Dict[str, str]:
    return asdict(config)


# =========================
# Config File Scanning and Loading
# =========================

def scan_config_files(config_dir: str) -> List[str]:
    """
    Scan a directory for JSON config files.
    Returns a sorted list of file paths.
    """
    if not os.path.isdir(config_dir):
        return []
    pattern = os.path.join(config_dir, "*.json")
    files = glob.glob(pattern)
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files


def load_config_file(cfg_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON config file and return its contents as a dict.
    Returns None if loading fails.
    """
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        return cfg
    except Exception:
        return None


def update_config_file(
    config_path: str,
    updates: Dict[str, Any],
    merge: bool = True,
) -> None:
    """Update a JSON config file with new values."""
    config: Dict[str, Any]

    if merge and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as file_handle:
            config = json.load(file_handle)
        config.update(updates)
    else:
        config = updates

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as file_handle:
        json.dump(config, file_handle, indent=4)
