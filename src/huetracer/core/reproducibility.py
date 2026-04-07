"""Utilities for reproducible execution across HueTracer pipelines."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import scanpy as sc


def set_global_seed(
    seed: int = 42,
    *,
    set_scanpy_seed: bool = True,
    set_scvi_seed: bool = True,
    set_torch_seed: bool = True,
    deterministic_torch: bool = True,
) -> int:
    """Set global random seeds for deterministic behavior where possible.

    This follows the notebook style:
      - np.random.seed(SEED)
      - random.seed(SEED)
      - torch.manual_seed(SEED)
    and also aligns Scanpy/scvi-tools seeds.
    """

    seed = int(seed)

    np.random.seed(seed)
    random.seed(seed)

    if set_scanpy_seed:
        # Scanpy uses this value when APIs do not receive random_state explicitly.
        sc.settings.seed = seed

    if set_scvi_seed:
        try:
            import scvi

            scvi.settings.seed = seed
        except Exception:
            pass

    if set_torch_seed:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = bool(deterministic_torch)
                torch.backends.cudnn.benchmark = not bool(deterministic_torch)
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(bool(deterministic_torch), warn_only=True)
        except Exception:
            pass

    return seed


def get_seed_from_env(default: int = 42, env_key: str = "HUETRACER_SEED") -> int:
    """Read seed from environment when set, otherwise return default."""

    import os

    value: Optional[str] = os.getenv(env_key)
    if value is None:
        return int(default)

    try:
        return int(value)
    except ValueError:
        return int(default)
