"""Configuration loading helpers.

Version-controlled settings live in ``configs/*.yaml``. Machine-specific settings
(such as where the raw CSVs are stored) live in ``configs/local.yaml``, which is
git-ignored.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_DATA_CONFIG = CONFIGS_DIR / "data.yaml"
DEFAULT_LOCAL_CONFIG = CONFIGS_DIR / "local.yaml"


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file that must contain a mapping at the top level."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping at the top level: {path}")
    return data


def resolve_raw_data_dir(local_config_path: str | Path | None = None) -> Path:
    """Return the directory holding the raw H&M CSV files, read from the local config."""
    path = Path(local_config_path) if local_config_path else DEFAULT_LOCAL_CONFIG
    if not path.is_file():
        raise FileNotFoundError(
            f"Local config not found: {path}. Copy configs/local.example.yaml to "
            "configs/local.yaml and set raw_data_dir to the folder containing the CSV files."
        )
    value = load_yaml(path).get("raw_data_dir")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'raw_data_dir' must be a non-empty string in {path}")
    raw_dir = Path(value).expanduser()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw_data_dir does not exist or is not a directory: {raw_dir}")
    return raw_dir
