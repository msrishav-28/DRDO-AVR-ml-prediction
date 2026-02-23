"""
Configuration loader for AVR-PHM project.

Provides centralized YAML config loading with path resolution.
All configuration files are loaded relative to the avr_phm package root.
"""

import os
from pathlib import Path
from typing import Any

import yaml


# Package root directory (parent of config/)
_PACKAGE_ROOT: Path = Path(__file__).resolve().parent.parent


def get_package_root() -> Path:
    """Returns the absolute path to the avr_phm package root directory."""
    return _PACKAGE_ROOT


def load_yaml(config_name: str) -> dict[str, Any]:
    """
    Load a YAML configuration file by name.

    Purpose:
        Provides a single entry point for loading any project config file.
        All paths in the returned dict remain as strings; callers resolve
        them using get_package_root() / path.

    Inputs:
        config_name: One of 'scenarios', 'milstd', 'model', 'paths'.

    Outputs:
        Parsed YAML dict with all keys and values.

    Raises:
        FileNotFoundError: If the requested config file does not exist.
        ValueError: If config_name is not recognized.
    """
    valid_configs: dict[str, str] = {
        "scenarios": "scenarios.yaml",
        "milstd": "milstd.yaml",
        "model": "model.yaml",
        "paths": "paths.yaml",
    }

    if config_name not in valid_configs:
        raise ValueError(
            f"Unknown config '{config_name}'. "
            f"Valid options: {list(valid_configs.keys())}"
        )

    config_path: Path = _PACKAGE_ROOT / "config" / valid_configs[config_name]

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    return data


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a relative path from paths.yaml to an absolute path.

    Purpose:
        Converts relative paths stored in paths.yaml to absolute paths
        anchored at the package root directory.

    Inputs:
        relative_path: A path string from paths.yaml (e.g., 'data/raw').

    Outputs:
        Absolute Path object.
    """
    return _PACKAGE_ROOT / relative_path


def get_device() -> str:
    """
    Returns the compute device string from model config.
    Falls back to 'cpu' if CUDA is not available.
    """
    import torch

    model_cfg: dict[str, Any] = load_yaml("model")
    requested: str = model_cfg["global"]["device"]

    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def set_all_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Purpose:
        Ensures deterministic behavior across all random number generators
        used in the project (Python, NumPy, PyTorch).

    Inputs:
        seed: Integer seed value. Default 42 per spec.

    Mathematical basis:
        N/A — this is a reproducibility utility.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_tests() -> None:
    """Sanity checks for the config module."""
    # Test 1: All configs load without error
    for name in ["scenarios", "milstd", "model", "paths"]:
        cfg = load_yaml(name)
        assert isinstance(cfg, dict), f"Config '{name}' did not return a dict"

    # Test 2: Scenario config has expected structure
    scenarios_cfg = load_yaml("scenarios")
    assert "global" in scenarios_cfg, "scenarios.yaml missing 'global' key"
    assert "scenarios" in scenarios_cfg, "scenarios.yaml missing 'scenarios' key"
    assert scenarios_cfg["global"]["nominal_voltage_v"] == 28.0, (
        "Nominal voltage must be 28.0V per MIL-STD-1275E"
    )

    # Test 3: Package root resolves correctly
    root = get_package_root()
    assert root.exists(), f"Package root does not exist: {root}"
    assert (root / "config").is_dir(), "config/ directory not found under package root"

    print("[PASS] config/__init__.py — all tests passed.")


if __name__ == "__main__":
    run_tests()
