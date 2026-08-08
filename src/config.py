"""
config.py
=========
Centralized configuration loader for PredictGuard.
Reads parameters from config.yaml with sensible fallbacks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger("predictguard.config")

# Default path to config.yaml relative to project root
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load project configuration from YAML file.

    Parameters
    ----------
    config_path : Path, optional
        Path to config.yaml. Defaults to root config.yaml.

    Returns
    -------
    dict
        Nested dictionary of configuration parameters.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.warning("Config file not found at %s. Using default parameters.", path)
        return _get_defaults()

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    logger.debug("Loaded configuration from %s", path)
    return config


def _get_defaults() -> Dict[str, Any]:
    """Fallback configuration parameters if config.yaml is missing."""
    return {
        "target": {"prediction_horizon_hours": 24, "window_inclusive_end": True},
        "split": {"test_ratio": 0.20, "n_folds": 5, "random_seed": 42},
        "cost_decision": {
            "C_FN": 10000,
            "C_FP": 500,
            "threshold_min": 0.05,
            "threshold_max": 0.95,
            "threshold_step": 0.01,
        },
        "risk_segmentation": {
            "tier_thresholds": {
                "LOW": [0, 20],
                "MEDIUM": [20, 50],
                "HIGH": [50, 75],
                "CRITICAL": [75, 100],
            }
        },
        "explainability": {"shap_global_sample_size": 5000, "random_state": 42},
        "component_classifier": {"n_estimators": 300, "max_depth": 6, "random_state": 42},
    }
