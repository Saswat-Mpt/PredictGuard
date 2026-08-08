"""
test_config.py
==============
Unit tests for src/config.py configuration loader.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, _get_defaults


def test_load_config_default_file():
    """Verify that root config.yaml loads correctly with expected sections."""
    cfg = load_config()
    assert "target" in cfg
    assert "cost_decision" in cfg
    assert cfg["cost_decision"]["C_FN"] == 10000
    assert cfg["cost_decision"]["C_FP"] == 500


def test_load_config_missing_file_fallback(tmp_path):
    """Verify that missing config file gracefully falls back to default dict."""
    missing_path = tmp_path / "non_existent_config.yaml"
    cfg = load_config(missing_path)
    assert cfg == _get_defaults()
    assert cfg["target"]["prediction_horizon_hours"] == 24
