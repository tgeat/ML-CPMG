"""Centralized workspace path definitions for the ML-CPMG package."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"
SPE10_INPUTS_DIR = INPUTS_DIR / "spe10"
OLYMPUS_INPUTS_DIR = INPUTS_DIR / "olympus"
BRUGGE_INPUTS_DIR = INPUTS_DIR / "brugge"
RESULTS_DIR = DATA_DIR / "results"
FIGURE_ARRAYS_DIR = DATA_DIR / "figure_arrays"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VENDOR_DIR = PROJECT_ROOT / ".paper_vendor"


def ensure_vendor_on_path() -> None:
    """Expose vendored PyAMG 5.4.1 if present and not yet on sys.path."""
    if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
        sys.path.append(str(VENDOR_DIR))
