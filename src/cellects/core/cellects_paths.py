#!/usr/bin/env python3
"""
Generate the different paths used by cellects.
Adjust the path names according to the current position of the software.
"""

import os
from pathlib import Path

# Current file -> src/cellects/cellect_paths.py
CURR_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
CELLECTS_DIR = CURR_DIR.parent  # = src/cellects

# Package-internal dirs
CONFIG_DIR = CELLECTS_DIR / "config"
CORE_DIR = CELLECTS_DIR / "core"
GUI_DIR = CELLECTS_DIR / "gui"
ICONS_DIR = CELLECTS_DIR / "icons"
IMAGE_ANALYSIS_DIR = CELLECTS_DIR / "image_analysis"
UTILS_DIR = CELLECTS_DIR / "utils"

# Repo root (src/..)
REPO_ROOT = CELLECTS_DIR.parent.parent

# Repo-level dirs
DATA_DIR = REPO_ROOT / "data"
EXPERIMENTS_DIR = DATA_DIR / "single_experiment"
TEST_DIR = REPO_ROOT / "tests"

# Example packaged file
ALL_VARS_PKL_FILE = CONFIG_DIR / "all_vars.pkl"