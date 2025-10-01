#!/usr/bin/env python3
"""
Generate the different paths used by cellects.
Adjust the path names according to the current position of the software
"""

import os
from pathlib import Path
CURR_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
# Add a .parent when trying to build an executable (remove it when trying to build a python package)
CELLECTS_DIR = CURR_DIR.parent#.parent # Add the ".parent" when creating a windows executable

CONFIG_DIR = CELLECTS_DIR / "config"
CORE_DIR = CELLECTS_DIR / "core"
GUI_DIR = CELLECTS_DIR / "gui"
ICONS_DIR = CELLECTS_DIR / "icons"
IMAGE_ANALYSIS_DIR = CELLECTS_DIR / "image_analysis"
UTILS_DIR = CELLECTS_DIR / "utils"
DATA_DIR = CELLECTS_DIR / "data"
TEST_DIR = CELLECTS_DIR / "test"
ALL_VARS_PKL_FILE = CONFIG_DIR / "all_vars.pkl"
