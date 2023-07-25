#!/usr/bin/env python3
"""Contains the paths used by cellects."""

import os
from pathlib import Path
CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
split_dir = os.path.split(CURRENT_DIR)
while split_dir[1] != "Cellects":
    split_dir = os.path.split(split_dir[0])

if "config" in os.listdir(Path(split_dir[0]) / "Cellects"):
    CELLECTS_DIR = Path(split_dir[0]) / "Cellects"
else:
    SRC_DIR = Path(split_dir[0]) / "Cellects" / "src"
    CELLECTS_DIR = SRC_DIR / "cellects"
# CORE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
# # # Add a .parent when trying to build a python package or an executable
# CELLECTS_DIR = CORE_DIR#.parent#.parent.parent # Add the ".parent" when creating a windows executable
# CELLECTS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))

CONFIG_DIR = CELLECTS_DIR / "config"
CORE_DIR = CELLECTS_DIR / "core"
GUI_DIR = CELLECTS_DIR / "gui"
ICONS_DIR = CELLECTS_DIR / "icons"
IMAGE_ANALYSIS_DIR = CELLECTS_DIR / "image_analysis"
UTILS_DIR = CELLECTS_DIR / "utils"
DATA_DIR = CELLECTS_DIR / "data"
TEST_DIR = CELLECTS_DIR / "test"
ALL_VARS_PKL_FILE = CONFIG_DIR / "all_vars.pkl"
