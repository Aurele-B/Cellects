#!/usr/bin/env python3
"""Launcher of cellects software."""
import logging
import os
import sys
from pathlib import Path

import coloredlogs

from PySide6 import QtWidgets

LOGLEVEL = "INFO" #"DEBUG"


def _initialize_coloredlogs(loglevel: str = 'DEBUG') -> None:
    """Initialize logging parameters with coloredlogs library."""
    coloredlogs.install(
        logger=logging.basicConfig(),
        level=loglevel,
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S')


def add_import_path():
    CURRENT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
    split_dir = os.path.split(CURRENT_DIR)
    while split_dir[1] != "Cellects":
        split_dir = os.path.split(split_dir[0])
    SRC_DIR = Path(split_dir[0]) / "Cellects" / "src"
    sys.path.append(str(SRC_DIR))

def run_cellects():
    """Entry point of cellects software."""
    _initialize_coloredlogs(LOGLEVEL)
    add_import_path()
    # from src.cellects.gui.cellects import CellectsMainWidget
    from cellects.gui.cellects import CellectsMainWidget
    app = QtWidgets.QApplication([])
    session = CellectsMainWidget()
    session.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_cellects()

