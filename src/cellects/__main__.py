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


def run_cellects():
    """Entry point of cellects software."""
    _initialize_coloredlogs(LOGLEVEL)
    from cellects.gui.cellects import CellectsMainWidget
    app = QtWidgets.QApplication([])
    session = CellectsMainWidget()
    session.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_cellects()

