#!/usr/bin/env python3
"""Launcher of cellects software.

 """
import logging
import sys
import coloredlogs
from PySide6 import QtWidgets, QtGui
from cellects.core.cellects_paths import ICONS_DIR

# These two lines allow the taskbar icon to be cellects_icon instead if python icon.
if sys.platform.startswith('win'):
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('company.app.1')


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

    # Add the icon file to the app
    icon = QtGui.QIcon()
    if sys.platform.startswith('win'):
        icon.addPixmap(QtGui.QPixmap(ICONS_DIR / "cellects_icon.ico"))
    else:
        icon.addPixmap(QtGui.QPixmap(ICONS_DIR / "cellects_icon.icns"))
    app.setWindowIcon(icon)
    # Start session
    session = CellectsMainWidget()
    session.instantiate()
    session.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_cellects()
