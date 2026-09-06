#!/usr/bin/env python3
"""
Launcher for the Cellects software.

This module initializes logging configuration, creates the Qt application instance,
loads Cellects icon, and launches the main GUI interface.
"""
import os
import sys
import logging
import tempfile
import pathlib
import traceback
import faulthandler
import atexit
import threading

def setup_debug_logging():
    log_path = pathlib.Path(tempfile.gettempdir()) / "Cellects_debug.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove duplicate handlers if called multiple times
    for h in list(logger.handlers):
        logger.removeHandler(h)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(process)s | %(threadName)s | %(levelname)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    faulthandler_file = open(log_path, "a", encoding="utf-8")
    atexit.register(faulthandler_file.close)
    faulthandler.enable(file=faulthandler_file)

    logging.info(
        "Debug log path: %s",
        log_path
    )
    logging.info(
        "cwd=%s executable=%s _MEIPASS=%s",
        os.getcwd(),
        sys.executable,
        getattr(sys, "_MEIPASS", None)
    )

setup_debug_logging()

import sys
import logging
import coloredlogs
from pathlib import Path

from cellects.gui.numba_precompilation import warming_up_numba_functions


def get_icon_path():
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "icons"
    return Path(__file__).parent / "icons"


ICONS_DIR = get_icon_path()
# from cellects.core.cellects_paths import ICONS_DIR

if sys.platform.startswith('win'):
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("cellects.app")
    except Exception as e:
        logging.getLogger(__name__).debug(f"Windows taskbar icon setup failed: {e}")

from PySide6 import QtWidgets, QtGui

LOGLEVEL = "INFO" # Set to DEBUG for development


def _initialize_coloredlogs(loglevel: str = 'DEBUG') -> None:

    """Initialize colored console logging with custom format.

    Parameters
    ----------
    loglevel : str, optional
        Logging threshold level (default is DEBUG). Accepts standard Python
        logging level strings like 'DEBUG', 'INFO', or 'WARNING'.

    Notes
    -----
    This function must be called before any other logging setup to ensure proper
    configuration of colored output.
    """
    # Configure basic logging before applying colored logs
    logging.basicConfig(level=loglevel)

    # Apply colored formatting to the root logger
    coloredlogs.install(
        logger=logging.basicConfig(),
        level=loglevel,
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S')


def run_cellects():
    """Run the Cellects application entry point.

        This function initializes the Qt application, loads platform-specific icons,
        creates and displays the main window widget, then starts the event loop.

        Raises
        ------
        ImportError
            If required GUI components cannot be loaded
        """
    _initialize_coloredlogs(LOGLEVEL)

    try:
        from cellects.gui.cellects import LoadingPopup

        # Initialize application
        app = QtWidgets.QApplication([])
        # Set custom window icon for taskbar
        icon = None
        if sys.platform.startswith('win'):
            icon = QtGui.QIcon(str(ICONS_DIR / "cellects_icon.ico"))
        elif sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
            icon = QtGui.QIcon(str(ICONS_DIR / "cellects_icon.png"))
        if icon is not None:
            app.setWindowIcon(icon)

        # Create and display loading bar and main window
        loading = LoadingPopup()
        loading.show()
        warming_up_numba_functions(loading)
        loading.start_application()

        # Set exit
        sys.exit(app.exec())
    except Exception as e:
        logging.getLogger(__name__).critical("Cellects failed to start", exc_info=True)
        raise


if __name__ == "__main__":
    run_cellects()
