#!/usr/bin/env python3
"""
Launcher for the Cellects software.

This module initializes logging configuration, creates the Qt application instance,
loads Cellects icon, and launches the main GUI interface.
"""

import sys
import logging
import coloredlogs
from PySide6 import QtWidgets, QtGui
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("cellects.app")
    except Exception as e:
        logging.getLogger(__name__).debug(f"Windows taskbar icon setup failed: {e}")

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
        from cellects.gui.cellects import CellectsMainWidget

        # Initialize application
        app = QtWidgets.QApplication([])

        # Set custom window icon for taskbar (platform-specific handling)
        icon = QtGui.QIcon()
        if hasattr(sys, "_MEIPASS"):
            ICONS_DIR = Path(sys._MEIPASS) / "icons"
        else:
            ICONS_DIR = Path(__file__).parent / "icons"
        platform_icon_path = (
            ICONS_DIR / "cellects_icon.ico" if sys.platform.startswith('win')
            else ICONS_DIR / "cellects_icon.icns"
        )
        icon.addPixmap(QtGui.QPixmap(str(platform_icon_path)))
        app.setWindowIcon(icon)

        # Create and display main window
        session = CellectsMainWidget()
        session.instantiate()
        session.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.getLogger(__name__).critical("Application failed to start", exc_info=True)
        raise


if __name__ == "__main__":
    run_cellects()
