#!/usr/bin/env python3

"""
Main window for Cellects application managing stacked workflow steps.

This module implements a user-assisted image analysis workflow using a QStackedWidget to navigate between configuration
 and analysis phases. It provides windows for data setup, image/video processing, output requirements, and
 advanced parameters. Automatic algorithm suggestions are offered at each step while allowing full user customization.
 Uses SaveAllVarsThread in background operations to maintain UI responsiveness.

Main Components
CellectsMainWidget : Central stacked widget managing workflow navigation
LoadingPopup : Progress bar during Cellects launch

Notes
Uses QThread for background operations to maintain UI responsiveness.
"""
import logging
import time
import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore
from cellects.core.program_organizer import ProgramOrganizer
from cellects.core.cellects_threads import SaveAllVarsThread, PrecompileNJITThread
from cellects.gui.custom_widgets import backgroundcolor, night_background_color, FixedText
from cellects.gui.advanced_parameters import AdvancedParameters
from cellects.gui.first_window import FirstWindow
from cellects.gui.if_several_folders_window import IfSeveralFoldersWindow
from cellects.gui.image_analysis_window import ImageAnalysisWindow
from cellects.gui.required_output import RequiredOutput
from cellects.gui.video_analysis_window import VideoAnalysisWindow


class CellectsMainWidget(QtWidgets.QStackedWidget):
    """ Main widget: this is the main window of the Cellects application. """

    def __init__(self):
        """

        Initializes the Cellects application window and sets up initial state.

        Sets the title, dimensions, and default values for various attributes
        required to manage the GUI's state and display settings.
        Initializes a ProgramOrganizer object and loads its variable dictionary.

        Attributes
        ----------
        pre_processing_done : bool
            Indicates whether pre-processing has been completed.
        last_is_first : bool
            Tracks if the last operation was the first in sequence.
        last_tab : str
            The most recently accessed tab name (default: "data_specifications").
        screen_height : int
            Height of the monitor in pixels.
        screen_width : int
            Width of the monitor in pixels.
        image_window_width_diff : int
            Difference in width between image window and max image size.
        image_window_height_diff : int
            Difference in height between image window and max image size.
        image_to_display : ndarray
            Placeholder for the image to be displayed (initialized as zeros).
        i : int
            Counter or index used in the application.
        po : ProgramOrganizer
            Instance managing the organization and variables of the program.
        """
        super().__init__()

        self.setWindowTitle('Cellects')
        self.thread_dict = {}
        # self.thread_dict['PrecompileNJIT'] = PrecompileNJITThread()
        # self.thread_dict['PrecompileNJIT'].start()
        self.pre_processing_done: bool = False
        self.last_is_first: bool = True
        self.last_tab: str = "data_specifications"
        self.pre_processing_done: bool = False
        self.i = 1

        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        self.po = ProgramOrganizer()
        self.po.load_variable_dict()
        self.apply_window_limits()

    def instantiate_cellects(self):
        """
        Initiates the Cellects application by setting up the main window and starting various threads.

        Extended Description
        ---------------------
        This method is responsible for initializing the Cellects application. It sets up the main window, creates necessary widgets, and starts the required threads for background operations.

        Other Parameters
        ----------------
        night_mode : bool, optional
            Indicates whether the application should run in night mode. This parameter is managed by another part of
            the code and should not be set directly.
        """
        logging.info("Instantiate Cellects")
        self.firstwindow = FirstWindow(self.po,
            self,
            night_mode=self.po.all['night_mode'])
        self.insertWidget(0, self.firstwindow)

        self.instantiate_widgets()

        self.thread_dict['SaveAllVars'] = SaveAllVarsThread(self.po, self)
        self.change_widget(0)
        self.center()

    def instantiate_widgets(self, several_folder_included: bool=True):
        """
        Instantiate various windows for the application's GUI.

        This function configures the main GUI windows for image and video analysis,
        output requirements, and advanced parameters.

        Parameters
        ----------
        several_folder_included: bool, optional
            A flag to determine whether the `IfSeveralFoldersWindow` should be instantiated. Default is `True`.
        """
        logging.info("Other widgets are instantiating")
        if several_folder_included:
            self.ifseveralfolderswindow = IfSeveralFoldersWindow(self.po, self, night_mode=self.po.all['night_mode'])
            self.insertWidget(1, self.ifseveralfolderswindow)
        self.imageanalysiswindow = ImageAnalysisWindow(self.po, self, night_mode=self.po.all['night_mode'])
        self.insertWidget(2, self.imageanalysiswindow)

        self.videoanalysiswindow = VideoAnalysisWindow(self.po, self, night_mode=self.po.all['night_mode'])
        self.insertWidget(3, self.videoanalysiswindow)

        self.requiredoutputwindow = RequiredOutput(self.po, self, night_mode=self.po.all['night_mode'])
        self.insertWidget(4, self.requiredoutputwindow)

        self.advancedparameterswindow = AdvancedParameters(self.po, self, night_mode=self.po.all['night_mode'])
        self.insertWidget(5, self.advancedparameterswindow)


    def update_widget(self, idx: int, widget_to_call):
        """ Update widget at its position (idx) in the stack """
        self.insertWidget(idx, widget_to_call)

    def change_widget(self, idx: int):
        """ Display a widget using its position (idx) in the stack """
        self.setCurrentIndex(idx)  # Index that new widget
        self.updateGeometry()
        self.currentWidget().setVisible(True)
        if idx == 3 or idx == 5:
            self.currentWidget().display_conditionally_visible_widgets()

    def center(self):
        """
        Centers the window on the screen.

        Moves the window to the center of the available screen geometry.
        Allows users to always see the application's windows in a consistent
        position, regardless of screen resolution or window size.
        """
        qr = self.frameGeometry()
        if self.size().height() < self.screen_height - 150:
            # cp = QtWidgets.QDesktopWidget().availableGeometry().center()  # PyQt 5/*
            cp = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()  # Pyside 6
            qr.moveCenter(cp)
        self.move(qr.topLeft())

    def apply_window_limits(self):
        screen = QtGui.QGuiApplication.primaryScreen()
        if not screen:
            return
        geom = screen.availableGeometry()
        self.screen_width, self.screen_height = geom.width(), geom.height()
        self.win_width, self.win_height = min(self.screen_width, 1380), min(self.screen_height, 750)

        self.setMaximumSize(self.screen_width, self.screen_height)
        self.resize(self.win_width, self.win_height)

    def closeEvent(self, event):
        """
        Close the application window and handle cleanup.

        Parameters
        ----------
        event : QCloseEvent
            The close event that triggered this function.

        Notes
        -----
        This function does not return any value and is intended for event
        handling purposes only.
        """
        reply = QtWidgets.QMessageBox.question(
            self,
            'Closing Cellects',
            'Are you sure you want to exit?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            logging.info("Interrupting all threads.")
            if self.count() >= 6:
                windows = [self.thread_dict.items(), self.imageanalysiswindow.thread_dict.items(),
                           self.ifseveralfolderswindow.thread_dict.items(),
                           self.videoanalysiswindow.thread_dict.items(), self.firstwindow.thread_dict.items()]
                for window in windows:
                    for thread_name, thread in window:
                        thread.requestInterruption()
                        thread.wait(20000)
            logging.info("Closing main window.")
            event.accept()
        else:
            event.ignore()


class LoadingPopup(QtWidgets.QWidget):
    """
    A progress bar displaying njit precompilation progress during Cellects launches.
    """
    def __init__(self):
        super().__init__()

        self.i: int = 0
        self.total: int = 18

        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint |
                            QtCore.Qt.WindowType.WindowStaysOnTopHint |
                            QtCore.Qt.WindowType.NoDropShadowWindowHint)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(300, 6)  # Fixed dimensions for a clean bar look

        # Zero margins so only the bar fills the widget
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setTextVisible(False)  # No text allowed
        self.progress.setRange(0, 100)  # Infinite marquee loading animation
        self.progress.setValue(0)

        self.progress.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: rgba(0, 0, 0, 10);
                border-radius: 15px;
            }
            
            QProgressBar::chunk {
                /* Forces pill-shaped rounded ends on both sides */
                border-radius: 15px; 
                
                /* Radial gradient creates a spherical glass orb effect rather than a flat sheet */
                background: qradialgradient(
                    cx: 0.5, cy: 0.4, radius: 0.9,
                    fx: 0.5, fy: 0.2,
                    stop: 0 #ffffff,       /* Ultra-bright white reflection spot at the top-center */
                    stop: 0.2 #555555,     /* Soft silver reflection falloff */
                    stop: 0.5 #1a1a1a,     /* Deep obsidian core */
                    stop: 0.8 #000000,     /* Pitch black base */
                    stop: 1.0 #0d0d0d      /* Outer shadow definition */
                );
            
                /* Clean, bright edge highlight to accentuate the curved glass profile */
                border: 1px solid rgba(255, 255, 255, 0.25);
            }
        """)

        layout.addWidget(self.progress)

    def start_application(self):
        """
        Start Cellects application
        """
        self.session = CellectsMainWidget()
        self.session.instantiate_cellects()
        self.add_progress()
        self.session.show()
        self.close()

    def add_progress(self):
        """
        Helper to increment the progress bar
        """
        self.i += 1
        self.progress.setValue(int(self.i / self.total * 100))
        QtWidgets.QApplication.processEvents()