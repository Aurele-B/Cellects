#!/usr/bin/env python3

"""
Main window for Cellects application managing stacked workflow steps.

This module implements a user-assisted image analysis workflow using a QStackedWidget to navigate between configuration
 and analysis phases. It provides windows for data setup, image/video processing, output requirements, and
 advanced parameters. Automatic algorithm suggestions are offered at each step while allowing full user customization.
 Uses SaveAllVarsThread in background operations to maintain UI responsiveness.

Main Components
CellectsMainWidget : Central stacked widget managing workflow navigation

Notes
Uses QThread for background operations to maintain UI responsiveness.
"""
import logging
import signal
import numpy as np
from PySide6 import QtWidgets, QtGui
from screeninfo import get_monitors
from cellects.core.program_organizer import ProgramOrganizer
from cellects.core.cellects_threads import SaveAllVarsThread
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
        im_max_width : int
            Maximum width allowed for displayed images (default: 570).
        im_max_height : int
            Maximum height allowed for displayed images (default: 266).
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
        self.pre_processing_done: bool = False
        self.last_is_first: bool = True
        self.last_tab: str = "data_specifications"
        self.pre_processing_done: bool = False
        self.screen_height = get_monitors()[0].height
        self.screen_width = get_monitors()[0].width
        self.im_max_width = 570
        self.im_max_height = 266
        self.image_window_width_diff = 1380 - 570
        self.image_window_height_diff = 750 - 266
        self.image_to_display = np.zeros((self.im_max_height, self.im_max_width, 3), np.uint8)
        self.i = 1
        self.po = ProgramOrganizer()

        self.po.load_variable_dict()
        self.resize(1380, 750)

    def instantiate(self):
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
        self.firstwindow = FirstWindow(
            self,
            night_mode=self.po.all['night_mode'])
        self.insertWidget(0, self.firstwindow)

        self.instantiate_widgets()

        self.thread_dict = {}
        self.thread_dict['SaveAllVars'] = SaveAllVarsThread(self)
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
        logging.info("Widgets are instantiating")
        if several_folder_included:
            self.ifseveralfolderswindow = IfSeveralFoldersWindow(self, night_mode=self.po.all['night_mode'])
            self.insertWidget(1, self.ifseveralfolderswindow)
        self.imageanalysiswindow = ImageAnalysisWindow(self, night_mode=self.po.all['night_mode'])
        self.insertWidget(2, self.imageanalysiswindow)

        self.videoanalysiswindow = VideoAnalysisWindow(self, night_mode=self.po.all['night_mode'])
        self.insertWidget(3, self.videoanalysiswindow)

        self.requiredoutputwindow = RequiredOutput(self, night_mode=self.po.all['night_mode'])
        self.insertWidget(4, self.requiredoutputwindow)

        self.advancedparameterswindow = AdvancedParameters(self, night_mode=self.po.all['night_mode'])
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
        # cp = QtWidgets.QDesktopWidget().availableGeometry().center()  # PyQt 5
        cp = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()  # Pyside 6
        qr.moveCenter(cp)
        self.move(qr.topLeft())

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
            for _, thread in self.imageanalysiswindow.thread_dict.items():
                thread.wait()
            for _, thread  in self.ifseveralfolderswindow.thread_dict.items():
                thread.wait()
            for _, thread  in self.videoanalysiswindow.thread_dict.items():
                thread.wait()
            for _, thread  in self.firstwindow.thread_dict.items():
                thread.wait()
            logging.info("Closing main window.")
            event.accept()
        else:
            event.ignore()
