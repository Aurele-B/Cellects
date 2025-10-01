#!/usr/bin/env python3
"""
This module contains the Widget that stacks all Cellects widget.
An important note is that user-friendliness and versatility are at the core of Cellects’ development.
Henceforth, the general workflow is in fact, a user-assisted workflow. Every step of the general workflow can
be fine-tuned by the user to cope with various imaging conditions. At every step, automatic algorithms suggest options
to help the user to find the best parameters.
"""
import logging
import signal

# necessary on OSX
# pip install cython pyobjus
import sys
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

from cellects.core.cellects_paths import ICONS_DIR


class CellectsMainWidget(QtWidgets.QStackedWidget):
    """ Main widget: this is the main window """

    def __init__(self):
        super().__init__()

        self.setWindowTitle('Cellects')
        self.pre_processing_done: bool = False
        self.last_is_first: bool = True
        self.last_tab: str = "data_specifications"
        self.pre_processing_done: bool = False
        self.screen_height = get_monitors()[0].height
        self.screen_width = get_monitors()[0].width
        self.im_max_width = 570  # self.screen_width // 5 374, 296
        self.im_max_height = 266  # self.screen_height // 5 (1369, 778) (1380, 834)
        # self.image_window_width_ratio = 570/1380
        # self.image_window_height_ratio = 266/750
        self.image_window_width_diff = 1380 - 570
        self.image_window_height_diff = 750 - 266
        self.image_to_display = np.zeros((self.im_max_height, self.im_max_width, 3), np.uint8)
        # self.im_max_height = (2 * self.screen_height // 3) // 3
        # self.im_max_width = (2 * self.screen_width // 3) // 4
        self.i = 1
        self.po = ProgramOrganizer()
        # self.subwidgets_stack.po = self.po
        self.po.load_variable_dict()
        # self.parent().po.all['night_mode'] = True

        # self.resize(4 * self.screen_width // 5, 4 * self.screen_height // 5)
        self.resize(1380, 750)

        # self.setMaximumWidth(self.screen_width)
        # self.setMaximumHeight(self.screen_height)
        #
        # self.setSizePolicy(
        #     QtWidgets.QSizePolicy.Maximum,
        #     QtWidgets.QSizePolicy.Maximum)
        #
        # self.setSizePolicy(
        #     QtWidgets.QSizePolicy.Minimum,
        #     QtWidgets.QSizePolicy.Minimum)
        # self.setSizePolicy(
        #     QtWidgets.QSizePolicy.Expanding,
        #     QtWidgets.QSizePolicy.Expanding)

    def instantiate(self):

        logging.info("instantiate")
        self.firstwindow = FirstWindow(
            self,
            night_mode=self.po.all['night_mode'])
        self.insertWidget(0, self.firstwindow)

        self.instantiate_widgets()

        self.thread = {}
        self.thread['SaveAllVars'] = SaveAllVarsThread(self)
        self.change_widget(0)
        self.center()

    def instantiate_widgets(self, severalfolder_included=True):
        print("Widgets are instantiating")
        # for widg_i in np.arange(1, 6):
        #     widget = self.widget(1)
        #     if widget is not None:
        #         self.removeWidget(widget)
        #         # widget.deleteLater()
        if severalfolder_included:
            self.ifseveralfolderswindow = IfSeveralFoldersWindow(self, night_mode=self.po.all['night_mode'])
            self.insertWidget(1, self.ifseveralfolderswindow)
            # self.ifseveralfolderswindow.setVisible(True)
        self.imageanalysiswindow = ImageAnalysisWindow(self, night_mode=self.po.all['night_mode'])
        self.insertWidget(2, self.imageanalysiswindow)

        self.videoanalysiswindow = VideoAnalysisWindow(self, night_mode=self.po.all['night_mode'])
        self.insertWidget(3, self.videoanalysiswindow)

        self.requiredoutputwindow = RequiredOutput(self, night_mode=self.po.all['night_mode'])
        self.insertWidget(4, self.requiredoutputwindow)

        self.advancedparameterswindow = AdvancedParameters(self, night_mode=self.po.all['night_mode'])
        self.insertWidget(5, self.advancedparameterswindow)

        # self.requiredoutputwindow = RequiredOutput(
        #     self, night_mode=self.po.all['night_mode'])
        # self.insertWidget(4, self.requiredoutputwindow)
        #
        # self.advancedparameterswindow = AdvancedParameters(
        #     self, night_mode=self.po.all['night_mode'])
        # self.insertWidget(5, self.requiredoutputwindow)


    def update_widget(self, idx, widget_to_call):
        """ Update widget at its position (idx) in the stack """
        self.insertWidget(idx, widget_to_call)

    def change_widget(self, idx):
        """ Display a widget using its position (idx) in the stack """
        self.setCurrentIndex(idx)  # Index that new widget
        self.updateGeometry()
        self.currentWidget().setVisible(True)
        if idx == 3 or idx == 5:
            self.currentWidget().display_conditionally_visible_widgets()

    def center(self):
        qr = self.frameGeometry()
        # cp = QtWidgets.QDesktopWidget().availableGeometry().center()  # PyQt 5
        cp = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()  # Pyside 6
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(
            self,
            'Closing Cellects',
            'Are you sure you want to exit?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            signal.signal(signal.SIGSEGV, signal.SIG_IGN)
            logging.info("Closing main window.")
            event.accept()
            # QtWidgets.QApplication.quit()
            self.close()
        else:
            event.ignore()
        # self.hide()

    # def update_all_settings(self):
    #     self.firstwindow
    #     self.imageanalysiswindow
    #     self.videoanalysiswindow
    #     requiredoutputwindow = self.widget(4)
    #     advancedparameterswindow = self.widget(5)



"""
Ajouter
are_gravity_centers_moving
Retirer advanced mode de third widget
Inverser gauche et droite de third widget

drop_nak1 Réduction de la taille de la forme d'origine lors du passage de
luminosity_segmentation à luminosity_segmentation + gradient_segmentation

Catégories d'Advanced parameters:

- Spatio-temporal scales: time interval, timmings, convert,
- Analysis parameters: crop, subtract background
- Computer resources: Parallel, proc, ram,
- Video saving: fps, over unaltered, keep unaltered, save processed
- Special cases: correct error around initial shape, connect distant shape,
  appearing size threshold, appearing detection method, oscillation period

if image_analysis is done:
    get_average_pixel_size
"""
