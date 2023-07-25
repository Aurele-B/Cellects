#!/usr/bin/env python3
"""ADD DETAIL OF THE MODULE"""
import logging
import signal

# necessary on OSX
# pip install cython pyobjus

from numpy import min, max, all, any, uint8, zeros, arange
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
    """ Main widget: this is the main window """

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cellects')
        self.pre_processing_done: bool = False
        self.last_is_first: bool = True
        self.last_is_image_analysis: bool = False
        self.pre_processing_done: bool = False
        self.screen_height = get_monitors()[0].height
        self.screen_width = get_monitors()[0].width
        self.im_max_width = self.screen_width // 3
        self.im_max_height = self.screen_height // 3
        self.image_to_display = zeros(
            (self.im_max_height, self.im_max_width, 3),
            uint8)
        # self.im_max_height = (2 * self.screen_height // 3) // 3
        # self.im_max_width = (2 * self.screen_width // 3) // 4
        self.i = 1
        self.po = ProgramOrganizer()
        # self.subwidgets_stack.po = self.po
        self.po.load_variable_dict()
        # self.parent().po.all['night_mode'] = True

        #self.resize(4 * self.screen_width // 5, 4 * self.screen_height // 5)
        self.resize(4 * self.screen_width // 5, 4 * self.screen_height // 5)
        #self.setMaximumWidth(self.screen_width)
        #self.setMaximumHeight(self.screen_height)
        self.setMaximumWidth(822)
        self.setMaximumHeight(462)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum,
            QtWidgets.QSizePolicy.Maximum)

        self.firstwindow = FirstWindow(
            self,
            night_mode=self.po.all['night_mode'])
        self.insertWidget(0, self.firstwindow)

        self.instantiate_widgets()

        # self.insertWidget(
        #     3,
        #     VideoAnalysisWindow(self),
        #     night_mode=self.po.all['night_mode'])
        # self.insertWidget(
        #     4,
        #     RequiredOutput(self),
        #     night_mode=self.po.all['night_mode'])
        # self.insertWidget(
        #     5,
        #     AdvancedParameters(self),
        #     night_mode=self.po.all['night_mode'])
        self.thread = {}
        self.thread['SaveAllVars'] = SaveAllVarsThread(self)
        self.change_widget(0)
        self.center()

    def instantiate_widgets(self, severalfolder_included=True):
        print("Widgets are instantiating")
        # for widg_i in arange(1, 6):
        #     widget = self.widget(1)
        #     if widget is not None:
        #         self.removeWidget(widget)
        #         # widget.deleteLater()

        if severalfolder_included:
            self.ifseveralfolderswindow = IfSeveralFoldersWindow(
                self, night_mode=self.po.all['night_mode'])
            self.insertWidget(1, self.ifseveralfolderswindow)
            # self.ifseveralfolderswindow.setVisible(True)
        self.imageanalysiswindow = ImageAnalysisWindow(
            self, night_mode=self.po.all['night_mode'])
        self.insertWidget(2, self.imageanalysiswindow)

        self.videoanalysiswindow = VideoAnalysisWindow(
            self, night_mode=self.po.all['night_mode'])
        self.insertWidget(3, self.videoanalysiswindow)

        self.insertWidget(4, RequiredOutput(
            self, night_mode=self.po.all['night_mode']))

        self.insertWidget(5, AdvancedParameters(
            self, night_mode=self.po.all['night_mode']))

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
