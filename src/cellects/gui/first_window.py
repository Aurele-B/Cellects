#!/usr/bin/env python3
"""This module creates the First window of the user interface of Cellects"""

import os
import logging
from pathlib import Path
import numpy as np
import cv2
from PySide6 import QtWidgets, QtCore

from cellects.core.cellects_threads import (
    GetFirstImThread, GetExifDataThread, RunAllThread, LookForDataThreadInFirstW, LoadDataToRunCellectsQuicklyThread)
from cellects.gui.custom_widgets import (
    MainTabsType, InsertImage, FullScreenImage, PButton, Spinbox,
    Combobox, FixedText, EditText, LineWidget)


class FirstWindow(MainTabsType):
    def __init__(self, parent, night_mode):
        super().__init__(parent, night_mode)
        logging.info("Initialize first window")
        self.setParent(parent)
        self.data_tab.set_in_use()
        self.image_tab.set_not_usable()
        self.video_tab.set_not_usable()
        # self.night_mode_switch(False)
        self.thread = {}
        self.thread["LookForData"] = LookForDataThreadInFirstW(self.parent())
        self.thread["RunAll"] = RunAllThread(self.parent())
        self.thread["LoadDataToRunCellectsQuickly"] = LoadDataToRunCellectsQuicklyThread(self.parent())
        self.thread["GetFirstIm"] = GetFirstImThread(self.parent())
        self.thread["GetExifDataThread"] = GetExifDataThread(self.parent())
        self.instantiate: bool = True
        ##
        self.title_label = FixedText('Cellects', police=60, night_mode=self.parent().po.all['night_mode'])
        self.title_label.setAlignment(QtCore.Qt.AlignHCenter)
        # self.subtitle_label = FixedText('A Cell Expansion Computer Tracking Software', police=18, night_mode=self.parent().po.all['night_mode'])
        # self.subtitle_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.subtitle_line = LineWidget(size=[1, 50], night_mode=self.parent().po.all['night_mode'])

        self.Vlayout.addWidget(self.title_label)
        # self.Vlayout.addWidget(self.subtitle_label)
        self.Vlayout.addWidget(self.subtitle_line)
        self.Vlayout.addItem(self.vertical_space)

        # 1) Set if this a Image list or Videos
        # Open the layout:
        self.second_row_widget = QtWidgets.QWidget()
        self.second_row_layout = QtWidgets.QHBoxLayout()
        self.im_or_vid_label = FixedText('Image list or Videos:', tip="What type of data do(es) contain(s) folder(s)?", night_mode=self.parent().po.all['night_mode'])
        self.im_or_vid = Combobox(["Image list", "Videos"], self.parent().po.all['im_or_vid'], night_mode=self.parent().po.all['night_mode'])
        self.im_or_vid.setFixedWidth(150)
        # Set their positions on layout
        self.second_row_layout.addItem(self.horizontal_space)
        self.second_row_layout.addWidget(self.im_or_vid_label)
        self.second_row_layout.addWidget(self.im_or_vid)
        self.second_row_layout.addItem(self.horizontal_space)
        self.second_row_widget.setLayout(self.second_row_layout)
        self.Vlayout.addWidget(self.second_row_widget)

        # 2) Open the third row layout
        self.third_row_widget = QtWidgets.QWidget()
        self.third_row_layout = QtWidgets.QHBoxLayout()
        # Set default images radical and extension widgets
        if self.parent().po.all['im_or_vid'] == 0:
            what = 'Images'
            if self.parent().po.all['extension'] == '.mp4':
                self.parent().po.all['radical'] = 'IMG_'
                self.parent().po.all['extension'] = '.JPG'
            self.arena_number_label = FixedText('Arena number per folder:', tip="If this number is not always the same (depending on the folder), it can be changed later",
                                                night_mode=self.parent().po.all['night_mode'])

        else:
            if self.parent().po.all['extension'] == '.JPG':
                self.parent().po.all['radical'] = ''
                self.parent().po.all['extension'] = '.mp4'
            self.arena_number_label = FixedText('Arena number per video:',
                                                tip="If this number is not always the same (depending on the video), it can be changed later",
                                                night_mode=self.parent().po.all['night_mode'])
            what = 'Videos'
        self.arena_number_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.arena_number = Spinbox(min=0, max=255, val=self.parent().po.all['first_folder_sample_number'],
                                     decimals=0, night_mode=self.parent().po.all['night_mode'])
        self.arena_number.valueChanged.connect(self.re_instantiate_widgets)
        self.radical_label = FixedText(what + ' prefix:', tip="Inform the prefix common to each name, if it exists", night_mode=self.parent().po.all['night_mode'])
        self.radical_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.radical = EditText(self.parent().po.all['radical'],
                                       night_mode=self.parent().po.all['night_mode'])
        self.radical.textChanged.connect(self.re_instantiate_widgets)

        self.extension_label = FixedText(what + ' extension:', tip="Caps sensitive", night_mode=self.parent().po.all['night_mode'])
        self.extension_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.extension = EditText(self.parent().po.all['extension'],
                                       night_mode=self.parent().po.all['night_mode'])
        self.extension.textChanged.connect(self.re_instantiate_widgets)

        # Set their positions on layout
        self.third_row_layout.addItem(self.horizontal_space)
        self.third_row_layout.addWidget(self.radical_label)
        self.third_row_layout.addWidget(self.radical)
        # self.third_row_layout.addItem(self.horizontal_space)
        self.third_row_layout.addWidget(self.extension_label)
        self.third_row_layout.addWidget(self.extension)
        self.third_row_layout.addItem(self.horizontal_space)
        self.third_row_widget.setLayout(self.third_row_layout)
        self.Vlayout.addWidget(self.third_row_widget)
        # If im_or_vid changes, adjust these 2 widgets
        self.im_or_vid.currentTextChanged.connect(self.im2vid)

        # 3) Get the path to the right folder:
        # Open the layout:
        self.first_row_widget = QtWidgets.QWidget()
        self.first_row_layout = QtWidgets.QHBoxLayout()

        self.folder_label = FixedText('Folder:',
                                      tip="Path to the folder containing images or videos\nThe selected folder may also contain several folders of data",
                                      night_mode=self.parent().po.all['night_mode'])
        self.folder_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.global_pathway = EditText(self.parent().po.all['global_pathway'],
                                       night_mode=self.parent().po.all['night_mode'])
        self.global_pathway.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        self.global_pathway.textChanged.connect(self.pathway_changed)
        self.browse = PButton('Browse', night_mode=self.parent().po.all['night_mode'])
        self.browse.clicked.connect(self.browse_is_clicked)

        # Set their positions on layout
        self.first_row_layout.addWidget(self.folder_label)
        self.first_row_layout.addWidget(self.global_pathway)
        self.first_row_layout.addWidget(self.browse)
        self.first_row_widget.setLayout(self.first_row_layout)
        self.Vlayout.addWidget(self.first_row_widget)

        self.fourth_row_widget = QtWidgets.QWidget()
        self.fourth_row_layout = QtWidgets.QHBoxLayout()
        self.fourth_row_layout.addItem(self.horizontal_space)
        self.fourth_row_layout.addWidget(self.arena_number_label)
        self.fourth_row_layout.addWidget(self.arena_number)
        self.fourth_row_layout.addItem(self.horizontal_space)
        self.fourth_row_widget.setLayout(self.fourth_row_layout)
        self.Vlayout.addWidget(self.fourth_row_widget)
        self.Vlayout.addItem(self.vertical_space)

        # Add the central image display widget
        self.display_image = np.zeros((self.parent().im_max_height, self.parent().im_max_width, 3), np.uint8)
        self.display_image = InsertImage(self.display_image, self.parent().im_max_height, self.parent().im_max_width)
        self.Vlayout.addWidget(self.display_image, alignment=QtCore.Qt.AlignCenter)
        self.display_image.setVisible(False)
        self.display_image.mousePressEvent = self.full_screen_display

        # Add the display shortcuts option
        #self.shortcut_cb = Checkbox(self.parent().po.all['display_shortcuts'])
        #self.shortcut_cb.stateChanged.connect(self.display_shortcuts_checked)
        #self.shortcut_label = FixedText('Display shortcuts', night_mode=self.parent().po.all['night_mode'])
        #self.shortcut_label.setAlignment(QtCore.Qt.AlignVCenter)

        # 4) Create the shortcuts row
        self.shortcuts_widget = QtWidgets.QWidget()
        self.shortcuts_layout = QtWidgets.QHBoxLayout()
        # Add shortcuts: Video_analysis and Run directly
        # Shortcut 1 : Advanced Parameters
        self.advanced_parameters = PButton('Advanced Parameters', night_mode=self.parent().po.all['night_mode'])
        self.advanced_parameters.clicked.connect(self.advanced_parameters_is_clicked)
        # Shortcut 2 : Required Outputs
        self.required_outputs = PButton('Required Outputs', night_mode=self.parent().po.all['night_mode'])
        self.required_outputs.clicked.connect(self.required_outputs_is_clicked)
        # Shortcut 3 :
        # self.Video_analysis_window = PButton("Video tracking window", night_mode=self.parent().po.all['night_mode'])
        # self.Video_analysis_window.clicked.connect(self.video_analysis_window_is_clicked)
        self.video_tab.clicked.connect(self.video_analysis_window_is_clicked)
        # Shortcut 4 :
        self.Run_all_directly = PButton("Run all directly", night_mode=self.parent().po.all['night_mode'])
        self.Run_all_directly.clicked.connect(self.Run_all_directly_is_clicked)
        # self.Video_analysis_window.setVisible(False)
        self.Run_all_directly.setVisible(False)

        self.shortcuts_layout.addItem(self.horizontal_space)
        self.shortcuts_layout.addWidget(self.advanced_parameters)
        self.shortcuts_layout.addWidget(self.required_outputs)
        # self.shortcuts_layout.addWidget(self.Video_analysis_window)
        self.shortcuts_layout.addWidget(self.Run_all_directly)
        self.shortcuts_layout.addItem(self.horizontal_space)
        self.shortcuts_widget.setLayout(self.shortcuts_layout)
        self.Vlayout.addWidget(self.shortcuts_widget)

        # 5) Open the last row layout
        self.last_row_widget = QtWidgets.QWidget()
        self.last_row_layout = QtWidgets.QHBoxLayout()

        # Message
        self.message = FixedText('', halign='r', night_mode=self.parent().po.all['night_mode'])
        self.message.setStyleSheet("color: rgb(230, 145, 18)")
        # Next button
        self.next = PButton('Next', night_mode=self.parent().po.all['night_mode'])
        self.image_tab.clicked.connect(self.next_is_clicked)
        self.next.clicked.connect(self.next_is_clicked)
        # Add widgets to the last_row_layout
        #self.last_row_layout.addWidget(self.shortcut_cb)
        #self.last_row_layout.addWidget(self.shortcut_label)
        self.last_row_layout.addItem(self.horizontal_space)
        self.last_row_layout.addWidget(self.message)
        self.last_row_layout.addWidget(self.next)
        # Close the last_row_layout
        self.last_row_widget.setLayout(self.last_row_layout)
        self.Vlayout.addWidget(self.last_row_widget)
        self.setLayout(self.Vlayout)

        # Check if there is data in the saved folder
        self.pathway_changed()

    def full_screen_display(self, event):
        self.popup_img = FullScreenImage(self.parent().image_to_display, self.parent().screen_width, self.parent().screen_height)
        self.popup_img.show()

    def browse_is_clicked(self):
        dialog = QtWidgets.QFileDialog()
        dialog.setDirectory(str(self.parent().po.all['global_pathway']))
        self.parent().po.all['global_pathway'] = dialog.getExistingDirectory(self,
                                                                             'Select a folder containing images (/videos) or folders of data images (/videos)')
        self.global_pathway.setText(self.parent().po.all['global_pathway'])

    def im2vid(self):
        if self.im_or_vid.currentText() == "Image list":
            what = 'Images'
            if self.parent().po.all['extension'] == '.mp4':
                self.parent().po.all['radical'] = 'IMG_'
                self.parent().po.all['extension'] = '.JPG'
        else:
            if self.parent().po.all['extension'] == '.JPG':
                self.parent().po.all['radical'] = ''
                self.parent().po.all['extension'] = '.mp4'
            what = 'Videos'
        self.radical_label.setText(what + ' prefix:')
        self.extension_label.setText(what + ' extension:')
        self.radical.setText(self.parent().po.all['radical'])
        self.extension.setText(self.parent().po.all['extension'])

    def display_message_from_thread(self, text_from_thread):
        self.message.setText(text_from_thread)

    def display_image_during_thread(self, dictionary):
        self.message.setText(dictionary['message'])
        self.parent().image_to_display = dictionary['current_image']
        self.display_image.update_image(dictionary['current_image'])

    def next_is_clicked(self):
        if not self.thread["LookForData"].isRunning() and not self.thread["RunAll"].isRunning():
            self.parent().po.all['im_or_vid'] = self.im_or_vid.currentIndex()
            self.parent().po.all['radical'] = self.radical.text()
            self.parent().po.all['extension'] = self.extension.text()
            #self.parent().po.all['display_shortcuts'] = self.shortcut_cb.isChecked()
            self.parent().po.sample_number = int(self.arena_number.value())
            self.parent().po.all['first_folder_sample_number'] = self.parent().po.sample_number
            self.parent().po.all['sample_number_per_folder'] = [self.parent().po.sample_number]
            if not self.instantiate:  # not self.parent().imageanalysiswindow.initialized:
                logging.info("No need to look for data, images or videos already found previously.")
                self.first_im_read(True)
            else:
                self.parent().po.all['global_pathway'] = Path(self.global_pathway.text())
                if not os.path.isdir(Path(self.parent().po.all['global_pathway'])):
                    self.message.setText('The folder selected is not valid')
                else:
                    self.message.setText('')
                    # self.parent().po.all['im_or_vid'] = self.im_or_vid.currentIndex()
                    # self.parent().po.all['radical'] = self.radical.text()
                    # self.parent().po.all['extension'] = self.extension.text()
                    # self.parent().po.all['display_shortcuts'] = self.shortcut_cb.isChecked()

                    self.message.setText(f"Looking for {self.parent().po.all['radical']}***{self.parent().po.all['extension']} Wait...")
                    self.message.setStyleSheet("color: rgb(230, 145, 18)")
                    self.thread["LookForData"].start()
                    self.thread["LookForData"].finished.connect(self.when_look_for_data_finished)
        else:
            self.message.setText('Analysis has already begun, wait or restart Cellects.')

    def when_look_for_data_finished(self):
        if len(self.parent().po.all['folder_list']) == 0 and len(self.parent().po.data_list) == 0:
            if self.parent().po.all['im_or_vid'] == 1:
                error_message = f"There is no videos ({self.parent().po.all['extension']})in the selected folder and its sub-folders"
            else:
                error_message = f"There is no images ({self.parent().po.all['extension']}) in the selected folder and its sub-folders"
            self.message.setText(error_message)
            #     = FixedText(error_message, align='r')
            # self.message.setStyleSheet("color: rgb(230, 145, 18)")
            # self.layout.addWidget(self.message, 12, 0, 12, 3)
        else:
            self.message.setText('')
            # if len(self.parent().po.all['folder_list']) > 0:
            #     self.parent().po.update_folder_id(self.parent().po.all['first_folder_sample_number'],
            #                                       self.parent().po.all['folder_list'][0])
            # if self.instantiate:  # not self.parent().imageanalysiswindow.initialized:
            #     self.thread["GetFirstIm"].start()
            #     self.thread["GetFirstIm"].message_when_thread_finished.connect(self.first_im_read)
            # else:
            #     self.first_im_read(True)
            # if isinstance(self.parent().po.all['sample_number_per_folder'], int):
            #     self.parent().po.all['folder_number'] = 1
            if self.parent().po.all['folder_number'] > 1:
                self.parent().instantiate_widgets()
                self.parent().ifseveralfolderswindow.true_init()
                self.instantiate = False
                self.parent().change_widget(1) # IfSeveralFoldersWindow
            else:
                self.thread["GetFirstIm"].start()
                self.thread["GetFirstIm"].message_when_thread_finished.connect(self.first_im_read)

    def first_im_read(self, greyscale):
        self.parent().instantiate_widgets()
        self.parent().imageanalysiswindow.true_init()
        self.instantiate = False
        if self.parent().po.first_exp_ready_to_run:
            self.parent().imageanalysiswindow.video_tab.set_not_in_use()
        self.parent().change_widget(2) # imageanalysiswindow
        # From now on, image analysis will be available from video analysis:
        self.parent().videoanalysiswindow.image_tab.set_not_in_use()
        self.thread["GetExifDataThread"].start()

    def required_outputs_is_clicked(self):
        self.parent().last_is_first = True
        self.parent().change_widget(4)  # RequiredOutput

    def advanced_parameters_is_clicked(self):
        if self.thread["GetExifDataThread"].isRunning():
            self.message.setText("Reading data, wait or restart Cellects")
        else:
            self.parent().last_is_first = True
            self.parent().widget(5).update_csc_editing_display()
            self.parent().change_widget(5) # AdvancedParameters

    def video_analysis_window_is_clicked(self):
        if self.video_tab.state != "not_usable":
            if self.thread["LookForData"].isRunning() or self.thread["LoadDataToRunCellectsQuickly"].isRunning() or self.thread["GetFirstIm"].isRunning() or self.thread["RunAll"].isRunning():
                self.message.setText("Wait for the analysis to end, or restart Cellects")
            else:
                self.parent().last_tab = "data_specifications"
                # self.parent().po.first_exp_ready_to_run = False
                self.parent().change_widget(3) # Should be VideoAnalysisW

    def Run_all_directly_is_clicked(self):
        if not self.thread["LookForData"].isRunning() and not self.thread["RunAll"].isRunning():
            self.parent().po.motion = None
            self.message.setText("Complete analysis has started, wait until this message disappear...")
            # if not self.parent().po.first_exp_ready_to_run:
            #     self.parent().po.use_data_to_run_cellects_quickly = True
            self.thread["RunAll"].start()
            self.thread["RunAll"].message_from_thread.connect(self.display_message_from_thread)
            self.thread["RunAll"].image_from_thread.connect(self.display_image_during_thread)
            self.display_image.setVisible(True)

    def pathway_changed(self):
        if self.thread["LoadDataToRunCellectsQuickly"].isRunning():
            self.thread["LoadDataToRunCellectsQuickly"].wait()
        if os.path.isdir(Path(self.global_pathway.text())):
            self.parent().po.all['global_pathway'] = self.global_pathway.text()
            logging.info(f"Dir: {self.parent().po.all['global_pathway']}")
            os.chdir(Path(self.parent().po.all['global_pathway']))
            # 1) Put invisible widgets
            self.radical.setVisible(False)
            self.extension.setVisible(False)
            self.arena_number.setVisible(False)
            self.im_or_vid.setVisible(False)
            self.advanced_parameters.setVisible(False)
            self.required_outputs.setVisible(False)
            # self.Video_analysis_window.setVisible(False)
            self.Run_all_directly.setVisible(False)
            self.next.setVisible(False)
            # 2) Load the dict
            self.thread["LoadDataToRunCellectsQuickly"].start()
            self.thread["LoadDataToRunCellectsQuickly"].message_from_thread.connect(self.load_data_quickly_finished)
            # 3) go to another func to change, put visible and re_instantiate
        else:
            # self.Video_analysis_window.setVisible(False)
            self.Run_all_directly.setVisible(False)
            self.image_tab.set_not_usable()
            self.video_tab.set_not_usable()
            self.message.setText("Please, enter a valid path")

    def load_data_quickly_finished(self, message):
        self.image_tab.set_not_in_use()
        self.message.setText(message)
        self.radical.setVisible(True)
        self.extension.setVisible(True)
        self.arena_number.setVisible(True)
        self.im_or_vid.setVisible(True)
        self.advanced_parameters.setVisible(True)
        self.required_outputs.setVisible(True)
        self.next.setVisible(True)

        if self.parent().po.first_exp_ready_to_run:
            self.parent().po.all['folder_number'] = 1
            self.parent().instantiate_widgets(True)
            self.arena_number.setValue(self.parent().po.all['first_folder_sample_number'])
            self.im_or_vid.setCurrentIndex(self.parent().po.all['im_or_vid'])
            self.radical.setText(self.parent().po.all['radical'])
            self.extension.setText(self.parent().po.all['extension'])
            #self.shortcut_cb.setChecked(self.parent().po.all['display_shortcuts'])
            #self.display_shortcuts_checked()
            # self.Video_analysis_window.setVisible(True)
            self.Run_all_directly.setVisible(True)
            self.video_tab.set_not_in_use()


    def re_instantiate_widgets(self):
        """

        :return:
        """
        self.instantiate = True
        # Since we re-instantiate everything, image analysis will no longer be available from video analysis:
        self.parent().videoanalysiswindow.image_tab.set_not_usable()

        # self.parent().po.all['radical'] = self.radical.text()
        # self.parent().po.all['extension'] = self.extension.text()
        # self.parent().po.sample_number = int(self.arena_number.value())
        # self.parent().po.all['first_folder_sample_number'] = self.parent().po.sample_number
        # self.parent().po.all['sample_number_per_folder'] = [self.parent().po.sample_number]


        # Mettre ça en thread ? PB : conflict entre all et vars
        # if os.path.isfile('Data to run Cellects quickly.pkl'):
        #     try:
        #         with open('Data to run Cellects quickly.pkl', 'rb') as fileopen:
        #             data_to_run_cellects_quickly = pickle.load(fileopen)
        #         if 'vars' in data_to_run_cellects_quickly:
        #             self.vars = data_to_run_cellects_quickly['vars']
        #     except EOFError:
        #         print("Pickle error: could not load vars from the data folder")
        #         self.instantiate = True
        # else:
        #     self.instantiate = True
        # if self.instantiate:
        #     self.parent().po.all['radical'] = self.radical.text()
        #     self.parent().po.all['extension'] = self.extension.text()
        #     self.parent().po.sample_number = int(self.arena_number.value())
        #     self.parent().po.all['first_folder_sample_number'] = self.parent().po.sample_number
        #     self.parent().po.all['sample_number_per_folder'] = [self.parent().po.sample_number]

    def closeEvent(self, event):
        event.accept


# if __name__ == "__main__":
#     from cellects.gui.cellects import CellectsMainWidget
#     import sys
#     app = QtWidgets.QApplication([])
#     parent = CellectsMainWidget()
#     session = FirstWindow(parent, False)
#     parent.insertWidget(0, session)
#     parent.show()
#     sys.exit(app.exec())
