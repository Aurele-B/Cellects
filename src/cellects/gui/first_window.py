#!/usr/bin/env python3
"""First window of the Cellects graphical user interface (GUI).

This module implements the initial setup UI for Cellects data processing. It provides
widgets for selecting image/video inputs, configuring folder paths, arena numbers,
and prefixes/extensions. Threaded operations ensure UI responsiveness during background tasks.

Main Components
FirstWindow : QWidget subclass implementing the first GUI window with tabs and interactive widgets
"""

import os
import logging
from pathlib import Path
import numpy as np
from PySide6 import QtWidgets, QtCore
from cellects.core.cellects_threads import (
    GetFirstImThread, GetExifDataThread, RunAllThread, LookForDataThreadInFirstW, LoadDataToRunCellectsQuicklyThread)
from cellects.gui.custom_widgets import (
    MainTabsType, InsertImage, FullScreenImage, PButton, Spinbox,
    Combobox, FixedText, EditText, LineWidget)
from cellects.gui.ui_strings import FW


class FirstWindow(MainTabsType):
    """
    First window of the Cellects GUI.
    """
    def __init__(self, parent, night_mode):
        """
        Initialize the First window with a parent widget and night mode setting.

        Parameters
        ----------
        parent : QWidget
            The parent widget to which this window will be attached.
        night_mode : bool
            A boolean indicating whether the night mode should be enabled.


        Examples
        --------
        >>> from PySide6 import QtWidgets
        >>> from cellects.gui.cellects import CellectsMainWidget
        >>> from cellects.gui.first_window import FirstWindow
        >>> import sys
        >>> app = QtWidgets.QApplication([])
        >>> parent = CellectsMainWidget()
        >>> session = FirstWindow(parent, False)
        >>> session.true_init()
        >>> parent.insertWidget(0, session)
        >>> parent.show()
        >>> sys.exit(app.exec())
        """
        super().__init__(parent, night_mode)
        logging.info("Initialize first window")
        self.setParent(parent)

        self.true_init()

    def true_init(self):
        """
        Initialize the FirstWindow components and setup its layout.

        Sets up various widgets, layouts, and threading components for the Cellects GUI,
        including image or video selection, folder path input, arena number management,
        and display setup.

        Notes
        -----
        This method assumes that the parent widget has a 'po' attribute with specific settings and variables.
        """
        self.data_tab.set_in_use()
        self.image_tab.set_not_usable()
        self.video_tab.set_not_usable()
        self.thread_dict = {}
        self.thread_dict["LookForData"] = LookForDataThreadInFirstW(self.parent())
        self.thread_dict["RunAll"] = RunAllThread(self.parent())
        self.thread_dict["LoadDataToRunCellectsQuickly"] = LoadDataToRunCellectsQuicklyThread(self.parent())
        self.thread_dict["GetFirstIm"] = GetFirstImThread(self.parent())
        self.thread_dict["GetExifDataThread"] = GetExifDataThread(self.parent())
        self.instantiate: bool = True
        self.title_label = FixedText('Cellects', police=60, night_mode=self.parent().po.all['night_mode'])
        self.title_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.subtitle_line = LineWidget(size=[1, 50], night_mode=self.parent().po.all['night_mode'])

        self.Vlayout.addWidget(self.title_label)
        self.Vlayout.addWidget(self.subtitle_line)
        self.Vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))

        # 1) Set if this a Image list or Videos
        # Open the layout:
        self.second_row_widget = QtWidgets.QWidget()
        self.second_row_layout = QtWidgets.QHBoxLayout()
        self.im_or_vid_label = FixedText(FW['Image_list_or_videos']['label'], tip=FW['Image_list_or_videos']['tips'],
                                         night_mode=self.parent().po.all['night_mode'])
        # self.im_or_vid_label = FixedText('Image list or Videos:', tip="What type of data do(es) contain(s) folder(s)?", night_mode=self.parent().po.all['night_mode'])
        self.im_or_vid = Combobox(["Image list", "Videos"], self.parent().po.all['im_or_vid'], night_mode=self.parent().po.all['night_mode'])
        self.im_or_vid.setFixedWidth(150)
        self.im_or_vid.currentTextChanged.connect(self.im2vid)
        # Set their positions on layout
        self.second_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.second_row_layout.addWidget(self.im_or_vid_label)
        self.second_row_layout.addWidget(self.im_or_vid)
        self.second_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
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
            self.arena_number_label = FixedText('Arena number per folder:',
                                                tip=FW["Arena_number_per_folder"]["tips"] , #"If this number is not always the same (depending on the folder), it can be changed later",
                                                night_mode=self.parent().po.all['night_mode'])
        else:
            if self.parent().po.all['extension'] == '.JPG':
                self.parent().po.all['radical'] = ''
                self.parent().po.all['extension'] = '.mp4'
            self.arena_number_label = FixedText('Arena number per folder:',
                                                tip=FW["Arena_number_per_folder"]["tips"], #"If this number is not always the same (depending on the video), it can be changed later",
                                                night_mode=self.parent().po.all['night_mode'])
            what = 'Videos'
        self.arena_number_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.arena_number = Spinbox(min=0, max=255, val=self.parent().po.all['first_folder_sample_number'],
                                     decimals=0, night_mode=self.parent().po.all['night_mode'])
        self.arena_number.valueChanged.connect(self.re_instantiate_widgets)
        self.radical_label = FixedText(what + ' prefix:', tip=FW["Image_prefix_and_extension"]["tips"], night_mode=self.parent().po.all['night_mode'])
        self.radical_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.radical = EditText(self.parent().po.all['radical'],
                                       night_mode=self.parent().po.all['night_mode'])
        self.radical.textChanged.connect(self.re_instantiate_widgets)

        self.extension_label = FixedText(what + ' extension:', tip=FW["Image_prefix_and_extension"]["tips"], night_mode=self.parent().po.all['night_mode'])
        self.extension_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.extension = EditText(self.parent().po.all['extension'],
                                       night_mode=self.parent().po.all['night_mode'])
        self.extension.textChanged.connect(self.re_instantiate_widgets)

        # Set their positions on layout
        self.third_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.third_row_layout.addWidget(self.radical_label)
        self.third_row_layout.addWidget(self.radical)
        # self.third_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.third_row_layout.addWidget(self.extension_label)
        self.third_row_layout.addWidget(self.extension)
        self.third_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.third_row_widget.setLayout(self.third_row_layout)
        self.Vlayout.addWidget(self.third_row_widget)
        # If im_or_vid changes, adjust these 2 widgets

        # 3) Get the path to the right folder:
        # Open the layout:
        self.first_row_widget = QtWidgets.QWidget()
        self.first_row_layout = QtWidgets.QHBoxLayout()

        self.folder_label = FixedText(FW["Folder"]["label"] + ':',
                                      tip=FW["Folder"]["tips"],#"Path to the folder containing images or videos\nThe selected folder may also contain several folders of data",
                                      night_mode=self.parent().po.all['night_mode'])
        self.folder_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.global_pathway = EditText(self.parent().po.all['global_pathway'],
                                       night_mode=self.parent().po.all['night_mode'])
        self.global_pathway.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        self.global_pathway.textChanged.connect(self.pathway_changed)
        self.browse = PButton(FW["Browse"]["label"], tip=FW["Browse"]["tips"],
                              night_mode=self.parent().po.all['night_mode'])
        self.browse.clicked.connect(self.browse_is_clicked)

        # Set their positions on layout
        self.first_row_layout.addWidget(self.folder_label)
        self.first_row_layout.addWidget(self.global_pathway)
        self.first_row_layout.addWidget(self.browse)
        self.first_row_widget.setLayout(self.first_row_layout)
        self.Vlayout.addWidget(self.first_row_widget)

        self.fourth_row_widget = QtWidgets.QWidget()
        self.fourth_row_layout = QtWidgets.QHBoxLayout()
        self.fourth_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.fourth_row_layout.addWidget(self.arena_number_label)
        self.fourth_row_layout.addWidget(self.arena_number)
        self.fourth_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.fourth_row_widget.setLayout(self.fourth_row_layout)
        self.Vlayout.addWidget(self.fourth_row_widget)
        self.Vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))

        # Add the central image display widget
        self.display_image = np.zeros((self.parent().im_max_height, self.parent().im_max_width, 3), np.uint8)
        self.display_image = InsertImage(self.display_image, self.parent().im_max_height, self.parent().im_max_width)
        self.Vlayout.addWidget(self.display_image, alignment=QtCore.Qt.AlignCenter)
        self.display_image.setVisible(False)
        self.display_image.mousePressEvent = self.full_screen_display

        # 4) Create the shortcuts row
        self.shortcuts_widget = QtWidgets.QWidget()
        self.shortcuts_layout = QtWidgets.QHBoxLayout()
        # Add shortcuts: Video_analysis and Run directly
        # Shortcut 1 : Advanced Parameters
        self.advanced_parameters = PButton(FW["Advanced_parameters"]["label"], tip=FW["Advanced_parameters"]["tips"],
                                           night_mode=self.parent().po.all['night_mode'])
        self.advanced_parameters.clicked.connect(self.advanced_parameters_is_clicked)
        # Shortcut 2 : Required Outputs
        self.required_outputs = PButton(FW["Required_outputs"]["label"], tip=FW["Required_outputs"]["tips"],
                                        night_mode=self.parent().po.all['night_mode'])
        self.required_outputs.clicked.connect(self.required_outputs_is_clicked)
        # Shortcut 3 :
        self.video_tab.clicked.connect(self.video_analysis_window_is_clicked)
        # Shortcut 4 :
        self.Run_all_directly = PButton(FW["Run_all_directly"]["label"], tip=FW["Run_all_directly"]["tips"],
                                        night_mode=self.parent().po.all['night_mode'])
        self.Run_all_directly.clicked.connect(self.Run_all_directly_is_clicked)
        self.Run_all_directly.setVisible(False)

        self.shortcuts_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.shortcuts_layout.addWidget(self.advanced_parameters)
        self.shortcuts_layout.addWidget(self.required_outputs)
        self.shortcuts_layout.addWidget(self.Run_all_directly)
        self.shortcuts_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.shortcuts_widget.setLayout(self.shortcuts_layout)
        self.Vlayout.addWidget(self.shortcuts_widget)

        # 5) Open the last row layout
        self.last_row_widget = QtWidgets.QWidget()
        self.last_row_layout = QtWidgets.QHBoxLayout()

        # Message
        self.message = FixedText('', halign='r', night_mode=self.parent().po.all['night_mode'])
        self.message.setStyleSheet("color: rgb(230, 145, 18)")
        # Next button
        self.next = PButton(FW['Next']['label'], tip=FW['Next']['tips'],
                            night_mode=self.parent().po.all['night_mode'])
        self.image_tab.clicked.connect(self.next_is_clicked)
        self.next.clicked.connect(self.next_is_clicked)
        # Add widgets to the last_row_layout
        self.last_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.last_row_layout.addWidget(self.message)
        self.last_row_layout.addWidget(self.next)
        # Close the last_row_layout
        self.last_row_widget.setLayout(self.last_row_layout)
        self.Vlayout.addWidget(self.last_row_widget)
        self.setLayout(self.Vlayout)

        # Check if there is data in the saved folder
        self.pathway_changed()

    def full_screen_display(self, event):
        """
        Display an image in full screen.

        Displays the current `image_to_display` of the parent window
        in a separate full-screen window.

        Parameters
        ----------
        event : QEvent
            The event that triggers the full-screen display.

        Other Parameters
        ----------------
        popup_img : FullScreenImage
            The instance of `FullScreenImage` created to display the image.

        Notes
        -----
        The method creates a new instance of `FullScreenImage` and displays it.
        This is intended to provide a full-screen view of the image currently
        displayed in the parent window.
        """
        self.popup_img = FullScreenImage(self.parent().image_to_display, self.parent().screen_width, self.parent().screen_height)
        self.popup_img.show()

    def browse_is_clicked(self):
        """
        Handles the logic for when a "Browse" button is clicked in the interface.

        Opens a file dialog to select a directory and updates the global pathway.

        Notes
        -----
        This function assumes that `self.parent().po.all` is a dictionary with a key `'global_pathway'`.
        """
        dialog = QtWidgets.QFileDialog()
        dialog.setDirectory(str(self.parent().po.all['global_pathway']))
        self.parent().po.all['global_pathway'] = dialog.getExistingDirectory(self,
                                                                             'Select a folder containing images (/videos) or folders of data images (/videos)')
        self.global_pathway.setText(self.parent().po.all['global_pathway'])

    def im2vid(self):
        """
        Toggle between processing images or videos based on UI selection.
        """
        self.parent().po.all['im_or_vid'] = self.im_or_vid.currentIndex()
        if self.im_or_vid.currentIndex() == 0:
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

    def display_message_from_thread(self, text_from_thread: str):
        """
        Updates the message displayed in the UI with text from a thread.

        Parameters
        ----------
        text_from_thread : str
            The text to be displayed in the UI message.
        """
        self.message.setText(text_from_thread)

    def display_image_during_thread(self, dictionary: dict):
        """
        Display an image and set a message during a thread operation.

        Parameters
        ----------
        dictionary : dict
            A dictionary containing the 'message' and 'current_image'.
                The message is a string to display.
                The current_image is the image data that will be displayed.
        """
        self.message.setText(dictionary['message'])
        self.parent().image_to_display = dictionary['current_image']
        self.display_image.update_image(dictionary['current_image'])

    def next_is_clicked(self):
        """
        Handles the logic for when a "Next" button is clicked in the interface.

        Checks if certain threads are running, updates parent object's attributes,
        and starts a data-looking thread if conditions are met.
        """
        if not self.thread_dict["LookForData"].isRunning() and not self.thread_dict["RunAll"].isRunning():
            self.parent().po.all['im_or_vid'] = self.im_or_vid.currentIndex()
            self.parent().po.all['radical'] = self.radical.text()
            self.parent().po.all['extension'] = self.extension.text()
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
                    self.message.setText(f"Looking for {self.parent().po.all['radical']}***{self.parent().po.all['extension']} Wait...")
                    self.message.setStyleSheet("color: rgb(230, 145, 18)")
                    self.thread_dict["LookForData"].start()
                    self.thread_dict["LookForData"].finished.connect(self.when_look_for_data_finished)
        else:
            self.message.setText('Analysis has already begun, wait or restart Cellects.')

    def when_look_for_data_finished(self):
        """
        Check if there are any data items left in the selected folder and its sub-folders.
        Display appropriate error messages or proceed with further actions based on the data availability.

        Notes
        -----
        This function checks if there are any data items (images or videos) left in the selected folder and its sub-folders.
        If no data is found, it displays an error message. Otherwise, it proceeds with instantiating widgets or starting a thread.
        """
        if len(self.parent().po.all['folder_list']) == 0 and len(self.parent().po.data_list) == 0:
            if self.parent().po.all['im_or_vid'] == 1:
                error_message = f"There is no videos ({self.parent().po.all['extension']}) in the selected folder and its sub-folders"
            else:
                error_message = f"There is no images ({self.parent().po.all['extension']}) in the selected folder and its sub-folders"
            self.message.setText(error_message)
        else:
            self.message.setText('')
            if self.parent().po.all['folder_number'] > 1:
                self.parent().instantiate_widgets()
                self.parent().ifseveralfolderswindow.true_init()
                self.instantiate = False
                self.parent().change_widget(1) # IfSeveralFoldersWindow
            else:
                self.thread_dict["GetFirstIm"].start()
                self.thread_dict["GetFirstIm"].message_when_thread_finished.connect(self.first_im_read)

    def first_im_read(self, greyscale):
        """
        Initialize the image analysis window and prepare for reading images.

        Notes
        -----
        This function prepares the image analysis window and sets it to be ready for
        reading images. It also ensures that certain tabs are set as not in use.
        """
        self.parent().instantiate_widgets()
        self.parent().imageanalysiswindow.true_init()
        self.instantiate = False
        if self.parent().po.first_exp_ready_to_run and (self.parent().po.all["im_or_vid"] == 1 or len(self.parent().po.data_list) > 1):
            self.parent().imageanalysiswindow.video_tab.set_not_in_use()
        self.parent().change_widget(2) # imageanalysiswindow
        # From now on, image analysis will be available from video analysis:
        self.parent().videoanalysiswindow.image_tab.set_not_in_use()
        self.thread_dict["GetExifDataThread"].start()

    def required_outputs_is_clicked(self):
        """
        Handle the click event for switching to required outputs.

        This function sets the `last_is_first` attribute of the parent to True
        and changes the widget to the Required Outputs view.
        """
        self.parent().last_is_first = True
        self.parent().change_widget(4)  # RequiredOutput

    def advanced_parameters_is_clicked(self):
        """
        Handle the click event for switching to advanced parameters.

        Checks if an Exif data reading thread is running and acts accordingly.
        If not, it updates the display for advanced parameters.

        Notes
        -----
        This function updates the display for advanced parameters only if no Exif data reading thread is running.
        If a thread is active, it informs the user to wait or restart Cellects.
        """
        if self.thread_dict["GetExifDataThread"].isRunning():
            self.message.setText("Reading data, wait or restart Cellects")
        else:
            self.parent().last_is_first = True
            self.parent().widget(5).update_csc_editing_display()
            self.parent().change_widget(5) # AdvancedParameters

    def video_analysis_window_is_clicked(self):
        """
        Handles the logic for when the "Video tracking" button is clicked in the interface,
        leading to the video analysis window.

        Notes
        -----
        This function displays an error message when a thread relative to the current window is running.
        This function also save the id of the following window for later use.
        """
        if self.video_tab.state != "not_usable":
            if self.thread_dict["LookForData"].isRunning() or self.thread_dict["LoadDataToRunCellectsQuickly"].isRunning() or self.thread_dict["GetFirstIm"].isRunning() or self.thread_dict["RunAll"].isRunning():
                self.message.setText("Wait for the analysis to end, or restart Cellects")
            else:
                self.parent().last_tab = "data_specifications"
                self.parent().change_widget(3) # Should be VideoAnalysisW

    def Run_all_directly_is_clicked(self):
        """
        Run_all_directly_is_clicked

        This method initiates a complete analysis process by starting the `RunAll` thread
        after ensuring no other relevant threads are currently running.

        Notes
        -----
        - This method ensures that the `LookForData` and `RunAll` threads are not running
          before initiating a new analysis.
        - The method updates the UI to indicate that an analysis has started and displays
          progress messages.
        """
        if not self.thread_dict["LookForData"].isRunning() and not self.thread_dict["RunAll"].isRunning():
            self.parent().po.motion = None
            self.message.setText("Complete analysis has started, wait until this message disappear...")
            self.thread_dict["RunAll"].start()
            self.thread_dict["RunAll"].message_from_thread.connect(self.display_message_from_thread)
            self.thread_dict["RunAll"].image_from_thread.connect(self.display_image_during_thread)
            self.display_image.setVisible(True)

    def pathway_changed(self):
        """
        Method for handling pathway changes in the application.

        This method performs several operations when a new global pathway is set:
        1. Waits for any running thread to complete.
        2. Updates the global pathway if a valid directory is found.
        3. Changes the current working directory to the new global pathway.
        4. Hides various widgets associated with advanced options and outputs.
        5. Starts a background thread to load data quickly.
        6. If the provided pathway is invalid, it hides relevant tabs and outputs an error message.

        Notes
        -----
        This method performs actions to prepare the application for loading data from a new pathway.
        It ensures that certain widgets are hidden and starts necessary background processes.
        """
        if self.thread_dict["LoadDataToRunCellectsQuickly"].isRunning():
            self.thread_dict["LoadDataToRunCellectsQuickly"].wait()
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
            self.Run_all_directly.setVisible(False)
            self.next.setVisible(False)
            self.instantiate = True
            self.video_tab.set_not_usable()
            # 2) Load the dict
            self.thread_dict["LoadDataToRunCellectsQuickly"].start()
            self.thread_dict["LoadDataToRunCellectsQuickly"].message_from_thread.connect(self.load_data_quickly_finished)
            # 3) go to another func to change, put visible and re_instantiate
        else:
            self.Run_all_directly.setVisible(False)
            self.image_tab.set_not_usable()
            self.video_tab.set_not_usable()
            self.message.setText("Please, enter a valid path")

    def load_data_quickly_finished(self, message: str):
        """
        Set up the UI components for a new one_experiment.

        Parameters
        ----------
        message : str
            The message to be displayed on the UI component.

        Notes
        -----
        This function sets several visibility flags and values for UI components
        in preparation for starting an one_experiment.
        """
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
            self.Run_all_directly.setVisible(True)
            if self.parent().po.all["im_or_vid"] == 1 or len(self.parent().po.data_list) > 1:
                self.video_tab.set_not_in_use()


    def re_instantiate_widgets(self):
        """
        Reinstantiate the videoanalysis window from the parent of the current window.
        """
        self.instantiate = True
        # Since we re-instantiate everything, image analysis will no longer be available from video analysis:
        self.parent().videoanalysiswindow.image_tab.set_not_usable()