#!/usr/bin/env python3
"""Third main widget for Cellects GUI enabling video analysis configuration and execution.

This module implements a video tracking interface for analyzing cell movement through configurable parameters like
 arena selection, segmentation methods, and smoothing thresholds. It provides interactive controls including spinboxes,
  comboboxes, buttons for detection/post-processing, and an image display area with full-screen support. Threaded
  operations (VideoReaderThread, VideoTrackingThread) handle background processing to maintain UI responsiveness.

Main Components
VideoAnalysisWindow : QWidget subclass implementing the video analysis interface with tab navigation, parameter
controls, and thread coordination

Notes
Uses QThread for background operations to maintain UI responsiveness.
"""
import logging
import numpy as np
from PySide6 import QtWidgets, QtCore

from cellects.core.cellects_threads import VideoTrackingThread, VideoReaderThread, WriteVideoThread
from cellects.gui.custom_widgets import (
    MainTabsType, InsertImage, FullScreenImage, PButton, Spinbox,
    Combobox, Checkbox, FixedText)
from cellects.gui.ui_strings import FW, VAW


class VideoAnalysisWindow(MainTabsType):
    def __init__(self, po, parent, night_mode):
        """
        Initialize the VideoAnalysis window with a parent widget and night mode setting.

        Parameters
        ----------
        po: ProgramOrganizer
            The object containing current analysis parameters and connecting all methods of the software.
        parent : QWidget
            The parent widget to which this window will be attached.
        night_mode : bool
            A boolean indicating whether the night mode should be enabled.

        Examples
        --------
        >>> from PySide6 import QtWidgets
        >>> from cellects.gui.cellects import CellectsMainWidget
        >>> from cellects.gui.video_analysis_window import VideoAnalysisWindow
        >>> import sys
        >>> app = QtWidgets.QApplication([])
        >>> parent = CellectsMainWidget()
        >>> session = VideoAnalysisWindow(parent, False)
        >>> parent.insertWidget(0, session)
        >>> parent.show()
        >>> sys.exit(app.exec())
        """
        super().__init__(parent, night_mode)
        logging.info("Initialize VideoAnalysisWindow")
        self.setParent(parent)
        self.po = po
        self.previous_arena = 0
        self.true_init()

    def true_init(self):
        """
        Initialize the video tracking interface and set up its UI components.

        Extended Description
        --------------------
        This method initializes various tabs, threads, and UI elements for the video tracking interface. It sets up
        event handlers for tab clicks and configures layout components such as labels, buttons, spinboxes, and
        comboboxes.

        Notes
        -----
        This method assumes that the parent widget has a 'po' attribute with specific settings and variables.
        """
        self.data_tab.set_not_in_use()
        self.image_tab.set_not_usable()
        self.video_tab.set_in_use()
        self.data_tab.clicked.connect(self.data_tab_is_clicked)
        self.image_tab.clicked.connect(self.image_tab_is_clicked)
        self.thread_dict['VideoReader'] = VideoReaderThread(self.po, self.parent())
        self.thread_dict['VideoTracking'] = VideoTrackingThread(self.po, self.parent())
        curr_row_main_layout = 0
        self.Vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))#, curr_row_main_layout, 0, 1, ncol)
        curr_row_main_layout += 1

        # Open subtitle
        self.title_label = FixedText("Video tracking", police=60)
        self.title_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.Vlayout.addWidget(self.title_label)
        curr_row_main_layout += 1

        # Open central widget
        self.video_display_widget = QtWidgets.QWidget()
        self.video_display_layout = QtWidgets.QHBoxLayout()
        self.video_display_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        #   Open left widget
        self.left_options_widget = QtWidgets.QWidget()
        self.left_options_layout = QtWidgets.QVBoxLayout()
        self.left_options_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        self.specimen_activity_widget = QtWidgets.QWidget()
        self.specimen_activity_layout = QtWidgets.QHBoxLayout()
        self.specimen_activity_label = FixedText(VAW["Specimen_activity"]["label"] + ':',
                                       tip=VAW["Specimen_activity"]["tips"],
                                       night_mode=self.po.all['night_mode'])
        self.specimen_activity = Combobox(['move', 'grow', 'move and grow'],
                                       night_mode=self.po.all['night_mode'])
        self.specimen_activity.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.specimen_activity.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.specimen_activity.setFixedWidth(175)
        self.specimen_activity.setCurrentText(self.po.vars['specimen_activity'])
        self.specimen_activity.currentTextChanged.connect(self.specimen_activity_changed)
        self.specimen_activity_layout.addWidget(self.specimen_activity_label)
        self.specimen_activity_layout.addWidget(self.specimen_activity)
        self.specimen_activity_widget.setLayout(self.specimen_activity_layout)
        self.left_options_layout.addWidget(self.specimen_activity_widget)

        self.fading_widget = QtWidgets.QWidget()
        self.fading_layout = QtWidgets.QHBoxLayout()
        self.fading_label = FixedText(VAW["Fading_detection"]["label"],
                                       tip=VAW["Fading_detection"]["tips"] + ':',
                                       night_mode=self.po.all['night_mode'])
        self.fading = Spinbox(min=- 1, max=1, val=self.po.vars['fading'], decimals=2,
                               night_mode=self.po.all['night_mode'])
        self.fading.valueChanged.connect(self.fading_changed)
        self.fading_layout.addWidget(self.fading_label)
        self.fading_layout.addWidget(self.fading)
        self.fading_widget.setLayout(self.fading_layout)
        self.left_options_layout.addWidget(self.fading_widget)

        self.growth_per_frame_widget = QtWidgets.QWidget()
        self.growth_per_frame_layout = QtWidgets.QHBoxLayout()
        try:
            self.po.vars['maximal_growth_factor']
        except KeyError:
            self.po.vars['maximal_growth_factor'] = 0.02
        self.maximal_growth_factor = Spinbox(min=0, max=0.5, val=self.po.vars['maximal_growth_factor'],
                                            decimals=3, night_mode=self.po.all['night_mode'])
        self.maximal_growth_factor_label = FixedText(VAW["Maximal_growth_factor"]["label"] + ':',
                                                    tip=VAW["Maximal_growth_factor"]["tips"],
                                                    night_mode=self.po.all['night_mode'])
        self.maximal_growth_factor.valueChanged.connect(self.maximal_growth_factor_changed)
        self.growth_per_frame_layout.addWidget(self.maximal_growth_factor_label)
        self.growth_per_frame_layout.addWidget(self.maximal_growth_factor)
        self.growth_per_frame_widget.setLayout(self.growth_per_frame_layout)
        self.left_options_layout.addWidget(self.growth_per_frame_widget)


        self.select_option_label = FixedText(VAW["Segmentation_method"]["label"] + ':',
                                             tip=VAW["Segmentation_method"]["tips"],
                                             night_mode=self.po.all['night_mode'])
        self.select_option = Combobox([], night_mode=self.po.all['night_mode'])
        self.select_option_label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.select_option.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.select_option.setFixedWidth(175)
        self.select_option.addItem("1. Frame by frame")
        self.select_option.addItem("2. Dynamical threshold")
        self.select_option.addItem("3. Dynamical slope")
        self.select_option.addItem("4. Threshold and Slope")
        self.select_option.addItem("5. Threshold or Slope")
        self.select_option.setCurrentIndex(self.po.all['video_option'])
        self.select_option.currentTextChanged.connect(self.option_changed)

        # Open the choose best option row layout
        self.options_row_widget = QtWidgets.QWidget()
        self.options_row_layout = QtWidgets.QHBoxLayout()
        self.options_row_layout.addWidget(self.select_option_label)
        self.options_row_layout.addWidget(self.select_option)
        self.options_row_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.options_row_layout.setAlignment(QtCore.Qt.AlignVCenter)
        self.options_row_widget.setLayout(self.options_row_layout)
        self.left_options_layout.addWidget(self.options_row_widget)

        #   Close left widget
        self.left_options_widget.setLayout(self.left_options_layout)
        self.video_display_layout.addWidget(self.left_options_widget)


        # Add the central video display widget
        self.display_image = np.zeros((self.parent().im_max_height, self.parent().im_max_width, 3), np.uint8)
        self.display_image = InsertImage(self.display_image, self.parent().im_max_height, self.parent().im_max_width)
        self.display_image.mousePressEvent = self.full_screen_display
        self.video_display_layout.addWidget(self.display_image)


        #   Open right widget
        self.right_options_widget = QtWidgets.QWidget()
        self.right_options_layout = QtWidgets.QVBoxLayout()
        self.right_options_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        self.test_one_arena_label = FixedText('Test one arena:', tip="",
                                         night_mode=self.po.all['night_mode'])
        self.right_options_layout.addWidget(self.test_one_arena_label)
        # I/B/ Create the box
        self.one_arena_box_layout = QtWidgets.QVBoxLayout()
        self.one_arena_box_widget = QtWidgets.QWidget()
        # boxstylesheet = \
        #     ".QWidget {\n" \
        #     + "border: 1px solid black;\n" \
        #     + "border-radius: 20px;\n" \
        #     + "}"
        self.one_arena_box_widget.setObjectName("box_widget")
        self.one_arena_box_widget.setStyleSheet("""
            #box_widget {
                border: 1px solid black;
                border-radius: 20px;
            }
        """)
        # I/C/ Create widgets
        self.arena_widget = QtWidgets.QWidget()
        self.arena_layout = QtWidgets.QHBoxLayout()
        self.arena_label = FixedText(VAW["Arena_to_analyze"]["label"] + ':',
                                       tip=VAW["Arena_to_analyze"]["tips"],
                                       night_mode=self.po.all['night_mode'])
        sample_size = self.po.all['sample_number_per_folder'][0]
        if self.po.all['arena'] > sample_size:
            self.po.all['arena'] = 1

        self.arena = Spinbox(min=1, max=1000000, val=self.po.all['arena'],
                               night_mode=self.po.all['night_mode'])
        self.arena.valueChanged.connect(self.arena_changed)

        self.arena_layout.addWidget(self.arena_label)
        self.arena_layout.addWidget(self.arena)
        self.arena_widget.setLayout(self.arena_layout)
        self.one_arena_box_layout.addWidget(self.arena_widget)

        self.operation_widget = QtWidgets.QWidget()
        self.operation_layout = QtWidgets.QHBoxLayout()
        self.operation_label = FixedText(VAW["Operation"]["label"] + ':', tip=VAW["Operation"]["tips"],
                                      night_mode=self.po.all['night_mode'])
        self.operation_list = ['load', 'quick detect', 'full detect']
        self.operation = Combobox(self.operation_list,
                                       night_mode=self.po.all['night_mode'])
        self.operation.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.operation.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.operation.setFixedWidth(150)
        self.operation.setCurrentText(self.operation_list[2])
        self.operation.currentTextChanged.connect(self.operation_changed)

        self.operation_layout.addWidget(self.operation_label)
        self.operation_layout.addWidget(self.operation)
        self.operation_widget.setLayout(self.operation_layout)
        self.one_arena_box_layout.addWidget(self.operation_widget)

        self.all_options_row_widget = QtWidgets.QWidget()
        self.all_options_row_layout = QtWidgets.QHBoxLayout()
        self.compute_all_options_label = FixedText('Try the 5 segmentation methods:',
                                                   tip=VAW["Segmentation_method"]["tips"],
                                                   night_mode=self.po.all['night_mode'])
        self.compute_all_options_cb = Checkbox(self.po.all['compute_all_options'])
        self.compute_all_options_cb.setStyleSheet("QCheckBox::indicator {width: 12px;height: 12px;background-color: transparent;"
                            "border-radius: 5px;border-style: solid;border-width: 1px;"
                            "border-color: rgb(100,100,100);}"
                            "QCheckBox::indicator:checked {background-color: rgb(70,130,180);}"
                            "QCheckBox:checked, QCheckBox::indicator:checked {border-color: black black white white;}"
                            "QCheckBox:checked {background-color: transparent;}"
                            "QCheckBox:margin-left {0%}"
                            "QCheckBox:margin-right {0%}")
        self.compute_all_options_cb.stateChanged.connect(self.compute_all_options_check)
        self.all_options_row_layout.addWidget(self.compute_all_options_label)
        self.all_options_row_layout.addWidget(self.compute_all_options_cb)
        self.all_options_row_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.all_options_row_layout.setAlignment(QtCore.Qt.AlignVCenter)
        self.all_options_row_widget.setLayout(self.all_options_row_layout)
        self.one_arena_box_layout.addWidget(self.all_options_row_widget)

        self.post_analysis_widget = QtWidgets.QWidget()
        self.post_analysis_layout = QtWidgets.QHBoxLayout()
        self.run_one = PButton(VAW["Run_one"]["label"], tip=VAW["Run_one"]["tips"],
                                      night_mode=self.po.all['night_mode'])
        self.run_one.clicked.connect(self.run_one_arena_thread)

        self.read = PButton(VAW["Read"]["label"], tip=VAW["Read"]["tips"], night_mode=self.po.all['night_mode'])
        self.read.clicked.connect(self.read_is_clicked)
        self.read.setVisible(False)

        self.save_one_result = PButton(VAW["Save_one_result"]["label"], tip=VAW["Save_one_result"]["tips"],
                                       night_mode=self.po.all['night_mode'])
        self.save_one_result.clicked.connect(self.save_one_result_is_clicked)
        self.save_one_result.setVisible(False)

        self.post_analysis_layout.addWidget(self.run_one)
        self.post_analysis_layout.addWidget(self.read)
        self.post_analysis_layout.addWidget(self.save_one_result)
        self.post_analysis_widget.setLayout(self.post_analysis_layout)
        self.one_arena_box_layout.addWidget(self.post_analysis_widget)

        self.one_arena_box_widget.setLayout(self.one_arena_box_layout)
        self.right_options_layout.addWidget(self.one_arena_box_widget)

        self.right_options_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.right_options_widget.setLayout(self.right_options_layout)
        self.video_display_layout.addWidget(self.right_options_widget)
        self.video_display_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        # Close central widget
        self.video_display_widget.setLayout(self.video_display_layout)
        self.Vlayout.addWidget(self.video_display_widget)#, curr_row_main_layout, 0)
        curr_row_main_layout += 1

        # Open last options row widget
        self.last_options_widget = QtWidgets.QWidget()
        self.last_options_layout = QtWidgets.QHBoxLayout()
        self.last_options_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))

        self.advanced_parameters = PButton(FW["Advanced_parameters"]["label"], tip=FW["Advanced_parameters"]["tips"],
                                           night_mode=self.po.all['night_mode'])
        self.advanced_parameters.clicked.connect(self.advanced_parameters_is_clicked)
        self.last_options_layout.addWidget(self.advanced_parameters)

        #  Required Outputs widget
        self.required_outputs = PButton(FW["Required_outputs"]["label"], tip=FW["Required_outputs"]["tips"],
                                        night_mode=self.po.all['night_mode'])
        self.required_outputs.clicked.connect(self.required_outputs_is_clicked)
        self.last_options_layout.addWidget(self.required_outputs)

        #  Save all choices widget
        self.save_all_vars = PButton(VAW["Save_all_choices"]["label"], tip=VAW["Save_all_choices"]["tips"],
                                     night_mode=self.po.all['night_mode'])
        self.save_all_vars.clicked.connect(self.save_current_settings)
        self.last_options_layout.addWidget(self.save_all_vars)

        # Close last options widget
        self.last_options_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.last_options_widget.setLayout(self.last_options_layout)
        self.Vlayout.addWidget(self.last_options_widget)#, curr_row_main_layout, 0)
        curr_row_main_layout += 1

        self.message = QtWidgets.QLabel(self)
        self.message.setText('')
        self.message.setStyleSheet("color: rgb(230, 145, 18)")
        self.message.setAlignment(QtCore.Qt.AlignLeft)

        self.previous = PButton('Previous', night_mode=self.po.all['night_mode'])
        self.previous.clicked.connect(self.previous_is_clicked)

        self.run_all = PButton(VAW["Run_All"]["label"], tip=VAW["Run_All"]["tips"],
                               night_mode=self.po.all['night_mode'])
        self.run_all.clicked.connect(self.run_all_is_clicked)

        # Open last row widget
        self.last_row_widget = QtWidgets.QWidget()
        self.last_row_layout = QtWidgets.QHBoxLayout()
        self.last_row_layout.addWidget(self.previous)
        self.last_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.last_row_layout.addWidget(self.message)
        self.last_row_layout.addWidget(self.run_all)
        # Close last row widget
        self.last_row_widget.setLayout(self.last_row_layout)
        self.Vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))#, curr_row_main_layout, 0, 1, ncol)
        self.Vlayout.addWidget(self.last_row_widget)#, curr_row_main_layout + 1, 0)

        self.setLayout(self.Vlayout)

    def display_conditionally_visible_widgets(self):
        """
        Display Conditionally Visible Widgets
        """
        self.select_option_label.setVisible(self.po.vars["color_number"] == 2)
        self.select_option.setVisible(self.po.vars["color_number"] == 2)
        do_fading = self.po.vars['specimen_activity'] == 'move and grow'
        self.fading.setVisible(do_fading)
        self.fading_label.setVisible(do_fading)

    def full_screen_display(self, event):
        """
        Full-screen display of an image.

        This method creates a full-screen image popup and displays it. The
        full-screen image is initialized with the current image to display,
        and its size is set to match the screen dimensions.
        """
        self.popup_img = FullScreenImage(self.parent().image_to_display, self.parent().screen_width, self.parent().screen_height)
        self.popup_img.show()

    def option_changed(self):
        """
        Handles the logic for changing video option settings and logging the appropriate actions.

        This method is responsible for updating various flags and configuration variables
        based on the selected video option. It also logs informational messages regarding
        the behavior of the segmentation algorithms being enabled or disabled.

        Notes
        -----
        This function updates the parent object's configuration variables and logs messages
        based on the selected video option. The behavior changes depending on the number of
        colors detected and the specific video option chosen.
        """
        if self.thread_dict['VideoReader'].isRunning():
            self.thread_dict['VideoReader'].requestInterruption()
            self.thread_dict['VideoReader'].wait()
            self.message.setText("")
        self.po.all['video_option'] = self.select_option.currentIndex()
        self.po.vars['frame_by_frame_segmentation'] = False
        self.po.vars['do_threshold_segmentation'] = False
        self.po.vars['do_slope_segmentation'] = False

        if self.po.vars['color_number'] > 2 or self.po.all['video_option'] == 0:
            logging.info(f"This option will detect {self.po.vars['color_number']} distinct luminosity groups for each frame.")
            self.po.vars['frame_by_frame_segmentation'] = True
        else:
            self.po.vars['frame_by_frame_segmentation'] = False
            if self.po.all['video_option'] == 1:
                logging.info(f"This option will detect cell(s) using a dynamic threshold algorithm with a maximal growth factor of {self.po.vars['maximal_growth_factor']}")
                self.po.vars['do_threshold_segmentation'] = True
            elif self.po.all['video_option'] == 2:
                logging.info(f"This option will detect cell(s) using a dynamic slope algorithm with a maximal growth factor of {self.po.vars['maximal_growth_factor']}")
                self.po.vars['do_slope_segmentation'] = True
            elif self.po.all['video_option'] > 2:
                self.po.vars['do_threshold_segmentation'] = True
                self.po.vars['do_slope_segmentation'] = True
                if self.po.all['video_option'] == 3:
                    logging.info(f"This option will detect cell(s) using the dynamic threshold AND slope algorithms with a maximal growth factor of {self.po.vars['maximal_growth_factor']}")
                    self.po.vars['true_if_use_light_AND_slope_else_OR'] = True
                elif self.po.all['video_option'] == 4:
                    logging.info(f"This option will detect cell(s) using the dynamic threshold OR slope algorithms with a maximal growth factor of {self.po.vars['maximal_growth_factor']}")
                    self.po.vars['true_if_use_light_AND_slope_else_OR'] = False

    def operation_changed(self):
        """
        Update the "load_quick_full" attribute based on the current UI operation index.

        Returns
        -------
        None
            The method only mutates self.po.load_quick_full.
        """
        self.po.load_quick_full = self.operation.currentIndex()

    def data_tab_is_clicked(self):
        """
        Handles the logic for when the "Data specifications" button is clicked in the interface,
        leading to the FirstWindow.

        Notes
        -----
        This function displays an error message when a thread relative to the current window is running.
        This function also save the id of the following window for later use.
        """
        if self.thread_dict['VideoReader'].isRunning() or self.thread_dict['VideoTracking'].isRunning() or self.parent().firstwindow.thread_dict["VideoTracking"].isRunning():
            self.message.setText("Wait for the analysis to end, or restart Cellects")
        else:
            self.parent().last_tab = "data_specifications"
            self.parent().change_widget(0)  # FirstWidget

    def image_tab_is_clicked(self):
        """
        Handles the logic for when the "Image analysis" button is clicked in the interface,
        leading to the image analysis window.

        Notes
        -----
        This function displays an error message when a thread relative to the current window is running.
        This function also save the id of the following window for later use.
        """
        if self.image_tab.state != "not_usable":
            if self.thread_dict['VideoReader'].isRunning() or self.thread_dict['VideoTracking'].isRunning() or self.parent().firstwindow.thread_dict["VideoTracking"].isRunning():
                self.message.setText("Wait for the analysis to end, or restart Cellects")
            else:
                # Reset the VideoTracking thread for one arena
                if not self.thread_dict['VideoTracking'].isRunning():
                    self.po.motion = None
                self.parent().last_tab = "video_analysis"
                self.parent().change_widget(2)
                self.parent().imageanalysiswindow.advanced_mode_cb.setVisible(True)
                self.parent().imageanalysiswindow.advanced_mode_label.setVisible(True)


    def required_outputs_is_clicked(self):
        """
        Sets the required outputs flag and changes the widget to the "Required Output" window.
        """
        self.parent().last_is_first = False
        self.parent().change_widget(4)  # RequiredOutput

    def advanced_parameters_is_clicked(self):
        """
        Modifies the interface to display advanced parameters.
        """
        # Reset the VideoTracking thread for one arena
        if not self.thread_dict['VideoTracking'].isRunning():
            self.po.motion = None
        self.parent().last_is_first = False
        self.parent().widget(5).update_csc_editing_display()
        self.parent().change_widget(5)  # AdvancedParameters

    def previous_is_clicked(self):
        """
        Transition to the previous tab based on current tab history.

        This method handles the logic for navigating back through the
        application's tabs when "previous" is clicked. It updates the current
        tab to the one that was last visited, cycling through the predefined
        order of tabs.

        Notes
        -----
        This function is part of a state-machine-like navigation system that
        tracks tab history. It assumes the parent widget has methods `last_tab`
        and `change_widget` for managing the current view.
        """
        if self.parent().last_tab == "data_specifications":
            self.parent().change_widget(0)  # FirstWidget
        elif self.parent().last_tab == "image":
            self.parent().change_widget(2)  # ThirdWidget
        self.parent().last_tab = "video_analysis"

    def save_all_vars_thread(self):
        """
        Start the 'SaveAllVars' thread if it is not already running.

        This method is used to ensure that variable saving operations are performed
        in a separate thread to avoid blocking the main application.
        """
        if not self.parent().thread_dict['SaveAllVars'].isRunning():
            self.parent().thread_dict['SaveAllVars'].start()  # SaveAllVarsThreadInThirdWidget

    def save_current_settings(self):
        """
        Saves the current settings from UI components to persistent storage.

        This method captures the values of various UI components and stores
        them in a persistent data structure to ensure settings are saved across
        sessions.
        """
        self.po.vars['maximal_growth_factor'] = self.maximal_growth_factor.value()
        self.po.vars['specimen_activity'] = self.specimen_activity.currentText()
        self.po.vars['fading'] = self.fading.value()
        self.po.all['compute_all_options'] = self.compute_all_options_cb.isChecked()
        self.option_changed()
        self.save_all_vars_thread()

    def specimen_activity_changed(self):
        """
        Save the fading checkbox value to allow cases where pixels can be left by the specimen(s).
        """
        self.po.vars['specimen_activity'] = self.specimen_activity.currentText()
        do_fading = self.po.vars['specimen_activity'] == 'move and grow'
        self.fading_label.setVisible(do_fading)
        self.fading.setVisible(do_fading)

    def fading_changed(self):
        """
        Save the fading spinbox value to modify how intensity must decrease to detect a pixel left by the specimen(s).
        """
        self.po.vars['fading'] = self.fading.value()

    def maximal_growth_factor_changed(self):
        """
        Save the maximal_growth_factor spinbox value to modulate the maximal growth between two frames.
        """
        self.po.vars['maximal_growth_factor'] = self.maximal_growth_factor.value()

    def arena_changed(self):
        """
        Resets the loaded arena when its video and processing threads are not running.

        Notes
        -----
        This function is part of a larger class responsible for managing video and
        arena processing threads. It should be called when all relevant threads are not
        running to ensure the arena's state is properly reset.
        """
        if self.thread_dict['VideoTracking'].isRunning():
            self.message.setText("Wait for the analysis to end, or restart Cellects")
        else:
            if self.thread_dict['VideoReader'].isRunning():
                self.thread_dict['VideoReader'].requestInterruption()
                self.thread_dict['VideoReader'].wait()
                self.message.setText("")
            self.po.motion = None
            self.reset_general_step()
            self.po.computed_video_options = np.zeros(5, bool)
            self.po.all['arena'] = int(np.round(self.arena.value()))

    def compute_all_options_check(self):
        """
        Save the compute_all_options checkbox value to process every video segmentation algorithms during the next run.
        """
        self.po.all['compute_all_options'] = self.compute_all_options_cb.isChecked()

    def run_one_arena_thread(self):
        """
        Run the VideoTracking thread for processing one arena.

        Executes the thread to load one video, initialize analysis,
        stop any running instance of the thread, save settings, and connect
        signals for displaying messages, images, and handling completion events.

        Notes
        -----
        Ensures that the previous arena settings are cleared and connects signals
        to display messages and images during thread execution.
        """
        if self.thread_dict['VideoTracking'].isRunning():
            self.message.setText("A video tracking task is already running, wait or restart Cellects")
        else:
            if self.thread_dict['VideoReader'].isRunning():
                self.thread_dict['VideoReader'].requestInterruption()
                self.thread_dict['VideoReader'].wait()
            self.message.setText("Load the video and initialize analysis, wait...")
            self.save_current_settings()
            if self.previous_arena != self.po.all['arena']:
                self.po.motion = None
            self.po.video_task = 'one_arena'
            self.thread_dict['VideoTracking'].start()
            self.thread_dict['VideoTracking'].message_from_thread.connect(self.display_message_from_thread)
            self.thread_dict['VideoTracking'].when_loading_finished.connect(self.when_loading_thread_finished)
            self.thread_dict['VideoTracking'].when_detection_finished.connect(self.when_detection_finished)
            self.thread_dict['VideoTracking'].image_from_thread.connect(self.display_image_during_thread)

    def when_loading_thread_finished(self, save_loaded_video: bool):
        """
        Ends the loading thread process and handles post-loading actions.

        Notes
        ----------
        This method assumes that the parent object has a `po` attribute with an
        'arena' key and a `load_quick_full` attribute. It also assumes that the
        parent object has a 'thread' dictionary and a message UI component.
        """
        self.previous_arena = self.po.all['arena']
        if save_loaded_video:
            self.thread_dict['WriteVideo'] = WriteVideoThread(self.po, self.parent())
            self.thread_dict['WriteVideo'].start()
        if self.po.load_quick_full == 0:
            self.message.setText("Loading done, you can watch the video")
        self.read.setVisible(True)

    def when_detection_finished(self, message: str):
        """
        Handles the completion of video detection and updates the UI accordingly.

        When the video detection is finished, this function waits for the
        VideoReader thread to complete if it's running. It then processes the
        last frame of the video based on the configured visualization and motion
        detection settings. Finally, it updates the UI with the processed image
        and sets appropriate labels' visibility.

        Parameters
        ----------
        message : str
            The message to display upon completion of detection.
            This could be a status update or any relevant information.

        Notes
        -----
        This function assumes that the parent object has attributes `po` and
        `image_to_display`, and methods like `display_image.update_image`.
        """
        self.previous_arena = self.po.all['arena']
        if self.thread_dict['VideoReader'].isRunning():  # VideoReaderThreadInThirdWidget
            self.thread_dict['VideoReader'].wait()
        if self.po.load_quick_full > 0:
            image = self.po.motion.segmented[-1, ...]
        if self.po.motion.visu is None:
            image = self.po.motion.converted_video[-1, ...] * (1 - image)
            image = np.round(image).astype(np.uint8)
            image = np.stack((image, image, image), axis=2)
        else:
            image = np.stack((image, image, image), axis=2)
            image = self.po.motion.visu[-1, ...] * (1 - image)
        self.parent().image_to_display = image
        self.display_image.update_image(image)
        self.message.setText(message)
        self.select_option_label.setVisible(self.po.vars["color_number"] == 2)
        self.select_option.setVisible(self.po.vars["color_number"] == 2)
        self.read.setVisible(True)
        self.save_one_result.setVisible(self.po.load_quick_full == 2)

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

    def save_one_result_is_clicked(self):
        """
        Finalize one arena analysis and save the result if conditions are met.

        This function checks various conditions before starting a thread to
        finalize the analysis and save the result. It ensures that certain
        threads are not running before proceeding.
        """
        if self.thread_dict['VideoTracking'].isRunning():
            self.message.setText("A video tracking task is already running, wait or restart Cellects")
        else:
            if self.po.motion is None or self.po.load_quick_full < 2:
                self.message.setText("Run Post processing first")
            else:
                if self.thread_dict['VideoReader'].isRunning():
                    self.thread_dict['VideoReader'].requestInterruption()
                    self.thread_dict['VideoReader'].wait()
                self.message.setText(f"Arena {self.po.all['arena']}: Finalize analysis and save, wait...")
                self.po.video_task = 'change_one_arena_result'
                self.compute_all_options_cb.setChecked(False)
                self.thread_dict['VideoTracking'].start()
                self.thread_dict['VideoTracking'].message_from_thread.connect(self.display_message_from_thread)
                self.message.setText("Complete analysis + change that result")

    def read_is_clicked(self):
        """
        Read a video corresponding to a numbered arena (numbered using natural sorting) in the image

        This function checks if the detection has been run and if the video reader or analysis thread is running.
        If both threads are idle, it starts the video reading process. Otherwise, it updates the message accordingly.
        """
        if self.po.motion is None or self.po.motion.segmented is None:
            self.message.setText("Run detection first")
        else:
            if self.thread_dict['VideoTracking'].isRunning():
                self.message.setText("A video tracking task is running, wait or restart Cellects")
            else:
                if self.thread_dict['VideoReader'].isRunning():
                    self.thread_dict['VideoReader'].requestInterruption()
                    self.thread_dict['VideoReader'].wait()
                self.thread_dict['VideoReader'].start()
                self.thread_dict['VideoReader'].message_from_thread.connect(self.display_image_during_thread)

    def run_all_is_clicked(self):
        """
        Handle the click event to start the complete analysis.

        This function checks if any threads are running and starts the
        'VideoTracking' thread for all arenas. It also updates
        various attributes and messages related to the analysis process.

        Notes
        -----
        This function will only start the analysis if no other threads
        are running. It updates several attributes of the parent object.
        """
        if self.thread_dict['VideoTracking'].isRunning():
            self.message.setText("A video tracking task is already running, wait or restart Cellects")
        else:
            if self.thread_dict['VideoReader'].isRunning():
                self.thread_dict['VideoReader'].requestInterruption()
                self.thread_dict['VideoReader'].wait()
            if self.parent().firstwindow.thread_dict["VideoTracking"].isRunning():
                self.message.setText('Analysis has already begun in the first window.')
            else:
                self.save_one_result.setVisible(False)
                self.read.setVisible(False)
                self.save_current_settings()
                self.po.motion = None
                self.po.converted_video = None
                self.po.converted_video2 = None
                self.po.visu = None
                self.message.setText("Complete analysis has started, wait...")
                self.po.video_task = 'all'
                self.compute_all_options_cb.setChecked(False)
                self.thread_dict['VideoTracking'].start()
                self.thread_dict['VideoTracking'].message_from_thread.connect(self.display_message_from_thread)
                self.thread_dict['VideoTracking'].image_from_thread.connect(self.display_image_during_thread)

    def display_message_from_thread(self, text_from_thread: str):
        """
        Updates the message displayed in the UI with text from a thread.

        Parameters
        ----------
        text_from_thread : str
            The text to be displayed in the UI message.
        """
        self.message.setText(text_from_thread)

