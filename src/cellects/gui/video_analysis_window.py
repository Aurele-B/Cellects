#!/usr/bin/env python3
"""Third main widget for Cellects GUI enabling video analysis configuration and execution.

This module implements a video tracking interface for analyzing cell movement through configurable parameters like
 arena selection, segmentation methods, and smoothing thresholds. It provides interactive controls including spinboxes,
  comboboxes, buttons for detection/post-processing, and an image display area with full-screen support. Threaded
  operations (VideoReaderThread, OneArenaThread) handle background processing to maintain UI responsiveness.

Main Components
VideoAnalysisWindow : QWidget subclass implementing the video analysis interface with tab navigation, parameter
controls, and thread coordination

Notes
Uses QThread for background operations to maintain UI responsiveness.
"""
import logging
import numpy as np
from PySide6 import QtWidgets, QtCore

from cellects.core.cellects_threads import (
    RunAllThread, OneArenaThread, VideoReaderThread, ChangeOneRepResultThread,
    WriteVideoThread)
from cellects.gui.custom_widgets import (
    MainTabsType, InsertImage, FullScreenImage, PButton, Spinbox,
    Combobox, Checkbox, FixedText)
from cellects.gui.ui_strings import FW, VAW


class VideoAnalysisWindow(MainTabsType):
    def __init__(self, parent, night_mode):
        """
        Initialize the VideoAnalysis window with a parent widget and night mode setting.

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
        >>> from cellects.gui.video_analysis_window import VideoAnalysisWindow
        >>> import sys
        >>> app = QtWidgets.QApplication([])
        >>> parent = CellectsMainWidget()
        >>> session = VideoAnalysisWindow(parent, False)
        >>> session.true_init()
        >>> parent.insertWidget(0, session)
        >>> parent.show()
        >>> sys.exit(app.exec())
        """
        super().__init__(parent, night_mode)
        logging.info("Initialize VideoAnalysisWindow")
        self.setParent(parent)
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
        self.thread_dict = {}
        self.thread_dict['VideoReader'] = VideoReaderThread(self.parent())
        self.thread_dict['OneArena'] = OneArenaThread(self.parent())
        self.thread_dict['ChangeOneRepResult'] = ChangeOneRepResultThread(self.parent())
        self.thread_dict['RunAll'] = RunAllThread(self.parent())
        self.previous_arena = 0
        curr_row_main_layout = 0
        ncol = 1
        self.Vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))#, curr_row_main_layout, 0, 1, ncol)
        curr_row_main_layout += 1

        # Open subtitle
        self.general_step_widget = QtWidgets.QWidget()
        self.general_step_layout = QtWidgets.QHBoxLayout()
        self.current_step = 0
        self.general_step_label = FixedText('Step 1: Tune parameters to improve Detection', night_mode=self.parent().po.all['night_mode'])
        self.general_step_button = PButton('Done', night_mode=self.parent().po.all['night_mode'])
        self.general_step_button.clicked.connect(self.step_done_is_clicked)
        self.general_step_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.general_step_layout.addWidget(self.general_step_label)
        self.general_step_layout.addWidget(self.general_step_button)
        self.general_step_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))

        self.general_step_widget.setLayout(self.general_step_layout)
        self.Vlayout.addWidget(self.general_step_widget)#, curr_row_main_layout, 0, 1, ncol)
        curr_row_main_layout += 1

        # Open central widget
        self.video_display_widget = QtWidgets.QWidget()
        self.video_display_layout = QtWidgets.QHBoxLayout()
        self.video_display_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        #   Open left widget
        self.left_options_widget = QtWidgets.QWidget()
        self.left_options_layout = QtWidgets.QVBoxLayout()
        self.left_options_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        self.arena_widget = QtWidgets.QWidget()
        self.arena_layout = QtWidgets.QHBoxLayout()
        self.arena_label = FixedText(VAW["Arena_to_analyze"]["label"] + ':',
                                       tip=VAW["Arena_to_analyze"]["tips"],
                                       night_mode=self.parent().po.all['night_mode'])
        sample_size = self.parent().po.all['sample_number_per_folder'][0]
        if self.parent().po.all['arena'] > sample_size:
            self.parent().po.all['arena'] = 1

        self.arena = Spinbox(min=1, max=1000000, val=self.parent().po.all['arena'],
                               night_mode=self.parent().po.all['night_mode'])
        self.arena.valueChanged.connect(self.arena_changed)

        self.arena_layout.addWidget(self.arena_label)
        self.arena_layout.addWidget(self.arena)
        self.arena_widget.setLayout(self.arena_layout)
        self.left_options_layout.addWidget(self.arena_widget)


        self.growth_per_frame_widget = QtWidgets.QWidget()
        self.growth_per_frame_layout = QtWidgets.QHBoxLayout()
        try:
            self.parent().po.vars['maximal_growth_factor']
        except KeyError:
            self.parent().po.vars['maximal_growth_factor'] = 0.02
            self.parent().po.vars['repeat_video_smoothing'] = self.parent().po.vars['iterate_smoothing']
        self.maximal_growth_factor = Spinbox(min=0, max=0.5, val=self.parent().po.vars['maximal_growth_factor'],
                                            decimals=3, night_mode=self.parent().po.all['night_mode'])
        self.maximal_growth_factor_label = FixedText(VAW["Maximal_growth_factor"]["label"] + ':',
                                                    tip=VAW["Maximal_growth_factor"]["tips"],
                                                    night_mode=self.parent().po.all['night_mode'])
        self.maximal_growth_factor.valueChanged.connect(self.maximal_growth_factor_changed)
        self.growth_per_frame_layout.addWidget(self.maximal_growth_factor_label)
        self.growth_per_frame_layout.addWidget(self.maximal_growth_factor)
        self.growth_per_frame_widget.setLayout(self.growth_per_frame_layout)
        self.left_options_layout.addWidget(self.growth_per_frame_widget)

        self.iterate_widget = QtWidgets.QWidget()
        self.iterate_layout = QtWidgets.QHBoxLayout()
        self.repeat_video_smoothing = Spinbox(min=0, max=10, val=self.parent().po.vars['repeat_video_smoothing'],
                                         night_mode=self.parent().po.all['night_mode'])
        self.repeat_video_smoothing_label = FixedText(VAW["Temporal_smoothing"]["label"] + ':',
                                                 tip=VAW["Temporal_smoothing"]["tips"],
                                                 night_mode=self.parent().po.all['night_mode'])
        self.repeat_video_smoothing.valueChanged.connect(self.repeat_video_smoothing_changed)
        self.iterate_layout.addWidget(self.repeat_video_smoothing_label)
        self.iterate_layout.addWidget(self.repeat_video_smoothing)
        self.iterate_widget.setLayout(self.iterate_layout)
        self.left_options_layout.addWidget(self.iterate_widget)


        self.select_option_label = FixedText(VAW["Segmentation_method"]["label"] + ':',
                                             tip=VAW["Segmentation_method"]["tips"],
                                             night_mode=self.parent().po.all['night_mode'])
        self.select_option = Combobox([], night_mode=self.parent().po.all['night_mode'])
        self.select_option_label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.select_option.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.select_option.setFixedWidth(175)
        self.select_option.addItem("1. Frame by frame")
        self.select_option.addItem("2. Dynamical threshold")
        self.select_option.addItem("3. Dynamical slope")
        self.select_option.addItem("4. Threshold and Slope")
        self.select_option.addItem("5. Threshold or Slope")
        self.select_option.setCurrentIndex(self.parent().po.all['video_option'])
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

        self.compute_all_options_label = FixedText('Compute all options',
                                                   tip=VAW["Segmentation_method"]["tips"],
                                                   night_mode=self.parent().po.all['night_mode'])
        self.compute_all_options_cb = Checkbox(self.parent().po.all['compute_all_options'])
        self.compute_all_options_cb.setStyleSheet("QCheckBox::indicator {width: 12px;height: 12px;background-color: transparent;"
                            "border-radius: 5px;border-style: solid;border-width: 1px;"
                            "border-color: rgb(100,100,100);}"
                            "QCheckBox::indicator:checked {background-color: rgb(70,130,180);}"
                            "QCheckBox:checked, QCheckBox::indicator:checked {border-color: black black white white;}"
                            "QCheckBox:checked {background-color: transparent;}"
                            "QCheckBox:margin-left {0%}"
                            "QCheckBox:margin-right {0%}")
        self.compute_all_options_cb.stateChanged.connect(self.compute_all_options_check)
        self.all_options_row_widget = QtWidgets.QWidget()
        self.all_options_row_layout = QtWidgets.QHBoxLayout()
        self.all_options_row_layout.addWidget(self.compute_all_options_cb)
        self.all_options_row_layout.addWidget(self.compute_all_options_label)
        self.all_options_row_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.all_options_row_layout.setAlignment(QtCore.Qt.AlignVCenter)
        self.all_options_row_widget.setLayout(self.all_options_row_layout)
        self.right_options_layout.addWidget(self.all_options_row_widget)

        self.load_one_arena = PButton(VAW["Load_one_arena"]["label"], tip=VAW["Load_one_arena"]["tips"],
                                      night_mode=self.parent().po.all['night_mode'])
        self.load_one_arena.clicked.connect(self.load_one_arena_is_clicked)
        self.detection = PButton(VAW["Detection"]["label"], tip=VAW["Detection"]["tips"],
                                 night_mode=self.parent().po.all['night_mode'])
        self.detection.clicked.connect(self.detection_is_clicked)
        self.read = PButton(VAW["Read"]["label"], tip=VAW["Read"]["tips"], night_mode=self.parent().po.all['night_mode'])
        self.read.clicked.connect(self.read_is_clicked)
        self.read.setVisible(False)
        self.right_options_layout.addWidget(self.load_one_arena, alignment=QtCore.Qt.AlignCenter)
        self.right_options_layout.addWidget(self.detection, alignment=QtCore.Qt.AlignCenter)
        self.right_options_layout.addWidget(self.read, alignment=QtCore.Qt.AlignCenter)


        self.right_options_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.right_options_widget.setLayout(self.right_options_layout)
        self.video_display_layout.addWidget(self.right_options_widget)
        self.video_display_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        # Close central widget
        self.video_display_widget.setLayout(self.video_display_layout)
        self.Vlayout.addWidget(self.video_display_widget)#, curr_row_main_layout, 0)
        curr_row_main_layout += 1

        # Open Second step row
        self.second_step_widget = QtWidgets.QWidget()
        self.second_step_layout = QtWidgets.QHBoxLayout()
        self.second_step_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.second_step_widget.setVisible(False)

        self.fading_widget = QtWidgets.QWidget()
        self.fading_layout = QtWidgets.QHBoxLayout()
        self.do_fading = Checkbox(self.parent().po.vars['do_fading'])
        self.do_fading.setStyleSheet("QCheckBox::indicator {width: 12px;height: 12px;background-color: transparent;"
                            "border-radius: 5px;border-style: solid;border-width: 1px;"
                            "border-color: rgb(100,100,100);}"
                            "QCheckBox::indicator:checked {background-color: rgb(70,130,180);}"
                            "QCheckBox:checked, QCheckBox::indicator:checked {border-color: black black white white;}"
                            "QCheckBox:checked {background-color: transparent;}"
                            "QCheckBox:margin-left {0%}"
                            "QCheckBox:margin-right {0%}")
        self.do_fading.stateChanged.connect(self.do_fading_check)
        self.fading = Spinbox(min=- 1, max=1, val=self.parent().po.vars['fading'], decimals=2,
                               night_mode=self.parent().po.all['night_mode'])
        self.fading_label = FixedText(VAW["Fading_detection"]["label"],
                                       tip=VAW["Fading_detection"]["tips"],
                                       night_mode=self.parent().po.all['night_mode'])
        self.fading.valueChanged.connect(self.fading_changed)
        self.fading_layout.addWidget(self.do_fading)
        self.fading_layout.addWidget(self.fading_label)
        self.fading_layout.addWidget(self.fading)
        self.fading_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.fading_widget.setLayout(self.fading_layout)
        self.second_step_layout.addWidget(self.fading_widget)

        self.post_processing = PButton(VAW["Post_processing"]["label"], tip=VAW["Post_processing"]["tips"],
                                       night_mode=self.parent().po.all['night_mode'])
        self.post_processing.clicked.connect(self.post_processing_is_clicked)
        self.second_step_layout.addWidget(self.post_processing)

        self.save_one_result = PButton(VAW["Save_one_result"]["label"], tip=VAW["Save_one_result"]["tips"],
                                       night_mode=self.parent().po.all['night_mode'])
        self.save_one_result.clicked.connect(self.save_one_result_is_clicked)
        self.second_step_layout.addWidget(self.save_one_result)

        # Close Second step row
        self.second_step_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.second_step_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.second_step_widget.setLayout(self.second_step_layout)
        self.Vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))#, curr_row_main_layout, 0, 1, ncol)
        curr_row_main_layout += 1
        self.Vlayout.addWidget(self.second_step_widget)#, curr_row_main_layout, 0)
        curr_row_main_layout += 1

        # Open last options row widget
        self.last_options_widget = QtWidgets.QWidget()
        self.last_options_layout = QtWidgets.QHBoxLayout()
        self.last_options_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))

        self.advanced_parameters = PButton(FW["Advanced_parameters"]["label"], tip=FW["Advanced_parameters"]["tips"],
                                           night_mode=self.parent().po.all['night_mode'])
        self.advanced_parameters.clicked.connect(self.advanced_parameters_is_clicked)
        self.last_options_layout.addWidget(self.advanced_parameters)

        #  Required Outputs widget
        self.required_outputs = PButton(FW["Required_outputs"]["label"], tip=FW["Required_outputs"]["tips"],
                                        night_mode=self.parent().po.all['night_mode'])
        self.required_outputs.clicked.connect(self.required_outputs_is_clicked)
        self.last_options_layout.addWidget(self.required_outputs)

        #  Save all choices widget
        self.save_all_vars = PButton(VAW["Save_all_choices"]["label"], tip=VAW["Save_all_choices"]["tips"],
                                     night_mode=self.parent().po.all['night_mode'])
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

        self.previous = PButton('Previous', night_mode=self.parent().po.all['night_mode'])
        self.previous.clicked.connect(self.previous_is_clicked)

        self.run_all = PButton(VAW["Run_All"]["label"], tip=VAW["Run_All"]["tips"],
                               night_mode=self.parent().po.all['night_mode'])
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
        self.select_option_label.setVisible(self.parent().po.vars["color_number"] == 2)
        self.select_option.setVisible(self.parent().po.vars["color_number"] == 2)
        self.fading.setVisible(self.parent().po.vars['do_fading'])

    def step_done_is_clicked(self):
        """
        Step the analysis progress when 'Done' button is clicked.

        Increments the current step and updates the UI accordingly based on the
        new step value. Updates labels, tooltips, and visibility of widgets.

        Notes
        -----
        This method is automatically called when the 'Done' button is clicked.
        It updates the GUI elements to reflect progress in a multi-step
        analysis process.
        """
        self.current_step += 1
        if self.current_step == 1:
            self.general_step_label.setText('Step 2: Tune fading and advanced parameters to improve Post processing')
            self.general_step_label.setToolTip('Post processing is slower than Detection.\nIt improves detection with the following optional algorithms:\n - Fading detection\n - Correct errors around initial shape\n - Organism internal oscillation period\n - Connect distant shape\n - Appearing cell selection')
            self.second_step_widget.setVisible(True)
            self.fading.setVisible(self.parent().po.vars['do_fading'])
            self.save_one_result.setVisible(False)
        elif self.current_step == 2:
            self.general_step_label.setText('Step 3: Run the full analysis or save the result of one arena.')
            self.general_step_label.setToolTip('Once all settings are correct for a arena, click on "Run All" to start the full analysis.\nIf the detection is unsatisfactory for a arena, you can repeat the detection for this\narena and save the results by clicking "Save One Result".\nRepeat the process for as many arenas as necessary.')
            self.save_one_result.setVisible(True)
            self.general_step_button.setVisible(False)

    def reset_general_step(self):
        """
        Reset the general step counter and update UI labels.
        """
        self.current_step = 0
        self.general_step_label.setText('Step 1: Tune parameters to improve Detection')
        self.general_step_label.setToolTip('Detection uses only the visible parameters and those\npreviously determined on the first or last image.')
        self.general_step_button.setVisible(True)
        self.second_step_widget.setVisible(False)

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
        self.parent().po.all['video_option'] = self.select_option.currentIndex()
        self.parent().po.vars['frame_by_frame_segmentation'] = False
        self.parent().po.vars['do_threshold_segmentation'] = False
        self.parent().po.vars['do_slope_segmentation'] = False

        if self.parent().po.vars['color_number'] > 2 or self.parent().po.all['video_option'] == 0:
            logging.info(f"This option will detect {self.parent().po.vars['color_number']} distinct luminosity groups for each frame.")
            self.parent().po.vars['frame_by_frame_segmentation'] = True
        else:
            self.parent().po.vars['frame_by_frame_segmentation'] = False
            if self.parent().po.all['video_option'] == 1:
                logging.info(f"This option will detect cell(s) using a dynamic threshold algorithm with a maximal growth factor of {self.parent().po.vars['maximal_growth_factor']}")
                self.parent().po.vars['do_threshold_segmentation'] = True
            elif self.parent().po.all['video_option'] == 2:
                logging.info(f"This option will detect cell(s) using a dynamic slope algorithm with a maximal growth factor of {self.parent().po.vars['maximal_growth_factor']}")
                self.parent().po.vars['do_slope_segmentation'] = True
            elif self.parent().po.all['video_option'] > 2:
                self.parent().po.vars['do_threshold_segmentation'] = True
                self.parent().po.vars['do_slope_segmentation'] = True
                if self.parent().po.all['video_option'] == 3:
                    logging.info(f"This option will detect cell(s) using the dynamic threshold AND slope algorithms with a maximal growth factor of {self.parent().po.vars['maximal_growth_factor']}")
                    self.parent().po.vars['true_if_use_light_AND_slope_else_OR'] = True
                elif self.parent().po.all['video_option'] == 4:
                    logging.info(f"This option will detect cell(s) using the dynamic threshold OR slope algorithms with a maximal growth factor of {self.parent().po.vars['maximal_growth_factor']}")
                    self.parent().po.vars['true_if_use_light_AND_slope_else_OR'] = False

    def data_tab_is_clicked(self):
        """
        Handles the logic for when the "Data specifications" button is clicked in the interface,
        leading to the FirstWindow.

        Notes
        -----
        This function displays an error message when a thread relative to the current window is running.
        This function also save the id of the following window for later use.
        """
        if self.thread_dict['VideoReader'].isRunning() or self.thread_dict['OneArena'].isRunning() or self.thread_dict['ChangeOneRepResult'].isRunning() or self.parent().firstwindow.thread_dict["RunAll"].isRunning():
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
            if self.thread_dict['VideoReader'].isRunning() or self.thread_dict['OneArena'].isRunning() or self.thread_dict[
                'ChangeOneRepResult'].isRunning() or self.parent().firstwindow.thread_dict["RunAll"].isRunning():
                self.message.setText("Wait for the analysis to end, or restart Cellects")
            else:
                self.parent().last_tab = "video_analysis"
                self.parent().change_widget(2)


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
        elif self.parent().last_tab == "image_analysis":
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
        self.parent().po.vars['maximal_growth_factor'] = self.maximal_growth_factor.value()
        self.parent().po.vars['repeat_video_smoothing'] = int(np.round(self.repeat_video_smoothing.value()))
        self.parent().po.vars['do_fading'] = self.do_fading.isChecked()
        self.parent().po.vars['fading'] = self.fading.value()
        self.parent().po.all['compute_all_options'] = self.compute_all_options_cb.isChecked()
        self.option_changed()
        self.save_all_vars_thread()

    def repeat_video_smoothing_changed(self):
        """
        Save the repeat_video_smoothing spinbox value to set how many times the pixel intensity dynamics will be
        smoothed.
        """
        self.parent().po.vars['repeat_video_smoothing'] = int(np.round(self.repeat_video_smoothing.value()))

    def do_fading_check(self):
        """
        Save the fading checkbox value to allow cases where pixels can be left by the specimen(s).
        """
        self.parent().po.vars['do_fading'] = self.do_fading.isChecked()
        self.fading.setVisible(self.parent().po.vars['do_fading'])

    def fading_changed(self):
        """
        Save the fading spinbox value to modify how intensity must decrease to detect a pixel left by the specimen(s).
        """
        self.parent().po.vars['fading'] = self.fading.value()

    def maximal_growth_factor_changed(self):
        """
        Save the maximal_growth_factor spinbox value to modulate the maximal growth between two frames.
        """
        self.parent().po.vars['maximal_growth_factor'] = self.maximal_growth_factor.value()

    def arena_changed(self):
        """
        Resets the loaded arena when its video and processing threads are not running.

        Notes
        -----
        This function is part of a larger class responsible for managing video and
        arena processing threads. It should be called when all relevant threads are not
        running to ensure the arena's state is properly reset.
        """
        if not self.thread_dict['VideoReader'].isRunning() and not self.thread_dict['OneArena'].isRunning() and not self.thread_dict['ChangeOneRepResult'].isRunning():
            self.parent().po.motion = None
            self.reset_general_step()
            self.parent().po.computed_video_options = np.zeros(5, bool)
            self.parent().po.all['arena'] = int(np.round(self.arena.value()))

    def load_one_arena_is_clicked(self):
        """
        Load one arena if clicked.

        Resets the general step, sets `load_quick_full` to 0, and runs the arena in a separate thread.
        """
        self.reset_general_step()
        self.parent().po.load_quick_full = 0
        self.run_one_arena_thread()

    def compute_all_options_check(self):
        """
        Save the compute_all_options checkbox value to process every video segmentation algorithms during the next run.
        """
        self.parent().po.all['compute_all_options'] = self.compute_all_options_cb.isChecked()

    def detection_is_clicked(self):
        """
        Trigger detection when a button is clicked.

        This method handles the logic when the user clicks the "detection" button.
        It resets certain states, sets a flag for quick full processing,
        and starts a thread to run the detection in one arena.

        Notes
        -----
        This method is part of a larger state machine for handling user interactions.
        It assumes that the parent object has a `po` attribute with a `load_quick_full`
        flag and a method to run an arena thread.
        """
        self.reset_general_step()
        self.parent().po.load_quick_full = 1
        self.run_one_arena_thread()

    def post_processing_is_clicked(self):
        """
        Trigger post-processing when a button is clicked.

        Extended Description
        -------------------
        This function updates the parent object's load_quick_full attribute,
        logs a specific variable value, and runs an arena thread.
        """
        self.parent().po.load_quick_full = 2
        logging.info(self.parent().po.vars['maximal_growth_factor'])
        self.run_one_arena_thread()

    def run_one_arena_thread(self):
        """
        Run the OneArena thread for processing.

        Executes the OneArena thread to load video, initialize analysis,
        stop any running instance of the thread, save settings, and connect
        signals for displaying messages, images, and handling completion events.

        Notes
        -----
        Ensures that the previous arena settings are cleared and connects signals
        to display messages and images during thread execution.
        """
        if self.thread_dict['OneArena']._isRunning:
            self.thread_dict['OneArena'].stop()
        self.save_current_settings()
        if self.previous_arena != self.parent().po.all['arena']:
            self.parent().po.motion = None
        self.message.setText("Load the video and initialize analysis, wait...")
        self.thread_dict['OneArena'].start()  # OneArenaThreadInThirdWidget
        self.thread_dict['OneArena'].message_from_thread_starting.connect(self.display_message_from_thread)
        self.thread_dict['OneArena'].when_loading_finished.connect(self.when_loading_thread_finished)
        self.thread_dict['OneArena'].when_detection_finished.connect(self.when_detection_finished)
        self.thread_dict['OneArena'].image_from_thread.connect(self.display_image_during_thread)

    def when_loading_thread_finished(self, save_loaded_video: bool):
        """
        Ends the loading thread process and handles post-loading actions.

        Notes
        ----------
        This method assumes that the parent object has a `po` attribute with an
        'arena' key and a `load_quick_full` attribute. It also assumes that the
        parent object has a 'thread' dictionary and a message UI component.
        """
        self.previous_arena = self.parent().po.all['arena']
        if save_loaded_video:
            self.thread_dict['WriteVideo'] = WriteVideoThread(self.parent())
            self.thread_dict['WriteVideo'].start()
        if self.parent().po.load_quick_full == 0:
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
        self.previous_arena = self.parent().po.all['arena']
        if self.thread_dict['VideoReader'].isRunning():  # VideoReaderThreadInThirdWidget
            self.thread_dict['VideoReader'].wait()
        if self.parent().po.load_quick_full > 0:
            image = self.parent().po.motion.segmented[-1, ...]
        if self.parent().po.motion.visu is None:
            image = self.parent().po.motion.converted_video[-1, ...] * (1 - image)
            image = np.round(image).astype(np.uint8)
            image = np.stack((image, image, image), axis=2)
        else:
            image = np.stack((image, image, image), axis=2)
            image = self.parent().po.motion.visu[-1, ...] * (1 - image)
        self.parent().image_to_display = image
        self.display_image.update_image(image)
        self.message.setText(message)
        self.select_option_label.setVisible(self.parent().po.vars["color_number"] == 2)
        self.select_option.setVisible(self.parent().po.vars["color_number"] == 2)
        self.read.setVisible(True)

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
        if self.parent().po.motion is not None:
            if self.parent().po.load_quick_full == 2:
                if not self.thread_dict['OneArena'].isRunning() and not self.thread_dict['ChangeOneRepResult'].isRunning():
                    self.message.setText(f"Arena {self.parent().po.all['arena']}: Finalize analysis and save, wait...")
                    self.thread_dict['ChangeOneRepResult'].start()  # ChangeOneRepResultThreadInThirdWidget
                    self.thread_dict['ChangeOneRepResult'].message_from_thread.connect(self.display_message_from_thread)
                    self.message.setText("Complete analysis + change that result")
                else:
                    self.message.setText("Wait for the analysis to end")
            else:
                self.message.setText("Run Post processing first")
        else:
            self.message.setText("Run Post processing first")

    def read_is_clicked(self):
        """
        Read a video corresponding to a numbered arena (numbered using natural sorting) in the image

        This function checks if the detection has been run and if the video reader or analysis thread is running.
        If both threads are idle, it starts the video reading process. Otherwise, it updates the message accordingly.
        """
        if self.parent().po.motion is not None:
            if self.parent().po.motion.segmented is not None:
                if not self.thread_dict['OneArena'].isRunning() and not self.thread_dict['VideoReader'].isRunning():
                    self.thread_dict['VideoReader'].start()  # VideoReaderThreadInThirdWidget
                    self.thread_dict['VideoReader'].message_from_thread.connect(self.display_image_during_thread)
                else:
                    self.message.setText("Wait for the analysis to end")
            else:
                self.message.setText("Run detection first")
        else:
            self.message.setText("Run detection first")

    def run_all_is_clicked(self):
        """
        Handle the click event to start the complete analysis.

        This function checks if any threads are running and starts the
        'RunAll' thread if none of them are active. It also updates
        various attributes and messages related to the analysis process.

        Notes
        -----
        This function will only start the analysis if no other threads
        are running. It updates several attributes of the parent object.
        """
        if self.thread_dict['OneArena'].isRunning() or self.thread_dict['ChangeOneRepResult'].isRunning():
            self.message.setText("Wait for the current analysis to end")
        else:
            if self.thread_dict['VideoReader'].isRunning():
                self.thread_dict['VideoReader'].wait()
            if self.parent().firstwindow.thread_dict["RunAll"].isRunning():
                self.message.setText('Analysis has already begun in the first window.')
            else:
                if not self.thread_dict['RunAll'].isRunning():
                    self.save_current_settings()
                    self.parent().po.motion = None
                    self.parent().po.converted_video = None
                    self.parent().po.converted_video2 = None
                    self.parent().po.visu = None
                    self.message.setText("Complete analysis has started, wait...")
                    self.thread_dict['RunAll'].start()  # RunAllThread
                    self.thread_dict['RunAll'].message_from_thread.connect(self.display_message_from_thread)
                    self.thread_dict['RunAll'].image_from_thread.connect(self.display_image_during_thread)

    def display_message_from_thread(self, text_from_thread: str):
        """
        Updates the message displayed in the UI with text from a thread.

        Parameters
        ----------
        text_from_thread : str
            The text to be displayed in the UI message.
        """
        self.message.setText(text_from_thread)

