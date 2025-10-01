#!/usr/bin/env python3
"""
This is the third main widget of the GUI of Cellects
It process the video analysis computations by running threads connected to the program_organizer,
especially, most computation are then processed by the MotionAnalysis class
"""
import logging
import numpy as np
import cv2
from PySide6 import QtWidgets, QtCore

from cellects.core.cellects_threads import (
    RunAllThread, OneArenaThread, VideoReaderThread, ChangeOneRepResultThread,
    WriteVideoThread)
from cellects.gui.custom_widgets import (
    MainTabsType, InsertImage, FullScreenImage, PButton, Spinbox,
    Combobox, Checkbox, FixedText)


class VideoAnalysisWindow(MainTabsType):
    def __init__(self, parent, night_mode):
        super().__init__(parent, night_mode)
        logging.info("Initialize VideoAnalysisWindow")
        self.setParent(parent)
        self.data_tab.set_not_in_use()
        self.image_tab.set_not_usable()
        self.video_tab.set_in_use()
        self.data_tab.clicked.connect(self.data_tab_is_clicked)
        self.image_tab.clicked.connect(self.image_tab_is_clicked)
        self.thread = {}
        self.thread['VideoReader'] = VideoReaderThread(self.parent())
        self.thread['OneArena'] = OneArenaThread(self.parent())
        self.thread['ChangeOneRepResult'] = ChangeOneRepResultThread(self.parent())
        self.thread['RunAll'] = RunAllThread(self.parent())
        self.previous_arena = 0

        self.layout = QtWidgets.QGridLayout()
        self.grid_widget = QtWidgets.QWidget()

        # self.title = FixedText('Video tracking', police=30, night_mode=self.parent().po.all['night_mode'])
        # self.title.setAlignment(QtCore.Qt.AlignHCenter)
        curr_row_main_layout = 0
        ncol = 1
        # self.layout.addWidget(self.title, curr_row_main_layout, 0, 2, ncol)
        # curr_row_main_layout += 2
        self.layout.addItem(self.vertical_space, curr_row_main_layout, 0, 1, ncol)
        curr_row_main_layout += 1
        #
        # self.layout.addWidget(self.arena_widget, curr_row_main_layout, 0)
        # curr_row_main_layout += 1

        # Open subtitle
        self.general_step_widget = QtWidgets.QWidget()
        self.general_step_layout = QtWidgets.QHBoxLayout()
        self.current_step = 0
        self.general_step_label = FixedText('Step 1: Tune parameters to improve Detection', night_mode=self.parent().po.all['night_mode'])
        self.general_step_button = PButton('Done', night_mode=self.parent().po.all['night_mode'])
        self.general_step_button.clicked.connect(self.step_done_is_clicked)
        self.general_step_layout.addItem(self.horizontal_space)
        self.general_step_layout.addWidget(self.general_step_label)
        self.general_step_layout.addWidget(self.general_step_button)
        self.general_step_layout.addItem(self.horizontal_space)
        self.general_step_widget.setLayout(self.general_step_layout)
        self.layout.addWidget(self.general_step_widget, curr_row_main_layout, 0, 1, ncol)
        curr_row_main_layout += 1

        # Open central widget
        self.video_display_widget = QtWidgets.QWidget()
        self.video_display_layout = QtWidgets.QHBoxLayout()
        self.video_display_layout.addItem(self.horizontal_space)
        #   Open left widget
        self.left_options_widget = QtWidgets.QWidget()
        self.left_options_layout = QtWidgets.QVBoxLayout()
        self.left_options_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # self.left_options_widget.setFixedWidth(420)

        self.arena_widget = QtWidgets.QWidget()
        self.arena_layout = QtWidgets.QHBoxLayout()
        self.arena_label = FixedText('Arena to analyze:',
                                       tip="Among selected folders, choose a arena from the first folder\nThen, click on *Quick (or *Full) detection* to load and analyse one arena\nFinally, click on *Read* to see the resulting analysis\n\nSupplementary information:\nLoading will be faster if videos are already saved as ind_*.npy\n*Post processing* automatically runs *Detection* and *Detection* automatically runs *Load One arena*\nEach being faster than the previous one",
                                       night_mode=self.parent().po.all['night_mode'])
        # if isinstance(self.parent().po.all['sample_number_per_folder'], int):
        #     self.parent().po.all['folder_number'] = 1
        # if self.parent().po.all['folder_number'] > 1:
        #     sample_size = self.parent().po.all['sample_number_per_folder']
        # else:
        sample_size = self.parent().po.all['sample_number_per_folder'][0]
        if self.parent().po.all['arena'] > sample_size:
            self.parent().po.all['arena'] = 1

        self.arena = Spinbox(min=1, max=1000000, val=self.parent().po.all['arena'],
                               night_mode=self.parent().po.all['night_mode'])
        self.arena.valueChanged.connect(self.arena_changed)

        # self.arena_layout.addItem(self.horizontal_space)
        self.arena_layout.addWidget(self.arena_label)
        self.arena_layout.addWidget(self.arena)
        # self.arena_layout.addItem(self.horizontal_space)
        self.arena_widget.setLayout(self.arena_layout)
        # self.arena_layout.setAlignment(QtCore.Qt.AlignHCenter)
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
        self.maximal_growth_factor_label = FixedText('Maximal growth factor:',
                                                    tip="This factor should be tried and increased (resp. decreases)\nif the analysis underestimates (resp. overestimates) the cell size.\nThe maximal growth factor is a proportion of pixels in the image. \nIt tells Cellects how much the cell(s) can possibly move or grow from one image to the next.\nIn other words, this is the upper limit of the proportion of the image\nthat can change from being the background to being covered by the cell(s).",
                                                    night_mode=self.parent().po.all['night_mode'])
        self.maximal_growth_factor.valueChanged.connect(self.maximal_growth_factor_changed)
        self.growth_per_frame_layout.addWidget(self.maximal_growth_factor_label)
        self.growth_per_frame_layout.addWidget(self.maximal_growth_factor)
        # self.growth_per_frame_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.growth_per_frame_widget.setLayout(self.growth_per_frame_layout)
        self.left_options_layout.addWidget(self.growth_per_frame_widget)

        self.iterate_widget = QtWidgets.QWidget()
        self.iterate_layout = QtWidgets.QHBoxLayout()
        self.repeat_video_smoothing = Spinbox(min=0, max=10, val=self.parent().po.vars['repeat_video_smoothing'],
                                         night_mode=self.parent().po.all['night_mode'])
        self.repeat_video_smoothing_label = FixedText('Repeat video smoothing:',
                                                 tip="Increase (with steps of 1) if video noise is the source of detection failure",
                                                 night_mode=self.parent().po.all['night_mode'])
        self.repeat_video_smoothing.valueChanged.connect(self.repeat_video_smoothing_changed)
        self.iterate_layout.addWidget(self.repeat_video_smoothing_label)
        self.iterate_layout.addWidget(self.repeat_video_smoothing)
        # self.iterate_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.iterate_widget.setLayout(self.iterate_layout)
        self.left_options_layout.addWidget(self.iterate_widget)


        self.select_option_label = FixedText('Segmentation method:',
                                             tip='Select the option allowing the best cell delimitation.',
                                             night_mode=self.parent().po.all['night_mode'])
        self.select_option = Combobox([], night_mode=self.parent().po.all['night_mode'])
        self.select_option_label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.select_option.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.select_option.setFixedWidth(175)
        # self.select_option_label.setFixedWidth(265)
        # self.select_option.setFixedWidth(150)
        self.select_option.addItem("1. Frame by frame")
        self.select_option.addItem("2. Dynamical threshold")
        self.select_option.addItem("3. Dynamical slope")
        self.select_option.addItem("4. Threshold and Slope")
        self.select_option.addItem("5. Threshold or Slope")
        # for option in range(5):
        #     self.select_option.addItem(f"Option {option + 1}")
        self.select_option.setCurrentIndex(self.parent().po.all['video_option'])
        self.select_option.currentTextChanged.connect(self.option_changed)
        # self.select_option_label.setVisible(self.parent().po.vars["color_number"] == 2)
        # self.select_option.setVisible(self.parent().po.vars["color_number"] == 2)

        # Open the choose best option row layout
        self.options_row_widget = QtWidgets.QWidget()
        self.options_row_layout = QtWidgets.QHBoxLayout()
        # self.options_row_layout.addItem(self.horizontal_space)
        self.options_row_layout.addWidget(self.select_option_label)
        self.options_row_layout.addWidget(self.select_option)
        # self.options_row_layout.addItem(self.horizontal_space)
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
        # self.right_options_widget.setFixedWidth(420)

        self.compute_all_options_label = FixedText('Compute all options',
                                                   tip='Uncheck to do a Post processing on only one option and earn computation time\nSelecting one of the remaining options will display the result from a Detection',
                                                   night_mode=self.parent().po.all['night_mode'])
        self.compute_all_options_cb = Checkbox(self.parent().po.all['compute_all_options'])
        self.compute_all_options_cb.setStyleSheet("margin-left:0%; margin-right:0%;")
        self.compute_all_options_cb.stateChanged.connect(self.compute_all_options_check)
        self.all_options_row_widget = QtWidgets.QWidget()
        self.all_options_row_layout = QtWidgets.QHBoxLayout()
        self.all_options_row_layout.addWidget(self.compute_all_options_cb)
        self.all_options_row_layout.addWidget(self.compute_all_options_label)
        self.all_options_row_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.all_options_row_layout.setAlignment(QtCore.Qt.AlignVCenter)
        self.all_options_row_widget.setLayout(self.all_options_row_layout)
        self.right_options_layout.addWidget(self.all_options_row_widget)

        self.load_one_arena = PButton('Load One arena', night_mode=self.parent().po.all['night_mode'])
        self.load_one_arena.clicked.connect(self.load_one_arena_is_clicked)
        self.detection = PButton('Detection', night_mode=self.parent().po.all['night_mode'])
        self.detection.clicked.connect(self.detection_is_clicked)
        self.read = PButton('Read', night_mode=self.parent().po.all['night_mode'])
        self.read.clicked.connect(self.read_is_clicked)
        self.read.setVisible(False)
        self.right_options_layout.addWidget(self.load_one_arena, alignment=QtCore.Qt.AlignCenter)
        self.right_options_layout.addWidget(self.detection, alignment=QtCore.Qt.AlignCenter)
        self.right_options_layout.addWidget(self.read, alignment=QtCore.Qt.AlignCenter)


        self.right_options_layout.addItem(self.horizontal_space)
        self.right_options_widget.setLayout(self.right_options_layout)
        self.video_display_layout.addWidget(self.right_options_widget)
        self.video_display_layout.addItem(self.horizontal_space)
        # Close central widget
        self.video_display_widget.setLayout(self.video_display_layout)
        # self.video_display_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # self.video_display_widget.setAlignment(QtCore.Qt.AlignHCenter)
        # self.video_display_widget.setFixedHeight(500)
        self.layout.addWidget(self.video_display_widget, curr_row_main_layout, 0)
        curr_row_main_layout += 1

        # Open Second step row
        self.second_step_widget = QtWidgets.QWidget()
        self.second_step_layout = QtWidgets.QHBoxLayout()
        self.second_step_layout.addItem(self.horizontal_space)
        self.second_step_widget.setVisible(False)

        self.fading_widget = QtWidgets.QWidget()
        self.fading_layout = QtWidgets.QHBoxLayout()
        self.do_fading = Checkbox(self.parent().po.vars['do_fading'])
        self.do_fading.setStyleSheet("margin-left:0%; margin-right:0%;")
        self.do_fading.stateChanged.connect(self.do_fading_check)
        self.fading = Spinbox(min=- 1, max=1, val=self.parent().po.vars['fading'], decimals=2,
                               night_mode=self.parent().po.all['night_mode'])
        self.fading_label = FixedText('Fading detection',
                                       tip="Set a value between -1 and 1\nnear - 1: it will never detect when the cell leaves an area\nnear 1: it may stop detecting cell (because cell will be considered left from any area)",
                                       night_mode=self.parent().po.all['night_mode'])
        self.fading.valueChanged.connect(self.fading_changed)
        # self.fading.setVisible(self.parent().po.vars['do_fading'])
        self.fading_layout.addWidget(self.do_fading)
        self.fading_layout.addWidget(self.fading_label)
        self.fading_layout.addWidget(self.fading)
        self.fading_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.fading_widget.setLayout(self.fading_layout)
        self.second_step_layout.addWidget(self.fading_widget)

        self.post_processing = PButton('Post processing', night_mode=self.parent().po.all['night_mode'])
        self.post_processing.clicked.connect(self.post_processing_is_clicked)
        self.second_step_layout.addWidget(self.post_processing)
        # self.second_step_layout.addWidget(self.post_processing, alignment=QtCore.Qt.AlignCenter)

        self.save_one_result = PButton('Save One Result', night_mode=self.parent().po.all['night_mode'])
        self.save_one_result.clicked.connect(self.save_one_result_is_clicked)
        self.second_step_layout.addWidget(self.save_one_result)
        # self.second_step_layout.addWidget(self.save_one_result, alignment=QtCore.Qt.AlignCenter)

        # Close Second step row
        self.second_step_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.second_step_layout.addItem(self.horizontal_space)
        self.second_step_widget.setLayout(self.second_step_layout)
        self.layout.addItem(self.vertical_space, curr_row_main_layout, 0, 1, ncol)
        curr_row_main_layout += 1
        self.layout.addWidget(self.second_step_widget, curr_row_main_layout, 0)
        curr_row_main_layout += 1

        # Open last options row widget
        self.last_options_widget = QtWidgets.QWidget()
        self.last_options_layout = QtWidgets.QHBoxLayout()
        self.last_options_layout.addItem(self.horizontal_space)

        # #  advanced mode widget
        # self.advanced_mode_widget = QtWidgets.QWidget()
        # self.advanced_mode_layout = QtWidgets.QHBoxLayout()
        # advanced_mode = self.parent().po.all['expert_mode']
        # self.advanced_mode_cb = Checkbox(self.parent().po.all['expert_mode'])
        # self.advanced_mode_cb.setStyleSheet("margin-left:0%; margin-right:0%;")
        # self.advanced_mode_cb.stateChanged.connect(self.advanced_mode_check)
        # self.advanced_mode_label = FixedText('Go to step 2 directly', align='l',
        #                                      tip="Allow the user to try Post processing before having tuned the parameters related to Detection.",
        #                                      night_mode=self.parent().po.all['night_mode'])
        # # self.advanced_mode_label.setAlignment(QtCore.Qt.AlignTop)
        # self.advanced_mode_layout.addWidget(self.advanced_mode_cb)
        # self.advanced_mode_layout.addWidget(self.advanced_mode_label)
        # self.advanced_mode_layout.addItem(self.horizontal_space)
        # self.advanced_mode_layout.setAlignment(QtCore.Qt.AlignHCenter)
        # self.advanced_mode_widget.setLayout(self.advanced_mode_layout)
        # self.last_options_layout.addWidget(self.advanced_mode_widget)

        self.advanced_parameters = PButton('Advanced Parameters', night_mode=self.parent().po.all['night_mode'])
        self.advanced_parameters.clicked.connect(self.advanced_parameters_is_clicked)
        self.last_options_layout.addWidget(self.advanced_parameters)

        #  Required Outputs widget
        self.required_outputs = PButton('Required Outputs', night_mode=self.parent().po.all['night_mode'])
        self.required_outputs.clicked.connect(self.required_outputs_is_clicked)
        self.last_options_layout.addWidget(self.required_outputs)

        #  Save all choices widget
        self.save_all_vars = PButton('Save all choices', night_mode=self.parent().po.all['night_mode'])
        self.save_all_vars.clicked.connect(self.save_current_settings)
        self.last_options_layout.addWidget(self.save_all_vars)

        # Close last options widget
        self.last_options_layout.addItem(self.horizontal_space)
        self.last_options_widget.setLayout(self.last_options_layout)
        self.layout.addWidget(self.last_options_widget, curr_row_main_layout, 0)
        curr_row_main_layout += 1

        self.message = QtWidgets.QLabel(self)
        self.message.setText('')
        self.message.setStyleSheet("color: rgb(230, 145, 18)")
        self.message.setAlignment(QtCore.Qt.AlignLeft)

        self.previous = PButton('Previous', night_mode=self.parent().po.all['night_mode'])
        self.previous.clicked.connect(self.previous_is_clicked)

        self.run_all = PButton('Run All', night_mode=self.parent().po.all['night_mode'])
        self.run_all.clicked.connect(self.run_all_is_clicked)

        # Open last row widget
        self.last_row_widget = QtWidgets.QWidget()
        self.last_row_layout = QtWidgets.QHBoxLayout()
        self.last_row_layout.addWidget(self.previous)
        self.last_row_layout.addItem(self.horizontal_space)
        self.last_row_layout.addWidget(self.message)
        self.last_row_layout.addWidget(self.run_all)
        # Close last row widget
        self.last_row_widget.setLayout(self.last_row_layout)
        self.layout.addItem(self.vertical_space, curr_row_main_layout, 0, 1, ncol)
        self.layout.addWidget(self.last_row_widget, curr_row_main_layout, 0)

        self.grid_widget.setLayout(self.layout)
        self.Vlayout.addItem(self.vertical_space)
        self.Vlayout.addWidget(self.grid_widget)
        self.setLayout(self.Vlayout)
        # self.advanced_mode_check()

    def display_conditionally_visible_widgets(self):
        self.select_option_label.setVisible(self.parent().po.vars["color_number"] == 2)
        self.select_option.setVisible(self.parent().po.vars["color_number"] == 2)
        self.fading.setVisible(self.parent().po.vars['do_fading'])

    def step_done_is_clicked(self):
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
        self.current_step = 0
        self.general_step_label.setText('Step 1: Tune parameters to improve Detection')
        self.general_step_label.setToolTip('Detection uses only the visible parameters and those\npreviously determined on the first or last image.')
        self.general_step_button.setVisible(True)
        self.second_step_widget.setVisible(False)

    def full_screen_display(self, event):
        self.popup_img = FullScreenImage(self.parent().image_to_display, self.parent().screen_width, self.parent().screen_height)
        self.popup_img.show()

    def option_changed(self):
        """
        Update the video, save parameters
        :return:
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
        # self.parent().po.motion

    # def advanced_mode_check(self):
    #     advanced_mode = self.advanced_mode_cb.isChecked()
    #     self.parent().po.all['expert_mode'] = advanced_mode
    #     self.second_step_widget.setVisible(advanced_mode or self.current_step > 0)
    #     if advanced_mode:
    #         if self.current_step == 0:
    #             self.current_step += 1
    #             self.general_step_label.setText('Step 2: Tune fading and advanced parameters to improve Post processing')
    #         self.save_one_result.setVisible(self.current_step == 2)

        # self.maximal_growth_factor.setVisible(advanced_mode)
        # self.maximal_growth_factor_label.setVisible(advanced_mode)
        # self.fading.setVisible(advanced_mode)
        # self.fading_label.setVisible(advanced_mode)
        # self.repeat_video_smoothing.setVisible(advanced_mode)
        # self.repeat_video_smoothing_label.setVisible(advanced_mode)

    def data_tab_is_clicked(self):
        if self.thread['VideoReader'].isRunning() or self.thread['OneArena'].isRunning() or self.thread['ChangeOneRepResult'].isRunning() or self.parent().firstwindow.thread["RunAll"].isRunning():
            self.message.setText("Wait for the analysis to end, or restart Cellects")
        else:
            self.parent().last_tab = "data_specifications"
            self.parent().change_widget(0)  # FirstWidget

    def image_tab_is_clicked(self):
        if self.image_tab.state != "not_usable":
            if self.thread['VideoReader'].isRunning() or self.thread['OneArena'].isRunning() or self.thread[
                'ChangeOneRepResult'].isRunning() or self.parent().firstwindow.thread["RunAll"].isRunning():
                self.message.setText("Wait for the analysis to end, or restart Cellects")
            else:
                self.parent().last_tab = "video_analysis"
                self.parent().change_widget(2)


    def required_outputs_is_clicked(self):
        self.parent().last_is_first = False
        self.parent().change_widget(4)  # RequiredOutput

    def advanced_parameters_is_clicked(self):
        self.parent().last_is_first = False
        self.parent().widget(5).update_csc_editing_display()
        self.parent().change_widget(5)  # AdvancedParameters

    def previous_is_clicked(self):
        if self.parent().last_tab == "data_specifications":
            self.parent().change_widget(0)  # FirstWidget
        elif self.parent().last_tab == "image_analysis":
            self.parent().change_widget(2)  # ThirdWidget
        self.parent().last_tab = "video_analysis"
        # self.parent().change_widget(2)  # SecondWidget

    def save_all_vars_thread(self):
        if not self.parent().thread['SaveAllVars'].isRunning():
            self.parent().thread['SaveAllVars'].start()  # SaveAllVarsThreadInThirdWidget

    def save_current_settings(self):
        self.parent().po.vars['maximal_growth_factor'] = self.maximal_growth_factor.value()
        self.parent().po.vars['repeat_video_smoothing'] = int(np.round(self.repeat_video_smoothing.value()))
        self.parent().po.vars['do_fading'] = self.do_fading.isChecked()
        self.parent().po.vars['fading'] = self.fading.value()
        self.parent().po.all['compute_all_options'] = self.compute_all_options_cb.isChecked()
        self.option_changed()
        self.save_all_vars_thread()

    def repeat_video_smoothing_changed(self):
        self.parent().po.vars['repeat_video_smoothing'] = int(np.round(self.repeat_video_smoothing.value()))
        # self.save_all_vars_is_clicked()

    def do_fading_check(self):
        self.parent().po.vars['do_fading'] = self.do_fading.isChecked()
        self.fading.setVisible(self.parent().po.vars['do_fading'])

    def fading_changed(self):
        self.parent().po.vars['fading'] = self.fading.value()
        # self.save_all_vars_is_clicked()

    def maximal_growth_factor_changed(self):
        self.parent().po.vars['maximal_growth_factor'] = self.maximal_growth_factor.value()
        # self.save_all_vars_is_clicked()

    def arena_changed(self):
        """
            Put motion to None allows the class OneArenaThread to load the selected arena
        """
        if not self.thread['VideoReader'].isRunning() and not self.thread['OneArena'].isRunning() and not self.thread['ChangeOneRepResult'].isRunning():
            self.parent().po.motion = None
            self.reset_general_step()
            self.parent().po.computed_video_options = np.zeros(5, bool)
            self.parent().po.all['arena'] = int(np.round(self.arena.value()))

    def load_one_arena_is_clicked(self):
        self.reset_general_step()
        # self.save_all_vars_is_clicked()
        self.parent().po.load_quick_full = 0
        self.run_one_arena_thread()

    def compute_all_options_check(self):
        self.parent().po.all['compute_all_options'] = self.compute_all_options_cb.isChecked()

    def detection_is_clicked(self):
        self.reset_general_step()
        # self.save_all_vars_is_clicked()
        self.parent().po.load_quick_full = 1
        self.run_one_arena_thread()

    def post_processing_is_clicked(self):
        # self.save_all_vars_is_clicked()
        self.parent().po.load_quick_full = 2
        logging.info(self.parent().po.vars['maximal_growth_factor'])
        self.run_one_arena_thread()

    def run_one_arena_thread(self):
        """
            Make sure that the previous thread is not running and start the OneArenaThread
            According to the button clicked, this class will only load, load and quick segment,
            or load, quickly segment and fully detect the cell(s) dynamic
        """
        if self.thread['OneArena']._isRunning:
            self.thread['OneArena'].stop()
        self.save_current_settings()
        if self.previous_arena != self.parent().po.all['arena']:
            self.parent().po.motion = None
        self.message.setText("Load the video and initialize analysis, wait...")
        # if not self.parent().po.first_exp_ready_to_run:
        #     self.parent().po.use_data_to_run_cellects_quickly = True
        self.thread['OneArena'].start()  # OneArenaThreadInThirdWidget
        self.thread['OneArena'].message_from_thread_starting.connect(self.display_message_from_thread)
        self.thread['OneArena'].when_loading_finished.connect(self.when_loading_thread_finished)
        self.thread['OneArena'].when_detection_finished.connect(self.when_detection_finished)
        self.thread['OneArena'].image_from_thread.connect(self.display_image_during_thread)

    def when_loading_thread_finished(self, save_loaded_video):
        self.previous_arena = self.parent().po.all['arena']
        # if not self.parent().po.vars['already_greyscale']:
        #     self.parent().po.motion.analysis_instance = self.parent().po.motion.visu.copy()
        # else:
        #     self.parent().po.motion.analysis_instance = self.parent().po.motion.converted_video.copy()
        if save_loaded_video:
            self.thread['WriteVideo'] = WriteVideoThread(self.parent())
            self.thread['WriteVideo'].start()
        if self.parent().po.load_quick_full == 0:
            self.message.setText("Loading done, you can watch the video")
        self.read.setVisible(True)

    def when_detection_finished(self, message):
        # self.option_changed()
        self.previous_arena = self.parent().po.all['arena']
        if self.thread['VideoReader'].isRunning():  # VideoReaderThreadInThirdWidget
            self.thread['VideoReader'].wait()
        if self.parent().po.load_quick_full > 0:
            image = self.parent().po.motion.segmentation[-1, ...]
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
        # self.save_one_result.setVisible(True)

    def display_image_during_thread(self, dictionary):
        self.message.setText(dictionary['message'])
        self.parent().image_to_display = dictionary['current_image']
        self.display_image.update_image(dictionary['current_image'])

    def save_one_result_is_clicked(self):
        if self.parent().po.motion is not None:
            if self.parent().po.load_quick_full == 2:
                if not self.thread['OneArena'].isRunning() and not self.thread['ChangeOneRepResult'].isRunning():
                    self.message.setText(f"Arena {self.parent().po.all['arena']}: Finalize analysis and save, wait...")
                    self.thread['ChangeOneRepResult'].start()  # ChangeOneRepResultThreadInThirdWidget
                    self.thread['ChangeOneRepResult'].message_from_thread.connect(self.display_message_from_thread)
                    self.message.setText("Complete analysis + change that result")
                else:
                    self.message.setText("Wait for the analysis to end")
            else:
                self.message.setText("Run Post processing first")
        else:
            self.message.setText("Run Post processing first")

    def read_is_clicked(self):
        if self.parent().po.motion is not None:
            if self.parent().po.motion.segmentation is not None:
                if not self.thread['OneArena'].isRunning() and not self.thread['VideoReader'].isRunning():
                    self.thread['VideoReader'].start()  # VideoReaderThreadInThirdWidget
                    self.thread['VideoReader'].message_from_thread.connect(self.display_image_during_thread)
                    # if self.parent().po.computed_video_options[self.parent().po.all['video_option']]:
                    #     self.message.setText("Run detection to visualize analysis")
                else:
                    self.message.setText("Wait for the analysis to end")
            else:
                self.message.setText("Run detection first")
        else:
            self.message.setText("Run detection first")
    #
    # def video_display(self, dictionary):
    #     self.drawn_image = dictionary['image']
    #     self.display_image.update_image(dictionary['image'], self.parent().po.vars['contour_color'])
    #     self.message.setText(dictionary['message'])
    #     # self.message.setText(f"Reading done, try to change parameters if necessary")

    def run_all_is_clicked(self):
        if self.thread['OneArena'].isRunning() or self.thread['ChangeOneRepResult'].isRunning():
            self.message.setText("Wait for the current analysis to end")
        else:
            if self.thread['VideoReader'].isRunning():
                self.thread['VideoReader'].wait()
            # self.save_all_vars_is_clicked()
            # self.save_current_settings()
            if self.parent().firstwindow.thread["RunAll"].isRunning():
                self.message.setText('Analysis has already begun in the first window.')
            else:
                if not self.thread['RunAll'].isRunning():
                    # self.save_all_vars_is_clicked()
                    self.save_current_settings()
                    self.parent().po.motion = None
                    self.parent().po.converted_video = None
                    self.parent().po.converted_video2 = None
                    self.parent().po.visu = None
                    self.message.setText("Complete analysis has started, wait...")
                    # if not self.parent().po.first_exp_ready_to_run:
                    #     self.parent().po.use_data_to_run_cellects_quickly = True
                    self.thread['RunAll'].start()  # RunAllThread
                    self.thread['RunAll'].message_from_thread.connect(self.display_message_from_thread)
                    self.thread['RunAll'].image_from_thread.connect(self.display_image_during_thread)
                    # self.parent().imageanalysiswindow.true_init()

    def display_message_from_thread(self, text_from_thread):
        self.message.setText(text_from_thread)

    def closeEvent(self, event):
        event.accept


# if __name__ == "__main__":
#     from cellects.gui.cellects import CellectsMainWidget
#     import sys
#     app = QtWidgets.QApplication([])
#     parent = CellectsMainWidget()
#     session = VideoAnalysisWindow(parent, False)
#     parent.insertWidget(0, session)
#     parent.show()
#     sys.exit(app.exec())
