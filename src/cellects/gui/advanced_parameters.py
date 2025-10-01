#!/usr/bin/env python3
"""This module creates the Advanced Parameters window of the user interface of Cellects
This windows contains most parameters
"""


import logging
import os
from copy import deepcopy
from pathlib import Path

from PySide6 import QtWidgets, QtCore
import numpy as np
from cellects.config.all_vars_dict import DefaultDicts
from cellects.core.cellects_paths import CELLECTS_DIR, CONFIG_DIR
from cellects.gui.custom_widgets import (
    WindowType, PButton, Spinbox, Combobox, Checkbox, FixedText)


class AdvancedParameters(WindowType):
    """
        This class creates the Advanced Parameters window.
        In the app, it is accessible from the first and the Video tracking window. It allows the user to fill in
        some parameters stored in the directory po.all (in RAM) and in all_vars.pkl (in ROM).
        Clicking "Ok" save the directory in RAM and in ROM.
    """
    def __init__(self, parent, night_mode):
        super().__init__(parent, night_mode)

        logging.info("Initialize AdvancedParameters window")
        self.setParent(parent)
        try:
            self.true_init()
        except KeyError:
            default_dicts = DefaultDicts()
            self.parent().po.all = default_dicts.all
            self.parent().po.vars = default_dicts.vars
            self.true_init()

    def true_init(self):
        self.layout = QtWidgets.QVBoxLayout()

        self.left_scroll_table = QtWidgets.QScrollArea()  # QTableWidget()  # Scroll Area which contains the widgets, set as the centralWidget
        self.left_scroll_table.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.left_scroll_table.setMinimumHeight(150)#self.parent().im_max_height - 100
        self.left_scroll_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.left_scroll_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.left_scroll_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self.left_col_layout = QtWidgets.QVBoxLayout()
        self.right_col_layout = QtWidgets.QVBoxLayout()
        self.left_col_widget = QtWidgets.QWidget()
        self.right_col_widget = QtWidgets.QWidget()
        # curr_row_1st_col = 0
        ncol = 11
        # Create the main Title
        self.title = FixedText('Advanced parameters', police=30, night_mode=self.parent().po.all['night_mode'])
        self.title.setAlignment(QtCore.Qt.AlignHCenter)
        # Create the main layout
        self.layout.addWidget(self.title)
        # self.layout.addItem(self.vertical_space)
        # self.layout.addWidget(self.title, curr_row_1st_col, 0, 2, ncol)
        # curr_row_1st_col += 2
        # horzspaceItem = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding)
        # self.layout.addItem(horzspaceItem, 2, 0, 1, 5)
        # curr_row_1st_col += 1
        # Create the stylesheet for the boxes allowing to categorize advanced parameters.
        boxstylesheet = \
            ".QWidget {\n" \
            + "border: 1px solid black;\n" \
            + "border-radius: 20px;\n" \
            + "}"


        # I/ First box: General parameters
        # I/A/ Title
        self.general_param_box_label = FixedText('General parameters:', tip="",
                                         night_mode=self.parent().po.all['night_mode'])
        self.left_col_layout.addWidget(self.general_param_box_label)
        # self.layout.addWidget(self.general_param_box_label, curr_row_1st_col, 1)
        # curr_row_1st_col += 1
        # I/B/ Create the box
        self.general_param_box_layout = QtWidgets.QGridLayout()
        self.general_param_box_widget = QtWidgets.QWidget()
        self.general_param_box_widget.setStyleSheet(boxstylesheet)
        # I/C/ Create widgets
        self.automatically_crop = Checkbox(self.parent().po.all['automatically_crop'])
        self.automatically_crop_label = FixedText('Automatically crop images', tip="If more than one cell shape are (or may appear) in each arena", night_mode=self.parent().po.all['night_mode'])

        self.subtract_background = Checkbox(self.parent().po.vars['subtract_background'])
        self.subtract_background.stateChanged.connect(self.subtract_background_check)
        self.subtract_background_label = FixedText('Subtract background', tip="Apply an algorithm allowing to remove a potential brightness gradient from images during analysis", night_mode=self.parent().po.all['night_mode'])

        self.keep_cell_and_back_for_all_folders = Checkbox(self.parent().po.all['keep_cell_and_back_for_all_folders'])
        self.keep_cell_and_back_for_all_folders_label = FixedText('Keep Cell and Back drawings for all folders',
                                               tip="During the first image analysis, if the user drew cell and back to help detection\n- Keep this information for all folders (if checked)\n- Only use this information for the current folder (if unchecked)",
                                               night_mode=self.parent().po.all['night_mode'])

        self.correct_errors_around_initial = Checkbox(self.parent().po.vars['correct_errors_around_initial'])
        self.correct_errors_around_initial_label = FixedText('Correct errors around initial shape',
                                               tip="Apply an algorithm allowing to correct some failure around the initial shape\nThese errors are most likely due to color variations\n themselves due to substrate width differences crossed by light\naround initial cell lying on an opaque substrate",
                                               night_mode=self.parent().po.all['night_mode'])

        self.prevent_fast_growth_near_periphery = Checkbox(self.parent().po.vars['prevent_fast_growth_near_periphery'])
        self.prevent_fast_growth_near_periphery_label = FixedText('Prevent fast growth near periphery',
                                               tip="During video analysis, the borders of the arena may create wrong detection\n- Remove the detection of the specimen(s) that move too fast near periphery (if checked)\n- Do not change the detection (if unchecked)",
                                               night_mode=self.parent().po.all['night_mode'])

        self.prevent_fast_growth_near_periphery.stateChanged.connect(self.prevent_fast_growth_near_periphery_check)
        self.periphery_width = Spinbox(min=1, max=1000, val=self.parent().po.vars['periphery_width'],
                                            decimals=0, night_mode=self.parent().po.all['night_mode'])
        self.periphery_width_label = FixedText('Periphery width',
                                               tip="The width, in pixels, of the arena s border designated as the peripheral region",
                                               night_mode=self.parent().po.all['night_mode'])
        self.max_periphery_growth = Spinbox(min=1, max=1000000, val=self.parent().po.vars['max_periphery_growth'],
                                            decimals=0, night_mode=self.parent().po.all['night_mode'])
        self.max_periphery_growth_label = FixedText('Max periphery growth',
                                               tip="The maximum detectable size (in pixels) of a shape in a single frame near the periphery of the arena.\nLarger shapes will be considered as noise.",
                                               night_mode=self.parent().po.all['night_mode'])
        self.prevent_fast_growth_near_periphery_check()


        # I/D/ Arrange widgets in the box
        self.general_param_box_layout.addWidget(self.automatically_crop, 0, 0)
        self.general_param_box_layout.addWidget(self.automatically_crop_label, 0, 1)
        self.general_param_box_layout.addWidget(self.subtract_background, 1, 0)
        self.general_param_box_layout.addWidget(self.subtract_background_label, 1, 1)
        self.general_param_box_layout.addWidget(self.keep_cell_and_back_for_all_folders, 2, 0)
        self.general_param_box_layout.addWidget(self.keep_cell_and_back_for_all_folders_label, 2, 1)
        self.general_param_box_layout.addWidget(self.correct_errors_around_initial, 3, 0)
        self.general_param_box_layout.addWidget(self.correct_errors_around_initial_label, 3, 1)
        self.general_param_box_layout.addWidget(self.prevent_fast_growth_near_periphery, 4, 0)
        self.general_param_box_layout.addWidget(self.prevent_fast_growth_near_periphery_label, 4, 1)
        self.general_param_box_layout.addWidget(self.periphery_width, 5, 0)
        self.general_param_box_layout.addWidget(self.periphery_width_label, 5, 1)
        self.general_param_box_layout.addWidget(self.max_periphery_growth, 6, 0)
        self.general_param_box_layout.addWidget(self.max_periphery_growth_label, 6, 1)
        self.general_param_box_widget.setLayout(self.general_param_box_layout)
        self.left_col_layout.addWidget(self.general_param_box_widget)
        # self.layout.addWidget(self.general_param_box_widget, curr_row_1st_col, 1, 2, 2)
        # curr_row_1st_col += 2

        # II/ Second box: One cell/colony per arena
        # II/A/ Title
        self.one_per_arena_label = FixedText('One cell/colony per arena parameters:', tip="",
                                            night_mode=self.parent().po.all['night_mode'])
        self.left_col_layout.addWidget(self.one_per_arena_label)
        # II/B/ Create the box
        self.one_per_arena_box_layout = QtWidgets.QGridLayout()
        self.one_per_arena_box_widget = QtWidgets.QWidget()
        self.one_per_arena_box_widget.setStyleSheet(boxstylesheet)

        # II/C/ Create widgets
        self.all_specimens_have_same_direction = Checkbox(self.parent().po.all['all_specimens_have_same_direction'])
        # self.all_specimens_have_same_direction.stateChanged.connect(self.all_specimens_have_same_direction_changed)
        self.all_specimens_have_same_direction_label = FixedText('All specimens have the same direction',
                                                         tip="This parameter only affects the slow algorithm of automatic arena detection.\nChecking it will improve the chances to correctly detect arenas when\n all cells move in the same direction",
                                                         night_mode=self.parent().po.all['night_mode'])


        connect_distant_shape = self.parent().po.all['connect_distant_shape_during_segmentation']
        self.connect_distant_shape_during_segmentation = Checkbox(connect_distant_shape)
        self.connect_distant_shape_during_segmentation.stateChanged.connect(self.do_distant_shape_int_changed)
        self.connect_distant_shape_label = FixedText('Connect distant shapes',
                                                         tip="Allows a homemade algorithm allowing to\nprogressively (i.e. at the growth rate speed of neighboring pixels)\nconnect distant shapes to original shape(s)\nWarning: this option can drastically increase the duration of the analysis",
                                                         night_mode=self.parent().po.all['night_mode'])
        self.detection_range_factor = Spinbox(min=0, max=1000000,
                                                      val=self.parent().po.vars['detection_range_factor'],
                                                      night_mode=self.parent().po.all['night_mode'])
        self.detection_range_factor_label = FixedText('Detection range factor:',
                                                              tip="From 1 to 10, increase the allowed distance from original shape(s) to connect distant shapes",
                                                              night_mode=self.parent().po.all['night_mode'])

        # Connect distant shape algo:
        do_use_max_size = self.parent().po.vars['max_size_for_connection'] is not None and connect_distant_shape
        do_use_min_size = self.parent().po.vars['min_size_for_connection'] is not None and connect_distant_shape
        self.use_max_size = Checkbox(do_use_max_size, night_mode=self.parent().po.all['night_mode'])
        self.use_min_size = Checkbox(do_use_min_size, night_mode=self.parent().po.all['night_mode'])
        self.use_max_size.stateChanged.connect(self.use_max_size_changed)
        self.use_min_size.stateChanged.connect(self.use_min_size_changed)

        self.use_max_size_label = FixedText('Use max size as a threshold',
                                            tip="To decide whether distant shapes should get connected",
                                            night_mode=self.parent().po.all['night_mode'])
        self.use_min_size_label = FixedText('Use min size as a threshold',
                                            tip="To decide whether distant shapes should get connected",
                                            night_mode=self.parent().po.all['night_mode'])
        self.max_size_for_connection_label = FixedText('Max (pixels):', night_mode=self.parent().po.all['night_mode'])
        self.min_size_for_connection_label = FixedText('Min (pixels):', night_mode=self.parent().po.all['night_mode'])
        if do_use_max_size:
            self.max_size_for_connection = Spinbox(min=0, max=1000000,
                                                  val=self.parent().po.vars['max_size_for_connection'],
                                                  night_mode=self.parent().po.all['night_mode'])
        else:
            self.max_size_for_connection = Spinbox(min=0, max=1000000, val=50,
                                                  night_mode=self.parent().po.all['night_mode'])
        if do_use_min_size:
            self.min_size_for_connection = Spinbox(min=0, max=1000000,
                                                  val=self.parent().po.vars['min_size_for_connection'],
                                                  night_mode=self.parent().po.all['night_mode'])
        else:
            self.min_size_for_connection = Spinbox(min=0, max=1000000, val=0,
                                                  night_mode=self.parent().po.all['night_mode'])
        # set things visible or invisible:
        # self.detection_range_factor.setVisible(connect_distant_shape)
        # self.detection_range_factor_label.setVisible(connect_distant_shape)
        # self.use_max_size.setVisible(connect_distant_shape)
        # self.use_min_size.setVisible(connect_distant_shape)
        # self.use_max_size_label.setVisible(connect_distant_shape)
        # self.use_min_size_label.setVisible(connect_distant_shape)
        #
        # self.max_size_for_connection.setVisible(do_use_max_size)
        # self.max_size_for_connection_label.setVisible(do_use_max_size)
        # self.min_size_for_connection.setVisible(do_use_min_size)
        # self.min_size_for_connection_label.setVisible(do_use_min_size)

        self.use_min_size.setStyleSheet("margin-left:100%; margin-right:0%;")
        self.min_size_for_connection_label.setAlignment(QtCore.Qt.AlignRight)
        self.use_max_size.setStyleSheet("margin-left:100%; margin-right:0%;")
        self.max_size_for_connection_label.setAlignment(QtCore.Qt.AlignRight)

        # II/D/ Arrange widgets in the box
        curr_box_row = 0
        self.one_per_arena_box_layout.addWidget(self.connect_distant_shape_during_segmentation, curr_box_row, 0)
        self.one_per_arena_box_layout.addWidget(self.connect_distant_shape_label, curr_box_row, 1)
        curr_box_row += 1
        self.one_per_arena_box_layout.addWidget(self.detection_range_factor, curr_box_row, 0)
        self.one_per_arena_box_layout.addWidget(self.detection_range_factor_label, curr_box_row, 1)
        curr_box_row += 1
        self.one_per_arena_box_layout.addWidget(self.use_min_size, curr_box_row, 0)
        self.one_per_arena_box_layout.addWidget(self.use_min_size_label, curr_box_row, 1)
        curr_box_row += 1
        self.one_per_arena_box_layout.addWidget(self.min_size_for_connection_label, curr_box_row, 0)
        self.one_per_arena_box_layout.addWidget(self.min_size_for_connection, curr_box_row, 1)
        curr_box_row += 1
        self.one_per_arena_box_layout.addWidget(self.use_max_size, curr_box_row, 0)
        self.one_per_arena_box_layout.addWidget(self.use_max_size_label, curr_box_row, 1)
        curr_box_row += 1
        self.one_per_arena_box_layout.addWidget(self.max_size_for_connection_label, curr_box_row, 0)
        self.one_per_arena_box_layout.addWidget(self.max_size_for_connection, curr_box_row, 1)
        curr_box_row += 1
        self.one_per_arena_box_layout.addWidget(self.all_specimens_have_same_direction, curr_box_row, 0)
        self.one_per_arena_box_layout.addWidget(self.all_specimens_have_same_direction_label, curr_box_row, 1)
        curr_box_row += 1

        self.one_per_arena_box_widget.setLayout(self.one_per_arena_box_layout)
        self.left_col_layout.addWidget(self.one_per_arena_box_widget)
        # self.layout.addWidget(self.one_per_arena_box_widget, curr_row_1st_col, 1, 3, 2)
        # curr_row_1st_col += 2# curr_box_row

        # III/ Third box: Appearing cell/colony
        # III/A/ Title
        self.appearing_cell_label = FixedText('Appearing cell/colony parameters:', tip="",
                                              night_mode=self.parent().po.all['night_mode'])

        self.left_col_layout.addWidget(self.appearing_cell_label)
        # self.layout.addWidget(self.appearing_cell_label, curr_row_1st_col, 1)
        # curr_row_1st_col += 1
        # III/B/ Create the box
        self.appearing_cell_box_layout = QtWidgets.QGridLayout()
        self.appearing_cell_box_widget = QtWidgets.QWidget()
        self.appearing_cell_box_widget.setStyleSheet(boxstylesheet)

        # III/C/ Create widgets
        self.first_move_threshold = Spinbox(min=0, max=1000000, val=self.parent().po.all['first_move_threshold_in_mm²'],
                                            decimals=6, night_mode=self.parent().po.all['night_mode'])
        self.first_move_threshold_label = FixedText('Minimal size to detect a cell/colony',
                                                    tip="In mm². All appearing cell/colony lesser than this value will be considered as noise",
                                                    night_mode=self.parent().po.all['night_mode'])
        # self.first_move_threshold.setVisible(not self.parent().po.all['automatic_size_thresholding'])
        # self.first_move_threshold_label.setVisible(not self.parent().po.all['automatic_size_thresholding'])
        self.do_automatic_size_thresholding = Checkbox(self.parent().po.all['automatic_size_thresholding'])
        self.do_automatic_size_thresholding_label = FixedText('Automatic size threshold for appearance/motion',
                                                              night_mode=self.parent().po.all['night_mode'])
        self.do_automatic_size_thresholding.stateChanged.connect(self.do_automatic_size_thresholding_changed)
        self.appearing_selection = Combobox(["largest", "most_central"], night_mode=self.parent().po.all['night_mode'])
        self.appearing_selection_label = FixedText('Appearance detection method',
                                                   tip="When specimen(s) are invisible at the beginning of the experiment and appear progressively",
                                                   night_mode=self.parent().po.all['night_mode'])
        self.appearing_selection.setCurrentText(self.parent().po.vars['appearance_detection_method'])
        self.appearing_selection.setFixedWidth(190)

        # III/D/ Arrange widgets in the box
        curr_box_row = 0
        self.appearing_cell_box_layout.addWidget(self.do_automatic_size_thresholding, curr_box_row, 0)
        self.appearing_cell_box_layout.addWidget(self.do_automatic_size_thresholding_label, curr_box_row, 1)
        curr_box_row += 1
        self.appearing_cell_box_layout.addWidget(self.first_move_threshold, curr_box_row, 0)
        self.appearing_cell_box_layout.addWidget(self.first_move_threshold_label, curr_box_row, 1)
        curr_box_row += 1
        self.appearing_cell_box_layout.addWidget(self.appearing_selection, curr_box_row, 0)
        self.appearing_cell_box_layout.addWidget(self.appearing_selection_label, curr_box_row, 1)
        curr_box_row += 1

        self.appearing_cell_box_widget.setLayout(self.appearing_cell_box_layout)
        self.left_col_layout.addWidget(self.appearing_cell_box_widget)
        # self.layout.addWidget(self.appearing_cell_box_widget, curr_row_1st_col, 1, 2, 2)
        # curr_row_1st_col += curr_box_row

        # IV/ Fourth box: Oscillation period:#
        # IV/A/ Title
        self.oscillation_label = FixedText('Oscillatory parameters:', tip="",
                                              night_mode=self.parent().po.all['night_mode'])
        self.left_col_layout.addWidget(self.oscillation_label)

        self.oscillation_period_layout = QtWidgets.QGridLayout()
        self.oscillation_period_widget = QtWidgets.QWidget()
        self.oscillation_period_widget.setStyleSheet(boxstylesheet)

        self.oscillation_period = Spinbox(min=0, max=10000, val=self.parent().po.vars['expected_oscillation_period'], decimals=2,
                                          night_mode=self.parent().po.all['night_mode'])
        self.oscillation_period_label = FixedText('Expected oscillation period (min)',
                                                  tip="If one expect biotic oscillations to occur",
                                                  night_mode=self.parent().po.all['night_mode'])

        self.minimal_oscillating_cluster_size = Spinbox(min=1, max=1000000000, decimals=0, val=self.parent().po.vars['minimal_oscillating_cluster_size'],
                                          night_mode=self.parent().po.all['night_mode'])
        self.minimal_oscillating_cluster_size_label = FixedText('Minimal oscillating cluster size',
                                                  tip="In pixels\nWhen analyzing oscillations within the detected specimen(s)\nCellects looks for clusters of pixels that oscillate synchronously\nThis parameter sets the minimal size (in pixels) of these clusters.",
                                                  night_mode=self.parent().po.all['night_mode'])

        self.oscillation_period_layout.addWidget(self.oscillation_period, 0, 0)
        self.oscillation_period_layout.addWidget(self.oscillation_period_label, 0, 1)
        self.oscillation_period_layout.addWidget(self.minimal_oscillating_cluster_size, 1, 0)
        self.oscillation_period_layout.addWidget(self.minimal_oscillating_cluster_size_label, 1, 1)

        self.oscillation_period_widget.setLayout(self.oscillation_period_layout)
        self.left_col_layout.addWidget(self.oscillation_period_widget)


        # V/ Fifth box: Fractal parameters:#
        # IV/A/ Title
        # self.fractal_label = FixedText('Fractal parameters:', tip="",
        #                                       night_mode=self.parent().po.all['night_mode'])
        # self.left_col_layout.addWidget(self.fractal_label)
        #
        # self.fractal_layout = QtWidgets.QGridLayout()
        # self.fractal_widget = QtWidgets.QWidget()
        # self.fractal_widget.setStyleSheet(boxstylesheet)
        #
        # self.fractal_box_side_threshold = Spinbox(min=0, max=100000, val=self.parent().po.vars['fractal_box_side_threshold'], decimals=0,
        #                                   night_mode=self.parent().po.all['night_mode'])
        # self.fractal_box_side_threshold_label = FixedText('Fractal box side threshold',
        #                                           tip="Increase/decrease to adjust the minimal side length (pixels) of an image\nto compute the Minkowski dimension using the box counting method.",
        #                                           night_mode=self.parent().po.all['night_mode'])
        # self.fractal_layout.addWidget(self.fractal_box_side_threshold, 3, 0)
        # self.fractal_layout.addWidget(self.fractal_box_side_threshold_label, 3, 1)
        # self.fractal_zoom_step = Spinbox(min=0, max=100000, val=self.parent().po.vars['fractal_zoom_step'], decimals=0,
        #                                   night_mode=self.parent().po.all['night_mode'])
        # self.fractal_zoom_step_label = FixedText('Fractal zoom step',
        #                                           tip="When using the box counting method to compute the Minkowski dimension\nThe zoom step is the side length (pixels) difference between each zoom level.\nWhen set to 0, the default zoom step is all possible powers of two.",
        #                                           night_mode=self.parent().po.all['night_mode'])
        # self.fractal_layout.addWidget(self.fractal_zoom_step, 4, 0)
        # self.fractal_layout.addWidget(self.fractal_zoom_step_label, 4, 1)
        #
        # self.fractal_widget.setLayout(self.fractal_layout)
        # self.left_col_layout.addWidget(self.fractal_widget)

        # V/ Fifth box: Network detection parameters:#
        # IV/A/ Title
        self.network_label = FixedText('Network parameters:', tip="",
                                              night_mode=self.parent().po.all['night_mode'])
        self.left_col_layout.addWidget(self.network_label)

        self.network_layout = QtWidgets.QGridLayout()
        self.network_widget = QtWidgets.QWidget()
        self.network_widget.setStyleSheet(boxstylesheet)


        self.network_detection_threshold = Spinbox(min=0, max=255, val=self.parent().po.vars['network_detection_threshold'], decimals=0,
                                          night_mode=self.parent().po.all['night_mode'])
        self.network_detection_threshold_label = FixedText('Network detection threshold',
                                                  tip="To detect the network, Cellects segment small parts of the image using a sliding window.\nThis threshold is an intensity value [0, 255]\napplied to the sliding window to not consider homogeneous substes of the image\ni.e. This is the minimal variation in intensity to consider that some pixels are parts of the network.",
                                                  night_mode=self.parent().po.all['night_mode'])
        # self.mesh_side_length = Spinbox(min=2, max=1000000, val=self.parent().po.vars['network_mesh_side_length'], decimals=0,
        #                                   night_mode=self.parent().po.all['night_mode'])
        # self.mesh_side_length_label = FixedText('Mesh side length',
        #                                           tip="This is the side length (in pixels) of the sliding window used to detect the network.\nHigh values are faster but less precise.\nWhen too high, straight vertical or horizontal lines appear in the detected network.",
        #                                           night_mode=self.parent().po.all['night_mode'])
        # self.mesh_step_length = Spinbox(min=1, max=100, val=self.parent().po.vars['network_mesh_step_length'], decimals=0,
        #                                   night_mode=self.parent().po.all['night_mode'])
        # self.mesh_step_length_label = FixedText('Mesh step length',
        #                                           tip="This is the distance (in pixels) travelled by the sliding window\n(used to detect the network) at each stage.\nHigh values are faster but less precise.",
        #                                           night_mode=self.parent().po.all['night_mode'])

        self.network_layout.addWidget(self.network_detection_threshold, 0, 0)
        self.network_layout.addWidget(self.network_detection_threshold_label, 0, 1)
        # self.network_layout.addWidget(self.mesh_side_length, 1, 0)
        # self.network_layout.addWidget(self.mesh_side_length_label, 1, 1)
        # self.network_layout.addWidget(self.mesh_step_length, 2, 0)
        # self.network_layout.addWidget(self.mesh_step_length_label, 2, 1)

        self.network_widget.setLayout(self.network_layout)
        self.left_col_layout.addWidget(self.network_widget)


        # self.layout.addWidget(self.oscillation_period_widget, curr_row_1st_col, 1)
        # curr_row_1st_col + 1


        # From here start the 2nd column of boxes in the advanced parameters window
        # vertspaceItem = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding,
        #                                       QtWidgets.QSizePolicy.Maximum)
        # self.layout.addItem(vertspaceItem, 0, 3, 10, 1)
        # curr_row_2nd_col = 3


        # I/ First box: Scales
        # I/A/ Title

        self.right_scroll_table = QtWidgets.QScrollArea()  # QTableWidget()  # Scroll Area which contains the widgets, set as the centralWidget
        self.right_scroll_table.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.right_scroll_table.setMinimumHeight(150)#self.parent().im_max_height - 100
        self.right_scroll_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.right_scroll_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.right_scroll_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)


        self.scale_box_label = FixedText('Spatio-temporal scaling:', tip="",
                                         night_mode=self.parent().po.all['night_mode'])
        self.right_col_layout.addWidget(self.scale_box_label)
        # self.layout.addWidget(self.scale_box_label, curr_row_2nd_col, 4)
        # curr_row_2nd_col += 1
        # I/B/ Create the box
        self.scale_box_layout = QtWidgets.QGridLayout()
        self.scale_box_widget = QtWidgets.QWidget()
        self.scale_box_widget.setStyleSheet(boxstylesheet)
        # I/C/ Create widgets

        self.extract_time = Checkbox(self.parent().po.all['extract_time_interval'])
        self.extract_time.clicked.connect(self.extract_time_is_clicked)

        self.time_step = Spinbox(min=0, max=100000, val=self.parent().po.vars['time_step'], decimals=3,
                                 night_mode=self.parent().po.all['night_mode'])
        self.time_step.setFixedWidth(60)
        if self.parent().po.all['extract_time_interval']:
            self.time_step.setVisible(False)
            self.time_step_label = FixedText('Automatically extract time interval between images',
                                         tip="Uses the exif data of the images (if available), to extract these intervals\nOtherwise, default time interval is 1 min",
                                         night_mode=self.parent().po.all['night_mode'])
        else:
            self.time_step_label = FixedText('Set the time interval between images',
                                         tip="In minutes",
                                         night_mode=self.parent().po.all['night_mode'])
        # self.overwrite_cellects_data = Checkbox(self.parent().po.all['overwrite_cellects_data'],
        #                                night_mode=self.parent().po.all['night_mode'])
        # self.overwrite_cellects_data_label = FixedText('Do overwrite cellects data',
        #                                       tip="The file Data to run Cellects quickly.pkl allow to run\na complete analysis from the first and the video anaysis window",
        #                                       night_mode=self.parent().po.all['night_mode'])
        self.pixels_to_mm = Checkbox(self.parent().po.vars['output_in_mm'])
        self.pixels_to_mm_label = FixedText('Convert areas and distances from pixels to mm',
                                            tip="Check if you want output variables to be in mm\nUncheck if you want output variables to be in pixels",
                                            night_mode=self.parent().po.all['night_mode'])
        # I/D/ Arrange widgets in the box
        self.scale_box_layout.addWidget(self.extract_time, 0, 0)
        self.scale_box_layout.addWidget(self.time_step_label, 0, 1)
        self.scale_box_layout.addWidget(self.time_step, 0, 2)
        # self.scale_box_layout.addWidget(self.overwrite_cellects_data, 1, 0)
        # self.scale_box_layout.addWidget(self.overwrite_cellects_data_label, 1, 1)
        self.scale_box_layout.addWidget(self.pixels_to_mm, 2, 0)
        self.scale_box_layout.addWidget(self.pixels_to_mm_label, 2, 1)
        self.scale_box_widget.setLayout(self.scale_box_layout)
        self.right_col_layout.addWidget(self.scale_box_widget)
        # self.layout.addWidget(self.scale_box_widget, curr_row_2nd_col, 4, 2, 2)
        # curr_row_2nd_col += 3

        # IV/ Fourth box: Computer resources
        # IV/A/ Title
        self.resources_label = FixedText('Computer resources:', tip="",
                                            night_mode=self.parent().po.all['night_mode'])
        self.right_col_layout.addWidget(self.resources_label)
        # self.layout.addWidget(self.resources_label, curr_row_2nd_col, 4)
        # curr_row_2nd_col += 1

        # IV/B/ Create the box
        self.resources_box_layout = QtWidgets.QGridLayout()
        self.resources_box_widget = QtWidgets.QWidget()
        self.resources_box_widget.setStyleSheet(boxstylesheet)

        # IV/C/ Create widgets
        self.do_multiprocessing = Checkbox(self.parent().po.all['do_multiprocessing'])
        self.do_multiprocessing_label = FixedText('Run analysis in parallel', tip="Allow the use of more than one core of the computer processor", night_mode=self.parent().po.all['night_mode'])
        self.do_multiprocessing.stateChanged.connect(self.do_multiprocessing_is_clicked)
        self.max_core_nb = Spinbox(min=0, max=256, val=self.parent().po.all['cores'],
                                   night_mode=self.parent().po.all['night_mode'])
        self.max_core_nb_label = FixedText('Proc max core number', night_mode=self.parent().po.all['night_mode'])
        # self.max_core_nb.setVisible(self.parent().po.all['do_multiprocessing'])
        # self.max_core_nb_label.setVisible(self.parent().po.all['do_multiprocessing'])

        self.min_memory_left = Spinbox(min=0, max=1024, val=self.parent().po.vars['min_ram_free'], decimals=1,
                                       night_mode=self.parent().po.all['night_mode'])
        self.min_memory_left_label = FixedText('Minimal RAM let free (Go)', night_mode=self.parent().po.all['night_mode'])

        self.lose_accuracy_to_save_memory = Checkbox(self.parent().po.vars['lose_accuracy_to_save_memory'])
        self.lose_accuracy_to_save_memory_label = FixedText('Lose accuracy to save RAM',
                                                  tip="Use 8 bits instead of 64 to study each pixel",
                                                  night_mode=self.parent().po.all['night_mode'])

        # IV/D/ Arrange widgets in the box
        self.resources_box_layout.addWidget(self.do_multiprocessing, 0, 0)
        self.resources_box_layout.addWidget(self.do_multiprocessing_label, 0, 1)
        self.resources_box_layout.addWidget(self.max_core_nb, 1, 0)
        self.resources_box_layout.addWidget(self.max_core_nb_label, 1, 1)
        self.resources_box_layout.addWidget(self.min_memory_left, 2, 0)
        self.resources_box_layout.addWidget(self.min_memory_left_label, 2, 1)
        self.resources_box_layout.addWidget(self.lose_accuracy_to_save_memory, 3, 0)
        self.resources_box_layout.addWidget(self.lose_accuracy_to_save_memory_label, 3, 1)
        self.resources_box_widget.setLayout(self.resources_box_layout)
        self.right_col_layout.addWidget(self.resources_box_widget)
        # self.layout.addWidget(self.resources_box_widget, curr_row_2nd_col, 4, 2, 2)
        # curr_row_2nd_col += 3

        # V/ Fifth box: Video saving
        # V/A/ Title
        self.video_saving_label = FixedText('Video saving:', tip="",
                                         night_mode=self.parent().po.all['night_mode'])
        self.right_col_layout.addWidget(self.video_saving_label)
        # self.layout.addWidget(self.video_saving_label, curr_row_2nd_col, 4)
        # curr_row_2nd_col += 1
        # V/B/ Create the box
        self.video_saving_layout = QtWidgets.QGridLayout()
        self.video_saving_widget = QtWidgets.QWidget()
        self.video_saving_widget.setStyleSheet(boxstylesheet)

        # V/C/ Create widgets
        self.video_fps = Spinbox(min=0, max=10000, val=self.parent().po.vars['video_fps'], decimals=2,
                                 night_mode=self.parent().po.all['night_mode'])
        self.video_fps_label = FixedText('Video fps', night_mode=self.parent().po.all['night_mode'])
        # self.overwrite_unaltered_videos = Checkbox(self.parent().po.all['overwrite_unaltered_videos'])
        # self.overwrite_unaltered_videos_label = FixedText('Do overwrite unaltered videos (.npy)', tip="If the analysis fails because of a bad detection of arenas\nChecking this may resolve failures during image analysis", night_mode=self.parent().po.all['night_mode'])
        self.keep_unaltered_videos = Checkbox(self.parent().po.vars['keep_unaltered_videos'])
        self.keep_unaltered_videos_label = FixedText('Keep unaltered videos', tip="Unaltered videos (.npy) takes a lot of hard drive space\nUsers should only keep these videos\nif they plan to redo the analysis soon and faster", night_mode=self.parent().po.all['night_mode'])
        self.save_processed_videos = Checkbox(self.parent().po.vars['save_processed_videos'])
        self.save_processed_videos_label = FixedText('Save processed videos', tip="Processed videos allow to check analysis accuracy\nThey do not take a lot of space", night_mode=self.parent().po.all['night_mode'])

        # V/D/ Arrange widgets in the box
        curr_box_row = 0
        self.video_saving_layout.addWidget(self.video_fps, curr_box_row, 0)
        self.video_saving_layout.addWidget(self.video_fps_label, curr_box_row, 1)
        curr_box_row += 1
        # self.video_saving_layout.addWidget(self.overwrite_unaltered_videos, curr_box_row, 0)
        # self.video_saving_layout.addWidget(self.overwrite_unaltered_videos_label, curr_box_row, 1)
        # curr_box_row += 1
        self.video_saving_layout.addWidget(self.keep_unaltered_videos, curr_box_row, 0)
        self.video_saving_layout.addWidget(self.keep_unaltered_videos_label, curr_box_row, 1)
        curr_box_row += 1
        self.video_saving_layout.addWidget(self.save_processed_videos, curr_box_row, 0)
        self.video_saving_layout.addWidget(self.save_processed_videos_label, curr_box_row, 1)
        curr_box_row += 1

        self.video_saving_widget.setLayout(self.video_saving_layout)
        self.right_col_layout.addWidget(self.video_saving_widget)
        # self.layout.addWidget(self.video_saving_widget, curr_row_2nd_col, 4, 2, 2)
        # curr_row_2nd_col += 2

        # VII/ Seventh box: csc
        # VII/A/ Title
        # self.video_csc_label = FixedText('Color space combination for video analysis:', tip="",
        #                                     night_mode=self.parent().po.all['night_mode'])
        # self.right_col_layout.addWidget(self.video_csc_label)
        # self.layout.addWidget(self.video_csc_label, curr_row_2nd_col, 4)
        # curr_row_2nd_col += 1

        # VII/C/ Create widgets
        self.generate_csc_editing()
        # VII/D/ Arrange widgets in the box
        self.right_col_layout.addWidget(self.edit_widget)
        # self.layout.addWidget(self.edit_widget, curr_row_2nd_col, 4, 2, 2)
        # curr_row_2nd_col += 3

        # VIII/ Finalize layout and add the night mode option and the ok button
        self.left_col_layout.addItem(self.vertical_space)
        self.right_col_layout.addItem(self.vertical_space)
        self.left_col_widget.setLayout(self.left_col_layout)


        self.right_col_widget.setLayout(self.right_col_layout)
        self.central_layout = QtWidgets.QHBoxLayout()
        self.central_layout.addItem(self.horizontal_space)
        #self.central_layout.addWidget(self.left_col_widget)

        self.left_scroll_table.setWidget(self.left_col_widget)
        self.left_scroll_table.setWidgetResizable(True)
        self.central_layout.addWidget(self.left_scroll_table)


        self.central_layout.addItem(self.horizontal_space)
        self.right_scroll_table.setWidget(self.right_col_widget)
        self.right_scroll_table.setWidgetResizable(True)
        self.central_layout.addWidget(self.right_scroll_table)
        # self.central_layout.addWidget(self.right_col_widget)
        self.central_layout.addItem(self.horizontal_space)
        self.central_widget = QtWidgets.QWidget()
        self.central_widget.setLayout(self.central_layout)
        self.layout.addWidget(self.central_widget)
        self.layout.addItem(self.vertical_space)
        # Last row
        self.last_row_layout = QtWidgets.QHBoxLayout()
        self.last_row_widget = QtWidgets.QWidget()
        self.night_mode_cb = Checkbox(self.parent().po.all['night_mode'])
        self.night_mode_cb.clicked.connect(self.night_mode_is_clicked)
        self.night_mode_label = FixedText('Night mode', night_mode=self.parent().po.all['night_mode'])
        self.reset_all_settings = PButton("Reset all settings", night_mode=self.parent().po.all['night_mode'])
        self.reset_all_settings.clicked.connect(self.reset_all_settings_is_clicked)
        self.message = FixedText('', night_mode=self.parent().po.all['night_mode'])
        self.cancel = PButton('Cancel', night_mode=self.parent().po.all['night_mode'])
        self.cancel.clicked.connect(self.cancel_is_clicked)
        self.ok = PButton('Ok', night_mode=self.parent().po.all['night_mode'])
        self.ok.clicked.connect(self.ok_is_clicked)
        self.last_row_layout.addWidget(self.night_mode_cb)
        self.last_row_layout.addWidget(self.night_mode_label)
        self.last_row_layout.addWidget(self.reset_all_settings)
        self.last_row_layout.addItem(self.horizontal_space)
        self.last_row_layout.addWidget(self.message)
        self.last_row_layout.addWidget(self.cancel)
        self.last_row_layout.addWidget(self.ok)
        self.last_row_widget.setLayout(self.last_row_layout)
        self.layout.addWidget(self.last_row_widget)

        self.setLayout(self.layout)

    def display_conditionally_visible_widgets(self):
        self.max_core_nb.setVisible(self.parent().po.all['do_multiprocessing'])
        self.max_core_nb_label.setVisible(self.parent().po.all['do_multiprocessing'])
        self.first_move_threshold.setVisible(not self.parent().po.all['automatic_size_thresholding'])
        self.first_move_threshold_label.setVisible(not self.parent().po.all['automatic_size_thresholding'])
        connect_distant_shape = self.parent().po.all['connect_distant_shape_during_segmentation']
        do_use_max_size = self.parent().po.vars['max_size_for_connection'] is not None and connect_distant_shape
        do_use_min_size = self.parent().po.vars['min_size_for_connection'] is not None and connect_distant_shape
        self.detection_range_factor.setVisible(connect_distant_shape)
        self.detection_range_factor_label.setVisible(connect_distant_shape)
        self.use_max_size.setVisible(connect_distant_shape)
        self.use_min_size.setVisible(connect_distant_shape)
        self.use_max_size_label.setVisible(connect_distant_shape)
        self.use_min_size_label.setVisible(connect_distant_shape)

        self.max_size_for_connection.setVisible(do_use_max_size)
        self.max_size_for_connection_label.setVisible(do_use_max_size)
        self.min_size_for_connection.setVisible(do_use_min_size)
        self.min_size_for_connection_label.setVisible(do_use_min_size)
        self.display_more_than_two_colors_option()

        if self.parent().po.vars['convert_for_motion'] is not None:
            self.update_csc_editing_display()
        else:
            self.row1[0].setCurrentIndex(4)
            self.row1[3].setValue(1)
            self.row21[0].setCurrentIndex(0)
            self.row21[3].setValue(0)

    def subtract_background_check(self):
        self.parent().po.motion = None
        if self.subtract_background.isChecked():
            self.parent().po.first_exp_ready_to_run = False

    def prevent_fast_growth_near_periphery_check(self):
        checked_status = self.prevent_fast_growth_near_periphery.isChecked()
        self.periphery_width.setVisible(checked_status)
        self.periphery_width_label.setVisible(checked_status)
        self.max_periphery_growth.setVisible(checked_status)
        self.max_periphery_growth_label.setVisible(checked_status)

    def do_automatic_size_thresholding_changed(self):
        """ Triggered when do_automatic_size_thresholding check status changes"""
        self.first_move_threshold.setVisible(not self.do_automatic_size_thresholding.isChecked())
        self.first_move_threshold_label.setVisible(not self.do_automatic_size_thresholding.isChecked())

    def extract_time_is_clicked(self):
        self.time_step.setVisible(not self.extract_time.isChecked())
        if self.extract_time.isChecked():
            self.time_step_label.setText("Automatically extract time interval between images")
            self.time_step_label.setToolTip("Uses the exif data of the images (if available), to extract these intervals\nOtherwise, default time interval is 1 min")
        else:
            self.time_step_label.setText("Set the time interval between images")
            self.time_step_label.setToolTip("In minutes")

    def do_multiprocessing_is_clicked(self):
        self.max_core_nb.setVisible(self.do_multiprocessing.isChecked())
        self.max_core_nb_label.setVisible(self.do_multiprocessing.isChecked())

    # def all_specimens_have_same_direction_changed(self):
    #     """ Triggered when all_specimens_have_same_direction check status changes"""
    #     self.parent().po.all['all_specimens_have_same_direction'] = self.all_specimens_have_same_direction.isChecked()

    def do_distant_shape_int_changed(self):
        """ Triggered when connect_distant_shape_during_segmentation check status changes"""
        do_distant_shape_int = self.connect_distant_shape_during_segmentation.isChecked()
        self.detection_range_factor.setVisible(do_distant_shape_int)
        if do_distant_shape_int:
            self.detection_range_factor.setValue(2)
        self.detection_range_factor_label.setVisible(do_distant_shape_int)
        self.use_max_size.setVisible(do_distant_shape_int)
        self.use_min_size.setVisible(do_distant_shape_int)
        self.use_max_size_label.setVisible(do_distant_shape_int)
        self.use_min_size_label.setVisible(do_distant_shape_int)
        do_use_max_size = do_distant_shape_int and self.use_max_size.isChecked()
        self.max_size_for_connection.setVisible(do_use_max_size)
        self.max_size_for_connection_label.setVisible(do_use_max_size)
        do_use_min_size = do_distant_shape_int and self.use_min_size.isChecked()
        self.min_size_for_connection.setVisible(do_use_min_size)
        self.min_size_for_connection_label.setVisible(do_use_min_size)

    def use_max_size_changed(self):
        """ Triggered when use_max_size check status changes"""
        do_use_max_size = self.use_max_size.isChecked()
        self.max_size_for_connection.setVisible(do_use_max_size)
        self.max_size_for_connection_label.setVisible(do_use_max_size)
        if do_use_max_size:
            self.max_size_for_connection.setValue(300)

    def use_min_size_changed(self):
        """ Triggered when use_min_size check status changes"""
        do_use_min_size = self.use_min_size.isChecked()
        self.min_size_for_connection.setVisible(do_use_min_size)
        self.min_size_for_connection_label.setVisible(do_use_min_size)
        if do_use_min_size:
            self.min_size_for_connection.setValue(30)

    def generate_csc_editing(self):
        # self.edit_layout = QtWidgets.QGridLayout()
        self.edit_widget = QtWidgets.QWidget()
        # self.edit_widget.setVisible(False)
        # self.edit_widget.setStyleSheet(boxstylesheet)
        self.edit_layout = QtWidgets.QVBoxLayout()

        # self.csc_scroll_table = QtWidgets.QScrollArea()  # QTableWidget()  # Scroll Area which contains the widgets, set as the centralWidget
        # # self.csc_scroll_table.setVisible(False)
        # # self.csc_scroll_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # self.csc_scroll_table.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        # self.csc_scroll_table.setMinimumHeight(150)#self.parent().im_max_height - 100
        # # self.csc_scroll_table.setMinimumWidth(300)
        # self.csc_scroll_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        # self.csc_scroll_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # self.csc_scroll_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.csc_table_widget = QtWidgets.QWidget()
        self.csc_table_layout = QtWidgets.QVBoxLayout()

        # 2) Titles
        # self.edit_labels_widget = QtWidgets.QWidget()
        # self.edit_labels_widget.setFixedHeight(50)
        # self.edit_labels_layout = QtWidgets.QHBoxLayout()
        # self.space_label = FixedText('Space', align='c',
        #                             tip="Color spaces are transformations of the original BGR (Blue Green Red) image\nInstead of defining an image by 3 colors,\n they transform it into 3 different visual properties\n  - hsv: hue (color), saturation, value (lightness)\n  - hls: hue (color), lightness, saturation\n  - lab: Lightness, Red/Green, Blue/Yellow\n  - luv and yuv: l and y are Lightness, u and v are related to colors\n",
        #                             night_mode=self.parent().po.all['night_mode'])
        # self.c1 = FixedText('  C1', align='c', tip="Increase if it increase cell detection", night_mode=self.parent().po.all['night_mode'])
        # self.c2 = FixedText('  C2', align='c', tip="Increase if it increase cell detection", night_mode=self.parent().po.all['night_mode'])
        # self.c3 = FixedText('  C3', align='c', tip="Increase if it increase cell detection", night_mode=self.parent().po.all['night_mode'])
        #
        # self.edit_labels_layout.addWidget(self.space_label)
        # self.edit_labels_layout.addWidget(self.c1)
        # self.edit_labels_layout.addWidget(self.c2)
        # self.edit_labels_layout.addWidget(self.c3)
        # self.edit_labels_layout.addItem(self.horizontal_space)
        # self.edit_labels_widget.setLayout(self.edit_labels_layout)
        # # self.edit_layout.addWidget(self.edit_labels_widget)
        # self.csc_table_layout.addWidget(self.edit_labels_widget)
        self.video_csc_label = FixedText('Color space combination for video analysis:', tip="",
                                         night_mode=self.parent().po.all['night_mode'])
        self.video_csc_label.setFixedHeight(30)
        self.csc_table_layout.addWidget(self.video_csc_label)

        self.both_csc_widget = QtWidgets.QWidget()
        self.both_csc_layout = QtWidgets.QHBoxLayout()
        # 3) First CSC
        self.first_csc_widget = QtWidgets.QWidget()
        self.first_csc_layout = QtWidgets.QGridLayout()
        self.row1 = self.one_csc_editing()
        self.row1[4].clicked.connect(self.display_row2)
        self.row2 = self.one_csc_editing()
        self.row2[4].clicked.connect(self.display_row3)
        self.row3 = self.one_csc_editing()# Second CSC
        self.logical_operator_between_combination_result = Combobox(["None", "Or", "And", "Xor"],
                                                                    night_mode=self.parent().po.all['night_mode'])
        self.logical_operator_between_combination_result.setCurrentText(self.parent().po.vars['convert_for_motion']['logical'])
        self.logical_operator_between_combination_result.currentTextChanged.connect(self.logical_op_changed)
        self.logical_operator_between_combination_result.setFixedWidth(100)
        # self.logical_operator_between_combination_result.cha
        self.logical_operator_label = FixedText("Logical operator", halign='c', tip="Between selected color space combinations",
                                                night_mode=self.parent().po.all['night_mode'])

        self.row21 = self.one_csc_editing()
        self.row21[4].clicked.connect(self.display_row22)
        self.row22 = self.one_csc_editing()
        self.row22[4].clicked.connect(self.display_row23)
        self.row23 = self.one_csc_editing()


        for i in range(5):
            self.first_csc_layout.addWidget(self.row1[i], 0, i, 1, 1)
            self.first_csc_layout.addWidget(self.row2[i], 1, i, 1, 1)
            self.first_csc_layout.addWidget(self.row3[i], 2, i, 1, 1)
            self.row1[i].setVisible(False)
            self.row2[i].setVisible(False)
            self.row3[i].setVisible(False)
        self.first_csc_layout.setHorizontalSpacing(0)
        self.first_csc_layout.addItem(self.horizontal_space, 0, 5, 3, 1)
        self.first_csc_widget.setLayout(self.first_csc_layout)
        self.both_csc_layout.addWidget(self.first_csc_widget)
        # self.csc_table_layout.addWidget(self.first_csc_widget)
        # self.edit_layout.addWidget(self.first_csc_widget)


        # 5) Second CSC
        self.second_csc_widget = QtWidgets.QWidget()
        self.second_csc_layout = QtWidgets.QGridLayout()
        for i in range(5):
            self.second_csc_layout.addWidget(self.row21[i], 0, i, 1, 1)
            self.second_csc_layout.addWidget(self.row22[i], 1, i, 1, 1)
            self.second_csc_layout.addWidget(self.row23[i], 2, i, 1, 1)
            self.row21[i].setVisible(False)
            self.row22[i].setVisible(False)
            self.row23[i].setVisible(False)
        self.second_csc_layout.setHorizontalSpacing(0)
        self.second_csc_layout.addItem(self.horizontal_space, 0, 5, 3, 1)
        self.second_csc_widget.setLayout(self.second_csc_layout)
        self.both_csc_layout.addItem(self.horizontal_space)
        self.both_csc_layout.addWidget(self.second_csc_widget)
        self.both_csc_widget.setLayout(self.both_csc_layout)
        self.csc_table_layout.addWidget(self.both_csc_widget)
        # self.csc_table_layout.addWidget(self.second_csc_widget)


        # 4) logical_operator
        self.logical_op_widget = QtWidgets.QWidget()
        self.logical_op_widget.setFixedHeight(30)
        self.logical_op_layout = QtWidgets.QHBoxLayout()
        self.logical_op_layout.addWidget(self.logical_operator_label)
        self.logical_op_layout.addWidget(self.logical_operator_between_combination_result)
        self.logical_op_layout.addItem(self.horizontal_space)
        self.logical_operator_between_combination_result.setVisible(False)
        self.logical_operator_label.setVisible(False)
        self.logical_op_widget.setLayout(self.logical_op_layout)
        self.logical_op_widget.setFixedHeight(50)
        self.csc_table_layout.addWidget(self.logical_op_widget)
        # self.edit_layout.addWidget(self.logical_op_widget)

        # 6) Open the more_than_2_colors row layout
        self.more_than_2_colors_widget = QtWidgets.QWidget()
        self.more_than_2_colors_layout = QtWidgets.QHBoxLayout()
        self.more_than_two_colors = Checkbox(self.parent().po.all["more_than_two_colors"])
        self.more_than_two_colors.setStyleSheet("margin-left:0%; margin-right:-10%;")
        self.more_than_two_colors.stateChanged.connect(self.display_more_than_two_colors_option)

        self.more_than_two_colors_label = FixedText("Heterogeneous back",
                                                    tip="The program will split the image into categories", night_mode=self.parent().po.all['night_mode'])
        self.more_than_two_colors_label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # self.more_than_two_colors_label.setFixedWidth(300)
        self.more_than_two_colors_label.setAlignment(QtCore.Qt.AlignLeft)
        self.distinct_colors_number = Spinbox(min=2, max=5, val=self.parent().po.vars["color_number"], night_mode=self.parent().po.all['night_mode'])
        # self.distinct_colors_number.valueChanged.connect(self.distinct_colors_number_changed)
        # self.display_more_than_two_colors_option()
        # self.more_than_two_colors.setVisible(False)
        # self.more_than_two_colors_label.setVisible(False)
        # self.distinct_colors_number.setVisible(False)
        self.more_than_2_colors_layout.addWidget(self.more_than_two_colors)
        self.more_than_2_colors_layout.addWidget(self.more_than_two_colors_label)
        self.more_than_2_colors_layout.addWidget(self.distinct_colors_number)
        self.more_than_2_colors_layout.addItem(self.horizontal_space)
        self.more_than_2_colors_widget.setLayout(self.more_than_2_colors_layout)
        self.more_than_2_colors_widget.setFixedHeight(50)
        self.csc_table_layout.addWidget(self.more_than_2_colors_widget)
        self.csc_table_layout.addItem(self.vertical_space)
        self.csc_table_widget.setLayout(self.csc_table_layout)

        self.edit_layout.addWidget(self.csc_table_widget)
        # self.csc_scroll_table.setWidget(self.csc_table_widget)
        # self.csc_scroll_table.setWidgetResizable(True)
        # self.edit_layout.addWidget(self.csc_scroll_table)

        # self.more_than_2_colors_layout.addWidget(self.more_than_two_colors)
        # self.more_than_2_colors_layout.addWidget(self.more_than_two_colors_label)
        # self.more_than_2_colors_layout.addWidget(self.distinct_colors_number)
        # self.more_than_2_colors_layout.addItem(self.horizontal_space)
        # self.more_than_2_colors_widget.setLayout(self.more_than_2_colors_layout)
        # self.edit_layout.addWidget(self.more_than_2_colors_widget)

        self.edit_widget.setLayout(self.edit_layout)

    def one_csc_editing(self):
        widget_list = []
        widget_list.insert(0, Combobox(["None", "bgr", "hsv", "hls", "lab", "luv", "yuv"],
                                       night_mode=self.parent().po.all['night_mode']))
        widget_list[0].setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        widget_list[0].setFixedWidth(100)
        for i in [1, 2, 3]:
            widget_list.insert(i, Spinbox(min=-126, max=126, val=0, night_mode=self.parent().po.all['night_mode']))
            widget_list[i].setFixedWidth(45)
        widget_list.insert(i + 1, PButton("+", night_mode=self.parent().po.all['night_mode']))

        return widget_list

    def logical_op_changed(self):
        # show = self.logical_operator_between_combination_result.currentText() != 'None'
        if self.logical_operator_between_combination_result.currentText() == 'None':
            self.row21[0].setVisible(False)
            self.row21[0].setCurrentIndex(0)
            for i1 in [1, 2, 3]:
                self.row21[i1].setVisible(False)
                self.row21[i1].setValue(0)
            self.row21[i1 + 1].setVisible(False)

            self.row22[0].setVisible(False)
            self.row22[0].setCurrentIndex(0)
            for i1 in [1, 2, 3]:
                self.row22[i1].setVisible(False)
                self.row22[i1].setValue(0)
            self.row22[i1 + 1].setVisible(False)

            self.row23[0].setVisible(False)
            self.row23[0].setCurrentIndex(0)
            for i1 in [1, 2, 3]:
                self.row23[i1].setVisible(False)
                self.row23[i1].setValue(0)
            self.row23[i1 + 1].setVisible(False)
        else:
            self.row21[0].setVisible(True)
            for i1 in [1, 2, 3]:
                self.row21[i1].setVisible(True)
            self.row21[i1 + 1].setVisible(True)

    def display_logical_operator(self):
        self.logical_operator_between_combination_result.setVisible(True)
        self.logical_operator_label.setVisible(True)

    def display_row2(self):
        self.row1[4].setVisible(False)
        for i in range(5):
            self.row2[i].setVisible(True)
        self.display_logical_operator()

    def display_row3(self):
        self.row2[4].setVisible(False)
        for i in range(4):
            self.row3[i].setVisible(True)
        self.display_logical_operator()

    def display_row22(self):
        self.row21[4].setVisible(False)
        for i in range(5):
            self.row22[i].setVisible(True)
        self.display_logical_operator()

    def display_row23(self):
        self.row22[4].setVisible(False)
        for i in range(4):
            self.row23[i].setVisible(True)
        self.display_logical_operator()

    def update_csc_editing_display(self):
        c_space_order = ["None", "bgr", "hsv", "hls", "lab", "luv", "yuv"]
        remaining_c_spaces = []
        row_number1 = 0
        row_number2 = 0
        for i, (k, v) in enumerate(self.parent().po.vars['convert_for_motion'].items()):
            if k != "logical":
                if k[-1] != "2":
                    if row_number1 == 0:
                        row_to_change = self.row1
                    elif row_number1 == 1:
                        row_to_change = self.row2
                    elif row_number1 == 2:
                        row_to_change = self.row3
                    else:
                        remaining_c_spaces.append(k + " " + str(v))
                    row_number1 += 1
                    current_row_number = row_number1
                else:
                    if row_number2 == 0:
                        row_to_change = self.row21
                    elif row_number2 == 1:
                        row_to_change = self.row22
                    elif row_number2 == 2:
                        row_to_change = self.row23
                    else:
                        remaining_c_spaces.append(k + " " + str(v))
                    row_number2 += 1
                    current_row_number = row_number2
                    k = k[:-1]
                if current_row_number <= 3:
                    row_to_change[0].setCurrentIndex(np.nonzero(np.isin(c_space_order, k))[0][0])
                    row_to_change[0].setVisible(True)
                    for i1, i2 in zip([1, 2, 3], [0, 1, 2]):
                        row_to_change[i1].setValue(v[i2])
                        row_to_change[i1].setVisible(True)
                    if current_row_number < 3:
                        row_to_change[i1 + 1].setVisible(True)

        # If not all color space combinations are filled, put None and 0 in boxes
        if row_number1 < 3:
            self.row3[0].setVisible(False)
            self.row3[0].setCurrentIndex(0)
            for i1 in [1, 2, 3]:
                self.row3[i1].setVisible(False)
                self.row3[i1].setValue(0)
            if row_number1 < 2:
                self.row2[0].setVisible(False)
                self.row2[0].setCurrentIndex(0)
                for i1 in [1, 2, 3]:
                    self.row2[i1].setVisible(False)
                    self.row2[i1].setValue(0)
                self.row2[i1 + 1].setVisible(False)

        self.row1[4].setVisible(row_number1 == 1)
        self.row2[4].setVisible(row_number1 == 2)
        self.row21[4].setVisible(row_number2 == 1)
        self.row22[4].setVisible(row_number2 == 2)
        if row_number2 > 0:
            self.logical_operator_between_combination_result.setCurrentText(self.parent().po.vars['convert_for_motion']['logical'])
        if row_number2 == 0:
            self.logical_operator_between_combination_result.setCurrentText('None')
            self.logical_operator_between_combination_result.setVisible(False)
            self.logical_operator_label.setVisible(False)
            self.row21[0].setVisible(False)
            self.row21[0].setCurrentIndex(0)
            for i1 in [1, 2, 3]:
                self.row21[i1].setVisible(False)
                self.row21[i1].setValue(0)
            self.row21[i1 + 1].setVisible(False)

        self.logical_operator_between_combination_result.setVisible((row_number2 > 0))
        self.logical_operator_label.setVisible((row_number2 > 0))

        if row_number2 < 3:
            self.row23[0].setVisible(False)
            self.row23[0].setCurrentIndex(0)
            for i1 in [1, 2, 3]:
                self.row23[i1].setVisible(False)
                self.row23[i1].setValue(0)
            self.row23[i1 + 1].setVisible(False)
            self.row22[4].setVisible(False)
            if row_number2 < 2:
                self.row22[0].setVisible(False)
                self.row22[0].setCurrentIndex(0)
                for i1 in [1, 2, 3]:
                    self.row22[i1].setVisible(False)
                    self.row22[i1].setValue(0)
                self.row22[i1 + 1].setVisible(False)

    def save_user_defined_csc(self):
        self.parent().po.vars['convert_for_motion'] = {}
        spaces = np.array((self.row1[0].currentText(), self.row2[0].currentText(), self.row3[0].currentText()))
        channels = np.array(
            ((self.row1[1].value(), self.row1[2].value(), self.row1[3].value()),
             (self.row2[1].value(), self.row2[2].value(), self.row2[3].value()),
             (self.row3[1].value(), self.row3[2].value(), self.row3[3].value()),
             (self.row21[1].value(), self.row21[2].value(), self.row21[3].value()),
             (self.row22[1].value(), self.row22[2].value(), self.row22[3].value()),
             (self.row23[1].value(), self.row23[2].value(), self.row23[3].value())),
            dtype=np.int8)
        if self.logical_operator_between_combination_result.currentText() != 'None':
            spaces = np.concatenate((spaces, np.array((
                        self.row21[0].currentText() + "2", self.row22[0].currentText() + "2",
                        self.row23[0].currentText() + "2"))))
            channels = np.concatenate((channels, np.array(((self.row21[1].value(), self.row21[2].value(), self.row21[3].value()),
             (self.row22[1].value(), self.row22[2].value(), self.row22[3].value()),
             (self.row23[1].value(), self.row23[2].value(), self.row23[3].value())),
             dtype=np.int8)))
            self.parent().po.vars['convert_for_motion']['logical'] = self.logical_operator_between_combination_result.currentText()
        else:
            self.parent().po.vars['convert_for_motion']['logical'] = 'None'
        if not np.all(spaces == "None"):
            for i, space in enumerate(spaces):
                if space != "None" and space != "None2":
                    self.parent().po.vars['convert_for_motion'][space] = channels[i, :]
        if len(self.parent().po.vars['convert_for_motion']) == 1 or channels.sum() == 0:
            self.csc_dict_is_empty = True
        else:
            self.csc_dict_is_empty = False

        self.parent().po.all["more_than_two_colors"] = self.more_than_two_colors.isChecked()
        if self.more_than_two_colors.isChecked():
            self.parent().po.vars["color_number"] = int(self.distinct_colors_number.value())
            self.parent().videoanalysiswindow.select_option.setVisible(True)
            self.parent().videoanalysiswindow.select_option_label.setVisible(True)

    def display_more_than_two_colors_option(self):
        if self.more_than_two_colors.isChecked():
            self.distinct_colors_number.setVisible(True)
            self.more_than_two_colors_label.setText("How many distinct colors?")
            self.distinct_colors_number.setValue(3)
        else:
            self.more_than_two_colors_label.setText("Heterogeneous background")
            self.distinct_colors_number.setVisible(False)
            self.distinct_colors_number.setValue(2)


            # self.parent().po.vars["color_number"] = 2

    # def distinct_colors_number_changed(self):
    #     self.parent().po.vars["color_number"] = int(self.distinct_colors_number.value())

    def night_mode_is_clicked(self):
        """ Triggered when night_mode_cb check status changes"""
        self.parent().po.all['night_mode'] = self.night_mode_cb.isChecked()
        self.message.setText('Close and restart Cellects to apply night or light mode')
        self.message.setStyleSheet("color: rgb(230, 145, 18)")

    def reset_all_settings_is_clicked(self):
        if os.path.isfile('Data to run Cellects quickly.pkl'):
            os.remove('Data to run Cellects quickly.pkl')
        if os.path.isfile('PickleRick.pkl'):
            os.remove('PickleRick.pkl')
        if os.path.isfile('PickleRick0.pkl'):
            os.remove('PickleRick0.pkl')
        if os.path.isfile(Path(CELLECTS_DIR.parent / 'PickleRick.pkl')):
            os.remove(Path(CELLECTS_DIR.parent / 'PickleRick.pkl'))
        if os.path.isfile(Path(CELLECTS_DIR.parent / 'PickleRick0.pkl')):
            os.remove(Path(CELLECTS_DIR.parent / 'PickleRick0.pkl'))
        if os.path.isfile(Path(CONFIG_DIR / 'PickleRick1.pkl')):
            os.remove(Path(CONFIG_DIR / 'PickleRick1.pkl'))
        current_dir = os.getcwd()
        os.chdir(CONFIG_DIR)
        DefaultDicts().save_as_pkl(self.parent().po)
        os.chdir(current_dir)
        self.message.setText('Close and restart Cellects to apply the settings reset')
        self.message.setStyleSheet("color: rgb(230, 145, 18)")

    def cancel_is_clicked(self):
        self.automatically_crop.setChecked(self.parent().po.all['automatically_crop'])
        self.subtract_background.setChecked(self.parent().po.vars['subtract_background'])
        self.keep_cell_and_back_for_all_folders.setChecked(self.parent().po.all['keep_cell_and_back_for_all_folders'])
        self.correct_errors_around_initial.setChecked(self.parent().po.vars['correct_errors_around_initial'])
        self.prevent_fast_growth_near_periphery.setChecked(self.parent().po.vars['prevent_fast_growth_near_periphery'])
        self.periphery_width.setValue(self.parent().po.vars['periphery_width'])
        self.max_periphery_growth.setValue(self.parent().po.vars['max_periphery_growth'])


        self.first_move_threshold.setValue(self.parent().po.all['first_move_threshold_in_mm²'])
        self.pixels_to_mm.setChecked(self.parent().po.vars['output_in_mm'])
        self.do_automatic_size_thresholding.setChecked(self.parent().po.all['automatic_size_thresholding'])
        self.appearing_selection.setCurrentText(self.parent().po.vars['appearance_detection_method'])
        self.oscillation_period.setValue(self.parent().po.vars['expected_oscillation_period'])
        self.minimal_oscillating_cluster_size.setValue(self.parent().po.vars['minimal_oscillating_cluster_size'])

        self.network_detection_threshold.setValue(self.parent().po.vars['network_detection_threshold'])

        self.do_multiprocessing.setChecked(self.parent().po.all['do_multiprocessing'])
        self.max_core_nb.setValue(self.parent().po.all['cores'])
        self.min_memory_left.setValue(self.parent().po.vars['min_ram_free'])
        self.lose_accuracy_to_save_memory.setChecked(self.parent().po.vars['lose_accuracy_to_save_memory'])
        self.video_fps.setValue(self.parent().po.vars['video_fps'])
        self.keep_unaltered_videos.setChecked(self.parent().po.vars['keep_unaltered_videos'])
        self.save_processed_videos.setChecked(self.parent().po.vars['save_processed_videos'])
        self.time_step.setValue(self.parent().po.vars['time_step'])
        # self.parent().po.all['overwrite_cellects_data'] = self.overwrite_cellects_data.isChecked()

        self.connect_distant_shape_during_segmentation.setChecked(self.parent().po.all['connect_distant_shape_during_segmentation'])
        do_use_max_size = self.parent().po.vars['max_size_for_connection'] is not None and self.parent().po.all['connect_distant_shape_during_segmentation']
        do_use_min_size = self.parent().po.vars['min_size_for_connection'] is not None and self.parent().po.all['connect_distant_shape_during_segmentation']
        self.use_max_size.setChecked(do_use_max_size)
        self.use_min_size.setChecked(do_use_min_size)
        if do_use_max_size:
            self.max_size_for_connection.setValue(self.parent().po.vars['max_size_for_connection'])
        else:
            self.max_size_for_connection.setValue(50)
        if do_use_min_size:
            self.min_size_for_connection.setValue(self.parent().po.vars['min_size_for_connection'])
        else:
            self.min_size_for_connection.setValue(0)

        self.detection_range_factor.setValue(self.parent().po.vars['detection_range_factor'])
        self.all_specimens_have_same_direction.setChecked(self.parent().po.all['all_specimens_have_same_direction'])
        self.update_csc_editing_display()

        if self.parent().last_is_first:
            self.parent().change_widget(0) # FirstWidget
        else:
            self.parent().change_widget(3) # ImageAnalysisWindow ThirdWidget

    def ok_is_clicked(self):
        """ Triggered when ok is clicked, save the directory all_vars.pkl and go back to the previous window"""
        # if self.mesh_side_length.value() <= self.mesh_step_length.value():
        #     self.message.setText('The mesh side has to be inferior to the mesh step')
        #     self.message.setStyleSheet("color: rgb(230, 145, 18)")
        # else:
        self.parent().po.all['automatically_crop'] = self.automatically_crop.isChecked()
        self.parent().po.vars['subtract_background'] = self.subtract_background.isChecked()
        self.parent().po.all['keep_cell_and_back_for_all_folders'] = self.keep_cell_and_back_for_all_folders.isChecked()
        self.parent().po.vars['correct_errors_around_initial'] = self.correct_errors_around_initial.isChecked()
        self.parent().po.vars['prevent_fast_growth_near_periphery'] = self.prevent_fast_growth_near_periphery.isChecked()
        self.parent().po.vars['periphery_width'] = int(self.periphery_width.value())
        self.parent().po.vars['max_periphery_growth'] = int(self.max_periphery_growth.value())

        # if self.parent().po.vars['origin_state'] == "invisible":
        self.parent().po.all['first_move_threshold_in_mm²'] = self.first_move_threshold.value()
        self.parent().po.vars['output_in_mm'] = self.pixels_to_mm.isChecked()
        self.parent().po.all['automatic_size_thresholding'] = self.do_automatic_size_thresholding.isChecked()
        self.parent().po.vars['appearance_detection_method'] = self.appearing_selection.currentText()
        self.parent().po.vars['expected_oscillation_period'] = self.oscillation_period.value()
        self.parent().po.vars['minimal_oscillating_cluster_size'] = int(self.minimal_oscillating_cluster_size.value())

        self.parent().po.vars['network_detection_threshold'] = int(np.round(self.network_detection_threshold.value()))

        self.parent().po.all['do_multiprocessing'] = self.do_multiprocessing.isChecked()
        self.parent().po.all['cores'] = np.uint8(self.max_core_nb.value())
        self.parent().po.vars['min_ram_free'] = self.min_memory_left.value()
        self.parent().po.vars['lose_accuracy_to_save_memory'] = self.lose_accuracy_to_save_memory.isChecked()
        self.parent().po.vars['video_fps'] = float(self.video_fps.value())
        self.parent().po.vars['keep_unaltered_videos'] = self.keep_unaltered_videos.isChecked()
        self.parent().po.vars['save_processed_videos'] = self.save_processed_videos.isChecked()
        self.parent().po.all['extract_time_interval'] = self.extract_time.isChecked()
        self.parent().po.vars['time_step'] = float(self.time_step.value())

        do_distant_shape_int = self.connect_distant_shape_during_segmentation.isChecked()
        self.parent().po.all['connect_distant_shape_during_segmentation'] = do_distant_shape_int
        if do_distant_shape_int:
            self.parent().po.vars['detection_range_factor'] = int(
                np.round(self.detection_range_factor.value()))
        else:
            self.parent().po.vars['detection_range_factor'] = 0
        if self.use_max_size.isChecked():
            self.parent().po.vars['max_size_for_connection'] = int(np.round(self.max_size_for_connection.value()))
        else:
            self.parent().po.vars['max_size_for_connection'] = None
        if self.use_min_size.isChecked():
            self.parent().po.vars['min_size_for_connection'] = int(np.round(self.min_size_for_connection.value()))
        else:
            self.parent().po.vars['min_size_for_connection'] = None

        self.parent().po.all['all_specimens_have_same_direction'] = self.all_specimens_have_same_direction.isChecked()

        previous_csc = deepcopy(self.parent().po.vars['convert_for_motion'])
        self.save_user_defined_csc()
        print(self.parent().po.vars['convert_for_motion'])
        if self.parent().po.first_exp_ready_to_run:
            are_dicts_equal: bool = True
            for key in previous_csc.keys():
                if key != 'logical':
                    are_dicts_equal = are_dicts_equal and np.all(key in self.parent().po.vars['convert_for_motion'] and previous_csc[key] == self.parent().po.vars['convert_for_motion'][key])
            for key in self.parent().po.vars['convert_for_motion'].keys():
                if key != 'logical':
                    are_dicts_equal = are_dicts_equal and np.all(
                        key in previous_csc and self.parent().po.vars['convert_for_motion'][key] ==
                        previous_csc[key])
            if not are_dicts_equal:
                self.parent().po.find_if_lighter_background()

        if not self.parent().thread['SaveAllVars'].isRunning():
            self.parent().thread['SaveAllVars'].start()

        if self.parent().last_is_first:
            self.parent().change_widget(0) # FirstWidget
        else:
            self.parent().change_widget(3) # ImageAnalysisWindow ThirdWidget
    
    def closeEvent(self, event):
        event.accept


# if __name__ == "__main__":
#     from cellects.gui.cellects import CellectsMainWidget
#     import sys
#     app = QtWidgets.QApplication([])
#     parent = CellectsMainWidget()
#     session = AdvancedParameters(parent, False)
#     parent.insertWidget(0, session)
#     parent.show()
#     sys.exit(app.exec())
