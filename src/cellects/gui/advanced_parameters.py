#!/usr/bin/env python3
"""GUI module implementing the Advanced Parameters configuration window for Cellects.

This module provides an interactive dialog allowing users to configure advanced image
analysis and processing settings. The UI organizes parameter controls into categorized boxes:
general parameters, cell detection rules, spatiotemporal scaling, computer resources,
video saving options, and color space conversion (CSC) settings. It maintains user preferences
in both RAM and persistent storage via "Ok" button click.

Main Components
AdvancedParameters : QWidget subclass for advanced parameter configuration window

Notes
Uses QThread for background operations to maintain UI responsiveness during parameter saving.
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
from cellects.gui.ui_strings import AP, IAW


class AdvancedParameters(WindowType):
    """
        This class creates the Advanced Parameters window.
        In the app, it is accessible from the first and the Video tracking window. It allows the user to fill in
        some parameters stored in the directory po.all (in RAM) and in all_vars.pkl (in ROM).
        Clicking "Ok" save the directory in RAM and in ROM.
    """
    def __init__(self, parent, night_mode):
        """
        Initialize the AdvancedParameters window with a parent widget and night mode setting.

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
        >>> from cellects.gui.advanced_parameters import AdvancedParameters
        >>> import sys
        >>> app = QtWidgets.QApplication([])
        >>> parent = CellectsMainWidget()
        >>> session = AdvancedParameters(parent, False)
        >>> session.true_init()
        >>> parent.insertWidget(0, session)
        >>> parent.show()
        >>> sys.exit(app.exec())
        """
        super().__init__(parent, night_mode)

        self.setParent(parent)
        try:
            self.true_init()
        except KeyError:
            default_dicts = DefaultDicts()
            self.parent().po.all = default_dicts.all
            self.parent().po.vars = default_dicts.vars
            self.true_init()

    def true_init(self):
        """
        Initialize the AdvancedParameters window.

        This method sets up the layout and widgets for the AdvancedParameters window,
        including scroll areas, layouts, and various UI components for configuring
        advanced parameters, including 'Cancel' and 'Ok' buttons.

        Notes
        -----
        This method assumes that the parent widget has a 'po' attribute with specific settings and variables.
        """
        logging.info("Initialize AdvancedParameters window")
        self.layout = QtWidgets.QVBoxLayout()

        self.left_scroll_table = QtWidgets.QScrollArea()  #   # Scroll Area which contains the widgets, set as the centralWidget
        self.left_scroll_table.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.left_scroll_table.setMinimumHeight(150)
        self.left_scroll_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.left_scroll_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.left_scroll_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self.left_col_layout = QtWidgets.QVBoxLayout()
        self.right_col_layout = QtWidgets.QVBoxLayout()
        self.left_col_widget = QtWidgets.QWidget()
        self.right_col_widget = QtWidgets.QWidget()
        # Create the main Title
        self.title = FixedText('Advanced parameters', police=30, night_mode=self.parent().po.all['night_mode'])
        self.title.setAlignment(QtCore.Qt.AlignHCenter)
        # Create the main layout
        self.layout.addWidget(self.title)
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
        # I/B/ Create the box
        self.general_param_box_layout = QtWidgets.QGridLayout()
        self.general_param_box_widget = QtWidgets.QWidget()
        self.general_param_box_widget.setStyleSheet(boxstylesheet)
        # I/C/ Create widgets
        self.automatically_crop = Checkbox(self.parent().po.all['automatically_crop'])
        self.automatically_crop_label = FixedText(AP["Crop_images"]["label"], tip=AP["Crop_images"]["tips"],
                                                  night_mode=self.parent().po.all['night_mode'])

        self.subtract_background = Checkbox(self.parent().po.vars['subtract_background'])
        self.subtract_background.stateChanged.connect(self.subtract_background_check)
        self.subtract_background_label = FixedText(AP["Subtract_background"]["label"], tip=AP["Subtract_background"]["tips"], night_mode=self.parent().po.all['night_mode'])

        self.keep_cell_and_back_for_all_folders = Checkbox(self.parent().po.all['keep_cell_and_back_for_all_folders'])
        self.keep_cell_and_back_for_all_folders_label = FixedText(AP["Keep_drawings"]["label"],
                                               tip=AP["Keep_drawings"]["tips"],
                                               night_mode=self.parent().po.all['night_mode'])

        self.correct_errors_around_initial = Checkbox(self.parent().po.vars['correct_errors_around_initial'])
        self.correct_errors_around_initial_label = FixedText(AP["Correct_errors_around_initial"]["label"],
                                               tip=AP["Correct_errors_around_initial"]["tips"],
                                               night_mode=self.parent().po.all['night_mode'])

        self.prevent_fast_growth_near_periphery = Checkbox(self.parent().po.vars['prevent_fast_growth_near_periphery'])
        self.prevent_fast_growth_near_periphery_label = FixedText(AP["Prevent_fast_growth_near_periphery"]["label"],
                                               tip=AP["Prevent_fast_growth_near_periphery"]["tips"],
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
        self.all_specimens_have_same_direction_label = FixedText(AP["Specimens_have_same_direction"]["label"],
                                                         tip=AP["Specimens_have_same_direction"]["tips"],
                                                         night_mode=self.parent().po.all['night_mode'])


        connect_distant_shape = self.parent().po.all['connect_distant_shape_during_segmentation']
        self.connect_distant_shape_during_segmentation = Checkbox(connect_distant_shape)
        self.connect_distant_shape_during_segmentation.stateChanged.connect(self.do_distant_shape_int_changed)
        self.connect_distant_shape_label = FixedText(AP["Connect_distant_shapes"]["label"],
                                                         tip=AP["Connect_distant_shapes"]["tips"],
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

        self.use_min_size.setStyleSheet("QCheckBox::indicator {width: 12px;height: 12px;background-color: transparent;"
                            "border-radius: 5px;border-style: solid;border-width: 1px;"
                            "border-color: rgb(100,100,100);}"
                            "QCheckBox::indicator:checked {background-color: rgb(70,130,180);}"
                            "QCheckBox:checked, QCheckBox::indicator:checked {border-color: black black white white;}"
                            "QCheckBox:checked {background-color: transparent;}"
                            "QCheckBox:margin-left {100%}"
                            "QCheckBox:margin-right {0%}")
        self.min_size_for_connection_label.setAlignment(QtCore.Qt.AlignRight)
        self.use_max_size.setStyleSheet("QCheckBox::indicator {width: 12px;height: 12px;background-color: transparent;"
                            "border-radius: 5px;border-style: solid;border-width: 1px;"
                            "border-color: rgb(100,100,100);}"
                            "QCheckBox::indicator:checked {background-color: rgb(70,130,180);}"
                            "QCheckBox:checked, QCheckBox::indicator:checked {border-color: black black white white;}"
                            "QCheckBox:checked {background-color: transparent;}"
                            "QCheckBox:margin-left {100%}"
                            "QCheckBox:margin-right {0%}")
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

        # III/ Third box: Appearing cell/colony
        # III/A/ Title
        self.appearing_cell_label = FixedText('Appearing cell/colony parameters:', tip="",
                                              night_mode=self.parent().po.all['night_mode'])

        self.left_col_layout.addWidget(self.appearing_cell_label)
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
        self.do_automatic_size_thresholding = Checkbox(self.parent().po.all['automatic_size_thresholding'])
        self.do_automatic_size_thresholding_label = FixedText(AP["Appearance_size_threshold"]["label"],
                                                              tip=AP["Appearance_size_threshold"]["tips"],
                                                              night_mode=self.parent().po.all['night_mode'])
        self.do_automatic_size_thresholding.stateChanged.connect(self.do_automatic_size_thresholding_changed)
        self.appearing_selection = Combobox(["largest", "most_central"], night_mode=self.parent().po.all['night_mode'])
        self.appearing_selection_label = FixedText(AP["Appearance_detection_method"]["label"],
                                                   tip=AP["Appearance_detection_method"]["tips"],
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

        # V/ Fifth box: Network detection parameters:#
        # IV/A/ Title
        self.rolling_window_s_label = FixedText(IAW["Rolling_window_segmentation"]["label"] + ': (auto if checked)',
                                                tip=IAW["Rolling_window_segmentation"]["tips"],
                                              night_mode=self.parent().po.all['night_mode'])
        self.left_col_layout.addWidget(self.rolling_window_s_label)
        self.rolling_window_s_layout = QtWidgets.QGridLayout()
        self.rolling_window_s_widget = QtWidgets.QWidget()
        self.rolling_window_s_widget.setStyleSheet(boxstylesheet)
        self.mesh_side_length_cb = Checkbox(self.parent().po.all['auto_mesh_side_length'],
                                        night_mode=self.parent().po.all['night_mode'])
        self.mesh_side_length_cb.stateChanged.connect(self.mesh_side_length_cb_changed)
        self.mesh_side_length_label = FixedText(AP["Mesh_side_length"]["label"], tip=AP["Mesh_side_length"]["tips"],
                                                night_mode=self.parent().po.all['night_mode'])
        if self.parent().po.vars['rolling_window_segmentation']['side_len'] is None:
            self.mesh_side_length = Spinbox(min=0, max=1000000, val=4, decimals=0,
                                            night_mode=self.parent().po.all['night_mode'])
            self.mesh_side_length.setVisible(False)
        else:
            self.mesh_side_length = Spinbox(min=0, max=1000000, val=self.parent().po.vars['rolling_window_segmentation']['side_len'], decimals=0,
                                            night_mode=self.parent().po.all['night_mode'])


        self.mesh_step_length_cb = Checkbox(self.parent().po.all['auto_mesh_step_length'],
                                        night_mode=self.parent().po.all['night_mode'])
        self.mesh_step_length_cb.stateChanged.connect(self.mesh_step_length_cb_changed)
        self.mesh_step_length_label = FixedText(AP["Mesh_step_length"]["label"], tip=AP["Mesh_step_length"]["tips"],
                                                night_mode=self.parent().po.all['night_mode'])
        if self.parent().po.vars['rolling_window_segmentation']['side_len'] is None:
            self.mesh_step_length = Spinbox(min=0, max=1000, val=2, decimals=0,
                                            night_mode=self.parent().po.all['night_mode'])
            self.mesh_step_length.setVisible(False)
        else:
            self.mesh_step_length = Spinbox(min=0, max=1000, val=self.parent().po.vars['rolling_window_segmentation']['step'], decimals=0,
                                            night_mode=self.parent().po.all['night_mode'])


        self.mesh_min_int_var_cb = Checkbox(self.parent().po.all['auto_mesh_min_int_var'],
                                        night_mode=self.parent().po.all['night_mode'])
        self.mesh_min_int_var_cb.stateChanged.connect(self.mesh_min_int_var_cb_changed)
        if self.parent().po.vars['rolling_window_segmentation']['side_len'] is None:
            self.mesh_min_int_var = Spinbox(min=0, max=1000, val=2, decimals=0,
                                            night_mode=self.parent().po.all['night_mode'])
            self.mesh_min_int_var.setVisible(False)
        else:
            self.mesh_min_int_var = Spinbox(min=0, max=1000, val=self.parent().po.vars['rolling_window_segmentation']['min_int_var'], decimals=0,
                                            night_mode=self.parent().po.all['night_mode'])
        self.mesh_min_int_var_label = FixedText(AP["Mesh_minimal_intensity_variation"]["label"],
                                                tip=AP["Mesh_minimal_intensity_variation"]["tips"],
                                                night_mode=self.parent().po.all['night_mode'])
        self.rolling_window_s_layout.addWidget(self.mesh_side_length_cb, 0, 0)
        self.rolling_window_s_layout.addWidget(self.mesh_side_length_label, 0, 1)
        self.rolling_window_s_layout.addWidget(self.mesh_side_length, 0, 2)
        self.rolling_window_s_layout.addWidget(self.mesh_step_length_cb, 1, 0)
        self.rolling_window_s_layout.addWidget(self.mesh_step_length_label, 1, 1)
        self.rolling_window_s_layout.addWidget(self.mesh_step_length, 1, 2)
        self.rolling_window_s_layout.addWidget(self.mesh_min_int_var_cb, 2, 0)
        self.rolling_window_s_layout.addWidget(self.mesh_min_int_var_label, 2, 1)
        self.rolling_window_s_layout.addWidget(self.mesh_min_int_var, 2, 2)
        self.rolling_window_s_widget.setLayout(self.rolling_window_s_layout)
        self.left_col_layout.addWidget(self.rolling_window_s_widget)

        # IV/ Fourth box: Oscillation period:
        # IV/A/ Title
        self.oscillation_label = FixedText('Oscillatory parameters:', tip="",
                                              night_mode=self.parent().po.all['night_mode'])
        self.left_col_layout.addWidget(self.oscillation_label)

        self.oscillation_period_layout = QtWidgets.QGridLayout()
        self.oscillation_period_widget = QtWidgets.QWidget()
        self.oscillation_period_widget.setStyleSheet(boxstylesheet)

        self.oscillation_period = Spinbox(min=0, max=10000, val=self.parent().po.vars['expected_oscillation_period'], decimals=2,
                                          night_mode=self.parent().po.all['night_mode'])
        self.oscillation_period_label = FixedText(AP["Expected_oscillation_period"]["label"],
                                                  tip=AP["Expected_oscillation_period"]["tips"],
                                                  night_mode=self.parent().po.all['night_mode'])

        self.minimal_oscillating_cluster_size = Spinbox(min=1, max=1000000000, decimals=0, val=self.parent().po.vars['minimal_oscillating_cluster_size'],
                                          night_mode=self.parent().po.all['night_mode'])
        self.minimal_oscillating_cluster_size_label = FixedText(AP["Minimal_oscillating_cluster_size"]["label"],
                                                  tip=AP["Minimal_oscillating_cluster_size"]["tips"],
                                                  night_mode=self.parent().po.all['night_mode'])

        self.oscillation_period_layout.addWidget(self.oscillation_period, 0, 0)
        self.oscillation_period_layout.addWidget(self.oscillation_period_label, 0, 1)
        self.oscillation_period_layout.addWidget(self.minimal_oscillating_cluster_size, 1, 0)
        self.oscillation_period_layout.addWidget(self.minimal_oscillating_cluster_size_label, 1, 1)

        self.oscillation_period_widget.setLayout(self.oscillation_period_layout)
        self.left_col_layout.addWidget(self.oscillation_period_widget)

        # I/ First box: Scales
        # I/A/ Title
        self.right_scroll_table = QtWidgets.QScrollArea()   # Scroll Area which contains the widgets, set as the centralWidget
        self.right_scroll_table.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.right_scroll_table.setMinimumHeight(150)#self.parent().im_max_height - 100
        self.right_scroll_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.right_scroll_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.right_scroll_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scale_box_label = FixedText(AP["Spatio_temporal_scaling"]["label"] + ':',
                                         tip=AP["Spatio_temporal_scaling"]["tips"],
                                         night_mode=self.parent().po.all['night_mode'])
        self.right_col_layout.addWidget(self.scale_box_label)

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
        self.pixels_to_mm = Checkbox(self.parent().po.vars['output_in_mm'])
        self.pixels_to_mm_label = FixedText('Convert areas and distances from pixels to mm',
                                            tip="Check if you want output variables to be in mm\nUncheck if you want output variables to be in pixels",
                                            night_mode=self.parent().po.all['night_mode'])

        # I/D/ Arrange widgets in the box
        self.scale_box_layout.addWidget(self.extract_time, 0, 0)
        self.scale_box_layout.addWidget(self.time_step_label, 0, 1)
        self.scale_box_layout.addWidget(self.time_step, 0, 2)
        self.scale_box_layout.addWidget(self.pixels_to_mm, 2, 0)
        self.scale_box_layout.addWidget(self.pixels_to_mm_label, 2, 1)
        self.scale_box_widget.setLayout(self.scale_box_layout)
        self.right_col_layout.addWidget(self.scale_box_widget)

        # IV/ Fourth box: Computer resources
        # IV/A/ Title
        self.resources_label = FixedText('Computer resources:', tip="",
                                            night_mode=self.parent().po.all['night_mode'])
        self.right_col_layout.addWidget(self.resources_label)

        # IV/B/ Create the box
        self.resources_box_layout = QtWidgets.QGridLayout()
        self.resources_box_widget = QtWidgets.QWidget()
        self.resources_box_widget.setStyleSheet(boxstylesheet)

        # IV/C/ Create widgets
        self.do_multiprocessing = Checkbox(self.parent().po.all['do_multiprocessing'])
        self.do_multiprocessing_label = FixedText(AP["Parallel_analysis"]["label"], tip=AP["Parallel_analysis"]["tips"],
                                                  night_mode=self.parent().po.all['night_mode'])
        self.do_multiprocessing.stateChanged.connect(self.do_multiprocessing_is_clicked)
        self.max_core_nb = Spinbox(min=0, max=256, val=self.parent().po.all['cores'],
                                   night_mode=self.parent().po.all['night_mode'])
        self.max_core_nb_label = FixedText(AP["Proc_max_core_nb"]["label"],
                                           tip=AP["Proc_max_core_nb"]["tips"],
                                           night_mode=self.parent().po.all['night_mode'])
        self.min_memory_left = Spinbox(min=0, max=1024, val=self.parent().po.vars['min_ram_free'], decimals=1,
                                       night_mode=self.parent().po.all['night_mode'])
        self.min_memory_left_label = FixedText(AP["Minimal_RAM_let_free"]["label"],
                                                tip=AP["Minimal_RAM_let_free"]["tips"],
                                               night_mode=self.parent().po.all['night_mode'])

        self.lose_accuracy_to_save_memory = Checkbox(self.parent().po.vars['lose_accuracy_to_save_memory'])
        self.lose_accuracy_to_save_memory_label = FixedText(AP["Lose_accuracy_to_save_RAM"]["label"],
                                                  tip=AP["Lose_accuracy_to_save_RAM"]["tips"],
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

        # V/ Fifth box: Video saving
        # V/A/ Title
        self.video_saving_label = FixedText('Video saving:', tip="",
                                         night_mode=self.parent().po.all['night_mode'])
        self.right_col_layout.addWidget(self.video_saving_label)

        # V/B/ Create the box
        self.video_saving_layout = QtWidgets.QGridLayout()
        self.video_saving_widget = QtWidgets.QWidget()
        self.video_saving_widget.setStyleSheet(boxstylesheet)

        # V/C/ Create widgets
        self.video_fps = Spinbox(min=0, max=10000, val=self.parent().po.vars['video_fps'], decimals=2,
                                 night_mode=self.parent().po.all['night_mode'])
        self.video_fps_label = FixedText(AP["Video_fps"]["label"], tip=AP["Video_fps"]["tips"],
                                         night_mode=self.parent().po.all['night_mode'])
        self.keep_unaltered_videos = Checkbox(self.parent().po.vars['keep_unaltered_videos'])
        self.keep_unaltered_videos_label = FixedText(AP["Keep_unaltered_videos"]["label"],
                                                     tip=AP["Keep_unaltered_videos"]["tips"],
                                                     night_mode=self.parent().po.all['night_mode'])
        self.save_processed_videos = Checkbox(self.parent().po.vars['save_processed_videos'])
        self.save_processed_videos_label = FixedText(AP["Save_processed_videos"]["label"],
                                                     tip=AP["Save_processed_videos"]["tips"],
                                                     night_mode=self.parent().po.all['night_mode'])

        # V/D/ Arrange widgets in the box
        curr_box_row = 0
        self.video_saving_layout.addWidget(self.video_fps, curr_box_row, 0)
        self.video_saving_layout.addWidget(self.video_fps_label, curr_box_row, 1)
        curr_box_row += 1
        self.video_saving_layout.addWidget(self.keep_unaltered_videos, curr_box_row, 0)
        self.video_saving_layout.addWidget(self.keep_unaltered_videos_label, curr_box_row, 1)
        curr_box_row += 1
        self.video_saving_layout.addWidget(self.save_processed_videos, curr_box_row, 0)
        self.video_saving_layout.addWidget(self.save_processed_videos_label, curr_box_row, 1)
        curr_box_row += 1
        self.video_saving_widget.setLayout(self.video_saving_layout)
        self.right_col_layout.addWidget(self.video_saving_widget)

        # VII/ Seventh box: csc
        # VII/A/ Title
        # VII/C/ Create widgets
        self.generate_csc_editing()
        # VII/D/ Arrange widgets in the box
        self.right_col_layout.addWidget(self.edit_widget)

        # VIII/ Finalize layout and add the night mode option and the ok button
        self.left_col_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))
        self.right_col_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))
        self.left_col_widget.setLayout(self.left_col_layout)
        self.right_col_widget.setLayout(self.right_col_layout)
        self.central_widget = QtWidgets.QWidget()
        self.central_layout = QtWidgets.QHBoxLayout()
        self.central_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.left_scroll_table.setWidget(self.left_col_widget)
        self.left_scroll_table.setWidgetResizable(True)
        self.left_scroll_table.setParent(self.central_widget)
        self.central_layout.addWidget(self.left_scroll_table)
        self.central_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.right_scroll_table.setWidget(self.right_col_widget)
        self.right_scroll_table.setWidgetResizable(True)
        self.right_scroll_table.setParent(self.central_widget)
        self.central_layout.addWidget(self.right_scroll_table)
        self.central_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.central_widget.setLayout(self.central_layout)
        self.layout.addWidget(self.central_widget)

        # Last row
        self.last_row_layout = QtWidgets.QHBoxLayout()
        self.last_row_widget = QtWidgets.QWidget()
        self.night_mode_cb = Checkbox(self.parent().po.all['night_mode'])
        self.night_mode_cb.clicked.connect(self.night_mode_is_clicked)
        self.night_mode_label = FixedText(AP["Night_mode"]["label"], tip=AP["Night_mode"]["tips"],
                                          night_mode=self.parent().po.all['night_mode'])
        self.reset_all_settings = PButton(AP["Reset_all_settings"]["label"], tip=AP["Reset_all_settings"]["tips"],
                                          night_mode=self.parent().po.all['night_mode'])
        self.reset_all_settings.clicked.connect(self.reset_all_settings_is_clicked)
        self.message = FixedText('', night_mode=self.parent().po.all['night_mode'])
        self.cancel = PButton('Cancel', night_mode=self.parent().po.all['night_mode'])
        self.cancel.clicked.connect(self.cancel_is_clicked)
        self.ok = PButton('Ok', night_mode=self.parent().po.all['night_mode'])
        self.ok.clicked.connect(self.ok_is_clicked)
        self.last_row_layout.addWidget(self.night_mode_cb)
        self.last_row_layout.addWidget(self.night_mode_label)
        self.last_row_layout.addWidget(self.reset_all_settings)
        self.last_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.last_row_layout.addWidget(self.message)
        self.last_row_layout.addWidget(self.cancel)
        self.last_row_layout.addWidget(self.ok)
        self.last_row_widget.setLayout(self.last_row_layout)
        self.layout.addWidget(self.last_row_widget)

        self.setLayout(self.layout)

    def display_conditionally_visible_widgets(self):
        """
        Conditionally displays widgets based on various settings within the parent object.

        This function controls the visibility of several UI elements based on the
        values in the parent object's `all` dictionary and `vars` dictionary. It ensures
        that only relevant widgets are shown to the user, depending on the current settings.
        """
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
        """
        Handles the logic for using background subtraction or not during image segmentation.
        """
        self.parent().po.motion = None
        if self.subtract_background.isChecked():
            self.parent().po.first_exp_ready_to_run = False

    def prevent_fast_growth_near_periphery_check(self):
        """
        Handles the logic for using a special algorithm on growth near the periphery during video segmentation.
        """
        checked_status = self.prevent_fast_growth_near_periphery.isChecked()
        self.periphery_width.setVisible(checked_status)
        self.periphery_width_label.setVisible(checked_status)
        self.max_periphery_growth.setVisible(checked_status)
        self.max_periphery_growth_label.setVisible(checked_status)

    def do_automatic_size_thresholding_changed(self):
        """
        This function toggles the visibility of `first_move_threshold` and
        `first_move_threshold_label` UI elements based on whether the
        `do_automatic_size_thresholding` checkbox is checked or not.
        """
        self.first_move_threshold.setVisible(not self.do_automatic_size_thresholding.isChecked())
        self.first_move_threshold_label.setVisible(not self.do_automatic_size_thresholding.isChecked())

    def mesh_side_length_cb_changed(self):
        self.mesh_side_length.setVisible(self.mesh_side_length_cb.isChecked())

    def mesh_step_length_cb_changed(self):
        self.mesh_step_length.setVisible(self.mesh_step_length_cb.isChecked())

    def mesh_min_int_var_cb_changed(self):
        self.mesh_min_int_var.setVisible(self.mesh_min_int_var_cb.isChecked())

    def extract_time_is_clicked(self):
        """
        Toggle the visibility of time_step_label and update its text/tooltip based on
        whether extract_time is checked.
        """
        self.time_step.setVisible(not self.extract_time.isChecked())
        if self.extract_time.isChecked():
            self.time_step_label.setText("Automatically extract time interval between images")
            self.time_step_label.setToolTip("Uses the exif data of the images (if available), to extract these intervals\nOtherwise, default time interval is 1 min")
        else:
            self.time_step_label.setText("Set the time interval between images")
            self.time_step_label.setToolTip("In minutes")

    def do_multiprocessing_is_clicked(self):
        """
        Update the visibility of `max_core_nb` and `max_core_nb_label` based on the checkbox state of `do_multiprocessing`.
        """
        self.max_core_nb.setVisible(self.do_multiprocessing.isChecked())
        self.max_core_nb_label.setVisible(self.do_multiprocessing.isChecked())

    def do_distant_shape_int_changed(self):
        """
        Toggles the visibility of widgets based the use of an algorithm allowing to connect distant shapes
        during segmentation.
        """
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
        """
        Toggles the visibility of max size input fields based on checkbox state.
        """
        do_use_max_size = self.use_max_size.isChecked()
        self.max_size_for_connection.setVisible(do_use_max_size)
        self.max_size_for_connection_label.setVisible(do_use_max_size)
        if do_use_max_size:
            self.max_size_for_connection.setValue(300)

    def use_min_size_changed(self):
        """
        Updates the visibility and value of UI elements based on whether a checkbox is checked.
        """
        do_use_min_size = self.use_min_size.isChecked()
        self.min_size_for_connection.setVisible(do_use_min_size)
        self.min_size_for_connection_label.setVisible(do_use_min_size)
        if do_use_min_size:
            self.min_size_for_connection.setValue(30)

    def generate_csc_editing(self):
        """
        Generate CSC Editing Layout

        Creates and configures the layout for Color Space Combination (CSC) editing in the video analysis window,
        initializing widgets and connecting signals to slots for dynamic UI handling.
        """
        self.edit_widget = QtWidgets.QWidget()
        self.edit_layout = QtWidgets.QVBoxLayout()

        # 2) Titles
        self.video_csc_label = FixedText(AP["Csc_for_video_analysis"]["label"] + ':',
                                         tip=AP["Csc_for_video_analysis"]["tips"],
                                         night_mode=self.parent().po.all['night_mode'])
        self.video_csc_label.setFixedHeight(30)
        self.edit_layout.addWidget(self.video_csc_label)
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
        self.logical_operator_label = FixedText(IAW["Logical_operator"]["label"], halign='c', tip=IAW["Logical_operator"]["tips"],
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
        self.first_csc_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum), 0, 5, 3, 1)
        self.first_csc_widget.setLayout(self.first_csc_layout)
        self.both_csc_layout.addWidget(self.first_csc_widget)

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
        self.second_csc_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum), 0, 5, 3, 1)
        self.second_csc_widget.setLayout(self.second_csc_layout)
        self.both_csc_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.both_csc_layout.addWidget(self.second_csc_widget)
        self.both_csc_widget.setLayout(self.both_csc_layout)
        self.edit_layout.addWidget(self.both_csc_widget)

        # 4) logical_operator
        self.logical_op_widget = QtWidgets.QWidget()
        self.logical_op_widget.setFixedHeight(30)
        self.logical_op_layout = QtWidgets.QHBoxLayout()
        self.logical_op_layout.addWidget(self.logical_operator_label)
        self.logical_op_layout.addWidget(self.logical_operator_between_combination_result)
        self.logical_op_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.logical_operator_between_combination_result.setVisible(False)
        self.logical_operator_label.setVisible(False)
        self.logical_op_widget.setLayout(self.logical_op_layout)
        self.logical_op_widget.setFixedHeight(50)
        self.edit_layout.addWidget(self.logical_op_widget)

        # 6) Open the more_than_2_colors row layout
        self.more_than_2_colors_widget = QtWidgets.QWidget()
        self.more_than_2_colors_layout = QtWidgets.QHBoxLayout()
        self.more_than_two_colors = Checkbox(self.parent().po.all["more_than_two_colors"])
        self.more_than_two_colors.setStyleSheet("QCheckBox::indicator {width: 12px;height: 12px;background-color: transparent;"
                            "border-radius: 5px;border-style: solid;border-width: 1px;"
                            "border-color: rgb(100,100,100);}"
                            "QCheckBox::indicator:checked {background-color: rgb(70,130,180);}"
                            "QCheckBox:checked, QCheckBox::indicator:checked {border-color: black black white white;}"
                            "QCheckBox:checked {background-color: transparent;}"
                            "QCheckBox:margin-left {0%}"
                            "QCheckBox:margin-right {-10%}")
        self.more_than_two_colors.stateChanged.connect(self.display_more_than_two_colors_option)

        self.more_than_two_colors_label = FixedText(IAW["Kmeans"]["label"],
                                                    tip=IAW["Kmeans"]["tips"], night_mode=self.parent().po.all['night_mode'])
        self.more_than_two_colors_label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.more_than_two_colors_label.setAlignment(QtCore.Qt.AlignLeft)
        self.distinct_colors_number = Spinbox(min=2, max=5, val=self.parent().po.vars["color_number"], night_mode=self.parent().po.all['night_mode'])
        self.more_than_2_colors_layout.addWidget(self.more_than_two_colors)
        self.more_than_2_colors_layout.addWidget(self.more_than_two_colors_label)
        self.more_than_2_colors_layout.addWidget(self.distinct_colors_number)
        self.more_than_2_colors_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.more_than_2_colors_widget.setLayout(self.more_than_2_colors_layout)
        self.more_than_2_colors_widget.setFixedHeight(50)
        self.edit_layout.addWidget(self.more_than_2_colors_widget)
        self.edit_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))
        self.edit_widget.setLayout(self.edit_layout)

    def one_csc_editing(self):
        """
        Creates a list of widgets for color space editing.

        Returns
        -------
        widget_list : List[QtWidgets.QWidget]
            A list containing a Combobox, three Spinboxes, and a PButton.

        Notes
        -----
        The Combobox widget allows selection from predefined color spaces,
        the Spinboxes are for editing numerical values, and the PButton is
        for adding new entries.
        """
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
        """
        Update the visibility and values of UI components based on the logical operator selection.
        """
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
        """
        Display logical operator components in the user interface.
        """
        self.logical_operator_between_combination_result.setVisible(True)
        self.logical_operator_label.setVisible(True)

    def display_row2(self):
        """
        Display or hide the second row of the csc editing widgets.
        """
        self.row1[4].setVisible(False)
        for i in range(5):
            self.row2[i].setVisible(True)
        self.display_logical_operator()

    def display_row3(self):
        """
        Display or hide the third row of the csc editing widgets.
        """
        self.row2[4].setVisible(False)
        for i in range(4):
            self.row3[i].setVisible(True)
        self.display_logical_operator()

    def display_row22(self):
        """
        Display or hide the second row (for the second image segmentation pipeline) of the csc editing widgets.
        """
        self.row21[4].setVisible(False)
        for i in range(5):
            self.row22[i].setVisible(True)
        self.display_logical_operator()

    def display_row23(self):
        """
        Display or hide the third row (for the second image segmentation pipeline) of the csc editing widgets.
        """
        self.row22[4].setVisible(False)
        for i in range(4):
            self.row23[i].setVisible(True)
        self.display_logical_operator()

    def update_csc_editing_display(self):
        """
        Update the color space conversion (CSC) editing display.

        This method updates the visibility and values of UI elements related to color
        space conversions based on the current state of `self.csc_dict`. It handles
        the display logic for different color spaces and their combinations, ensuring
        that the UI reflects the current configuration accurately.
        """
        c_space_order = ["None", "bgr", "hsv", "hls", "lab", "luv", "yuv"]
        remaining_c_spaces = []
        row_number1 = 0
        row_number2 = 0
        if "PCA" in self.parent().po.vars['convert_for_motion'].keys():
            self.row1[0].setCurrentIndex(0)
            self.row1[0].setVisible(True)
            for i in range(1, 4):
                self.row1[i].setVisible(False)
        else:
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
        """
        Save user-defined combination of color spaces and channels.
        """
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
                    if np.any(channels[i, :] < 0.):
                        channels[i, :] + channels[i, :].min()
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
        """
        Display the More Than Two Colors Options

        This method manages the visibility and state of UI elements related to selecting
        more than two colors for displaying biological masks in advanced mode.
        """
        if self.more_than_two_colors.isChecked():
            self.distinct_colors_number.setVisible(True)
            self.more_than_two_colors_label.setText("How many distinct colors?")
            self.distinct_colors_number.setValue(3)
        else:
            self.more_than_two_colors_label.setText("Heterogeneous background")
            self.distinct_colors_number.setVisible(False)
            self.distinct_colors_number.setValue(2)

    def night_mode_is_clicked(self):
        """ Triggered when night_mode_cb check status changes"""
        self.parent().po.all['night_mode'] = self.night_mode_cb.isChecked()
        self.message.setText('Close and restart Cellects to apply night or light mode')
        self.message.setStyleSheet("color: rgb(230, 145, 18)")

    def reset_all_settings_is_clicked(self):
        """
        Reset All Settings on Click

        Resets the application settings to their default state by removing specific pickle files and saving new default dictionaries.

        Notes
        -----
        - This function removes specific pickle files to reset settings.
        - The function changes the current working directory temporarily.
        """
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
        """
        Instead of saving the widgets values to the saved states, use the saved states to fill in the widgets.

        This function updates the state of several checkboxes based on saved variables
        and descriptors. It also changes the active widget to either the first or third
        widget depending on a condition.
        """
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
        """
        Updates the parent object's processing options with the current state of various UI elements.

        Summary
        -------
        Saves the current state of UI components to the parent object's processing options dictionary.

        Extended Description
        --------------------
        This method iterates through various UI components such as checkboxes, sliders,
        and dropdowns to save their current state into the parent object's processing
        options variables. This allows the software to retain user preferences across
        sessions and ensures that all settings are correctly applied before processing.
        """
        self.parent().po.all['automatically_crop'] = self.automatically_crop.isChecked()
        self.parent().po.vars['subtract_background'] = self.subtract_background.isChecked()
        self.parent().po.all['keep_cell_and_back_for_all_folders'] = self.keep_cell_and_back_for_all_folders.isChecked()
        self.parent().po.vars['correct_errors_around_initial'] = self.correct_errors_around_initial.isChecked()
        self.parent().po.vars['prevent_fast_growth_near_periphery'] = self.prevent_fast_growth_near_periphery.isChecked()
        self.parent().po.vars['periphery_width'] = int(self.periphery_width.value())
        self.parent().po.vars['max_periphery_growth'] = int(self.max_periphery_growth.value())

        self.parent().po.all['first_move_threshold_in_mm²'] = self.first_move_threshold.value()
        self.parent().po.vars['output_in_mm'] = self.pixels_to_mm.isChecked()
        self.parent().po.all['automatic_size_thresholding'] = self.do_automatic_size_thresholding.isChecked()
        self.parent().po.vars['appearance_detection_method'] = self.appearing_selection.currentText()

        self.parent().po.all['auto_mesh_step_length'] = self.mesh_step_length_cb.isChecked()
        if self.parent().po.all['auto_mesh_step_length']:
            self.parent().po.vars['rolling_window_segmentation']['step'] = None
        else:
            self.parent().po.vars['rolling_window_segmentation']['step'] = int(self.mesh_step_length.value())

        self.parent().po.all['auto_mesh_side_length'] = self.mesh_side_length_cb.isChecked()
        if self.parent().po.all['auto_mesh_side_length']:
            self.parent().po.vars['rolling_window_segmentation']['side_len'] = None
        else:
            self.parent().po.vars['rolling_window_segmentation']['side_len'] = int(self.mesh_side_length.value())

        self.parent().po.all['auto_mesh_min_int_var'] = self.mesh_min_int_var_cb.isChecked()
        if self.parent().po.all['auto_mesh_min_int_var']:
            self.parent().po.vars['rolling_window_segmentation']['min_int_var'] = None
        else:
            self.parent().po.vars['rolling_window_segmentation']['min_int_var'] = int(self.mesh_min_int_var.value())

        self.parent().po.vars['expected_oscillation_period'] = self.oscillation_period.value()
        self.parent().po.vars['minimal_oscillating_cluster_size'] = int(self.minimal_oscillating_cluster_size.value())

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

        if not self.parent().thread_dict['SaveAllVars'].isRunning():
            self.parent().thread_dict['SaveAllVars'].start()

        if self.parent().last_is_first:
            self.parent().change_widget(0) # FirstWidget
        else:
            self.parent().change_widget(3) # ImageAnalysisWindow ThirdWidget
