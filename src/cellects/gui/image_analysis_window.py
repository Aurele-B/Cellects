#!/usr/bin/env python3
"""
Image analysis GUI module for Cellects application

This module provides a user interface for configuring and performing image analysis with the Cellects system.
It allows users to adjust scaling parameters, manually label cell/background regions, select segmentation methods
(quick/careful), visualize results, and validate analysis outcomes through interactive decision prompts. The UI supports
manual arena delineation when automatic detection fails, using threaded operations for background processing.

Main Components
ImageAnalysisWindow : Main UI window for image analysis configuration and execution.

Includes parameter controls (scaling, spot shape/size), segmentation options (quick/careful/visualize)
Provides cell/background selection buttons with manual drawing capabilities
Features decision prompts via Yes/No buttons to validate intermediate results
Displays real-time image updates with user-defined annotations
Notes
Uses QThread for background operations to maintain UI responsiveness.
"""
import logging
import time
from copy import deepcopy
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from cellects.core.cellects_threads import (
    GetFirstImThread, GetLastImThread, FirstImageAnalysisThread,
    CropScaleSubtractDelineateThread, UpdateImageThread, CompleteImageAnalysisThread,
    LastImageAnalysisThread, SaveManualDelineationThread, PrepareVideoAnalysisThread)
from cellects.gui.ui_strings import IAW
from cellects.gui.custom_widgets import (
    MainTabsType, InsertImage, FullScreenImage, PButton, Spinbox,
    Combobox, Checkbox, FixedText)
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.image_analysis.image_segmentation import filter_dict
from cellects.utils.formulas import bracket_to_uint8_image_contrast


class ImageAnalysisWindow(MainTabsType):
    def __init__(self, parent: object, night_mode: bool):
        """
        Initialize the ImageAnalysis window with a parent widget and night mode setting.

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
        >>> from cellects.gui.image_analysis_window import ImageAnalysisWindow
        >>> from cellects.core.program_organizer import ProgramOrganizer
        >>> import numpy as np
        >>> import sys
        >>> app = QtWidgets.QApplication([])
        >>> parent = CellectsMainWidget()
        >>> parent.po = ProgramOrganizer()
        >>> parent.po.update_variable_dict()
        >>> parent.po.get_first_image(np.zeros((10, 10), dtype=np.uint8), 1)
        >>> session = ImageAnalysisWindow(parent, False)
        >>> session.true_init()
        >>> parent.insertWidget(0, session)
        >>> parent.show()
        >>> sys.exit(app.exec())
        """
        super().__init__(parent, night_mode)
        self.setParent(parent)
        self.csc_dict = self.parent().po.vars['convert_for_origin'] # To change
        self.manual_delineation_flag: bool = False

    def true_init(self):
        """
        Initialize the ImageAnalysisWindow class with default settings and UI components.

        This function sets up the initial state of the ImageAnalysisWindow, including various flags,
        labels, input fields, and layout configurations. It also initializes the display image
        and connects UI elements to their respective event handlers.

        Notes
        -----
        This method assumes that the parent widget has a 'po' attribute with specific settings and variables.
        """
        logging.info("Initialize ImageAnalysisWindow")
        self.data_tab.set_not_in_use()
        self.image_tab.set_in_use()
        self.video_tab.set_not_usable()
        self.hold_click_flag: bool = False
        self.is_first_image_flag: bool = True
        self.is_image_analysis_running: bool = False
        self.is_image_analysis_display_running: bool = False
        self.asking_first_im_parameters_flag: bool = True
        self.first_im_parameters_answered: bool = False
        self.auto_delineation_flag: bool = False
        self.delineation_done: bool = False
        self.asking_delineation_flag: bool = False
        self.asking_slower_or_manual_delineation_flag: bool = False
        self.slower_delineation_flag: bool = False
        self.asking_last_image_flag: bool = False
        self.step = 0
        self.temporary_mask_coord = []
        self.saved_coord = []
        self.back1_bio2 = 0
        self.bio_masks_number = 0
        self.back_masks_number = 0
        self.arena_masks_number = 0
        self.available_bio_names = np.arange(1, 1000, dtype=np.uint16)
        self.available_back_names = np.arange(1, 1000, dtype=np.uint16)
        self.parent().po.current_combination_id = 0

        self.display_image = np.zeros((self.parent().im_max_width, self.parent().im_max_width, 3), np.uint8)
        self.display_image = InsertImage(self.display_image, self.parent().im_max_height, self.parent().im_max_width)
        self.display_image.mousePressEvent = self.get_click_coordinates
        self.display_image.mouseMoveEvent = self.get_mouse_move_coordinates
        self.display_image.mouseReleaseEvent = self.get_mouse_release_coordinates

        ## Title
        self.image_number_label = FixedText(IAW["Image_number"]["label"],
                                            tip=IAW["Image_number"]["tips"],
                                            night_mode=self.parent().po.all['night_mode'])
        self.image_number_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.image_number = Spinbox(min=0, max=self.parent().po.vars['img_number'] - 1, val=self.parent().po.vars['first_detection_frame'], night_mode=self.parent().po.all['night_mode'])
        self.read = PButton("Read", night_mode=self.parent().po.all['night_mode'])
        self.read.clicked.connect(self.read_is_clicked)
        if self.parent().po.all["im_or_vid"] == 0 and len(self.parent().po.data_list) == 1:
            # If there is only one image in the folder
            self.image_number.setVisible(False)
            self.image_number_label.setVisible(False)
            self.read.setVisible(False)

        self.one_blob_per_arena = Checkbox(not self.parent().po.vars['several_blob_per_arena'])
        self.one_blob_per_arena.stateChanged.connect(self.several_blob_per_arena_check)
        self.one_blob_per_arena_label = FixedText(IAW["several_blob_per_arena"]["label"], valign="c",
                                                  tip=IAW["several_blob_per_arena"]["tips"],
                                                  night_mode=self.parent().po.all['night_mode'])


        self.scale_with_label = FixedText(IAW["Scale_with"]["label"] + ':', valign="c",
                                        tip=IAW["Scale_with"]["tips"],
                                        night_mode=self.parent().po.all['night_mode'])
        self.scale_with = Combobox(["Image horizontal size", "Cell(s) horizontal size"], night_mode=self.parent().po.all['night_mode'])
        self.scale_with.setFixedWidth(280)
        self.scale_with.setCurrentIndex(self.parent().po.all['scale_with_image_or_cells'])
        self.scale_size_label = FixedText(IAW["Scale_size"]["label"] + ':', valign="c",
                                          tip=IAW["Scale_size"]["tips"],
                                          night_mode=self.parent().po.all['night_mode'])
        if self.parent().po.all['scale_with_image_or_cells'] == 0:
            self.horizontal_size = Spinbox(min=0, max=100000,
                                        val=self.parent().po.all['image_horizontal_size_in_mm'],
                                        night_mode=self.parent().po.all['night_mode'])
        else:
            self.horizontal_size = Spinbox(min=0, max=100000,
                                        val=self.parent().po.all['starting_blob_hsize_in_mm'],
                                        night_mode=self.parent().po.all['night_mode'])
        self.horizontal_size.valueChanged.connect(self.horizontal_size_changed)
        self.scale_with.currentTextChanged.connect(self.scale_with_changed)
        self.scale_unit_label = FixedText(' mm', night_mode=self.parent().po.all['night_mode'])

        # 1) Open the first row layout
        self.row1_widget = QtWidgets.QWidget()
        self.row1_layout = QtWidgets.QHBoxLayout()
        self.row1_layout.addWidget(self.image_number_label)
        self.row1_layout.addWidget(self.image_number)
        self.row1_layout.addWidget(self.read)
        self.row1_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.row1_layout.addWidget(self.one_blob_per_arena_label)
        self.row1_layout.addWidget(self.one_blob_per_arena)
        self.row1_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.row1_layout.addWidget(self.scale_with_label)
        self.row1_layout.addWidget(self.scale_with)
        self.row1_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.row1_layout.addWidget(self.scale_size_label)
        self.row1_layout.addWidget(self.horizontal_size)

        self.row1_widget.setLayout(self.row1_layout)
        self.Vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))
        self.Vlayout.addWidget(self.row1_widget)
        self.Vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))

        # 2) Open the central row layout
        self.central_row_widget = QtWidgets.QWidget()
        self.central_row_layout = QtWidgets.QGridLayout()

        # it will contain a) the user drawn lines, b) the image, c) the csc
        # 2)a) the user drawn lines
        self.user_drawn_lines_widget = QtWidgets.QWidget()
        self.user_drawn_lines_layout = QtWidgets.QVBoxLayout()
        self.user_drawn_lines_label = FixedText(IAW["Select_and_draw"]["label"] + ":",
                                                tip=IAW["Select_and_draw"]["tips"],
                                                night_mode=self.parent().po.all['night_mode'])
        self.user_drawn_lines_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.user_drawn_lines_layout.addWidget(self.user_drawn_lines_label)
        self.pbuttons_widget = QtWidgets.QWidget()
        self.pbuttons_layout = QtWidgets.QHBoxLayout()
        self.cell = PButton("Cell", False, tip=IAW["Draw_buttons"]["tips"],
                            night_mode=self.parent().po.all['night_mode'])
        self.cell.setFixedWidth(150)
        self.background = PButton("Back", False, tip=IAW["Draw_buttons"]["tips"],
                                  night_mode=self.parent().po.all['night_mode'])
        self.background.setFixedWidth(150)
        self.cell.clicked.connect(self.cell_is_clicked)
        self.background.clicked.connect(self.background_is_clicked)
        self.pbuttons_layout.addWidget(self.cell)
        self.pbuttons_layout.addWidget(self.background)
        self.pbuttons_widget.setLayout(self.pbuttons_layout)
        self.user_drawn_lines_layout.addWidget(self.pbuttons_widget)

        self.pbuttons_tables_widget = QtWidgets.QWidget()
        self.pbuttons_tables_layout = QtWidgets.QHBoxLayout()
        self.pbuttons_tables_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.bio_pbuttons_table = QtWidgets.QScrollArea()#QTableWidget()  # Scroll Area which contains the widgets, set as the centralWidget
        self.bio_pbuttons_table.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.bio_pbuttons_table.setMinimumHeight(self.parent().im_max_height // 2)
        self.bio_pbuttons_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.bio_pbuttons_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.back_pbuttons_table = QtWidgets.QScrollArea()#QTableWidget()  # Scroll Area which contains the widgets, set as the centralWidget
        self.back_pbuttons_table.setMinimumHeight(self.parent().im_max_height // 2)
        self.back_pbuttons_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.back_pbuttons_table.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.back_pbuttons_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self.bio_added_lines_widget = QtWidgets.QWidget()
        self.back_added_lines_widget = QtWidgets.QWidget()
        self.bio_added_lines_layout = QtWidgets.QVBoxLayout()
        self.back_added_lines_layout = QtWidgets.QVBoxLayout()
        self.back_added_lines_widget.setLayout(self.back_added_lines_layout)
        self.bio_added_lines_widget.setLayout(self.bio_added_lines_layout)
        self.bio_pbuttons_table.setWidget(self.bio_added_lines_widget)
        self.back_pbuttons_table.setWidget(self.back_added_lines_widget)
        self.bio_pbuttons_table.setWidgetResizable(True)
        self.back_pbuttons_table.setWidgetResizable(True)

        self.pbuttons_tables_layout.addWidget(self.bio_pbuttons_table)
        self.pbuttons_tables_layout.addWidget(self.back_pbuttons_table)
        self.pbuttons_tables_widget.setLayout(self.pbuttons_tables_layout)
        self.user_drawn_lines_layout.addWidget(self.pbuttons_tables_widget)

        # # Dynamically add the lines
        self.bio_lines = {}
        self.back_lines = {}
        self.arena_lines = {}

        self.user_drawn_lines_widget.setLayout(self.user_drawn_lines_layout)
        self.user_drawn_lines_widget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.central_row_layout.addWidget(self.user_drawn_lines_widget, 0, 0)

        # 2)b) the image
        self.central_row_layout.addWidget(self.display_image, 0, 1)

        # Need to create this before self.generate_csc_editing()
        self.message = FixedText("", halign="r", night_mode=self.parent().po.all['night_mode'])
        self.message.setStyleSheet("color: rgb(230, 145, 18)")

        # 2)c) The csc editing
        self.generate_csc_editing()

        self.central_row_layout.addWidget(self.central_right_widget, 0, 2)
        self.central_row_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.central_row_layout.setAlignment(QtCore.Qt.AlignHCenter)
        # 2) Close the central row layout
        self.central_row_widget.setLayout(self.central_row_layout)
        self.Vlayout.addWidget(self.central_row_widget)
        self.Vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))

        # 3) Add Set supplementary parameters row 1
        self.sup_param_row1_widget = QtWidgets.QWidget()
        self.sup_param_row1_layout = QtWidgets.QHBoxLayout()

        # 4) Add Set supplementary parameters row2
        self.sup_param_row2_widget = QtWidgets.QWidget()
        self.sup_param_row2_layout = QtWidgets.QHBoxLayout()

        self.arena_shape_label = FixedText(IAW["Arena_shape"]["label"], tip=IAW["Arena_shape"]["tips"],
                                           night_mode=self.parent().po.all['night_mode'])
        self.arena_shape = Combobox(['circle', 'rectangle'], night_mode=self.parent().po.all['night_mode'])
        self.arena_shape.setFixedWidth(160)
        self.arena_shape.setCurrentText(self.parent().po.vars['arena_shape'])
        self.arena_shape.currentTextChanged.connect(self.arena_shape_changed)
        self.set_spot_shape = Checkbox(self.parent().po.all['set_spot_shape'])
        self.set_spot_shape.stateChanged.connect(self.set_spot_shape_check)
        self.spot_shape_label = FixedText(IAW["Spot_shape"]["label"], tip=IAW["Spot_shape"]["tips"], night_mode=self.parent().po.all['night_mode'])
        self.spot_shape = Combobox(['circle', 'rectangle'], night_mode=self.parent().po.all['night_mode'])
        self.spot_shape.setFixedWidth(160)
        if self.parent().po.all['starting_blob_shape'] is None:
            self.spot_shape.setCurrentIndex(0)
        else:
            self.spot_shape.setCurrentText(self.parent().po.all['starting_blob_shape'])
        self.spot_shape.currentTextChanged.connect(self.spot_shape_changed)
        self.set_spot_size = Checkbox(self.parent().po.all['set_spot_size'])
        self.set_spot_size.stateChanged.connect(self.set_spot_size_check)
        self.spot_size_label = FixedText(IAW["Spot_size"]["label"], tip=IAW["Spot_size"]["tips"],
                                         night_mode=self.parent().po.all['night_mode'])
        self.spot_size = Spinbox(min=0, max=100000, val=self.parent().po.all['starting_blob_hsize_in_mm'], decimals=2,
                                 night_mode=self.parent().po.all['night_mode'])
        self.spot_size.valueChanged.connect(self.spot_size_changed)
        self.sup_param_row2_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.sup_param_row2_layout.addWidget(self.arena_shape_label)
        self.sup_param_row2_layout.addWidget(self.arena_shape)
        self.sup_param_row2_layout.addWidget(self.set_spot_shape)
        self.sup_param_row2_layout.addWidget(self.spot_shape_label)
        self.sup_param_row2_layout.addWidget(self.spot_shape)
        self.sup_param_row2_layout.addWidget(self.set_spot_size)
        self.sup_param_row2_layout.addWidget(self.spot_size_label)
        self.sup_param_row2_layout.addWidget(self.spot_size)
        self.sup_param_row2_widget.setLayout(self.sup_param_row2_layout)
        self.sup_param_row2_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.Vlayout.addWidget(self.sup_param_row2_widget)

        self.one_blob_per_arena.setVisible(True)
        self.one_blob_per_arena_label.setVisible(True)
        self.set_spot_shape.setVisible(False)
        self.spot_shape_label.setVisible(False)
        self.spot_shape.setVisible(False)
        self.arena_shape_label.setVisible(False)
        self.arena_shape.setVisible(False)
        self.set_spot_size.setVisible(False)
        self.spot_size_label.setVisible(False)
        self.spot_size.setVisible(False)

        # 5) Add the generate option row
        self.generate_analysis_options = FixedText(IAW["Generate_analysis_options"]["label"] + ": ",
                                                   tip=IAW["Generate_analysis_options"]["tips"],
                                                   night_mode=self.parent().po.all['night_mode'])
        self.basic = PButton("Basic", night_mode=self.parent().po.all['night_mode'])
        self.basic.clicked.connect(self.basic_is_clicked)
        self.network_shaped = PButton("Network-shaped", night_mode=self.parent().po.all['night_mode'])
        self.network_shaped.clicked.connect(self.network_shaped_is_clicked)
        self.network_shaped.setVisible(False)
        self.visualize = PButton('Apply current config', night_mode=self.parent().po.all['night_mode'])
        self.visualize.clicked.connect(self.visualize_is_clicked)
        if self.parent().po.vars['already_greyscale']:
            self.visualize_label = FixedText("Directly: ", night_mode=self.parent().po.all['night_mode'])
        else:
            self.visualize_label = FixedText("Or directly: ", night_mode=self.parent().po.all['night_mode'])

        self.sup_param_row1_layout.addWidget(self.generate_analysis_options)
        self.sup_param_row1_layout.addWidget(self.basic)
        self.sup_param_row1_layout.addWidget(self.network_shaped)
        self.sup_param_row1_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.sup_param_row1_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.sup_param_row1_layout.addWidget(self.visualize_label)
        self.sup_param_row1_layout.addWidget(self.visualize)

        self.sup_param_row1_widget.setLayout(self.sup_param_row1_layout)
        self.Vlayout.addWidget(self.sup_param_row1_widget)

        # 6) Open the choose best option row layout
        self.options_row_widget = QtWidgets.QWidget()
        self.options_row_layout = QtWidgets.QHBoxLayout()
        self.select_option_label = FixedText(IAW["Select_option_to_read"]["label"],
                                             tip=IAW["Select_option_to_read"]["tips"],
                                             night_mode=self.parent().po.all['night_mode'])
        self.select_option = Combobox([], night_mode=self.parent().po.all['night_mode'])
        if self.parent().po.vars['color_number'] == 2:
            self.select_option.setCurrentIndex(self.parent().po.all['video_option'])
        self.select_option.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.select_option.setMinimumWidth(145)
        self.select_option.currentTextChanged.connect(self.option_changed)
        self.n_shapes_detected = FixedText(f'', night_mode=self.parent().po.all['night_mode'])
        self.select_option_label.setVisible(False)
        self.select_option.setVisible(False)
        self.n_shapes_detected.setVisible(False)
        self.options_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.options_row_layout.addWidget(self.select_option_label)
        self.options_row_layout.addWidget(self.select_option)
        self.options_row_layout.addWidget(self.n_shapes_detected)
        self.options_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.options_row_widget.setLayout(self.options_row_layout)
        self.Vlayout.addWidget(self.options_row_widget)

        # 7) Open decision row layout
        self.decision_row_widget = QtWidgets.QWidget()
        self.decision_row_layout = QtWidgets.QHBoxLayout()
        self.decision_label = FixedText("", night_mode=self.parent().po.all['night_mode'])
        self.yes = PButton("Yes", night_mode=self.parent().po.all['night_mode'])
        self.yes.clicked.connect(self.when_yes_is_clicked)
        self.no = PButton("No", night_mode=self.parent().po.all['night_mode'])
        self.no.clicked.connect(self.when_no_is_clicked)

        self.decision_label.setVisible(False)
        self.yes.setVisible(False)
        self.no.setVisible(False)
        self.decision_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.decision_row_layout.addWidget(self.decision_label)
        self.decision_row_layout.addWidget(self.yes)
        self.decision_row_layout.addWidget(self.no)
        self.decision_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.decision_row_widget.setLayout(self.decision_row_layout)
        self.Vlayout.addWidget(self.decision_row_widget)

        # 8) Open the special cases layout
        self.special_cases_widget = QtWidgets.QWidget()
        self.special_cases_layout = QtWidgets.QHBoxLayout()
        self.starting_differs_from_growing_cb = Checkbox(self.parent().po.vars['origin_state'] == 'constant')
        self.starting_differs_from_growing_cb.stateChanged.connect(self.starting_differs_from_growing_check)
        self.starting_differs_from_growing_label = FixedText(IAW["Start_differs_from_arena"]["label"],
                                                             tip=IAW["Start_differs_from_arena"]["tips"],
                                                             night_mode=self.parent().po.all['night_mode'])
        self.starting_differs_from_growing_cb.setVisible(False)
        self.starting_differs_from_growing_label.setVisible(False)
        self.special_cases_layout.addWidget(self.starting_differs_from_growing_cb)
        self.special_cases_layout.addWidget(self.starting_differs_from_growing_label)
        self.special_cases_widget.setLayout(self.special_cases_layout)
        self.Vlayout.addWidget(self.special_cases_widget)

        # 9) Open the last row layout
        self.last_row_widget = QtWidgets.QWidget()
        self.last_row_layout = QtWidgets.QHBoxLayout()
        self.previous = PButton('Previous', night_mode=self.parent().po.all['night_mode'])
        self.previous.clicked.connect(self.previous_is_clicked)
        self.data_tab.clicked.connect(self.data_is_clicked)
        self.video_tab.clicked.connect(self.video_is_clicked)
        self.complete_image_analysis = PButton(IAW["Save_image_analysis"]["label"],
                                               tip=IAW["Save_image_analysis"]["tips"],
                                               night_mode=self.parent().po.all['night_mode'])
        self.complete_image_analysis.setVisible(False)
        self.complete_image_analysis.clicked.connect(self.complete_image_analysis_is_clicked)
        self.next = PButton("Next", night_mode=self.parent().po.all['night_mode'])
        self.next.setVisible(False)
        self.next.clicked.connect(self.go_to_next_widget)
        self.last_row_layout.addWidget(self.previous)
        self.last_row_layout.addWidget(self.message)
        self.last_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.last_row_layout.addWidget(self.complete_image_analysis)
        self.last_row_layout.addWidget(self.next)
        self.last_row_widget.setLayout(self.last_row_layout)
        self.Vlayout.addWidget(self.last_row_widget)
        self.Vlayout.setSpacing(0)
        self.setLayout(self.Vlayout)

        self.advanced_mode_check()

        self.thread_dict = {}
        self.thread_dict["GetFirstIm"] = GetFirstImThread(self.parent())
        self.reinitialize_image_and_masks(self.parent().po.first_im)
        self.thread_dict["GetLastIm"] = GetLastImThread(self.parent())
        if self.parent().po.all['im_or_vid'] == 0:
            self.thread_dict["GetLastIm"].start()
        self.parent().po.first_image = OneImageAnalysis(self.parent().po.first_im)
        self.thread_dict["FirstImageAnalysis"] = FirstImageAnalysisThread(self.parent())
        self.thread_dict["LastImageAnalysis"] = LastImageAnalysisThread(self.parent())
        self.thread_dict['UpdateImage'] = UpdateImageThread(self.parent())
        self.thread_dict['CropScaleSubtractDelineate'] = CropScaleSubtractDelineateThread(self.parent())
        self.thread_dict['SaveManualDelineation'] = SaveManualDelineationThread(self.parent())
        self.thread_dict['CompleteImageAnalysisThread'] = CompleteImageAnalysisThread(self.parent())
        self.thread_dict['PrepareVideoAnalysis'] = PrepareVideoAnalysisThread(self.parent())

    def previous_is_clicked(self):
        """
        Handles the logic for when a "Previous" button is clicked in the interface, leading to the FirstWindow.

        This method resets various flags and variables related to image analysis
        to their initial state. It is called when the "Previous" button is clicked,
        preparing the application for new input and reinitialization.
        """
        if self.is_image_analysis_running:
            self.message.setText("Wait for the analysis to end, or restart Cellects")
        else:
            self.parent().firstwindow.instantiate = True
            self.hold_click_flag: bool = False
            self.is_first_image_flag: bool = True
            self.is_image_analysis_running: bool = False
            self.is_image_analysis_display_running: bool = False
            self.asking_first_im_parameters_flag: bool = True
            self.first_im_parameters_answered: bool = False
            self.auto_delineation_flag: bool = False
            self.delineation_done: bool = False
            self.asking_delineation_flag: bool = False
            self.asking_slower_or_manual_delineation_flag: bool = False
            self.slower_delineation_flag: bool = False
            self.asking_last_image_flag: bool = False
            self.step = 0
            self.temporary_mask_coord = []
            self.saved_coord = []
            self.back1_bio2 = 0
            self.bio_masks_number = 0
            self.back_masks_number = 0
            self.arena_masks_number = 0
            self.available_bio_names = np.arange(1, 1000, dtype=np.uint16)
            self.available_back_names = np.arange(1, 1000, dtype=np.uint16)
            self.parent().po.current_combination_id = 0
            self.parent().last_tab = "data_specifications"
            self.parent().change_widget(0)  # First

    def data_is_clicked(self):
        """
        Handles the logic for when the "Data specifications" button is clicked in the interface,
        leading to the FirstWindow.

        Notes
        -----
        This function displays an error message when a thread relative to the current window is running.
        This function also save the id of this tab for later use.
        """
        if self.is_image_analysis_running:
            self.message.setText("Wait for the analysis to end, or restart Cellects")
        else:
            self.parent().last_tab = "data_specifications"
            self.parent().change_widget(0)  # First

    def video_is_clicked(self):
        """
        Handles the logic for when the "Video tracking" button is clicked in the interface,
        leading to the video analysis window.

        Notes
        -----
        This function displays an error message when a thread relative to the current window is running.
        This function also save the id of the following window for later use.
        """
        if self.video_tab.state != "not_usable":
            if self.is_image_analysis_running:
                self.message.setText("Wait for the analysis to end, or restart Cellects")
            else:
                self.parent().last_tab = "image_analysis"
                self.parent().change_widget(3)

    def read_is_clicked(self):
        """
        Read an image (numbered using natural sorting) from the selected folder

        This method handles the logic for starting image reading when the "Read" button is clicked.
        It ensures that only one thread runs at a time, updates the UI with relevant messages,
        and resets visual components once processing begins.
        """
        if not self.thread_dict["GetFirstIm"].isRunning():
            self.parent().po.vars['first_detection_frame'] = int(self.image_number.value())
            self.message.setText(f"Reading image n°{self.parent().po.vars['first_detection_frame']}")
            self.thread_dict["GetFirstIm"].start()
            self.thread_dict["GetFirstIm"].message_when_thread_finished.connect(self.reinitialize_image_and_masks)
            self.reinitialize_bio_and_back_legend()


    def several_blob_per_arena_check(self):
        """
        Checks or unchecks the option for having several blobs per arena.
        """
        is_checked = self.one_blob_per_arena.isChecked()
        self.parent().po.vars['several_blob_per_arena'] = not is_checked
        self.set_spot_size.setVisible(is_checked)
        self.spot_size_label.setVisible(is_checked)
        self.spot_size.setVisible(is_checked and self.set_spot_size.isChecked())

    def set_spot_size_check(self):
        """
        Set the visibility of spot size based on checkbox state.
        """
        is_checked = self.set_spot_size.isChecked()
        if self.step == 1:
            self.spot_size.setVisible(is_checked)
        self.parent().po.all['set_spot_size'] = is_checked

    def spot_size_changed(self):
        """
        Update the starting blob size and corresponding horizontal size based on user input.
        """
        self.parent().po.all['starting_blob_hsize_in_mm'] = self.spot_size.value()
        if self.parent().po.all['scale_with_image_or_cells'] == 1:
            self.horizontal_size.setValue(self.parent().po.all['starting_blob_hsize_in_mm'])
        self.set_spot_size_check()

    def set_spot_shape_check(self):
        """
        Set the spot shape setting visibility.
        """
        is_checked = self.set_spot_shape.isChecked()
        self.spot_shape.setVisible(is_checked)
        self.parent().po.all['set_spot_shape'] = is_checked
        if not is_checked:
            self.parent().po.all['starting_blob_shape'] = None

    def spot_shape_changed(self):
        """
        Save the user selection of shape.
        """
        self.parent().po.all['starting_blob_shape'] = self.spot_shape.currentText()
        self.set_spot_shape_check()

    def arena_shape_changed(self):
        """
        Calculate and update the arena shape in response to user input and manage threading operations.

        Extended Description
        --------------------
        This method updates the arena shape variable based on user selection from a dropdown menu.
        It ensures that certain background threading operations are completed before proceeding with updates
        and reinitializes necessary components to reflect the new arena shape.

        Notes
        -----
        This method handles threading operations to ensure proper synchronization and updates.
        It reinitializes the biological legend, image, and masks when the arena shape is changed.
        """
        self.parent().po.vars['arena_shape'] = self.arena_shape.currentText()
        if self.asking_delineation_flag:
            if self.thread_dict['CropScaleSubtractDelineate'].isRunning():
                self.thread_dict['CropScaleSubtractDelineate'].wait()
            if self.thread_dict['UpdateImage'].isRunning():
                self.thread_dict['UpdateImage'].wait()
            self.message.setText("Updating display...")
            self.decision_label.setVisible(False)
            self.yes.setVisible(False)
            self.no.setVisible(False)
            self.reinitialize_bio_and_back_legend()
            self.reinitialize_image_and_masks(self.parent().po.first_image.bgr)
            self.delineation_done = True
            if self.thread_dict["UpdateImage"].isRunning():
                self.thread_dict["UpdateImage"].wait()
            self.thread_dict["UpdateImage"].start()
            self.thread_dict["UpdateImage"].message_when_thread_finished.connect(self.automatic_delineation_display_done)

    def reinitialize_bio_and_back_legend(self):
        """
        Reinitialize the bio and back legend.

        Reinitializes the bio and back legends, removing all existing lines
        and resetting counters for masks. This function ensures that the UI
        components associated with bio and back lines are correctly cleaned up.
        """
        lines_names_to_remove = []
        for line_number, back_line_dict in self.back_lines.items():
            line_name = u"\u00D7" + " Back" + str(line_number)
            self.back_added_lines_layout.removeWidget(back_line_dict[line_name])
            back_line_dict[line_name].deleteLater()
            lines_names_to_remove.append(line_number)
        for line_number in lines_names_to_remove:
            self.back_lines.pop(line_number)
        lines_names_to_remove = []
        for line_number, bio_line_dict in self.bio_lines.items():
            line_name = u"\u00D7" + " Cell" + str(line_number)
            self.bio_added_lines_layout.removeWidget(bio_line_dict[line_name])
            bio_line_dict[line_name].deleteLater()
            lines_names_to_remove.append(line_number)
        for line_number in lines_names_to_remove:
            self.bio_lines.pop(line_number)
        if len(self.arena_lines) > 0:
            lines_names_to_remove = []
            for i, (line_number, arena_line_dict) in enumerate(self.arena_lines.items()):
                line_name = u"\u00D7" + " Arena" + str(line_number)
                if i % 2 == 0:
                    self.bio_added_lines_layout.removeWidget(arena_line_dict[line_name])
                else:
                    self.back_added_lines_layout.removeWidget(arena_line_dict[line_name])
                arena_line_dict[line_name].deleteLater()
                lines_names_to_remove.append(line_number)
            for line_number in lines_names_to_remove:
                self.arena_lines.pop(line_number)
        self.bio_masks_number = 0
        self.back_masks_number = 0

    def reinitialize_image_and_masks(self, image: np.ndarray):
        """
        Reinitialize the image and masks for analysis.

        This method reinitializes the current image and its associated masks
        used in the analysis process. It checks if the input image is grayscale
        and converts it to a 3-channel RGB image, stacking identical channels.
        It also updates the visibility of various UI components based on
        the image type and reinitializes masks to prepare for new analysis.
        """
        if len(image.shape) == 2:
            self.parent().po.current_image = np.stack((image, image, image), axis=2)

            self.generate_analysis_options.setVisible(False)
            self.network_shaped.setVisible(False)
            self.basic.setVisible(False)
            self.select_option.setVisible(False)
            self.select_option_label.setVisible(False)
            self.visualize.setVisible(True)
            self.visualize_label.setVisible(True)
        else:
            self.parent().po.current_image = deepcopy(image)
        self.drawn_image = deepcopy(self.parent().po.current_image)
        self.display_image.update_image(self.parent().po.current_image)
        self.arena_mask = None
        self.bio_mask = np.zeros(self.parent().po.current_image.shape[:2], dtype=np.uint16)
        self.back_mask = np.zeros(self.parent().po.current_image.shape[:2], dtype=np.uint16)

    def scale_with_changed(self):
        """
        Modifies how the image scale is computed: using the image width or the blob unitary size (horizontal diameter).
        """
        self.parent().po.all['scale_with_image_or_cells'] = self.scale_with.currentIndex()
        if self.parent().po.all['scale_with_image_or_cells'] == 0:
            self.horizontal_size.setValue(self.parent().po.all['image_horizontal_size_in_mm'])
        else:
            self.horizontal_size.setValue(self.parent().po.all['starting_blob_hsize_in_mm'])

    def horizontal_size_changed(self):
        """
        Changes the horizontal size value of the image or of the blobs in the image, depending on user's choice.
        """
        if self.parent().po.all['scale_with_image_or_cells'] == 0:
            self.parent().po.all['image_horizontal_size_in_mm'] = self.horizontal_size.value()
        else:
            self.parent().po.all['starting_blob_hsize_in_mm'] = self.horizontal_size.value()
            self.spot_size.setValue(self.parent().po.all['starting_blob_hsize_in_mm'])

    def advanced_mode_check(self):
        """
        Update widget visibility based on advanced mode check.

        This function updates the visbility of various UI elements depending on
        the state of the advanced mode check box and other conditions.
        """
        is_checked = self.advanced_mode_cb.isChecked()
        color_analysis = is_checked and not self.parent().po.vars['already_greyscale']
        self.parent().po.all['expert_mode'] = is_checked

        if is_checked and (self.asking_first_im_parameters_flag or self.auto_delineation_flag):
            self.arena_shape_label.setVisible(True)
            self.arena_shape.setVisible(True)
            self.set_spot_shape.setVisible(True)
            self.spot_shape_label.setVisible(True)
            self.spot_shape.setVisible(self.set_spot_shape.isChecked())
            self.set_spot_size.setVisible(self.one_blob_per_arena.isChecked())
            self.spot_size_label.setVisible(self.one_blob_per_arena.isChecked())
            self.spot_size.setVisible(
                self.one_blob_per_arena.isChecked() and self.set_spot_size.isChecked())
            self.first_im_parameters_answered = True

        self.space_label.setVisible(color_analysis)
        display_logical = self.logical_operator_between_combination_result.currentText() != 'None'
        self.logical_operator_between_combination_result.setVisible(color_analysis and display_logical)
        self.logical_operator_label.setVisible(color_analysis and display_logical)

        at_least_one_line_drawn = self.bio_masks_number > 0
        self.more_than_two_colors.setVisible(is_checked and at_least_one_line_drawn)
        self.more_than_two_colors_label.setVisible(is_checked and at_least_one_line_drawn)
        self.distinct_colors_number.setVisible(is_checked and at_least_one_line_drawn and self.parent().po.all["more_than_two_colors"])

        # Check whether filter 1 and its potential parameters should be visible
        self.filter1.setVisible(is_checked)
        self.filter1_label.setVisible(is_checked)
        has_param1 = is_checked and 'Param1' in filter_dict[self.filter1.currentText()]
        self.filter1_param1.setVisible(has_param1)
        self.filter1_param1_label.setVisible(has_param1)
        has_param2 = is_checked and 'Param2' in filter_dict[self.filter1.currentText()]
        self.filter1_param2.setVisible(has_param2)
        self.filter1_param2_label.setVisible(has_param2)

        # Check whether filter 2 and its potential parameters should be visible
        self.filter2.setVisible(is_checked and display_logical)
        self.filter2_label.setVisible(is_checked and display_logical)
        has_param1 = is_checked and display_logical and 'Param1' in filter_dict[self.filter2.currentText()]
        self.filter2_param1.setVisible(has_param1)
        self.filter2_param1_label.setVisible(has_param1)
        has_param2 = is_checked and display_logical and 'Param2' in filter_dict[self.filter2.currentText()]
        self.filter2_param2.setVisible(has_param2)
        self.filter2_param2_label.setVisible(has_param2)

        self.rolling_window_segmentation.setVisible(is_checked)
        self.rolling_window_segmentation_label.setVisible(is_checked)

        for i in range(5):
            if i == 0:
                self.row1[i].setVisible(color_analysis)
            else:
                self.row1[i].setVisible(color_analysis and not "PCA" in self.csc_dict)
            self.row21[i].setVisible(color_analysis and self.row21[0].currentText() != "None")
            self.row2[i].setVisible(color_analysis and self.row2[0].currentText() != "None")
            self.row22[i].setVisible(color_analysis and self.row22[0].currentText() != "None")
            if i < 4:
                self.row3[i].setVisible(color_analysis and self.row3[0].currentText() != "None")
                self.row23[i].setVisible(color_analysis and self.row23[0].currentText() != "None")
        if color_analysis:
            if self.row1[0].currentText() != "PCA":
                if self.row2[0].currentText() == "None":
                    self.row1[4].setVisible(True)
                else:
                    self.row2[4].setVisible(True)
            if self.row21[0].currentText() != "None":
                if self.row22[0].currentText() == "None":
                    self.row21[4].setVisible(True)
                else:
                    self.row22[4].setVisible(True)
        else:
            self.row1[4].setVisible(False)
            self.row2[4].setVisible(False)
            self.row21[4].setVisible(False)
            self.row22[4].setVisible(False)

    def cell_is_clicked(self):
        """
        Handles the logic for when a "cell" button is clicked in the interface,
        allowing the user to draw cells on the image.
        """
        if self.back1_bio2 == 2:
            self.cell.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
            self.back1_bio2 = 0
        else:
            self.cell.color("rgb(230, 145, 18)")
            self.background.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
            self.back1_bio2 = 2
        self.saved_coord = []

    def background_is_clicked(self):
        """
        Handles the logic for when a "back" button is clicked in the interface,
        allowing the user to draw where there is background on the image.
        """
        if self.back1_bio2 == 1:
            self.background.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
            self.back1_bio2 = 0
        else:
            self.background.color("rgb(81, 160, 224)")
            self.cell.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
            self.back1_bio2 = 1
        self.saved_coord = []

    def get_click_coordinates(self, event):
        """
        Handle mouse click events to capture coordinate data or display an image.

        This function determines the handling of click events based on various
        flags and states, including whether image analysis is running or if a
        manual delineation flag is set.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event that triggered the function.
        """
        if self.back1_bio2 > 0 or self.manual_delineation_flag:
            if not self.is_image_analysis_display_running and not self.thread_dict["UpdateImage"].isRunning():
                self.hold_click_flag = True
                self.saved_coord.append([event.pos().y(), event.pos().x()])
        else:
            self.popup_img = FullScreenImage(self.drawn_image, self.parent().screen_width, self.parent().screen_height)
            self.popup_img.show()

    def get_mouse_move_coordinates(self, event):
        """
        Handles mouse movement events to update the temporary mask coordinate.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event object containing position information.
        """
        if self.hold_click_flag:
            if not self.thread_dict["UpdateImage"].isRunning():
                if self.saved_coord[0][0] != event.pos().y() and self.saved_coord[0][1] != event.pos().x():
                    self.temporary_mask_coord = [self.saved_coord[0], [event.pos().y(), event.pos().x()]]
                    self.thread_dict["UpdateImage"].start()

    def get_mouse_release_coordinates(self, event):
        """
        Process mouse release event to save coordinates and manage image update thread.

        This method handles the logic for saving mouse release coordinates during
        manual delineation, checks conditions to prevent exceeding the number of arenas,
        and manages an image update thread for display purposes.

        Parameters
        ----------
        event : QMouseEvent
            The mouse event containing the release position.

        Notes
        -----
        This method requires an active image update thread and assumes certain attributes
        like `hold_click_flag`, `manual_delineation_flag`, etc., are part of the class
        state.
        """
        if self.hold_click_flag:
            if self.thread_dict["UpdateImage"].isRunning():
                self.thread_dict["UpdateImage"].wait()
            self.temporary_mask_coord = []
            if self.manual_delineation_flag and len(self.parent().imageanalysiswindow.available_arena_names) == 0:
                self.message.setText(f"The total number of arenas are already drawn ({self.parent().po.sample_number})")
                self.saved_coord = []
            else:
                self.saved_coord.append([event.pos().y(), event.pos().x()])
                self.thread_dict["UpdateImage"].start()
                self.thread_dict["UpdateImage"].message_when_thread_finished.connect(self.user_defined_shape_displayed)
            self.hold_click_flag = False

    def user_defined_shape_displayed(self, when_finished: bool):
        """
        Display user-defined shapes or elements based on specific conditions and update the UI accordingly.

        Parameters
        ----------
        when_finished : bool
            A flag indicating whether a certain operation has finished.

        Notes
        -----
        This method modifies the user interface by adding buttons and updating layouts based on the current state and conditions.
        """
        if self.back1_bio2 == 1:
            back_name = self.parent().imageanalysiswindow.available_back_names[0]
            self.back_lines[back_name] = {}
            pbutton_name = u"\u00D7" + " Back" + str(back_name)
            self.back_lines[back_name][pbutton_name] = self.new_pbutton_on_the_left(pbutton_name)
            self.back_added_lines_layout.addWidget(self.back_lines[back_name][pbutton_name])
            self.background.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
            self.available_back_names = self.available_back_names[1:]
        elif self.back1_bio2 == 2:
            bio_name = self.parent().imageanalysiswindow.available_bio_names[0]
            self.bio_lines[bio_name] = {}
            pbutton_name = u"\u00D7" + " Cell" + str(bio_name)
            self.bio_lines[bio_name][pbutton_name] = self.new_pbutton_on_the_left(pbutton_name)
            self.bio_added_lines_layout.addWidget(self.bio_lines[bio_name][pbutton_name])
            self.cell.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
            self.available_bio_names = self.available_bio_names[1:]
            if self.bio_masks_number == 0:
                self.display_more_than_two_colors_option()

            self.more_than_two_colors.setVisible(self.advanced_mode_cb.isChecked())
            self.more_than_two_colors_label.setVisible(self.advanced_mode_cb.isChecked())
            self.distinct_colors_number.setVisible(self.advanced_mode_cb.isChecked() and self.more_than_two_colors.isChecked())
        elif self.manual_delineation_flag:
            arena_name = self.parent().imageanalysiswindow.available_arena_names[0]
            self.arena_lines[arena_name] = {}
            pbutton_name = u"\u00D7" + " Arena" + str(arena_name)
            self.arena_lines[arena_name][pbutton_name] = self.new_pbutton_on_the_left(pbutton_name)
            if self.arena_masks_number % 2 == 1:
                self.bio_added_lines_layout.addWidget(self.arena_lines[arena_name][pbutton_name])
            else:
                self.back_added_lines_layout.addWidget(self.arena_lines[arena_name][pbutton_name])
            self.available_arena_names = self.available_arena_names[1:]
        self.saved_coord = []
        self.back1_bio2 = 0

        self.thread_dict["UpdateImage"].message_when_thread_finished.disconnect()

    def new_pbutton_on_the_left(self, pbutton_name: str):
        """
        Create a styled PButton instance positioned on the left of the image.

        Notes
        -----
        The button's appearance is customized based on the value of
        `self.back1_bio2`, which affects its color. The button also has a fixed
        size and specific font settings.
        """
        pbutton = PButton(pbutton_name, False, night_mode=self.parent().po.all['night_mode'])
        pbutton.setFixedHeight(20)
        pbutton.setFixedWidth(100)
        pbutton.setFont(QtGui.QFont("Segoe UI Semibold", 8, QtGui.QFont.Thin))
        pbutton.textcolor("rgb(0, 0, 0)")
        pbutton.border("0px")
        pbutton.angles("10px")
        if self.back1_bio2 == 1:
            pbutton.color("rgb(81, 160, 224)")
        elif self.back1_bio2 == 2:
            pbutton.color("rgb(230, 145, 18)")
        else:
            pbutton.color("rgb(126, 126, 126)")
        pbutton.clicked.connect(self.remove_line)
        return pbutton

    def remove_line(self):
        """
        Remove the specified line from the image analysis display.

        This method removes a line identified by its button name from the appropriate mask
        and updates the layout and available names accordingly. It starts the image update thread
        after removing the line.
        """
        if not self.is_image_analysis_display_running and not self.thread_dict["UpdateImage"].isRunning() and hasattr(self.sender(), 'text'):
            pbutton_name = self.sender().text()
            if pbutton_name[2:6] == "Back":
                line_name = np.uint8(pbutton_name[6:])
                self.back_mask[self.back_mask == line_name] = 0
                self.back_added_lines_layout.removeWidget(self.back_lines[line_name][pbutton_name])
                self.back_lines[line_name][pbutton_name].deleteLater()
                self.back_lines.pop(line_name)
                self.back_masks_number -= 1
                self.available_back_names = np.sort(np.concatenate(([line_name], self.available_back_names)))
            elif pbutton_name[2:6] == "Cell":
                line_name = np.uint8(pbutton_name[6:])
                self.bio_mask[self.bio_mask == line_name] = 0
                self.bio_added_lines_layout.removeWidget(self.bio_lines[line_name][pbutton_name])
                self.bio_lines[line_name][pbutton_name].deleteLater()
                self.bio_lines.pop(line_name)
                self.bio_masks_number -= 1
                self.available_bio_names = np.sort(np.concatenate(([line_name], self.available_bio_names)))
                self.display_more_than_two_colors_option()
            else:
                line_name = np.uint8(pbutton_name[7:])
                self.arena_mask[self.arena_mask == line_name] = 0
                if line_name % 2 == 1:
                    self.bio_added_lines_layout.removeWidget(self.arena_lines[line_name][pbutton_name])
                else:
                    self.back_added_lines_layout.removeWidget(self.arena_lines[line_name][pbutton_name])
                self.arena_lines[line_name][pbutton_name].deleteLater()
                self.arena_lines.pop(line_name)

                self.arena_masks_number -= 1
                self.available_arena_names = np.sort(np.concatenate(([line_name], self.available_arena_names)))
            self.thread_dict["UpdateImage"].start()

    def network_shaped_is_clicked(self):
        """
        Sets the GUI state for analyzing a network-shaped image when clicked.

        This method triggers the analysis process for a network-shaped image. It ensures that image analysis is not
        already running, updates GUI elements accordingly, and starts the appropriate analysis function based on a flag.
        """
        if not self.is_image_analysis_running:
            self.is_image_analysis_running = True
            self.message.setText('Loading, wait...')
            self.parent().po.visualize = False
            self.parent().po.basic = False
            self.parent().po.network_shaped = True
            self.select_option.clear()
            if self.is_first_image_flag:
                self.run_first_image_analysis()
            else:
                self.run_last_image_analysis()

    def basic_is_clicked(self):
        """
        Toggle image analysis mode and trigger appropriate image analysis process.

        This method enables the image analysis mode, sets a loading message,
        and initiates either the first or last image analysis based on
        the current state.
        """
        if not self.is_image_analysis_running:
            self.is_image_analysis_running = True
            self.message.setText('Loading, wait...')
            self.parent().po.visualize = False
            self.parent().po.basic = True
            self.parent().po.network_shaped = False
            if self.is_first_image_flag:
                self.run_first_image_analysis()
            else:
                self.run_last_image_analysis()

    def visualize_is_clicked(self):
        """
        Instructs the system to perform an image analysis and updates the UI accordingly.

        If image analysis is not currently running, this method triggers the analysis process
        and updates the UI message to indicate loading.
        """
        if not self.is_image_analysis_running:
            self.is_image_analysis_running = True
            self.message.setText('Loading, wait...')
            self.parent().po.visualize = True
            self.parent().po.basic = False
            self.parent().po.network_shaped = False
            if self.is_first_image_flag:
                self.run_first_image_analysis()
            else:
                self.run_last_image_analysis()

    def run_first_image_analysis(self):
        """
        Run the first image analysis.

        This method performs a series of checks and updates based on user-defined parameters
        before running the first image analysis. If visualization is enabled, it saves user-defined
        combinations and checks for empty color selection dictionaries. It then starts the thread
        for image analysis.

        Notes
        -----
        This method assumes that the parent object has already been initialized and contains all
        necessary variables for image analysis.
        """
        if self.first_im_parameters_answered:
            self.several_blob_per_arena_check()
            self.horizontal_size_changed()
            self.spot_shape_changed()
            self.arena_shape_changed()

        if self.parent().po.visualize:
            self.save_user_defined_csc()
            self.parent().po.vars["color_number"] = int(self.distinct_colors_number.value())
            if self.csc_dict_is_empty:
                self.message.setText('Select non null value(s) to combine colors')
                self.message.setStyleSheet("color: rgb(230, 145, 18)")
                self.is_image_analysis_running = False
        if not self.parent().po.visualize or not self.csc_dict_is_empty:
            self.parent().po.vars['convert_for_origin'] = self.csc_dict.copy()
            self.thread_dict["FirstImageAnalysis"].start()
            self.thread_dict["FirstImageAnalysis"].message_from_thread.connect(self.display_message_from_thread)
            self.thread_dict["FirstImageAnalysis"].message_when_thread_finished.connect(self.when_image_analysis_finishes)

    def run_last_image_analysis(self):
        """
        Run the last image analysis thread.

        This function updates relevant variables, saves user-defined color-space configurations (CSC),
        and manages thread operations for image analysis. The function does not handle any direct processing but
        prepares the environment by setting variables and starting threads.
        """
        self.save_user_defined_csc()
        self.parent().po.vars["color_number"] = int(self.distinct_colors_number.value())
        if not self.csc_dict_is_empty:
            self.parent().po.vars['convert_for_motion'] = self.csc_dict.copy()
        if self.parent().po.visualize and self.csc_dict_is_empty:
            self.message.setText('Select non null value(s) to combine colors')
            self.message.setStyleSheet("color: rgb(230, 145, 18)")
        else:
            self.thread_dict["LastImageAnalysis"].start()
            self.thread_dict["LastImageAnalysis"].message_from_thread.connect(self.display_message_from_thread)
            self.thread_dict["LastImageAnalysis"].message_when_thread_finished.connect(self.when_image_analysis_finishes)

    def when_image_analysis_finishes(self):
        """
        Logs the completion of an image analysis operation, updates the current combination ID,
        handles visualization settings, manages image combinations, and updates the display.

        Notes
        -----
        - This method interacts with the parent object's properties and thread management.
        - The `is_first_image_flag` determines which set of image combinations to use.
        """

        if self.is_first_image_flag:
            im_combinations = self.parent().po.first_image.im_combinations
        else:
            im_combinations = self.parent().po.last_image.im_combinations
        self.init_drawn_image(im_combinations)
        if self.parent().po.visualize:
            if self.parent().po.current_combination_id != self.select_option.currentIndex():
                self.select_option.setCurrentIndex(self.parent().po.current_combination_id)
        else:
            self.parent().po.current_combination_id = 0
            if len(im_combinations) > 0:
                self.csc_dict = im_combinations[self.parent().po.current_combination_id]["csc"]
                if self.is_first_image_flag:
                    self.parent().po.vars['convert_for_origin'] = self.csc_dict.copy()
                else:
                    self.parent().po.vars['convert_for_motion'] = self.csc_dict.copy()
                option_number = len(im_combinations)

                if option_number > 1:
                    # Update the available options of the scrolling menu
                    self.select_option.clear()
                    for option in range(option_number):
                        self.select_option.addItem(f"Option {option + 1}")
                self.update_csc_editing_display()
            else:
                self.message.setText("No options could be generated automatically, use the advanced mode")
                self.is_image_analysis_running = False

        if self.parent().po.visualize or len(im_combinations) > 0:
            self.is_image_analysis_display_running = True
            # Update image display
            if self.thread_dict["UpdateImage"].isRunning():
                self.thread_dict["UpdateImage"].wait()
            self.thread_dict["UpdateImage"].start()
            self.thread_dict["UpdateImage"].message_when_thread_finished.connect(self.image_analysis_displayed)

    def image_analysis_displayed(self):
        """
        Display results of image analysis based on the current step and configuration.

        Update the user interface elements based on the current step of image analysis,
        the detected number of shapes, and whether color analysis is enabled. Handles
        visibilities of buttons and labels to guide the user through the process.

        Notes
        -----
        This method updates the user interface based on the current state of image analysis.
        """
        color_analysis = not self.parent().po.vars['already_greyscale']
        self.message.setText("")

        if self.step < 2:
            detected_shape_nb = self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id][
                'shape_number']
            if detected_shape_nb == self.parent().po.sample_number or self.parent().po.vars['several_blob_per_arena']:
                self.decision_label.setText(
                    f"{detected_shape_nb} distinct specimen(s) detected in {self.parent().po.sample_number} arena(s). Does the color match the cell(s)?")
                if self.step == 1:
                    self.yes.setVisible(True)
                    self.message.setText("If not, draw more Cell and Back ellipses on the image and retry")
            else:
                if self.no.isVisible():
                    self.decision_label.setText(
                        f"{detected_shape_nb} distinct specimen(s) detected in {self.parent().po.sample_number} arena(s). Click Yes when satisfied, Click No to fill in more parameters")
                    self.yes.setVisible(True)
                    self.no.setVisible(True)
                else:
                    self.decision_label.setText(
                        f"{detected_shape_nb} distinct specimen(s) detected in {self.parent().po.sample_number} arena(s). Click Yes when satisfied")
                    self.yes.setVisible(True)

            if self.parent().po.vars['several_blob_per_arena'] and (detected_shape_nb == self.parent().po.sample_number):
                self.message.setText("Beware: Contrary to what has been checked, there is one spot per arena")

        if not self.parent().po.visualize:
            self.select_option.setVisible(color_analysis)
            self.select_option_label.setVisible(color_analysis)
        if self.step == 0:
            if self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id]['shape_number'] == 0:
                self.message.setText("Make sure that scaling metric and spot size are correct")
            self.decision_label.setVisible(True)
            self.yes.setVisible(True)
            self.no.setVisible(True)
            self.arena_shape.setVisible(True)
            self.arena_shape_label.setVisible(True)
            self.n_shapes_detected.setVisible(True)

        elif self.step == 2:
            self.generate_analysis_options.setVisible(color_analysis)
            self.network_shaped.setVisible(True)
            self.basic.setVisible(color_analysis)
            self.visualize.setVisible(True)

            self.decision_label.setText("Adjust parameters until the color delimits the specimen(s) correctly")
            self.yes.setVisible(False)
            self.no.setVisible(False)
            if self.parent().po.all["im_or_vid"] == 1 or len(self.parent().po.data_list) > 1:
                self.next.setVisible(True)
                self.message.setText('When the resulting segmentation of the last image seems good, click next.')
            else:
                self.video_tab.set_not_usable()
                self.message.setText('When the resulting segmentation of the last image seems good, save image analysis.')
            self.complete_image_analysis.setVisible(True)

        self.thread_dict["UpdateImage"].message_when_thread_finished.disconnect()
        self.is_image_analysis_running = False
        self.is_image_analysis_display_running = False

    def init_drawn_image(self, im_combinations: list=None):
        """
        Initialize the drawn image from a list of image combinations.

        Parameters
        ----------
        im_combinations : list or None, optional
            List of image combinations to initialize the drawn image from.
            Each combination should be a dictionary containing 'csc' and
            'converted_image'. If None, the current state is maintained.
        """
        if im_combinations is not None and len(im_combinations) > 0:
            if self.parent().po.current_combination_id + 1 > len(im_combinations):
                self.parent().po.current_combination_id = 0
            self.csc_dict = im_combinations[self.parent().po.current_combination_id]["csc"]
            self.parent().po.current_image = np.stack((im_combinations[self.parent().po.current_combination_id]['converted_image'],
                                                    im_combinations[self.parent().po.current_combination_id]['converted_image'],
                                                    im_combinations[self.parent().po.current_combination_id]['converted_image']), axis=2)
            self.drawn_image = deepcopy(self.parent().po.current_image)

    def option_changed(self):
        """
        Update the current image and related display information based on the selected image segmentation option.

        Notes
        -----
        This function updates several properties of the parent object, including the current image,
        combination ID, and display settings. It also handles thread management for updating the
        image display.
        """
        # Update the current image
        self.parent().po.current_combination_id = self.select_option.currentIndex()
        if self.is_first_image_flag:
            im_combinations = self.parent().po.first_image.im_combinations
        else:
            im_combinations = self.parent().po.last_image.im_combinations
        self.init_drawn_image(im_combinations)
        if im_combinations is not None and len(im_combinations) > 0:
            # Update image display
            if self.thread_dict["UpdateImage"].isRunning():
                self.thread_dict["UpdateImage"].wait()
            self.thread_dict["UpdateImage"].start()
            # Update csc editing
            self.update_csc_editing_display()

            # Update the detected shape number
            if self.is_first_image_flag:
                self.parent().po.vars['convert_for_origin'] = im_combinations[self.parent().po.current_combination_id]["csc"]
                detected_shape_nb = im_combinations[self.parent().po.current_combination_id]['shape_number']
                if self.parent().po.vars['several_blob_per_arena']:
                    if detected_shape_nb == self.parent().po.sample_number:
                        self.message.setText("Beware: Contrary to what has been checked, there is one spot per arena")
                else:
                    if detected_shape_nb == self.parent().po.sample_number:
                        self.decision_label.setText(
                            f"{detected_shape_nb} distinct specimen(s) detected in {self.parent().po.sample_number} arena(s). Does the color match the cell(s)?")
                        self.yes.setVisible(True)
                    else:
                        self.decision_label.setText(
                            f"{detected_shape_nb} distinct specimen(s) detected in {self.parent().po.sample_number} arena(s). Adjust settings, draw more cells and background, and try again")
                        self.yes.setVisible(False)
                if im_combinations[self.parent().po.current_combination_id]['shape_number'] == 0:
                    self.message.setText("Make sure that scaling metric and spot size are correct")
            else:
                self.parent().po.vars['convert_for_motion'] = im_combinations[self.parent().po.current_combination_id]["csc"]
                self.decision_label.setText("Do colored contours correctly match cell(s) contours?")
            if "rolling_window" in im_combinations[self.parent().po.current_combination_id]:
                self.parent().po.vars['rolling_window_segmentation']['do'] = im_combinations[self.parent().po.current_combination_id]["rolling_window"]
            if "filter_spec" in im_combinations[self.parent().po.current_combination_id]:
                self.parent().po.vars['filter_spec'] = im_combinations[self.parent().po.current_combination_id][
                    "filter_spec"]
                self.update_filter_display()

    def generate_csc_editing(self):
        """
        Create and configure a user interface for color space combination editing.

        This method sets up the UI components needed to edit color space combinations,
        including checkboxes, labels, and drop-down menus. It also configures the layout
        and connections between components.
        """
        self.central_right_widget = QtWidgets.QWidget()
        self.central_right_layout = QtWidgets.QVBoxLayout()

        # 1) Advanced mode option
        self.advanced_mode_widget = QtWidgets.QWidget()
        self.advanced_mode_layout = QtWidgets.QHBoxLayout()
        self.advanced_mode_cb = Checkbox(self.parent().po.all['expert_mode'])
        self.advanced_mode_cb.setStyleSheet("QCheckBox::indicator {width: 12px;height: 12px;background-color: transparent;"
                            "border-radius: 5px;border-style: solid;border-width: 1px;"
                            "border-color: rgb(100,100,100);}"
                            "QCheckBox::indicator:checked {background-color: rgb(70,130,180);}"
                            "QCheckBox:checked, QCheckBox::indicator:checked {border-color: black black white white;}"
                            "QCheckBox:checked {background-color: transparent;}"
                            "QCheckBox:margin-left {0%}"
                            "QCheckBox:margin-right {0%}")
        self.advanced_mode_cb.stateChanged.connect(self.advanced_mode_check)
        self.advanced_mode_label = FixedText(IAW["Advanced_mode"]["label"], halign='l',
                                             tip=IAW["Advanced_mode"]["tips"],
                                             night_mode=self.parent().po.all['night_mode'])
        self.advanced_mode_label.setAlignment(QtCore.Qt.AlignTop)
        self.advanced_mode_layout.addWidget(self.advanced_mode_cb)
        self.advanced_mode_layout.addWidget(self.advanced_mode_label)
        self.advanced_mode_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.advanced_mode_widget.setLayout(self.advanced_mode_layout)
        self.central_right_layout.addWidget(self.advanced_mode_widget)

        self.csc_scroll_table = QtWidgets.QScrollArea()  # QTableWidget()  # Scroll Area which contains the widgets, set as the centralWidget
        self.csc_scroll_table.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.csc_scroll_table.setMinimumHeight(self.parent().im_max_height - 100)
        self.csc_scroll_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.csc_scroll_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.csc_scroll_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.csc_table_widget = QtWidgets.QWidget()
        self.csc_table_layout = QtWidgets.QVBoxLayout()

        # 2) Titles
        self.edit_labels_widget = QtWidgets.QWidget()
        self.edit_labels_layout = QtWidgets.QHBoxLayout()

        self.space_label = FixedText(IAW["Color_combination"]["label"] + ':', halign='l',
                                    tip=IAW["Color_combination"]["tips"],
                                    night_mode=self.parent().po.all['night_mode'])

        self.edit_labels_layout.addWidget(self.space_label)
        self.edit_labels_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.space_label.setVisible(False)
        self.edit_labels_widget.setLayout(self.edit_labels_layout)
        self.csc_table_layout.addWidget(self.edit_labels_widget)

        # 3) First CSC
        self.first_csc_widget = QtWidgets.QWidget()
        self.first_csc_layout = QtWidgets.QGridLayout()
        self.row1 = self.one_csc_editing(with_PCA=True)
        self.row1[4].clicked.connect(self.display_row2)
        self.row2 = self.one_csc_editing()
        self.row2[4].clicked.connect(self.display_row3)
        self.row3 = self.one_csc_editing()# Second CSC
        self.logical_operator_between_combination_result = Combobox(["None", "Or", "And", "Xor"],
                                                                    night_mode=self.parent().po.all['night_mode'])
        self.logical_operator_between_combination_result.setCurrentText(self.parent().po.vars['convert_for_motion']['logical'])
        self.logical_operator_between_combination_result.currentTextChanged.connect(self.logical_op_changed)
        self.logical_operator_between_combination_result.setFixedWidth(100)
        self.logical_operator_label = FixedText(IAW["Logical_operator"]["label"], tip=IAW["Logical_operator"]["tips"],
                                                night_mode=self.parent().po.all['night_mode'])

        self.row21 = self.one_csc_editing()
        self.row21[4].clicked.connect(self.display_row22)
        self.row22 = self.one_csc_editing()
        self.row22[4].clicked.connect(self.display_row23)
        self.row23 = self.one_csc_editing()
        if self.csc_dict is not None:
            self.update_csc_editing_display()
        else:
            self.row1[0].setCurrentIndex(4)
            self.row1[3].setValue(1)
            self.row21[0].setCurrentIndex(0)
            self.row21[3].setValue(0)

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
        self.csc_table_layout.addWidget(self.first_csc_widget)

        # First filters
        self.filter1_label = FixedText(IAW["Filter"]["label"] + ': ', halign='l',
                                    tip=IAW["Filter"]["tips"],
                                    night_mode=self.parent().po.all['night_mode'])
        self.csc_table_layout.addWidget(self.filter1_label)
        self.filter1_widget = QtWidgets.QWidget()
        self.filter1_layout = QtWidgets.QHBoxLayout()
        self.filter1 = Combobox(list(filter_dict.keys()), night_mode=self.parent().po.all['night_mode'])
        self.filter1.setCurrentText(self.parent().po.vars['filter_spec']['filter1_type'])
        self.filter1.currentTextChanged.connect(self.filter1_changed)
        self.filter1.setFixedWidth(100)
        if "Param1" in filter_dict[self.parent().po.vars['filter_spec']['filter1_type']].keys():
            param1_name = filter_dict[self.parent().po.vars['filter_spec']['filter1_type']]["Param1"]["Name"]
        else:
            param1_name = ""
        self.filter1_param1_label = FixedText(param1_name, halign='l', tip="The parameter to adjust the filter effect",
                                    night_mode=self.parent().po.all['night_mode'])
        filter_param_spinbox_width = 60
        self.filter1_param1 = Spinbox(min=-1000, max=1000, val=self.parent().po.vars['filter_spec']['filter1_param'][0], decimals=3, night_mode=self.parent().po.all['night_mode'])
        self.filter1_param1.setFixedWidth(filter_param_spinbox_width)
        self.filter1_param1.valueChanged.connect(self.filter1_param1_changed)
        if "Param2" in filter_dict[self.parent().po.vars['filter_spec']['filter1_type']].keys():
            param2_name = filter_dict[self.parent().po.vars['filter_spec']['filter1_type']]["Param2"]["Name"]
        else:
            param2_name = ""
        self.filter1_param2_label = FixedText(param2_name, halign='l', tip="The parameter to adjust the filter effect",
            night_mode=self.parent().po.all['night_mode'])
        self.filter1_param2 = Spinbox(min=-1000, max=1000, val=self.parent().po.vars['filter_spec']['filter1_param'][1], decimals=3, night_mode=self.parent().po.all['night_mode'])
        self.filter1_param2.setFixedWidth(filter_param_spinbox_width)
        self.filter1_param2.valueChanged.connect(self.filter1_param2_changed)
        self.filter1_layout.addWidget(self.filter1)
        # self.filter1_layout.addWidget(self.filter1_label)
        self.filter1_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.filter1_layout.addWidget(self.filter1_param1_label)
        self.filter1_layout.addWidget(self.filter1_param1)
        self.filter1_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.filter1_layout.addWidget(self.filter1_param2_label)
        self.filter1_layout.addWidget(self.filter1_param2)
        self.filter1.setVisible(False)
        self.filter1_label.setVisible(False)
        self.filter1_param1_label.setVisible(False)
        self.filter1_param1.setVisible(False)
        self.filter1_param2_label.setVisible(False)
        self.filter1_param2.setVisible(False)
        self.filter1_widget.setLayout(self.filter1_layout)
        self.csc_table_layout.addWidget(self.filter1_widget)

        # 4) logical_operator
        self.logical_op_widget = QtWidgets.QWidget()
        self.logical_op_layout = QtWidgets.QHBoxLayout()
        self.logical_op_layout.addWidget(self.logical_operator_label)
        self.logical_op_layout.addWidget(self.logical_operator_between_combination_result)
        self.logical_op_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.logical_operator_between_combination_result.setVisible(False)
        self.logical_operator_label.setVisible(False)
        self.logical_op_widget.setLayout(self.logical_op_layout)
        self.csc_table_layout.addWidget(self.logical_op_widget)

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
        self.csc_table_layout.addWidget(self.second_csc_widget)

        self.csc_table_widget.setLayout(self.csc_table_layout)
        self.csc_scroll_table.setWidget(self.csc_table_widget)
        self.csc_scroll_table.setWidgetResizable(True)
        self.central_right_layout.addWidget(self.csc_scroll_table)
        self.central_right_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))

        # Second filters
        self.filter2_label = FixedText(IAW["Filter"]["label"] + ': ', halign='l',
                                    tip=IAW["Filter"]["tips"],
                                    night_mode=self.parent().po.all['night_mode'])
        self.csc_table_layout.addWidget(self.filter2_label)
        self.filter2_widget = QtWidgets.QWidget()
        self.filter2_layout = QtWidgets.QHBoxLayout()
        self.filter2 = Combobox(list(filter_dict.keys()), night_mode=self.parent().po.all['night_mode'])
        self.filter2.setCurrentText(self.parent().po.vars['filter_spec']['filter2_type'])
        self.filter2.currentTextChanged.connect(self.filter2_changed)
        self.filter2.setFixedWidth(100)
        if "Param1" in filter_dict[self.parent().po.vars['filter_spec']['filter2_type']].keys():
            param1_name = filter_dict[self.parent().po.vars['filter_spec']['filter2_type']]["Param1"]["Name"]
        else:
            param1_name = ""
        self.filter2_param1_label = FixedText(param1_name, halign='l',
                                    tip="The parameter to adjust the filter effect",
                                    night_mode=self.parent().po.all['night_mode'])
        self.filter2_param1 = Spinbox(min=-1000, max=1000, val=self.parent().po.vars['filter_spec']['filter2_param'][0], decimals=3, night_mode=self.parent().po.all['night_mode'])
        self.filter2_param1.setFixedWidth(filter_param_spinbox_width)
        self.filter2_param1.valueChanged.connect(self.filter2_param1_changed)
        if "Param2" in filter_dict[self.parent().po.vars['filter_spec']['filter2_type']].keys():
            param2_name = filter_dict[self.parent().po.vars['filter_spec']['filter2_type']]["Param2"]["Name"]
        else:
            param2_name = ""
        self.filter2_param2_label = FixedText(param2_name, halign='l', tip="The parameter to adjust the filter effect",
            night_mode=self.parent().po.all['night_mode'])
        self.filter2_param2 = Spinbox(min=-1000, max=1000, val=self.parent().po.vars['filter_spec']['filter2_param'][1], decimals=3, night_mode=self.parent().po.all['night_mode'])
        self.filter2_param2.setFixedWidth(filter_param_spinbox_width)

        self.filter1_param2.valueChanged.connect(self.filter2_param2_changed)
        self.filter2_layout.addWidget(self.filter2)
        self.filter2_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.filter2_layout.addWidget(self.filter2_param1_label)
        self.filter2_layout.addWidget(self.filter2_param1)
        self.filter2_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.filter2_layout.addWidget(self.filter2_param2_label)
        self.filter2_layout.addWidget(self.filter2_param2)
        self.filter2.setVisible(False)
        self.filter2_label.setVisible(False)
        self.filter2_widget.setLayout(self.filter2_layout)
        self.csc_table_layout.addWidget(self.filter2_widget)

        # 6) Open the rolling_window_segmentation row layout
        self.rolling_window_segmentation_widget = QtWidgets.QWidget()
        self.rolling_window_segmentation_layout = QtWidgets.QHBoxLayout()
        try:
            self.parent().po.vars["rolling_window_segmentation"]
        except KeyError:
            self.parent().po.vars["rolling_window_segmentation"] = False
        self.rolling_window_segmentation = Checkbox(self.parent().po.vars["rolling_window_segmentation"]['do'])
        self.rolling_window_segmentation.setStyleSheet("QCheckBox::indicator {width: 12px;height: 12px;background-color: transparent;"
                            "border-radius: 5px;border-style: solid;border-width: 1px;"
                            "border-color: rgb(100,100,100);}"
                            "QCheckBox::indicator:checked {background-color: rgb(70,130,180);}"
                            "QCheckBox:checked, QCheckBox::indicator:checked {border-color: black black white white;}"
                            "QCheckBox:checked {background-color: transparent;}"
                            "QCheckBox:margin-left {0%}"
                            "QCheckBox:margin-right {-10%}")
        self.rolling_window_segmentation.stateChanged.connect(self.rolling_window_segmentation_option)

        self.rolling_window_segmentation_label = FixedText(IAW["Rolling_window_segmentation"]["label"],
                                                    tip=IAW["Rolling_window_segmentation"]["tips"], night_mode=self.parent().po.all['night_mode'])
        self.rolling_window_segmentation_label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.rolling_window_segmentation_label.setAlignment(QtCore.Qt.AlignLeft)

        self.rolling_window_segmentation_layout.addWidget(self.rolling_window_segmentation)
        self.rolling_window_segmentation_layout.addWidget(self.rolling_window_segmentation_label)
        self.rolling_window_segmentation_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.rolling_window_segmentation_widget.setLayout(self.rolling_window_segmentation_layout)
        self.central_right_layout.addWidget(self.rolling_window_segmentation_widget)

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

        self.distinct_colors_number.valueChanged.connect(self.distinct_colors_number_changed)
        self.display_more_than_two_colors_option()
        self.more_than_two_colors.setVisible(False)
        self.more_than_two_colors_label.setVisible(False)
        self.distinct_colors_number.setVisible(False)
        self.rolling_window_segmentation.setVisible(False)
        self.rolling_window_segmentation_label.setVisible(False)

        self.more_than_2_colors_layout.addWidget(self.more_than_two_colors)
        self.more_than_2_colors_layout.addWidget(self.more_than_two_colors_label)
        self.more_than_2_colors_layout.addWidget(self.distinct_colors_number)
        self.more_than_2_colors_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.more_than_2_colors_widget.setLayout(self.more_than_2_colors_layout)
        self.central_right_layout.addWidget(self.more_than_2_colors_widget)

        self.central_right_widget.setLayout(self.central_right_layout)

    def update_filter_display(self):
        self.filter1.setCurrentText(self.parent().po.vars['filter_spec']['filter1_type'])
        self.filter1_param1.setValue(self.parent().po.vars['filter_spec']['filter1_param'][0])
        if len(self.parent().po.vars['filter_spec']['filter1_param']) > 1:
            self.filter1_param2.setValue(self.parent().po.vars['filter_spec']['filter1_param'][1])
        if 'filter2_type' in self.parent().po.vars['filter_spec']:
            self.filter2.setCurrentText(self.parent().po.vars['filter_spec']['filter2_type'])
            self.filter2_param1.setValue(self.parent().po.vars['filter_spec']['filter2_param'][0])
            if len(self.parent().po.vars['filter_spec']['filter2_param']) > 1:
                self.filter2_param2.setValue(self.parent().po.vars['filter_spec']['filter2_param'][1])

    def filter1_changed(self):
        """
        Update the UI elements and internal state when the `filter1` selection changes.

        This method updates labels, visibility, and values of filter parameters
        based on the currently selected filter type.

        Parameters
        ----------
        self : object
            The instance of the class containing this method.
        """
        current_filter = self.filter1.currentText()
        self.parent().po.vars['filter_spec']['filter1_type'] = current_filter
        show_param1 = "Param1" in filter_dict[current_filter].keys()
        if self.advanced_mode_cb.isChecked():
            self.filter1_param1_label.setVisible(show_param1)
            self.filter1_param1.setVisible(show_param1)
        if show_param1:
            self.filter1_param1_label.setText(filter_dict[current_filter]['Param1']['Name'])
            self.filter1_param1.setMinimum(filter_dict[current_filter]['Param1']['Minimum'])
            self.filter1_param1.setMaximum(filter_dict[current_filter]['Param1']['Maximum'])
            if self.filter1_param1.value() < filter_dict[current_filter]['Param1']['Minimum'] or self.filter1_param1.value() > filter_dict[current_filter]['Param1']['Maximum']:
                self.filter1_param1.setValue(filter_dict[current_filter]['Param1']['Default'])
        if 'Param2' in list(filter_dict[current_filter].keys()):
            self.filter1_param2_label.setText(filter_dict[current_filter]['Param2']['Name'])
            self.filter1_param2.setMinimum(filter_dict[current_filter]['Param2']['Minimum'])
            self.filter1_param2.setMaximum(filter_dict[current_filter]['Param2']['Maximum'])
            if self.filter1_param2.value() < filter_dict[current_filter]['Param2']['Minimum'] or self.filter1_param2.value() > filter_dict[current_filter]['Param2']['Maximum']:
                self.filter1_param2.setValue(filter_dict[current_filter]['Param2']['Default'])
            if self.advanced_mode_cb.isChecked():
                self.filter1_param2_label.setVisible(True)
                self.filter1_param2.setVisible(True)
        else:
            self.filter1_param2_label.setVisible(False)
            self.filter1_param2.setVisible(False)

    def filter1_param1_changed(self):
        """
        Save the first parameter (most often the lower bound) of the first filter.
        """
        self.parent().po.vars['filter_spec']['filter1_param'][0] = float(self.filter1_param1.value())

    def filter1_param2_changed(self):
        """
        Save the second parameter (most often the higher bound) of the first filter.
        """
        self.parent().po.vars['filter_spec']['filter1_param'][1] = float(self.filter1_param2.value())

    def filter2_changed(self):
        """
        Update the UI elements and internal state when the `filter2` selection changes.

        This method updates labels, visibility, and values of filter parameters
        based on the currently selected filter type.

        Parameters
        ----------
        self : object
            The instance of the class containing this method.
        """
        current_filter = self.filter2.currentText()
        self.parent().po.vars['filter_spec']['filter2_type'] = current_filter
        show_param1 = "Param1" in filter_dict[current_filter].keys()
        if self.advanced_mode_cb.isChecked():
            self.filter2_param1_label.setVisible(show_param1)
            self.filter2_param1.setVisible(show_param1)
        if show_param1:
            self.filter2_param1_label.setText(filter_dict[current_filter]['Param1']['Name'])
            self.filter2_param1.setMinimum(filter_dict[current_filter]['Param1']['Minimum'])
            self.filter2_param1.setMaximum(filter_dict[current_filter]['Param1']['Maximum'])
            if self.filter2_param1.value() < filter_dict[current_filter]['Param1']['Minimum'] or self.filter2_param1.value() > filter_dict[current_filter]['Param1']['Maximum']:
                self.filter2_param1.setValue(filter_dict[current_filter]['Param1']['Default'])
        if 'Param2' in list(filter_dict[current_filter].keys()):
            self.filter2_param2_label.setText(filter_dict[current_filter]['Param2']['Name'])
            self.filter2_param2.setMinimum(filter_dict[current_filter]['Param2']['Minimum'])
            self.filter2_param2.setMaximum(filter_dict[current_filter]['Param2']['Maximum'])
            if self.filter2_param2.value() < filter_dict[current_filter]['Param2']['Minimum'] or self.filter2_param2.value() > filter_dict[current_filter]['Param2']['Maximum']:
                self.filter2_param2.setValue(filter_dict[current_filter]['Param2']['Default'])
            if self.advanced_mode_cb.isChecked():
                self.filter2_param2_label.setVisible(True)
                self.filter2_param2.setVisible(True)
        else:
            self.filter2_param2_label.setVisible(False)
            self.filter2_param2.setVisible(False)

    def filter2_param1_changed(self):
        """
        Save the first parameter (most often the lower bound) of the second filter.
        """
        self.parent().po.vars['filter_spec']['filter2_param'][0] = float(self.filter2_param1.value())

    def filter2_param2_changed(self):
        """
        Save the second parameter (most often the higher bound) of the second filter.
        """
        self.parent().po.vars['filter_spec']['filter2_param'][1] = float(self.filter2_param2.value())

    def one_csc_editing(self, with_PCA: bool=False):
        """
        Summary
        --------
        Edit the color space configuration and add widgets for PCA or other options.

        Parameters
        ----------
        with_PCA : bool, optional
            Flag indicating whether to include PCA options.
            Default is False.

        Returns
        -------
        list
            List of widgets for color space configuration.
        """
        widget_list = []
        if with_PCA:
            widget_list.insert(0, Combobox(["PCA", "bgr", "hsv", "hls", "lab", "luv", "yuv"],
                                           night_mode=self.parent().po.all['night_mode']))
            widget_list[0].currentTextChanged.connect(self.pca_changed)
        else:
            widget_list.insert(0, Combobox(["None", "bgr", "hsv", "hls", "lab", "luv", "yuv"],
                                           night_mode=self.parent().po.all['night_mode']))
        widget_list[0].setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        widget_list[0].setFixedWidth(100)
        for i in [1, 2, 3]:
            widget_list.insert(i, Spinbox(min=-126, max=126, val=0, night_mode=self.parent().po.all['night_mode']))
            widget_list[i].setFixedWidth(45)
        widget_list.insert(i + 1, PButton("+", night_mode=self.parent().po.all['night_mode']))
        return widget_list

    def pca_changed(self):
        """
        Handles the UI changes when 'PCA' is selected in dropdown menu.

        Notes
        -----
        This function modifies the visibility of UI elements based on the selection in a dropdown menu.
        It is triggered when 'PCA' is selected, and hides elements related to logical operators.
        """
        if self.row1[0].currentText() == 'PCA':
            self.logical_operator_between_combination_result.setCurrentText('None')
            for i in range(1, 5):
                self.row1[i].setVisible(False)
                self.row2[i].setVisible(False)
                self.row3[i].setVisible(False)
            self.logical_operator_label.setVisible(False)
            self.logical_operator_between_combination_result.setVisible(False)
        else:
            for i in range(1, 5):
                self.row1[i].setVisible(True)


    def logical_op_changed(self):
        """
        Handles the visibility and values of UI elements based on the current
        logical operator selection in a combination result dropdown.
        """
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
            self.filter2_label.setVisible(self.parent().po.all['expert_mode'])
            self.filter2.setVisible(self.parent().po.all['expert_mode'])
            self.filter2_changed()
            self.row21[0].setVisible(self.parent().po.all['expert_mode'])
            for i1 in [1, 2, 3]:
                self.row21[i1].setVisible(self.parent().po.all['expert_mode'])
            self.row21[i1 + 1].setVisible(self.parent().po.all['expert_mode'])

    def display_logical_operator(self):
        """
        Displays the logical operator UI elements based on expert mode setting.
        """
        self.logical_operator_between_combination_result.setVisible(self.parent().po.all['expert_mode'])
        self.logical_operator_label.setVisible(self.parent().po.all['expert_mode'])

    def display_row2(self):
        """
        Display or hide the second row of the csc editing widgets based on expert mode.
        """
        self.row1[4].setVisible(False)
        for i in range(5):
            self.row2[i].setVisible(self.parent().po.all['expert_mode'])
        self.display_logical_operator()

    def display_row3(self):
        """
        Display or hide the third row of the csc editing widgets based on expert mode.
        """
        self.row2[4].setVisible(False)
        for i in range(4):
            self.row3[i].setVisible(self.parent().po.all['expert_mode'])
        self.display_logical_operator()

    def display_row22(self):
        """
        Display or hide the second row (for the second image segmentation pipeline) of the csc editing widgets based on expert mode.
        """
        self.row21[4].setVisible(False)
        for i in range(5):
            self.row22[i].setVisible(self.parent().po.all['expert_mode'])
        self.display_logical_operator()

    def display_row23(self):
        """
        Display or hide the third row (for the second image segmentation pipeline) of the csc editing widgets based on expert mode.
        """
        self.row22[4].setVisible(False)
        for i in range(4):
            self.row23[i].setVisible(self.parent().po.all['expert_mode'])
        self.display_logical_operator()

    def update_csc_editing_display(self):
        """
        Update the color space conversion (CSC) editing display.

        This method updates the visibility and values of UI elements related to color
        space conversions based on the current state of `self.csc_dict`. It handles
        the display logic for different color spaces and their combinations, ensuring
        that the UI reflects the current configuration accurately.
        """
        remaining_c_spaces = []
        row_number1 = 0
        row_number2 = 0
        if "PCA" in self.csc_dict.keys():
            self.row1[0].setCurrentIndex(0)
            for i in range(1, 4):
                self.row1[i].setVisible(False)
        else:
            c_space_order = ["PCA", "bgr", "hsv", "hls", "lab", "luv", "yuv"]
            for i, (k, v) in enumerate(self.csc_dict.items()):
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
                        row_to_change[0].setVisible(self.parent().po.all['expert_mode'])
                        for i1, i2 in zip([1, 2, 3], [0, 1, 2]):
                            row_to_change[i1].setValue(v[i2])
                            row_to_change[i1].setVisible(self.parent().po.all['expert_mode'])
                        if current_row_number < 3:
                            row_to_change[i1 + 1].setVisible(self.parent().po.all['expert_mode'])

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

        self.row1[4].setVisible(self.parent().po.all['expert_mode'] and row_number1 == 1)
        self.row2[4].setVisible(self.parent().po.all['expert_mode'] and row_number1 == 2)
        self.row21[4].setVisible(self.parent().po.all['expert_mode'] and row_number2 == 1)
        self.row22[4].setVisible(self.parent().po.all['expert_mode'] and row_number2 == 2)
        if row_number2 > 0:
            self.logical_operator_between_combination_result.setCurrentText(self.csc_dict['logical'])
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

        self.logical_operator_between_combination_result.setVisible((row_number2 > 0) and self.parent().po.all['expert_mode'])
        self.logical_operator_label.setVisible((row_number2 > 0) and self.parent().po.all['expert_mode'])

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

        if self.advanced_mode_cb.isChecked():
            if len(remaining_c_spaces) > 0:
                self.message.setText(f'Combination also includes {remaining_c_spaces}')
                self.message.setStyleSheet("color: rgb(230, 145, 18)")
            else:
                self.message.setText(f'')

    def save_user_defined_csc(self):
        """
        Save user-defined combination of color spaces and channels.
        """
        self.csc_dict = {}
        spaces = np.array((self.row1[0].currentText(), self.row2[0].currentText(), self.row3[0].currentText()))
        channels = np.array(
            ((self.row1[1].value(), self.row1[2].value(), self.row1[3].value()),
             (self.row2[1].value(), self.row2[2].value(), self.row2[3].value()),
             (self.row3[1].value(), self.row3[2].value(), self.row3[3].value()),
             (self.row21[1].value(), self.row21[2].value(), self.row21[3].value()),
             (self.row22[1].value(), self.row22[2].value(), self.row22[3].value()),
             (self.row23[1].value(), self.row23[2].value(), self.row23[3].value())),
            dtype=np.float64)
        if self.logical_operator_between_combination_result.currentText() != 'None':
            spaces = np.concatenate((spaces, np.array((
                        self.row21[0].currentText() + "2", self.row22[0].currentText() + "2",
                        self.row23[0].currentText() + "2"))))
            channels = np.concatenate((channels, np.array(((self.row21[1].value(), self.row21[2].value(), self.row21[3].value()),
             (self.row22[1].value(), self.row22[2].value(), self.row22[3].value()),
             (self.row23[1].value(), self.row23[2].value(), self.row23[3].value())),
             dtype=np.float64)))
            self.csc_dict['logical'] = self.logical_operator_between_combination_result.currentText()
        else:
            self.csc_dict['logical'] = 'None'
        if not np.all(spaces == "None"):
            for i, space in enumerate(spaces):
                if space != "None" and space != "None2":
                    self.csc_dict[space] = channels[i, :]
        if not 'PCA' in self.csc_dict and (len(self.csc_dict) == 1 or np.absolute(channels).sum() == 0):
            self.csc_dict_is_empty = True
        else:
            self.csc_dict_is_empty = False

    def rolling_window_segmentation_option(self):
        """
        Set True the grid segmentation option for future image analysis.
        """
        self.parent().po.vars["rolling_window_segmentation"]['do'] = self.rolling_window_segmentation.isChecked()

    def display_more_than_two_colors_option(self):
        """
        Display the More Than Two Colors Options

        This method manages the visibility and state of UI elements related to selecting
        more than two colors for displaying biological masks in advanced mode.
        """
        if self.bio_masks_number > 0 and self.advanced_mode_cb.isChecked():
            self.more_than_two_colors.setVisible(True)
            self.more_than_two_colors_label.setVisible(True)
            if self.more_than_two_colors.isChecked():
                self.distinct_colors_number.setVisible(True)
                self.more_than_two_colors_label.setText("How many distinct colors?")
                self.distinct_colors_number.setValue(3)
            else:
                self.more_than_two_colors_label.setText("Heterogeneous background")
                self.distinct_colors_number.setVisible(False)
                self.distinct_colors_number.setValue(2)
            self.parent().po.all["more_than_two_colors"] = self.more_than_two_colors.isChecked()
        else:
            self.more_than_two_colors.setChecked(False)
            self.more_than_two_colors.setVisible(False)
            self.more_than_two_colors_label.setVisible(False)
            self.distinct_colors_number.setVisible(False)
            self.distinct_colors_number.setValue(2)
            # self.parent().po.vars["color_number"] = 2

    def distinct_colors_number_changed(self):
        """
        Update the parent object's color number variable based on the current value of a distinct colors control.

        Notes
        -----
        This function expects that the parent object has an attribute `po` with a dictionary-like 'vars' that can be updated.
        """
        self.parent().po.vars["color_number"] = int(self.distinct_colors_number.value())

    def start_crop_scale_subtract_delineate(self):
        """
        Start the crop, scale, subtract, and delineate process.

        Extended Description
        --------------------
        This function initiates a background thread to perform the crop, scale,
        subtract, and delineate operations on the image. It also updates the
        UI elements to reflect the ongoing process.
        """
        if not self.thread_dict['CropScaleSubtractDelineate'].isRunning():
            self.message.setText("Looking for each arena contour, wait...")
            self.thread_dict['CropScaleSubtractDelineate'].start()
            self.thread_dict['CropScaleSubtractDelineate'].message_from_thread.connect(self.display_message_from_thread)
            self.thread_dict['CropScaleSubtractDelineate'].message_when_thread_finished.connect(self.delineate_is_done)

            self.yes.setVisible(False)
            self.no.setVisible(False)
            self.reinitialize_bio_and_back_legend()
            self.user_drawn_lines_label.setVisible(False)
            self.cell.setVisible(False)
            self.background.setVisible(False)
            self.one_blob_per_arena.setVisible(False)
            self.one_blob_per_arena_label.setVisible(False)
            self.set_spot_shape.setVisible(False)
            self.spot_shape.setVisible(False)
            self.spot_shape_label.setVisible(False)
            self.set_spot_size.setVisible(False)
            self.spot_size.setVisible(False)
            self.spot_size_label.setVisible(False)
            self.advanced_mode_cb.setChecked(False)
            self.advanced_mode_cb.setVisible(False)
            self.advanced_mode_label.setVisible(False)
            self.generate_analysis_options.setVisible(False)
            self.network_shaped.setVisible(False)
            self.basic.setVisible(False)
            self.visualize.setVisible(False)
            self.visualize_label.setVisible(False)
            self.select_option.setVisible(False)
            self.select_option_label.setVisible(False)

    def delineate_is_done(self, analysis_status: dict):
        """
        Update GUI after delineation is complete.
        """
        if analysis_status['continue']:
            logging.info("Delineation is done, update GUI")
            self.message.setText(analysis_status["message"])
            self.arena_shape_label.setVisible(False)
            self.arena_shape.setVisible(False)
            self.reinitialize_bio_and_back_legend()
            self.reinitialize_image_and_masks(self.parent().po.first_image.bgr)
            self.delineation_done = True
            if self.thread_dict["UpdateImage"].isRunning():
                self.thread_dict["UpdateImage"].wait()
            self.thread_dict["UpdateImage"].start()
            self.thread_dict["UpdateImage"].message_when_thread_finished.connect(self.automatic_delineation_display_done)

            try:
                self.thread_dict['CropScaleSubtractDelineate'].message_from_thread.disconnect()
                self.thread_dict['CropScaleSubtractDelineate'].message_when_thread_finished.disconnect()
            except RuntimeError:
                pass
            if not self.slower_delineation_flag:
                self.asking_delineation_flag = True
        else:
            self.delineation_done = False
            self.asking_delineation_flag = False
            self.auto_delineation_flag = False
            self.asking_slower_or_manual_delineation_flag = False
            self.slower_delineation_flag = False
            self.manual_delineation()

    def automatic_delineation_display_done(self, boole):
        """
        Automatically handles the delineation display status for the user interface.

        This function updates the visibility of various UI elements and resets
        certain flags to ensure that delineation is not redrawn unnecessarily.
        """
        # Remove this flag to not draw it again next time UpdateImage runs for another reason
        self.delineation_done = False
        self.auto_delineation_flag = False
        self.select_option_label.setVisible(False)
        self.select_option.setVisible(False)
        self.arena_shape_label.setVisible(True)
        self.arena_shape.setVisible(True)

        self.decision_label.setText('Is arena delineation correct?')
        self.decision_label.setToolTip(IAW["Video_delimitation"]["tips"])
        self.decision_label.setVisible(True)
        self.user_drawn_lines_label.setText('Draw each arena on the image')
        self.yes.setVisible(True)
        self.no.setVisible(True)

        self.thread_dict["UpdateImage"].message_when_thread_finished.disconnect()

    def display_message_from_thread(self, text_from_thread: str):
        """
        Display a message from a thread.

        Parameters
        ----------
        text_from_thread : str
            The message to display.
        """
        self.message.setText(text_from_thread)

    def starting_differs_from_growing_check(self):
        """
        Set the `origin_state` variable based on checkbox state and frame detection.
        """
        if self.parent().po.vars['first_detection_frame'] > 1:
            self.parent().po.vars['origin_state'] = 'invisible'
        else:
            if self.starting_differs_from_growing_cb.isChecked():
                self.parent().po.vars['origin_state'] = 'constant'
            else:
                self.parent().po.vars['origin_state'] = 'fluctuating'

    def when_yes_is_clicked(self):
        """
        Handles the event when the 'Yes' button is clicked.

        If image analysis is not running, trigger the decision tree process.
        """
        if not self.is_image_analysis_running:
            # self.message.setText('Loading, wait...')
            self.decision_tree(True)

    def when_no_is_clicked(self):
        """
        Handles the event when the 'No' button is clicked.

        If image analysis is not running, trigger the decision tree process.
        """
        if not self.is_image_analysis_running:
            self.decision_tree(False)

    def decision_tree(self, is_yes: bool):
        """
        Determine the next step in image processing based on user interaction.

        Parameters
        ----------
        is_yes : bool
            Boolean indicating the user's choice (Yes or No).

        Notes
        -----
        This function handles various flags and states to determine the next step in
        image processing workflow. It updates internal state variables and triggers
        appropriate methods based on the user's input.
        """
        color_analysis = not self.parent().po.vars['already_greyscale']
        if self.is_first_image_flag:
            if self.asking_first_im_parameters_flag:
                # Ask for the right number of distinct arenas, if not add parameters
                if not is_yes:
                    self.first_im_parameters()
                else:
                    self.auto_delineation()
                self.asking_first_im_parameters_flag = False

            elif self.auto_delineation_flag:
                self.auto_delineation()

            # Is automatic Video delineation correct?
            elif self.asking_delineation_flag:
                self.decision_label.setToolTip("")
                if not is_yes:
                    self.asking_slower_or_manual_delineation()
                else:
                    self.last_image_question()
                self.asking_delineation_flag = False

            # Slower or manual delineation?
            elif self.asking_slower_or_manual_delineation_flag:
                self.back1_bio2 = 0
                if not is_yes:
                    self.manual_delineation()
                else:
                    self.slower_delineation_flag = True
                    self.slower_delineation()
                self.asking_slower_or_manual_delineation_flag = False

            # Is slower delineation correct?
            elif self.slower_delineation_flag:
                self.yes.setText("Yes")
                self.no.setText("No")
                if not is_yes:
                    self.manual_delineation()
                else:
                    self.last_image_question()
                self.slower_delineation_flag = False

            elif self.manual_delineation_flag:
                if is_yes:
                    if self.parent().po.sample_number == self.arena_masks_number:
                        self.thread_dict['SaveManualDelineation'].start()
                        self.last_image_question()
                        self.manual_delineation_flag = False
                    else:
                        self.message.setText(
                            f"{self.arena_masks_number} arenas are drawn over the {self.parent().po.sample_number} expected")

            elif self.asking_last_image_flag:
                self.decision_label.setToolTip("")
                self.parent().po.first_image.im_combinations = None
                self.select_option.clear()
                self.arena_shape.setVisible(False)
                self.arena_shape_label.setVisible(False)
                if is_yes:
                    self.start_last_image()
                else:
                    if "PCA" in self.csc_dict:
                        if self.parent().po.last_image.first_pc_vector is None:
                            self.csc_dict = {"bgr": bracket_to_uint8_image_contrast(self.parent().po.first_image.first_pc_vector), "logical": None}
                        else:
                            self.csc_dict = {"bgr": bracket_to_uint8_image_contrast(self.parent().po.last_image.first_pc_vector), "logical": None}
                    self.parent().po.vars['convert_for_origin'] = deepcopy(self.csc_dict)
                    self.parent().po.vars['convert_for_motion'] = deepcopy(self.csc_dict)
                    self.go_to_next_widget()
                self.asking_last_image_flag = False
        else:
            if is_yes:
                self.parent().po.vars['convert_for_motion'] = deepcopy(self.csc_dict)
                self.go_to_next_widget()

    def first_im_parameters(self):
        """
        Reset UI components and prepare for first image parameters adjustment.

        This method resets various UI elements to their initial states, hides
        confirmation buttons, and shows controls for adjusting spot shapes and sizes.
        It also sets flags to indicate that the user has not yet answered the first
        image parameters prompt.
        """
        self.step = 1
        self.decision_label.setText("Adjust settings, draw more cells and background, and try again")
        self.yes.setVisible(False)
        self.no.setVisible(False)
        self.set_spot_shape.setVisible(True)
        self.spot_shape_label.setVisible(True)
        self.spot_shape.setVisible(self.parent().po.all['set_spot_shape'])
        self.set_spot_size.setVisible(self.one_blob_per_arena.isChecked())
        self.spot_size_label.setVisible(self.one_blob_per_arena.isChecked())
        self.spot_size.setVisible(
            self.one_blob_per_arena.isChecked() and self.set_spot_size.isChecked())
        self.auto_delineation_flag = True
        self.first_im_parameters_answered = True

    def auto_delineation(self):
        """
        Auto delineation process for image analysis.

        Automatically delineate or start manual delineation based on the number of arenas containing distinct specimen(s).

        Notes
        -----
        - The automatic delineation algorithm cannot handle situations where there are more than one arena containing distinct specimen(s). In such cases, manual delineation is initiated.
        - This function updates the current mask and its stats, removes unnecessary memory, initiates image processing steps including cropping, scaling, subtracting, and delineating.
        - The visualization labels are hidden during this process.
        """
        # Do not proceed automatic delineation if there are more than one arena containing distinct specimen(s)
        # The automatic delineation algorithm cannot handle this situation
        if self.parent().po.vars['several_blob_per_arena'] and self.parent().po.sample_number > 1:
            self.manual_delineation()
        else:
            self.decision_label.setText(f"")
            # Save the current mask, its stats, remove useless memory and start delineation
            self.parent().po.first_image.update_current_images(self.parent().po.current_combination_id)
            self.parent().po.get_average_pixel_size()
            self.parent().po.all['are_gravity_centers_moving'] = 0
            self.start_crop_scale_subtract_delineate()
            self.visualize_label.setVisible(False)
            self.visualize.setVisible(False)

    def asking_slower_or_manual_delineation(self):
        """
        Sets the asking_slower_or_manual_delineation_flag to True, updates decision_label and message.

        Extended Description
        --------------------
        This method is used to prompt the user to choose between a slower but more efficient delineation algorithm and manual delineation.

        Notes
        -----
        This function directly modifies instance attributes `asking_slower_or_manual_delineation_flag`, `decision_label`, and `message`.

        """
        self.asking_slower_or_manual_delineation_flag = True
        self.decision_label.setText(f"Click 'yes' to try a slower but more efficient delineation algorithm. Click 'no' to do it manually")
        self.message.setText(f"Clicking no will allow you to draw each arena manually")

    def slower_delineation(self):
        """
        Perform slower delineation process and clear the decision label.

        Execute a sequence of operations that prepare for a slower
        delineation process.
        """
        self.decision_label.setText(f"")
        self.arena_shape.setVisible(False)
        self.arena_shape_label.setVisible(False)
        # Save the current mask, its stats, remove useless memory and start delineation
        self.parent().po.first_image.update_current_images(self.parent().po.current_combination_id)
        self.parent().po.all['are_gravity_centers_moving'] = 1
        self.start_crop_scale_subtract_delineate()

    def manual_delineation(self):
        """
        Manually delineates the analysis arena on the image by enabling user interaction and
        preparing the necessary attributes for manual drawing of arenas on the image.
        """
        self.manual_delineation_flag = True
        self.parent().po.cropping(is_first_image=True)
        self.parent().po.get_average_pixel_size()
        self.reinitialize_image_and_masks(self.parent().po.first_image.bgr)
        self.reinitialize_bio_and_back_legend()
        self.available_arena_names = np.arange(1, self.parent().po.sample_number + 1)
        self.saved_coord = []
        self.arena_mask = np.zeros(self.parent().po.current_image.shape[:2], dtype=np.uint16)
        # self.next.setVisible(True)
        self.decision_label.setVisible(True)
        self.yes.setVisible(True)
        self.cell.setVisible(False)
        self.background.setVisible(False)
        self.no.setVisible(False)
        self.one_blob_per_arena.setVisible(False)
        self.one_blob_per_arena_label.setVisible(False)
        self.generate_analysis_options.setVisible(False)
        self.network_shaped.setVisible(False)
        self.basic.setVisible(False)
        self.visualize.setVisible(False)
        self.visualize_label.setVisible(False)
        self.select_option.setVisible(False)
        self.select_option_label.setVisible(False)
        self.user_drawn_lines_label.setText("Draw each arena")
        self.user_drawn_lines_label.setVisible(True)
        self.decision_label.setText(
            f"Hold click to draw {self.parent().po.sample_number} arena(s) on the image. Once done, click yes.")
        self.message.setText('An error? Hit one button on the left to remove any drawn arena.')

    def last_image_question(self):
        """
        Last image question.

        Queries the user if they want to check parameters for the last image,
        informing them that the best segmentation pipeline may change during analysis.
        """

        self.image_number.setVisible(False)
        self.image_number_label.setVisible(False)
        self.read.setVisible(False)
        self.step = 2
        if self.parent().po.all["im_or_vid"] == 0 and len(self.parent().po.data_list) == 1:
            self.starting_differs_from_growing_cb.setChecked(False)
            self.start_last_image()
        else:
            self.asking_last_image_flag = True
            self.decision_label.setText("Click 'yes' to improve the segmentation using the last image")
            self.decision_label.setToolTip(IAW["Last_image_question"]["tips"])
            self.message.setText('This is useful when the specimen(s) is more visible.')
            self.starting_differs_from_growing_cb.setVisible(True)
            self.starting_differs_from_growing_label.setVisible(True)
            self.yes.setVisible(True)
            self.no.setVisible(True)

    def start_last_image(self):
        """
        Start the process of analyzing the last image in the time-lapse or the video.

        This method initializes various UI elements, retrieves the last image,
        waits for any running threads to complete, processes the image without
        considering it as the first image, and updates the visualization.
        """
        self.is_first_image_flag = False
        self.decision_label.setText('')
        self.yes.setVisible(False)
        self.no.setVisible(False)
        self.spot_size.setVisible(False)
        self.starting_differs_from_growing_cb.setVisible(False)
        self.starting_differs_from_growing_label.setVisible(False)
        self.message.setText('Gathering data and visualizing last image analysis result')
        self.parent().po.get_last_image()
        if self.thread_dict['SaveManualDelineation'].isRunning():
            self.thread_dict['SaveManualDelineation'].wait()
        self.parent().po.cropping(is_first_image=False)
        self.reinitialize_image_and_masks(self.parent().po.last_image.bgr)
        self.reinitialize_bio_and_back_legend()
        self.parent().po.current_combination_id = 0
        self.visualize_is_clicked()
        self.user_drawn_lines_label.setText('Select and draw')
        self.user_drawn_lines_label.setVisible(True)
        self.cell.setVisible(True)
        self.background.setVisible(True)
        self.advanced_mode_cb.setVisible(True)
        self.advanced_mode_label.setVisible(True)
        self.visualize_label.setVisible(True)
        self.visualize.setVisible(True)
        self.row1_widget.setVisible(False)

    def complete_image_analysis_is_clicked(self):
        """
        Completes the image analysis process if no listed threads are running.
        """
        if (not self.thread_dict['SaveManualDelineation'].isRunning() or not self.thread_dict[
            'PrepareVideoAnalysis'].isRunning() or not self.thread_dict['SaveData'].isRunning() or not
        self.thread_dict['CompleteImageAnalysisThread'].isRunning()):
            self.message.setText(f"Analyzing and saving the segmentation result, wait... ")
            self.thread_dict['CompleteImageAnalysisThread'].start()
            self.thread_dict['CompleteImageAnalysisThread'].message_when_thread_finished.connect(self.complete_image_analysis_done)

    def complete_image_analysis_done(self, res):
        self.message.setText(f"Complete image analysis done.")

    def go_to_next_widget(self):
        """
        Advances the user interface to the next widget after performing final checks.

        Notes
        -----
        This function performs several actions in sequence:
            - Displays a message box to inform the user about final checks.
            - Waits for some background threads to complete their execution.
            - Advances the UI to the video analysis window if certain conditions are met.
        """
        if not self.thread_dict['SaveManualDelineation'].isRunning() or not self.thread_dict['PrepareVideoAnalysis'].isRunning() or not self.thread_dict['SaveData'].isRunning():

            self.popup = QtWidgets.QMessageBox()
            self.popup.setWindowTitle("Info")
            self.popup.setText("Final checks...")
            self.popup.setInformativeText("Close and wait until the video tracking window appears.")
            self.popup.setStandardButtons(QtWidgets.QMessageBox.Close)
            x = self.popup.exec_()
            self.decision_label.setVisible(False)
            self.yes.setVisible(False)
            self.no.setVisible(False)
            self.next.setVisible(True)


            self.message.setText(f"Final checks, wait... ")
            self.parent().last_tab = "image_analysis"
            self.thread_dict['PrepareVideoAnalysis'].start()
            if self.parent().po.vars["color_number"] > 2:
                self.parent().videoanalysiswindow.select_option.clear()
                self.parent().videoanalysiswindow.select_option.addItem(f"1) Kmeans")
                self.parent().videoanalysiswindow.select_option.setCurrentIndex(0)
                self.parent().po.all['video_option'] = 0
            time.sleep(1 / 10)
            self.thread_dict['PrepareVideoAnalysis'].wait()
            self.message.setText(f"")

            self.video_tab.set_not_in_use()
            self.parent().last_tab = "image_analysis"
            self.parent().change_widget(3)  # VideoAnalysisWindow

            self.popup.close()
