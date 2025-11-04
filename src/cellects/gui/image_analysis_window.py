#!/usr/bin/env python3
"""
Genereate the Image analysis window of the user interface of Cellects

Cellects transforms the color images into grayscale images in a way that maximizes the contrast between the specimens and the background.
It has an automatic procedure, processed by the one_image_analysis class. If this automatic procedure does not produce good enough results, the user can manually label some areas of the picture as “cell” or “background” to help find a better color space combination. This is particularly useful when the background is heterogeneous, and Cellects can use this information in two ways: First; it can simply ignore the parts labeled as background (e.g. objects or manual writings). Second, it can  use the manual annotation to train a more sophisticated segmentation method: A k-means algorithm to split the image into as many categories as necessary and use the “Cell” labelling to infer to what category the specimens are related to.
Then, Cellects will take into account the user’s input as follows: For each of the segmentations created in the previous steps, it will count the amount of pixels labeled as specimens by the user that were correctly labeled as cell in the segmentation, and will select the segmentation that achieves the highest number. Then, it will do the same thing for the pixels labeled as background. Then, it will use the AND operator between the two results having the best match with the areas labeled as specimens, the AND operator between the two results having the best match with the areas labeled as background, and the OR operator between the result having the best match with the areas labeled as specimens and the result having the best match with the areas labeled as background. Therefore, this optional labeling adds three new segmentations that take into account the user-labeled regions. If the results are still unsatisfactory, the user can continue labeling more areas until one of the segmentations matches their expectations.

"""
import logging
import time
from copy import deepcopy
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui

from cellects.core.cellects_threads import (
    GetFirstImThread, GetLastImThread, FirstImageAnalysisThread,
    CropScaleSubtractDelineateThread, UpdateImageThread,
    LastImageAnalysisThread, SaveManualDelineationThread, FinalizeImageAnalysisThread)
from cellects.gui.custom_widgets import (
    MainTabsType, InsertImage, FullScreenImage, PButton, Spinbox,
    Combobox, Checkbox, FixedText)
from cellects.core.one_image_analysis import OneImageAnalysis
from cellects.image_analysis.image_segmentation import filter_dict


class ImageAnalysisWindow(MainTabsType):
    def __init__(self, parent, night_mode):
        super().__init__(parent, night_mode)
        self.setParent(parent)
        self.csc_dict = self.parent().po.vars['convert_for_origin'] # To change
        self.manual_delineation_flag: bool = False

    def true_init(self):

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
        greyscale = len(self.parent().po.first_im.shape) == 2

        self.display_image = np.zeros((self.parent().im_max_width, self.parent().im_max_width, 3), np.uint8)
        self.display_image = InsertImage(self.display_image, self.parent().im_max_height, self.parent().im_max_width)
        self.display_image.mousePressEvent = self.get_click_coordinates
        self.display_image.mouseMoveEvent = self.get_mouse_move_coordinates
        self.display_image.mouseReleaseEvent = self.get_mouse_release_coordinates

        ## Title
        # self.title_label = FixedText('One Image Analysis', police=30, night_mode=self.parent().po.all['night_mode'])
        # self.title_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.image_number_label = FixedText('Image number',
                                            tip="Change this number if cells are invisible on the first image, never otherwise\nIf they cannot be seen on the first image, increase this number and read until all cells have appeared.",
                                            night_mode=self.parent().po.all['night_mode'])
        self.image_number_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.image_number = Spinbox(min=1, max=self.parent().po.vars['img_number'], val=self.parent().po.all['first_detection_frame'], night_mode=self.parent().po.all['night_mode'])
        self.read = PButton("Read", night_mode=self.parent().po.all['night_mode'])
        self.read.clicked.connect(self.read_is_clicked)

        self.one_blob_per_arena = Checkbox(not self.parent().po.vars['several_blob_per_arena'])
        self.one_blob_per_arena.stateChanged.connect(self.several_blob_per_arena_check)
        self.one_blob_per_arena_label = FixedText("One cell/colony per arena", valign="c",
                                                  tip="Check if there is always only one cell/colony per arena.\nUncheck if each experimental arena can contain several disconnected cells/colonies.",
                                                  night_mode=self.parent().po.all['night_mode'])


        self.scale_with_label = FixedText('Scale with:', valign="c",
                                        tip="What, on the image, should be considered to calculate pixel size in mm",
                                        night_mode=self.parent().po.all['night_mode'])
        self.scale_with = Combobox(["Image horizontal size", "Cell(s) horizontal size"], night_mode=self.parent().po.all['night_mode'])
        self.scale_with.setFixedWidth(280)
        self.scale_with.setCurrentIndex(self.parent().po.all['scale_with_image_or_cells'])
        self.scale_size_label = FixedText('Scale size:', valign="c",
                                          tip="True size (in mm) of the item(s) used for scaling",
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
        self.row1_layout.addItem(self.horizontal_space)
        self.row1_layout.addWidget(self.one_blob_per_arena_label)
        self.row1_layout.addWidget(self.one_blob_per_arena)
        self.row1_layout.addItem(self.horizontal_space)
        self.row1_layout.addWidget(self.scale_with_label)
        self.row1_layout.addWidget(self.scale_with)
        self.row1_layout.addItem(self.horizontal_space)
        self.row1_layout.addWidget(self.scale_size_label)
        self.row1_layout.addWidget(self.horizontal_size)

        # self.row1_widget = QtWidgets.QWidget()
        # self.row1_layout = QtWidgets.QHBoxLayout()
        # self.row1_col1_widget = QtWidgets.QWidget()
        # self.row1_col1_layout = QtWidgets.QVBoxLayout()
        # self.row1_col2_widget = QtWidgets.QWidget()
        # self.row1_col2_layout = QtWidgets.QVBoxLayout()
        #
        # self.im_number_widget = QtWidgets.QWidget()
        # self.im_number_layout = QtWidgets.QHBoxLayout()
        # self.im_number_layout.addWidget(self.image_number_label)
        # self.im_number_layout.addWidget(self.image_number)
        # self.im_number_layout.addWidget(self.read)
        # self.im_number_widget.setLayout(self.im_number_layout)
        # self.row1_col1_layout.addWidget(self.im_number_widget)
        #
        # self.specimen_number_widget = QtWidgets.QWidget()
        # self.specimen_number_layout = QtWidgets.QHBoxLayout()
        # self.specimen_number_layout.addWidget(self.one_blob_per_arena)
        # self.specimen_number_layout.addWidget(self.one_blob_per_arena_label)
        # self.specimen_number_widget.setLayout(self.specimen_number_layout)
        # self.row1_col1_layout.addWidget(self.specimen_number_widget)
        # self.row1_col1_widget.setLayout(self.row1_col1_layout)
        # self.row1_layout.addWidget(self.row1_col1_widget)
        #
        # # self.row1_layout.addItem(self.horizontal_space)
        # # self.row1_layout.addWidget(self.title_label)
        # self.row1_layout.addItem(self.horizontal_space)
        #
        # self.scale_with_widget = QtWidgets.QWidget()
        # self.scale_with_layout = QtWidgets.QHBoxLayout()
        # self.scale_with_layout.addWidget(self.scale_with_label)
        # self.scale_with_layout.addWidget(self.scale_with)
        # self.scale_with_widget.setLayout(self.scale_with_layout)
        # self.row1_col2_layout.addWidget(self.scale_with_widget)
        #
        # self.scale_size_widget = QtWidgets.QWidget()
        # self.scale_size_layout = QtWidgets.QHBoxLayout()
        # self.scale_size_layout.addWidget(self.scale_size_label)
        # self.scale_size_layout.addWidget(self.horizontal_size)
        # self.scale_size_widget.setLayout(self.scale_size_layout)
        # self.row1_col2_layout.addWidget(self.scale_size_widget)
        # self.row1_col2_widget.setLayout(self.row1_col2_layout)
        # self.row1_layout.addWidget(self.row1_col2_widget)

        self.row1_widget.setLayout(self.row1_layout)
        self.Vlayout.addItem(self.vertical_space)
        self.Vlayout.addWidget(self.row1_widget)
        self.Vlayout.addItem(self.vertical_space)
        self.Vlayout.setSpacing(0)

        # 2) Open the central row layout
        self.central_row_widget = QtWidgets.QWidget()
        self.central_row_layout = QtWidgets.QGridLayout()
        # self.central_row_widget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)

        # it will contain a) the user drawn lines, b) the image, c) the csc
        # 2)a) the user drawn lines
        self.user_drawn_lines_widget = QtWidgets.QWidget()
        self.user_drawn_lines_layout = QtWidgets.QVBoxLayout()
        self.user_drawn_lines_label = FixedText("Select and draw:",
                                                tip='By holding down mouse button on the image',
                                                night_mode=self.parent().po.all['night_mode'])
        self.user_drawn_lines_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.user_drawn_lines_layout.addWidget(self.user_drawn_lines_label)
        self.pbuttons_widget = QtWidgets.QWidget()
        self.pbuttons_layout = QtWidgets.QHBoxLayout()
        self.cell = PButton("Cell", False, night_mode=self.parent().po.all['night_mode'])
        self.cell.setFixedWidth(150)
        self.background = PButton("Back", False, night_mode=self.parent().po.all['night_mode'])
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
        # self.bio_pbuttons_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.bio_pbuttons_table.setMinimumHeight(self.parent().im_max_height // 2)
        self.bio_pbuttons_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.bio_pbuttons_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # self.bio_pbuttons_table.setColumnCount(1)
        # self.bio_pbuttons_table.verticalHeader().hide()
        # self.bio_pbuttons_table.horizontalHeader().hide()
        self.back_pbuttons_table = QtWidgets.QScrollArea()#QTableWidget()  # Scroll Area which contains the widgets, set as the centralWidget
        self.back_pbuttons_table.setMinimumHeight(self.parent().im_max_height // 2)
        self.back_pbuttons_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        # self.back_pbuttons_table.setShowGrid(False)
        self.back_pbuttons_table.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        # self.back_pbuttons_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.back_pbuttons_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # self.back_pbuttons_table.setColumnCount(1)
        # self.back_pbuttons_table.verticalHeader().hide()
        # self.back_pbuttons_table.horizontalHeader().hide()

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

        # self.added_lines_widget = QtWidgets.QWidget()
        # self.added_lines_layout = QtWidgets.QHBoxLayout()
        # self.bio_added_lines_widget = QtWidgets.QWidget()
        # self.bio_added_lines_layout = QtWidgets.QVBoxLayout()
        # self.back_added_lines_widget = QtWidgets.QWidget()
        # self.back_added_lines_layout = QtWidgets.QVBoxLayout()
        # # Dynamically add the lines
        self.bio_lines = {}
        self.back_lines = {}
        self.arena_lines = {}
        # self.bio_added_lines_widget.setLayout(self.bio_added_lines_layout)
        # self.back_added_lines_widget.setLayout(self.back_added_lines_layout)
        # self.added_lines_layout.addWidget(self.bio_added_lines_widget)
        # self.added_lines_layout.addWidget(self.back_added_lines_widget)
        # self.added_lines_widget.setLayout(self.added_lines_layout)
        # self.user_drawn_lines_layout.addWidget(self.added_lines_widget)
        # self.user_drawn_lines_layout.addItem(self.vertical_space)

        self.user_drawn_lines_widget.setLayout(self.user_drawn_lines_layout)
        # self.user_drawn_lines_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # self.user_drawn_lines_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.user_drawn_lines_widget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        # self.user_drawn_lines_widget.setFixedWidth(450)
        self.central_row_layout.addWidget(self.user_drawn_lines_widget, 0, 0)

        # 2)b) the image
        # self.central_row_layout.columnStretch(1)
        self.central_row_layout.addWidget(self.display_image, 0, 1)
        # self.central_row_layout.columnStretch(2)

        # Need to create this before self.generate_csc_editing()
        self.message = FixedText("", halign="r", night_mode=self.parent().po.all['night_mode'])
        self.message.setStyleSheet("color: rgb(230, 145, 18)")

        # 2)c) The csc editing
        self.central_right_widget = QtWidgets.QWidget()
        self.central_right_layout = QtWidgets.QVBoxLayout()
        self.generate_csc_editing()
        self.central_right_layout.addWidget(self.edit_widget)
        self.central_right_widget.setLayout(self.central_right_layout)
        # self.central_right_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # self.central_right_widget.setFixedWidth(450)

        self.central_row_layout.addWidget(self.central_right_widget, 0, 2)
        self.central_row_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.central_row_layout.setAlignment(QtCore.Qt.AlignHCenter)
        # 2) Close the central row layout
        self.central_row_widget.setLayout(self.central_row_layout)
        # self.central_row_widget.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # self.central_row_widget.setFixedHeight(self.parent().im_max_height)
        self.Vlayout.addWidget(self.central_row_widget)
        # self.Vlayout.setSpacing(0)
        self.Vlayout.addItem(self.vertical_space)

        # 3) Add Set supplementary parameters row 1
        self.sup_param_row1_widget = QtWidgets.QWidget()
        self.sup_param_row1_layout = QtWidgets.QHBoxLayout()
        # self.sample_number = Spinbox(min=0, max=255, val=self.parent().po.all['first_folder_sample_number'],
        #                             decimals=0, night_mode=self.parent().po.all['night_mode'])
        # self.sample_number_label = FixedText("Arena per image", night_mode=self.parent().po.all['night_mode'])
        # self.sample_number.valueChanged.connect(self.sample_number_changed)

        #HERE

        # 4) Add Set supplementary parameters row2
        self.sup_param_row2_widget = QtWidgets.QWidget()
        self.sup_param_row2_layout = QtWidgets.QHBoxLayout()

        self.arena_shape_label = FixedText("Arena shape", night_mode=self.parent().po.all['night_mode'])
        self.arena_shape = Combobox(['circle', 'rectangle'], night_mode=self.parent().po.all['night_mode'])
        self.arena_shape.setFixedWidth(160)
        self.arena_shape.setCurrentText(self.parent().po.vars['arena_shape'])
        self.arena_shape.currentTextChanged.connect(self.arena_shape_changed)
        self.set_spot_shape = Checkbox(self.parent().po.all['set_spot_shape'])
        self.set_spot_shape.stateChanged.connect(self.set_spot_shape_check)
        self.spot_shape_label = FixedText("Set spot shape", tip="horizontal size in mm", night_mode=self.parent().po.all['night_mode'])
        self.spot_shape = Combobox(['circle', 'rectangle'], night_mode=self.parent().po.all['night_mode'])
        self.spot_shape.setFixedWidth(160)
        if self.parent().po.all['starting_blob_shape'] is None:
            self.spot_shape.setCurrentText('circle')
        else:
            self.spot_shape.setCurrentText(self.parent().po.all['starting_blob_shape'])
        self.spot_shape.currentTextChanged.connect(self.spot_shape_changed)
        self.set_spot_size = Checkbox(self.parent().po.all['set_spot_size'])
        self.set_spot_size.stateChanged.connect(self.set_spot_size_check)
        self.spot_size_label = FixedText("Set spot size", night_mode=self.parent().po.all['night_mode'])
        self.spot_size = Spinbox(min=0, max=100000, val=self.parent().po.all['starting_blob_hsize_in_mm'], decimals=2,
                                 night_mode=self.parent().po.all['night_mode'])
        self.spot_size.valueChanged.connect(self.spot_size_changed)
        self.sup_param_row2_layout.addItem(self.horizontal_space)
        self.sup_param_row2_layout.addWidget(self.arena_shape_label)
        self.sup_param_row2_layout.addWidget(self.arena_shape)
        self.sup_param_row2_layout.addWidget(self.set_spot_shape)
        self.sup_param_row2_layout.addWidget(self.spot_shape_label)
        self.sup_param_row2_layout.addWidget(self.spot_shape)
        self.sup_param_row2_layout.addWidget(self.set_spot_size)
        self.sup_param_row2_layout.addWidget(self.spot_size_label)
        self.sup_param_row2_layout.addWidget(self.spot_size)
        self.sup_param_row2_widget.setLayout(self.sup_param_row2_layout)
        self.sup_param_row2_layout.addItem(self.horizontal_space)
        self.Vlayout.addWidget(self.sup_param_row2_widget)
        self.Vlayout.setSpacing(0)

        # self.sample_number.setVisible(False)
        # self.sample_number_label.setVisible(False)
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
        self.generate_analysis_options = FixedText("Generate analysis options: ", night_mode=self.parent().po.all['night_mode'])
        self.quickly = PButton("Quickly", night_mode=self.parent().po.all['night_mode'])
        self.carefully = PButton("Carefully", night_mode=self.parent().po.all['night_mode'])
        self.quickly.clicked.connect(self.quickly_is_clicked)
        self.carefully.clicked.connect(self.carefully_is_clicked)
        self.visualize = PButton('Visualize', night_mode=self.parent().po.all['night_mode'])
        self.visualize.clicked.connect(self.visualize_is_clicked)
        if self.parent().po.vars['already_greyscale']:
            self.visualize_label = FixedText("Directly: ", night_mode=self.parent().po.all['night_mode'])
        else:
            self.visualize_label = FixedText("Or directly: ", night_mode=self.parent().po.all['night_mode'])

        self.sup_param_row1_layout.addWidget(self.generate_analysis_options)
        self.sup_param_row1_layout.addWidget(self.quickly)
        self.sup_param_row1_layout.addWidget(self.carefully)
        self.sup_param_row1_layout.addItem(self.horizontal_space)
        # self.sup_param_row1_layout.addWidget(self.sample_number)
        # self.sup_param_row1_layout.addWidget(self.sample_number_label)
        self.sup_param_row1_layout.addItem(self.horizontal_space)
        self.sup_param_row1_layout.addWidget(self.visualize_label)
        self.sup_param_row1_layout.addWidget(self.visualize)

        self.sup_param_row1_widget.setLayout(self.sup_param_row1_layout)
        self.Vlayout.addWidget(self.sup_param_row1_widget)
        self.Vlayout.setSpacing(0)

        # 6) Open the choose best option row layout
        self.options_row_widget = QtWidgets.QWidget()
        self.options_row_layout = QtWidgets.QHBoxLayout()
        self.select_option_label = FixedText('Select option to read', tip='Select the option allowing the best segmentation between the cell and the background',
                                             night_mode=self.parent().po.all['night_mode'])
        self.select_option = Combobox([], night_mode=self.parent().po.all['night_mode'])
        if self.parent().po.vars['color_number'] == 2:
            self.select_option.setCurrentIndex(self.parent().po.all['video_option'])
        self.select_option.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # self.select_option.setFixedWidth(120)
        self.select_option.setMinimumWidth(145)
        self.select_option.currentTextChanged.connect(self.option_changed)
        self.n_shapes_detected = FixedText(f'', night_mode=self.parent().po.all['night_mode'])
        self.select_option_label.setVisible(False)
        self.select_option.setVisible(False)
        self.n_shapes_detected.setVisible(False)
        self.options_row_layout.addItem(self.horizontal_space)
        self.options_row_layout.addWidget(self.select_option_label)
        self.options_row_layout.addWidget(self.select_option)
        self.options_row_layout.addWidget(self.n_shapes_detected)
        self.options_row_layout.addItem(self.horizontal_space)
        self.options_row_widget.setLayout(self.options_row_layout)
        self.Vlayout.addWidget(self.options_row_widget)
        self.Vlayout.setSpacing(0)

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
        self.decision_row_layout.addItem(self.horizontal_space)
        self.decision_row_layout.addWidget(self.decision_label)
        self.decision_row_layout.addWidget(self.yes)
        self.decision_row_layout.addWidget(self.no)
        self.decision_row_layout.addItem(self.horizontal_space)
        self.decision_row_widget.setLayout(self.decision_row_layout)
        self.Vlayout.addWidget(self.decision_row_widget)

        # 8) Open the special cases layout
        self.special_cases_widget = QtWidgets.QWidget()
        self.special_cases_layout = QtWidgets.QHBoxLayout()
        self.starting_differs_from_growing_cb = Checkbox(self.parent().po.vars['origin_state'] == 'constant')
        self.starting_differs_from_growing_cb.stateChanged.connect(self.starting_differs_from_growing_check)
        self.starting_differs_from_growing_label = FixedText("Check if the starting area differs from the growing area", tip="This option is only relevant for experiments in which the medium\n(e.g. agar) on which the cells grow is heterogeneous.\nMore precisely when the exploration areas on which the cells will grow and/or move\nare not the same color as the one they were initially on.", night_mode=self.parent().po.all['night_mode'])
        self.starting_differs_from_growing_cb.setVisible(False)
        self.starting_differs_from_growing_label.setVisible(False)
        self.special_cases_layout.addWidget(self.starting_differs_from_growing_cb)
        self.special_cases_layout.addWidget(self.starting_differs_from_growing_label)
        self.special_cases_widget.setLayout(self.special_cases_layout)
        self.Vlayout.addWidget(self.special_cases_widget)

        self.Vlayout.setSpacing(0)

        # 9) Open the last row layout
        self.last_row_widget = QtWidgets.QWidget()
        self.last_row_layout = QtWidgets.QHBoxLayout()
        self.previous = PButton('Previous', night_mode=self.parent().po.all['night_mode'])
        self.previous.clicked.connect(self.previous_is_clicked)
        self.data_tab.clicked.connect(self.data_is_clicked)
        self.video_tab.clicked.connect(self.video_is_clicked)
        self.next = PButton("Next", night_mode=self.parent().po.all['night_mode'])
        self.next.setVisible(False)
        self.next.clicked.connect(self.go_to_next_widget)
        self.last_row_layout.addWidget(self.previous)
        self.last_row_layout.addWidget(self.message)
        self.last_row_layout.addItem(self.horizontal_space)
        self.last_row_layout.addWidget(self.next)
        self.last_row_widget.setLayout(self.last_row_layout)
        self.Vlayout.addWidget(self.last_row_widget)
        self.setLayout(self.Vlayout)
        self.Vlayout.setSpacing(0)

        self.advanced_mode_check()

        self.thread = {}
        self.thread["GetFirstIm"] = GetFirstImThread(self.parent())
        # self.thread["GetFirstIm"].start()
        # self.thread["GetFirstIm"].message_when_thread_finished.connect(self.first_im_read)
        self.reinitialize_image_and_masks(self.parent().po.first_im)
        self.thread["GetLastIm"] = GetLastImThread(self.parent())
        self.thread["GetLastIm"].start()
        self.parent().po.first_image = OneImageAnalysis(self.parent().po.first_im)
        self.thread["FirstImageAnalysis"] = FirstImageAnalysisThread(self.parent())
        self.thread["LastImageAnalysis"] = LastImageAnalysisThread(self.parent())
        self.thread['UpdateImage'] = UpdateImageThread(self.parent())
        self.thread['CropScaleSubtractDelineate'] = CropScaleSubtractDelineateThread(self.parent())
        self.thread['SaveManualDelineation'] = SaveManualDelineationThread(self.parent())
        self.thread['FinalizeImageAnalysis'] = FinalizeImageAnalysisThread(self.parent())

    def previous_is_clicked(self):
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
        if self.is_image_analysis_running:
            self.message.setText("Wait for the analysis to end, or restart Cellects")
        else:
            self.parent().last_tab = "data_specifications"
            self.parent().change_widget(0)  # First

    def video_is_clicked(self):

        if self.video_tab.state != "not_usable":
            if self.is_image_analysis_running:
                self.message.setText("Wait for the analysis to end, or restart Cellects")
            else:
                self.parent().last_tab = "image_analysis"
                self.parent().change_widget(3)

    def read_is_clicked(self):
        if not self.thread["GetFirstIm"].isRunning():
            self.parent().po.all['first_detection_frame'] = int(self.image_number.value())
            self.message.setText(f"Reading image n°{self.parent().po.all['first_detection_frame']}")
            self.thread["GetFirstIm"].start()
            # self.thread["GetFirstIm"].message_when_thread_finished.connect(self.reinitialize_image_and_masks)
            self.reinitialize_bio_and_back_legend()
            self.reinitialize_image_and_masks(self.parent().po.first_im)
            

    def several_blob_per_arena_check(self):
        is_checked = self.one_blob_per_arena.isChecked()
        self.parent().po.vars['several_blob_per_arena'] = not is_checked
        self.set_spot_size.setVisible(is_checked)
        self.spot_size_label.setVisible(is_checked)
        self.spot_size.setVisible(is_checked and self.set_spot_size.isChecked())

    def set_spot_size_check(self):
        is_checked = self.set_spot_size.isChecked()
        if self.step == 1:
            self.spot_size.setVisible(is_checked)
        self.parent().po.all['set_spot_size'] = is_checked

    def spot_size_changed(self):
        self.parent().po.all['starting_blob_hsize_in_mm'] = self.spot_size.value()
        if self.parent().po.all['scale_with_image_or_cells'] == 1:
            self.horizontal_size.setValue(self.parent().po.all['starting_blob_hsize_in_mm'])
        self.set_spot_size_check()

    def set_spot_shape_check(self):
        is_checked = self.set_spot_shape.isChecked()
        self.spot_shape.setVisible(is_checked)
        self.parent().po.all['set_spot_shape'] = is_checked
        if not is_checked:
            self.parent().po.all['starting_blob_shape'] = None

    def spot_shape_changed(self):
        self.parent().po.all['starting_blob_shape'] = self.spot_shape.currentText()
        self.set_spot_shape_check()

    def arena_shape_changed(self):
        self.parent().po.vars['arena_shape'] = self.arena_shape.currentText()
        if self.asking_delineation_flag:
            if self.thread['CropScaleSubtractDelineate'].isRunning():
                self.thread['CropScaleSubtractDelineate'].wait()
            if self.thread['UpdateImage'].isRunning():
                self.thread['UpdateImage'].wait()
            self.message.setText("Updating display...")
            self.decision_label.setVisible(False)
            self.yes.setVisible(False)
            self.no.setVisible(False)
            self.reinitialize_bio_and_back_legend()
            self.reinitialize_image_and_masks(self.parent().po.first_image.bgr)
            self.delineation_done = True
            if self.thread["UpdateImage"].isRunning():
                self.thread["UpdateImage"].wait()
            self.thread["UpdateImage"].start()
            self.thread["UpdateImage"].message_when_thread_finished.connect(self.automatic_delineation_display_done)

        # self.start_crop_scale_subtract_delineate()

    def reinitialize_bio_and_back_legend(self):
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

    def reinitialize_image_and_masks(self, image):
        if len(image.shape) == 2:
            self.parent().po.current_image = np.stack((image, image, image), axis=2)

            self.generate_analysis_options.setVisible(False)
            self.quickly.setVisible(False)
            self.carefully.setVisible(False)
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
        self.parent().po.all['scale_with_image_or_cells'] = self.scale_with.currentIndex()
        if self.parent().po.all['scale_with_image_or_cells'] == 0:
            self.horizontal_size.setValue(self.parent().po.all['image_horizontal_size_in_mm'])
        else:
            self.horizontal_size.setValue(self.parent().po.all['starting_blob_hsize_in_mm'])

    def horizontal_size_changed(self):
        if self.parent().po.all['scale_with_image_or_cells'] == 0:
            self.parent().po.all['image_horizontal_size_in_mm'] = self.horizontal_size.value()
        else:
            self.parent().po.all['starting_blob_hsize_in_mm'] = self.horizontal_size.value()
            self.spot_size.setValue(self.parent().po.all['starting_blob_hsize_in_mm'])

    def advanced_mode_check(self):
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
        # self.c1.setVisible(color_analysis)
        # self.c2.setVisible(color_analysis)
        # self.c3.setVisible(color_analysis)
        display_logical = self.logical_operator_between_combination_result.currentText() != 'None'
        self.logical_operator_between_combination_result.setVisible(color_analysis and display_logical)
        self.logical_operator_label.setVisible(color_analysis and display_logical)

        # if not self.parent().po.vars['already_greyscale']:
        # self.visualize.setVisible(is_checked)
        # self.visualize_label.setVisible(is_checked)
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
        self.filter2.setVisible(is_checked and at_least_one_line_drawn)
        self.filter2_label.setVisible(is_checked and at_least_one_line_drawn)
        has_param1 = is_checked and at_least_one_line_drawn and 'Param1' in filter_dict[self.filter2.currentText()]
        self.filter2_param1.setVisible(has_param1)
        self.filter2_param1_label.setVisible(has_param1)
        has_param2 = is_checked and at_least_one_line_drawn and 'Param2' in filter_dict[self.filter2.currentText()]
        self.filter2_param2.setVisible(has_param2)
        self.filter2_param2_label.setVisible(has_param2)

        self.filter1_param1.setVisible(is_checked)
        self.grid_segmentation.setVisible(is_checked)
        self.grid_segmentation_label.setVisible(is_checked)

        for i in range(5):
            self.row1[i].setVisible(color_analysis and self.row1[0].currentText() != "None")
            self.row21[i].setVisible(color_analysis and self.row21[0].currentText() != "None")
            self.row2[i].setVisible(color_analysis and self.row2[0].currentText() != "None")
            self.row22[i].setVisible(color_analysis and self.row22[0].currentText() != "None")
            if i < 4:
                self.row3[i].setVisible(color_analysis and self.row3[0].currentText() != "None")
                self.row23[i].setVisible(color_analysis and self.row23[0].currentText() != "None")
        if color_analysis:
            if self.row1[0].currentText() != "None":
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
        if self.back1_bio2 == 2:
            self.cell.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
            self.back1_bio2 = 0
        else:
            self.cell.color("rgb(230, 145, 18)")
            self.background.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
            self.back1_bio2 = 2
        self.saved_coord = []

    def background_is_clicked(self):
        if self.back1_bio2 == 1:
            self.background.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
            self.back1_bio2 = 0
        else:
            self.background.color("rgb(81, 160, 224)")
            self.cell.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
            self.back1_bio2 = 1
        self.saved_coord = []

    def get_click_coordinates(self, event):
        if self.back1_bio2 > 0 or self.manual_delineation_flag:
            if not self.is_image_analysis_display_running and not self.thread["UpdateImage"].isRunning():
                self.hold_click_flag = True
                self.saved_coord.append([event.pos().y(), event.pos().x()])
        else:
            self.popup_img = FullScreenImage(self.drawn_image, self.parent().screen_width, self.parent().screen_height)
            self.popup_img.show()
            # img = resize(self.drawn_image, (self.parent().screen_width, self.parent().screen_height))
            # cv2.imshow("Full screen image display", img)
            # waitKey(0)
            # destroyAllWindows()

    def get_mouse_move_coordinates(self, event):
        # if not self.is_image_analysis_display_running:
        #     if self.back1_bio2 > 0 or self.manual_delineation_flag:
        #         if not self.thread["UpdateImage"].isRunning() and len(self.saved_coord) > 0:
        #             if self.saved_coord[0][0] != event.pos().y() and self.saved_coord[0][1] != event.pos().x():
        #                 self.temporary_mask_coord = [self.saved_coord[0], [event.pos().y(), event.pos().x()]]
        #                 self.thread["UpdateImage"].start()
        if self.hold_click_flag:
            if not self.thread["UpdateImage"].isRunning():
                if self.saved_coord[0][0] != event.pos().y() and self.saved_coord[0][1] != event.pos().x():
                    self.temporary_mask_coord = [self.saved_coord[0], [event.pos().y(), event.pos().x()]]
                    self.thread["UpdateImage"].start()

    def get_mouse_release_coordinates(self, event):
        if self.hold_click_flag:
            if self.thread["UpdateImage"].isRunning():
                self.thread["UpdateImage"].wait()
                # self.saved_coord = []
                # self.background.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
                # self.background.night_mode_switch(night_mode=self.parent().po.all['night_mode'])
                # self.back1_bio2 = 0
            self.temporary_mask_coord = []
            if self.manual_delineation_flag and len(self.parent().imageanalysiswindow.available_arena_names) == 0:
                self.message.setText(f"The total number of arenas are already drawn ({self.parent().po.sample_number})")
                self.saved_coord = []
            else:
                self.saved_coord.append([event.pos().y(), event.pos().x()])
                self.thread["UpdateImage"].start()
                self.thread["UpdateImage"].message_when_thread_finished.connect(self.user_defined_shape_displayed)
            self.hold_click_flag = False


        # if not self.is_image_analysis_display_running:
        #     if self.back1_bio2 > 0 or self.manual_delineation_flag:
        #         if len(self.saved_coord) > 0 and self.saved_coord[0][0] != event.pos().y() and self.saved_coord[0][1] != event.pos().x():
        #             if self.thread["UpdateImage"].isRunning():
        #                 self.thread["UpdateImage"].wait()
        #             self.temporary_mask_coord = []
        #             if self.manual_delineation_flag and len(self.parent().imageanalysiswindow.available_arena_names) == 0:
        #                 self.message.setText(f"The total number of arenas are already drawn ({self.parent().po.sample_number})")
        #                 self.saved_coord = []
        #             else:
        #                 self.saved_coord.append([event.pos().y(), event.pos().x()])
        #                 self.thread["UpdateImage"].start()
        #                 self.thread["UpdateImage"].message_when_thread_finished.connect(self.user_defined_shape_displayed)

    def user_defined_shape_displayed(self, boole):
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
        try:
            self.thread["UpdateImage"].message_when_thread_finished.disconnect()
        except RuntimeError:
            pass

    def new_pbutton_on_the_left(self, pbutton_name):
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
        if not self.is_image_analysis_display_running and not self.thread["UpdateImage"].isRunning() and hasattr(self.sender(), 'text'):
            pbutton_name = self.sender().text()
            if pbutton_name[2:6] == "Back":
                line_name = np.uint8(pbutton_name[6:])
                self.back_mask[self.back_mask == line_name] = 0
                self.back_added_lines_layout.removeWidget(self.back_lines[line_name][pbutton_name])
                self.back_lines[line_name][pbutton_name].deleteLater()
                self.back_lines.pop(line_name)
                self.back_masks_number -= 1
                self.available_back_names = np.sort(np.concatenate(([line_name], self.available_back_names)))
                # self.back_pbuttons_table.removeRow(line_name - 1)
            elif pbutton_name[2:6] == "Cell":
                line_name = np.uint8(pbutton_name[6:])
                self.bio_mask[self.bio_mask == line_name] = 0
                self.bio_added_lines_layout.removeWidget(self.bio_lines[line_name][pbutton_name])
                self.bio_lines[line_name][pbutton_name].deleteLater()
                self.bio_lines.pop(line_name)
                self.bio_masks_number -= 1
                self.available_bio_names = np.sort(np.concatenate(([line_name], self.available_bio_names)))
                # self.bio_pbuttons_table.removeRow(line_name - 1)
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
                # if line_name % 2 == 1:
                #     self.bio_pbuttons_table.removeRow((line_name + 1) // 2)
                # else:
                #     self.back_pbuttons_table.removeRow((line_name + 1) // 2)
            # if self.parent().po.first_image.im_combinations is not None:
            # if self.thread["UpdateImage"].isRunning():
            #     self.thread["UpdateImage"].wait()
            self.thread["UpdateImage"].start()

    def quickly_is_clicked(self):
        if not self.is_image_analysis_running:
            self.is_image_analysis_running = True
            self.message.setText('Loading, wait...')
            self.parent().po.carefully = False
            self.parent().po.visualize = False
            if self.is_first_image_flag:
                self.run_first_image_analysis()
            else:
                self.run_last_image_analysis()

    def carefully_is_clicked(self):
        if not self.is_image_analysis_running:
            self.is_image_analysis_running = True
            self.message.setText('Loading, wait...')
            self.parent().po.carefully = True
            self.parent().po.visualize = False
            if self.is_first_image_flag:
                self.run_first_image_analysis()
            else:
                self.run_last_image_analysis()

    def visualize_is_clicked(self):
        if not self.is_image_analysis_running:
            self.is_image_analysis_running = True
            self.message.setText('Loading, wait...')
            self.parent().po.visualize = True
            # if self.step == 0:
            #     self.select_option_label.setVisible(False)
            #     self.select_option.setVisible(False)
            if self.is_first_image_flag:
                self.run_first_image_analysis()
            else:
                self.run_last_image_analysis()

    def run_first_image_analysis(self):
        # logging.info('runfim' +str(self.parent().po.sample_number))
        if self.first_im_parameters_answered:
            # self.sample_number_changed()
            self.several_blob_per_arena_check()
            self.horizontal_size_changed()
            self.spot_shape_changed()
            self.arena_shape_changed()
        logging.info(self.parent().po.sample_number)
        logging.info(self.parent().po.vars['several_blob_per_arena'])
        logging.info(self.parent().po.all['starting_blob_shape'])
        logging.info(self.parent().po.vars['arena_shape'])

        if self.parent().po.visualize:
            self.save_user_defined_csc()
            self.parent().po.vars["color_number"] = int(self.distinct_colors_number.value())
            if self.csc_dict_is_empty:
                self.message.setText('Choose a color space, modify a channel and visualize')
                self.message.setStyleSheet("color: rgb(230, 145, 18)")
        if not self.parent().po.visualize or not self.csc_dict_is_empty:
            self.parent().po.vars['convert_for_origin'] = deepcopy(self.csc_dict)
            self.thread["FirstImageAnalysis"].start()
            self.thread["FirstImageAnalysis"].message_from_thread.connect(self.display_message_from_thread)
            self.thread["FirstImageAnalysis"].message_when_thread_finished.connect(self.when_image_analysis_finishes)

    def run_last_image_analysis(self):
        logging.info('runlim')
        if self.parent().po.visualize:
            self.save_user_defined_csc()
            self.parent().po.vars["color_number"] = int(self.distinct_colors_number.value())
            if self.csc_dict_is_empty:
                self.message.setText('Choose a color space, increase a channel and visualize')
                self.message.setStyleSheet("color: rgb(230, 145, 18)")
            else:
                self.parent().po.vars['convert_for_motion'] = deepcopy(self.csc_dict)
                self.thread["LastImageAnalysis"].start()
                self.thread["LastImageAnalysis"].message_from_thread.connect(self.display_message_from_thread)
                self.thread["LastImageAnalysis"].message_when_thread_finished.connect(
                    self.when_image_analysis_finishes)
        else:
            self.thread["LastImageAnalysis"].start()
            self.thread["LastImageAnalysis"].message_from_thread.connect(self.display_message_from_thread)
            self.thread["LastImageAnalysis"].message_when_thread_finished.connect(self.when_image_analysis_finishes)

    def when_image_analysis_finishes(self):
        logging.info('im_finish' + str(self.parent().po.sample_number))

        if self.parent().po.visualize:
            if self.parent().po.current_combination_id != self.select_option.currentIndex():
                self.select_option.setCurrentIndex(self.parent().po.current_combination_id)
        else:
            self.parent().po.current_combination_id = 0
            if self.is_first_image_flag:
                im_combinations = self.parent().po.first_image.im_combinations
            else:
                im_combinations = self.parent().po.last_image.im_combinations
            if len(im_combinations) > 0:
                self.csc_dict = im_combinations[self.parent().po.current_combination_id]["csc"]

                if self.is_first_image_flag:
                    self.parent().po.vars['convert_for_origin'] = deepcopy(self.csc_dict)
                else:
                    self.parent().po.vars['convert_for_motion'] = deepcopy(self.csc_dict)
                option_number = len(im_combinations)

                if option_number > 1:
                    # Update the available options of the scrolling menu
                    self.select_option.clear()
                    for option in range(option_number):
                        self.select_option.addItem(f"Option {option + 1}")
                self.update_csc_editing_display()
            else:
                self.message.setText("No options could be generated automatically, use the advanced mode")

        if self.parent().po.visualize or len(im_combinations) > 0:
            self.is_image_analysis_display_running = True
            # Update image display
            if self.thread["UpdateImage"].isRunning():
                self.thread["UpdateImage"].wait()
            self.thread["UpdateImage"].start()
            self.thread["UpdateImage"].message_when_thread_finished.connect(self.image_analysis_displayed)

    def image_analysis_displayed(self):
        color_analysis = not self.parent().po.vars['already_greyscale']
        self.message.setText("")


        if self.step < 2:
            detected_shape_nb = self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id][
                'shape_number']
            if detected_shape_nb == self.parent().po.sample_number or self.parent().po.vars['several_blob_per_arena']:
                self.decision_label.setText(
                    f"{detected_shape_nb} distinct spots detected in {self.parent().po.sample_number} arena(s). Does the color match the cell(s)?")
                if self.step == 1:
                    self.yes.setVisible(True)
                    self.message.setText("If not, draw more Cell and Back ellipses on the image and retry")
            else:
                if self.no.isVisible():
                    self.decision_label.setText(
                        f"{detected_shape_nb} distinct spots detected in {self.parent().po.sample_number} arena(s). Click Yes when satisfied, Click No to fill in more parameters")
                    self.yes.setVisible(True)
                    self.no.setVisible(True)
                else:
                    self.decision_label.setText(
                        f"{detected_shape_nb} distinct spots detected in {self.parent().po.sample_number} arena(s). Click Yes when satisfied")
                    self.yes.setVisible(True)

            if self.parent().po.vars['several_blob_per_arena'] and (detected_shape_nb == self.parent().po.sample_number):
                self.message.setText("Beware: Contrary to what has been checked, there is one spot per arena")

        if not self.parent().po.visualize:
            self.select_option.setVisible(color_analysis)
            self.select_option_label.setVisible(color_analysis)
        if self.step == 0:
            # self.decision_label.setText(f"Does the color correctly cover the cells? And, is {detected_shape_nb} the number of distinct arenas?")
            if self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id]['shape_number'] == 0:
                self.message.setText("Make sure that scaling metric and spot size are correct")
            self.decision_label.setVisible(True)
            self.yes.setVisible(True)
            self.no.setVisible(True)
            self.arena_shape.setVisible(True)
            self.arena_shape_label.setVisible(True)
            # self.select_option_label.setVisible(color_analysis)
            # self.select_option.setVisible(color_analysis)
            self.n_shapes_detected.setVisible(True)

        elif self.step == 2:
            self.generate_analysis_options.setVisible(color_analysis)
            self.quickly.setVisible(color_analysis)
            self.carefully.setVisible(color_analysis)
            self.visualize.setVisible(True)

            self.decision_label.setText("Click next when color delimits the cell(s) correctly")
            self.yes.setVisible(False)
            self.no.setVisible(False)
            self.message.setText('When the resulting segmentation of the last image seems good, click next.')
            self.next.setVisible(True)

        try:
            self.thread["UpdateImage"].message_when_thread_finished.disconnect()
        except RuntimeError:
            pass
        self.is_image_analysis_running = False
        self.is_image_analysis_display_running = False

    def option_changed(self):
        """
        Save the csc, change the image displayed, the csc editing
        :return:
        """
        # Update the current image
        if self.is_first_image_flag:
            im_combinations = self.parent().po.first_image.im_combinations
        else:
            im_combinations = self.parent().po.last_image.im_combinations
        self.parent().po.current_combination_id = self.select_option.currentIndex()
        logging.info(im_combinations is None)
        if im_combinations is not None and len(im_combinations) > 0:
            if self.parent().po.current_combination_id + 1 > len(im_combinations):
                self.parent().po.current_combination_id = 0
            self.csc_dict = im_combinations[self.parent().po.current_combination_id]["csc"]
            self.parent().po.current_image = np.stack((im_combinations[self.parent().po.current_combination_id]['converted_image'],
                                                    im_combinations[self.parent().po.current_combination_id]['converted_image'],
                                                    im_combinations[self.parent().po.current_combination_id]['converted_image']), axis=2)
            self.drawn_image = deepcopy(self.parent().po.current_image)

            # Update image display
            if self.thread["UpdateImage"].isRunning():
                self.thread["UpdateImage"].wait()
            self.thread["UpdateImage"].start()
            # Update csc editing
            self.update_csc_editing_display()

            # Update the detected shape number
            if self.is_first_image_flag:
                self.parent().po.vars['convert_for_origin'] = im_combinations[self.parent().po.current_combination_id]["csc"]
                detected_shape_nb = \
                self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id]['shape_number']
                if self.parent().po.vars['several_blob_per_arena']:
                    if detected_shape_nb == self.parent().po.sample_number:
                        self.message.setText("Beware: Contrary to what has been checked, there is one spot per arena")
                else:
                    if detected_shape_nb == self.parent().po.sample_number:
                        self.decision_label.setText(
                            f"{detected_shape_nb} distinct spots detected in {self.parent().po.sample_number} arena(s). Does the color match the cell(s)?")
                        self.yes.setVisible(True)
                    else:
                        self.decision_label.setText(
                            f"{detected_shape_nb} distinct spots detected in {self.parent().po.sample_number} arena(s). Adjust settings, draw more cells and background, and try again")
                        self.yes.setVisible(False)
                # self.decision_label.setText(f"Does the color correctly cover the cells?")
                # detected_shape_nb = \
                # self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id]['shape_number']
                # self.decision_label.setText(
                #     f"Does the color correctly cover the cells? And, is {detected_shape_nb} the number of distinct arenas?")
                if self.parent().po.first_image.im_combinations[self.parent().po.current_combination_id]['shape_number'] == 0:
                    self.message.setText("Make sure that scaling metric and spot size are correct")
            else:
                self.parent().po.vars['convert_for_motion'] = im_combinations[self.parent().po.current_combination_id]["csc"]
                self.decision_label.setText("Do colored contours correctly match cell(s) contours?")

    def generate_csc_editing(self):
        # self.edit_layout = QtWidgets.QGridLayout()
        self.edit_widget = QtWidgets.QWidget()
        self.edit_layout = QtWidgets.QVBoxLayout()

        # 1) Advanced mode option
        self.advanced_mode_widget = QtWidgets.QWidget()
        self.advanced_mode_layout = QtWidgets.QHBoxLayout()
        self.advanced_mode_cb = Checkbox(self.parent().po.all['expert_mode'])
        self.advanced_mode_cb.setStyleSheet("margin-left:0%; margin-right:0%;")
        self.advanced_mode_cb.stateChanged.connect(self.advanced_mode_check)
        self.advanced_mode_label = FixedText('Advanced mode', halign='l',
                                             tip="Display the color space combination corresponding to the selected option",
                                             night_mode=self.parent().po.all['night_mode'])
        self.advanced_mode_label.setAlignment(QtCore.Qt.AlignTop)
        self.advanced_mode_layout.addWidget(self.advanced_mode_cb)
        self.advanced_mode_layout.addWidget(self.advanced_mode_label)
        self.advanced_mode_layout.addItem(self.horizontal_space)
        self.advanced_mode_widget.setLayout(self.advanced_mode_layout)
        self.edit_layout.addWidget(self.advanced_mode_widget)

        self.csc_scroll_table = QtWidgets.QScrollArea()  # QTableWidget()  # Scroll Area which contains the widgets, set as the centralWidget
        # self.csc_scroll_table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.csc_scroll_table.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.csc_scroll_table.setMinimumHeight(self.parent().im_max_height - 100)
        # self.csc_scroll_table.setMinimumWidth(300)
        self.csc_scroll_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.csc_scroll_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.csc_scroll_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.csc_table_widget = QtWidgets.QWidget()
        self.csc_table_layout = QtWidgets.QVBoxLayout()

        # 2) Titles
        self.edit_labels_widget = QtWidgets.QWidget()
        self.edit_labels_layout = QtWidgets.QHBoxLayout()

        self.space_label = FixedText('Color space:', halign='l',
                                    tip="Color spaces are transformations of the original BGR (Blue Green Red) image\nInstead of defining an image by 3 colors,\n they transform it into 3 different visual properties\n  - hsv: hue (color), saturation, value (lightness)\n  - hls: hue (color), lightness, saturation\n  - lab: Lightness, Red/Green, Blue/Yellow\n  - luv and yuv: l and y are Lightness, u and v are related to colors\n",
                                    night_mode=self.parent().po.all['night_mode'])
        # self.c1 = FixedText('  C1', halign='c', tip="Increase if it increase cell detection", night_mode=self.parent().po.all['night_mode'])
        # self.c2 = FixedText('  C2', halign='c', tip="Increase if it increase cell detection", night_mode=self.parent().po.all['night_mode'])
        # self.c3 = FixedText('  C3', halign='c', tip="Increase if it increase cell detection", night_mode=self.parent().po.all['night_mode'])

        self.edit_labels_layout.addWidget(self.space_label)
        # self.edit_labels_layout.addWidget(self.c1)
        # self.edit_labels_layout.addWidget(self.c2)
        # self.edit_labels_layout.addWidget(self.c3)
        self.edit_labels_layout.addItem(self.horizontal_space)
        self.space_label.setVisible(False)
        # self.c1.setVisible(False)
        # self.c2.setVisible(False)
        # self.c3.setVisible(False)
        self.edit_labels_widget.setLayout(self.edit_labels_layout)
        # self.edit_layout.addWidget(self.edit_labels_widget)
        self.csc_table_layout.addWidget(self.edit_labels_widget)

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
        self.logical_operator_label = FixedText("Logical operator", tip="Between selected color space combinations",
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
        self.first_csc_layout.addItem(self.horizontal_space, 0, 5, 3, 1)
        self.first_csc_widget.setLayout(self.first_csc_layout)
        self.csc_table_layout.addWidget(self.first_csc_widget)
        # self.edit_layout.addWidget(self.first_csc_widget)

        # First filters

        self.filter1_label = FixedText('Filter: ', halign='l',
                                    tip="The filter to apply to the image before segmentation",
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
        self.filter1_layout.addItem(self.horizontal_space)
        self.filter1_layout.addWidget(self.filter1_param1_label)
        self.filter1_layout.addWidget(self.filter1_param1)
        self.filter1_layout.addItem(self.horizontal_space)
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
        self.logical_op_layout.addItem(self.horizontal_space)
        self.logical_operator_between_combination_result.setVisible(False)
        self.logical_operator_label.setVisible(False)
        self.logical_op_widget.setLayout(self.logical_op_layout)
        self.csc_table_layout.addWidget(self.logical_op_widget)
        # self.edit_layout.addWidget(self.logical_op_widget)

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
        self.csc_table_layout.addWidget(self.second_csc_widget)

        self.csc_table_widget.setLayout(self.csc_table_layout)
        self.csc_scroll_table.setWidget(self.csc_table_widget)
        self.csc_scroll_table.setWidgetResizable(True)
        # self.edit_layout.addWidget(self.second_csc_widget)
        self.edit_layout.addWidget(self.csc_scroll_table)
        self.edit_layout.addItem(self.vertical_space)

        # Second filters
        self.filter2_label = FixedText('Filter: ', halign='l',
                                    tip="The filter to apply to the image before segmentation",
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
        # self.filter2_layout.addWidget(self.filter2_label)
        self.filter2_layout.addItem(self.horizontal_space)
        self.filter2_layout.addWidget(self.filter2_param1_label)
        self.filter2_layout.addWidget(self.filter2_param1)
        self.filter2_layout.addItem(self.horizontal_space)
        self.filter2_layout.addWidget(self.filter2_param2_label)
        self.filter2_layout.addWidget(self.filter2_param2)
        self.filter2.setVisible(False)
        self.filter2_label.setVisible(False)
        self.filter2_widget.setLayout(self.filter2_layout)
        self.csc_table_layout.addWidget(self.filter2_widget)

        # 6) Open the grid_segmentation row layout
        self.grid_segmentation_widget = QtWidgets.QWidget()
        self.grid_segmentation_layout = QtWidgets.QHBoxLayout()
        try:
            self.parent().po.vars["grid_segmentation"]
        except KeyError:
            self.parent().po.vars["grid_segmentation"] = False
        self.grid_segmentation = Checkbox(self.parent().po.vars["grid_segmentation"])
        self.grid_segmentation.setStyleSheet("margin-left:0%; margin-right:-10%;")
        self.grid_segmentation.stateChanged.connect(self.grid_segmentation_option)

        self.grid_segmentation_label = FixedText("Grid segmentation",
                                                    tip="Segment small squares of the images to detect local intensity valleys\nThis method segment the image locally using otsu thresholding on a rolling window", night_mode=self.parent().po.all['night_mode'])
        self.grid_segmentation_label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # self.more_than_two_colors_label.setFixedWidth(300)
        self.grid_segmentation_label.setAlignment(QtCore.Qt.AlignLeft)

        self.grid_segmentation_layout.addWidget(self.grid_segmentation)
        self.grid_segmentation_layout.addWidget(self.grid_segmentation_label)
        self.grid_segmentation_layout.addItem(self.horizontal_space)
        self.grid_segmentation_widget.setLayout(self.grid_segmentation_layout)
        self.edit_layout.addWidget(self.grid_segmentation_widget)

        # 6) Open the more_than_2_colors row layout
        self.more_than_2_colors_widget = QtWidgets.QWidget()
        self.more_than_2_colors_layout = QtWidgets.QHBoxLayout()
        self.more_than_two_colors = Checkbox(self.parent().po.all["more_than_two_colors"])
        self.more_than_two_colors.setStyleSheet("margin-left:0%; margin-right:-10%;")
        self.more_than_two_colors.stateChanged.connect(self.display_more_than_two_colors_option)

        self.more_than_two_colors_label = FixedText("More than two colors",
                                                    tip="The program will split the image into categories\nand find the one corresponding to the cell(s)", night_mode=self.parent().po.all['night_mode'])
        self.more_than_two_colors_label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # self.more_than_two_colors_label.setFixedWidth(300)
        self.more_than_two_colors_label.setAlignment(QtCore.Qt.AlignLeft)
        self.distinct_colors_number = Spinbox(min=2, max=5, val=self.parent().po.vars["color_number"], night_mode=self.parent().po.all['night_mode'])

        self.distinct_colors_number.valueChanged.connect(self.distinct_colors_number_changed)
        self.display_more_than_two_colors_option()
        self.more_than_two_colors.setVisible(False)
        self.more_than_two_colors_label.setVisible(False)
        self.distinct_colors_number.setVisible(False)
        self.grid_segmentation.setVisible(False)
        self.grid_segmentation_label.setVisible(False)

        self.more_than_2_colors_layout.addWidget(self.more_than_two_colors)
        self.more_than_2_colors_layout.addWidget(self.more_than_two_colors_label)
        self.more_than_2_colors_layout.addWidget(self.distinct_colors_number)
        self.more_than_2_colors_layout.addItem(self.horizontal_space)
        self.more_than_2_colors_widget.setLayout(self.more_than_2_colors_layout)
        self.edit_layout.addWidget(self.more_than_2_colors_widget)

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

    def filter1_changed(self):
        current_filter = self.filter1.currentText()
        self.parent().po.vars['filter_spec']['filter1_type'] = current_filter
        show_param1 = "Param1" in filter_dict[current_filter].keys()
        self.filter1_param1_label.setVisible(show_param1)
        self.filter1_param1.setVisible(show_param1)
        if show_param1:
            self.filter1_param1_label.setText(filter_dict[current_filter]['Param1']['Name'])
            self.filter1_param1.setMinimum(filter_dict[current_filter]['Param1']['Minimum'])
            self.filter1_param1.setMaximum(filter_dict[current_filter]['Param1']['Maximum'])
            self.filter1_param1.setValue(filter_dict[current_filter]['Param1']['Default'])
        if 'Param2' in list(filter_dict[current_filter].keys()):
            self.filter1_param2_label.setText(filter_dict[current_filter]['Param2']['Name'])
            self.filter1_param2.setMinimum(filter_dict[current_filter]['Param2']['Minimum'])
            self.filter1_param2.setMaximum(filter_dict[current_filter]['Param2']['Maximum'])
            self.filter1_param2.setValue(filter_dict[current_filter]['Param2']['Default'])
            self.filter1_param2_label.setVisible(True)
            self.filter1_param2.setVisible(True)
        else:
            self.filter1_param2_label.setVisible(False)
            self.filter1_param2.setVisible(False)

    def filter1_param1_changed(self):
        self.parent().po.vars['filter_spec']['filter1_param'][0] = float(self.filter1_param1.value())

    def filter1_param2_changed(self):
        self.parent().po.vars['filter_spec']['filter1_param'][1] = float(self.filter1_param2.value())

    def filter2_changed(self):
        current_filter = self.filter2.currentText()
        self.parent().po.vars['filter_spec']['filter2_type'] = current_filter
        show_param1 = "Param1" in filter_dict[current_filter].keys()
        self.filter2_param1_label.setVisible(show_param1)
        self.filter2_param1.setVisible(show_param1)
        if show_param1:
            self.filter2_param1_label.setText(filter_dict[current_filter]['Param1']['Name'])
            self.filter2_param1.setMinimum(filter_dict[current_filter]['Param1']['Minimum'])
            self.filter2_param1.setMaximum(filter_dict[current_filter]['Param1']['Maximum'])
            self.filter2_param1.setValue(filter_dict[current_filter]['Param2']['Default'])
        if 'Param2' in list(filter_dict[current_filter].keys()):
            self.filter2_param2_label.setText(filter_dict[current_filter]['Param2']['Name'])
            self.filter2_param2.setMinimum(filter_dict[current_filter]['Param2']['Minimum'])
            self.filter2_param2.setMaximum(filter_dict[current_filter]['Param2']['Maximum'])
            self.filter2_param2.setValue(filter_dict[current_filter]['Param2']['Default'])
            self.filter2_param2_label.setVisible(True)
            self.filter2_param2.setVisible(True)
        else:
            self.filter2_param2_label.setVisible(False)
            self.filter2_param2.setVisible(False)

    def filter2_param1_changed(self):
        self.parent().po.vars['filter_spec']['filter2_param'][0] = float(self.filter2_param1.value())

    def filter2_param2_changed(self):
        self.parent().po.vars['filter_spec']['filter2_param'][1] = float(self.filter2_param2.value())

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
            self.filter2_label.setVisible(self.parent().po.all['expert_mode'])
            self.filter2.setVisible(self.parent().po.all['expert_mode'])
            self.filter2_changed()
            self.row21[0].setVisible(self.parent().po.all['expert_mode'])
            for i1 in [1, 2, 3]:
                self.row21[i1].setVisible(self.parent().po.all['expert_mode'])
            self.row21[i1 + 1].setVisible(self.parent().po.all['expert_mode'])

    def display_logical_operator(self):
        self.logical_operator_between_combination_result.setVisible(self.parent().po.all['expert_mode'])
        self.logical_operator_label.setVisible(self.parent().po.all['expert_mode'])

    def display_row2(self):
        self.row1[4].setVisible(False)
        for i in range(5):
            self.row2[i].setVisible(self.parent().po.all['expert_mode'])
        self.display_logical_operator()

    def display_row3(self):
        self.row2[4].setVisible(False)
        for i in range(4):
            self.row3[i].setVisible(self.parent().po.all['expert_mode'])
        self.display_logical_operator()

    def display_row22(self):
        self.row21[4].setVisible(False)
        for i in range(5):
            self.row22[i].setVisible(self.parent().po.all['expert_mode'])
        self.display_logical_operator()

    def display_row23(self):
        self.row22[4].setVisible(False)
        for i in range(4):
            self.row23[i].setVisible(self.parent().po.all['expert_mode'])
        self.display_logical_operator()

    def update_csc_editing_display(self):
        c_space_order = ["None", "bgr", "hsv", "hls", "lab", "luv", "yuv"]
        remaining_c_spaces = []
        row_number1 = 0
        row_number2 = 0
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
        self.csc_dict = {}
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
            self.csc_dict['logical'] = self.logical_operator_between_combination_result.currentText()
        else:
            self.csc_dict['logical'] = 'None'
        if not np.all(spaces == "None"):
            for i, space in enumerate(spaces):
                if space != "None" and space != "None2":
                    self.csc_dict[space] = channels[i, :]
        if len(self.csc_dict) == 1 or channels.sum() == 0:
            self.csc_dict_is_empty = True
        else:
            self.csc_dict_is_empty = False

    def grid_segmentation_option(self):
        self.parent().po.vars["grid_segmentation"] = self.grid_segmentation.isChecked()

    def display_more_than_two_colors_option(self):
        """ should not do

            self.parent().po.all["more_than_two_colors"] = self.more_than_two_colors.isChecked()
            when init
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
        self.parent().po.vars["color_number"] = int(self.distinct_colors_number.value())

    def start_crop_scale_subtract_delineate(self):
        if not self.thread['CropScaleSubtractDelineate'].isRunning():
            self.message.setText("Looking for each arena contour, wait...")
            self.thread['CropScaleSubtractDelineate'].start()
            self.thread['CropScaleSubtractDelineate'].message_from_thread.connect(self.display_message_from_thread)
            self.thread['CropScaleSubtractDelineate'].message_when_thread_finished.connect(self.delineate_is_done)

            self.yes.setVisible(False)
            self.no.setVisible(False)
            # self.times_clicked_yes += 1
            self.reinitialize_bio_and_back_legend()
            self.user_drawn_lines_label.setVisible(False)
            self.cell.setVisible(False)
            self.background.setVisible(False)
            # self.sample_number.setVisible(False)
            # self.sample_number_label.setVisible(False)
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
            self.quickly.setVisible(False)
            self.carefully.setVisible(False)
            self.visualize.setVisible(False)
            self.visualize_label.setVisible(False)
            self.select_option.setVisible(False)
            self.select_option_label.setVisible(False)

    def delineate_is_done(self, message):
        logging.info("Delineation is done, update GUI")
        self.message.setText(message)
        self.arena_shape_label.setVisible(False)
        self.arena_shape.setVisible(False)
        self.reinitialize_bio_and_back_legend()
        self.reinitialize_image_and_masks(self.parent().po.first_image.bgr)
        self.delineation_done = True
        if self.thread["UpdateImage"].isRunning():
            self.thread["UpdateImage"].wait()
        self.thread["UpdateImage"].start()
        self.thread["UpdateImage"].message_when_thread_finished.connect(self.automatic_delineation_display_done)

        try:
            self.thread['CropScaleSubtractDelineate'].message_from_thread.disconnect()
            self.thread['CropScaleSubtractDelineate'].message_when_thread_finished.disconnect()
        except RuntimeError:
            pass
        if not self.slower_delineation_flag:
            self.asking_delineation_flag = True

    def automatic_delineation_display_done(self, boole):
        # Remove this flag to not draw it again next time UpdateImage runs for another reason
        self.delineation_done = False
        self.auto_delineation_flag = False
        self.select_option_label.setVisible(False)
        self.select_option.setVisible(False)

        self.arena_shape_label.setVisible(True)
        self.arena_shape.setVisible(True)

        self.decision_label.setText('Is video delineation correct?')
        self.decision_label.setVisible(True)
        # self.message.setText('If not, restart the analysis (Previous) or manually draw each arena (No)')
        self.user_drawn_lines_label.setText('Draw each arena on the image')
        self.yes.setVisible(True)
        self.no.setVisible(True)
        try:
            self.thread["UpdateImage"].message_when_thread_finished.disconnect()
        except RuntimeError:
            pass

    def display_message_from_thread(self, text_from_thread):
        self.message.setText(text_from_thread)

    def starting_differs_from_growing_check(self):
        if self.parent().po.all['first_detection_frame'] > 1:
            self.parent().po.vars['origin_state'] = 'invisible'
        else:
            if self.starting_differs_from_growing_cb.isChecked():
                self.parent().po.vars['origin_state'] = 'constant'
            else:
                self.parent().po.vars['origin_state'] = 'fluctuating'

    def when_yes_is_clicked(self):
        if not self.is_image_analysis_running:
            # self.message.setText('Loading, wait...')
            self.decision_tree(True)

    def when_no_is_clicked(self):
        if not self.is_image_analysis_running:
            # self.message.setText('Loading, wait...')
            self.decision_tree(False)

    def decision_tree(self, is_yes):
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
                if not is_yes:
                    self.asking_slower_or_manual_delineation()
                else:
                    self.last_image_question()
                self.asking_delineation_flag = False

            # Slower or manual delineation?
            elif self.asking_slower_or_manual_delineation_flag:
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
                        self.thread['SaveManualDelineation'].start()
                        self.last_image_question()
                        self.manual_delineation_flag = False
                    else:
                        self.message.setText(
                            f"{self.arena_masks_number} arenas are drawn over the {self.parent().po.sample_number} expected")

            elif self.asking_last_image_flag:
                self.parent().po.first_image.im_combinations = None
                self.select_option.clear()
                self.arena_shape.setVisible(False)
                self.arena_shape_label.setVisible(False)
                if is_yes:
                    self.start_last_image()
                    # if self.parent().po.vars['origin_state'] != 'invisible':
                    #     self.parent().po.vars['origin_state'] = "constant"
                else:
                    # if self.parent().po.vars['origin_state'] != 'invisible':
                    #     self.parent().po.vars['origin_state'] = "fluctuating"
                    self.parent().po.vars['convert_for_origin'] = deepcopy(self.csc_dict)
                    self.parent().po.vars['convert_for_motion'] = deepcopy(self.csc_dict)
                    self.go_to_next_widget()
                self.asking_last_image_flag = False
        else:
            if is_yes:
                self.parent().po.vars['convert_for_motion'] = deepcopy(self.csc_dict)
                self.go_to_next_widget()

    def first_im_parameters(self):
        """ Method called in the decision tree"""
        self.step = 1
        self.decision_label.setText("Adjust settings, draw more cells and background, and try again")
        self.yes.setVisible(False)
        self.no.setVisible(False)
        # self.one_blob_per_arena.setVisible(True)
        # self.one_blob_per_arena_label.setVisible(True)
        self.set_spot_shape.setVisible(True)
        self.spot_shape_label.setVisible(True)
        self.spot_shape.setVisible(self.parent().po.all['set_spot_shape'])
        self.set_spot_size.setVisible(self.one_blob_per_arena.isChecked())
        self.spot_size_label.setVisible(self.one_blob_per_arena.isChecked())
        self.spot_size.setVisible(
            self.one_blob_per_arena.isChecked() and self.set_spot_size.isChecked())
        # self.arena_shape.setVisible(True)
        # self.arena_shape_label.setVisible(True)
        self.auto_delineation_flag = True
        self.first_im_parameters_answered = True

    def auto_delineation(self):
        """ Method called in the decision tree"""
        # Do not proceed automatic delineation if there are more than one arena containing distinct spots
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
        self.asking_slower_or_manual_delineation_flag = True
        self.decision_label.setText(f"Click yes to try a slower but more efficient delineation algorithm, no to do it manually")
        self.message.setText(f"Clicking no will allow you to draw each arena manually")

    def slower_delineation(self):
        self.decision_label.setText(f"")
        self.arena_shape.setVisible(False)
        self.arena_shape_label.setVisible(False)
        # Save the current mask, its stats, remove useless memory and start delineation
        self.parent().po.first_image.update_current_images(self.parent().po.current_combination_id)
        self.parent().po.all['are_gravity_centers_moving'] = 1
        self.start_crop_scale_subtract_delineate()

    def manual_delineation(self):
        """ Method called in the decision tree"""
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
        self.arena_shape_label.setVisible(False)
        self.arena_shape.setVisible(False)
        self.no.setVisible(False)
        self.one_blob_per_arena.setVisible(False)
        self.one_blob_per_arena_label.setVisible(False)
        self.generate_analysis_options.setVisible(False)
        self.quickly.setVisible(False)
        self.carefully.setVisible(False)
        self.visualize.setVisible(False)
        self.visualize_label.setVisible(False)
        self.select_option.setVisible(False)
        self.select_option_label.setVisible(False)
        self.user_drawn_lines_label.setText("Draw each arena")
        self.user_drawn_lines_label.setVisible(True)
        self.decision_label.setText(
            f"Hold click to draw {self.parent().po.sample_number} arenas on the image")
        self.message.setText('Click Yes when it is done')

    def last_image_question(self):
        """ Method called in the decision tree"""
        self.decision_label.setText(
            'Do you want to check if the current parameters work for the last image:')
        self.message.setText('Click Yes if the cell color may change during the analysis.')
        self.yes.setVisible(True)
        self.no.setVisible(True)
        self.starting_differs_from_growing_cb.setVisible(True)
        self.starting_differs_from_growing_label.setVisible(True)
        self.image_number.setVisible(False)
        self.image_number_label.setVisible(False)
        self.read.setVisible(False)
        self.asking_last_image_flag = True
        # self.title_label.setVisible(False)
        self.step = 2

    def start_last_image(self):
        self.is_first_image_flag = False
        # self.parent().po.vars["color_number"] = 2
        self.decision_label.setText('')
        self.yes.setVisible(False)
        self.no.setVisible(False)
        self.spot_size.setVisible(False)
        self.starting_differs_from_growing_cb.setVisible(False)
        self.starting_differs_from_growing_label.setVisible(False)
        self.message.setText('Gathering data and visualizing last image analysis result')
        self.parent().po.get_last_image()
        if self.thread['SaveManualDelineation'].isRunning():
            self.thread['SaveManualDelineation'].wait()
        self.parent().po.cropping(is_first_image=False)
        # self.parent().po.last_image = OneImageAnalysis(self.parent().po.last_im)
        self.reinitialize_image_and_masks(self.parent().po.last_image.bgr)
        self.reinitialize_bio_and_back_legend()
        self.parent().po.current_combination_id = 0
        # self.advanced_mode_cb.setChecked(True)
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
        # self.title_label.setVisible(True)
        # self.row1_col1_widget.setVisible(False)
        # self.row1_col2_widget.setVisible(False)

    def go_to_next_widget(self):
        if not self.thread['SaveManualDelineation'].isRunning() or not self.thread['FinalizeImageAnalysis'].isRunning() or not self.thread['SaveData'].isRunning():

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
            self.thread['FinalizeImageAnalysis'].start()
            if self.parent().po.vars["color_number"] > 2:
                self.parent().videoanalysiswindow.select_option.clear()
                self.parent().videoanalysiswindow.select_option.addItem(f"1) Kmeans")
                self.parent().videoanalysiswindow.select_option.setCurrentIndex(0)
                self.parent().po.all['video_option'] = 0
            time.sleep(1 / 10)
            self.thread['FinalizeImageAnalysis'].wait()
            self.message.setText(f"")

            self.video_tab.set_not_in_use()
            self.parent().last_tab = "image_analysis"
            self.parent().change_widget(3)  # VideoAnalysisWindow

            self.popup.close()

    def closeEvent(self, event):
        event.accept


# if __name__ == "__main__":
#     from cellects.gui.cellects import CellectsMainWidget
#     import sys
#     app = QtWidgets.QApplication([])
#     parent = CellectsMainWidget()
#     session = ImageAnalysisWindow(parent, False)
#     parent.insertWidget(0, session)
#     parent.show()
#     sys.exit(app.exec())
