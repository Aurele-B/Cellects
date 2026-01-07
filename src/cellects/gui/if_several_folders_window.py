#!/usr/bin/env python3
"""GUI window for selecting folders when multiple options exist in Cellects analysis workflow.

This module implements a second-stage GUI dialog that appears when multiple experiment folders are available.
It provides an interface for folder selection via checkboxes and table visualization, with navigation controls to
proceed to image analysis or return to prior steps. Includes thread-safe background operations for loading data without
freezing the UI.

Main Components
IfSeveralFoldersWindow : QWidget subclass managing folder selection and analysis workflow navigation
"""
import logging
import numpy as np
from PySide6 import QtWidgets, QtCore
from cellects.core.cellects_threads import LoadFirstFolderIfSeveralThread
from cellects.gui.custom_widgets import (WindowType, PButton, FixedText)
from cellects.gui.ui_strings import MF, VAW


class IfSeveralFoldersWindow(WindowType):
    """
    Second window of the Cellects GUI, only appears when there are multiple folders.
    """
    def __init__(self, parent, night_mode):
        """
        Initialize the IfSeveralFolders window with a parent widget and night mode setting.

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
        >>> from cellects.gui.if_several_folders_window import IfSeveralFoldersWindow
        >>> import sys
        >>> app = QtWidgets.QApplication([])
        >>> parent = CellectsMainWidget()
        >>> session = IfSeveralFoldersWindow(parent, False)
        >>> session.true_init()
        >>> parent.insertWidget(0, session)
        >>> parent.show()
        >>> sys.exit(app.exec())
        """
        super().__init__(parent, night_mode)
        self.setParent(parent)

    def true_init(self):
        """
        Initialize the IfSeveralFoldersWindow with UI components and settings.

        Extended Description
        --------------------
        This method sets up the user interface for the IfSeveralFoldersWindow,
        including labels, a table widget for folders and sample sizes, checkboxes,
        buttons for video analysis and running tasks directly, and navigation
        buttons for previous and next steps. The window supports multiple folder
        selection and provides a means to control the analysis process.

        Notes
        -----
        This method assumes that the parent widget has a 'po' attribute with specific settings and variables.
        """
        logging.info("Initialize IfSeveralFoldersWindow")
        self.thread_dict = {}
        self.thread_dict["LoadFirstFolderIfSeveral"] = LoadFirstFolderIfSeveralThread(self.parent())
        self.next_clicked_once:bool = False
        self.layout = QtWidgets.QVBoxLayout()

        self.title_label = FixedText('Select folders to analyze', police=30, night_mode=self.parent().po.all['night_mode'])
        self.title_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.layout.addWidget(self.title_label)
        self.layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))

        # 1) add a check box allowing to select every folders
        self.cb_layout = QtWidgets.QHBoxLayout()
        self.cb_widget = QtWidgets.QWidget()
        self.cb_label = FixedText(MF["Check_to_select_all_folders"]["label"] + ':', tip=MF["Check_to_select_all_folders"]["tips"], night_mode=self.parent().po.all['night_mode'])
        self.cb = QtWidgets.QCheckBox()
        self.cb.setChecked(True)
        self.cb.clicked.connect(self.checked)
        self.cb_layout.addWidget(self.cb_label)
        self.cb_layout.addWidget(self.cb)
        self.cb_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.cb_widget.setLayout(self.cb_layout)
        self.layout.addWidget(self.cb_widget)

        # 2) Create a folder list and sample number per folder
        self.tableau = QtWidgets.QTableWidget()  # Scroll Area which contains the widgets, set as the centralWidget
        self.tableau.setColumnCount(2)
        self.tableau.setRowCount(len(self.parent().po.all['folder_list']))
        self.tableau.setHorizontalHeaderLabels(['Folders', 'Sample size'])
        self.parent().po.all['sample_number_per_folder'] = np.repeat(int(self.parent().po.all['first_folder_sample_number']), self.parent().po.all['folder_number'])

        for i, folder in enumerate(self.parent().po.all['folder_list']):
            self.tableau.setItem(i, 0, QtWidgets.QTableWidgetItem(folder))
            self.tableau.setItem(i, 1, QtWidgets.QTableWidgetItem(str(self.parent().po.all['sample_number_per_folder'][i])))
        self.tableau.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tableau.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tableau.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableau.selectAll()
        self.tableau.itemSelectionChanged.connect(self.item_selection_changed)

        self.tableau.setShowGrid(False)
        self.tableau.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.tableau.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tableau.horizontalHeader().hide()
        self.tableau.verticalHeader().hide()
        self.layout.addWidget(self.tableau)

        # Create the shortcuts row
        self.shortcuts_widget = QtWidgets.QWidget()
        self.shortcuts_layout = QtWidgets.QHBoxLayout()
        self.Video_analysis_window = PButton("Video tracking window", night_mode=self.parent().po.all['night_mode'])
        self.Video_analysis_window.clicked.connect(self.Video_analysis_window_is_clicked)
        self.Run_all_directly = PButton("Run all directly", tip=VAW["Run_All"]["tips"],
                                        night_mode=self.parent().po.all['night_mode'])
        self.Run_all_directly.clicked.connect(self.Run_all_directly_is_clicked)
        self.Video_analysis_window.setVisible(False)
        self.Run_all_directly.setVisible(False)
        self.shortcuts_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))
        self.shortcuts_layout.addWidget(self.Video_analysis_window)
        self.shortcuts_layout.addWidget(self.Run_all_directly)
        self.shortcuts_widget.setLayout(self.shortcuts_layout)
        self.layout.addWidget(self.shortcuts_widget)

        # 3) Previous button
        self.last_row_layout = QtWidgets.QHBoxLayout()
        self.last_row_widget = QtWidgets.QWidget()
        self.previous = PButton('Previous', night_mode=self.parent().po.all['night_mode'])
        self.previous.clicked.connect(self.previous_is_clicked)

        # 4) Message
        self.message = QtWidgets.QLabel(self)
        self.message.setText('')
        self.message.setStyleSheet("color: rgb(230, 145, 18)")
        self.message.setAlignment(QtCore.Qt.AlignRight)

        # 5) Next button
        self.next = PButton('Next', night_mode=self.parent().po.all['night_mode'])
        self.next.clicked.connect(self.next_is_clicked)
        self.last_row_layout.addWidget(self.previous)
        self.last_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.last_row_layout.addWidget(self.message)
        self.last_row_layout.addWidget(self.next)
        self.last_row_widget.setLayout(self.last_row_layout)
        self.layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))
        self.layout.addWidget(self.last_row_widget)
        self.setLayout(self.layout)

    def checked(self):
        """
        Check or uncheck all entries in the tableau based on checkbox state.

        If the associated checkbox is checked, select all items in the
        tableau. Otherwise, clear the selection.
        """
        if self.cb.isChecked():
            self.tableau.selectAll()
        else:
            self.tableau.clearSelection()

    def item_selection_changed(self):
        """
        Update the checkbox state based on the number of selected items.
        """
        if (len(self.tableau.selectedItems()) // 2) == len(self.parent().po.all['folder_list']):
            self.cb.setChecked(True)
        else:
            self.cb.setChecked(False)

    def previous_is_clicked(self):
        """
        Handles the logic for when a "Previous" button is clicked in the interface, leading to the FirstWindow.

        It modifies internal state to force the decision between IfSeveralFoldersWindow and ImageAnalysisWindow to be
        done once again if the user clicks on the "Next" button of the FirstWindow.
        """
        self.next_clicked_once = False
        self.parent().firstwindow.instantiate = True
        self.parent().change_widget(0)

    def next_is_clicked(self):
        """
        Handles the logic for when a "Next" button is clicked in the interface, leading to the ImageAnalysisWindow.

        If `self.next_clicked_once` is True, instanties widgets and performs image analysis.
        Otherwise, checks for selected folders and samples. Updates internal state and starts
        a thread for loading the first folder if multiple folders are selected.

        Notes
        -----
        This function updates the internal state based on user selection and starts a thread
        for loading data. The `self.parent().po.update_folder_id` method is called to update
        folder IDs.
        """
        if self.next_clicked_once:
            self.instantiates_widgets_and_do_image_analysis()
        else:
            self.message.setText("Loading, wait...")
            item_number = len(self.tableau.selectedItems())
            if item_number == 0:
                self.message.setText("Select at least one folder")
            else:
                sample_number_per_folder = []
                folder_list = []
                for i in np.arange(item_number):
                    if i % 2 == 0:
                        folder_list.append(self.tableau.selectedItems()[i].text())
                    else:
                        sample_number_per_folder.append(int(self.tableau.selectedItems()[i].text()))
                self.parent().po.all['first_folder_sample_number'] = int(self.tableau.selectedItems()[1].text())

                self.parent().po.all['folder_list'] = folder_list
                self.parent().po.all['sample_number_per_folder'] = sample_number_per_folder
                self.parent().po.update_folder_id(self.parent().po.all['first_folder_sample_number'],
                                                  self.parent().po.all['folder_list'][0])
                self.thread_dict["LoadFirstFolderIfSeveral"].start()
                self.thread_dict["LoadFirstFolderIfSeveral"].message_when_thread_finished.connect(self.first_folder_loaded)

    def first_folder_loaded(self, first_exp_ready_to_run: bool):
        """
        Set the visibility of widgets and messages based on whether data is found.

        Parameters
        ----------
        first_exp_ready_to_run : bool
            Indicates if the one_experiment data is ready to be run.
        """
        if first_exp_ready_to_run:
            self.cb_widget.setVisible(False)
            self.tableau.setVisible(False)
            if len(self.parent().po.vars['analyzed_individuals']) != self.parent().po.all['first_folder_sample_number']:
                self.parent().po.vars['analyzed_individuals'] = np.arange(
                    self.parent().po.all['first_folder_sample_number']) + 1
                self.parent().po.sample_number = self.parent().po.all['first_folder_sample_number']
            self.message.setText("Data found, shortcuts are available. Click Next again to redo/improve the image analysis")
            self.next_clicked_once = True
            self.Video_analysis_window.setVisible(True)
            self.Run_all_directly.setVisible(True)
            self.parent().firstwindow.Video_analysis_window.setVisible(True)
            self.parent().firstwindow.Run_all_directly.setVisible(True)
        else:
            self.instantiates_widgets_and_do_image_analysis()

    def instantiates_widgets_and_do_image_analysis(self):
        """
        Instantiate widgets and initialize the image analysis window.
        -----
        This function is responsible for:
            - Instantiating widgets with a specific condition.
            - Initializing the true initialization of the image analysis window.
            - Remember to not re-instantiate the image analysis window if the user goes back to the first window.
            - Changing the current widget to ImageAnalysisWindow.
        """
        self.parent().instantiate_widgets(severalfolder_included=False)
        self.parent().imageanalysiswindow.true_init()
        self.parent().firstwindow.instantiate = False
        self.parent().change_widget(2)# ImageAnalysisWindow

    def Video_analysis_window_is_clicked(self):
        """
        Save the identity of the current widget (for future navigation) and change to the video analysis window.
        """
        self.parent().last_tab = "data_specifications"
        self.parent().change_widget(3)

    def Run_all_directly_is_clicked(self):
        """
        Run the "Run all directly" operation in the parent window.

        This function triggers the execution of the "Run all directly"
        functionality by calling the corresponding method in the parent
        window and then updates the state accordingly.
        """
        self.parent().firstwindow.Run_all_directly_is_clicked()
        self.previous_is_clicked()