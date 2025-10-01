#!/usr/bin/env python3
"""ADD DETAIL OF THE MODULE"""
import os
import logging
from numpy import min, max, all, any, arange, repeat
from PySide6 import QtWidgets, QtCore

from cellects.core.cellects_threads import LoadFirstFolderIfSeveralThread
from cellects.gui.custom_widgets import (
    WindowType, PButton, FixedText)


class IfSeveralFoldersWindow(WindowType):
    def __init__(self, parent, night_mode):
        super().__init__(parent, night_mode)
        self.setParent(parent)

    def true_init(self):
        logging.info("Initialize IfSeveralFoldersWindow")
        self.thread = {}
        self.thread["LoadFirstFolderIfSeveral"] = LoadFirstFolderIfSeveralThread(self.parent())
        self.next_clicked_once:bool = False
        self.layout = QtWidgets.QVBoxLayout()
        # self.layout.setAlignment(QtCore.Qt.AlignLeading)
        # self.layout.setGeometry(QtCore.QRect(9, 9, 2 * self.parent().screen_width // 3, 2 * self.parent().screen_height // 3))

        self.title_label = FixedText('Select folders to analyze', police=30, night_mode=self.parent().po.all['night_mode'])
        self.title_label.setAlignment(QtCore.Qt.AlignHCenter)
        # self.layout.addWidget(self.title_label, 0, 0, 1, - 1)
        self.layout.addWidget(self.title_label)
        self.layout.addItem(self.vertical_space)

        # 1) add a check box allowing to select every folders
        self.cb_layout = QtWidgets.QHBoxLayout()
        self.cb_widget = QtWidgets.QWidget()
        self.cb_label = FixedText('Check to select all folders:', tip="Otherwise, use Ctrl/Cmd to select the folders to analyze", night_mode=self.parent().po.all['night_mode'])
        self.cb = QtWidgets.QCheckBox()
        self.cb.setChecked(True)
        self.cb.clicked.connect(self.checked)
        # self.cb.stateChanged.connect(self.checked)

        # self.layout.addWidget(self.cb_label, 1, 0, 1, 1)
        # self.layout.addWidget(self.cb, 1, 1, 1, 1)
        self.cb_layout.addWidget(self.cb_label)
        self.cb_layout.addWidget(self.cb)
        self.cb_layout.addItem(self.horizontal_space)
        self.cb_widget.setLayout(self.cb_layout)
        self.layout.addWidget(self.cb_widget)
        # spacerItem = QtWidgets.QSpacerItem(0, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        # self.layout.addItem(spacerItem, 1, 2, 1, 3)

        # 2) Create a folder list and sample number per folder
        self.tableau = QtWidgets.QTableWidget()  # Scroll Area which contains the widgets, set as the centralWidget
        self.tableau.setColumnCount(2)
        self.tableau.setRowCount(len(self.parent().po.all['folder_list']))
        self.tableau.setHorizontalHeaderLabels(['Folders', 'Sample size'])
        # if len(self.parent().po.all['sample_number_per_folder']) < 2:
        self.parent().po.all['sample_number_per_folder'] = np.repeat(int(self.parent().po.all['first_folder_sample_number']), self.parent().po.all['folder_number'])

        for i, folder in enumerate(self.parent().po.all['folder_list']):
            self.tableau.setItem(i, 0, QtWidgets.QTableWidgetItem(folder))
            self.tableau.setItem(i, 1, QtWidgets.QTableWidgetItem(str(self.parent().po.all['sample_number_per_folder'][i])))
            # if isinstance(self.parent().po.all['sample_number_per_folder'], int):
            #     self.tableau.setItem(i, 1, QtWidgets.QTableWidgetItem(str(self.parent().po.all['sample_number_per_folder'])))
            # else:
            #     self.tableau.setItem(i, 1, QtWidgets.QTableWidgetItem(str(self.parent().po.all['sample_number_per_folder'][i])))
        self.tableau.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tableau.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tableau.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableau.selectAll()
        self.tableau.itemSelectionChanged.connect(self.item_selection_changed)

        # self.tableau.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tableau.setShowGrid(False)
        self.tableau.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.tableau.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.tableau.horizontalHeader().hide()
        self.tableau.verticalHeader().hide()
        # self.layout.addWidget(self.tableau, 2, 0, 1, 1)
        self.layout.addWidget(self.tableau)

        # spaceItem = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        # self.layout.addItem(spaceItem, 3, 0)

        # Create the shortcuts row
        self.shortcuts_widget = QtWidgets.QWidget()
        self.shortcuts_layout = QtWidgets.QHBoxLayout()
        self.Video_analysis_window = PButton("Video tracking window", night_mode=self.parent().po.all['night_mode'])
        self.Video_analysis_window.clicked.connect(self.Video_analysis_window_is_clicked)
        self.Run_all_directly = PButton("Run all directly", night_mode=self.parent().po.all['night_mode'])
        self.Run_all_directly.clicked.connect(self.Run_all_directly_is_clicked)
        self.Video_analysis_window.setVisible(False)
        self.Run_all_directly.setVisible(False)
        self.shortcuts_layout.addItem(self.vertical_space)
        self.shortcuts_layout.addWidget(self.Video_analysis_window)
        self.shortcuts_layout.addWidget(self.Run_all_directly)
        self.shortcuts_widget.setLayout(self.shortcuts_layout)
        self.layout.addWidget(self.shortcuts_widget)

        # 3) Previous button
        self.last_row_layout = QtWidgets.QHBoxLayout()
        self.last_row_widget = QtWidgets.QWidget()
        self.previous = PButton('Previous', night_mode=self.parent().po.all['night_mode'])
        # self.layout.addWidget(self.previous, 4, 0)
        self.previous.clicked.connect(self.previous_is_clicked)

        # 4) Message
        self.message = QtWidgets.QLabel(self)
        self.message.setText('')
        self.message.setStyleSheet("color: rgb(230, 145, 18)")
        self.message.setAlignment(QtCore.Qt.AlignRight)
        # self.layout.addWidget(self.message, 4, 1)

        # 5) Next button
        self.next = PButton('Next', night_mode=self.parent().po.all['night_mode'])
        # self.layout.addWidget(self.next, 4, 2)
        self.next.clicked.connect(self.next_is_clicked)
        # self.setLayout(self.layout)
        self.last_row_layout.addWidget(self.previous)
        self.last_row_layout.addItem(self.horizontal_space)
        self.last_row_layout.addWidget(self.message)
        self.last_row_layout.addWidget(self.next)
        self.last_row_widget.setLayout(self.last_row_layout)
        self.layout.addItem(self.vertical_space)
        self.layout.addWidget(self.last_row_widget)
        self.setLayout(self.layout)

    def checked(self):
        if self.cb.isChecked():
            self.tableau.selectAll()
        else:
            self.tableau.clearSelection()

    def item_selection_changed(self):
        if (len(self.tableau.selectedItems()) // 2) == len(self.parent().po.all['folder_list']):
            self.cb.setChecked(True)
        else:
            self.cb.setChecked(False)

    def previous_is_clicked(self):
        self.next_clicked_once = False
        self.parent().firstwindow.instantiate = True
        self.parent().change_widget(0)

    def next_is_clicked(self):
        if self.next_clicked_once:
            self.instantiates_widgets_and_do_image_analysis()
        else:
            self.message.setText("Loading, wait...")
            item_number = len(self.tableau.selectedItems())
            if item_number == 0:
                self.message.setText("Select at least one folder")
            else:
                # self.tableau.selectedItems()
                sample_number_per_folder = []
                folder_list = []
                # sample_number =
                # if isinstance(self.parent().po.all['sample_number_per_folder'], int):
                #     sample_number = self.parent().po.all['sample_number_per_folder']
                for i in np.arange(item_number):
                    if i % 2 == 0:
                        folder_list.append(self.tableau.selectedItems()[i].text())
                    else:
                        sample_number_per_folder.append(int(self.tableau.selectedItems()[i].text()))
                self.parent().po.all['first_folder_sample_number'] = int(self.tableau.selectedItems()[1].text())

                #
                # for index in self.tableau.selectionModel().selectedRows():
                #     folder_list.append(self.parent().po.all['folder_list'][index.row()])
                #     # if not isinstance(self.parent().po.all['sample_number_per_folder'], int):
                #     sample_number = self.parent().po.all['sample_number_per_folder'][index.row()]
                #     sample_number_per_folder.append(sample_number)

                self.parent().po.all['folder_list'] = folder_list
                self.parent().po.all['sample_number_per_folder'] = sample_number_per_folder
                self.parent().po.update_folder_id(self.parent().po.all['first_folder_sample_number'],
                                                  self.parent().po.all['folder_list'][0])
                print(os.getcwd())
                # if not isinstance(self.parent().po.all['sample_number_per_folder'], int):
                #     self.parent().po.update_folder_id(self.parent().po.all['sample_number_per_folder'][0], self.parent().po.all['folder_list'][0])
                # else:
                #     self.parent().po.update_folder_id(self.parent().po.all['sample_number_per_folder'], self.parent().po.all['folder_list'][0])

                # if self.parent.subwidgets_stack.count() == 5:
                #     self.parent.instantiate_widget(ImageAnalysisWindow(self.parent))
                # if not self.parent().imageanalysiswindow.initialized:

                self.thread["LoadFirstFolderIfSeveral"].start()
                self.thread["LoadFirstFolderIfSeveral"].message_when_thread_finished.connect(self.first_folder_loaded)

    def first_folder_loaded(self, first_exp_ready_to_run):
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
        print(os.getcwd())
        self.parent().instantiate_widgets(severalfolder_included=False)
        self.parent().imageanalysiswindow.true_init()
        self.parent().firstwindow.instantiate = False
        print(os.getcwd())
        self.parent().change_widget(2)# ImageAnalysisWindow
        # self.parent().change_widget(3)  # VideoAnalysisWindow

    def Video_analysis_window_is_clicked(self):
        self.parent().last_tab = "data_specifications"
        # self.parent().po.update_folder_id(self.parent().po.all['sample_number_per_folder'][0],
        #                                   self.parent().po.all['folder_list'][0])
        self.parent().change_widget(3)

    def Run_all_directly_is_clicked(self):
        self.parent().firstwindow.Run_all_directly_is_clicked()
        self.previous_is_clicked()

    def closeEvent(self, event):
        event.accept


# if __name__ == "__main__":
#     from cellects.gui.cellects import CellectsMainWidget
#     import sys
#     app = QtWidgets.QApplication([])
#     parent = CellectsMainWidget()
#     session = IfSeveralFoldersWindow(parent, False)
#     parent.insertWidget(0, session)
#     parent.show()
#     sys.exit(app.exec())
