#!/usr/bin/env python3
"""
This module contains the widget allowing the user to set which variables Cellects will compute during analysis.
A first kind of variable is raw data: presence/absence coordinates of the specimens, network, oscillating pixels
A second kind of variable describe the specimen at each time frame and for each arena of the image stack or video
"""

from PySide6 import QtWidgets, QtCore
import logging
from cellects.gui.custom_widgets import (
    WindowType, PButton, Checkbox, FixedText)
from cellects.image_analysis.shape_descriptors import descriptors_names_to_display


class RequiredOutput(WindowType):
    def __init__(self, parent, night_mode):
        super().__init__(parent, night_mode)
        self.setParent(parent)
        # Create the main Title
        self.true_init(night_mode)

    def true_init(self, night_mode):

        logging.info("Initialize RequiredOutput window")
        self.title = FixedText('Required Output', police=30, night_mode=self.parent().po.all['night_mode'])
        self.title.setAlignment(QtCore.Qt.AlignHCenter)
        # Create the main layout
        self.vlayout = QtWidgets.QVBoxLayout()
        self.vlayout.addWidget(self.title) #
        horzspaceItem = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.vlayout.addItem(horzspaceItem) #

        # Create the stylesheet for the boxes allowing to categorize required outputs.
        boxstylesheet = \
            ".QWidget {\n" \
            + "border: 1px solid black;\n" \
            + "border-radius: 20px;\n" \
            + "}"

        # I/ First box: Save presence coordinates
        # I/A/ Title
        self.save_presence_coordinates_label = FixedText('Save presence coordinates:', tip="Saved in the python numpy format: .npy",
                                                 night_mode=self.parent().po.all['night_mode'])
        self.vlayout.addWidget(self.save_presence_coordinates_label) #

        # I/B/ Create the box
        self.save_presence_coordinates_layout = QtWidgets.QGridLayout()
        self.save_presence_coordinates_widget = QtWidgets.QWidget()
        self.save_presence_coordinates_widget.setStyleSheet(boxstylesheet)

        # I/C/ Create widgets
        self.save_coord_specimen = Checkbox(self.parent().po.vars['save_coord_specimen'])
        # self.save_coord_specimen.stateChanged.connect(self.save_coord_specimen_saving)
        self.save_coord_specimen_label = FixedText('All pixels covered by the specimen(s)', tip="",
                                           night_mode=self.parent().po.all['night_mode'])

        self.save_coord_contour = Checkbox(self.parent().po.vars['save_coord_contour'])
        # self.save_coord_contour.stateChanged.connect(self.save_coord_contour_saving)
        self.save_coord_contour_label = FixedText('Contours of the specimen(s)', tip="",
                                           night_mode=self.parent().po.all['night_mode'])
        self.save_coord_thickening_slimming = Checkbox(self.parent().po.vars['save_coord_thickening_slimming'])
        # self.save_coord_thickening_slimming.stateChanged.connect(self.save_coord_thickening_slimming_saving)
        self.save_coord_thickening_slimming_label = FixedText('Thickening and slimming areas in the specimen(s)', tip="",
                                           night_mode=self.parent().po.all['night_mode'])
        self.save_coord_network = Checkbox(self.parent().po.vars['save_coord_network'])
        # self.save_coord_network.stateChanged.connect(self.save_coord_network_saving)
        self.save_coord_network_label = FixedText('Tubular network in the specimen(s)', tip="",
                                           night_mode=self.parent().po.all['night_mode'])

        # I/D/ Arrange widgets in the box
        self.save_presence_coordinates_layout.addWidget(self.save_coord_specimen_label, 0, 0)
        self.save_presence_coordinates_layout.addWidget(self.save_coord_specimen, 0, 1)
        self.save_presence_coordinates_layout.addWidget(self.save_coord_contour_label, 1, 0)
        self.save_presence_coordinates_layout.addWidget(self.save_coord_contour, 1, 1)

        self.save_presence_coordinates_layout.addWidget(self.save_coord_thickening_slimming_label, 0, 2)
        self.save_presence_coordinates_layout.addWidget(self.save_coord_thickening_slimming, 0, 3)
        self.save_presence_coordinates_layout.addWidget(self.save_coord_network_label, 1, 2)
        self.save_presence_coordinates_layout.addWidget(self.save_coord_network, 1, 3)

        self.save_presence_coordinates_widget.setLayout(self.save_presence_coordinates_layout)
        self.vlayout.addWidget(self.save_presence_coordinates_widget)

        # II/ Second box: Save descriptors
        # II/A/ Title
        self.save_descriptors_label = FixedText('Save descriptors:',
                                                         tip="Saved in .csv",
                                                         night_mode=self.parent().po.all['night_mode'])
        self.vlayout.addWidget(self.save_descriptors_label)  #

        # II/B/ Create the box
        self.save_descriptors_layout = QtWidgets.QGridLayout()
        self.save_descriptors_widget = QtWidgets.QWidget()
        self.save_descriptors_widget.setStyleSheet(boxstylesheet)

        # II/C/ Create widgets

        self.descriptor_widgets_list = []

        # Create the table of the main output the user can select
        self.create_check_boxes_table()
        # II/D/ Set the layout
        self.save_descriptors_widget.setLayout(self.save_descriptors_layout)
        self.vlayout.addWidget(self.save_descriptors_widget)


        # Create the last row layout that will contain a few more output and the ok button.
        self.last_row_layout = QtWidgets.QHBoxLayout()
        self.last_row_widget = QtWidgets.QWidget()

        self.cancel = PButton('Cancel', night_mode=self.parent().po.all['night_mode'])
        self.cancel.clicked.connect(self.cancel_is_clicked)
        self.ok = PButton('Ok', night_mode=self.parent().po.all['night_mode'])
        self.ok.clicked.connect(self.ok_is_clicked)
        self.last_row_layout.addItem(self.horizontal_space)
        self.last_row_layout.addWidget(self.cancel)
        self.last_row_layout.addWidget(self.ok)

        self.vlayout.addItem(horzspaceItem)
        vertspaceItem = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)

        self.last_row_widget.setLayout(self.last_row_layout)
        self.vlayout.addWidget(self.last_row_widget)

        self.hlayout = QtWidgets.QHBoxLayout()
        self.vwidget = QtWidgets.QWidget()
        self.vwidget.setLayout(self.vlayout)
        self.hlayout.addItem(vertspaceItem)
        self.hlayout.addWidget(self.vwidget)
        self.hlayout.addItem(vertspaceItem)
        self.setLayout(self.hlayout)

    def create_check_boxes_table(self):
        """
            Loop over all main outputs. An output is a variable allowing to describe the binary image
            showing the presence/absence of the cell/colony at one time frame.
            This function
        """
        descriptor_names = self.parent().po.all['descriptors']

        for i, name in enumerate(descriptor_names):
            label_index = i * 2
            if i > 9:
                row = i - 10 + 1 + 3
                col = 4
            else:
                row = i + 1 + 3
                col = 1
            self.descriptor_widgets_list.append(FixedText(descriptors_names_to_display[i], 14, night_mode=self.parent().po.all['night_mode']))
            self.save_descriptors_layout.addWidget(self.descriptor_widgets_list[label_index], row, col)
            self.descriptor_widgets_list.append(Checkbox(self.parent().po.all['descriptors'][name]))
            cb_index = label_index + 1

            if name == 'fractal_analysis':# or name == 'network_analysis':
                self.descriptor_widgets_list[label_index].setVisible(False)
                self.descriptor_widgets_list[cb_index].setVisible(False)

            self.save_descriptors_layout.addWidget(self.descriptor_widgets_list[cb_index], row, col + 1)

    def cancel_is_clicked(self):
        self.save_coord_specimen.setChecked(self.parent().po.vars['save_coord_specimen'])
        self.save_coord_contour.setChecked(self.parent().po.vars['save_coord_contour'])
        self.save_coord_thickening_slimming.setChecked(self.parent().po.vars['save_coord_thickening_slimming'])
        self.save_coord_network.setChecked(self.parent().po.vars['save_coord_network'])

        descriptor_names = self.parent().po.all['descriptors']
        for i, name in enumerate(descriptor_names):
            k = i * 2 + 1
            if name == 'iso_digi_analysis':
                self.descriptor_widgets_list[k].setChecked(self.parent().po.vars['iso_digi_analysis'])
            elif name == 'oscilacyto_analysis':
                self.descriptor_widgets_list[k].setChecked(self.parent().po.vars['oscilacyto_analysis'])
            elif name == 'fractal_analysis':
                self.descriptor_widgets_list[k].setChecked(self.parent().po.vars['fractal_analysis'])
            elif name == 'network_analysis':
                self.descriptor_widgets_list[k].setChecked(self.parent().po.vars['network_analysis'])
            elif name == 'graph_extraction':
                self.descriptor_widgets_list[k].setChecked(self.parent().po.vars['graph_extraction'])
            else:
                self.descriptor_widgets_list[k].setChecked(self.parent().po.all['descriptors'][name])

        if self.parent().last_is_first:
            self.parent().change_widget(0) # FirstWidget
        else:
            self.parent().change_widget(3) # ThirdWidget

    def ok_is_clicked(self):
        self.parent().po.vars['save_coord_specimen'] = self.save_coord_specimen.isChecked()
        self.parent().po.vars['save_coord_contour'] = self.save_coord_contour.isChecked()
        self.parent().po.vars['save_coord_thickening_slimming'] = self.save_coord_thickening_slimming.isChecked()
        self.parent().po.vars['save_coord_network'] = self.save_coord_network.isChecked()
        descriptor_names = self.parent().po.all['descriptors'].keys()
        for i, name in enumerate(descriptor_names):
            k = i * 2 + 1
            checked_status = self.descriptor_widgets_list[k].isChecked()
            self.parent().po.all['descriptors'][name] = checked_status
            if name == 'iso_digi_analysis':
                self.parent().po.vars['iso_digi_analysis'] = checked_status
            if name == 'oscilacyto_analysis':
                self.parent().po.vars['oscilacyto_analysis'] = checked_status
            if name == 'fractal_analysis':
                self.parent().po.vars['fractal_analysis'] = checked_status
            if name == 'network_analysis':
                self.parent().po.vars['network_analysis'] = checked_status
            if name == 'graph_extraction':
                self.parent().po.vars['graph_extraction'] = checked_status

            # for j in [0, 1, 2]:
            #     k = i * 4 + j + 1
            #     self.parent().po.all['descriptors'][name][j] = self.descriptor_widgets_list[k].isChecked()
        if not self.parent().thread['SaveAllVars'].isRunning():
            self.parent().thread['SaveAllVars'].start()
        self.parent().po.update_output_list()
        if self.parent().last_is_first:
            self.parent().change_widget(0) # FirstWidget
        else:
            self.parent().change_widget(3) # ThirdWidget

    def closeEvent(self, event):
        event.accept


# if __name__ == "__main__":
#     from cellects.gui.cellects import CellectsMainWidget
#     import sys
#
#     app = QtWidgets.QApplication([])
#     parent = CellectsMainWidget()
#     session = RequiredOutput(parent, False)
#     parent.insertWidget(0, session)
#     parent.show()
#     sys.exit(app.exec())
