#!/usr/bin/env python3
"""ADD DETAIL OF THE MODULE"""

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
        self.create_gui(night_mode)

    def create_gui(self, night_mode):

        logging.info("Initialize RequiredOutput window")
        self.title = FixedText('Required Output', police=30, night_mode=self.parent().po.all['night_mode'])
        self.title.setAlignment(QtCore.Qt.AlignHCenter)
        # Create the main layout
        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(self.title, 1, 1, 1, 9)
        horzspaceItem = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.layout.addItem(horzspaceItem, 2, 0, 1, 9)

        # Create the column titles
        self.descriptor_label1 = FixedText('Descriptor', align="l", tip="Check all wanted descriptors as output of all analyses\nThese outputs will be stored in tables named statistics.csv and descriptors_in_long_format.csv", night_mode=self.parent().po.all['night_mode'])
        self.layout.addWidget(self.descriptor_label1, 3, 1)
        self.save_all_label1 = FixedText('Save', align="c", tip="Will add a column containing that descriptor in descriptors_in_long_format.csv\nEach row will contain the value of that descriptor at one time for one arena", night_mode=self.parent().po.all['night_mode'])
        self.layout.addWidget(self.save_all_label1, 3, 2)
        # self.save_mean_label1 = FixedText('Mean', align="c", tip="Will use descriptors_in_long_format.csv to store the average of that descriptor in statistics.csv")
        # self.layout.addWidget(self.save_mean_label1, 3, 3)
        # self.save_reg_label1 = FixedText('LinReg', align="c", tip="Will use descriptors_in_long_format.csv to make a linear regression\nof that descriptor when it varies the most\nand store this time interval,\nregression intercept and slope in statistics.csv")
        # self.layout.addWidget(self.save_reg_label1, 3, 4)

        self.descriptor_label2 = FixedText('Descriptor', align="l", tip="Check all wanted descriptors as output of all analyses\nThese outputs will be stored in tables named statistics.csv and descriptors_in_long_format.csv", night_mode=self.parent().po.all['night_mode'])
        self.layout.addWidget(self.descriptor_label2, 3, 4)
        self.save_all_label2 = FixedText('Save', align="c", tip="Will add a column containing that descriptor in descriptors_in_long_format.csv\nEach row will contain the value of that descriptor at one time for one arena", night_mode=self.parent().po.all['night_mode'])
        self.layout.addWidget(self.save_all_label2, 3, 5)
        # self.save_mean_label2 = FixedText('Mean', align="c", tip="Will use descriptors_in_long_format.csv to store the average of that descriptor in statistics.csv")
        # self.layout.addWidget(self.save_mean_label2, 3, 8)
        # self.save_reg_label2 = FixedText('LinReg', align="c", tip="Will use descriptors_in_long_format.csv to make a linear regression\nof that descriptor when it varies the most\nand store this time interval,\nregression intercept and slope in statistics.csv")
        # self.layout.addWidget(self.save_reg_label2, 3, 9)

        self.descriptor_widgets_list = []

        # Create the table of the main output the user can select
        self.create_check_boxes_table()

        # Create the last row layout that will contain a few more output and the ok button.
        self.last_row_layout = QtWidgets.QHBoxLayout()
        self.last_row_widget = QtWidgets.QWidget()
        # self.iso_digi_label = FixedText('Try to detect growth transition', 14, tip="Detect if growth occurs in an isotropic manner\ni.e. in all directions simultaneously\nAnd if and when this isotropic growth breaks into a digitated one\ni.e. in some directions using pseudopods or pseudopods-like", night_mode=self.parent().po.all['night_mode'])
        # self.iso_digi = Checkbox(self.parent().po.vars['iso_digi_analysis'])
        # self.last_row_layout.addWidget(self.iso_digi_label)
        # self.last_row_layout.addWidget(self.iso_digi)

        # self.oscilacyto_label = FixedText('Proceed oscillation analysis', 14, tip="If checked, detects oscillating clusters within cell(s) and compute their\nperiod, phase, size and minimal distance to the cell(s) edge", night_mode=self.parent().po.all['night_mode'])
        # self.oscilacyto = Checkbox(self.parent().po.vars['oscilacyto_analysis'])
        # self.oscilacyto.stateChanged.connect(self.do_oscilacyto)
        # self.last_row_layout.addWidget(self.oscilacyto_label)
        # self.last_row_layout.addWidget(self.oscilacyto)

        self.binary_mask_label = FixedText('Save binary mask coordinates', 14, tip="If checked, saves the binary mask coordinates used to compute the selected descriptors.\nMost of them only require the cell presence mask.\nThe oscillatory and network analyses require additional masks (also saved if checked)\nWarning: these masks may take a lot of hard drive space.", night_mode=self.parent().po.all['night_mode'])

        try:
            self.parent().po.vars['save_binary_masks']
        except NameError:
            self.parent().po.vars['save_binary_masks'] = False
        self.binary_mask = Checkbox(self.parent().po.vars['save_binary_masks'])
        self.binary_mask.stateChanged.connect(self.binary_mask_saving)
        self.last_row_layout.addWidget(self.binary_mask_label)
        self.last_row_layout.addWidget(self.binary_mask)

        self.ok = PButton('Ok', night_mode=self.parent().po.all['night_mode'])
        self.ok.clicked.connect(self.ok_is_clicked)
        self.last_row_layout.addItem(self.horizontal_space)
        self.last_row_layout.addWidget(self.ok)

        self.layout.addItem(horzspaceItem, 14, 0, 1, 9)
        vertspaceItem = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        self.layout.addItem(vertspaceItem, 0, 10, 15, 1)
        self.layout.addItem(vertspaceItem, 0, 0, 15, 1)

        self.last_row_widget.setLayout(self.last_row_layout)
        self.layout.addWidget(self.last_row_widget, 15, 1, 1, 9)
        self.setLayout(self.layout)

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
            self.layout.addWidget(self.descriptor_widgets_list[label_index], row, col)
            self.descriptor_widgets_list.append(Checkbox(self.parent().po.all['descriptors'][name]))
            cb_index = label_index + 1

            # To remove:
            if name == 'fractal_analysis':# or name == 'network_detection':
                self.descriptor_widgets_list[label_index].setVisible(False)
                self.descriptor_widgets_list[cb_index].setVisible(False)

            self.layout.addWidget(self.descriptor_widgets_list[cb_index], row, col + 1)
        #if not self.parent().po.vars['oscilacyto_analysis']:
            #self.descriptor_widgets_list[cb_index].setVisible(False)

    def binary_mask_saving(self):
        self.parent().po.vars['save_binary_masks'] = self.binary_mask.isChecked()

    # def do_oscilacyto(self):
    #     do_it = self.oscilacyto.isChecked()
    #     self.parent().po.vars['oscilacyto_analysis'] = do_it
    #     self.parent().po.vars['descriptors']['cluster_number'] = do_it
    #     self.parent().po.vars['descriptors']['mean_cluster_area'] = do_it

    def ok_is_clicked(self):
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
            if name == 'network_detection':
                self.parent().po.vars['network_detection'] = checked_status

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
