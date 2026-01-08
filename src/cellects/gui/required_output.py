#!/usr/bin/env python3
"""Widget for configuring Cellects analysis variables and descriptors.

This GUI module provides checkboxes to select raw data outputs (presence/absence coordinates,
contours, tubular networks) and dynamic descriptors calculated per time frame. User selections are saved via parent
object's background thread when 'Ok' is clicked. Includes categorized sections with grouped options for clarity of
specimen tracking parameters.

Main Components
---------------
RequiredOutput : QWidget for configuring required output variables

Notes
-----
Saves user selections using parent object's SaveAllVars QThread.
"""

import numpy as np
from PySide6 import QtWidgets, QtCore
import logging
from cellects.gui.custom_widgets import (
    WindowType, PButton, Checkbox, FixedText)
from cellects.image_analysis.shape_descriptors import descriptors_names_to_display, descriptors_categories
from cellects.gui.ui_strings import RO

class RequiredOutput(WindowType):
    def __init__(self, parent, night_mode):
        """
        Initialize the RequiredOutput window with a parent widget and night mode setting.

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
        >>> from cellects.gui.required_output import RequiredOutput
        >>> import sys
        >>> app = QtWidgets.QApplication([])
        >>> parent = CellectsMainWidget()
        >>> session = RequiredOutput(parent, False)
        >>> session.true_init()
        >>> parent.insertWidget(0, session)
        >>> parent.show()
        >>> sys.exit(app.exec())
        """
        super().__init__(parent, night_mode)
        self.setParent(parent)
        # Create the main Title
        self.true_init()

    def true_init(self):
        """
        Initialize the RequiredOutput window with various checkboxes and buttons.

        This method sets up the entire UI layout for the RequiredOutput window,
        including a title, checkboxes for saving different types of coordinates and
        descriptors, and 'Cancel' and 'Ok' buttons.

        Notes
        -----
        This method assumes that the parent widget has a 'po' attribute with specific settings and variables.
        """
        logging.info("Initialize RequiredOutput window")
        self.title = FixedText('Required Output', police=30, night_mode=self.parent().po.all['night_mode'])
        self.title.setAlignment(QtCore.Qt.AlignHCenter)
        # Create the main layout
        self.vlayout = QtWidgets.QVBoxLayout()
        self.vlayout.addWidget(self.title) #
        self.vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding)) #

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
        self.save_coord_specimen_label = FixedText(RO["coord_specimen"]["label"], tip=RO["coord_specimen"]["tips"],
                                           night_mode=self.parent().po.all['night_mode'])
        self.save_graph = Checkbox(self.parent().po.vars['save_graph'])
        self.save_graph_label = FixedText(RO["Graph"]["label"], tip=RO["Graph"]["tips"],
                                           night_mode=self.parent().po.all['night_mode'])
        self.save_coord_thickening_slimming = Checkbox(self.parent().po.vars['save_coord_thickening_slimming'])
        self.save_coord_thickening_slimming_label = FixedText(RO["coord_oscillating"]["label"],
                                                              tip=RO["coord_oscillating"]["tips"],
                                           night_mode=self.parent().po.all['night_mode'])
        self.save_coord_network = Checkbox(self.parent().po.vars['save_coord_network'])
        self.save_coord_network_label = FixedText(RO["coord_network"]["label"], tip=RO["coord_network"]["tips"],
                                           night_mode=self.parent().po.all['night_mode'])

        # I/D/ Arrange widgets in the box
        self.save_presence_coordinates_layout.addWidget(self.save_coord_specimen_label, 0, 0)
        self.save_presence_coordinates_layout.addWidget(self.save_coord_specimen, 0, 1)
        self.save_presence_coordinates_layout.addWidget(self.save_coord_thickening_slimming_label, 1, 0)
        self.save_presence_coordinates_layout.addWidget(self.save_coord_thickening_slimming, 1, 1)
        self.save_presence_coordinates_layout.addWidget(self.save_coord_network_label, 0, 2)
        self.save_presence_coordinates_layout.addWidget(self.save_coord_network, 0, 3)
        self.save_presence_coordinates_layout.addWidget(self.save_graph_label, 1, 2)
        self.save_presence_coordinates_layout.addWidget(self.save_graph, 1, 3)

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
        self.last_row_layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.last_row_layout.addWidget(self.cancel)
        self.last_row_layout.addWidget(self.ok)

        self.vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))
        self.last_row_widget.setLayout(self.last_row_layout)
        self.vlayout.addWidget(self.last_row_widget)

        self.hlayout = QtWidgets.QHBoxLayout()
        self.vwidget = QtWidgets.QWidget()
        self.vwidget.setLayout(self.vlayout)
        self.hlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.hlayout.addWidget(self.vwidget)
        self.hlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.setLayout(self.hlayout)

    def create_check_boxes_table(self):
        """
        Create and populate a table of checkboxes for descriptors in the parent object.

        This function initializes or updates the descriptors and iterates through
        them to create a table of checkboxes, arranging them in a grid layout. The
        arrangement depends on the total number of descriptors and their visibility.

        Notes
        -----
        The layout of checkboxes changes based on the number of descriptors.
        """
        if not np.array_equal(self.parent().po.all['descriptors'], list(descriptors_categories.keys())):
            self.parent().po.all['descriptors'] = descriptors_categories

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

            # if name == 'fractal_analysis' or name == 'oscilacyto_analysis':
            #     self.descriptor_widgets_list[label_index].setVisible(False)
            #     self.descriptor_widgets_list[cb_index].setVisible(False)

            self.save_descriptors_layout.addWidget(self.descriptor_widgets_list[cb_index], row, col + 1)

    def cancel_is_clicked(self):
        """
        Instead of saving the widgets values to the saved states, use the saved states to fill in the widgets.

        This function updates the state of several checkboxes based on saved variables
        and descriptors. It also changes the active widget to either the first or third
        widget depending on a condition.
        """
        self.save_coord_specimen.setChecked(self.parent().po.vars['save_coord_specimen'])
        self.save_graph.setChecked(self.parent().po.vars['save_graph'])
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
            else:
                self.descriptor_widgets_list[k].setChecked(self.parent().po.all['descriptors'][name])

        if self.parent().last_is_first:
            self.parent().change_widget(0) # FirstWidget
        else:
            self.parent().change_widget(3) # ThirdWidget

    def ok_is_clicked(self):
        """
        Updates the parent object's variables and descriptor states based on UI checkboxes.

        This method updates various variables and descriptor states in the parent
        object based on the current state of checkboxes in the UI. It also starts a
        thread to save all variables and updates the output list accordingly.

        Notes
        -----
        This method does not return any value. It updates the internal state of the
        parent object, which saves all user defined parameters.
        """
        self.parent().po.vars['save_coord_specimen'] = self.save_coord_specimen.isChecked()
        self.parent().po.vars['save_graph'] = self.save_graph.isChecked()
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
        if not self.parent().thread_dict['SaveAllVars'].isRunning():
            self.parent().thread_dict['SaveAllVars'].start()
        self.parent().po.update_output_list()
        if self.parent().last_is_first:
            self.parent().change_widget(0) # FirstWidget
        else:
            self.parent().change_widget(3) # ThirdWidget
