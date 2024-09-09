#!/usr/bin/env python3
"""This module contains all modified/simplified widgets from PySide6
It is made to be easier to use and to be consistant in terms of colors and sizes."""
from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QImage, QPixmap, QFont, QPen, QFontMetrics, QPainter, QPainterPath, QColor, QDoubleValidator
from numpy import min, max, all, any
from cv2 import cvtColor, COLOR_BGR2RGB, resize

# colorblind-friendly : rgb(42, 251, 97) , rgb(126, 85, 197)
buttonfont = QFont("Segoe UI Semibold", 17, QFont.Bold)
# titlesize = 40

titlefont = f"Baskerville Old Face" #"40pt Baskerville Old Face"
textsize = 15
textfont = "Century Gothic"

backgroundcolor = "rgb(255,255,255)"
textColor = "rgb(0, 0, 0)"
bordercolor = "rgb(255,255,255)"
selectioncolor = "rgb(1, 122, 94)"
selectionbackgroundcolor = "rgb(240,240,240)"
selectionbordercolor = "rgb(255,0,0)"
buttoncolor = "rgb(255,255,255)"
buttonclickedcolor = "rgb(100,100,120)"
buttonborder = "2px solid rgb(0, 0, 0)"
buttonangles = "13px"
rollingborder = "1px solid rgb(127, 127, 127)"
rollingangles = "4px"

night_background_color = "rgb(50,50,65)"
night_text_Color = "rgb(255, 255, 255)"
night_border_color = "rgb(50,50,65)"
night_selection_color = "rgb(1, 152, 117)"
night_selection_background_color = "rgb(50,50,65)"
night_selection_border_color = "rgb(255,0,0)"
night_button_color = "rgb(50,50,65)"
night_button_clicked_color = "rgb(100,100,120)"
night_button_border = "2px solid rgb(255, 255, 255)"


class WindowType(QtWidgets.QWidget):
    resized = QtCore.Signal()
    def __init__(self, parent, night_mode=False):
        super().__init__()

        self.setVisible(False)
        self.setParent(parent)
        self.frame = QtWidgets.QFrame(self)
        self.frame.setGeometry(QtCore.QRect(0, 0, self.parent().screen_width, self.parent().screen_height))
        self.display_image = None
        # self.setFont(QFont(textfont, textsize, QFont.Medium))
        # self.setStyleSheet("background-color: %s; color: %s; font: %s; border-color: %s; selection-color: %s; selection-background-color: %s" % (backgroundcolor, textColor, f"{textsize}pt {textfont};", bordercolor, selectioncolor, selectionbackgroundcolor))
        # self.setStyleSheet("background-color: %s; color: %s; border-color: %s; selection-color: %s; selection-background-color: %s" % (backgroundcolor, textColor, bordercolor, selectioncolor, selectionbackgroundcolor))

        self.setFont(QFont(textfont, textsize, QFont.Medium))
        self.night_mode_switch(night_mode)
        # self.setStyleSheet("background-color: rgb(50,50,65);"
        #                    "color: rgb(255, 255, 255);\n" # 213, 251, 255
        #                    "font: 15pt \"Calibri\";\n"
        #                    "border-color: rgb(50,50,65);\n"
        #                    # "border-color: rgb(0, 150, 75);\n"
        #                    "selection-color: rgb(1, 152, 117);\n" # 0, 150, 75  0, 132, 66  0, 120, 215
        #                    "selection-background-color:rgb(50,50,65);\n"  # jsp
        #                    )
                           # "selection-background-color:rgb(60, 60, 60);\n"  # jsp
                           # "alternate-background-color: rgb(54, 54, 54);\n"  # jsp
                           # "QToolTip { color:  rgb(1, 152, 117); background-color: rgb(64,64,64); border: 0px; };\n"
                           # "")
        # self.titles_font = "font: 24pt"
        self.horizontal_space = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding,
                                              QtWidgets.QSizePolicy.Maximum)
        self.vertical_space = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum,
                                              QtWidgets.QSizePolicy.MinimumExpanding)
        # self.resized.connect(self.center_window)

    def resizeEvent(self, event):
        '''
        # Use this signal to detect a resize event and call center window function
        :param event:
        :return:
        '''
        self.resized.emit()
        if self.display_image is not None:
            win_width, win_height = self.size().width(), self.size().height()
            self.display_image.max_width = win_width - self.parent().image_window_width_diff
            self.display_image.max_height = win_height - self.parent().image_window_height_diff
            if self.display_image.max_width * self.display_image.height_width_ratio < self.display_image.max_height:
                self.display_image.scaled_shape = [round(self.display_image.max_width * self.display_image.height_width_ratio), self.display_image.max_width]
            else:
                self.display_image.scaled_shape = [self.display_image.max_height, round(self.display_image.max_height / self.display_image.height_width_ratio)]

            self.display_image.setMaximumHeight(self.display_image.scaled_shape[0])
            self.display_image.setMaximumWidth(self.display_image.scaled_shape[1])
        return super(WindowType, self).resizeEvent(event)

    def center_window(self):
        self.parent().center()

    def night_mode_switch(self, night_mode):
        if night_mode:
            self.setStyleSheet(
                "background-color: %s; color: %s; font: %s; border-color: %s; selection-color: %s; selection-background-color: %s" % (
                night_background_color, night_text_Color, f"{textsize}pt {textfont};", night_border_color,
                night_selection_color, night_selection_background_color))
        else:
            self.setStyleSheet(
                "background-color: %s; color: %s; font: %s; border-color: %s; selection-color: %s; selection-background-color: %s" % (
                backgroundcolor, textColor, f"{textsize}pt {textfont};", bordercolor, selectioncolor,
                selectionbackgroundcolor))


class FullScreenImage(QtWidgets.QLabel):
    def __init__(self, image, screen_height, screen_width):
        super().__init__()
        self.true_shape = image.shape
        self.max_height = screen_height
        self.max_width = screen_width
        # self.im_size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.im_size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                                    QtWidgets.QSizePolicy.MinimumExpanding)
        # self.im_size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(self.im_size_policy)

        self.height_width_ratio = image.shape[0] / image.shape[1]
        image = cvtColor(image, COLOR_BGR2RGB)
        image = QImage(image.data, image.shape[1], image.shape[0], 3 * image.shape[1], QImage.Format_RGB888)
        image = QPixmap(image)
        self.setScaledContents(True)
        if self.max_width * self.height_width_ratio < self.max_height:
            self.scaled_shape = [round(self.max_width * self.height_width_ratio), self.max_width]
        else:
            self.scaled_shape = [self.max_height, round(self.max_height / self.height_width_ratio)]
        # self.setFixedHeight(self.scaled_shape[0])
        # self.setFixedWidth(self.scaled_shape[1])
        # self.setMaximumHeight(self.scaled_shape[0])
        # self.setMaximumWidth(self.scaled_shape[1])
        self.setMinimumSize(self.scaled_shape[1], self.scaled_shape[0])
        self.setPixmap(QPixmap(image))
        self.adjustSize()


class InsertImage(QtWidgets.QLabel):
    def __init__(self, image, max_height, max_width):
        super().__init__()
        self.true_shape = image.shape
        self.max_height = max_height
        self.max_width = max_width
        # self.im_size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.im_size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                                    QtWidgets.QSizePolicy.MinimumExpanding)
        # self.im_size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(self.im_size_policy)

        self.height_width_ratio = image.shape[0] / image.shape[1]
        image = cvtColor(image, COLOR_BGR2RGB)
        image = QImage(image.data, image.shape[1], image.shape[0], 3 * image.shape[1], QImage.Format_RGB888)
        image = QPixmap(image)
        self.setScaledContents(True)
        if self.max_width * self.height_width_ratio < self.max_height:
            self.scaled_shape = [round(self.max_width * self.height_width_ratio), self.max_width]
        else:
            self.scaled_shape = [self.max_height, round(self.max_height / self.height_width_ratio)]
        # self.setFixedHeight(self.scaled_shape[0])
        # self.setFixedWidth(self.scaled_shape[1])
        self.setMaximumHeight(self.scaled_shape[0])
        self.setMaximumWidth(self.scaled_shape[1])
        # self.resize(self.scaled_shape[1], self.scaled_shape[0])
        # self.setMinimumSize(self.scaled_shape[0], self.scaled_shape[1])
        self.setPixmap(QPixmap(image))
        self.adjustSize()


    def update_image(self, image, text=None, color=255):
        self.true_shape = image.shape
        self.height_width_ratio = image.shape[0] / image.shape[1]
        image = cvtColor(image, COLOR_BGR2RGB)
        image = QImage(image.data, image.shape[1], image.shape[0], 3 * image.shape[1], QImage.Format_RGB888)
        image = QPixmap(image)
        self.setScaledContents(True)
        if self.max_width * self.height_width_ratio < self.max_height:
            self.scaled_shape = [int(round(self.max_width * self.height_width_ratio)), self.max_width]
        else:
            self.scaled_shape = [self.max_height, int(round(self.max_height / self.height_width_ratio))]
        # self.setFixedHeight(self.scaled_shape[0])
        # self.setFixedWidth(self.scaled_shape[1])
        self.setMaximumHeight(self.scaled_shape[0])
        self.setMaximumWidth(self.scaled_shape[1])
        # self.resize(self.scaled_shape[1], self.scaled_shape[0])

        # if text is not None:
        #     pass
            # pos = QtCore.QPoint(50, 50)
            # painter = QPainter(self)
            # painter.drawText(pos, text)
            # painter.setPen(QColor(color, color, color))
        self.setPixmap(QPixmap(image))

    def update_image_scaling_factors(self):
        self.scaling_factors = (self.true_shape[0] / self.scaled_shape[0]), (self.true_shape[1] / self.scaled_shape[1])


class PButton(QtWidgets.QPushButton):
    def __init__(self, text, fade=True, night_mode=False):
        """

        self.setStyleSheet("background-color: rgb(107, 145, 202);\n"
                                "border-color: rgb(255, 255, 255);\n"
                                "color: rgb(0, 0, 0);\n"
                                "font: 17pt \"Britannic Bold\";")
        :param text:
        """
        super().__init__()
        self.setText(text)
        self.night_mode_switch(night_mode)
        self.setFont(buttonfont)
        # self.setStyleSheet("background-color: rgb(50,50,65);\n" #50,50,65 150, 150, 150 153, 204, 205    122, 0, 61   30, 144, 220
        #                         "border-color: rgb(255, 255, 255);\n"
        #                         "color: rgb(255, 255, 255);\n"
        #                         "font: 17pt \"Segoe UI Semibold\";\n"
        #                         "border : 1px solid rgb(255, 255, 255);"
        #                         "border-radius : 12px;\n"
        #                    )
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFixedWidth(len(text)*15 + 25)
        # if fade:
        #     # self.mouseMoveEvent.connect(self.fade)
        #     self.clicked.connect(self.fade)

    def night_mode_switch(self, night_mode):
        if night_mode:
            self.style = {"buttoncolor": night_button_color, "buttontextColor": night_text_Color, "buttonborder": night_button_border,
                          "buttonangles": buttonangles}
        else:
            self.style = {"buttoncolor": buttoncolor, "buttontextColor": textColor, "buttonborder": buttonborder,
                          "buttonangles": buttonangles}
        self.update_style()

    def event_filter(self, event):
        if event.type() == QtCore.QEvent.MouseMove:
            if event.buttons() == QtCore.Qt.NoButton:
                self.fade()
            else:
                self.unfade()

    def update_style(self):
        self.setStyleSheet(
            "background-color: %s; color: %s; border: %s; border-radius: %s" % tuple(self.style.values()))

    def color(self, color):
        self.style["buttoncolor"] = color
        self.update_style()

    def textcolor(self, textcolor):
        self.style["buttontextColor"] = textcolor
        self.update_style()

    def border(self, border):
        self.style["buttonborder"] = border
        self.update_style()

    def angles(self, angles):
        self.style["buttonangles"] = angles
        self.update_style()

    def fade(self):
        self.setWindowOpacity(0.5)
        self.setStyleSheet("background-color: %s; color: %s; border: %s; border-radius: %s" % (buttonclickedcolor, textColor, buttonborder, buttonangles))
        QtCore.QTimer.singleShot(300, self.unfade)

    def unfade(self):
        self.setWindowOpacity(1)
        self.setStyleSheet("background-color: %s; color: %s; border: %s; border-radius: %s" % (buttoncolor, textColor, buttonborder, buttonangles))

import numpy as np
class Spinbox(QtWidgets.QAbstractSpinBox):
    valueChanged = QtCore.Signal(float)

    class StepType:
        DefaultStepType = 0
        AdaptiveDecimalStepType = 1
    def __init__(self, min=0, max=100000, val=0, decimals=None, night_mode=False):
        super().__init__()
        self.setFixedHeight(30)
        self._minimum = min
        self._maximum = max
        self._value = val
        if decimals is None:
            decimals = 0
        self._decimals = decimals
        if decimals == 0:
            self._singleStep = 1
        else:
            self._singleStep = 0.1
        self._stepType = self.StepType.DefaultStepType

        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.lineEdit().setAlignment(QtCore.Qt.AlignRight)
        self._updateDisplayValue()
        self._updateValidator()

        self.lineEdit().textChanged.connect(self._handleTextChanged)

        self.setMinimumWidth(100)

        self.setMinimum(min)
        self.setMaximum(max)
        self.setValue(val)
        self.decimals = decimals
        if decimals is not None:
            self.setDecimals(decimals)
            # self.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        else:
            self.setDecimals(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.setFont(QFont(textfont, textsize, QFont.Medium))
        self.night_mode_switch(night_mode)
        # self.setStyleSheet("""
        #     QSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 16px; }
        #     QSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 16px; }
        #     QSpinBox::up-arrow, QSpinBox::down-arrow { width: 10px; height: 10px; }
        # """)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        button_width = 20
        button_height = self.height() // 2
        up_rect = QtCore.QRect(self.width() - button_width, 0, button_width, button_height)
        down_rect = QtCore.QRect(self.width() - button_width, button_height, button_width, button_height)

        # Draw button backgrounds
        painter.fillRect(up_rect, QColor(240, 240, 240))
        painter.fillRect(down_rect, QColor(240, 240, 240))

        # Draw button borders
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawLine(self.width() - button_width, 0, self.width() - button_width, self.height())
        painter.drawLine(self.width() - button_width, button_height, self.width(), button_height)

        # Draw arrows
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QColor(100, 100, 100))

        # Up arrow
        up_arrow = QPainterPath()
        up_arrow.moveTo(up_rect.center().x() - 4, up_rect.center().y() + 2)
        up_arrow.lineTo(up_rect.center().x() + 4, up_rect.center().y() + 2)
        up_arrow.lineTo(up_rect.center().x(), up_rect.center().y() - 2)
        painter.drawPath(up_arrow)

        # Down arrow
        down_arrow = QPainterPath()
        down_arrow.moveTo(down_rect.center().x() - 4, down_rect.center().y() - 2)
        down_arrow.lineTo(down_rect.center().x() + 4, down_rect.center().y() - 2)
        down_arrow.lineTo(down_rect.center().x(), down_rect.center().y() + 2)
        painter.drawPath(down_arrow)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        button_width = 20
        text_rect = self.rect().adjusted(0, 0, -button_width, 0)
        self.lineEdit().setGeometry(text_rect)

    def mousePressEvent(self, event):
        button_width = 20
        button_height = self.height() // 2
        if event.x() > self.width() - button_width:
            if event.y() < button_height:
                self.stepUp()
            else:
                self.stepDown()
        else:
            super().mousePressEvent(event)

    def sizeHint(self):
        fm = QFontMetrics(self.font())
        max_str = f"{self._maximum:.{self._decimals}f}"
        w = fm.horizontalAdvance(max_str) + 30  # Add some padding for buttons
        h = super().sizeHint().height()
        return QtCore.QSize(np.max((w, 100)), h)  # Ensure a minimum width of 100 pixels

    def setMaximum(self, maximum):
        self._maximum = float(maximum)
        self._updateValidator()
        self.setValue(np.min((self._value, self._maximum)))
        self.updateGeometry()

    def setDecimals(self, decimals):
        try:
            self._decimals = np.max((0, int(decimals)))
        except (ValueError, TypeError):
            print(f"Invalid decimals value: {decimals}. Using 2 decimals as default.")
            self._decimals = 2
        self._updateValidator()
        self._updateDisplayValue()
        self.updateGeometry()

    def _updateDisplayValue(self):
        if self._decimals == 0:
            self.lineEdit().setText(f"{int(round(self._value))}")
        else:
            self.lineEdit().setText(f"{self._value:.{self._decimals}f}")

    def _updateValidator(self):
        self.lineEdit().setValidator(QDoubleValidator(self._minimum, self._maximum, self._decimals, self))

    def setValue(self, value):
        clamped_value = np.clip(value, self._minimum, self._maximum)
        if np.abs(clamped_value - self._value) >= 1e-8:  # Compare floats with small tolerance
            self._value = clamped_value
            self._updateDisplayValue()
            self.valueChanged.emit(self._value)

    def value(self):
        return self._value

    def setMinimum(self, minimum):
        self._minimum = float(minimum)
        self._updateValidator()
        self.setValue(np.max((self._value, self._minimum)))

    def setSingleStep(self, step):
        self._singleStep = float(step)

    def setStepType(self, step_type):
        if step_type not in (self.StepType.DefaultStepType, self.StepType.AdaptiveDecimalStepType):
            raise ValueError("Invalid step type")
        self._stepType = step_type

    def stepBy(self, steps):
        if self._stepType == self.StepType.DefaultStepType:
            new_value = self._value + steps * self._singleStep
        else:  # AdaptiveDecimalStepType
            abs_value = np.abs(self._value)
            if abs_value == 0:
                new_value = steps * self._singleStep
            else:
                decade = np.floor(np.log10(abs_value))
                step_size = np.power(10, decade) / 100.0
                new_value = self._value + steps * step_size

        if self._decimals == 0:
            new_value = round(new_value)

        self.setValue(new_value)

    def _handleTextChanged(self, text):
        if text:
            try:
                value = float(text)
                self.setValue(value)
            except ValueError:
                pass

    def stepEnabled(self):
        return self.StepUpEnabled | self.StepDownEnabled

    def event(self, event):
        if event.type() in [QtCore.QEvent.HoverEnter, QtCore.QEvent.HoverLeave, QtCore.QEvent.HoverMove]:
            self.update()
        return super().event(event)
    def night_mode_switch(self, night_mode):
        if night_mode:
            self.setStyleSheet(
                "background-color: %s; color: %s; border-color: %s; border: %s; border-radius: %s" % (
                    night_background_color, night_text_Color, night_border_color, rollingborder, rollingangles))
        else:
            self.setStyleSheet(
                "background-color: %s; color: %s; border-color: %s; border: %s; border-radius: %s" % (
                    backgroundcolor, textColor, bordercolor, rollingborder, rollingangles))

        # self.setStyleSheet("background-color: rgb(50,50,65);\n"
        #                    "font: 16pt \"Calibri\";\n"
        #                    "border-color: rgb(50,50,65);\n"
        #                    # "border-color: rgb(255, 255, 255);\n"
        #                    "color: rgb(213, 251, 255);\n")


class Combobox(QtWidgets.QComboBox):
    def __init__(self, items_list, current_idx=None, night_mode=False):
        super().__init__()
        for item in items_list:
            self.addItem(item)
        if current_idx is not None:
            self.setCurrentIndex(current_idx)
        self.setFixedHeight(30)
        #self.adjustSize()
        #max([len(item) for item in items_list])
        # width = self.minimumSizeHint().width()
        # self.setMinimumWidth(width+30)
        # self.setFixedWidth(self.minimumSizeHint().width() + 5)

        # self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.night_mode_switch(night_mode)

        # self.setStyleSheet("QComboBox::drop - down:!editable{background: transparent; border: none;};"
        #                    "QComboBox {border: '1px solid rgb(127, 127, 127)'; border-radius: 4px}")

    def night_mode_switch(self, night_mode):
        if night_mode:
            self.setStyleSheet("border-color: %s; border: %s; border-radius: %s" % (
                night_background_color, rollingborder, rollingangles))
        else:
            self.setStyleSheet("border-color: %s; border: %s; border-radius: %s" % (
                backgroundcolor, rollingborder, rollingangles))


class Checkbox(QtWidgets.QCheckBox):
    def __init__(self, set_checked, night_mode=None):
        super().__init__()
        self.setChecked(set_checked)
        self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.setMinimumWidth(75)
        # self.setStyleSheet("padding:5px")
        self.setStyleSheet("margin-left:50%; margin-right:50%;")


class Title(QtWidgets.QLabel):
    def __init__(self, text, tip=None):
        super().__init__()
        self.setText(text)
        self.setAlignment(QtCore.Qt.AlignHCenter)
        self.setStyleSheet("font: 24pt \"Segoe UI Semibold\";\n"
                           #"border: 1px solid gray;\n"
                           "margin-bottom: 30%;\n")
        self.setMargin(2)
        if tip is not None:
            self.setToolTip(tip)
            self.setStyleSheet("""QToolTip { 
                                          background-color: rgb(60, 60, 60); 
                                          color: rgb(1, 152, 117); 
                                          border: black solid 1px
                                          }""")


class EditText(QtWidgets.QLineEdit):
    def __init__(self, text, police=None, align='l', tip=None, night_mode=False):
        super().__init__()
        self.setMaximumHeight(36)
        self.setText(str(text))
        self.night_mode_switch(night_mode)
        # self.setAttribute(QtCore.Qt.WA_MacShowFocusRect, 0)

    def night_mode_switch(self, night_mode):
        # self.setStyleSheet("border: %s" % bordercolor)
        # self.setStyleSheet("border: %s" % night_border_color)
        if night_mode:
            self.setStyleSheet(
                "background-color: %s; color: %s; font: %s; border-bottom: %s; border-top: %s" % (
                    night_background_color, night_text_Color, f"{textsize}pt {textfont};", f"1px solid grey", f"1px solid grey"))
        else:
            self.setStyleSheet(
                "background-color: %s; color: %s; font: %s; border-bottom: %s; border-top: %s" % (
                    backgroundcolor, textColor, f"{textsize}pt {textfont};", f"1px solid grey", f"1px solid grey"))

            # "QLineEdit"
            #           "{"
            #           "border : 5px solid black;"
            #           "}"
            #           "QLineEdit::hover"
            #           "{"
            #           "border-color : red green blue yellow"
            #           "}")

        # if night_mode:
        #     self.setStyleSheet(
        #         "background-color: %s; color: %s; font: %s; border-color: %s; selection-color: %s; alternate-border-color: %s; selection-background-color: %s" % (
        #             night_background_color, night_text_Color, f"{textsize}pt {textfont};", night_border_color,
        #             night_selection_color, night_selection_border_color, night_selection_background_color))
        # else:
        #     self.setStyleSheet(
        #         "background-color: %s; color: %s; font: %s; border-color: %s; selection-color: %s; alternate-border-color: %s;  selection-background-color: %s" % (
        #             backgroundcolor, textColor, f"{textsize}pt {textfont};", bordercolor, selectioncolor,
        #             selectionbordercolor, selectionbackgroundcolor))


class FixedText(QtWidgets.QLabel):
    def __init__(self, text, police=None, align='l', tip=None, night_mode=False):
        super().__init__()
        self.setText(text)
        if align == 'l':
            self.setAlignment(QtCore.Qt.AlignLeft)
        elif align == 'r':
            self.setAlignment(QtCore.Qt.AlignRight)
        else:
            self.setAlignment(QtCore.Qt.AlignCenter)
        if police is not None:
            if police > 23:
                # self.setStyleSheet("font: %s; margin-bottom: %s" % (titlefont, "30%;"))

                # self.night_mode_switch(night_mode, titlefont)
                self.night_mode_switch(night_mode, f"{police}pt {titlefont}")

                # self.setFont(titlefont)
                # self.setStyleSheet("margin-bottom: 30%;\n")
                self.setMargin(2)
            # else:
            #     self.night_mode_switch(night_mode, f"{police}pt {textfont}")
                # self.night_mode_switch(night_mode, QFont(textfont, police + 2, QFont.Medium))
                # self.setFont(QFont(textfont, police + 2, QFont.Medium))

        else:
            self.setFont(QFont(textfont, textsize + 2, QFont.Medium))

        if tip is not None:
            self.setToolTip(tip)
            if night_mode:
                self.setStyleSheet("""QToolTip {background-color: rgb(70,70,85);color: rgb(1, 152, 117);
                                                            border: white solid 1px}""")
            else:
                self.setStyleSheet("""QToolTip {background-color: rgb(240,240,240);
                                                                                  color: rgb(1, 122, 94);
                                                                                  border: black solid 1px
                                                                                  }""")

            # self.setStyleSheet("""QToolTip {
            #                               background-color: rgb(150, 150, 150);
            #                               color: rgb(1, 1, 1);
            #                               border: black solid 1px
            #                               }""")

    def night_mode_switch(self, night_mode, font):
        if night_mode:
            self.setStyleSheet("font: %s; color: %s; margin-bottom: %s" % (font, night_text_Color, "30%"))
        else:
            self.setStyleSheet("font: %s; color: %s; margin-bottom: %s" % (font, textColor, "30%"))


class LineWidget(QtWidgets.QWidget):
    def __init__(self, ori="h", size=None, night_mode=False):
        super().__init__()
        self.layout = QtWidgets.QHBoxLayout()
        self.line = QtWidgets.QFrame()
        if ori == "h":
            self.line.setFrameShape(QtWidgets.QFrame.HLine)
            # self.setFrameShadow(QtWidgets.QFrame.Sunken)
        else:
            self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.night_mode_switch(night_mode)
        if size is None:
            size = [1, 4]
        self.line.setFixedHeight(size[0])
        self.line.setFixedWidth(size[1])
        horizontal_space = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding,
                                                      QtWidgets.QSizePolicy.Maximum)
        self.layout.addItem(horizontal_space)
        self.layout.addWidget(self.line)
        self.layout.addItem(horizontal_space)
        self.setLayout(self.layout)

    def night_mode_switch(self, night_mode):
        if night_mode:
            self.line.setStyleSheet("QFrame { background-color: rgb(255, 255, 255) }")
        else:
            self.line.setStyleSheet("QFrame { background-color: rgb(0, 0, 0) }")

