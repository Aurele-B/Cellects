#!/usr/bin/env python3
"""This module contains all modified/simplified widgets from PySide6
It is made to be easier to use and to be consistant in terms of colors and sizes."""
import os
from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import Qt, QImage, QPixmap, QFont,QFontMetrics, QPainter, QDoubleValidator
import numpy as np
from numpy.typing import NDArray
import cv2

"""
Get every available font:
from PySide6 import QtGui
fonts = QtGui.QFontDatabase()
font_names = fonts.families()
print(font_names)
"Baskerville" in font_names
"""

if os.name == 'nt':
    buttonfont = QFont("Century Gothic", 17, QFont.Bold)
    titlefont = f"Baskerville Old Face" #"40pt Baskerville Old Face"
    textfont = "Century Gothic"
    tabfont = "Baskerville Old Face" # 30pt Comic Sans MS
else:
    buttonfont = QFont("Times New Roman", 17, QFont.Bold)
    titlefont = "Baskerville"
    textfont = "Times New Roman"
    tabfont = "Baskerville"
textsize = 15

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

        self.thread_dict = {}
        self.setVisible(False)
        self.setParent(parent)

        self.setAttribute(Qt.WA_StyledBackground, True)

        self.frame = QtWidgets.QFrame(self)
        self.frame.setGeometry(QtCore.QRect(0, 0, self.parent().win_width, self.parent().win_height))
        self.frame.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        self.display_image = None
        self.setFont(QFont(textfont, textsize, QFont.Medium))
        self.night_mode = night_mode
        self.night_mode_switch(night_mode)

    def center_window(self):
        self.parent().center()

    def night_mode_switch(self, night_mode):
        if night_mode:
            self.setStyleSheet(
                "background-color: %s; color: %s; font: %s; selection-color: %s; selection-background-color: %s" % (
                night_background_color, night_text_Color, f"{textsize}pt {textfont};",
                night_selection_color, night_selection_background_color))
        else:
            self.setStyleSheet(
                "background-color: %s; color: %s; font: %s; selection-color: %s; selection-background-color: %s" % (
                backgroundcolor, textColor, f"{textsize}pt {textfont};", selectioncolor,
                selectionbackgroundcolor))

    def mouse_clicks(self, image_object, event):
        pass

    def mouse_moves(self, image_object, event):
        pass

    def mouse_releases(self, image_object, event):
        pass


class MainTabsType(WindowType):
    def __init__(self, parent, night_mode):
        super().__init__(parent, night_mode)
        self.thread_dict = {}
        self.Vlayout = QtWidgets.QVBoxLayout()
        self.Vlayout.setContentsMargins(9, 0, 9, 9)
        self.main_tabs_widget = QtWidgets.QWidget()
        self.main_tabs_layout = QtWidgets.QHBoxLayout()
        self.main_tabs_layout.setContentsMargins(0, 0, 0, 0)
        self.main_tabs_layout.setSpacing(0)
        self.data_tab = MainTabsWidget('Data localisation', night_mode=night_mode)
        self.image_tab = MainTabsWidget('Image analysis', night_mode=night_mode)
        self.video_tab = MainTabsWidget('Video tracking', night_mode=night_mode)
        self.main_tabs_layout.addWidget(self.data_tab)
        self.main_tabs_layout.addWidget(self.image_tab)
        self.main_tabs_layout.addWidget(self.video_tab)
        self.main_tabs_widget.setLayout(self.main_tabs_layout)
        self.Vlayout.addWidget(self.main_tabs_widget)
        self.Vlayout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding))


class InsertImage(QtWidgets.QLabel):
    def __init__(self, parent=None, track_mouse: bool=False):
        super().__init__()
        self.parent = parent
        self.image = None
        self.panning = False
        self.last_mouse = None
        self.track_mouse = track_mouse
        self.closed = False
        self.zoom = 1.0
        self.base_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.setMinimumWidth(200)
        self.setMinimumHeight(200)

        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._update_scaled)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )

    def update_image(self, image: NDArray[np.uint8]):
        dims = image.shape
        if not isinstance(image, np.uint8):
            image = image.astype(np.uint8)
        img_max_int = image.max()
        if img_max_int < 10:
            image *= 255 // img_max_int
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_format = QImage.Format_RGB888
        else:
            img_format = QImage.Format_Grayscale8
        self.image = QImage(image.data, dims[1], dims[0], image.strides[0], img_format)
        self.pixmap = QPixmap.fromImage(self.image)
        self._schedule_update()

    def update_screen_limits(self):
        screen = self.screen()
        if not screen:
            return

        geom = screen.availableGeometry()

        # Use a fraction of screen size for image cap
        self.max_size = (int(geom.width() * 0.8), int(geom.height() * 0.8))
        self._schedule_update()

    def _schedule_update(self):
        self.timer.start(20)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_update()

    def _update_scaled(self):
        if not self.image:
            return
        sx = self.width() / self.image.width()
        sy = self.height() / self.image.height()
        self.base_scale = min(sx, sy)
        self.update()

    def paintEvent(self, event):
        if not self.image:
            return

        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            True
        )

        scale = self.base_scale * self.zoom
        draw_w = self.image.width() * scale
        draw_h = self.image.height() * scale

        x = (self.width() - draw_w) / 2
        y = (self.height() - draw_h) / 2
        painter.translate(x + self.pan_x, y + self.pan_y)
        painter.scale(scale, scale)
        painter.drawPixmap(0, 0, self.pixmap)

    def showEvent(self, event):
        super().showEvent(event)
        self.update_screen_limits()

    def image_coordinates(self, pos):
        if self.image is None:
            return None

        scale = self.base_scale * self.zoom

        x0 = self.width() / 2 + self.pan_x - (self.image.width() * scale) / 2
        y0 = self.height() / 2 + self.pan_y - (self.image.height() * scale) / 2

        ix = (pos.x() - x0) / scale
        iy = (pos.y() - y0) / scale

        ix = int(np.clip(ix, 0, self.image.width() - 1))
        iy = int(np.clip(iy, 0, self.image.height() - 1))

        return iy, ix

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self.panning = True
            self.last_mouse = event.position()
        elif self.track_mouse:
            self.parent.mouse_clicks(self, event)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.position() - self.last_mouse
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self.last_mouse = event.position()
            self.update()
        elif self.track_mouse:
            self.parent.mouse_moves(self, event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self.panning = False
        elif self.track_mouse:
            self.parent.mouse_releases(self, event)

    def wheelEvent(self, event):
        old_zoom = self.zoom

        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = np.clip(self.zoom * factor, 0.05, 50.0)

        mouse = event.position()

        scale_old = self.base_scale * old_zoom
        scale_new = self.base_scale * self.zoom

        # convert mouse to image space BEFORE zoom
        ix = (mouse.x() - self.width() / 2 - self.pan_x) / scale_old
        iy = (mouse.y() - self.height() / 2 - self.pan_y) / scale_old

        # recompute pan so point stays fixed
        self.pan_x = mouse.x() - self.width() / 2 - ix * scale_new
        self.pan_y = mouse.y() - self.height() / 2 - iy * scale_new

        self.update()

    def closeEvent(self, event):
        self.closed = True
        super().closeEvent(event)


class PButton(QtWidgets.QPushButton):
    def __init__(self, text, fade=True, tip=None, night_mode=False):
        """

        self.setStyleSheet("background-color: rgb(107, 145, 202);\n"
                                "border-color: rgb(255, 255, 255);\n"
                                "color: rgb(0, 0, 0);\n"
                                "font: 17pt \"Britannic Bold\";")
        :param text:
        """
        super().__init__()
        self.setText(text)
        self.setToolTip(tip)
        self.night_mode_switch(night_mode)
        self.setFont(buttonfont)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFixedWidth(len(text)*15 + 25)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

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


class Spinbox(QtWidgets.QWidget):
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
        self._button_width = 16
        self._button_height = 12

        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._line_edit = QtWidgets.QLineEdit(self)
        self._line_edit.setAlignment(QtCore.Qt.AlignLeft)
        self._layout.addWidget(self._line_edit)

        button_layout = QtWidgets.QVBoxLayout()
        button_layout.setSpacing(0)
        button_layout.setContentsMargins(0, 0, 0, 0)

        self._up_button = QtWidgets.QPushButton("▲", self)
        self._up_button.setFixedSize(self._button_width, self._button_height)
        self._up_button.clicked.connect(self.stepUp)
        button_layout.addWidget(self._up_button)

        self._down_button = QtWidgets.QPushButton("▼", self)
        self._down_button.setFixedSize(self._button_width, self._button_height)
        self._down_button.clicked.connect(self.stepDown)
        button_layout.addWidget(self._down_button)

        self._layout.addLayout(button_layout)

        self._updateDisplayValue()
        self._updateValidator()

        self._line_edit.textChanged.connect(self._handleTextChanged)

        self.setMinimumWidth(120)

        self.setMinimum(min)
        self.setMaximum(max)
        self.setValue(val)
        self.decimals = decimals
        if decimals is not None:
            self.setDecimals(decimals)
        else:
            self.setDecimals(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.setFont(QFont(textfont, textsize, QFont.Medium))
        self.setStyleSheet("""
            QPushButton {
                font-size: 8px;
                padding: 0px;
                margin: 0px;
                border: 1px solid #c0c0c0;
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
        """)
        self.night_mode_switch(night_mode)

        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def sizeHint(self):
        fm = QFontMetrics(self.font())
        max_str = f"{self._maximum:.{self._decimals}f}"
        w = fm.horizontalAdvance(max_str) + self._button_width + 5  # Add some padding
        h = self._line_edit.sizeHint().height()
        return QtCore.QSize(np.max((w, 100)), h)  # Ensure a minimum width of 100 pixels

    def minimumSizeHint(self):
        return QtCore.QSize(self._button_width * 3, self._line_edit.minimumSizeHint().height())

    def _updateDisplayValue(self):
        if self._decimals == 0:
            self._line_edit.setText(f"{int(np.round(self._value))}")
        else:
            self._line_edit.setText(f"{self._value:.{self._decimals}f}")

    def _updateValidator(self):
        self._line_edit.setValidator(QDoubleValidator(self._minimum, self._maximum, self._decimals, self))

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

    def setMaximum(self, maximum):
        self._maximum = float(maximum)
        self._updateValidator()
        self.setValue(np.min((self._value, self._maximum)))

    def setSingleStep(self, step):
        self._singleStep = float(step)

    def setDecimals(self, decimals):
        self._decimals = int(decimals)
        self._updateValidator()
        self._updateDisplayValue()

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

        self.setValue(new_value)

    def stepUp(self):
        self.stepBy(1)

    def stepDown(self):
        self.stepBy(-1)

    def _handleTextChanged(self, text):
        if text:
            try:
                value = float(text)
                self.setValue(value)
            except ValueError:
                pass

    def night_mode_switch(self, night_mode):
        if night_mode:
            self.setStyleSheet(
                "background-color: %s; color: %s; border-color: %s; border: %s; border-radius: %s" % (
                    night_background_color, night_text_Color, night_border_color, rollingborder, rollingangles))
        else:
            self.setStyleSheet(
                "background-color: %s; color: %s; border-color: %s; border: %s; border-radius: %s" % (
                    backgroundcolor, textColor, bordercolor, rollingborder, rollingangles))


class Combobox(QtWidgets.QComboBox):
    def __init__(self, items_list, current_idx=None, night_mode=False):
        super().__init__()
        for item in items_list:
            self.addItem(item)
        if current_idx is not None:
            self.setCurrentIndex(current_idx)
        self.setFixedHeight(30)
        self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        self.night_mode_switch(night_mode)

        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
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
        self.setStyleSheet("QCheckBox::indicator {width: 12px;height: 12px;background-color: transparent;"
                            "border-radius: 5px;border-style: solid;border-width: 1px;"
                            "border-color: rgb(100,100,100);}"
                            "QCheckBox::indicator:checked {background-color: rgb(70,130,180);}"
                            "QCheckBox:checked, QCheckBox::indicator:checked {border-color: black black white white;}"
                            "QCheckBox:checked {background-color: transparent;}"
                            "QCheckBox:margin-left {50%}"
                            "QCheckBox:margin-right {50%}")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)


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
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def night_mode_switch(self, night_mode):
        if night_mode:
            self.setStyleSheet(
                "background-color: %s; color: %s; font: %s; border-bottom: %s; border-top: %s" % (
                    night_background_color, night_text_Color, f"{textsize}pt {textfont};", f"1px solid #adadad", f"1px dotted grey"))
        else:
            self.setStyleSheet(
                "background-color: %s; color: %s; font: %s; border-bottom: %s; border-top: %s" % (
                    backgroundcolor, textColor, f"{textsize}pt {textfont};", f"1px solid grey", f"1px dotted #adadad"))


class FixedText(QtWidgets.QLabel):
    def __init__(self, text, police=None, halign='l', valign="c", tip=None, night_mode=False):
        super().__init__()
        self.setText(text)
        if halign == 'l':
            self.setAlignment(QtCore.Qt.AlignLeft)
        elif halign == 'r':
            self.setAlignment(QtCore.Qt.AlignRight)
        else:
            self.setAlignment(QtCore.Qt.AlignCenter)
        if valign == 't':
            self.setAlignment(Qt.AlignTop)
        elif valign == 'b':
            self.setAlignment(Qt.AlignBottom)
        else:
            self.setAlignment(Qt.AlignVCenter)

        if police is not None:
            if police > 23:
                self.night_mode_switch(night_mode, f"{police}pt {titlefont}")
                self.setMargin(2)

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
        else:
            self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.night_mode_switch(night_mode)
        if size is None:
            size = [1, 4]
        self.line.setFixedHeight(size[0])
        self.line.setFixedWidth(size[1])
        self.layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.layout.addWidget(self.line)
        self.layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum))
        self.setLayout(self.layout)

    def night_mode_switch(self, night_mode):
        if night_mode:
            self.line.setStyleSheet("QFrame { background-color: rgb(255, 255, 255) }")
        else:
            self.line.setStyleSheet("QFrame { background-color: rgb(0, 0, 0) }")



class MainTabsWidget(QtWidgets.QPushButton):
    """
    A custom QPushButton that mimics an explorer tab appearance.

    Features:
    - Customizable text
    - Night mode support
    - Three states: not_in_use (grey border), in_use (black border), not_usable (grey text)
    - Tooltip support for not_usable state
    """

    def __init__(self, text="", night_mode=False, parent=None):
        super().__init__()

        self.setText(text)
        self.state = "not_in_use"  # States: "not_in_use", "in_use", "not_usable"
        self.setFont(buttonfont)

        # Set basic properties
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.setFixedHeight(35)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self.night_mode_switch(night_mode)


    def night_mode_switch(self, night_mode):
        self._night_mode = night_mode
        self.update_style()

    def update_style(self):
        """Update the widget's stylesheetµ"""

        if self.state == "not_usable":
            tab_text_color = "#888888"
            self.setCursor(QtCore.Qt.CursorShape.ForbiddenCursor)
        else:
            if self._night_mode:
                tab_text_color = night_text_Color
            else:
                tab_text_color = textColor
            if self.state == "not_in_use":
                self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        if self.state == "in_use":
            border_width = 2
            if self._night_mode:
                tab_border = f"{border_width}px solid #adadad"
            else:
                tab_border = f"{border_width}px solid #323241"
        else:
            border_width = 1
            if self._night_mode:
                tab_border = f"{border_width}px solid #323241"
            else:
                tab_border = f"{border_width}px solid #adadad"

        if self._night_mode:
            style = {"buttoncolor": night_button_color, "buttontextColor": tab_text_color, "border": tab_border,
                     "font_family": tabfont, "font_size": "22pt", #"font_weight": "bold",
                     "border-top-color": "#323241", "border-right-color": "#323241", # invisible
                     "border-top-left-radius": "10", "border-bottom-left-radius": "1"}
        else:
            style = {"buttoncolor": buttoncolor, "buttontextColor": tab_text_color, "border": tab_border,
                     "font_family": tabfont, "font_size": "22pt", #"font_weight": "bold",
                     "border-top-color": "#ffffff", "border-right-color": "#ffffff", # invisible
                     "border-top-left-radius": "10", "border-bottom-left-radius": "1"
                     }
        self.setStyleSheet(
            "background-color: %s; color: %s; border: %s; font-family: %s; font-size: %s;  border-top-color: %s; border-right-color: %s; border-top-left-radius: %s; border-bottom-left-radius: %s" % tuple(style.values()))


    def set_in_use(self):
        """Set the tab to 'in_use' state with black border."""
        self.state = "in_use"
        self.setToolTip("")  # Clear any tooltip
        self.update_style()

    def set_not_in_use(self):
        """Set the tab to 'not_in_use' state with grey border."""
        self.state = "not_in_use"
        self.setToolTip("")  # Clear any tooltip
        self.update_style()

    def set_not_usable(self, tooltip_text="This tab is not usable"):
        """
        Set the tab to 'not_usable' state with grey text.

        Args:
            tooltip_text (str): Custom tooltip text to show when hovering
        """
        self.state = "not_usable"
        self.setToolTip(tooltip_text)
        self.update_style()

    def get_state(self):
        """Get the current state of the tab."""
        return self.state

    def is_night_mode(self):
        """Check if night mode is enabled."""
        return self._night_mode