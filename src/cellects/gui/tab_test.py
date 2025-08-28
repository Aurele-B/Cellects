from PySide6 import QtWidgets, QtCore, QtGui


class ExplorerTab(QtWidgets.QPushButton):
    """
    A custom QPushButton that mimics an explorer tab appearance.

    Features:
    - Customizable text
    - Night mode support
    - Three states: not_in_use (grey border), in_use (black border), not_usable (grey text)
    - Tooltip support for not_usable state
    """

    def __init__(self, text="", night_mode=False, parent=None):
        super().__init__(text, parent)

        self._night_mode = night_mode
        self._state = "not_in_use"  # States: "not_in_use", "in_use", "not_usable"

        # Set basic properties
        self.setFixedHeight(32)
        self.setMinimumWidth(80)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        # Apply initial styling
        self._update_style()

    def _update_style(self):
        """Update the widget's stylesheet based on current state and mode."""

        # Base colors for light mode
        if not self._night_mode:
            bg_color = "#ffffff"
            text_color = "#000000"
            border_color = "#cccccc"
            hover_bg = "#f0f0f0"
        else:
            # Dark mode colors
            bg_color = "#2d2d2d"
            text_color = "#ffffff"
            border_color = "#555555"
            hover_bg = "#404040"

        # Adjust colors based on state
        if self._state == "in_use":
            border_color = "#000000" if not self._night_mode else "#ffffff"
            border_width = "2px"
        elif self._state == "not_usable":
            text_color = "#888888"
            border_width = "1px"
            self.setCursor(QtCore.Qt.CursorShape.ForbiddenCursor)
        else:  # not_in_use
            border_width = "1px"
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        # Create stylesheet
        style = f"""
            ExplorerTab {{
                background-color: {bg_color};
                color: {text_color};
                border: {border_width} solid {border_color};
                border-radius: 6px 6px 0px 0px;
                padding: 6px 12px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                font-weight: normal;
                text-align: center;
            }}
        """

        # Add hover effect only if not in not_usable state
        if self._state != "not_usable":
            style += f"""
                ExplorerTab:hover {{
                    background-color: {hover_bg};
                }}
            """

        # Add pressed effect only if not in not_usable state
        if self._state != "not_usable":
            pressed_bg = "#e0e0e0" if not self._night_mode else "#353535"
            style += f"""
                ExplorerTab:pressed {{
                    background-color: {pressed_bg};
                }}
            """

        self.setStyleSheet(style)

    def set_in_use(self):
        """Set the tab to 'in_use' state with black border."""
        self._state = "in_use"
        self.setToolTip("")  # Clear any tooltip
        self._update_style()

    def set_not_in_use(self):
        """Set the tab to 'not_in_use' state with grey border."""
        self._state = "not_in_use"
        self.setToolTip("")  # Clear any tooltip
        self._update_style()

    def set_not_usable(self, tooltip_text="This tab is not usable"):
        """
        Set the tab to 'not_usable' state with grey text.

        Args:
            tooltip_text (str): Custom tooltip text to show when hovering
        """
        self._state = "not_usable"
        self.setToolTip(tooltip_text)
        self._update_style()

    def set_night_mode(self, enabled):
        """
        Enable or disable night mode.

        Args:
            enabled (bool): True to enable night mode, False for light mode
        """
        self._night_mode = enabled
        self._update_style()

    def get_state(self):
        """Get the current state of the tab."""
        return self._state

    def is_night_mode(self):
        """Check if night mode is enabled."""
        return self._night_mode


# Example usage and demo
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    # Create main window
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Explorer Tab Demo")
    window.setGeometry(100, 100, 600, 200)

    # Create central widget and layout
    central_widget = QtWidgets.QWidget()
    window.setCentralWidget(central_widget)
    layout = QtWidgets.QVBoxLayout(central_widget)

    # Create tab container
    tab_container = QtWidgets.QHBoxLayout()

    # Create sample tabs
    tab1 = ExplorerTab("Home", night_mode=False)
    tab2 = ExplorerTab("Documents", night_mode=False)
    tab3 = ExplorerTab("Downloads", night_mode=False)
    tab4 = ExplorerTab("Pictures", night_mode=False)

    # Set different states
    tab1.set_in_use()  # Active tab
    tab2.set_not_in_use()  # Normal tab
    tab3.set_not_usable("This folder is currently being synchronized")  # Disabled tab
    tab4.set_not_in_use()  # Normal tab

    # Add tabs to container
    tab_container.addWidget(tab1)
    tab_container.addWidget(tab2)
    tab_container.addWidget(tab3)
    tab_container.addWidget(tab4)
    tab_container.addStretch()  # Push tabs to the left

    # Create controls
    controls_layout = QtWidgets.QHBoxLayout()

    night_mode_btn = QtWidgets.QPushButton("Toggle Night Mode")
    state_btn = QtWidgets.QPushButton("Cycle States (Tab 1)")


    def toggle_night_mode():
        current_mode = tab1.is_night_mode()
        for tab in [tab1, tab2, tab3, tab4]:
            tab.set_night_mode(not current_mode)


    def cycle_states():
        current_state = tab1.get_state()
        if current_state == "not_in_use":
            tab1.set_in_use()
        elif current_state == "in_use":
            tab1.set_not_usable("This tab is temporarily disabled")
        else:
            tab1.set_not_in_use()


    night_mode_btn.clicked.connect(toggle_night_mode)
    state_btn.clicked.connect(cycle_states)

    controls_layout.addWidget(night_mode_btn)
    controls_layout.addWidget(state_btn)
    controls_layout.addStretch()

    # Add to main layout
    layout.addLayout(tab_container)
    layout.addLayout(controls_layout)
    layout.addStretch()

    window.show()
    sys.exit(app.exec_())