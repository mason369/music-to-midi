"""Application-wide desktop colors and themed dialog options."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QFileDialog

DARK_FILE_DIALOG_OPTIONS = QFileDialog.Option.DontUseNativeDialog
DARK_DIRECTORY_DIALOG_OPTIONS = (
    QFileDialog.Option.DontUseNativeDialog | QFileDialog.Option.ShowDirsOnly
)

_DIALOG_THEME_MARKER = "/* music-to-midi-dark-dialog-theme */"
_DIALOG_STYLESHEET = f"""
{_DIALOG_THEME_MARKER}
QDialog,
QMessageBox,
QFileDialog {{
    background-color: #1a1a2e;
    color: #e0e0e0;
}}
QDialog QLabel,
QDialog QCheckBox,
QDialog QRadioButton {{
    background: transparent;
    color: #c8d3e6;
}}
QDialog QPushButton {{
    min-height: 28px;
    padding: 5px 14px;
    background: #2a3f5f;
    border: 1px solid #3a4a6a;
    border-radius: 5px;
    color: #e0e0e0;
}}
QDialog QPushButton:hover {{
    background: #3a5a7c;
    border-color: #4a9eff;
}}
QDialog QPushButton:pressed {{
    background: #203f68;
}}
QDialog QPushButton:default {{
    background: #315c8f;
    border-color: #4a9eff;
}}
QDialog QPushButton:disabled {{
    background: #202e49;
    border-color: #33425d;
    color: #65738a;
}}
QDialog QLineEdit,
QDialog QTextEdit,
QDialog QPlainTextEdit,
QDialog QComboBox,
QDialog QListView,
QDialog QTreeView,
QDialog QTableView {{
    background: #16213e;
    border: 1px solid #3a4a6a;
    border-radius: 4px;
    color: #e0e0e0;
    selection-background-color: #3a5a7c;
    selection-color: #ffffff;
}}
QDialog QLineEdit,
QDialog QComboBox {{
    min-height: 26px;
    padding: 3px 7px;
}}
QDialog QHeaderView::section {{
    background: #1f2940;
    border: 0;
    border-right: 1px solid #3a4a6a;
    border-bottom: 1px solid #3a4a6a;
    color: #c8d3e6;
    padding: 5px;
}}
QDialog QToolButton {{
    background: #2a3f5f;
    border: 1px solid #3a4a6a;
    border-radius: 4px;
    color: #e0e0e0;
    padding: 4px;
}}
QDialog QToolButton:hover {{
    background: #3a5a7c;
    border-color: #4a9eff;
}}
QDialog QMenu {{
    background: #1f2940;
    border: 1px solid #3a4a6a;
    color: #e0e0e0;
}}
QDialog QMenu::item:selected {{
    background: #3a5a7c;
}}
QDialog QScrollBar:vertical {{
    background: #16213e;
    width: 12px;
    margin: 0;
}}
QDialog QScrollBar:horizontal {{
    background: #16213e;
    height: 12px;
    margin: 0;
}}
QDialog QScrollBar::handle {{
    background: #3a4a6a;
    min-width: 28px;
    min-height: 28px;
    border-radius: 5px;
    margin: 2px;
}}
QDialog QScrollBar::handle:hover {{
    background: #4a5f80;
}}
QDialog QScrollBar::add-line,
QDialog QScrollBar::sub-line {{
    width: 0;
    height: 0;
    background: transparent;
}}
QDialog QScrollBar::add-page,
QDialog QScrollBar::sub-page {{
    background: transparent;
}}
"""


def apply_dark_application_theme(application: QApplication) -> None:
    """Apply the product palette to top-level Qt dialogs and their controls."""

    if not isinstance(application, QApplication):
        raise TypeError("application must be a QApplication")

    palette = application.palette()
    colors = {
        QPalette.ColorRole.Window: "#1a1a2e",
        QPalette.ColorRole.WindowText: "#e0e0e0",
        QPalette.ColorRole.Base: "#16213e",
        QPalette.ColorRole.AlternateBase: "#1f2940",
        QPalette.ColorRole.ToolTipBase: "#1f2940",
        QPalette.ColorRole.ToolTipText: "#e0e0e0",
        QPalette.ColorRole.Text: "#e0e0e0",
        QPalette.ColorRole.Button: "#2a3f5f",
        QPalette.ColorRole.ButtonText: "#e0e0e0",
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Link: "#4a9eff",
        QPalette.ColorRole.Highlight: "#3a5a7c",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.PlaceholderText: "#8292aa",
        QPalette.ColorRole.Light: "#3a4a6a",
        QPalette.ColorRole.Midlight: "#33425d",
        QPalette.ColorRole.Mid: "#2a3f5f",
        QPalette.ColorRole.Dark: "#101a32",
        QPalette.ColorRole.Shadow: "#080d19",
    }
    for role, value in colors.items():
        palette.setColor(role, QColor(value))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.PlaceholderText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#65738a"))
    application.setPalette(palette)

    existing = application.styleSheet().strip()
    if _DIALOG_THEME_MARKER not in existing:
        application.setStyleSheet(
            f"{existing}\n{_DIALOG_STYLESHEET}" if existing else _DIALOG_STYLESHEET
        )


__all__ = [
    "DARK_DIRECTORY_DIALOG_OPTIONS",
    "DARK_FILE_DIALOG_OPTIONS",
    "apply_dark_application_theme",
]
