import ast
import os
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.gui.main_window import MainWindow
from src.gui.theme import (
    DARK_DIRECTORY_DIALOG_OPTIONS,
    DARK_FILE_DIALOG_OPTIONS,
)
from src.models.data_models import Config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_about_dialog_uses_the_application_dark_surface():
    app = _app()
    captured = []
    with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
        window = MainWindow(Config(language="zh_CN"))
    try:
        with mock.patch.object(
            QMessageBox,
            "exec",
            lambda dialog: captured.append(dialog) or 0,
        ):
            window._show_about()

        dialog = captured[0]
        assert dialog.objectName() == "aboutDialog"
        assert dialog.palette().color(QPalette.ColorRole.Window).name() == "#1a1a2e"
        assert dialog.palette().color(QPalette.ColorRole.WindowText).name() == "#e0e0e0"
        assert "QMessageBox" in app.styleSheet()
        assert dialog.standardButtons() == QMessageBox.StandardButton.Ok
    finally:
        window.close()
        app.processEvents()


def test_file_dialog_palette_and_options_are_project_dark_theme():
    app = _app()
    with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
        window = MainWindow(Config(language="en_US"))
    dialog = QFileDialog(window)
    try:
        dialog.setOptions(DARK_FILE_DIALOG_OPTIONS)
        assert dialog.testOption(QFileDialog.Option.DontUseNativeDialog)
        assert dialog.palette().color(QPalette.ColorRole.Window).name() == "#1a1a2e"
        assert app.palette().color(QPalette.ColorRole.Base).name() == "#16213e"
        assert DARK_DIRECTORY_DIALOG_OPTIONS & QFileDialog.Option.ShowDirsOnly
        assert DARK_DIRECTORY_DIALOG_OPTIONS & QFileDialog.Option.DontUseNativeDialog
        assert "QFileDialog" in app.styleSheet()
    finally:
        dialog.close()
        window.close()
        app.processEvents()


def test_every_desktop_static_file_dialog_explicitly_disables_native_skin():
    paths = (
        REPO_ROOT / "src/gui/main_window.py",
        REPO_ROOT / "src/gui/widgets/dropzone.py",
        REPO_ROOT / "src/gui/widgets/audio_track_mixer.py",
        REPO_ROOT / "src/gui/widgets/muscriptor_result.py",
    )
    calls = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id == "QFileDialog"
                and node.func.attr.startswith("get")
            ):
                calls.append((path, node))

    assert len(calls) == 7
    for path, call in calls:
        assert any(
            keyword.arg == "options" for keyword in call.keywords
        ), f"{path}:{call.lineno} does not opt into the project-themed file dialog"
