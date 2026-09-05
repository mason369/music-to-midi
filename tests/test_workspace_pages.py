"""Drive the actual Qt workspace with events, without desktop automation."""

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.i18n.translator import set_language
from src.models.data_models import BeatInfo, Config, ProcessingResult


@pytest.mark.parametrize("language", ["zh_CN", "en_US"])
def test_default_controls_height_matches_wrapped_content(window, language):
    set_language(language)
    window._update_translations()
    window._show_muscriptor_streaming("guitar.wav", ["acoustic_guitar"])
    workbench = window.muscriptor_result_widget
    for width in (1280, 320, 1920, 1280):
        window.resize(width, 1080)
        for _ in range(10):
            QApplication.processEvents()
        controls = workbench.result_controls_panel
        assert controls.sizeHint().height() == controls._wrapped_height
        assert workbench.result_splitter.sizes()[0] == controls.minimumSizeHint().height()
        assert workbench.result_splitter.sizes()[1] >= workbench.roll_panel.minimumSizeHint().height()


@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    with patch.object(MainWindow, "_start_gpu_detection", return_value=None):
        value = MainWindow(Config(language="zh_CN", output_dir=str(tmp_path)))
    value.resize(1280, 900)
    value.show()
    app.processEvents()
    try:
        yield value
    finally:
        value.close()
        app.processEvents()


def test_pages_retain_state_and_never_start_work_on_navigation(window):
    assert window.workspace_tabs.currentIndex() == 0
    assert not window.workspace_tabs.isTabEnabled(1)
    window.track_panel.set_processing_mode("six_stem_split")
    result = ProcessingResult(midi_path="unchanged.mid", beat_info=BeatInfo(128.4))
    window._show_success_result(result, show_timeline=False)
    panel = window.result_panel
    assert window.workspace_tabs.currentIndex() == 1
    tabbar = window.workspace_tabs.tabBar()
    with patch.object(window, "_start_processing") as start:
        for _ in range(5):
            QTest.mouseClick(tabbar, Qt.MouseButton.LeftButton, pos=tabbar.tabRect(0).center())
            assert window.workspace_tabs.currentIndex() == 0
            QTest.mouseClick(tabbar, Qt.MouseButton.LeftButton, pos=tabbar.tabRect(1).center())
            assert window.workspace_tabs.currentIndex() == 1
        start.assert_not_called()
    assert window.result_panel is panel
    assert window._last_result is result
    assert window.track_panel.get_processing_mode() == "six_stem_split"
    assert window.worker is None


def test_results_show_uncertainty_and_language_change_does_not_navigate(window):
    result = ProcessingResult(
        midi_path="estimated.mid", beat_info=BeatInfo(128.4, fixed_tempo_reliable=False)
    )
    window._show_success_result(result, show_timeline=False)
    assert "BPM 为估计值" in window.result_info_label.text()
    window.workspace_tabs.setCurrentIndex(0)
    set_language("en_US")
    window._update_translations()
    assert window.workspace_tabs.currentIndex() == 0
    assert window.workspace_tabs.tabText(0) == "Setup"
    assert "BPM is an estimate" in window.result_info_label.text()


@pytest.mark.parametrize("width,height", [(320, 360), (1280, 900), (1920, 1080)])
def test_result_page_keeps_actions_reachable_at_each_size(window, width, height):
    window.resize(width, height)
    window._reveal_result_page()
    QApplication.processEvents()
    assert window.result_scroll.horizontalScrollBar().maximum() == 0
    for button in (window.start_btn, window.stop_btn):
        assert button.isVisible()
        assert window.rect().contains(button.mapTo(window, button.rect().topLeft()))
        assert window.rect().contains(button.mapTo(window, button.rect().bottomRight()))


def test_new_input_clears_stale_result_and_returns_to_setup(window):
    window._show_success_result(ProcessingResult(midi_path="old.mid"), show_timeline=False)
    window._on_file_selected("new-guitar-04-01a0613f.wav")
    assert window.workspace_tabs.currentIndex() == 0
    assert not window.workspace_tabs.isTabEnabled(1)
    assert window._last_result is None
    assert window.track_panel.get_muscriptor_instruments() == [
        "acoustic_guitar",
        "clean_electric_guitar",
        "distorted_electric_guitar",
    ]


@pytest.mark.parametrize("language", ["zh_CN", "en_US"])
def test_filled_workbench_reflows_without_clipping_controls(window, language):
    from PyQt6.QtWidgets import (
        QAbstractSpinBox,
        QBoxLayout,
        QCheckBox,
        QComboBox,
        QPushButton,
        QToolButton,
    )
    from src.core.muscriptor_result_assets import MuscriptorRollNote

    set_language(language)
    window._update_translations()
    instruments = ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"]
    window._show_muscriptor_streaming("test-guitar-04-01a0613f.wav", instruments)
    workbench = window.muscriptor_result_widget
    workbench._detected = instruments[:]
    workbench._rebuild_instrument_rows()
    workbench.set_bpm_context(128.4, 128.4)
    notes = tuple(
        MuscriptorRollNote(name, 60 + index, 90, 0.0, 1.0, program=24 + index)
        for index, name in enumerate(instruments)
    )
    workbench._begin_editor_session(notes, 184.57)
    workbench.roll.set_notes(notes, duration=184.57)
    workbench.edit_toggle.setChecked(True)
    for width in [1280, 320, 1280]:
        window.resize(width, 900)
        # Reflow invalidates several nested Qt layouts; drain their queued requests.
        for _ in range(10):
            QApplication.processEvents()
        assert window.width() == width
        assert window.result_scroll.horizontalScrollBar().maximum() == 0
        assert workbench._roll_content_layout.direction() == (
            QBoxLayout.Direction.TopToBottom if width == 320 else QBoxLayout.Direction.LeftToRight
        )
        for control in workbench.findChildren(
            (QPushButton, QToolButton, QComboBox, QAbstractSpinBox, QCheckBox)
        ):
            if not control.isVisible():
                continue
            assert control.parentWidget().rect().contains(control.geometry()), (
                control.metaObject().className(),
                control.geometry(),
                control.parentWidget().size(),
            )
        assert workbench.result_controls_panel.minimumHeight() == 0
        workbench.result_splitter.setSizes([0, 900])
        QApplication.processEvents()
        assert workbench.result_splitter.sizes()[0] == 0
        assert workbench.result_controls_panel.height() == 0
        workbench.result_splitter.setSizes(
            [workbench.result_controls_panel.minimumSizeHint().height(), 500]
        )
