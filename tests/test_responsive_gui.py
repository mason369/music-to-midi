# ruff: noqa: E402

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOptionSlider,
    QTextEdit,
    QWidget,
)

from src.core.separation_service import SeparationResult
from src.gui.main_window import MainWindow
from src.gui.widgets.progress_widget import ProgressWidget
from src.gui.widgets.track_panel import TrackPanel
from src.i18n.translator import set_language
from src.models.data_models import Config, ProcessingMode, ProcessingStage


REPO_ROOT = Path(__file__).resolve().parents[1]


class _ResizableFakeAudioTrackMixer(QWidget):
    """Media-free mixer double whose minimum width never drives overflow."""

    midi_conversion_requested = pyqtSignal(str, str, str)
    midi_open_requested = pyqtSignal(str)

    def __init__(self, tracks):
        super().__init__()
        self.tracks = dict(tracks)
        self.shutdown_called = False
        self.setObjectName("responsiveFakeAudioMixer")
        self.setMinimumSize(0, 96)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    def shutdown(self):
        self.shutdown_called = True


class ResponsiveGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _window(self, language="zh_CN"):
        with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
            return MainWindow(Config(language=language))

    def _split_result(self):
        root = Path(self._temporary_directory.name)
        vocals = root / "vocals.wav"
        accompaniment = root / "accompaniment.wav"
        vocals.write_bytes(b"vocals")
        accompaniment.write_bytes(b"accompaniment")
        return SeparationResult(
            mode=ProcessingMode.VOCAL_SPLIT.value,
            source_path=str(root / "source.wav"),
            output_dir=str(root),
            separated_audio={
                "vocals": str(vocals),
                "accompaniment": str(accompaniment),
            },
            processing_time=1.0,
        )

    def _finish_with_fake_mixer(self, window, result):
        with mock.patch(
            "src.gui.main_window.AudioTrackMixerWidget",
            _ResizableFakeAudioTrackMixer,
        ):
            window._on_separation_finished(result)
        self._app.processEvents()
        self._app.processEvents()

    def _grabbed_pixel_name(self, pixmap, logical_point):
        self.assertFalse(pixmap.isNull())
        image = pixmap.toImage()
        dpr = pixmap.devicePixelRatio()
        physical_point = QPoint(
            int(logical_point.x() * dpr),
            int(logical_point.y() * dpr),
        )
        self.assertTrue(image.rect().contains(physical_point))
        return image.pixelColor(physical_point).name()

    def test_main_window_uses_vertical_scroll_area_and_resizes_to_small_window(self):
        window = self._window()
        try:
            window.showNormal()
            window.resize(480, 360)
            self._app.processEvents()

            self.assertIsInstance(window.centralWidget(), QScrollArea)
            self.assertEqual(window.size().width(), 480)
            self.assertEqual(window.size().height(), 360)
            self.assertEqual(
                window.centralWidget().horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
            self.assertGreater(window.centralWidget().verticalScrollBar().maximum(), 0)
        finally:
            window.close()

    def test_scroll_content_and_dropzone_render_dark_theme_backgrounds(self):
        window = self._window()
        try:
            window.showNormal()
            window.resize(900, 700)
            self._app.processEvents()

            scroll = window.content_scroll
            viewport = scroll.viewport()
            content = scroll.widget()
            self.assertIsNotNone(content)

            frame = viewport.grab()
            content_point = content.mapTo(viewport, QPoint(4, 4))
            dropzone_point = window.dropzone.mapTo(
                viewport,
                QPoint(16, window.dropzone.height() // 2),
            )
            self.assertTrue(viewport.rect().contains(content_point))
            self.assertTrue(viewport.rect().contains(dropzone_point))

            content_color = self._grabbed_pixel_name(frame, content_point)
            dropzone_color = self._grabbed_pixel_name(frame, dropzone_point)

            scrollbar = scroll.verticalScrollBar()
            self.assertTrue(scrollbar.isVisible())
            scrollbar_option = QStyleOptionSlider()
            scrollbar.initStyleOption(scrollbar_option)
            handle_rect = scrollbar.style().subControlRect(
                QStyle.ComplexControl.CC_ScrollBar,
                scrollbar_option,
                QStyle.SubControl.SC_ScrollBarSlider,
                scrollbar,
            )
            groove_rect = scrollbar.style().subControlRect(
                QStyle.ComplexControl.CC_ScrollBar,
                scrollbar_option,
                QStyle.SubControl.SC_ScrollBarGroove,
                scrollbar,
            )
            track_candidates = (
                QPoint(groove_rect.center().x(), groove_rect.top() + 4),
                QPoint(groove_rect.center().x(), groove_rect.bottom() - 4),
            )
            track_point = next(
                point for point in track_candidates if not handle_rect.contains(point)
            )
            scrollbar_frame = scrollbar.grab()
            scrollbar_handle_color = self._grabbed_pixel_name(
                scrollbar_frame,
                handle_rect.center(),
            )
            scrollbar_track_color = self._grabbed_pixel_name(
                scrollbar_frame,
                track_point,
            )

            content.hide()
            self._app.processEvents()
            viewport_color = self._grabbed_pixel_name(
                viewport.grab(),
                viewport.rect().center(),
            )

            self.assertEqual(content_color, "#1a1a2e")
            self.assertEqual(viewport_color, "#1a1a2e")
            self.assertEqual(content_color, viewport_color)
            self.assertEqual(dropzone_color, "#1f2940")
            self.assertEqual(scrollbar_handle_color, "#3a4a6a")
            self.assertEqual(scrollbar_track_color, "#16213e")
        finally:
            window.close()

    def test_status_bar_long_values_stay_inside_narrow_window(self):
        window = self._window("en_US")
        status_text = "Processing - " + "very-long-audio-file-name-" * 5 + ".wav"
        device_text = "NVIDIA GeForce RTX 5090 Laptop GPU (CUDA)"
        try:
            window.showNormal()
            window.resize(320, 360)
            window.status_label.setText(status_text)
            window._on_gpu_detected(device_text, (15.8, 16.0))
            self._app.processEvents()

            self.assertLessEqual(window.minimumSizeHint().width(), window.width())
            for label in (
                window.status_label,
                window.device_label,
                window.memory_label,
            ):
                with self.subTest(label=label.toolTip()):
                    self.assertGreater(label.width(), 0)
                    self.assertLessEqual(
                        label.mapTo(window.statusbar, label.rect().topRight()).x(),
                        window.statusbar.rect().right(),
                    )
                    self.assertTrue(label.toolTip())
        finally:
            window.close()

    def test_main_content_keeps_children_at_or_above_their_minimum_height(self):
        window = self._window()
        try:
            window.showNormal()
            window.resize(900, 700)
            self._app.processEvents()

            for widget in (
                window.dropzone,
                window.track_panel,
                window.progress_widget,
                window.output_group,
            ):
                with self.subTest(widget=type(widget).__name__):
                    self.assertGreaterEqual(
                        widget.height(), widget.minimumSizeHint().height()
                    )
        finally:
            window.close()

    def test_separated_track_option_is_visible_only_for_split_modes(self):
        window = self._window()
        try:
            self.assertTrue(window.tracks_check.isHidden())
            self.assertTrue(window.tracks_check.isChecked())

            window.tracks_check.setChecked(False)
            for mode in ("vocal_split", "six_stem_split"):
                with self.subTest(mode=mode):
                    window.track_panel.set_processing_mode(mode)
                    self.assertTrue(window.tracks_check.isHidden())
                    self.assertTrue(window.tracks_check.isChecked())

            for mode in (
                "smart",
                "piano_transkun",
                "piano_transkun_v2_aug",
                "piano_aria_amt",
                "piano_bytedance_pedal",
            ):
                with self.subTest(mode=mode):
                    window.track_panel.set_processing_mode(mode)
                    self.assertTrue(window.tracks_check.isHidden())

            self.assertTrue(window.tracks_check.isChecked())
        finally:
            window.close()

    def test_track_panel_can_shrink_without_horizontal_clipping(self):
        panel = TrackPanel()
        panel.resize(320, panel.sizeHint().height())
        panel.show()
        self._app.processEvents()
        try:
            self.assertLessEqual(panel.minimumSizeHint().width(), 320)
            for combo in (
                panel.mode_combo,
                panel.model_combo,
                panel.yourmt3_model_combo,
            ):
                with self.subTest(combo=combo.objectName() or combo.currentText()):
                    self.assertLessEqual(
                        combo.geometry().right(), panel.contentsRect().right()
                    )
        finally:
            panel.close()

    def test_progress_stages_reflow_at_narrow_width(self):
        widget = ProgressWidget()
        widget.set_mode("vocal_split")
        widget.resize(320, 240)
        widget.show()
        self._app.processEvents()
        try:
            self.assertEqual(
                tuple(widget.stage_indicators),
                (
                    ProcessingStage.PREPROCESSING,
                    ProcessingStage.SEPARATION,
                ),
            )
        finally:
            widget.close()
    def test_all_modes_and_languages_keep_content_inside_narrow_viewport(self):
        modes = (
            "smart",
            "vocal_split",
            "six_stem_split",
            "piano_transkun",
            "piano_aria_amt",
            "piano_bytedance_pedal",
        )
        for language in ("zh_CN", "en_US"):
            for mode in modes:
                window = None
                try:
                    set_language(language)
                    window = self._window(language)
                    window.track_panel.set_processing_mode(mode)
                    window.progress_widget.set_mode(mode)
                    window.showNormal()
                    window.resize(320, 360)
                    self._app.processEvents()

                    scroll = window.content_scroll
                    with self.subTest(language=language, mode=mode):
                        self.assertLessEqual(
                            scroll.widget().width(), scroll.viewport().width()
                        )
                        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
                finally:
                    if window is not None:
                        window.close()
        set_language("zh_CN")

    def test_error_dialog_has_no_artificial_maximum_and_fits_parent(self):
        window = self._window()
        captured = []
        try:
            window.resize(480, 360)
            with mock.patch.object(
                QDialog, "exec", lambda dialog: captured.append(dialog)
            ):
                window._on_error("failure details")

            dialog = captured[0]
            self.assertEqual(dialog.maximumHeight(), QWidget().maximumHeight())
            self.assertLessEqual(dialog.minimumWidth(), 320)
            self.assertLessEqual(
                dialog.width(), window.screen().availableGeometry().width()
            )
            original = dialog.size()
            dialog.resize(original.width() + 20, original.height() + 20)
            self.assertGreater(dialog.width(), original.width())
            self.assertGreater(dialog.height(), original.height())

            dialog.show()
            dialog.resize(320, 200)
            self._app.processEvents()
            scroll = dialog.findChild(QScrollArea)
            self.assertIsNotNone(scroll)
            self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
            content = scroll.widget()
            error_text = dialog.findChild(QTextEdit)
            buttons = dialog.findChildren(QPushButton)
            self.assertLess(
                error_text.mapTo(content, error_text.rect().bottomLeft()).y(),
                min(
                    button.mapTo(content, button.rect().topLeft()).y()
                    for button in buttons
                ),
            )
        finally:
            window.close()

    def test_inline_completion_panel_reflows_actions_and_opens_timeline_dock(self):
        window = self._window("en_US")
        result = self._split_result()
        try:
            window.showNormal()
            window.resize(760, 620)
            self._finish_with_fake_mixer(window, result)

            self.assertTrue(window.result_panel.isVisible())
            self.assertTrue(window.audio_timeline_container.isVisible())
            self.assertIsInstance(
                window.audio_mixer,
                _ResizableFakeAudioTrackMixer,
            )
            self.assertEqual(
                window.audio_mixer.tracks,
                {
                    name: Path(path).resolve()
                    for name, path in result.separated_audio.items()
                },
            )

            buttons = window.result_panel.findChildren(QPushButton)
            self.assertEqual(len(buttons), 1)
            for width in (760, 320):
                window.resize(width, 480)
                self._app.processEvents()
                self._app.processEvents()
                button = buttons[0]
                self.assertTrue(
                    window.result_panel.rect().contains(
                        button.mapTo(
                            window.result_panel,
                            button.rect().bottomRight(),
                        )
                    )
                )
        finally:
            window.close()
    def test_error_dialog_keeps_dark_surface_and_top_alignment(self):
        window = self._window("en_US")
        error_dialogs = []
        try:
            with mock.patch.object(
                QDialog, "exec", lambda dialog: error_dialogs.append(dialog)
            ):
                window._on_error("failure details")

            dialog = error_dialogs[0]
            dialog.show()
            self._app.processEvents()

            scroll = dialog.findChild(QScrollArea, "dialogScrollArea")
            self.assertIsNotNone(scroll)
            content = dialog.findChild(QWidget, "dialogScrollContent")
            self.assertIsNotNone(content)
            self.assertEqual(
                self._grabbed_pixel_name(content.grab(), QPoint(5, 5)),
                "#1a1a2e",
            )
            self.assertTrue(
                content.layout().alignment() & Qt.AlignmentFlag.AlignTop
            )

            first_item = content.layout().itemAt(0)
            first_widget = first_item.widget()
            if first_widget is None:
                first_widget = first_item.layout().itemAt(0).widget()
            self.assertIsNotNone(first_widget)
            self.assertLessEqual(
                first_widget.mapTo(content, QPoint(0, 0)).y(),
                24,
            )
        finally:
            for dialog in error_dialogs:
                dialog.close()
            window.close()

    def test_inline_completion_title_keeps_visible_width(self):
        window = self._window("en_US")
        try:
            window.showNormal()
            window.resize(320, 480)
            self._finish_with_fake_mixer(window, self._split_result())
            self._app.processEvents()

            title = window.result_title_label
            self.assertIs(title.parentWidget(), window.result_panel)
            self.assertEqual(title.objectName(), "successResultTitle")
            self.assertTrue(title.isVisible())
            self.assertTrue(title.text())
            self.assertGreater(title.width(), 100)
            self.assertLessEqual(
                title.mapTo(
                    window.result_panel,
                    title.rect().topRight(),
                ).x(),
                window.result_panel.rect().right(),
            )
            self.assertGreater(
                title.palette().color(title.foregroundRole()).lightness(),
                150,
            )
            self.assertTrue(window.audio_timeline_container.isVisible())
        finally:
            window.close()

    def test_narrow_inline_result_and_timeline_stay_inside_window(self):
        window = self._window("en_US")
        try:
            window.showNormal()
            window.resize(320, 480)
            self._finish_with_fake_mixer(window, self._split_result())
            self._app.processEvents()

            scroll = window.content_scroll
            viewport = scroll.viewport()
            content = scroll.widget()
            self.assertLessEqual(content.width(), viewport.width())
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)

            panel = window.result_panel
            for button in panel.findChildren(QPushButton):
                with self.subTest(button=button.text()):
                    top_left = button.mapTo(panel, button.rect().topLeft())
                    bottom_right = button.mapTo(panel, button.rect().bottomRight())
                    self.assertGreaterEqual(top_left.x(), panel.rect().left())
                    self.assertLessEqual(bottom_right.x(), panel.rect().right())

            container = window.audio_timeline_container
            self.assertIs(container.parentWidget(), panel)
            self.assertIs(container.window(), window)
            container_left = container.mapTo(panel, container.rect().topLeft()).x()
            container_right = container.mapTo(panel, container.rect().topRight()).x()
            self.assertGreaterEqual(container_left, panel.rect().left())
            self.assertLessEqual(container_right, panel.rect().right())
            self.assertEqual(
                self._grabbed_pixel_name(
                    container.grab(),
                    QPoint(12, 12),
                ),
                "#1a1a2e",
            )

            mixer = window.audio_mixer
            self.assertIsInstance(mixer, _ResizableFakeAudioTrackMixer)
            self.assertEqual(mixer.minimumWidth(), 0)
            mixer_left = mixer.mapTo(container, mixer.rect().topLeft()).x()
            mixer_right = mixer.mapTo(container, mixer.rect().topRight()).x()
            self.assertGreaterEqual(mixer_left, container.rect().left())
            self.assertLessEqual(
                mixer_right,
                container.rect().right(),
            )
        finally:
            window.close()

    def test_inline_result_and_timeline_have_no_overflow_at_fractional_scales(self):
        probe = textwrap.dedent(
            """
            import json
            from pathlib import Path
            from tempfile import TemporaryDirectory
            from unittest import mock
            from PyQt6.QtCore import QPoint, Qt, pyqtSignal
            from PyQt6.QtWidgets import (
                QApplication,
                QPushButton,
                QSizePolicy,
                QWidget,
            )
            from src.gui.main_window import MainWindow
            from src.core.separation_service import SeparationResult
            from src.models.data_models import Config, ProcessingMode

            class FakeAudioTrackMixer(QWidget):
                midi_conversion_requested = pyqtSignal(str, str, str)
                midi_open_requested = pyqtSignal(str)

                def __init__(self, tracks):
                    super().__init__()
                    self.tracks = dict(tracks)
                    self.setMinimumSize(0, 96)
                    self.setSizePolicy(
                        QSizePolicy.Policy.Expanding,
                        QSizePolicy.Policy.Expanding,
                    )

                def shutdown(self):
                    pass

            def horizontal_bounds(widget, ancestor):
                return [
                    widget.mapTo(ancestor, widget.rect().topLeft()).x(),
                    widget.mapTo(ancestor, widget.rect().topRight()).x(),
                ]

            app = QApplication([])
            with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
                window = MainWindow(Config(language="en_US"))
            window.showNormal()
            window.resize(320, 360)
            temporary_directory = TemporaryDirectory()
            root = Path(temporary_directory.name)
            vocals = root / "vocals.wav"
            accompaniment = root / "accompaniment.wav"
            vocals.write_bytes(b"vocals")
            accompaniment.write_bytes(b"accompaniment")
            result = SeparationResult(
                mode=ProcessingMode.VOCAL_SPLIT.value,
                source_path=str(root / "source.wav"),
                output_dir=str(root),
                processing_time=1.0,
                separated_audio={
                    "vocals": str(vocals),
                    "accompaniment": str(accompaniment),
                },
            )
            with mock.patch(
                "src.gui.main_window.AudioTrackMixerWidget",
                FakeAudioTrackMixer,
            ):
                window._on_separation_finished(result)
            app.processEvents()
            app.processEvents()

            scroll = window.content_scroll
            viewport = scroll.viewport()
            panel = window.result_panel
            container = window.audio_timeline_container
            mixer = window.audio_mixer
            button_bounds = [
                horizontal_bounds(button, panel)
                for button in panel.findChildren(QPushButton)
            ]
            print(json.dumps({
                "result_visible": panel.isVisible(),
                "timeline_visible": container.isVisible(),
                "content_width": scroll.widget().width(),
                "viewport_width": viewport.width(),
                "horizontal_maximum": scroll.horizontalScrollBar().maximum(),
                "panel_right": panel.rect().right(),
                "button_bounds": button_bounds,
                "container_bounds": horizontal_bounds(container, panel),
                "mixer_bounds": horizontal_bounds(mixer, container),
                "container_right": container.rect().right(),
                "container_parent_is_panel": container.parentWidget() is panel,
                "container_window_is_main": container.window() is window,
            }))
            window.close()
            temporary_directory.cleanup()
            """
        )

        for scale in ("1", "1.25", "1.5", "2"):
            env = os.environ.copy()
            env["QT_QPA_PLATFORM"] = "offscreen"
            env["QT_SCALE_FACTOR"] = scale
            completed = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            measured = json.loads(completed.stdout.strip().splitlines()[-1])
            with self.subTest(scale=scale):
                self.assertTrue(measured["result_visible"])
                self.assertTrue(measured["timeline_visible"])
                self.assertLessEqual(
                    measured["content_width"], measured["viewport_width"]
                )
                self.assertEqual(measured["horizontal_maximum"], 0)
                for left, right in measured["button_bounds"]:
                    self.assertGreaterEqual(left, 0)
                    self.assertLessEqual(right, measured["panel_right"])
                self.assertGreaterEqual(measured["container_bounds"][0], 0)
                self.assertLessEqual(
                    measured["container_bounds"][1], measured["panel_right"]
                )
                self.assertGreaterEqual(measured["mixer_bounds"][0], 0)
                self.assertLessEqual(
                    measured["mixer_bounds"][1], measured["container_right"]
                )
                self.assertTrue(measured["container_parent_is_panel"])
                self.assertTrue(measured["container_window_is_main"])

    def test_initial_window_fits_available_geometry_at_fractional_scales(self):
        probe = textwrap.dedent(
            """
            import json
            from unittest import mock
            from PyQt6.QtWidgets import QApplication
            from src.gui.main_window import MainWindow
            from src.models.data_models import Config

            app = QApplication([])
            with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
                window = MainWindow(Config())
            window.showNormal()
            app.processEvents()
            available = app.primaryScreen().availableGeometry()
            print(json.dumps({
                "window": [window.width(), window.height()],
                "available": [available.width(), available.height()],
            }))
            window.close()
            """
        )

        for scale in ("1", "1.25", "1.5", "2"):
            env = os.environ.copy()
            env["QT_QPA_PLATFORM"] = "offscreen"
            env["QT_SCALE_FACTOR"] = scale
            completed = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            measured = json.loads(completed.stdout.strip().splitlines()[-1])
            with self.subTest(scale=scale):
                self.assertLessEqual(measured["window"][0], measured["available"][0])
                self.assertLessEqual(measured["window"][1], measured["available"][1])


if __name__ == "__main__":
    unittest.main()
