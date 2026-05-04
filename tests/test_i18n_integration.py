import os
import sys
import types
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import mido  # noqa: F401
except ImportError:
    mido_stub = types.ModuleType("mido")
    mido_stub.__spec__ = None

    class _Dummy:
        pass

    mido_stub.MidiFile = _Dummy
    mido_stub.MidiTrack = _Dummy
    mido_stub.Message = _Dummy
    mido_stub.MetaMessage = _Dummy
    sys.modules.setdefault("mido", mido_stub)

from PyQt6.QtWidgets import QApplication, QLabel

from src.gui.main_window import MainWindow
from src.i18n.translator import set_language, t
from src.models.data_models import Config


class TestDesktopI18nIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self):
        set_language("zh_CN")

    def _window(self, language="zh_CN"):
        with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
            return MainWindow(Config(language=language))

    def test_main_window_honors_config_language_at_startup(self):
        window = self._window("en_US")
        try:
            self.assertEqual(window.windowTitle(), "Music to MIDI")
            self.assertEqual(
                [action.text() for action in window.menuBar().actions()],
                ["File", "Edit", "View", "Help"],
            )
        finally:
            window.close()

    def test_language_switch_refreshes_menus_header_and_action_icons(self):
        window = self._window("zh_CN")
        try:
            set_language("en_US")
            window._update_translations()

            self.assertEqual(
                [action.text() for action in window.menuBar().actions()],
                ["File", "Edit", "View", "Help"],
            )
            labels = [label.text() for label in window.findChildren(QLabel)]
            self.assertIn("Music to MIDI", labels)
            self.assertIn("Intelligently convert music to MIDI files", labels)
            self.assertEqual(window.start_btn.text(), "▶  Start")
            self.assertEqual(window.stop_btn.text(), "■  Stop")
        finally:
            window.close()

    def test_language_switch_refreshes_selected_file_prefix(self):
        window = self._window("zh_CN")
        try:
            window.dropzone._set_file(r"C:\tmp\song.wav")

            set_language("en_US")
            window._update_translations()

            self.assertEqual(window.dropzone.file_label.text(), "Selected file: song.wav")
        finally:
            window.close()

    def test_completion_dialog_summary_keys_are_translated(self):
        set_language("en_US")
        self.assertEqual(t("dialogs.complete.processing_time"), "Processing Time")
        self.assertEqual(t("dialogs.complete.midi_file"), "MIDI File")
        self.assertEqual(t("dialogs.complete.track_count"), "Track Count")
        self.assertEqual(t("dialogs.complete.note_count"), "Note Count")
        self.assertEqual(t("dialogs.complete.seconds_suffix"), "s")

        set_language("zh_CN")
        self.assertEqual(t("dialogs.complete.processing_time"), "处理时间")
        self.assertEqual(t("dialogs.complete.midi_file"), "MIDI文件")
        self.assertEqual(t("dialogs.complete.track_count"), "轨道数")
        self.assertEqual(t("dialogs.complete.note_count"), "音符数")
        self.assertEqual(t("dialogs.complete.seconds_suffix"), "秒")

    def test_status_memory_key_is_translated(self):
        set_language("en_US")
        self.assertEqual(t("status.video_memory"), "VRAM")

        set_language("zh_CN")
        self.assertEqual(t("status.video_memory"), "显存")

    def test_cancelled_status_key_is_translated(self):
        set_language("en_US")
        self.assertEqual(t("status.cancelled"), "Processing cancelled")

        set_language("zh_CN")
        self.assertEqual(t("status.cancelled"), "处理已取消")

    def test_pipeline_progress_helpers_are_translated(self):
        from src.core.pipeline import MusicToMidiPipeline

        english_pipeline = MusicToMidiPipeline(Config(language="en_US"))
        self.assertEqual(english_pipeline._pt("progress.analyzing_audio"), "Analyzing audio...")
        self.assertEqual(
            english_pipeline._pt("progress.complete_elapsed", seconds="1.2"),
            "Complete, elapsed 1.2 seconds",
        )

        chinese_pipeline = MusicToMidiPipeline(Config(language="zh_CN"))
        self.assertEqual(chinese_pipeline._pt("progress.analyzing_audio"), "正在分析音频...")
        self.assertEqual(
            chinese_pipeline._pt("progress.complete_elapsed", seconds="1.2"),
            "处理完成，耗时 1.2 秒",
        )


if __name__ == "__main__":
    unittest.main()
