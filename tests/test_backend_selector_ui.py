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

from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.gui.widgets.track_panel import TrackPanel
from src.i18n.translator import set_language
from src.models.data_models import Config, QualityBehavior


class TestBackendSelectorUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])
        set_language("en_US")

    def test_track_panel_only_lists_supported_modes_and_backends(self):
        panel = TrackPanel()

        modes = [panel.mode_combo.itemData(index) for index in range(panel.mode_combo.count())]
        backends = [panel.model_combo.itemData(index) for index in range(panel.model_combo.count())]

        self.assertEqual(modes, ["smart", "vocal_split"])
        self.assertEqual(backends, ["yourmt3", "miros"])
        self.assertNotIn("six_stem_split", modes)
        self.assertNotIn("piano_transkun", modes)
        self.assertNotIn("piano_aria_amt", modes)
        self.assertNotIn("aria_amt", backends)

    def test_removed_mode_and_backend_fall_back_to_supported_defaults(self):
        panel = TrackPanel()

        panel.set_processing_mode("six_stem_split")
        panel.set_transcription_backend("aria_amt")

        self.assertEqual(panel.get_processing_mode(), "smart")
        self.assertEqual(panel.get_transcription_backend(), "yourmt3")

    def test_processing_controls_disable_mode_specific_checkboxes(self):
        panel = TrackPanel()
        panel.set_processing_mode("vocal_split")
        panel.set_processing_controls_enabled(False)
        self.assertFalse(panel._vocal_split_merge_check.isEnabled())

    def test_main_window_replaces_saved_aria_backend_with_saved_multi_backend(self):
        config = Config(transcription_backend="aria_amt", multi_instrument_model="miros")

        with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
            window = MainWindow(config)

        self.assertEqual(window.track_panel.get_transcription_backend(), "miros")
        self.assertEqual(window.track_panel.get_multi_instrument_model(), "miros")
        window.close()

    def test_track_panel_shows_midi_track_mode_for_yourmt3_backend(self):
        panel = TrackPanel()
        panel.set_processing_mode("smart")
        panel.set_transcription_backend("yourmt3")

        self.assertFalse(panel._midi_track_mode_row.isHidden())
        self.assertEqual(panel.get_midi_track_mode(), "multi_track")

        panel.set_midi_track_mode("single_track")
        self.assertEqual(panel.get_midi_track_mode(), "single_track")

    def test_track_panel_explains_official_yourmt3_single_multi_meaning(self):
        panel = TrackPanel()
        panel.set_processing_mode("smart")
        panel.set_transcription_backend("yourmt3")

        text = panel.yourmt3_arch_hint_label.text()

        self.assertFalse(panel.yourmt3_arch_hint_label.isHidden())
        self.assertIn("YPTF+Single", text)
        self.assertIn("mc13_full_plus_256", text)
        self.assertIn("multi-t5", text)
        self.assertIn("output layout", text)

    def test_track_panel_hides_midi_track_mode_for_non_yourmt3_backend(self):
        panel = TrackPanel()
        panel.set_processing_mode("smart")
        panel.set_transcription_backend("miros")
        panel.set_midi_track_mode("single_track")

        self.assertTrue(panel._midi_track_mode_row.isHidden())
        self.assertTrue(panel.yourmt3_arch_hint_label.isHidden())
        self.assertEqual(panel.get_midi_track_mode(), "multi_track")

    def test_removed_dedicated_piano_mode_uses_smart_quality_behavior(self):
        panel = TrackPanel()
        panel.set_processing_mode("piano_transkun")

        self.assertEqual(panel.get_processing_mode(), "smart")
        self.assertEqual(panel.get_quality_behavior(), QualityBehavior.CONFIGURABLE)


if __name__ == "__main__":
    unittest.main()
