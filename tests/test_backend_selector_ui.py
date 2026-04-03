import os
import sys
import types
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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

    def test_track_panel_preserves_explicit_multi_backend_when_aria_is_preferred(self):
        panel = TrackPanel()

        panel.set_transcription_backend("aria_amt")
        panel.set_multi_instrument_model("miros")

        self.assertEqual(panel.get_transcription_backend(), "aria_amt")
        self.assertEqual(panel.get_multi_instrument_model(), "miros")

    def test_track_panel_preserves_aria_fallback_when_switching_away_and_back(self):
        panel = TrackPanel()

        panel.set_transcription_backend("aria_amt")
        panel.set_multi_instrument_model("miros")
        panel.set_transcription_backend("yourmt3")
        panel.set_transcription_backend("aria_amt")

        self.assertEqual(panel.get_transcription_backend(), "aria_amt")
        self.assertEqual(panel.get_multi_instrument_model(), "miros")

    def test_processing_controls_disable_mode_specific_checkboxes(self):
        panel = TrackPanel()
        panel.set_processing_mode("vocal_split")
        panel.set_processing_controls_enabled(False)
        self.assertFalse(panel._vocal_split_merge_check.isEnabled())

        panel.set_processing_mode("six_stem_split")
        panel.set_processing_controls_enabled(False)
        self.assertFalse(panel._six_stem_only_selected_check.isEnabled())
        self.assertFalse(panel._six_stem_vocal_harmony_check.isEnabled())

    def test_main_window_loads_saved_aria_with_miros_fallback(self):
        config = Config(transcription_backend="aria_amt", multi_instrument_model="miros")

        with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
            window = MainWindow(config)

        self.assertEqual(window.track_panel.get_transcription_backend(), "aria_amt")
        self.assertEqual(window.track_panel.get_multi_instrument_model(), "miros")
        window.close()

    def test_track_panel_exposes_partial_quality_behavior_for_aria_plus_yourmt3_six_stem(self):
        panel = TrackPanel()
        panel.set_processing_mode("six_stem_split")
        panel.set_transcription_backend("aria_amt")
        panel.set_multi_instrument_model("yourmt3")

        self.assertEqual(panel.get_quality_behavior(), QualityBehavior.PARTIAL)
        self.assertIn("Quality selector only affects the YourMT3+ full-mix pass", panel.quality_hint_label.text())

    def test_track_panel_exposes_fixed_quality_behavior_for_dedicated_piano_mode(self):
        panel = TrackPanel()
        panel.set_processing_mode("piano_transkun")

        self.assertEqual(panel.get_quality_behavior(), QualityBehavior.FIXED)
        self.assertIn("fixed checkpoint quality", panel.quality_hint_label.text())


if __name__ == "__main__":
    unittest.main()
