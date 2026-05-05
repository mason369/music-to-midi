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
from src.models.data_models import Config


class TestBackendSelectorUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])
        set_language("en_US")

    def setUp(self):
        set_language("en_US")

    def test_track_panel_lists_restored_modes_and_backends(self):
        panel = TrackPanel()

        modes = [panel.mode_combo.itemData(index) for index in range(panel.mode_combo.count())]
        backends = [panel.model_combo.itemData(index) for index in range(panel.model_combo.count())]

        self.assertEqual(
            modes,
            [
                "smart",
                "vocal_split",
                "six_stem_split",
                "piano_transkun",
                "piano_aria_amt",
                "piano_bytedance_pedal",
            ],
        )
        self.assertEqual(backends, ["yourmt3", "miros"])

    def test_smart_mode_defaults_to_yourmt3_without_aria_amt_engine_display(self):
        panel = TrackPanel()

        self.assertEqual(panel.get_processing_mode(), "smart")
        self.assertEqual(panel.get_transcription_backend(), "yourmt3")
        self.assertEqual(panel.get_multi_instrument_model(), "yourmt3")
        self.assertEqual(panel._model_label.text(), "Multi-Instrument Backend:")
        self.assertEqual(
            [panel.model_combo.itemData(index) for index in range(panel.model_combo.count())],
            ["yourmt3", "miros"],
        )
        self.assertNotIn("Aria-AMT", [panel.model_combo.itemText(index) for index in range(panel.model_combo.count())])

    def test_smart_mode_rejects_aria_amt_as_displayed_full_mix_engine(self):
        panel = TrackPanel()

        panel.set_processing_mode("smart")
        panel.set_transcription_backend("aria_amt")

        self.assertEqual(panel.get_transcription_backend(), "yourmt3")
        self.assertEqual(
            [panel.model_combo.itemData(index) for index in range(panel.model_combo.count())],
            ["yourmt3", "miros"],
        )

    def test_restored_mode_and_backend_are_preserved(self):
        panel = TrackPanel()

        panel.set_processing_mode("six_stem_split")
        panel.set_transcription_backend("aria_amt")

        self.assertEqual(panel.get_processing_mode(), "six_stem_split")
        self.assertEqual(panel.get_transcription_backend(), "aria_amt")

    def test_processing_controls_disable_mode_specific_checkboxes(self):
        panel = TrackPanel()
        panel.set_processing_mode("vocal_split")
        panel.set_processing_controls_enabled(False)
        self.assertFalse(panel._vocal_split_merge_check.isEnabled())

    def test_main_window_does_not_display_aria_as_smart_full_mix_engine(self):
        config = Config(transcription_backend="aria_amt", multi_instrument_model="miros")

        with mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None):
            window = MainWindow(config)

        self.assertEqual(window.track_panel.get_transcription_backend(), "yourmt3")
        self.assertEqual(window.track_panel.get_multi_instrument_model(), "yourmt3")
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

    def test_track_panel_lists_official_yourmt3_checkpoint_modes(self):
        panel = TrackPanel()

        modes = [
            panel.yourmt3_model_combo.itemData(index)
            for index in range(panel.yourmt3_model_combo.count())
        ]

        self.assertEqual(
            modes,
            [
                "ymt3_plus",
                "yptf_single_nops",
                "yptf_multi_ps",
                "yptf_moe_multi_nops",
                "yptf_moe_multi_ps",
            ],
        )

    def test_track_panel_shows_selected_yourmt3_model_description(self):
        panel = TrackPanel()

        panel.set_yourmt3_model("yptf_moe_multi_nops")

        self.assertEqual(panel.get_yourmt3_model(), "yptf_moe_multi_nops")
        self.assertIn("official Hugging Face Space", panel.yourmt3_model_hint_label.text())
        self.assertIn("MoE", panel.yourmt3_model_hint_label.text())

    def test_track_panel_hides_yourmt3_model_picker_for_miros(self):
        panel = TrackPanel()
        panel.set_processing_mode("smart")
        panel.set_transcription_backend("miros")

        self.assertTrue(panel._yourmt3_model_row.isHidden())
        self.assertTrue(panel.yourmt3_model_hint_label.isHidden())

    def test_track_panel_hides_midi_track_mode_for_non_yourmt3_backend(self):
        panel = TrackPanel()
        panel.set_processing_mode("smart")
        panel.set_transcription_backend("miros")
        panel.set_midi_track_mode("single_track")

        self.assertTrue(panel._midi_track_mode_row.isHidden())
        self.assertTrue(panel.yourmt3_arch_hint_label.isHidden())
        self.assertEqual(panel.get_midi_track_mode(), "multi_track")

    def test_restored_dedicated_piano_mode_remains_selectable(self):
        panel = TrackPanel()
        panel.set_processing_mode("piano_bytedance_pedal")

        self.assertEqual(panel.get_processing_mode(), "piano_bytedance_pedal")

    def test_bytedance_pedal_mode_explains_pedal_scope_when_selected(self):
        panel = TrackPanel()
        panel.set_processing_mode("piano_bytedance_pedal")

        self.assertIn("Pedal-Aware", panel.mode_desc_label.text())
        self.assertIn("CC64", panel.hint_label.text())
        self.assertIn("Dedicated piano modes", panel.model_hint_label.text())


if __name__ == "__main__":
    unittest.main()
