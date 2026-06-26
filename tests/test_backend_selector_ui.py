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
        self.assertEqual(panel.get_yourmt3_model(), "yptf_moe_multi_nops")
        self.assertEqual(panel.mode_combo.currentText(), "Multi-Instrument MIDI")
        self.assertNotIn("YourMT3", panel.mode_combo.currentText())
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

    def test_six_stem_mode_does_not_expose_removed_selection_controls(self):
        panel = TrackPanel()

        panel.set_processing_mode("six_stem_split")

        self.assertFalse(hasattr(panel, "_six_stem_only_selected_check"))
        self.assertFalse(hasattr(panel, "_six_stem_vocal_harmony_check"))
        self.assertFalse(hasattr(panel, "get_selected_six_stem_targets"))
        self.assertFalse(hasattr(panel, "get_six_stem_vocal_harmony"))
        self.assertIn("six stem MIDI files", panel.hint_label.text())
        self.assertIn("all_stems merged MIDI", panel.hint_label.text())
        self.assertNotIn("selected", panel.hint_label.text().lower())
        self.assertNotIn("lead", panel.hint_label.text().lower())
        self.assertNotIn("harmony", panel.hint_label.text().lower())

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

    def test_track_panel_does_not_expose_non_official_midi_track_layout(self):
        panel = TrackPanel()
        panel.set_processing_mode("smart")
        panel.set_transcription_backend("yourmt3")

        self.assertEqual(panel.get_midi_track_mode(), "multi_track")
        self.assertFalse(hasattr(panel, "_midi_track_mode_row"))
        self.assertFalse(hasattr(panel, "midi_track_mode_combo"))
        self.assertFalse(hasattr(panel, "yourmt3_arch_hint_label"))

        panel.set_midi_track_mode("single_track")
        self.assertEqual(panel.get_midi_track_mode(), "multi_track")

    def test_vocal_split_hides_yourmt3_checkpoint_and_track_layout(self):
        panel = TrackPanel()
        panel.set_processing_mode("vocal_split")
        panel.set_transcription_backend("yourmt3")
        panel.set_midi_track_mode("single_track")

        self.assertEqual(panel._model_label.text(), "Post-Split MIDI Backend:")
        self.assertEqual(panel.model_combo.currentText(), "YourMT3+ (MIDI only)")
        self.assertTrue(panel._yourmt3_model_row.isHidden())
        self.assertTrue(panel.yourmt3_model_card.isHidden())
        self.assertFalse(hasattr(panel, "_midi_track_mode_row"))
        self.assertFalse(hasattr(panel, "yourmt3_arch_hint_label"))
        self.assertEqual(panel.get_midi_track_mode(), "multi_track")
        self.assertIn("RoFormer vocal_rvc/karaoke", panel.mode_desc_label.text())
        self.assertIn("does not use YourMT3+ / MIROS", panel.hint_label.text())
        self.assertIn("YourMT3+ is only the downstream MIDI backend", panel.model_hint_label.text())

    def test_track_panel_explains_official_yourmt3_single_multi_in_model_card(self):
        panel = TrackPanel()
        panel.set_processing_mode("smart")
        panel.set_transcription_backend("yourmt3")

        panel.set_yourmt3_model("yptf_multi_ps")
        text = panel.yourmt3_model_hint_label.text()

        self.assertFalse(panel.yourmt3_model_card.isHidden())
        self.assertIn("Perceiver-TF", text)
        self.assertIn("multi-t5", text)
        self.assertIn("multi-channel", text)
        self.assertNotIn("Track Layout", text)
        self.assertNotIn("single track", text)

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
        self.assertFalse(panel.yourmt3_model_card.isHidden())
        self.assertIn("YPTF.MoE+Multi (noPS)", panel.yourmt3_model_title_label.text())
        self.assertIn("Checkpoint: YPTF.MoE+Multi (noPS)", panel.yourmt3_model_hint_label.text())
        self.assertIn("Best for / traits:", panel.yourmt3_model_hint_label.text())
        self.assertIn("official Hugging Face Space", panel.yourmt3_model_hint_label.text())
        self.assertIn("MoE", panel.yourmt3_model_hint_label.text())

    def test_track_panel_localizes_yourmt3_model_feature_card(self):
        set_language("zh_CN")
        panel = TrackPanel()

        panel.set_yourmt3_model("yptf_multi_ps")
        text = panel.yourmt3_model_hint_label.text()

        self.assertIn("检查点：YPTF+Multi (PS)", text)
        self.assertIn("适合/特点：", text)
        self.assertIn("多通道解码", text)
        self.assertIn("使用音高偏移增强", text)
        self.assertNotIn("Checkpoint:", text)
        self.assertNotIn("Features:", text)
        self.assertNotIn("multi-channel decoding", text)
        self.assertNotIn("pitch-shift augmentation", text)

    def test_track_panel_hides_yourmt3_model_picker_for_miros(self):
        panel = TrackPanel()
        panel.set_processing_mode("smart")
        panel.set_transcription_backend("miros")

        self.assertTrue(panel._yourmt3_model_row.isHidden())
        self.assertTrue(panel.yourmt3_model_card.isHidden())
        self.assertTrue(panel.yourmt3_model_title_label.isHidden())
        self.assertTrue(panel.yourmt3_model_hint_label.isHidden())

    def test_miros_backend_displays_independent_sota_route_copy(self):
        panel = TrackPanel()
        panel.set_processing_mode("smart")
        panel.set_transcription_backend("miros")

        visible_copy = "\n".join(
            [
                panel.mode_combo.currentText(),
                panel.mode_desc_label.text(),
                panel.hint_label.text(),
                panel.model_hint_label.text(),
            ]
        )

        self.assertIn("MIROS", panel.mode_desc_label.text())
        self.assertIn("MusicFM multi-decoder", visible_copy)
        self.assertIn("2025 AI4Musician", visible_copy)
        self.assertNotIn("YourMT3", visible_copy)

    def test_track_panel_hides_midi_track_mode_for_non_yourmt3_backend(self):
        panel = TrackPanel()
        panel.set_processing_mode("smart")
        panel.set_transcription_backend("miros")
        panel.set_midi_track_mode("single_track")

        self.assertFalse(hasattr(panel, "_midi_track_mode_row"))
        self.assertFalse(hasattr(panel, "yourmt3_arch_hint_label"))
        self.assertEqual(panel.get_midi_track_mode(), "multi_track")

    def test_six_stem_mode_explains_telknet_separation_and_keeps_midi_extension(self):
        panel = TrackPanel()
        panel.set_processing_mode("six_stem_split")
        panel.set_transcription_backend("yourmt3")
        panel.set_midi_track_mode("single_track")

        self.assertEqual(panel._model_label.text(), "Stem MIDI Backend:")
        self.assertEqual(panel.model_combo.currentText(), "YourMT3+ (MIDI only)")
        self.assertTrue(panel._yourmt3_model_row.isHidden())
        self.assertTrue(panel.yourmt3_model_card.isHidden())
        self.assertFalse(hasattr(panel, "_midi_track_mode_row"))
        self.assertFalse(hasattr(panel, "yourmt3_arch_hint_label"))
        self.assertEqual(panel.get_midi_track_mode(), "multi_track")
        self.assertIn("BS-RoFormer SW", panel.mode_desc_label.text())
        self.assertIn("TelkNet/MVSep-aligned", panel.hint_label.text())
        self.assertIn("MIDI extension", panel.hint_label.text())
        self.assertIn("YourMT3+ is only the downstream Stem MIDI backend", panel.model_hint_label.text())

    def test_restored_dedicated_piano_mode_remains_selectable(self):
        panel = TrackPanel()
        panel.set_processing_mode("piano_bytedance_pedal")

        self.assertEqual(panel.get_processing_mode(), "piano_bytedance_pedal")

    def test_bytedance_pedal_mode_explains_pedal_scope_when_selected(self):
        panel = TrackPanel()
        panel.set_processing_mode("piano_bytedance_pedal")

        self.assertFalse(panel.model_info_card.isHidden())
        self.assertTrue(panel.yourmt3_model_card.isHidden())
        self.assertIn("Pedal-Aware", panel.mode_desc_label.text())
        self.assertIn("CC64", panel.hint_label.text())
        self.assertIn("Dedicated piano modes", panel.model_hint_label.text())

    def test_packaged_model_copy_does_not_prompt_download_in_track_panel(self):
        set_language("zh_CN")
        panel = TrackPanel()

        panel.set_processing_mode("piano_bytedance_pedal")
        self.assertIn("模型目录可用", panel.hint_label.text())
        self.assertNotIn("下载", panel.hint_label.text())

        panel.set_processing_mode("piano_aria_amt")
        self.assertIn("模型目录可用", panel.hint_label.text())
        self.assertNotIn("下载", panel.hint_label.text())

        panel.set_processing_mode("smart")
        self.assertIn("模型目录可用", panel.yourmt3_model_combo.toolTip())
        self.assertNotIn("下载", panel.yourmt3_model_combo.toolTip())


if __name__ == "__main__":
    unittest.main()
