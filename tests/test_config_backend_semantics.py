import unittest

from src.models.data_models import Config, QualityBehavior


class ConfigBackendSemanticsTests(unittest.TestCase):
    def test_default_backend_uses_yourmt3_directly(self):
        config = Config()

        self.assertEqual(config.transcription_backend, "yourmt3")
        self.assertEqual(config.multi_instrument_model, "yourmt3")

    def test_from_dict_keeps_legacy_multi_model_for_compatibility(self):
        config = Config.from_dict({"multi_instrument_model": "miros"})

        self.assertEqual(config.transcription_backend, "miros")
        self.assertEqual(config.multi_instrument_model, "miros")

    def test_from_dict_replaces_removed_aria_backend_with_saved_multi_backend(self):
        config = Config.from_dict(
            {
                "transcription_backend": "aria_amt",
                "multi_instrument_model": "miros",
            }
        )

        self.assertEqual(config.transcription_backend, "miros")
        self.assertEqual(config.multi_instrument_model, "miros")

    def test_invalid_backend_falls_back_to_yourmt3(self):
        config = Config(transcription_backend="unknown", multi_instrument_model="unknown")

        self.assertEqual(config.transcription_backend, "yourmt3")
        self.assertEqual(config.multi_instrument_model, "yourmt3")

    def test_midi_track_mode_defaults_to_multi_track(self):
        config = Config()

        self.assertEqual(config.midi_track_mode, "multi_track")

    def test_invalid_midi_track_mode_falls_back_to_multi_track(self):
        config = Config(midi_track_mode="official_single")

        self.assertEqual(config.midi_track_mode, "multi_track")

    def test_from_dict_preserves_single_midi_track_mode(self):
        config = Config.from_dict({"midi_track_mode": "single_track"})

        self.assertEqual(config.midi_track_mode, "single_track")
        self.assertEqual(config.to_dict()["midi_track_mode"], "single_track")

    def test_direct_multi_backend_overrides_stale_fallback_value(self):
        config = Config(transcription_backend="yourmt3", multi_instrument_model="miros")

        self.assertEqual(config.transcription_backend, "yourmt3")
        self.assertEqual(config.multi_instrument_model, "yourmt3")

    def test_quality_behavior_is_configurable_for_yourmt3_general_modes(self):
        config = Config(
            processing_mode="smart",
            transcription_backend="yourmt3",
            multi_instrument_model="miros",
        )

        self.assertEqual(config.get_quality_behavior(), QualityBehavior.CONFIGURABLE)

    def test_removed_processing_modes_fall_back_to_smart(self):
        for removed_mode in ("six_stem_split", "piano_transkun", "piano_aria_amt"):
            with self.subTest(removed_mode=removed_mode):
                config = Config(processing_mode=removed_mode)

                self.assertEqual(config.processing_mode, "smart")
                self.assertEqual(config.get_quality_behavior(), QualityBehavior.CONFIGURABLE)


if __name__ == "__main__":
    unittest.main()
