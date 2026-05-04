import unittest

from src.models.data_models import Config, QualityBehavior


class ConfigBackendSemanticsTests(unittest.TestCase):
    def test_default_backend_uses_yourmt3_multi_instrument_engine(self):
        config = Config()

        self.assertEqual(config.transcription_backend, "yourmt3")
        self.assertEqual(config.multi_instrument_model, "yourmt3")

    def test_from_dict_keeps_legacy_multi_model_for_compatibility(self):
        config = Config.from_dict({"multi_instrument_model": "miros"})

        self.assertEqual(config.transcription_backend, "miros")
        self.assertEqual(config.multi_instrument_model, "miros")

    def test_from_dict_resets_stale_miros_multi_backend_when_aria_is_preferred(self):
        config = Config.from_dict(
            {
                "transcription_backend": "aria_amt",
                "multi_instrument_model": "miros",
            }
        )

        self.assertEqual(config.transcription_backend, "aria_amt")
        self.assertEqual(config.multi_instrument_model, "yourmt3")

    def test_explicit_miros_backend_still_selects_miros(self):
        config = Config.from_dict(
            {
                "transcription_backend": "miros",
                "multi_instrument_model": "miros",
            }
        )

        self.assertEqual(config.transcription_backend, "miros")
        self.assertEqual(config.multi_instrument_model, "miros")

    def test_invalid_backend_raises_validation_error(self):
        with self.assertRaisesRegex(ValueError, "transcription_backend"):
            Config(transcription_backend="unknown", multi_instrument_model="unknown")

    def test_invalid_processing_mode_raises_validation_error(self):
        with self.assertRaisesRegex(ValueError, "processing_mode"):
            Config(processing_mode="unknown")

    def test_legacy_piano_processing_mode_maps_to_smart(self):
        config = Config(processing_mode="piano")

        self.assertEqual(config.processing_mode, "smart")

    def test_midi_track_mode_defaults_to_multi_track(self):
        config = Config()

        self.assertEqual(config.midi_track_mode, "multi_track")
        self.assertEqual(config.yourmt3_model, "yptf_moe_multi_ps")

    def test_invalid_yourmt3_model_raises_validation_error(self):
        with self.assertRaisesRegex(ValueError, "yourmt3_model"):
            Config(yourmt3_model="unknown_yourmt3_checkpoint")

    def test_from_dict_preserves_yourmt3_model_choice(self):
        config = Config.from_dict({"yourmt3_model": "yptf_moe_multi_nops"})

        self.assertEqual(config.yourmt3_model, "yptf_moe_multi_nops")
        self.assertEqual(config.to_dict()["yourmt3_model"], "yptf_moe_multi_nops")

    def test_invalid_midi_track_mode_raises_validation_error(self):
        with self.assertRaisesRegex(ValueError, "midi_track_mode"):
            Config(midi_track_mode="official_single")

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

    def test_restored_processing_modes_are_valid(self):
        for restored_mode in ("six_stem_split", "piano_transkun", "piano_aria_amt"):
            with self.subTest(restored_mode=restored_mode):
                config = Config(processing_mode=restored_mode)

                self.assertEqual(config.processing_mode, restored_mode)
        self.assertEqual(
            Config(processing_mode="piano_transkun").get_quality_behavior(),
            QualityBehavior.FIXED,
        )


if __name__ == "__main__":
    unittest.main()
