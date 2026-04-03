import unittest

from src.models.data_models import Config, QualityBehavior


class ConfigBackendSemanticsTests(unittest.TestCase):
    def test_default_backend_prefers_aria_with_yourmt3_fallback(self):
        config = Config()

        self.assertEqual(config.transcription_backend, "aria_amt")
        self.assertEqual(config.multi_instrument_model, "yourmt3")

    def test_from_dict_keeps_legacy_multi_model_for_compatibility(self):
        config = Config.from_dict({"multi_instrument_model": "miros"})

        self.assertEqual(config.transcription_backend, "miros")
        self.assertEqual(config.multi_instrument_model, "miros")

    def test_from_dict_preserves_explicit_aria_backend_with_miros_fallback(self):
        config = Config.from_dict(
            {
                "transcription_backend": "aria_amt",
                "multi_instrument_model": "miros",
            }
        )

        self.assertEqual(config.transcription_backend, "aria_amt")
        self.assertEqual(config.multi_instrument_model, "miros")

    def test_invalid_backend_falls_back_to_aria_and_yourmt3(self):
        config = Config(transcription_backend="unknown", multi_instrument_model="unknown")

        self.assertEqual(config.transcription_backend, "aria_amt")
        self.assertEqual(config.multi_instrument_model, "yourmt3")

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

    def test_quality_behavior_is_partial_for_six_stem_aria_with_yourmt3_fallback(self):
        config = Config(
            processing_mode="six_stem_split",
            transcription_backend="aria_amt",
            multi_instrument_model="yourmt3",
        )

        self.assertEqual(config.get_quality_behavior(), QualityBehavior.PARTIAL)

    def test_quality_behavior_is_fixed_for_six_stem_aria_with_miros_fallback(self):
        config = Config(
            processing_mode="six_stem_split",
            transcription_backend="aria_amt",
            multi_instrument_model="miros",
        )

        self.assertEqual(config.get_quality_behavior(), QualityBehavior.FIXED)

    def test_quality_behavior_is_fixed_for_dedicated_piano_modes(self):
        transkun = Config(processing_mode="piano_transkun")
        aria = Config(processing_mode="piano_aria_amt")

        self.assertEqual(transkun.get_quality_behavior(), QualityBehavior.FIXED)
        self.assertEqual(aria.get_quality_behavior(), QualityBehavior.FIXED)


if __name__ == "__main__":
    unittest.main()
