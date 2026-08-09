import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.pipeline import MusicToMidiPipeline
from src.models.data_models import Config


class TestRestoredProcessingModes(unittest.TestCase):
    def test_restored_backend_modules_are_importable(self):
        for module_name in (
            "src.core.aria_amt_transcriber",
            "src.core.bytedance_piano_transcriber",
            "src.core.transkun_transcriber",
            "src.core.transkun_v2_aug_transcriber",
            "src.core.multi_stem_separator",
        ):
            with self.subTest(module_name=module_name):
                self.assertIsNotNone(importlib.util.find_spec(module_name))

    def test_pipeline_constructs_restored_transcribers(self):
        pipeline = MusicToMidiPipeline(Config())

        self.assertTrue(hasattr(pipeline, "aria_amt_transcriber"))
        self.assertTrue(hasattr(pipeline, "bytedance_piano_transcriber"))
        self.assertTrue(hasattr(pipeline, "transkun_transcriber"))
        self.assertTrue(hasattr(pipeline, "transkun_v2_aug_transcriber"))

    def test_restored_modes_dispatch_to_their_specific_paths(self):
        cases = (
            ("six_stem_split", "_process_six_stem_split"),
            ("piano_transkun", "_process_piano_transkun"),
            ("piano_transkun_v2_aug", "_process_piano_transkun_v2_aug"),
            ("piano_aria_amt", "_process_piano_aria_amt"),
            ("piano_bytedance_pedal", "_process_piano_bytedance_pedal"),
        )
        for restored_mode, method_name in cases:
            with self.subTest(restored_mode=restored_mode):
                config = Config()
                config.processing_mode = restored_mode
                pipeline = MusicToMidiPipeline(config)

                calls = []

                def fake_handler(audio_path, output_dir):
                    calls.append((audio_path, output_dir))
                    return f"{restored_mode}-result"

                setattr(pipeline, method_name, fake_handler)
                result = pipeline.process("input.wav", "output")

                self.assertEqual(result, f"{restored_mode}-result")
                self.assertEqual(calls, [("input.wav", "output")])

    def test_aria_piano_stem_preference_fails_when_backend_is_unavailable(self):
        with self.assertRaisesRegex(ValueError, "requires transcription_backend"):
            Config(
                processing_mode="six_stem_split",
                transcription_backend="aria_amt",
            )

    def test_aria_preference_uses_yourmt3_multi_backend_when_saved_miros_is_stale(self):
        config = Config(
            processing_mode="six_stem_split",
            transcription_backend="miros",
            multi_instrument_model="miros",
        )
        pipeline = MusicToMidiPipeline(config)

        self.assertIs(pipeline._get_multi_instrument_transcriber(), pipeline.miros_transcriber)
        self.assertEqual(pipeline._get_multi_instrument_label(), "MIROS")

    def test_pipeline_revalidates_mutated_mode_before_audio_conversion(self):
        config = Config()
        pipeline = MusicToMidiPipeline(config)
        config.processing_mode = "not_a_mode"

        with patch.object(
            pipeline,
            "_ensure_wav",
            side_effect=AssertionError("invalid config must stop before conversion"),
        ):
            with self.assertRaisesRegex(ValueError, "processing_mode"):
                pipeline.process("input.mp3", "output")

    def test_six_stem_experimental_vocal_harmony_branch_is_removed(self):
        pipeline = MusicToMidiPipeline(Config(processing_mode="six_stem_split"))

        self.assertFalse(hasattr(pipeline, "_apply_vocal_harmony_split"))
        self.assertIsNone(importlib.util.find_spec("src.core.vocal_harmony_separator"))

    def test_backend_error_message_does_not_duplicate_existing_prefix(self):
        message = MusicToMidiPipeline._format_backend_error(
            "MIROS",
            "转写失败",
            RuntimeError("MIROS 转写失败:\nworker boom"),
        )

        self.assertEqual(message, "MIROS 转写失败:\nworker boom")

    def test_backend_error_message_adds_prefix_to_plain_errors(self):
        message = MusicToMidiPipeline._format_backend_error(
            "MIROS",
            "转写失败",
            RuntimeError("worker boom"),
        )

        self.assertEqual(message, "MIROS 转写失败: worker boom")


class TestVocalSplitMode(unittest.TestCase):
    def test_vocal_split_outputs_only_verified_wavs_even_with_legacy_midi_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"real input")
            output_dir = root / "out"

            class FakeVocalSeparator:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def is_model_available():
                    return True

                def __init__(self, *_args, **_kwargs):
                    self.cancel_check = None

                def set_cancel_check(self, cancel_check):
                    self.cancel_check = cancel_check

                def separate(self, audio_path, output_dir, progress_callback=None):
                    self.assert_not_cancelled()
                    base = Path(output_dir)
                    base.mkdir(parents=True, exist_ok=True)
                    vocals = base / f"{Path(audio_path).stem}_vocals.wav"
                    accompaniment = base / f"{Path(audio_path).stem}_accompaniment.wav"
                    vocals.write_bytes(b"vocals")
                    accompaniment.write_bytes(b"accompaniment")
                    if progress_callback is not None:
                        progress_callback(1.0, "separated")
                    return {
                        "vocals": str(vocals),
                        "accompaniment": str(accompaniment),
                    }

                def assert_not_cancelled(self):
                    if self.cancel_check is None or self.cancel_check():
                        raise AssertionError("cancel callback was not installed correctly")

            def unexpected_midi_work(*_args, **_kwargs):
                raise AssertionError(
                    "vocal split must stop before beat detection or MIDI transcription"
                )

            config = Config(
                processing_mode="vocal_split",
                vocal_split_merge_midi=True,
                save_separated_tracks=False,
            )
            pipeline = MusicToMidiPipeline(config)
            for transcriber in (
                pipeline.yourmt3_transcriber,
                pipeline.miros_transcriber,
                pipeline.muscriptor_transcriber,
            ):
                transcriber.transcribe_to_midi = unexpected_midi_work
            pipeline._detect_beat_or_raise = unexpected_midi_work
            pipeline._merge_stem_midis = unexpected_midi_work

            with patch("src.core.vocal_separator.VocalSeparator", FakeVocalSeparator):
                result = pipeline.process(str(audio_path), str(output_dir))

            self.assertEqual(result.midi_path, "")
            self.assertEqual(result.total_notes, 0)
            self.assertIsNone(result.beat_info)
            self.assertIsNone(result.vocal_midi_path)
            self.assertIsNone(result.accompaniment_midi_path)
            self.assertIsNone(result.merged_midi_path)
            self.assertEqual(set(result.separated_audio or {}), {"vocals", "accompaniment"})
            self.assertTrue(
                all(Path(path).is_file() for path in (result.separated_audio or {}).values())
            )
            self.assertEqual(list(output_dir.rglob("*.mid")), [])

    def test_vocal_split_rejects_legacy_no_vocals_without_canonical_accompaniment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"audio")
            vocals_path = root / "vocals.wav"
            no_vocals_path = root / "no_vocals.wav"
            vocals_path.write_bytes(b"wav")
            no_vocals_path.write_bytes(b"wav")

            class LegacySeparator:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def is_model_available():
                    return True

                def __init__(self, *_args, **_kwargs):
                    pass

                def set_cancel_check(self, _cancel_check):
                    return None

                def separate(self, **_kwargs):
                    return {
                        "vocals": str(vocals_path),
                        "no_vocals": str(no_vocals_path),
                    }

            pipeline = MusicToMidiPipeline(Config(processing_mode="vocal_split"))
            with patch("src.core.vocal_separator.VocalSeparator", LegacySeparator):
                with self.assertRaisesRegex(RuntimeError, "accompaniment"):
                    pipeline._process_vocal_split(str(audio_path), str(root / "out"))


if __name__ == "__main__":
    unittest.main()
