import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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

from src.core.pipeline import MusicToMidiPipeline
from src.models.data_models import BeatInfo, Config, NoteEvent


class TestRestoredProcessingModes(unittest.TestCase):
    def test_restored_backend_modules_are_importable(self):
        for module_name in (
            "src.core.aria_amt_transcriber",
            "src.core.bytedance_piano_transcriber",
            "src.core.transkun_transcriber",
            "src.core.multi_stem_separator",
        ):
            with self.subTest(module_name=module_name):
                self.assertIsNotNone(importlib.util.find_spec(module_name))

    def test_pipeline_constructs_restored_transcribers(self):
        pipeline = MusicToMidiPipeline(Config())

        self.assertTrue(hasattr(pipeline, "aria_amt_transcriber"))
        self.assertTrue(hasattr(pipeline, "bytedance_piano_transcriber"))
        self.assertTrue(hasattr(pipeline, "transkun_transcriber"))

    def test_restored_modes_dispatch_to_their_specific_paths(self):
        cases = (
            ("six_stem_split", "_process_six_stem_split"),
            ("piano_transkun", "_process_piano_transkun"),
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
        pipeline = MusicToMidiPipeline(Config(transcription_backend="aria_amt"))
        pipeline.aria_amt_transcriber.is_available = lambda: False

        with self.assertRaisesRegex(RuntimeError, "Aria-AMT"):
            pipeline._maybe_transcribe_piano_stem_with_aria("piano.wav", "output")

    def test_aria_preference_uses_yourmt3_multi_backend_when_saved_miros_is_stale(self):
        config = Config(transcription_backend="aria_amt", multi_instrument_model="miros")
        pipeline = MusicToMidiPipeline(config)

        self.assertIs(pipeline._get_multi_instrument_transcriber(), pipeline.yourmt3_transcriber)
        self.assertEqual(pipeline._get_multi_instrument_label(), "YourMT3+")

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
    def test_vocal_split_optionally_outputs_merged_midi(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            audio_path.write_bytes(b"audio")
            out_dir = tmp_path / "out"

            class FakeVocalSeparator:
                def __init__(self, *_args, **_kwargs):
                    pass

                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def is_model_available():
                    return True

                def set_cancel_check(self, _cancel_check):
                    return None

                def separate(self, audio_path, output_dir, progress_callback=None):
                    if progress_callback:
                        progress_callback(1.0, "ok")
                    base = Path(output_dir)
                    vocals = base / f"{Path(audio_path).stem}_vocals.wav"
                    no_vocals = base / f"{Path(audio_path).stem}_accompaniment.wav"
                    vocals.parent.mkdir(parents=True, exist_ok=True)
                    vocals.write_bytes(b"wav")
                    no_vocals.write_bytes(b"wav")
                    return {"vocals": str(vocals), "no_vocals": str(no_vocals)}

            class FakeTranscriber:
                def transcribe_precise(self, audio_path, quality, progress_callback=None):
                    if progress_callback:
                        progress_callback(1.0, "done")
                    return ({0: [NoteEvent(pitch=60, start_time=0.0, end_time=0.4)]}, {})

                def unload_model(self):
                    return None

            class FakeBeatDetector:
                def detect(self, _audio_path):
                    return BeatInfo(bpm=120.0)

            class FakeMidiGenerator:
                def generate_from_precise_instruments_v2(
                    self, instrument_notes, drum_notes, tempo, output_path, quality
                ):
                    path = Path(output_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"midi")
                    return str(path)

            merged_calls = []

            def fake_merge(stem_paths, output_path, _tempo):
                merged_calls.append(set(stem_paths.keys()))
                merged_path = Path(output_path)
                merged_path.parent.mkdir(parents=True, exist_ok=True)
                merged_path.write_bytes(b"merged")
                return str(merged_path)

            config = Config()
            config.processing_mode = "vocal_split"
            config.vocal_split_merge_midi = True
            pipeline = MusicToMidiPipeline(config)
            pipeline.yourmt3_transcriber = FakeTranscriber()
            pipeline.beat_detector = FakeBeatDetector()
            pipeline.midi_generator = FakeMidiGenerator()
            pipeline._merge_stem_midis = fake_merge

            with patch("src.core.vocal_separator.VocalSeparator", FakeVocalSeparator), patch(
                "src.core.pipeline.YourMT3Transcriber.is_available", return_value=True
            ):
                result = pipeline._process_vocal_split(str(audio_path), str(out_dir))

            self.assertEqual(result.merged_midi_path, result.midi_path)
            self.assertTrue(Path(result.midi_path).name.endswith("_vocal_accompaniment_merged.mid"))
            self.assertEqual(merged_calls, [{"accompaniment", "vocal"}])
            self.assertTrue(Path(result.accompaniment_midi_path).exists())
            self.assertTrue(Path(result.vocal_midi_path).exists())


if __name__ == "__main__":
    unittest.main()
