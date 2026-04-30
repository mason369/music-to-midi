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


class TestRemovedProcessingModes(unittest.TestCase):
    def test_removed_backend_modules_are_not_importable(self):
        for module_name in (
            "src.core.aria_amt_transcriber",
            "src.core.transkun_transcriber",
            "src.core.multi_stem_separator",
        ):
            with self.subTest(module_name=module_name):
                self.assertIsNone(importlib.util.find_spec(module_name))

    def test_pipeline_no_longer_constructs_removed_transcribers(self):
        pipeline = MusicToMidiPipeline(Config())

        self.assertFalse(hasattr(pipeline, "aria_amt_transcriber"))
        self.assertFalse(hasattr(pipeline, "transkun_transcriber"))

    def test_stale_removed_modes_dispatch_to_smart_path(self):
        for removed_mode in ("six_stem_split", "piano_transkun", "piano_aria_amt"):
            with self.subTest(removed_mode=removed_mode):
                config = Config()
                config.processing_mode = removed_mode
                pipeline = MusicToMidiPipeline(config)

                calls = []

                def fake_smart(audio_path, output_dir):
                    calls.append((audio_path, output_dir))
                    return "smart-result"

                pipeline._process_smart = fake_smart
                result = pipeline.process("input.wav", "output")

                self.assertEqual(result, "smart-result")
                self.assertEqual(calls, [("input.wav", "output")])


class TestVocalSplitMode(unittest.TestCase):
    def test_vocal_split_optionally_outputs_merged_midi(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            audio_path.write_bytes(b"audio")
            out_dir = tmp_path / "out"

            class FakeVocalSeparator:
                @staticmethod
                def is_available():
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
