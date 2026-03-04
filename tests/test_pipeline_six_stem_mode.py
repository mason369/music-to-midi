import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.pipeline import MusicToMidiPipeline
from src.models.data_models import BeatInfo, Config, NoteEvent, ProcessingResult


class TestPipelineSixStemMode(unittest.TestCase):
    def test_process_dispatches_to_six_stem_branch(self):
        config = Config()
        config.processing_mode = "six_stem_split"
        pipeline = MusicToMidiPipeline(config)

        sentinel = ProcessingResult(midi_path="merged.mid")

        def fake_six(audio_path, output_dir):
            self.assertEqual(audio_path, "input.wav")
            self.assertEqual(output_dir, "output")
            return sentinel

        def fail_other(*_args, **_kwargs):
            raise AssertionError("wrong branch dispatched")

        pipeline._process_six_stem_split = fake_six
        pipeline._process_smart = fail_other
        pipeline._process_vocal_split = fail_other

        result = pipeline.process("input.wav", "output")
        self.assertIs(result, sentinel)

    def test_process_dispatches_to_piano_aria_branch(self):
        config = Config()
        config.processing_mode = "piano_aria_amt"
        pipeline = MusicToMidiPipeline(config)

        sentinel = ProcessingResult(midi_path="piano.mid")

        def fake_piano(audio_path, output_dir):
            self.assertEqual(audio_path, "input.wav")
            self.assertEqual(output_dir, "output")
            return sentinel

        def fail_other(*_args, **_kwargs):
            raise AssertionError("wrong branch dispatched")

        pipeline._process_piano_aria_amt = fake_piano
        pipeline._process_six_stem_split = fail_other
        pipeline._process_smart = fail_other
        pipeline._process_vocal_split = fail_other

        result = pipeline.process("input.wav", "output")
        self.assertIs(result, sentinel)

    def test_processing_result_has_six_stem_fields(self):
        result = ProcessingResult(midi_path="merged.mid")
        self.assertTrue(hasattr(result, "stem_midi_paths"))
        self.assertTrue(hasattr(result, "merged_midi_path"))

    def test_six_stem_processing_outputs_stem_midis_and_merged_midi(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            audio_path.write_bytes(b"audio")
            out_dir = tmp_path / "out"

            class FakeSixStemSeparator:
                def __init__(self):
                    pass

                @staticmethod
                def is_available():
                    return True

                def separate(self, audio_path, output_dir, progress_callback=None):
                    if progress_callback:
                        progress_callback(1.0, "ok")
                    base = Path(output_dir)
                    result = {}
                    for stem in ("bass", "drums", "guitar", "piano", "vocals", "other"):
                        stem_path = base / f"{Path(audio_path).stem}_{stem}.wav"
                        stem_path.parent.mkdir(parents=True, exist_ok=True)
                        stem_path.write_bytes(b"wav")
                        result[stem] = str(stem_path)
                    return result

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

            def fake_merge(_stem_paths, output_path, _tempo):
                merged_path = Path(output_path)
                merged_path.parent.mkdir(parents=True, exist_ok=True)
                merged_path.write_bytes(b"merged")
                return str(merged_path)

            config = Config()
            config.processing_mode = "six_stem_split"
            pipeline = MusicToMidiPipeline(config)
            pipeline.yourmt3_transcriber = FakeTranscriber()
            pipeline.beat_detector = FakeBeatDetector()
            pipeline.midi_generator = FakeMidiGenerator()
            pipeline._merge_stem_midis = fake_merge

            with patch("src.core.multi_stem_separator.SixStemSeparator", FakeSixStemSeparator), patch(
                "src.core.pipeline.YourMT3Transcriber.is_available", return_value=True
            ):
                result = pipeline._process_six_stem_split(str(audio_path), str(out_dir))

            self.assertEqual(set(result.stem_midi_paths.keys()), {"bass", "drums", "guitar", "piano", "vocals", "other"})
            self.assertTrue(Path(result.midi_path).exists())
            self.assertEqual(result.merged_midi_path, result.midi_path)

    def test_six_stem_processing_only_transcribes_selected_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            audio_path.write_bytes(b"audio")
            out_dir = tmp_path / "out"

            class FakeSixStemSeparator:
                @staticmethod
                def is_available():
                    return True

                def separate(self, audio_path, output_dir, progress_callback=None):
                    if progress_callback:
                        progress_callback(1.0, "ok")
                    base = Path(output_dir)
                    result = {}
                    for stem in ("bass", "drums", "guitar", "piano", "vocals", "other"):
                        stem_path = base / f"{Path(audio_path).stem}_{stem}.wav"
                        stem_path.parent.mkdir(parents=True, exist_ok=True)
                        stem_path.write_bytes(b"wav")
                        result[stem] = str(stem_path)
                    return result

            transcribed_stems = []

            class FakeTranscriber:
                def transcribe_precise(self, audio_path, quality, progress_callback=None):
                    transcribed_stems.append(Path(audio_path).stem.split("_")[-1])
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

            merged_keys = []

            def fake_merge(stem_paths, output_path, _tempo):
                merged_keys.extend(stem_paths.keys())
                merged_path = Path(output_path)
                merged_path.parent.mkdir(parents=True, exist_ok=True)
                merged_path.write_bytes(b"merged")
                return str(merged_path)

            config = Config()
            config.processing_mode = "six_stem_split"
            config.six_stem_targets = ["drums", "bass"]
            pipeline = MusicToMidiPipeline(config)
            pipeline.yourmt3_transcriber = FakeTranscriber()
            pipeline.beat_detector = FakeBeatDetector()
            pipeline.midi_generator = FakeMidiGenerator()
            pipeline._merge_stem_midis = fake_merge

            with patch("src.core.multi_stem_separator.SixStemSeparator", FakeSixStemSeparator), patch(
                "src.core.pipeline.YourMT3Transcriber.is_available", return_value=True
            ):
                result = pipeline._process_six_stem_split(str(audio_path), str(out_dir))

            self.assertEqual(set(result.stem_midi_paths.keys()), {"bass", "drums"})
            self.assertEqual(set(transcribed_stems), {"bass", "drums"})
            self.assertEqual(set(merged_keys), {"bass", "drums"})
            self.assertTrue(Path(result.midi_path).name.endswith("_selected_stems_merged.mid"))

    def test_six_stem_can_split_vocals_into_lead_and_harmony(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            audio_path.write_bytes(b"audio")
            out_dir = tmp_path / "out"

            class FakeSixStemSeparator:
                @staticmethod
                def is_available():
                    return True

                def separate(self, audio_path, output_dir, progress_callback=None):
                    if progress_callback:
                        progress_callback(1.0, "ok")
                    base = Path(output_dir)
                    result = {}
                    for stem in ("bass", "drums", "guitar", "piano", "vocals", "other"):
                        stem_path = base / f"{Path(audio_path).stem}_{stem}.wav"
                        stem_path.parent.mkdir(parents=True, exist_ok=True)
                        stem_path.write_bytes(b"wav")
                        result[stem] = str(stem_path)
                    return result

            class FakeHarmonySeparator:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def is_model_available():
                    return True

                def separate(self, audio_path, output_dir, progress_callback=None):
                    if progress_callback:
                        progress_callback(1.0, "ok")
                    base = Path(output_dir)
                    lead = base / f"{Path(audio_path).stem}_lead_vocals.wav"
                    harmony = base / f"{Path(audio_path).stem}_harmony_vocals.wav"
                    lead.write_bytes(b"wav")
                    harmony.write_bytes(b"wav")
                    return {
                        "lead_vocals": str(lead),
                        "harmony_vocals": str(harmony),
                    }

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

            def fake_merge(_stem_paths, output_path, _tempo):
                merged_path = Path(output_path)
                merged_path.parent.mkdir(parents=True, exist_ok=True)
                merged_path.write_bytes(b"merged")
                return str(merged_path)

            config = Config()
            config.processing_mode = "six_stem_split"
            config.six_stem_split_vocal_harmony = True
            pipeline = MusicToMidiPipeline(config)
            pipeline.yourmt3_transcriber = FakeTranscriber()
            pipeline.beat_detector = FakeBeatDetector()
            pipeline.midi_generator = FakeMidiGenerator()
            pipeline._merge_stem_midis = fake_merge

            with patch("src.core.multi_stem_separator.SixStemSeparator", FakeSixStemSeparator), patch(
                "src.core.vocal_harmony_separator.VocalHarmonySeparator", FakeHarmonySeparator
            ), patch(
                "src.core.pipeline.YourMT3Transcriber.is_available", return_value=True
            ):
                result = pipeline._process_six_stem_split(str(audio_path), str(out_dir))

            self.assertIn("lead_vocals", result.stem_midi_paths)
            self.assertIn("harmony_vocals", result.stem_midi_paths)
            self.assertNotIn("vocals", result.stem_midi_paths)

    def test_piano_aria_mode_outputs_single_midi(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "solo_piano.wav"
            audio_path.write_bytes(b"audio")
            out_dir = tmp_path / "out"

            class FakeBeatDetector:
                def detect(self, _audio_path):
                    return BeatInfo(bpm=96.0)

            class FakeAriaTranscriber:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def is_model_available():
                    return True

                def transcribe(self, audio_path, output_path, progress_callback=None):
                    if progress_callback:
                        progress_callback(1.0, "done")
                    midi_path = Path(output_path)
                    midi_path.parent.mkdir(parents=True, exist_ok=True)
                    midi_path.write_bytes(b"midi")
                    return str(midi_path)

            config = Config()
            config.processing_mode = "piano_aria_amt"
            pipeline = MusicToMidiPipeline(config)
            pipeline.beat_detector = FakeBeatDetector()
            pipeline.aria_amt_transcriber = FakeAriaTranscriber()

            result = pipeline._process_piano_aria_amt(str(audio_path), str(out_dir))
            self.assertTrue(Path(result.midi_path).exists())
            self.assertEqual(result.total_notes, 0)

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
