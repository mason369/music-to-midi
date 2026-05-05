import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models.data_models import Config


class ByteDancePianoTranscriberTests(unittest.TestCase):
    def test_unavailable_when_inference_package_is_missing(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        with patch(
            "src.core.bytedance_piano_transcriber.importlib.util.find_spec",
            return_value=None,
        ):
            self.assertFalse(ByteDancePianoTranscriber.is_available())

    def test_model_availability_requires_explicit_checkpoint_file(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "missing.pth"
            transcriber = ByteDancePianoTranscriber(
                Config(),
                checkpoint_path=checkpoint_path,
            )

            self.assertFalse(transcriber.is_model_available())

    def test_transcribe_uses_explicit_checkpoint_and_keeps_written_midi(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        calls = {}

        class FakePianoTranscription:
            def __init__(self, device, checkpoint_path=None):
                calls["device"] = device
                calls["checkpoint_path"] = checkpoint_path

            def transcribe(self, audio, midi_path):
                calls["audio"] = audio
                calls["midi_path"] = midi_path
                Path(midi_path).write_bytes(b"MIDI_WITH_CC64")
                return {"est_pedal_events": [{"onset_time": 0.1, "offset_time": 0.9}]}

        fake_module = types.ModuleType("piano_transcription_inference")
        fake_module.PianoTranscription = FakePianoTranscription
        fake_module.sample_rate = 16000
        fake_librosa = types.ModuleType("librosa")
        fake_librosa.load = lambda path, sr, mono: ([0.0, 0.1], sr)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "note_F1=0.9677_pedal_F1=0.9186.pth"
            with checkpoint_path.open("wb") as checkpoint_file:
                checkpoint_file.truncate(170_000_000)
            audio_path = tmp_path / "piano.wav"
            audio_path.write_bytes(b"audio")
            output_path = tmp_path / "out.mid"

            with patch.dict(
                sys.modules,
                {
                    "piano_transcription_inference": fake_module,
                    "librosa": fake_librosa,
                },
            ), patch(
                "src.core.bytedance_piano_transcriber.importlib.util.find_spec",
                return_value=object(),
            ), patch(
                "src.core.bytedance_piano_transcriber.get_device",
                return_value="cuda:0",
            ):
                transcriber = ByteDancePianoTranscriber(
                    Config(use_gpu=True),
                    checkpoint_path=checkpoint_path,
                )
                result = transcriber.transcribe(str(audio_path), str(output_path))

            self.assertEqual(result, str(output_path))
            self.assertEqual(output_path.read_bytes(), b"MIDI_WITH_CC64")
            self.assertEqual(calls["device"], "cuda")
            self.assertEqual(calls["checkpoint_path"], str(checkpoint_path))
            self.assertEqual(calls["audio"], [0.0, 0.1])
            self.assertEqual(calls["midi_path"], str(output_path))

    def test_transcribe_uses_modern_librosa_loader_instead_of_upstream_legacy_loader(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        calls = {}

        class FakePianoTranscription:
            def __init__(self, device, checkpoint_path=None):
                pass

            def transcribe(self, audio, midi_path):
                calls["audio"] = audio
                Path(midi_path).write_bytes(b"midi")
                return {}

        def legacy_loader(*_args, **_kwargs):
            raise TypeError("legacy librosa positional resample failure")

        fake_module = types.ModuleType("piano_transcription_inference")
        fake_module.PianoTranscription = FakePianoTranscription
        fake_module.sample_rate = 16000
        fake_module.load_audio = legacy_loader

        fake_librosa = types.ModuleType("librosa")
        fake_librosa.load = lambda path, sr, mono: ([0.2, 0.3], sr)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "note_F1=0.9677_pedal_F1=0.9186.pth"
            with checkpoint_path.open("wb") as checkpoint_file:
                checkpoint_file.truncate(170_000_000)
            audio_path = tmp_path / "piano.wav"
            audio_path.write_bytes(b"audio")
            output_path = tmp_path / "out.mid"

            with patch.dict(
                sys.modules,
                {
                    "piano_transcription_inference": fake_module,
                    "librosa": fake_librosa,
                },
            ), patch(
                "src.core.bytedance_piano_transcriber.importlib.util.find_spec",
                return_value=object(),
            ), patch(
                "src.core.bytedance_piano_transcriber.get_device",
                return_value="cpu",
            ):
                transcriber = ByteDancePianoTranscriber(
                    Config(use_gpu=False),
                    checkpoint_path=checkpoint_path,
                )
                transcriber.transcribe(str(audio_path), str(output_path))

            self.assertEqual(calls["audio"], [0.2, 0.3])

    def test_transcribe_reports_expected_output_when_midi_is_missing(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        class FakePianoTranscription:
            def __init__(self, device, checkpoint_path=None):
                pass

            def transcribe(self, audio, midi_path):
                return {}

        fake_module = types.ModuleType("piano_transcription_inference")
        fake_module.PianoTranscription = FakePianoTranscription
        fake_module.sample_rate = 16000
        fake_librosa = types.ModuleType("librosa")
        fake_librosa.load = lambda path, sr, mono: ([0.0], sr)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "note_F1=0.9677_pedal_F1=0.9186.pth"
            with checkpoint_path.open("wb") as checkpoint_file:
                checkpoint_file.truncate(170_000_000)
            audio_path = tmp_path / "piano.wav"
            audio_path.write_bytes(b"audio")
            output_path = tmp_path / "out.mid"

            with patch.dict(
                sys.modules,
                {
                    "piano_transcription_inference": fake_module,
                    "librosa": fake_librosa,
                },
            ), patch(
                "src.core.bytedance_piano_transcriber.importlib.util.find_spec",
                return_value=object(),
            ), patch(
                "src.core.bytedance_piano_transcriber.get_device",
                return_value="cpu",
            ):
                transcriber = ByteDancePianoTranscriber(
                    Config(use_gpu=False),
                    checkpoint_path=checkpoint_path,
                )
                with self.assertRaisesRegex(RuntimeError, "期望输出") as cm:
                    transcriber.transcribe(str(audio_path), str(output_path))

        message = str(cm.exception)
        self.assertIn("ByteDance Piano 未生成 MIDI 输出", message)
        self.assertIn(str(output_path.resolve()), message)
        self.assertIn(str(audio_path.resolve()), message)


if __name__ == "__main__":
    unittest.main()
