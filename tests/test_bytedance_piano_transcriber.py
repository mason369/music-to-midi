import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from mido import Message, MidiFile, MidiTrack

from src.models.data_models import Config


def _write_valid_midi(path: Path) -> None:
    midi = MidiFile(type=1)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(Message("note_on", note=60, velocity=64, time=0))
    track.append(Message("note_off", note=60, velocity=0, time=240))
    midi.save(str(path))


class ByteDancePianoTranscriberTests(unittest.TestCase):
    def test_pinned_package_and_checkpoint_identity_constants(self):
        from src.core.bytedance_piano_transcriber import (
            BYTEDANCE_PIANO_CHECKPOINT_SHA256,
            BYTEDANCE_PIANO_CHECKPOINT_SIZE,
            BYTEDANCE_PIANO_PACKAGE_VERSION,
        )

        self.assertEqual(BYTEDANCE_PIANO_PACKAGE_VERSION, "0.0.6")
        self.assertEqual(BYTEDANCE_PIANO_CHECKPOINT_SIZE, 171_966_578)
        self.assertEqual(
            BYTEDANCE_PIANO_CHECKPOINT_SHA256,
            "c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141",
        )

    def test_unavailable_when_inference_package_is_missing(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        with patch(
            "src.core.bytedance_piano_transcriber.importlib.util.find_spec",
            return_value=None,
        ):
            self.assertFalse(ByteDancePianoTranscriber.is_available())

    def test_unavailable_when_inference_package_dependency_is_missing(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        with (
            patch(
                "src.core.bytedance_piano_transcriber.importlib.util.find_spec",
                return_value=object(),
            ),
            patch(
                "src.core.bytedance_piano_transcriber.importlib.import_module",
                side_effect=ModuleNotFoundError("No module named 'matplotlib'"),
            ),
        ):
            self.assertFalse(ByteDancePianoTranscriber.is_available())

    def test_unavailable_when_inference_package_version_is_not_pinned(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        with (
            patch(
                "src.core.bytedance_piano_transcriber.importlib.util.find_spec",
                return_value=object(),
            ),
            patch(
                "src.core.bytedance_piano_transcriber.metadata.version",
                return_value="0.0.7",
            ),
        ):
            reason = ByteDancePianoTranscriber.get_unavailable_reason()

        self.assertIn("expected 0.0.6, got 0.0.7", reason)

    def test_model_availability_requires_exact_checkpoint_identity(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        expected_payload = b"official-bytedance-checkpoint"
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pth"
            checkpoint_path.write_bytes(expected_payload)
            transcriber = ByteDancePianoTranscriber(checkpoint_path=checkpoint_path)

            with (
                patch(
                    "src.core.bytedance_piano_transcriber.BYTEDANCE_PIANO_CHECKPOINT_SIZE",
                    len(expected_payload),
                ),
                patch(
                    "src.core.bytedance_piano_transcriber.BYTEDANCE_PIANO_CHECKPOINT_SHA256",
                    hashlib.sha256(expected_payload).hexdigest(),
                ),
            ):
                self.assertTrue(transcriber.is_model_available())
                checkpoint_path.write_bytes(expected_payload[:-1] + b"X")
                self.assertFalse(transcriber.is_model_available())

    def test_model_availability_requires_explicit_checkpoint_file(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "missing.pth"
            transcriber = ByteDancePianoTranscriber(
                Config(),
                checkpoint_path=checkpoint_path,
            )

            self.assertFalse(transcriber.is_model_available())

    def test_missing_checkpoint_error_reports_expected_checkpoint_name(self):
        from src.core.bytedance_piano_transcriber import (
            BYTEDANCE_PIANO_CHECKPOINT_NAME,
            ByteDancePianoTranscriber,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "piano.wav"
            audio_path.write_bytes(b"audio")
            output_path = tmp_path / "out.mid"
            wrong_checkpoint_path = tmp_path / "matplotlib.pth"

            with patch.object(ByteDancePianoTranscriber, "is_available", return_value=True):
                transcriber = ByteDancePianoTranscriber(
                    Config(),
                    checkpoint_path=wrong_checkpoint_path,
                )
                with self.assertRaisesRegex(RuntimeError, BYTEDANCE_PIANO_CHECKPOINT_NAME):
                    transcriber.transcribe(str(audio_path), str(output_path))

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
                _write_valid_midi(Path(midi_path))
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

            with (
                patch.dict(
                    sys.modules,
                    {
                        "piano_transcription_inference": fake_module,
                        "librosa": fake_librosa,
                    },
                ),
                patch(
                    "src.core.bytedance_piano_transcriber.importlib.util.find_spec",
                    return_value=object(),
                ),
                patch(
                    "src.core.bytedance_piano_transcriber.get_device",
                    return_value="cuda:0",
                ),
                patch(
                    "src.core.bytedance_piano_transcriber.ensure_cuda_runtime_compatibility",
                ),
                patch.object(
                    ByteDancePianoTranscriber,
                    "is_model_available",
                    return_value=True,
                ),
            ):
                transcriber = ByteDancePianoTranscriber(
                    Config(use_gpu=True),
                    checkpoint_path=checkpoint_path,
                )
                result = transcriber.transcribe(str(audio_path), str(output_path))

            self.assertEqual(result, str(output_path))
            self.assertTrue(output_path.is_file())
            MidiFile(str(output_path))
            self.assertEqual(calls["device"], "cuda")
            self.assertEqual(calls["checkpoint_path"], str(checkpoint_path))
            self.assertEqual(calls["audio"], [0.0, 0.1])
            temporary_output = Path(calls["midi_path"])
            self.assertEqual(temporary_output.parent.resolve(), output_path.parent.resolve())
            self.assertTrue(temporary_output.name.startswith(".out.bytedance-piano."))
            self.assertFalse(temporary_output.exists())

    def test_transcribe_uses_modern_librosa_loader_instead_of_upstream_legacy_loader(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        calls = {}

        class FakePianoTranscription:
            def __init__(self, device, checkpoint_path=None):
                pass

            def transcribe(self, audio, midi_path):
                calls["audio"] = audio
                _write_valid_midi(Path(midi_path))
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

            with (
                patch.dict(
                    sys.modules,
                    {
                        "piano_transcription_inference": fake_module,
                        "librosa": fake_librosa,
                    },
                ),
                patch(
                    "src.core.bytedance_piano_transcriber.importlib.util.find_spec",
                    return_value=object(),
                ),
                patch(
                    "src.core.bytedance_piano_transcriber.get_device",
                    return_value="cpu",
                ),
                patch.object(
                    ByteDancePianoTranscriber,
                    "is_model_available",
                    return_value=True,
                ),
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
            _write_valid_midi(output_path)
            stale_bytes = output_path.read_bytes()

            with (
                patch.dict(
                    sys.modules,
                    {
                        "piano_transcription_inference": fake_module,
                        "librosa": fake_librosa,
                    },
                ),
                patch(
                    "src.core.bytedance_piano_transcriber.importlib.util.find_spec",
                    return_value=object(),
                ),
                patch(
                    "src.core.bytedance_piano_transcriber.get_device",
                    return_value="cpu",
                ),
                patch.object(
                    ByteDancePianoTranscriber,
                    "is_model_available",
                    return_value=True,
                ),
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
            self.assertEqual(output_path.read_bytes(), stale_bytes)
            self.assertEqual(list(output_path.parent.glob(".out.bytedance-piano.*.tmp.mid")), [])


if __name__ == "__main__":
    unittest.main()
