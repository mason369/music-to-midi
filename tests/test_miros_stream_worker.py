import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.miros_stream_worker import run_miros_stream_worker


class MirosStreamWorkerTests(unittest.TestCase):
    def test_normalized_wav_adapter_is_used_and_restored_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "ai4m-miros"
            repo.mkdir()
            (repo / "transcribe.py").write_text("# identity placeholder\n", encoding="utf-8")
            audio_path = root / "input.wav"
            audio_path.write_bytes(b"wav")
            output_path = root / "output.mid"

            original_load = Mock(name="original_torchaudio_load")
            original_resolve_device = Mock(return_value="xpu:0")
            original_validate_device = Mock()
            fake_torchaudio = types.SimpleNamespace(load=original_load)
            fake_transcribe = types.ModuleType("transcribe")
            fake_transcribe.torchaudio = fake_torchaudio
            fake_transcribe._resolve_runtime_device = original_resolve_device
            fake_transcribe._validate_model_device = original_validate_device
            loaded = {}
            waveform = object()
            model = object()

            def transcribe(input_path, destination_path):
                device = fake_transcribe._resolve_runtime_device()
                fake_transcribe._validate_model_device(model, device)
                loaded["audio"] = fake_torchaudio.load(uri=input_path)
                loaded["destination"] = destination_path
                raise RuntimeError("inference failed")

            fake_transcribe.transcribe = transcribe

            with (
                patch.dict("sys.modules", {"transcribe": fake_transcribe}),
                patch(
                    "src.core.miros_stream_worker.load_audio_tensor",
                    return_value=(waveform, 44100),
                ) as explicit_loader,
                patch.dict(os.environ, {"MUSIC_TO_MIDI_MIROS_DEVICE": "xpu:0"}),
                patch("builtins.print") as print_line,
                self.assertRaisesRegex(RuntimeError, "inference failed"),
            ):
                run_miros_stream_worker(repo, audio_path, output_path)

            explicit_loader.assert_called_once_with(str(audio_path.resolve()))
            self.assertEqual(loaded["audio"], (waveform, 44100))
            self.assertEqual(loaded["destination"], str(output_path.resolve()))
            self.assertIs(fake_torchaudio.load, original_load)
            self.assertIs(fake_transcribe._resolve_runtime_device, original_resolve_device)
            self.assertIs(fake_transcribe._validate_model_device, original_validate_device)
            original_resolve_device.assert_called_once_with()
            original_validate_device.assert_called_once_with(model, "xpu:0")
            printed = [call.args[0] for call in print_line.call_args_list]
            self.assertIn(
                "MIROS_RUNTIME_DEVICE_OK | requested=xpu:0 resolved=xpu:0",
                printed,
            )
            self.assertIn("MIROS_MODEL_DEVICE_OK | expected=xpu:0", printed)
            original_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
