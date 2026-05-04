import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.core.aria_amt_transcriber import AriaAmtTranscriber


ARIA_AMT_MODEL_CONFIG_NAME = "medium-double"


class AriaAmtTranscriberTests(unittest.TestCase):
    def test_is_available_returns_false_when_amt_package_is_missing(self):
        with patch(
            "src.core.aria_amt_transcriber.importlib.util.find_spec",
            side_effect=ModuleNotFoundError("No module named 'amt'"),
        ):
            self.assertFalse(AriaAmtTranscriber.is_available())

    def test_non_windows_dev_transcribe_uses_aria_amt_cli_argument_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            output_path = tmp_path / "out" / "song.mid"
            checkpoint_path = tmp_path / "piano-medium-double-1.0.safetensors"
            audio_path.write_bytes(b"wav")
            checkpoint_path.write_bytes(b"weights")
            calls = []

            class FakeProcess:
                returncode = 0

                def communicate(self):
                    temp_midi = output_path.parent / ".aria_amt_tmp" / "song.mid"
                    temp_midi.write_bytes(b"midi")
                    return ("", "")

            def fake_popen(command, *, stdout, stderr, text):
                self.assertIsNotNone(stdout)
                self.assertIsNotNone(stderr)
                self.assertTrue(text)
                calls.append(command)
                return FakeProcess()

            transcriber = AriaAmtTranscriber(checkpoint_path=checkpoint_path)

            with patch.object(AriaAmtTranscriber, "is_available", return_value=True), patch(
                "src.core.aria_amt_transcriber.platform.system",
                return_value="Linux",
            ), patch(
                "src.core.aria_amt_transcriber.is_frozen_app",
                return_value=False,
            ), patch(
                "src.core.aria_amt_transcriber.subprocess.Popen",
                side_effect=fake_popen,
            ):
                result = transcriber.transcribe(str(audio_path), str(output_path))

            self.assertEqual(result, str(output_path))
            command = calls[0]
            self.assertEqual(command[1:4], ["-m", "amt.run", "transcribe"])
            self.assertIn(ARIA_AMT_MODEL_CONFIG_NAME, command)
            self.assertIn(str(checkpoint_path), command)
            self.assertIn("-load_path", command)
            self.assertIn("-save_dir", command)
            self.assertNotIn("--load_path", command)
            self.assertNotIn("--save_dir", command)
            self.assertLess(command.index(ARIA_AMT_MODEL_CONFIG_NAME), command.index("-load_path"))
            self.assertLess(command.index(str(checkpoint_path)), command.index("-load_path"))

    def test_windows_transcribe_uses_single_file_path_instead_of_posix_batch_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            output_path = tmp_path / "out" / "song.mid"
            checkpoint_path = tmp_path / "piano-medium-double-1.0.safetensors"
            audio_path.write_bytes(b"wav")
            checkpoint_path.write_bytes(b"weights")
            calls = []

            def fake_windows_transcribe(input_path, temp_dir, progress_callback=None):
                calls.append((input_path, temp_dir, progress_callback))
                (temp_dir / "song.mid").write_bytes(b"midi")

            transcriber = AriaAmtTranscriber(checkpoint_path=checkpoint_path)

            with patch.object(AriaAmtTranscriber, "is_available", return_value=True), patch(
                "src.core.aria_amt_transcriber.platform.system",
                return_value="Windows",
            ), patch(
                "src.core.aria_amt_transcriber.subprocess.Popen"
            ) as popen, patch(
                "src.core.aria_amt_transcriber.importlib.import_module"
            ) as import_module, patch.object(
                transcriber,
                "_run_transcription_windows_single_file",
                side_effect=fake_windows_transcribe,
            ):
                result = transcriber.transcribe(str(audio_path), str(output_path))

            self.assertEqual(result, str(output_path))
            self.assertEqual(calls[0][0], audio_path)
            self.assertEqual(calls[0][1], output_path.parent / ".aria_amt_tmp")
            popen.assert_not_called()
            import_module.assert_not_called()

    def test_windows_segment_reader_does_not_use_torchaudio_stream_reader(self):
        waveform = torch.arange(0, 8, dtype=torch.float32).unsqueeze(0)
        with patch(
            "src.core.aria_amt_transcriber.torchaudio.load",
            return_value=(waveform, 4),
        ), patch(
            "src.core.aria_amt_transcriber.torchaudio.io.StreamReader"
        ) as stream_reader:
            segments = list(
                AriaAmtTranscriber._iter_windows_wav_segments(
                    Path("song.wav"),
                    sample_rate=4,
                    chunk_len_seconds=1,
                    stride_factor=2,
                )
            )

        stream_reader.assert_not_called()
        self.assertTrue(segments)
        self.assertTrue(all(segment.shape[0] == 4 for segment in segments))

    def test_frozen_non_windows_transcribe_runs_amt_in_process_instead_of_relaunching_exe(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            output_path = tmp_path / "out" / "song.mid"
            checkpoint_path = tmp_path / "piano-medium-double-1.0.safetensors"
            audio_path.write_bytes(b"wav")
            checkpoint_path.write_bytes(b"weights")
            calls = []

            def fake_transcribe(**kwargs):
                calls.append(kwargs)
                save_dir = Path(kwargs["save_dir"])
                save_dir.mkdir(parents=True, exist_ok=True)
                (save_dir / "song.mid").write_bytes(b"midi")

            fake_run_module = types.SimpleNamespace(transcribe=fake_transcribe)
            transcriber = AriaAmtTranscriber(checkpoint_path=checkpoint_path)

            with patch.object(AriaAmtTranscriber, "is_available", return_value=True), patch(
                "src.core.aria_amt_transcriber.platform.system",
                return_value="Linux",
            ), patch(
                "src.core.aria_amt_transcriber.subprocess.Popen"
            ) as popen, patch(
                "src.core.aria_amt_transcriber.is_frozen_app",
                return_value=True,
            ), patch(
                "src.core.aria_amt_transcriber.importlib.import_module",
                return_value=fake_run_module,
            ):
                result = transcriber.transcribe(str(audio_path), str(output_path))

            self.assertEqual(result, str(output_path))
            popen.assert_not_called()
            self.assertEqual(
                calls,
                [
                    {
                        "model_name": ARIA_AMT_MODEL_CONFIG_NAME,
                        "checkpoint_path": str(checkpoint_path),
                        "load_path": str(audio_path),
                        "load_dir": None,
                        "save_dir": str(output_path.parent / ".aria_amt_tmp"),
                        "batch_size": 1,
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
