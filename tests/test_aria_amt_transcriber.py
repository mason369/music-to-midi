import hashlib
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from mido import Message, MidiFile, MidiTrack

from src.core.aria_amt_transcriber import (
    ARIA_AMT_CHECKPOINT_REVISION,
    ARIA_AMT_CHECKPOINT_SHA256,
    ARIA_AMT_CHECKPOINT_SIZE,
    ARIA_AMT_CHECKPOINT_URL,
    ARIA_AMT_SOURCE_ARCHIVE_URL,
    ARIA_AMT_SOURCE_REVISION,
    AriaAmtTranscriber,
    get_aria_amt_runtime_unavailable_reason,
)

ARIA_AMT_MODEL_CONFIG_NAME = "medium-double"


def _write_valid_midi(path: Path) -> None:
    midi = MidiFile(type=1)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(Message("note_on", note=60, velocity=64, time=0))
    track.append(Message("note_off", note=60, velocity=0, time=240))
    midi.save(str(path))


class AriaAmtTranscriberTests(unittest.TestCase):
    def test_pre_cancelled_transcription_is_not_reset_at_entry(self):
        transcriber = AriaAmtTranscriber(checkpoint_path=Path("checkpoint.safetensors"))
        transcriber.cancel()

        with patch.object(transcriber, "is_available") as is_available:
            with self.assertRaisesRegex(InterruptedError, "已取消"):
                transcriber.transcribe("song.wav", "song.mid")

        is_available.assert_not_called()

    def test_subprocess_cancellation_precedes_nonzero_exit_failure(self):
        transcriber = AriaAmtTranscriber(checkpoint_path=Path("checkpoint.safetensors"))

        class FakeProcess:
            returncode = -15

            def communicate(self, timeout=None):
                transcriber._cancelled = True
                return ("", "terminated")

        with patch(
            "src.core.aria_amt_transcriber.subprocess.Popen",
            return_value=FakeProcess(),
        ):
            with self.assertRaisesRegex(InterruptedError, "已取消"):
                transcriber._run_transcription_subprocess(
                    Path("song.wav"),
                    Path("output"),
                )

        self.assertIsNone(transcriber._process)

    def test_cancel_requests_terminate_without_waiting_or_killing(self):
        class FakeProcess:
            def __init__(self):
                self.terminated = False

            @staticmethod
            def poll():
                return None

            def terminate(self):
                self.terminated = True

            def kill(self):
                raise AssertionError("GUI-side cancel must not kill")

            def wait(self, timeout):
                raise AssertionError("GUI-side cancel must not wait")

        transcriber = AriaAmtTranscriber(checkpoint_path=Path("checkpoint.safetensors"))
        process = FakeProcess()
        transcriber._process = process

        transcriber.cancel()

        self.assertTrue(transcriber._cancelled)
        self.assertTrue(process.terminated)

    def test_subprocess_worker_kills_and_reaps_after_cancel_deadline(self):
        class FakeProcess:
            returncode = None

            def __init__(self):
                self.terminated = False
                self.killed = False
                self.communicate_timeouts = []

            @staticmethod
            def poll():
                return None

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True
                self.returncode = -9

            def communicate(self, timeout=None):
                self.communicate_timeouts.append(timeout)
                if timeout == 0.1:
                    raise subprocess.TimeoutExpired("aria-amt", timeout)
                return ("", "killed")

        transcriber = AriaAmtTranscriber(checkpoint_path=Path("checkpoint.safetensors"))
        transcriber._cancelled = True
        process = FakeProcess()

        with (
            patch(
                "src.core.aria_amt_transcriber.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "src.core.aria_amt_transcriber.time.monotonic",
                side_effect=[0.0, 5.0],
            ),
        ):
            with self.assertRaisesRegex(InterruptedError, "已取消"):
                transcriber._run_transcription_subprocess(
                    Path("song.wav"),
                    Path("output"),
                )

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.communicate_timeouts, [0.1, 5.0])
        self.assertIsNone(transcriber._process)

    def test_pinned_source_and_checkpoint_identity_constants(self):
        self.assertEqual(
            ARIA_AMT_SOURCE_REVISION,
            "a1ab73fc901d1759ec3bc173c146b3c6a3040261",
        )
        self.assertEqual(
            ARIA_AMT_CHECKPOINT_REVISION,
            "8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b",
        )
        self.assertIn(ARIA_AMT_CHECKPOINT_REVISION, ARIA_AMT_CHECKPOINT_URL)
        self.assertEqual(ARIA_AMT_CHECKPOINT_SIZE, 446_577_344)
        self.assertEqual(
            ARIA_AMT_CHECKPOINT_SHA256,
            "089d3129dbe93246aeda55efe668c8a48af08afaf9dd15c64cef0a07c0fb30a4",
        )

    def test_is_available_returns_false_when_amt_package_is_missing(self):
        with patch(
            "src.core.aria_amt_transcriber.importlib.util.find_spec",
            side_effect=ModuleNotFoundError("No module named 'amt'"),
        ):
            self.assertFalse(AriaAmtTranscriber.is_available())

    def test_runtime_identity_accepts_only_the_pinned_source_commit(self):
        class FakeDistribution:
            @staticmethod
            def read_text(name):
                self.assertEqual(name, "direct_url.json")
                return json.dumps(
                    {
                        "url": "https://github.com/EleutherAI/aria-amt.git",
                        "vcs_info": {
                            "vcs": "git",
                            "commit_id": ARIA_AMT_SOURCE_REVISION,
                        },
                    }
                )

        with (
            patch(
                "src.core.aria_amt_transcriber.importlib.util.find_spec",
                return_value=object(),
            ),
            patch(
                "src.core.aria_amt_transcriber.metadata.distribution",
                return_value=FakeDistribution(),
            ),
        ):
            self.assertEqual(get_aria_amt_runtime_unavailable_reason(), "")

        class ArchiveDistribution:
            @staticmethod
            def read_text(_name):
                return json.dumps(
                    {
                        "url": ARIA_AMT_SOURCE_ARCHIVE_URL,
                        "archive_info": {},
                    }
                )

        with (
            patch(
                "src.core.aria_amt_transcriber.importlib.util.find_spec",
                return_value=object(),
            ),
            patch(
                "src.core.aria_amt_transcriber.metadata.distribution",
                return_value=ArchiveDistribution(),
            ),
        ):
            self.assertEqual(get_aria_amt_runtime_unavailable_reason(), "")

        class WrongDistribution:
            @staticmethod
            def read_text(_name):
                return json.dumps(
                    {
                        "url": ARIA_AMT_SOURCE_ARCHIVE_URL.replace(
                            ARIA_AMT_SOURCE_REVISION, "main"
                        ),
                        "vcs_info": {"commit_id": "0" * 40, "vcs": "git"},
                    }
                )

        with (
            patch(
                "src.core.aria_amt_transcriber.importlib.util.find_spec",
                return_value=object(),
            ),
            patch(
                "src.core.aria_amt_transcriber.metadata.distribution",
                return_value=WrongDistribution(),
            ),
        ):
            reason = get_aria_amt_runtime_unavailable_reason()

        self.assertIn(ARIA_AMT_SOURCE_REVISION, reason)
        self.assertIn("0" * 40, reason)

    def test_model_availability_requires_exact_checkpoint_identity(self):
        expected_payload = b"official-aria-checkpoint"
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.safetensors"
            checkpoint_path.write_bytes(expected_payload)
            transcriber = AriaAmtTranscriber(checkpoint_path=checkpoint_path)

            with (
                patch(
                    "src.core.aria_amt_transcriber.ARIA_AMT_CHECKPOINT_SIZE",
                    len(expected_payload),
                ),
                patch(
                    "src.core.aria_amt_transcriber.ARIA_AMT_CHECKPOINT_SHA256",
                    hashlib.sha256(expected_payload).hexdigest(),
                ),
            ):
                self.assertTrue(transcriber.is_model_available())
                checkpoint_path.write_bytes(expected_payload[:-1] + b"X")
                self.assertFalse(transcriber.is_model_available())

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
                def __init__(self, command):
                    self.command = command

                returncode = 0

                def communicate(self, timeout=None):
                    save_dir = Path(self.command[self.command.index("-save_dir") + 1])
                    _write_valid_midi(save_dir / "song.mid")
                    return ("", "")

            def fake_popen(
                command,
                *,
                stdout,
                stderr,
                text,
                encoding,
                errors,
                env,
            ):
                self.assertIsNotNone(stdout)
                self.assertIsNotNone(stderr)
                self.assertTrue(text)
                calls.append((command, encoding, errors, env))
                return FakeProcess(command)

            transcriber = AriaAmtTranscriber(checkpoint_path=checkpoint_path)

            with (
                patch.object(AriaAmtTranscriber, "is_available", return_value=True),
                patch.object(
                    transcriber,
                    "is_model_available",
                    return_value=True,
                ),
                patch(
                    "src.core.aria_amt_transcriber.platform.system",
                    return_value="Linux",
                ),
                patch(
                    "src.core.aria_amt_transcriber.is_frozen_app",
                    return_value=False,
                ),
                patch(
                    "src.core.aria_amt_transcriber.subprocess.Popen",
                    side_effect=fake_popen,
                ),
            ):
                result = transcriber.transcribe(str(audio_path), str(output_path))

            self.assertEqual(result, str(output_path))
            command, encoding, errors, env = calls[0]
            self.assertEqual(encoding, "utf-8")
            self.assertEqual(errors, "replace")
            self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
            self.assertEqual(env["PYTHONUTF8"], "1")
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
                _write_valid_midi(temp_dir / "song.mid")

            transcriber = AriaAmtTranscriber(checkpoint_path=checkpoint_path)

            with (
                patch.object(AriaAmtTranscriber, "is_available", return_value=True),
                patch.object(
                    transcriber,
                    "is_model_available",
                    return_value=True,
                ),
                patch(
                    "src.core.aria_amt_transcriber.platform.system",
                    return_value="Windows",
                ),
                patch("src.core.aria_amt_transcriber.subprocess.Popen") as popen,
                patch("src.core.aria_amt_transcriber.importlib.import_module") as import_module,
                patch.object(
                    transcriber,
                    "_run_transcription_windows_single_file",
                    side_effect=fake_windows_transcribe,
                ),
            ):
                result = transcriber.transcribe(str(audio_path), str(output_path))

            self.assertEqual(result, str(output_path))
            self.assertEqual(calls[0][0], audio_path)
            self.assertEqual(calls[0][1].parent, output_path.parent)
            self.assertTrue(calls[0][1].name.startswith(".aria_amt_"))
            self.assertFalse(calls[0][1].exists())
            popen.assert_not_called()
            import_module.assert_not_called()

    def test_windows_segment_reader_does_not_use_torchaudio_stream_reader(self):
        waveform = torch.arange(0, 8, dtype=torch.float32).unsqueeze(0)
        with (
            patch(
                "src.core.aria_amt_transcriber.torchaudio.load",
                return_value=(waveform, 4),
            ),
            patch("src.core.aria_amt_transcriber.torchaudio.io.StreamReader") as stream_reader,
        ):
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
                _write_valid_midi(save_dir / "song.mid")

            fake_run_module = types.SimpleNamespace(transcribe=fake_transcribe)
            transcriber = AriaAmtTranscriber(checkpoint_path=checkpoint_path)

            with (
                patch.object(AriaAmtTranscriber, "is_available", return_value=True),
                patch.object(
                    transcriber,
                    "is_model_available",
                    return_value=True,
                ),
                patch(
                    "src.core.aria_amt_transcriber.platform.system",
                    return_value="Linux",
                ),
                patch("src.core.aria_amt_transcriber.subprocess.Popen") as popen,
                patch(
                    "src.core.aria_amt_transcriber.is_frozen_app",
                    return_value=True,
                ),
                patch(
                    "src.core.aria_amt_transcriber.importlib.import_module",
                    return_value=fake_run_module,
                ),
            ):
                result = transcriber.transcribe(str(audio_path), str(output_path))

            self.assertEqual(result, str(output_path))
            popen.assert_not_called()
            self.assertEqual(len(calls), 1)
            call = calls[0]
            self.assertEqual(call["model_name"], ARIA_AMT_MODEL_CONFIG_NAME)
            self.assertEqual(call["checkpoint_path"], str(checkpoint_path))
            self.assertEqual(call["load_path"], str(audio_path))
            self.assertIsNone(call["load_dir"])
            self.assertEqual(call["batch_size"], 1)
            save_dir = Path(call["save_dir"])
            self.assertEqual(save_dir.parent, output_path.parent)
            self.assertTrue(save_dir.name.startswith(".aria_amt_"))
            self.assertFalse(save_dir.exists())

    def test_transcribe_reports_temp_dir_when_midi_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            output_path = tmp_path / "out" / "song.mid"
            checkpoint_path = tmp_path / "piano-medium-double-1.0.safetensors"
            audio_path.write_bytes(b"wav")
            checkpoint_path.write_bytes(b"weights")

            def fake_windows_transcribe(_input_path, temp_dir, progress_callback=None):
                (temp_dir / "debug.txt").write_text("no midi here", encoding="utf-8")

            transcriber = AriaAmtTranscriber(checkpoint_path=checkpoint_path)

            with (
                patch.object(AriaAmtTranscriber, "is_available", return_value=True),
                patch.object(
                    transcriber,
                    "is_model_available",
                    return_value=True,
                ),
                patch(
                    "src.core.aria_amt_transcriber.platform.system",
                    return_value="Windows",
                ),
                patch.object(
                    transcriber,
                    "_run_transcription_windows_single_file",
                    side_effect=fake_windows_transcribe,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "临时输出目录") as cm:
                    transcriber.transcribe(str(audio_path), str(output_path))

        message = str(cm.exception)
        self.assertIn("Aria-AMT 未生成 MIDI 输出", message)
        self.assertIn(str(output_path.resolve()), message)
        self.assertIn(str(output_path.parent.resolve()), message)
        self.assertIn(".aria_amt_", message)
        self.assertIn("debug.txt", message)

    def test_stale_final_midi_is_not_accepted_when_current_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            output_path = tmp_path / "out" / "song.mid"
            checkpoint_path = tmp_path / "checkpoint.safetensors"
            audio_path.write_bytes(b"wav")
            checkpoint_path.write_bytes(b"weights")
            output_path.parent.mkdir(parents=True)
            _write_valid_midi(output_path)
            stale_bytes = output_path.read_bytes()

            transcriber = AriaAmtTranscriber(checkpoint_path=checkpoint_path)
            with (
                patch.object(AriaAmtTranscriber, "is_available", return_value=True),
                patch.object(transcriber, "is_model_available", return_value=True),
                patch("src.core.aria_amt_transcriber.platform.system", return_value="Windows"),
                patch.object(
                    transcriber,
                    "_run_transcription_windows_single_file",
                    return_value=None,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "未生成 MIDI 输出"):
                    transcriber.transcribe(str(audio_path), str(output_path))

            self.assertEqual(output_path.read_bytes(), stale_bytes)
            self.assertEqual(list(output_path.parent.glob(".aria_amt_*")), [])


if __name__ == "__main__":
    unittest.main()
