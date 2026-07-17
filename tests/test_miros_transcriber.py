import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import src.core.miros_transcriber as miros_runtime
from src.core.miros_transcriber import MirosTranscriber
from src.models.data_models import Config


def _write_valid_midi(path: Path) -> None:
    import mido

    path.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(type=0)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    midi.save(str(path))


class _FakeNote:
    def __init__(self, pitch, start, end, velocity):
        self.pitch = pitch
        self.start = start
        self.end = end
        self.velocity = velocity


class _FakeInstrument:
    def __init__(self, program, is_drum, notes):
        self.program = program
        self.is_drum = is_drum
        self.notes = notes


class _FakePrettyMIDI:
    def __init__(self, _path):
        self.instruments = [
            _FakeInstrument(
                program=5,
                is_drum=False,
                notes=[
                    _FakeNote(pitch=60, start=0.5, end=1.25, velocity=90),
                    _FakeNote(pitch=64, start=1.5, end=2.0, velocity=70),
                ],
            ),
            _FakeInstrument(
                program=0,
                is_drum=True,
                notes=[
                    _FakeNote(pitch=36, start=0.25, end=0.35, velocity=110),
                ],
            ),
        ]


class MirosTranscriberTests(unittest.TestCase):
    def test_subprocess_enables_expandable_cuda_segments_without_overriding_user_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "external" / "ai4m-miros"
            repo.mkdir(parents=True)
            entrypoint = repo / "main.py"
            entrypoint.write_text("print('miros')", encoding="utf-8")
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"wav")
            captured = {}

            class FakeProcess:
                returncode = 0

                def __init__(self, command, **kwargs):
                    captured["command"] = command
                    captured["env"] = kwargs["env"]

                def communicate(self):
                    _write_valid_midi(
                        Path(captured["command"][captured["command"].index("-o") + 1])
                    )
                    return "", ""

            transcriber = MirosTranscriber(Config())
            with (
                patch.object(MirosTranscriber, "_repo_dir", return_value=repo),
                patch.object(MirosTranscriber, "_entrypoint_path", return_value=entrypoint),
                patch.object(MirosTranscriber, "get_unavailable_reason", return_value=""),
                patch.object(
                    subprocess,
                    "Popen",
                    side_effect=lambda command, **kwargs: FakeProcess(command, **kwargs),
                ),
                patch.dict(
                    os.environ,
                    {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"},
                ),
            ):
                transcriber.transcribe_to_midi(
                    str(audio_path),
                    str(root / "out" / "song.mid"),
                )

            self.assertEqual(
                captured["env"]["PYTORCH_CUDA_ALLOC_CONF"],
                "max_split_size_mb:64",
            )

    def test_source_without_weights_is_not_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "main.py").write_text("print('miros')", encoding="utf-8")
            (repo / "transcribe.py").write_text("print('miros')", encoding="utf-8")

            with (
                patch.object(MirosTranscriber, "_repo_dir", return_value=repo),
                patch.object(
                    MirosTranscriber,
                    "_missing_modules",
                    return_value=[],
                ),
                patch.object(
                    miros_runtime,
                    "get_miros_source_identity_error",
                    return_value="",
                ),
            ):
                reason = MirosTranscriber.get_unavailable_reason()
                available = MirosTranscriber.is_available()

        self.assertIn("模型权重身份校验失败", reason)
        self.assertFalse(available)

    def test_runtime_rejects_same_size_wrong_weight_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "main.py").write_text("print('miros')", encoding="utf-8")
            (repo / "transcribe.py").write_text("print('miros')", encoding="utf-8")
            pretrained = repo / MirosTranscriber.PRETRAINED_REL_PATH
            checkpoint = repo / MirosTranscriber.CHECKPOINT_REL_PATH
            pretrained.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            pretrained.write_bytes(b"bad-pretrained")
            checkpoint.write_bytes(b"fine-tuned")

            with (
                patch.object(MirosTranscriber, "_repo_dir", return_value=repo),
                patch.object(
                    MirosTranscriber,
                    "_missing_modules",
                    return_value=[],
                ),
                patch.object(
                    miros_runtime,
                    "get_miros_source_identity_error",
                    return_value="",
                ),
                patch.object(
                    miros_runtime,
                    "MIROS_PRETRAINED_EXACT_BYTES",
                    len(b"bad-pretrained"),
                ),
                patch.object(
                    miros_runtime,
                    "MIROS_PRETRAINED_SHA256",
                    hashlib.sha256(b"good-pretrained").hexdigest(),
                ),
                patch.object(
                    miros_runtime,
                    "MIROS_FINETUNED_EXACT_BYTES",
                    len(b"fine-tuned"),
                ),
                patch.object(
                    miros_runtime,
                    "MIROS_FINETUNED_SHA256",
                    hashlib.sha256(b"fine-tuned").hexdigest(),
                ),
            ):
                reason = MirosTranscriber.get_unavailable_reason()
                available = MirosTranscriber.is_model_available()

        self.assertIn("SHA256 mismatch", reason)
        self.assertFalse(available)

    def test_runtime_rejects_unapproved_source_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "main.py").write_text("print('miros')", encoding="utf-8")
            (repo / "transcribe.py").write_text("print('miros')", encoding="utf-8")

            with (
                patch.object(MirosTranscriber, "_repo_dir", return_value=repo),
                patch.object(
                    MirosTranscriber,
                    "_missing_modules",
                    return_value=[],
                ),
            ):
                reason = MirosTranscriber.get_unavailable_reason()
                available = MirosTranscriber.is_model_available()

        self.assertIn("源码身份校验失败", reason)
        self.assertIn("patched source tree SHA256 mismatch", reason)
        self.assertFalse(available)

    def test_transcribe_to_midi_passes_absolute_paths_to_repo_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "external" / "ai4m-miros"
            repo.mkdir(parents=True)
            entrypoint = repo / "main.py"
            entrypoint.write_text("print('miros')", encoding="utf-8")
            audio_path = root / "relative" / "song.wav"
            audio_path.parent.mkdir()
            audio_path.write_bytes(b"wav")
            captured = {}

            class FakeProcess:
                returncode = 0

                def __init__(self, command, **kwargs):
                    captured["command"] = command
                    captured["cwd"] = kwargs.get("cwd")

                def communicate(self):
                    _write_valid_midi(
                        Path(captured["command"][captured["command"].index("-o") + 1])
                    )
                    return "", ""

            transcriber = MirosTranscriber(Config())

            previous_cwd = os.getcwd()
            os.chdir(root)
            try:
                with (
                    patch.object(MirosTranscriber, "_repo_dir", return_value=repo),
                    patch.object(MirosTranscriber, "_entrypoint_path", return_value=entrypoint),
                    patch.object(MirosTranscriber, "get_unavailable_reason", return_value=""),
                    patch.object(
                        subprocess,
                        "Popen",
                        side_effect=lambda command, **kwargs: FakeProcess(command, **kwargs),
                    ),
                ):
                    result = transcriber.transcribe_to_midi("relative/song.wav", "out/song.mid")
            finally:
                os.chdir(previous_cwd)

        command = captured["command"]
        input_arg = Path(command[command.index("-i") + 1])
        output_arg = Path(command[command.index("-o") + 1])

        self.assertTrue(input_arg.is_absolute())
        self.assertTrue(output_arg.is_absolute())
        self.assertNotEqual(output_arg, (root / "out" / "song.mid").resolve())
        self.assertEqual(str(repo), captured["cwd"])
        self.assertEqual(str((root / "out" / "song.mid").resolve()), result)

    def test_transcribe_to_midi_preserves_official_cli_midi_bytes_exactly(self):
        import mido

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "external" / "ai4m-miros"
            repo.mkdir(parents=True)
            entrypoint = repo / "main.py"
            entrypoint.write_text("print('miros')", encoding="utf-8")
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"wav")
            output_path = root / "out" / "official.mid"
            captured = {}

            class FakeOfficialCliProcess:
                returncode = 0

                def __init__(self, command, **_kwargs):
                    self.command = command

                def communicate(self):
                    attempt_path = Path(self.command[self.command.index("-o") + 1])
                    midi = mido.MidiFile(type=1, ticks_per_beat=480)

                    conductor = mido.MidiTrack()
                    conductor.append(
                        mido.MetaMessage(
                            "track_name",
                            name="Official MIROS Conductor",
                            time=0,
                        )
                    )
                    conductor.append(mido.MetaMessage("set_tempo", tempo=612_345, time=0))
                    conductor.append(mido.MetaMessage("end_of_track", time=0))
                    midi.tracks.append(conductor)

                    instrument = mido.MidiTrack()
                    instrument.append(
                        mido.MetaMessage(
                            "track_name",
                            name="Official MIROS Track",
                            time=0,
                        )
                    )
                    instrument.append(
                        mido.Message(
                            "program_change",
                            channel=2,
                            program=37,
                            time=0,
                        )
                    )
                    instrument.append(
                        mido.Message(
                            "control_change",
                            channel=2,
                            control=11,
                            value=91,
                            time=0,
                        )
                    )
                    instrument.append(
                        mido.Message(
                            "pitchwheel",
                            channel=2,
                            pitch=2_048,
                            time=0,
                        )
                    )
                    instrument.append(
                        mido.Message(
                            "note_on",
                            channel=2,
                            note=64,
                            velocity=73,
                            time=0,
                        )
                    )
                    instrument.append(
                        mido.Message(
                            "note_off",
                            channel=2,
                            note=64,
                            velocity=0,
                            time=1,
                        )
                    )
                    instrument.append(mido.MetaMessage("end_of_track", time=0))
                    midi.tracks.append(instrument)

                    attempt_path.parent.mkdir(parents=True, exist_ok=True)
                    midi.save(str(attempt_path))
                    captured["attempt_path"] = attempt_path
                    captured["official_bytes"] = attempt_path.read_bytes()
                    return "official CLI complete", ""

            transcriber = MirosTranscriber(Config())
            with (
                patch.object(MirosTranscriber, "_repo_dir", return_value=repo),
                patch.object(
                    MirosTranscriber,
                    "_entrypoint_path",
                    return_value=entrypoint,
                ),
                patch.object(
                    MirosTranscriber,
                    "get_unavailable_reason",
                    return_value="",
                ),
                patch.object(
                    subprocess,
                    "Popen",
                    side_effect=lambda command, **kwargs: FakeOfficialCliProcess(
                        command,
                        **kwargs,
                    ),
                ),
            ):
                result = transcriber.transcribe_to_midi(
                    str(audio_path),
                    str(output_path),
                )

            self.assertEqual(result, str(output_path.resolve()))
            self.assertEqual(output_path.read_bytes(), captured["official_bytes"])
            self.assertFalse(captured["attempt_path"].exists())

            published = mido.MidiFile(str(output_path))
            message_types = [message.type for track in published.tracks for message in track]
            for message_type in (
                "set_tempo",
                "track_name",
                "program_change",
                "control_change",
                "pitchwheel",
                "note_on",
                "note_off",
            ):
                self.assertIn(message_type, message_types)
            note_off = next(
                message
                for track in published.tracks
                for message in track
                if message.type == "note_off"
            )
            self.assertEqual(note_off.time, 1)

    def test_frozen_transcribe_to_midi_uses_hidden_worker_mode_not_repo_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "external" / "ai4m-miros"
            repo.mkdir(parents=True)
            entrypoint = repo / "main.py"
            entrypoint.write_text("print('miros')", encoding="utf-8")
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"wav")
            captured = {}

            class FakeProcess:
                returncode = 0

                def __init__(self, command, **kwargs):
                    captured["command"] = command

                def communicate(self):
                    _write_valid_midi(
                        Path(captured["command"][captured["command"].index("-o") + 1])
                    )
                    return "", ""

            transcriber = MirosTranscriber(Config())

            with (
                patch.object(MirosTranscriber, "_repo_dir", return_value=repo),
                patch.object(MirosTranscriber, "_entrypoint_path", return_value=entrypoint),
                patch.object(MirosTranscriber, "get_unavailable_reason", return_value=""),
                patch.object(
                    subprocess,
                    "Popen",
                    side_effect=lambda command, **kwargs: FakeProcess(command, **kwargs),
                ),
                patch.object(sys, "executable", str(root / "MusicToMidi.exe")),
                patch.object(sys, "frozen", True, create=True),
            ):
                transcriber.transcribe_to_midi(str(audio_path), str(root / "out" / "song.mid"))

        command = captured["command"]
        self.assertEqual(command[0], str(root / "MusicToMidi.exe"))
        self.assertEqual(command[1], "--miros-worker")
        self.assertIn("--status-json", command)
        self.assertNotIn(str(entrypoint), command)

    def test_frozen_transcribe_to_midi_reports_worker_status_when_stdio_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "external" / "ai4m-miros"
            repo.mkdir(parents=True)
            entrypoint = repo / "main.py"
            entrypoint.write_text("print('miros')", encoding="utf-8")
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"wav")

            class FakeProcess:
                returncode = 1

                def __init__(self, command, **kwargs):
                    self.command = command

                def communicate(self):
                    status_path = Path(self.command[self.command.index("--status-json") + 1])
                    status_path.write_text(
                        json.dumps(
                            {
                                "ok": False,
                                "error": "worker boom",
                                "traceback": "Traceback line",
                            }
                        ),
                        encoding="utf-8",
                    )
                    return "", ""

            transcriber = MirosTranscriber(Config())

            with (
                patch.object(MirosTranscriber, "_repo_dir", return_value=repo),
                patch.object(MirosTranscriber, "_entrypoint_path", return_value=entrypoint),
                patch.object(MirosTranscriber, "get_unavailable_reason", return_value=""),
                patch.object(
                    subprocess,
                    "Popen",
                    side_effect=lambda command, **kwargs: FakeProcess(command, **kwargs),
                ),
                patch.object(sys, "executable", str(root / "MusicToMidi.exe")),
                patch.object(sys, "frozen", True, create=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "worker boom") as cm:
                    transcriber.transcribe_to_midi(str(audio_path), str(root / "out" / "song.mid"))

        self.assertIn("Traceback line", str(cm.exception))

    def test_transcribe_to_midi_reports_process_output_when_midi_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "external" / "ai4m-miros"
            repo.mkdir(parents=True)
            entrypoint = repo / "main.py"
            entrypoint.write_text("print('miros')", encoding="utf-8")
            output_path = root / "out" / "song.mid"
            output_path.parent.mkdir()
            (output_path.parent / "unexpected.mid").write_bytes(b"mid")

            class FakeProcess:
                returncode = 0

                def communicate(self):
                    return "Transcribing song.wav -> song.mid\nstdout clue", "stderr clue"

            transcriber = MirosTranscriber(Config())

            with (
                patch.object(MirosTranscriber, "_repo_dir", return_value=repo),
                patch.object(MirosTranscriber, "_entrypoint_path", return_value=entrypoint),
                patch.object(MirosTranscriber, "get_unavailable_reason", return_value=""),
                patch.object(subprocess, "Popen", return_value=FakeProcess()),
            ):
                with self.assertRaisesRegex(RuntimeError, "stdout clue") as cm:
                    transcriber.transcribe_to_midi(str(root / "song.wav"), str(output_path))

        message = str(cm.exception)
        self.assertIn("MIROS 未生成 MIDI 输出", message)
        self.assertIn(str(output_path.resolve()), message)
        self.assertIn("stderr clue", message)
        self.assertIn("unexpected.mid", message)

    def test_stale_final_midi_cannot_mask_missing_current_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "external" / "ai4m-miros"
            repo.mkdir(parents=True)
            entrypoint = repo / "main.py"
            entrypoint.write_text("print('miros')", encoding="utf-8")
            output_path = root / "out" / "song.mid"
            _write_valid_midi(output_path)
            previous_bytes = output_path.read_bytes()

            class FakeProcess:
                returncode = 0

                def communicate(self):
                    return "worker exited without output", ""

            transcriber = MirosTranscriber(Config())
            with (
                patch.object(MirosTranscriber, "_repo_dir", return_value=repo),
                patch.object(MirosTranscriber, "_entrypoint_path", return_value=entrypoint),
                patch.object(MirosTranscriber, "get_unavailable_reason", return_value=""),
                patch.object(subprocess, "Popen", return_value=FakeProcess()),
            ):
                with self.assertRaisesRegex(RuntimeError, "MIROS 未生成 MIDI 输出"):
                    transcriber.transcribe_to_midi(str(root / "song.wav"), str(output_path))

            self.assertEqual(output_path.read_bytes(), previous_bytes)
            self.assertEqual(list(output_path.parent.glob(".*.miros.*.tmp.mid")), [])

    def test_transcribe_precise_reads_seconds_from_pretty_midi(self):
        pretty_midi_stub = types.ModuleType("pretty_midi")
        pretty_midi_stub.PrettyMIDI = _FakePrettyMIDI

        transcriber = MirosTranscriber(Config())

        def fake_transcribe_to_midi(audio_path, output_path, progress_callback=None):
            Path(output_path).write_bytes(b"mid")
            return output_path

        with (
            patch.object(transcriber, "transcribe_to_midi", side_effect=fake_transcribe_to_midi),
            patch.dict(
                sys.modules,
                {"pretty_midi": pretty_midi_stub},
            ),
        ):
            instrument_notes, drum_notes = transcriber.transcribe_precise("song.wav")

        self.assertIn(5, instrument_notes)
        self.assertEqual(len(instrument_notes[5]), 2)
        self.assertAlmostEqual(instrument_notes[5][0].start_time, 0.5)
        self.assertAlmostEqual(instrument_notes[5][0].end_time, 1.25)

        self.assertIn(36, drum_notes)
        self.assertEqual(len(drum_notes[36]), 1)
        self.assertAlmostEqual(drum_notes[36][0].start_time, 0.25)
        self.assertAlmostEqual(drum_notes[36][0].end_time, 0.35)


if __name__ == "__main__":
    unittest.main()
