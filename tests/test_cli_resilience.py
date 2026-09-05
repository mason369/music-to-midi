"""CLI fault boundaries: real files, durable state, and machine-readable streams."""

from __future__ import annotations

import errno
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mido
import pytest

from src.cli import app


class RecordingEngine:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        path = kwargs["output_dir"] / "result.mid"
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        track.extend([
            mido.Message("note_on", note=60, velocity=90),
            mido.Message("note_off", note=60, time=480),
        ])
        midi.tracks.append(track)
        midi.save(path)
        return SimpleNamespace(
            result={"total_notes": 1},
            artifacts=(SimpleNamespace(id="midi", kind="midi", path=path),),
        )


def invoke(source, output, engine, *extra):
    stdout, stderr = io.StringIO(), io.StringIO()
    code = app.main(
        ["convert", str(source), "-o", str(output), *extra],
        engine_factory=lambda: engine,
        stdout=stdout,
        stderr=stderr,
        install_signal_handlers=False,
    )
    return code, stdout.getvalue(), stderr.getvalue()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "input.wav"
    path.write_bytes(b"input; engine boundary is injected")
    return path


@pytest.mark.parametrize("corruption", [b"null", b"[]", b"42", b'"text"', b"\xff\xfe"])
def test_corrupt_resume_is_reported_and_preserved_without_blocking_new_job(
    tmp_path, source, corruption
):
    output, engine = tmp_path / "out", RecordingEngine()
    assert invoke(source, output, engine)[0] == 0
    manifest = output / "input" / app.MANIFEST_NAME
    manifest.write_bytes(corruption)

    code, _, error = invoke(source, output, engine)

    assert code == 0, error
    assert len(engine.calls) == 2
    assert "无法通过校验" in error
    assert manifest.read_bytes() == corruption
    assert (output / "input-2" / app.MANIFEST_NAME).is_file()


@pytest.mark.parametrize("schema", [None, 999, "1", True])
def test_unknown_manifest_schema_cannot_skip_a_conversion(tmp_path, source, schema):
    output, engine = tmp_path / "out", RecordingEngine()
    assert invoke(source, output, engine)[0] == 0
    path = output / "input" / app.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema"] = schema
    path.write_text(json.dumps(manifest), encoding="utf-8")

    code, _, error = invoke(source, output, engine)

    assert code == 0, error
    assert len(engine.calls) == 2
    assert "无法通过校验" in error


def test_unreadable_candidate_does_not_block_a_later_verified_resume(
    tmp_path, source, monkeypatch
):
    output, engine = tmp_path / "out", RecordingEngine()
    assert invoke(source, output, engine)[0] == 0
    assert invoke(source, output, engine, "--rerun")[0] == 0
    original = app._sha256_file
    blocked = output / "input" / "result.mid"

    def read_hash(path):
        if path == blocked:
            raise PermissionError(errno.EACCES, "locked output", str(path))
        return original(path)

    monkeypatch.setattr(app, "_sha256_file", read_hash)
    code, text, error = invoke(source, output, engine)

    assert code == 0, error
    assert len(engine.calls) == 2
    assert "跳过" in text
    assert "无法通过校验" in error


def test_unreadable_subdirectory_fails_preflight_without_partial_success(tmp_path, monkeypatch):
    library = tmp_path / "library"
    blocked = library / "blocked"
    blocked.mkdir(parents=True)
    (library / "good.wav").write_bytes(b"readable")
    (blocked / "hidden.wav").write_bytes(b"inaccessible")
    original = os.scandir

    def scan(path):
        if Path(path) == blocked:
            raise PermissionError(errno.EACCES, "denied subtree", str(path))
        return original(path)

    monkeypatch.setattr(os, "scandir", scan)
    output, engine = tmp_path / "out", RecordingEngine()
    code, _, error = invoke(library, output, engine, "--recursive", "--dry-run")

    assert code == 2
    assert "denied subtree" in error
    assert not engine.calls
    assert not output.exists()


def test_empty_tempo_source_fails_even_in_dry_run(tmp_path, source):
    original = tmp_path / "original.wav"
    original.touch()
    output = tmp_path / "out"
    text, error = io.StringIO(), io.StringIO()
    engine = RecordingEngine()
    code = app.main(
        ["track-to-midi", str(source), "--route", "piano_transkun",
         "--tempo-source", str(original), "-o", str(output), "--dry-run"],
        engine_factory=lambda: engine, stdout=text, stderr=error,
        install_signal_handlers=False,
    )
    assert code == 2
    assert "empty" in error.getvalue()
    assert not engine.calls
    assert not output.exists()


def test_json_captures_engine_initialization_diagnostics(tmp_path, source, capsys):
    output, error = io.StringIO(), io.StringIO()

    def create_engine():
        print("initialization diagnostic")
        return RecordingEngine()

    code = app.main(
        [str(source), "-o", str(tmp_path / "out"), "--json"],
        engine_factory=create_engine, stdout=output, stderr=error,
        install_signal_handlers=False,
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    assert "initialization diagnostic" in error.getvalue()
    assert json.loads(output.getvalue().splitlines()[-1])["event"] == "summary_path"


def test_machine_stdout_excludes_native_and_inherited_child_diagnostics(tmp_path, source):
    repo = Path(__file__).resolve().parents[1]
    script = """
import os, runpy, subprocess, sys
from src.cli.app import main
Engine = runpy.run_path('tests/test_cli_resilience.py')['RecordingEngine']
class NativeEngine(Engine):
    def run(self, **kwargs):
        os.write(1, b'native diagnostic\\n')
        subprocess.run([sys.executable, '-c', "print('child diagnostic')"], check=True)
        return super().run(**kwargs)
raise SystemExit(main(sys.argv[1:], engine_factory=NativeEngine))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), "-o", str(tmp_path / "out"), "--json"],
        cwd=repo, capture_output=True, text=True, encoding="utf-8", timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
    )
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["event"] == "summary_path"
    assert "native diagnostic" in result.stderr
    assert "child diagnostic" in result.stderr


def test_summary_write_failure_never_relabels_a_committed_success(tmp_path, source, monkeypatch):
    output, engine = tmp_path / "out", RecordingEngine()
    original = app._write_run_summary

    def write_summary(path, summary, counts):
        if counts.succeeded:
            raise OSError(errno.ENOSPC, "disk full when saving batch summary")
        return original(path, summary, counts)

    monkeypatch.setattr(app, "_write_run_summary", write_summary)
    code, _, error = invoke(source, output, engine)

    assert code != 0
    assert "disk full" in error
    manifest = json.loads((output / "input" / app.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["error"] is None
    assert len(engine.calls) == 1


def test_closed_json_output_pipe_stops_once_and_reports_to_stderr(tmp_path, source):
    class ClosedOutput(io.StringIO):
        def write(self, value):
            raise BrokenPipeError(errno.EPIPE, "reader closed pipe")

    error = io.StringIO()
    engine = RecordingEngine()
    code = app.main(
        [str(source), "-o", str(tmp_path / "out"), "--json", "--dry-run"],
        engine_factory=lambda: engine, stdout=ClosedOutput(), stderr=error,
        install_signal_handlers=False,
    )
    assert code == 1
    assert error.getvalue().strip()
    assert not engine.calls


def test_broken_stdout_after_commit_preserves_successful_checkpoint(tmp_path, source):
    class DisconnectedReader(io.StringIO):
        disconnected = False

        def write(self, value):
            if '"event": "succeeded"' in value:
                self.disconnected = True
            if self.disconnected:
                raise BrokenPipeError(errno.EPIPE, "reader closed pipe")
            return super().write(value)

    error, output = io.StringIO(), tmp_path / "out"
    code = app.main(
        [str(source), "-o", str(output), "--json"],
        engine_factory=RecordingEngine, stdout=DisconnectedReader(), stderr=error,
        install_signal_handlers=False,
    )
    assert code == 1
    assert error.getvalue().strip()
    manifest = json.loads((output / "input" / app.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    summary = json.loads(next(output.glob("batch-run-*.json")).read_text(encoding="utf-8"))
    assert summary["counts"]["succeeded"] == 1
    assert summary["counts"]["failed"] == 0
    assert len(summary["items"]) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows UCRT reports EINVAL for a closed pipe")
def test_real_windows_closed_stdout_pipe_has_clean_nonzero_exit(source):
    repo = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "src.cli", str(source), "--json", "--dry-run"],
        cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    process.stdout.close()
    error = process.stderr.read().decode("utf-8")
    assert process.wait(timeout=15) == 1, error
    assert "管道" in error
    assert "Traceback" not in error and "Exception ignored" not in error


def test_engine_einval_is_preserved_as_a_model_failure(tmp_path, source):
    class InvalidEngine(RecordingEngine):
        def run(self, **kwargs):
            raise OSError(errno.EINVAL, "invalid model argument")

    code, text, _ = invoke(source, tmp_path / "out", InvalidEngine(), "--json")
    assert code == 1
    events = [json.loads(line) for line in text.splitlines()]
    assert events[-1]["status"] == "failed"
    assert "invalid model argument" in next(event["error"] for event in events if event["event"] == "failed")


@pytest.mark.skipif(os.name != "nt", reason="Windows UCRT closed-pipe boundary")
def test_real_windows_closed_human_output_pipe_has_clean_nonzero_exit(source):
    process = subprocess.Popen(
        [sys.executable, "-m", "src.cli", str(source), "--dry-run"],
        cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    process.stdout.close()
    error = process.stderr.read().decode("utf-8")
    assert process.wait(timeout=15) == 1, error
    assert "管道" in error
    assert "Traceback" not in error and "Exception ignored" not in error
