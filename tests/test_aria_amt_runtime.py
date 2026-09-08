"""Regression checks for read-only packages and failed single-file workers."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.utils.aria_amt_runtime import prepare_aria_audio_assets
from src.utils.subprocess_utils import hidden_subprocess_kwargs

ROOT = Path(__file__).resolve().parents[1]


def test_preparation_preserves_existing_assets_and_is_idempotent(tmp_path):
    root = tmp_path / "amt"
    asset = root / "assets" / "noise" / "training.wav"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"preserve the installed asset")
    first = prepare_aria_audio_assets(root)
    timestamps = [directory.stat().st_mtime_ns for directory in first]
    assert prepare_aria_audio_assets(root) == first
    assert [directory.stat().st_mtime_ns for directory in first] == timestamps
    assert asset.read_bytes() == b"preserve the installed asset"
    assert {path.name for path in first} == {"impulse", "noise", "applause"}


def test_missing_readonly_directories_fail_with_original_permission_error(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("read-only Aria package")

    monkeypatch.setattr(Path, "mkdir", denied)
    with pytest.raises(RuntimeError, match="read-only Aria package") as caught:
        prepare_aria_audio_assets(tmp_path / "amt")
    assert isinstance(caught.value.__cause__, PermissionError)
    assert "python -m src.utils.aria_amt_runtime" in str(caught.value)


def test_worker_model_failure_exits_and_preserves_the_original_traceback(tmp_path):
    # Run the actual process entrypoint and wrapper, with a deterministic failure
    # at model loading. No GPU or checkpoint download is needed for this failure.
    program = r'''
import runpy
import sys
from pathlib import Path
from unittest.mock import patch
from src.core.aria_amt_transcriber import AriaAmtTranscriber
from src.utils.aria_amt_runtime import prepare_aria_audio_assets
root = Path(sys.argv[1])
sys.argv = ["aria_amt_worker", "--checkpoint", str(root / "missing.safetensors"),
            "--input", str(root / "source.wav"), "--output-directory", str(root)]
with patch("src.core.aria_amt_transcriber.prepare_aria_audio_assets",
           side_effect=lambda: prepare_aria_audio_assets(root / "amt")), \
     patch.object(AriaAmtTranscriber, "_resolve_runtime_device", return_value="cuda:0"), \
     patch.object(AriaAmtTranscriber, "_load_aria_model",
                  side_effect=RuntimeError("ARIA_MODEL_FAILURE_PROOF")):
    runpy.run_module("src.core.aria_amt_worker", run_name="__main__")
'''
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
        timeout=30, **hidden_subprocess_kwargs(),
    )
    assert result.returncode == 1
    assert "Traceback" in result.stderr
    assert "ARIA_MODEL_FAILURE_PROOF" in result.stderr
    assert not list(tmp_path.glob("*.mid"))


def test_container_prepares_directories_before_switching_to_runtime_user():
    dockerfile = (ROOT / "docker" / "backend.Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.index("python -m src.utils.aria_amt_runtime") < dockerfile.index("USER 10001")


def test_source_worker_can_start_from_an_unrelated_working_directory(tmp_path, monkeypatch):
    from src.core.aria_amt_transcriber import AriaAmtTranscriber

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    transcriber = AriaAmtTranscriber(checkpoint_path=tmp_path / "missing.safetensors")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            transcriber._run_transcription_subprocess, tmp_path / "source.wav", tmp_path
        )
        try:
            with pytest.raises(RuntimeError, match="Aria-AMT 转写失败") as caught:
                future.result(timeout=30)
        finally:
            process = transcriber._process
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    message = str(caught.value)
    assert "aria_amt_worker.py" in message
    assert "No module named 'src'" not in message
    assert transcriber._process is None
    assert not list(tmp_path.glob("*.mid"))


def test_portable_directory_markers_cannot_be_scanned_as_audio(tmp_path):
    spec_text = (ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")
    assert 'f"amt/assets/{name}/.inference-runtime"' in spec_text
    source = ROOT / "resources" / "aria_amt" / "README.txt"
    for directory in prepare_aria_audio_assets(tmp_path / "amt"):
        marker = directory / ".inference-runtime" / source.name
        marker.parent.mkdir()
        marker.write_bytes(source.read_bytes())
        # Exactly the immediate-file scan used by the pinned upstream constructor.
        assert [path for path in directory.iterdir() if path.is_file()] == []
