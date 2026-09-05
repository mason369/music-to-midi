"""Guard native background process behavior, including real Windows child processes."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.utils import subprocess_utils


@pytest.mark.parametrize("platform", ["nt", "posix"])
def test_background_options_are_platform_specific(monkeypatch, platform):
    monkeypatch.setattr(subprocess_utils, "os", SimpleNamespace(name=platform))
    monkeypatch.setattr(
        subprocess_utils, "subprocess", SimpleNamespace(CREATE_NO_WINDOW=0x08000000)
    )
    assert subprocess_utils.hidden_subprocess_kwargs() == (
        {"creationflags": 0x08000000} if platform == "nt" else {}
    )


def test_background_child_preserves_output_and_failure_exit_code():
    source = (
        "import ctypes, json, os, sys; "
        "window = ctypes.windll.kernel32.GetConsoleWindow() if os.name == 'nt' else 0; "
        "print(json.dumps({'console_window': window, 'text': '中文输出'})); "
        "print('中文错误', file=sys.stderr); sys.exit(7)"
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        **subprocess_utils.hidden_subprocess_kwargs(),
    )
    assert result.returncode == 7
    assert json.loads(result.stdout) == {"console_window": 0, "text": "中文输出"}
    assert result.stderr.strip() == "中文错误"


def test_background_child_can_be_cancelled_and_reaped():
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.read()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **subprocess_utils.hidden_subprocess_kwargs(),
    )
    try:
        process.terminate()
        process.communicate(timeout=10)
        assert process.poll() is not None
        assert process.returncode != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_background_child_timeout_remains_an_error():
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            capture_output=True,
            timeout=0.1,
            **subprocess_utils.hidden_subprocess_kwargs(),
        )


@pytest.mark.parametrize(
    "failure",
    [
        None,
        FileNotFoundError("missing ffmpeg"),
        subprocess.CalledProcessError(3, "ffmpeg", output="broken runtime"),
    ],
)
def test_separator_ffmpeg_probe_uses_background_policy_and_exposes_errors(monkeypatch, failure):
    from src.utils import audio_separator_compat as compat

    # This unit exercises our process adapter only. Loading the GPU stack after
    # pytest-qt would violate the desktop's native-runtime-before-Qt bootstrap;
    # real separator inference is validated through the normal application path.
    upstream = ModuleType("audio_separator.separator")
    upstream.Separator = type("Separator", (), {})
    monkeypatch.setitem(sys.modules, "audio_separator.separator", upstream)
    separator_cls = compat.get_separator_cls()
    separator = object.__new__(separator_cls)
    separator.logger = logging.getLogger("separator-probe-test")
    seen = {}

    def run(command, **kwargs):
        seen.update(kwargs)
        assert command[-1] == "-version"
        if failure is not None:
            raise failure
        return subprocess.CompletedProcess(command, 0, "ffmpeg version test\n")

    monkeypatch.setattr(compat.subprocess, "run", run)
    if failure is None:
        separator.check_ffmpeg_installed()
    else:
        with pytest.raises(type(failure)):
            separator.check_ffmpeg_installed()
    assert seen["check"] is True
    for key, value in subprocess_utils.hidden_subprocess_kwargs().items():
        assert seen[key] == value
