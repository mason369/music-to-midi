"""Exercise the shipped launchers with the real CLI and Windows PowerShell 5.1."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher contract")


@pytest.mark.parametrize("wrapper", ["ps1", "bat"])
def test_windows_launchers_preserve_trailing_slashes_in_quoted_directories(tmp_path, wrapper):
    directory = tmp_path / "音频目录 with spaces"
    directory.mkdir()
    audio = directory / "sample.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(b"\0\0" * 441)
    result = _launch(wrapper, "batch", "--dry-run", "--json", str(directory) + "\\", "--recursive")
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert [event["event"] for event in events] == ["planned", "plan_summary"]
    assert Path(events[0]["source"]) == audio


@pytest.mark.parametrize("wrapper", ["ps1", "bat"])
@pytest.mark.parametrize("use_psdrive", [False, True])
def test_windows_launchers_use_powershell_current_location(tmp_path, wrapper, use_psdrive):
    caller = tmp_path / "调用目录 with spaces"
    caller.mkdir()
    source = caller / "音频 source.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(b"\0\0" * 441)

    quoted_caller = "'" + str(caller).replace("'", "''") + "'"
    if use_psdrive:
        location = (
            "New-PSDrive -Name AudioReview -PSProvider FileSystem -Root "
            + quoted_caller
            + " | Out-Null\nSet-Location -LiteralPath 'AudioReview:\\'\n"
        )
    else:
        location = "Set-Location -LiteralPath " + quoted_caller + "\n"
    launcher = str(REPO_ROOT / f"cli.{wrapper}").replace("'", "''")
    command = (
        location
        + f"& '{launcher}' 'batch' '{source.name}' '-o' '输出 MIDI' '--dry-run' '--json'"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert [event["event"] for event in events] == ["planned", "plan_summary"]
    assert Path(events[0]["source"]) == source
    assert not (caller / "输出 MIDI").exists(), "Dry runs must not create output directories"


def _launch(wrapper: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    if wrapper == "ps1":
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(REPO_ROOT / "cli.ps1"),
        ]
    else:
        command = ["cmd.exe", "/d", "/c", str(REPO_ROOT / "cli.bat")]
    return subprocess.run(
        [*command, *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


@pytest.mark.parametrize("wrapper", ["ps1", "bat"])
@pytest.mark.parametrize("output_flag,display_flag", [("-o", "--verbose"), ("--output", "--quiet")])
def test_windows_launchers_forward_native_flags_and_unicode_paths(
    tmp_path, wrapper, output_flag, display_flag
):
    audio = tmp_path / "音频 (a & b)'s file.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(b"\0\0" * 441)
    output = tmp_path / "输出 with spaces"
    result = _launch(
        wrapper,
        "batch",
        str(audio),
        output_flag,
        str(output),
        "--recursive",
        "--dry-run",
        "--json",
        display_flag,
    )
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert [event["event"] for event in events] == ["planned", "plan_summary"]
    assert Path(events[0]["source"]) == audio
    assert events[-1]["total"] == 1
    assert not output.exists(), "Planning must not execute a conversion or create its output"


@pytest.mark.parametrize("wrapper", ["ps1", "bat"])
def test_windows_launchers_forward_help_version_and_native_error_codes(wrapper):
    for arguments in [(), ("--help",), ("--version",), ("batch", "--help")]:
        result = _launch(wrapper, *arguments)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()
    invalid = _launch(wrapper, "batch", "--invalid-cli-argument")
    assert invalid.returncode == 2
    assert invalid.stderr.strip()


@pytest.mark.parametrize(
    "arguments",
    [
        ("-Accelerator", "cuda"),
        ("-Accelerator:CUDA",),
        ("-Accelerator", "cuda", "-VenvName", "venv"),
        ("-VenvName:venv",),
    ],
)
def test_windows_launcher_explicit_environment_selection(arguments):
    result = _launch("ps1", *arguments, "routes", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["count"] == 13


def test_windows_launcher_rejects_cuda_environment_for_xpu():
    result = _launch("ps1", "-Accelerator", "xpu", "-VenvName", "venv", "routes", "--json")
    assert result.returncode == 2
    assert "venv-xpu" in result.stderr
    assert not result.stdout.strip()


@pytest.mark.parametrize(
    "arguments,exit_code",
    [
        (("-Accelerator",), 2),
        (("-Accelerator", "invalid"), 2),
        (("-VenvName",), 2),
        (("-VenvName", "../venv"), 2),
        (("-VenvName", "does-not-exist-issue-11"), 1),
    ],
)
def test_windows_launcher_reports_invalid_or_missing_environment(arguments, exit_code):
    result = _launch("ps1", *arguments)
    assert result.returncode == exit_code
    assert result.stderr.strip()
    assert not result.stdout.strip()
