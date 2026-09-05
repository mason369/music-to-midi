"""Run the real container entrypoint with a path probe, without a GPU or image build."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bash_and_path(path):
    if os.name == "nt":
        git = shutil.which("git")
        if not git:
            pytest.skip("Git Bash is required for Windows shell integration")
        bash = Path(git).resolve().parents[1] / "bin" / "bash.exe"
        if not bash.is_file():
            pytest.skip("Git Bash executable is unavailable")
        converted = subprocess.run(
            [bash, "-c", 'cygpath -u "$1"', "path", str(path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        return str(bash), converted.stdout.strip()
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Bash is required for container entrypoint integration")
    return bash, str(path)


@pytest.mark.parametrize("command", ["cli", "verify-models"])
def test_container_entrypoint_preserves_cli_paths_and_verifies_models(tmp_path, command):
    caller = tmp_path / "调用目录 with spaces"
    caller.mkdir()
    source = caller / "输入 audio.wav"
    tempo = caller / "节拍 source.wav"
    output = caller / "输出 MIDI"
    source.write_text("input-", encoding="utf-8")
    tempo.write_text("tempo", encoding="utf-8")
    probe_bin = tmp_path / "probe-bin"
    probe_bin.mkdir()
    probe = probe_bin / "python"
    # This probe verifies module invocation order and filesystem semantics. It
    # does not represent successful runtime/model validation or real inference.
    probe.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        '[[ "$1" == "-m" ]] || exit 90\n'
        'printf "%s\\t%s\\t%s\\n" "$PWD" "${PYTHONPATH%%:*}" "$2" >> "$PROBE_LOG"\n'
        'case "$2" in\n'
        'src.utils.source_runtime|src.model_profiles) printf "preflight %s\\n" "$2"; exit 0 ;;\n'
        'src.cli) mkdir -p -- "$7"; cat -- "$3" "$5" > "$7/result.txt"; '
        'printf \'{"event":"summary"}\\n\' ;;\n'
        '*) exit 91 ;;\nesac\n',
        encoding="utf-8",
        newline="\n",
    )
    probe.chmod(0o755)
    log = tmp_path / "invocations.tsv"
    bash, entrypoint = _bash_and_path(REPO_ROOT / "docker" / "backend-entrypoint.sh")
    _, bin_path = _bash_and_path(probe_bin)
    _, log_path = _bash_and_path(log)
    arguments = (
        [source.name, "--tempo-source", tempo.name, "-o", output.name]
        if command == "cli" else []
    )
    result = subprocess.run(
        [
            bash,
            "-c",
            'export PATH="$1:$PATH"; exec bash "$2" "${@:3}"',
            "entrypoint-test",
            bin_path,
            entrypoint,
            command,
            *arguments,
        ],
        cwd=caller,
        env={
            **os.environ,
            "PROBE_LOG": log_path,
            "MUSIC_TO_MIDI_ENABLED_PROFILES": "piano_transkun",
            "MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES": "1",
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [line.split("\t") for line in log.read_text(encoding="utf-8").splitlines()]
    expected_modules = ["src.utils.source_runtime", "src.model_profiles"]
    if command == "cli":
        expected_modules.append("src.cli")
        assert (output / "result.txt").read_text(encoding="utf-8") == "input-tempo"
        assert json.loads(result.stdout) == {"event": "summary"}
        assert "preflight src.utils.source_runtime" in result.stderr
        assert "preflight src.model_profiles" in result.stderr
    else:
        assert not output.exists()
        assert result.stdout.splitlines() == [
            "preflight src.utils.source_runtime", "preflight src.model_profiles",
        ]
    assert [row[2] for row in rows] == expected_modules
    _, expected_cwd = _bash_and_path(caller if command == "cli" else REPO_ROOT)
    _, expected_import_root = _bash_and_path(REPO_ROOT)
    assert all(row[0] == expected_cwd for row in rows)
    assert all(row[1] == expected_import_root for row in rows)
