from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_pyinstaller_builds_console_cli_for_cuda_and_xpu_in_shared_collection():
    spec = _read("MusicToMidi.spec")

    assert "cli_name='MusicToMidiCLI'" in spec
    assert 'cli_name = "MusicToMidiCLIXpu"' in spec
    assert "cli_exe = EXE(" in spec
    assert "name=cli_name" in spec
    assert "console=True" in spec
    assert "cli_exe," in spec
    assert "'src.cli.app'" in spec


def test_source_launchers_are_native_on_windows_and_linux():
    powershell = _read("cli.ps1")
    batch = _read("cli.bat")
    shell = _read("cli.sh")
    pyproject = _read("pyproject.toml")

    assert '@("-m", "src.cli") + $arguments' in powershell
    assert "UseShellExecute = $false" in powershell
    assert "ConvertTo-NativeArgument" in powershell
    assert '"run.ps1"' not in powershell
    assert "MUSIC_TO_MIDI_ACCELERATOR" in powershell
    assert "PYTHONPATH" in powershell
    assert '"%~dp0cli.ps1" %*' in batch
    assert '"$VENV_PYTHON" -m src.cli "$@"' in shell
    assert 'run.sh" --cli "$@"' not in shell
    assert "global Python and CPU fallback are not used" in shell
    assert "Intentionally no console-script entry" in pyproject


def test_portable_roles_and_universal_package_require_the_cli():
    separator = _read("scripts/separate_windows_portable_roles.ps1")
    portable = _read("build_portable.ps1")
    web_build = _read("build_web_executables.ps1")
    universal = _read("build_universal_windows.ps1")
    launcher_build = _read("scripts/build_universal_windows_launchers.ps1")
    launcher_source = _read("tools/universal_launcher/UniversalLauncher.cs")

    assert '"MusicToMidiCLIXpu.exe"' in separator
    assert "Portable role is missing its native CLI executable" in separator
    assert "Portable native CLI command contract verified" in portable
    assert "$AppCliExe" in web_build and "$BackendCliExe" in web_build
    assert "Portable source role is missing its native CLI" in universal
    assert 'Join-Path $appRoot "MusicToMidiCLI.exe"' in universal
    assert 'Join-Path $backendRoot "MusicToMidiCLI.exe"' in universal
    assert "/define:CLI_LAUNCHER" in launcher_build
    assert 'return "cli";' in launcher_source
    assert '"MusicToMidiCLIXpu.exe"' in launcher_source


def test_release_smokes_both_role_clis_and_documents_cross_platform_commands():
    workflow = _read(".github/workflows/release.yml")
    universal_usage = _read("resources/universal/README-使用说明.txt")

    assert 'test -x "$APP_ROOT/MusicToMidiCLI"' in workflow
    assert 'test -x "$BACKEND_ROOT/MusicToMidiCLI"' in workflow
    assert "$routeContract.count -ne 13" in workflow
    assert 'CLI_ROUTES_JSON="$($cli_exe routes --json)"' in workflow
    assert "MusicToMidiCLI.exe batch D:\\Audio" in workflow
    assert "./MusicToMidiCLI batch /data/audio" in workflow
    assert "MusicToMidiCLI.exe D:\\Audio\\song.wav" in universal_usage


def test_docker_entrypoint_exposes_cli_without_cpu_or_model_fallback():
    entrypoint = _read("docker/backend-entrypoint.sh")
    workflow = _read(".github/workflows/container.yml")
    zh_guide = _read("docs/docker-deployment.md")
    en_guide = _read("docs/docker-deployment.en.md")

    assert "    cli)" in entrypoint
    assert 'exec python -m src.cli "$@"' in entrypoint
    assert "--require-ready" in entrypoint
    assert "-m src.cli routes --json" in workflow
    assert 'payload.get("count") != 13' in workflow
    assert "backend cli batch /input --recursive -o /output" in zh_guide
    assert "backend cli batch /input --recursive -o /output" in en_guide
    assert "不会改用 CPU" in zh_guide
    assert "uses the selected model and the configured GPU runtime" in en_guide
    assert "If either is unavailable, it stops with an error" in en_guide


def test_chinese_and_english_readmes_describe_resume_and_explicit_stem_contract():
    zh = _read("README.md")
    en = _read("docs/README.md")

    for text in (zh, en):
        assert "music-to-midi-job.json" in text
        assert "batch-run-*.json" in text
        assert "track-to-midi" in text
        assert "SHA-256" in text
        assert "MusicToMidiCLI" in text
