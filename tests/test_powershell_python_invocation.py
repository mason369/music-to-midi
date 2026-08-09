import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_powershell_scripts_do_not_pass_multiline_python_through_dash_c():
    for script_name in ("run.ps1", "install.ps1", "build_portable.ps1"):
        script = (REPO_ROOT / script_name).read_text(encoding="utf-8")
        assert not re.search(r"-c\s+\$[A-Za-z_][A-Za-z0-9_]*", script)
        assert "scripts\\powershell_helpers.ps1" in script


def test_windows_source_scripts_force_utf8_for_python_output():
    for script_name in ("run.ps1", "install.ps1"):
        script = (REPO_ROOT / script_name).read_text(encoding="utf-8")
        assert '$env:PYTHONIOENCODING = "utf-8"' in script


def test_windows_installer_passes_windows_ort_requirement_without_markers():
    script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

    assert '"onnxruntime-gpu==1.23.2"' in script
    assert "platform_system" not in script


def test_windows_installer_uses_and_cleans_a_unique_requirements_file():
    script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

    assert '"music-to-midi-requirements-{0}.txt"' in script
    assert "[System.Guid]::NewGuid()" in script
    assert "finally {" in script
    assert "Remove-Item -LiteralPath $tmpReq -Force -ErrorAction Stop" in script
    assert 'Join-Path $env:TEMP "requirements-without-aria-amt.txt"' not in script


def test_windows_installer_reads_utf8_requirements_without_merging_lines():
    script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

    assert 'Get-Content -Encoding UTF8 (Join-Path $REPO_DIR "requirements.txt")' in script


def test_source_launchers_require_the_runtime_used_by_real_pipeline_inference():
    windows = (REPO_ROOT / "run.ps1").read_text(encoding="utf-8")
    linux = (REPO_ROOT / "run.sh").read_text(encoding="utf-8")

    assert "beat_this, fastapi" in windows
    assert "beat_this, fastapi" in linux


def test_invoke_python_script_preserves_quotes_and_removes_temp_file(tmp_path):
    helper = REPO_ROOT / "scripts" / "powershell_helpers.ps1"
    working_dir = tmp_path / "working directory"
    working_dir.mkdir()
    (working_dir / "local_probe.py").write_text(
        'VALUE = "repository import preserved"\n', encoding="utf-8"
    )
    temp_dir = tmp_path / "powershell helper temp"
    temp_dir.mkdir(exist_ok=True)
    before = set(temp_dir.glob("music-to-midi-python-*.py"))
    command = "\n".join(
        (
            f"$env:TEMP = '{temp_dir}'",
            f"$env:TMP = '{temp_dir}'",
            "$env:PYTHONPATH = 'existing-path-sentinel'",
            f". '{helper}'",
            '$source = \'import local_probe\nvalues = ("small", "medium", "large")\nprint("|".join(values), local_probe.VALUE)\'',
            f"$code = Invoke-PythonScript -PythonExecutable '{sys.executable}' -Script $source",
            "if ($code -ne 0) { exit $code }",
            "if ($env:PYTHONPATH -ne 'existing-path-sentinel') { exit 9 }",
        )
    )

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "small|medium|large repository import preserved" in completed.stdout
    assert set(temp_dir.glob("music-to-midi-python-*.py")) == before
