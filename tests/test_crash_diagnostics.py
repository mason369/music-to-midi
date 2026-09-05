"""Exercise native diagnostics in isolated processes, never in the test runner."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run_child(tmp_path, body):
    prefix = (
        "import sys, os, faulthandler\n"
        "from pathlib import Path\n"
        "from src.utils.crash_diagnostics import install_native_fault_log\n"
        "directory = Path(sys.argv[1])\n"
    )
    return subprocess.run(
        [sys.executable, "-c", prefix + body, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )


def test_fault_log_survives_gc_and_is_idempotent(tmp_path):
    result = _run_child(
        tmp_path,
        "p = install_native_fault_log(directory)\n"
        "assert p == install_native_fault_log(directory / '.')\n"
        "import gc; gc.collect()\n"
        "from src.utils import crash_diagnostics as diagnostics\n"
        "faulthandler.dump_traceback(file=diagnostics._fault_stream)\n",
    )
    assert result.returncode == 0, result.stderr
    logs = list(tmp_path.glob("native-fault-*.log"))
    assert len(logs) == 1
    content = logs[0].read_text(encoding="utf-8")
    assert "PID=" in content and "Python=" in content
    assert 'File "<string>"' in content


def test_native_abort_preserves_failure_and_traceback(tmp_path):
    result = _run_child(
        tmp_path,
        "if os.name == 'nt':\n"
        "    import ctypes\n"
        "    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)\n"
        "    ctypes.CDLL('ucrtbase')._set_abort_behavior(0, 3)\n"
        "install_native_fault_log(directory)\n"
        "def deliberate_native_failure():\n"
        "    os.abort()\n"
        "deliberate_native_failure()\n",
    )
    assert result.returncode != 0
    logs = list(tmp_path.glob("native-fault-*.log"))
    assert len(logs) == 1
    content = logs[0].read_text(encoding="utf-8")
    assert "Fatal Python error: Aborted" in content
    assert "deliberate_native_failure" in content


@pytest.mark.parametrize("failure", ["directory", "enable", "redirect"])
def test_diagnostic_setup_failure_is_explicit(tmp_path, failure):
    bodies = {
        "directory": "directory.write_text('occupied')\ninstall_native_fault_log(directory)\n",
        "enable": (
            "from unittest.mock import patch\n"
            "with patch('faulthandler.enable', side_effect=OSError('handler rejected')):\n"
            "    install_native_fault_log(directory)\n"
        ),
        "redirect": (
            "install_native_fault_log(directory)\n"
            "install_native_fault_log(directory / 'different')\n"
        ),
    }
    target = tmp_path / "logs"
    result = _run_child(target, bodies[failure])
    assert result.returncode != 0
    assert "Traceback" in result.stderr


def test_gui_installs_fault_handler_before_native_runtime(tmp_path):
    result = _run_child(
        tmp_path,
        "from unittest.mock import patch\n"
        "import src.main as entry\n"
        "def stop_before_qt():\n"
        "    assert faulthandler.is_enabled()\n"
        "    assert list(directory.glob('native-fault-*.log'))\n"
        "    raise RuntimeError('native runtime sentinel')\n"
        "with patch.object(sys, 'argv', ['MusicToMidi']), "
        "patch.object(entry, 'get_logs_dir', return_value=directory), "
        "patch.object(entry, '_prepare_torch_runtime_before_pyqt', stop_before_qt):\n"
        "    entry.main()\n",
    )
    assert result.returncode != 0
    assert "native runtime sentinel" in result.stderr
    assert list(tmp_path.glob("native-fault-*.log"))
