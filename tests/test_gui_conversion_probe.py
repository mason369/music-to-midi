import sys
from unittest.mock import patch

import pytest
from PyQt6.QtCore import QTimer

from src.gui_conversion_probe import check_regex_identity, parse_probe_args, wait_for_gui


def test_probe_rejects_missing_input_without_creating_output(tmp_path):
    with pytest.raises(SystemExit) as exc:
        parse_probe_args(
            [
                "--input",
                str(tmp_path / "missing.wav"),
                "--output",
                str(tmp_path / "out"),
                "--route",
                "yourmt3:yptf_multi_ps",
            ]
        )
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()


def test_probe_refuses_to_overwrite_existing_content(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"test")
    with pytest.raises(SystemExit) as exc:
        parse_probe_args(
            ["--input", str(source), "--output", str(tmp_path), "--route", "yourmt3:yptf_multi_ps"]
        )
    assert exc.value.code == 2
    assert source.read_bytes() == b"test"


def test_probe_preserves_explicit_route_order(tmp_path):
    source = tmp_path / "source.wav"
    source.touch()
    args = parse_probe_args(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out"),
            "--route",
            "muscriptor:small",
            "--route",
            "yourmt3:yptf_multi_ps",
        ]
    )
    assert args.route == ["muscriptor:small", "yourmt3:yptf_multi_ps"]


def test_probe_regex_checks_use_actual_standard_library():
    assert check_regex_identity()["consistent"]


def test_probe_wait_runs_real_qt_events(qapp):
    state = []
    QTimer.singleShot(10, lambda: state.append(True))
    wait_for_gui(lambda: bool(state), "signal", 1)
    assert state == [True]


def test_probe_wait_timeout_is_failure(qapp):
    with pytest.raises(TimeoutError, match="timeout"):
        wait_for_gui(lambda: False, "timeout", 0.03)


def test_probe_wait_predicate_error_is_not_swallowed(qapp):
    def failed():
        raise ValueError("real predicate error")

    with pytest.raises(ValueError, match="real predicate error"):
        wait_for_gui(failed, "failure", 1)


def test_main_dispatch_preserves_probe_failure():
    import src.main as entry

    with (
        patch.object(sys, "argv", ["MusicToMidi", "--self-test-gui-conversion", "--input", "x"]),
        patch("src.gui_conversion_probe.run_gui_conversion_probe", return_value=1) as probe,
    ):
        with pytest.raises(SystemExit) as exc:
            entry.main()
    assert exc.value.code == 1
    probe.assert_called_once_with(["--input", "x"])


def test_probe_startup_error_writes_failure_report_and_traceback(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess

    source = tmp_path / "input.wav"
    source.touch()
    output = tmp_path / "probe"
    code = """
import sys
from unittest.mock import patch
from src.gui_conversion_probe import run_gui_conversion_probe
with patch('src.main._prepare_torch_runtime_before_pyqt', side_effect=RuntimeError('probe failure sentinel')):
    sys.exit(run_gui_conversion_probe(['--input', sys.argv[1], '--output', sys.argv[2], '--route', 'yourmt3:yptf_multi_ps']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(source), str(output)],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 1
    report = json.loads((output / "gui-probe.json").read_text(encoding="utf-8"))
    assert report["ok"] is False and report["stage"] == "failed"
    assert "probe failure sentinel" in report["error"]
    logs = "\n".join(path.read_text(encoding="utf-8") for path in (output / "logs").glob("*.log"))
    assert "Traceback" in logs and "probe failure sentinel" in logs
