from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.web_frontend import __main__ as frontend
from src.web_frontend.config import FrontendServerConfig


@pytest.fixture
def frozen_frontend(monkeypatch, tmp_path):
    message_box = Mock()
    monkeypatch.setattr(frontend.sys, "platform", "win32")
    monkeypatch.setattr(frontend.sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        frontend.ctypes,
        "windll",
        SimpleNamespace(user32=SimpleNamespace(MessageBoxW=message_box)),
        raising=False,
    )
    monkeypatch.setattr(frontend, "get_logs_dir", lambda: tmp_path)
    monkeypatch.setattr(frontend, "_install_windows_break_handler", lambda: None)
    return message_box


@pytest.mark.parametrize("failure_stage", ["logging", "config", "server"])
def test_no_window_startup_failures_exit_nonzero_without_dialog(
    monkeypatch, tmp_path, caplog, frozen_frontend, failure_stage
):
    error = RuntimeError("explicit startup failure")
    monkeypatch.setattr(frontend, "load_frontend_config", lambda _path: FrontendServerConfig())
    if failure_stage == "logging":
        monkeypatch.setattr(frontend, "get_logs_dir", Mock(side_effect=error))
    elif failure_stage == "config":
        monkeypatch.setattr(frontend, "load_frontend_config", Mock(side_effect=error))
    else:
        monkeypatch.setattr(frontend, "FrontendServer", Mock(side_effect=error))

    assert frontend.main(["--no-window", "--config", str(tmp_path / "frontend.json")]) == 1
    frozen_frontend.assert_not_called()
    assert "explicit startup failure" in caplog.text


def test_configured_headless_failure_closes_server_and_never_opens_a_window(
    monkeypatch, tmp_path, caplog, frozen_frontend
):
    monkeypatch.setattr(
        frontend, "load_frontend_config", lambda _path: FrontendServerConfig(open_app_window=False)
    )
    server = Mock()
    server.serve_forever.side_effect = OSError("listener failed")
    monkeypatch.setattr(frontend, "FrontendServer", lambda _config: server)

    assert frontend.main(["--config", str(tmp_path / "frontend.json")]) == 1
    server.close.assert_called_once_with()
    server.run_app_window.assert_not_called()
    frozen_frontend.assert_not_called()
    assert "listener failed" in caplog.text


def test_windowed_frontend_still_reports_errors_with_a_native_dialog(
    monkeypatch, tmp_path, frozen_frontend
):
    monkeypatch.setattr(frontend, "load_frontend_config", lambda _path: FrontendServerConfig())
    monkeypatch.setattr(frontend, "FrontendServer", Mock(side_effect=OSError("listener failed")))

    assert frontend.main(["--config", str(tmp_path / "frontend.json")]) == 1
    frozen_frontend.assert_called_once()
    assert frozen_frontend.call_args.args[1] == "listener failed"


def test_no_window_stop_during_startup_exits_without_an_error_dialog(
    monkeypatch, tmp_path, frozen_frontend
):
    monkeypatch.setattr(frontend, "load_frontend_config", Mock(side_effect=KeyboardInterrupt()))

    assert frontend.main(["--no-window", "--config", str(tmp_path / "frontend.json")]) == 0
    frozen_frontend.assert_not_called()
