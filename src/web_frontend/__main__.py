"""Executable entry point for the standalone frontend host."""

from __future__ import annotations

import argparse
import ctypes
import logging
import signal
import sys
from dataclasses import replace
from pathlib import Path

from src.utils.runtime_paths import get_logs_dir
from src.web_frontend.config import (
    default_frontend_config_path,
    load_frontend_config,
    write_frontend_config,
)
from src.web_frontend.server import FrontendServer


def _install_windows_break_handler() -> None:
    if sys.platform != "win32":
        return
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is None:
        raise RuntimeError("当前 Windows Python 运行时缺少 SIGBREAK 支持")
    signal.signal(sigbreak, signal.default_int_handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Music to MIDI frontend application")
    parser.add_argument("--config", default=None, help="JSON configuration path")
    parser.add_argument("--host", default=None, help="frontend listen IPv4/IPv6 address")
    parser.add_argument("--port", type=int, default=None, help="frontend listen port")
    parser.add_argument("--public-host", default=None, help="client-visible frontend address")
    parser.add_argument("--api-url", default=None, help="inference backend base URL")
    parser.add_argument("--edge-path", default=None, help="Microsoft Edge executable")
    parser.add_argument("--no-window", action="store_true", help="serve without opening app window")
    parser.add_argument("--save-config", action="store_true", help="persist CLI overrides")
    return parser


def main(argv: list[str] | None = None) -> int:
    _install_windows_break_handler()
    args = _parser().parse_args(argv)
    config_path = (
        Path(args.config).expanduser().resolve() if args.config else default_frontend_config_path()
    )
    config = load_frontend_config(config_path)
    overrides = {
        "host": args.host,
        "port": args.port,
        "public_host": args.public_host,
        "backend_url": args.api_url,
        "edge_path": args.edge_path,
    }
    config = replace(
        config,
        **{key: value for key, value in overrides.items() if value is not None},
    ).validated()
    if args.save_config:
        write_frontend_config(config_path, config)
    server = FrontendServer(config)
    if args.no_window or not config.open_app_window:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.close()
        return 0
    return server.run_app_window()


def _show_fatal_error(message: str) -> None:
    logging.getLogger(__name__).critical(message, exc_info=True)
    if sys.platform == "win32" and getattr(sys, "frozen", False):
        ctypes.windll.user32.MessageBoxW(0, message, "Music to MIDI Frontend", 0x10)
    else:
        print(message, file=sys.stderr)


if __name__ == "__main__":
    log_path = get_logs_dir() / "web-frontend.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        encoding="utf-8",
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        _show_fatal_error(str(exc))
        raise SystemExit(1) from exc
