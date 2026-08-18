"""Command-line launcher for the standalone inference service."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from src.gui.web.server_runtime import configure_uvicorn_websocket_protocol
from src.utils.source_runtime import require_source_runtime_identity
from src.web_api.app import create_app


def main() -> None:
    # Keep the standalone server on the same strict, pinned source runtime as
    # the desktop and headless CLI entry points.  Without this gate a global
    # Python can accept uploads successfully and then fail only inside the
    # inference worker when Beat This or another venv-only package is imported.
    require_source_runtime_identity()
    parser = argparse.ArgumentParser(description="Run the Music to MIDI inference API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--serve-frontend",
        action="store_true",
        help="also serve the separately built static client from / (optional convenience)",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    frontend_root = None
    if args.serve_frontend:
        frontend_root = Path(__file__).resolve().parents[2] / "web"
    app = create_app(data_root=args.data_dir, frontend_root=frontend_root)
    websocket_protocol = configure_uvicorn_websocket_protocol()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        ws=websocket_protocol,
    )


if __name__ == "__main__":
    main()
