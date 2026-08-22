"""Command-line launcher for the standalone inference service."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import uvicorn

from src.gui.web.server_runtime import configure_uvicorn_websocket_protocol
from src.utils.source_runtime import require_source_runtime_identity
from src.web_api.app import create_app
from src.web_api.server_config import (
    default_backend_config_path,
    load_backend_config,
    write_backend_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Music to MIDI inference API")
    parser.add_argument("--config", default=None, help="JSON configuration path")
    parser.add_argument("--host", default=None, help="listen IPv4/IPv6 address")
    parser.add_argument("--port", type=int, default=None, help="listen port")
    parser.add_argument("--data-dir", default=None, help="persistent job data directory")
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=None,
        help="allowed frontend origin; repeat for multiple origins",
    )
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--max-upload-bytes", type=int, default=None)
    parser.add_argument("--max-queued-jobs", type=int, default=None)
    parser.add_argument("--min-free-bytes", type=int, default=None)
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="delete terminal job families older than this many days; 0 disables age cleanup",
    )
    parser.add_argument(
        "--retention-max-jobs",
        type=int,
        default=None,
        help="maximum retained terminal job records; 0 disables count cleanup",
    )
    parser.add_argument(
        "--retention-max-bytes",
        type=int,
        default=None,
        help="maximum retained terminal-job bytes; 0 disables byte cleanup",
    )
    parser.add_argument("--save-config", action="store_true", help="persist CLI overrides")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Keep the standalone server on the same strict, pinned source runtime as
    # the desktop and headless CLI entry points.  Without this gate a global
    # Python can accept uploads successfully and then fail only inside the
    # inference worker when Beat This or another venv-only package is imported.
    require_source_runtime_identity()
    args = _parser().parse_args(argv)
    config_path = (
        Path(args.config).expanduser().resolve() if args.config else default_backend_config_path()
    )
    config = load_backend_config(config_path)
    overrides = {
        "host": args.host,
        "port": args.port,
        "data_dir": args.data_dir,
        "allowed_origins": tuple(args.cors_origin) if args.cors_origin is not None else None,
        "log_level": args.log_level,
        "max_upload_bytes": args.max_upload_bytes,
        "max_queued_jobs": args.max_queued_jobs,
        "min_free_bytes": args.min_free_bytes,
        "retention_days": args.retention_days,
        "retention_max_jobs": args.retention_max_jobs,
        "retention_max_bytes": args.retention_max_bytes,
    }
    config = replace(
        config,
        **{key: value for key, value in overrides.items() if value is not None},
    ).validated()
    if args.save_config:
        write_backend_config(config_path, config)

    logging.basicConfig(
        level=(
            logging.DEBUG
            if config.log_level == "trace"
            else getattr(logging, config.log_level.upper())
        ),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app = create_app(
        data_root=config.resolved_data_dir(),
        cors_origins=list(config.allowed_origins),
        max_upload_bytes=config.max_upload_bytes,
        max_queued_jobs=config.max_queued_jobs,
        min_free_bytes=config.min_free_bytes,
        retention_days=config.retention_days,
        retention_max_jobs=config.retention_max_jobs,
        retention_max_bytes=config.retention_max_bytes,
    )
    websocket_protocol = configure_uvicorn_websocket_protocol()
    logging.getLogger(__name__).info(
        "backend config=%s listen=%s:%d data=%s origins=%s max_upload_bytes=%d "
        "max_queued_jobs=%d min_free_bytes=%d retention_days=%d "
        "retention_max_jobs=%d retention_max_bytes=%d",
        config_path,
        config.host,
        config.port,
        config.resolved_data_dir(),
        ",".join(config.allowed_origins),
        config.max_upload_bytes,
        config.max_queued_jobs,
        config.min_free_bytes,
        config.retention_days,
        config.retention_max_jobs,
        config.retention_max_bytes,
    )
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        ws=websocket_protocol,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
