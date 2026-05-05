"""Download ByteDance pedal-aware piano transcription checkpoint."""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from src.core.bytedance_piano_transcriber import (
    BYTEDANCE_PIANO_CHECKPOINT_NAME,
    BYTEDANCE_PIANO_CHECKPOINT_URL,
    BYTEDANCE_PIANO_MIN_CHECKPOINT_BYTES,
    ByteDancePianoTranscriber,
)

logger = logging.getLogger(__name__)

BYTEDANCE_PIANO_DOWNLOAD_ATTEMPTS = 3
BYTEDANCE_PIANO_RETRY_DELAY_SECONDS = 5


def _log(printer: Optional[Callable[[str], None]], message: str) -> None:
    if printer:
        printer(message)
        return
    logger.info(message)


def get_bytedance_piano_model_path() -> Path:
    return ByteDancePianoTranscriber.default_checkpoint_path()


def is_bytedance_piano_model_available() -> bool:
    path = get_bytedance_piano_model_path()
    return path.exists() and path.stat().st_size >= BYTEDANCE_PIANO_MIN_CHECKPOINT_BYTES


def _is_retriable_download_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return 500 <= int(error.code) <= 599
    return isinstance(error, (TimeoutError, urllib.error.URLError))


def download_bytedance_piano_model(
    printer: Optional[Callable[[str], None]] = None,
) -> Path:
    target = get_bytedance_piano_model_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if is_bytedance_piano_model_available():
        _log(printer, f"ByteDance Piano checkpoint already exists: {target}")
        return target

    partial = target.with_suffix(target.suffix + ".download")
    if partial.exists():
        partial.unlink()

    for attempt in range(1, BYTEDANCE_PIANO_DOWNLOAD_ATTEMPTS + 1):
        _log(
            printer,
            f"Downloading {BYTEDANCE_PIANO_CHECKPOINT_NAME} "
            f"(attempt {attempt}/{BYTEDANCE_PIANO_DOWNLOAD_ATTEMPTS})...",
        )
        try:
            with urllib.request.urlopen(BYTEDANCE_PIANO_CHECKPOINT_URL, timeout=120) as response:
                with partial.open("wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
            break
        except Exception as exc:
            if partial.exists():
                partial.unlink()
            if attempt >= BYTEDANCE_PIANO_DOWNLOAD_ATTEMPTS or not _is_retriable_download_error(exc):
                raise
            _log(
                printer,
                "Transient ByteDance Piano checkpoint download error: "
                f"{exc}. Retrying in {BYTEDANCE_PIANO_RETRY_DELAY_SECONDS}s...",
            )
            time.sleep(BYTEDANCE_PIANO_RETRY_DELAY_SECONDS)

    size = partial.stat().st_size
    if size < BYTEDANCE_PIANO_MIN_CHECKPOINT_BYTES:
        partial.unlink()
        raise RuntimeError(
            "Downloaded ByteDance Piano checkpoint is incomplete: "
            f"{size} bytes < {BYTEDANCE_PIANO_MIN_CHECKPOINT_BYTES} bytes"
        )

    partial.replace(target)
    _log(printer, f"ByteDance Piano checkpoint saved: {target}")
    return target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_bytedance_piano_model(print)
