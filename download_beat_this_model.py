"""Download and strictly validate the required Beat This ``final0`` checkpoint."""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from src.core.beat_this_tracker import (
    BEAT_THIS_CHECKPOINT_NAME,
    BEAT_THIS_CHECKPOINT_SHA256,
    BEAT_THIS_CHECKPOINT_SIZE,
    BEAT_THIS_CHECKPOINT_URL,
    validate_beat_this_checkpoint,
)
from src.utils.runtime_paths import get_runtime_data_dir

logger = logging.getLogger(__name__)

BEAT_THIS_DOWNLOAD_ATTEMPTS = 3
BEAT_THIS_RETRY_DELAY_SECONDS = 5


def _log(printer: Optional[Callable[[str], None]], message: str) -> None:
    if printer is not None:
        printer(message)
    else:
        logger.info(message)


def get_beat_this_download_path() -> Path:
    """Return the single writable source-runtime destination."""

    return get_runtime_data_dir() / "models" / "beat_this" / BEAT_THIS_CHECKPOINT_NAME


def is_beat_this_model_available() -> bool:
    try:
        validate_beat_this_checkpoint(get_beat_this_download_path())
        return True
    except (OSError, RuntimeError):
        return False


def _is_retriable_download_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return 500 <= int(error.code) <= 599
    return isinstance(error, (TimeoutError, urllib.error.URLError))


def download_beat_this_model(
    printer: Optional[Callable[[str], None]] = None,
) -> Path:
    """Prepare the exact official artifact, or fail without substituting a model."""

    target = get_beat_this_download_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_beat_this_model_available():
        _log(printer, f"Beat This final0 checkpoint already exists: {target}")
        return target

    if target.exists():
        _log(
            printer,
            "Existing Beat This final0 checkpoint failed identity validation; "
            f"the pinned artifact must replace it: {target}",
        )

    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)

    for attempt in range(1, BEAT_THIS_DOWNLOAD_ATTEMPTS + 1):
        _log(
            printer,
            "Downloading Beat This final0 " f"(attempt {attempt}/{BEAT_THIS_DOWNLOAD_ATTEMPTS})...",
        )
        try:
            request = urllib.request.Request(
                BEAT_THIS_CHECKPOINT_URL,
                headers={"User-Agent": "music-to-midi/beat-this-final0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with partial.open("wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
            break
        except Exception as exc:
            partial.unlink(missing_ok=True)
            if attempt >= BEAT_THIS_DOWNLOAD_ATTEMPTS or not _is_retriable_download_error(exc):
                raise
            _log(
                printer,
                "Transient Beat This final0 download error: "
                f"{exc}. Retrying in {BEAT_THIS_RETRY_DELAY_SECONDS}s...",
            )
            time.sleep(BEAT_THIS_RETRY_DELAY_SECONDS)

    try:
        validate_beat_this_checkpoint(partial)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    partial.replace(target)
    _log(
        printer,
        "Beat This final0 checkpoint saved and validated: "
        f"{target} ({BEAT_THIS_CHECKPOINT_SIZE} bytes, "
        f"SHA-256 {BEAT_THIS_CHECKPOINT_SHA256})",
    )
    return target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_beat_this_model(print)
