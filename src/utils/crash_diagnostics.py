"""Keep a process-lifetime traceback sink for desktop native failures."""

import atexit
import faulthandler
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from src import __version__

_fault_stream: TextIO | None = None
_fault_path: Path | None = None


def _close_fault_log() -> None:
    """Disable the handler before closing its descriptor during normal exit."""
    global _fault_stream
    if _fault_stream is not None:
        faulthandler.disable()
        _fault_stream.close()
        _fault_stream = None


def install_native_fault_log(log_dir: str | Path) -> Path:
    """Install once; failures propagate rather than silently losing diagnostics.

    Python's handler records supported native faults, not every possible OS
    termination (for example, process kill or Windows fail-fast termination).
    The stream stays open until normal exit, as required by faulthandler.
    """
    global _fault_stream, _fault_path
    directory = Path(log_dir).resolve()
    if _fault_stream is not None:
        if _fault_path is None or _fault_path.parent != directory:
            raise RuntimeError("原生故障日志已经启用，不能切换日志目录")
        return _fault_path

    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"native-fault-{stamp}-{os.getpid()}.log"
    stream = path.open("x", encoding="utf-8")
    try:
        stream.write(
            f"原生故障记录已启用 UTC={stamp} PID={os.getpid()} "
            f"应用版本={__version__} Python={sys.version}\n"
        )
        stream.flush()
        faulthandler.enable(file=stream, all_threads=True)
    except BaseException:
        stream.close()
        raise
    _fault_stream, _fault_path = stream, path
    atexit.register(_close_fault_log)
    return path
