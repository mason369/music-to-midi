"""Cross-process accelerator ownership for desktop and HTTP inference entry points."""

from __future__ import annotations

import errno
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

CancelCheck = Callable[[], bool]
WaitCallback = Callable[[], None]

_LOCAL_GATES_GUARD = threading.Lock()
_LOCAL_GATES: dict[Path, threading.Lock] = {}
_THREAD_STATE = threading.local()


def accelerator_lock_path(device_index: int) -> Path:
    """Return the stable, per-user lock path for one accelerator index."""
    normalized_index = int(device_index)
    if normalized_index < 0:
        raise ValueError(f"accelerator device index must be non-negative: {device_index}")
    configured_root = os.environ.get("MUSIC_TO_MIDI_INFERENCE_LOCK_DIR", "").strip()
    root = (
        Path(os.path.expandvars(configured_root)).expanduser().resolve()
        if configured_root
        else (Path.home() / ".music-to-midi" / "locks").resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    return root / f"accelerator-{normalized_index}.lock"


def _local_gate(path: Path) -> threading.Lock:
    with _LOCAL_GATES_GUARD:
        return _LOCAL_GATES.setdefault(path, threading.Lock())


def _held_counts() -> dict[Path, int]:
    counts = getattr(_THREAD_STATE, "held_counts", None)
    if counts is None:
        counts = {}
        _THREAD_STATE.held_counts = counts
    return counts


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and bool(cancel_check()):
        raise InterruptedError("用户在等待独占加速器时取消了处理")


def _try_lock_file(stream) -> bool:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK} or getattr(
                exc, "winerror", None
            ) in {33, 36}:
                return False
            raise

    import fcntl

    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        raise


def _unlock_file(stream) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def acquire_accelerator_lock(
    device_index: int,
    *,
    cancel_check: CancelCheck | None = None,
    on_wait: WaitCallback | None = None,
    poll_interval: float = 0.25,
) -> Iterator[Path]:
    """Acquire a re-entrant, process-wide and OS-wide accelerator lock.

    The OS releases the byte-range lock when a process exits. Waiting has no
    hidden timeout: callers may cancel through ``cancel_check`` and otherwise
    receive the accelerator as soon as its current owner exits the critical
    section.
    """
    interval = float(poll_interval)
    if not 0.01 <= interval <= 5.0:
        raise ValueError(f"poll_interval must be between 0.01 and 5 seconds: {interval}")
    path = accelerator_lock_path(device_index)
    held_counts = _held_counts()
    if held_counts.get(path, 0):
        held_counts[path] += 1
        try:
            yield path
        finally:
            held_counts[path] -= 1
        return

    gate = _local_gate(path)
    gate_acquired = False
    stream = None
    file_acquired = False
    wait_notified = False

    def notify_waiting() -> None:
        nonlocal wait_notified
        if not wait_notified and on_wait is not None:
            on_wait()
        wait_notified = True

    try:
        while not gate_acquired:
            _raise_if_cancelled(cancel_check)
            gate_acquired = gate.acquire(timeout=interval)
            if not gate_acquired:
                notify_waiting()

        stream = path.open("a+b", buffering=0)
        if path.stat().st_size == 0:
            stream.write(b"\0")
        while not file_acquired:
            _raise_if_cancelled(cancel_check)
            file_acquired = _try_lock_file(stream)
            if not file_acquired:
                notify_waiting()
                time.sleep(interval)

        held_counts[path] = 1
        try:
            yield path
        finally:
            del held_counts[path]
    finally:
        try:
            if file_acquired and stream is not None:
                _unlock_file(stream)
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                if gate_acquired:
                    gate.release()


__all__ = ["accelerator_lock_path", "acquire_accelerator_lock"]
