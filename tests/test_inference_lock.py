from __future__ import annotations

import multiprocessing
import os
import time

import pytest

from src.utils.inference_lock import acquire_accelerator_lock


def _hold_accelerator_lock(lock_dir: str, ready, release) -> None:
    os.environ["MUSIC_TO_MIDI_INFERENCE_LOCK_DIR"] = lock_dir
    with acquire_accelerator_lock(0):
        ready.set()
        if not release.wait(15):
            raise RuntimeError("test holder timed out waiting for release")


def test_accelerator_lock_is_reentrant_and_blocks_another_process(tmp_path, monkeypatch):
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("MUSIC_TO_MIDI_INFERENCE_LOCK_DIR", str(lock_dir))
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_accelerator_lock,
        args=(str(lock_dir), ready, release),
    )
    holder.start()
    try:
        assert ready.wait(15), "child process did not acquire the accelerator lock"
        started = time.monotonic()
        waits: list[float] = []
        with pytest.raises(InterruptedError, match="等待独占加速器"):
            with acquire_accelerator_lock(
                0,
                cancel_check=lambda: time.monotonic() - started >= 0.35,
                on_wait=lambda: waits.append(time.monotonic()),
                poll_interval=0.05,
            ):
                raise AssertionError("contending process acquired a held accelerator lock")
        assert len(waits) == 1
    finally:
        release.set()
        holder.join(15)
        if holder.is_alive():
            holder.terminate()
            holder.join(5)
    assert holder.exitcode == 0

    with acquire_accelerator_lock(0) as outer_path:
        with acquire_accelerator_lock(0) as inner_path:
            assert outer_path == inner_path == (lock_dir / "accelerator-0.lock").resolve()
