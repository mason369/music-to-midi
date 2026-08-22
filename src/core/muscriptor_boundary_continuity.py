"""Narrow post-decode repair for exact MuScriptor five-second split notes."""

from __future__ import annotations

from typing import Any, Iterable

MUSCRIPTOR_CHUNK_SECONDS = 5.0
MUSCRIPTOR_MAX_BOUNDARY_GAP_SECONDS = 0.03


def continuous_chunk_events(
    events: Iterable[object],
    NoteStartEvent,
    NoteEndEvent,
) -> tuple[list[object], int]:
    """Merge only a non-drum note split at an exact model chunk boundary.

    The official state machine can close an open note at N*5 seconds when the
    following tie prologue omits it. If the identical instrument/pitch restarts
    on one of the next three 10 ms model frames, the pair is an architectural
    discontinuity. Ordinary repetitions, drums, and longer gaps are untouched.
    """

    rewritten = list(events)
    starts_by_key: dict[tuple[str, int], list[tuple[int, Any]]] = {}
    for index, event in enumerate(rewritten):
        if isinstance(event, NoteStartEvent):
            key = (str(event.instrument), int(event.pitch))
            starts_by_key.setdefault(key, []).append((index, event))

    removed: set[int] = set()
    merged = 0
    for end_index, event in enumerate(rewritten):
        if not isinstance(event, NoteEndEvent):
            continue
        original_start = event.start_event
        instrument = str(original_start.instrument)
        if instrument == "drums":
            continue
        boundary = round(float(event.end_time) / MUSCRIPTOR_CHUNK_SECONDS)
        boundary_time = boundary * MUSCRIPTOR_CHUNK_SECONDS
        if boundary <= 0 or abs(float(event.end_time) - boundary_time) > 1e-9:
            continue
        key = (instrument, int(original_start.pitch))
        successor = next(
            (
                (start_index, start_event)
                for start_index, start_event in starts_by_key.get(key, ())
                if start_index > end_index
                and 0.0
                < float(start_event.start_time) - boundary_time
                <= MUSCRIPTOR_MAX_BOUNDARY_GAP_SECONDS + 1e-9
            ),
            None,
        )
        if successor is None:
            continue
        successor_index, successor_start = successor
        successor_end_index = next(
            (
                index
                for index in range(successor_index + 1, len(rewritten))
                if isinstance(rewritten[index], NoteEndEvent)
                and rewritten[index].start_event is successor_start
            ),
            None,
        )
        if successor_end_index is None:
            continue
        rewritten[successor_end_index] = NoteEndEvent(
            end_time=float(rewritten[successor_end_index].end_time),
            start_event=original_start,
        )
        removed.update((end_index, successor_index))
        merged += 1
    return (
        [event for index, event in enumerate(rewritten) if index not in removed],
        merged,
    )


__all__ = [
    "MUSCRIPTOR_CHUNK_SECONDS",
    "MUSCRIPTOR_MAX_BOUNDARY_GAP_SECONDS",
    "continuous_chunk_events",
]
