"""Backend-neutral, real incremental MIDI event contract.

Every snapshot is built from notes already emitted by the selected model.  A
frontend may therefore unlock playback only after receiving a non-empty
snapshot; ordinary percentage progress is never treated as playable MIDI.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

StreamCallback = Callable[[dict[str, Any]], None]


def instrument_key(program: int, is_drum: bool) -> str:
    return "drums" if is_drum else f"gm:{int(program):03d}"


def model_notes_payload(
    notes: Iterable[Any],
    *,
    frontier_seconds: float,
    inverse_vocab: dict | None = None,
) -> list[dict[str, Any]]:
    """Serialize stable model Note objects for the Qt/process boundary."""

    payload: list[dict[str, Any]] = []
    frontier = max(0.0, float(frontier_seconds))
    for note in notes:
        onset = max(0.0, float(note.onset))
        offset = max(onset + 0.01, float(note.offset))
        if offset > frontier + 1e-6:
            continue
        is_drum = bool(note.is_drum)
        source_program = int(note.program)
        if is_drum:
            program = 0
        elif inverse_vocab is not None:
            program = int(inverse_vocab.get(source_program, (source_program,))[0])
        else:
            program = source_program
        if program == 100:
            program = 65
        elif program == 101:
            program = 53
        # YourMT3/MIROS token velocities are binary; their official MIDI writer
        # deliberately renders every positive event at velocity 100.
        velocity = 100 if int(getattr(note, "velocity", 1)) > 0 else 1
        payload.append(
            {
                "instrument": instrument_key(program, is_drum),
                "program": program,
                "is_drum": is_drum,
                "pitch": int(note.pitch),
                "velocity": velocity,
                "start": onset,
                "end": offset,
            }
        )
    payload.sort(key=lambda item: (item["start"], item["pitch"], item["instrument"]))
    return payload


def piano_notes_payload(
    notes: Iterable[Any],
    *,
    frontier_seconds: float,
) -> list[dict[str, Any]]:
    """Serialize piano backend note objects/dicts at a stable frontier."""

    payload: list[dict[str, Any]] = []
    frontier = max(0.0, float(frontier_seconds))
    for note in notes:
        if isinstance(note, dict):
            onset = float(note["onset_time"])
            offset = float(note["offset_time"])
            pitch = int(note["midi_note"])
            velocity = int(note.get("velocity", 100))
        else:
            onset = float(note.start)
            offset = float(note.end)
            pitch = int(note.pitch)
            velocity = int(getattr(note, "velocity", 100))
        onset = max(0.0, onset)
        offset = max(onset + 0.01, offset)
        if offset > frontier + 1e-6:
            continue
        payload.append(
            {
                "instrument": "gm:000",
                "program": 0,
                "is_drum": False,
                "pitch": pitch,
                "velocity": max(1, min(127, velocity)),
                "start": onset,
                "end": offset,
            }
        )
    payload.sort(key=lambda item: (item["start"], item["pitch"], item["end"]))
    return payload


def snapshot_event(
    *,
    backend: str,
    completed: int,
    total: int,
    frontier_seconds: float,
    duration_seconds: float,
    notes: list[dict[str, Any]],
) -> dict[str, Any]:
    if total <= 0 or completed < 0 or completed > total:
        raise ValueError(
            f"Invalid streamed transcription progress: completed={completed}, total={total}"
        )
    return {
        "type": "snapshot",
        "backend": str(backend),
        "completed": int(completed),
        "total": int(total),
        "frontier_seconds": max(0.0, float(frontier_seconds)),
        "duration_seconds": max(0.0, float(duration_seconds)),
        "notes": list(notes),
    }


def append_jsonl_event(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one complete UTF-8 event so the parent can tail it safely."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def read_new_jsonl_events(
    path: str | Path,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    """Read only complete newly appended lines; retain a partial last line."""

    source = Path(path)
    if not source.exists():
        return [], offset
    with source.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    if not data:
        return [], offset
    last_newline = data.rfind(b"\n")
    if last_newline < 0:
        return [], offset
    complete = data[: last_newline + 1]
    next_offset = offset + len(complete)
    events: list[dict[str, Any]] = []
    for raw_line in complete.splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line.decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Stream event is not an object: {payload!r}")
        events.append(payload)
    return events, next_offset
