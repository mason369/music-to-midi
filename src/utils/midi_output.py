"""Strict temporary-MIDI validation and atomic publication helpers."""

from __future__ import annotations

import os
import re
import uuid
from collections import defaultdict
from math import floor, isfinite
from pathlib import Path


def unique_midi_temp_path(final_path: str | Path, purpose: str) -> Path:
    """Return a unique, same-directory path without touching the final output."""
    final = Path(final_path).resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    safe_purpose = re.sub(r"[^A-Za-z0-9_-]+", "-", purpose).strip("-")
    if not safe_purpose:
        raise ValueError("MIDI temporary-output purpose must not be empty")
    return final.parent / (
        f".{final.stem}.{safe_purpose}.{uuid.uuid4().hex}.tmp.mid"
    )


def validate_midi_output(path: str | Path, backend_label: str) -> Path:
    """Require a non-empty, parseable Standard MIDI File."""
    midi_path = Path(path)
    if not midi_path.exists():
        raise RuntimeError(f"{backend_label} did not create a MIDI output: {midi_path}")
    if not midi_path.is_file():
        raise RuntimeError(f"{backend_label} MIDI output is not a file: {midi_path}")
    if midi_path.stat().st_size <= 0:
        raise RuntimeError(f"{backend_label} created an empty MIDI output: {midi_path}")

    try:
        import mido

        mido.MidiFile(str(midi_path))
    except Exception as exc:
        raise RuntimeError(
            f"{backend_label} created an invalid MIDI output: {midi_path}: {exc}"
        ) from exc
    return midi_path


def clip_midi_to_duration(
    path: str | Path,
    duration_seconds: float,
    backend_label: str,
) -> Path:
    """Clamp padded note-off/control events to the decoded audio duration."""
    import mido

    midi_path = validate_midi_output(path, backend_label)
    duration = float(duration_seconds)
    if not isfinite(duration) or duration <= 0.0:
        raise ValueError(f"Audio duration must be positive and finite, got {duration!r}")

    midi = mido.MidiFile(str(midi_path))
    tempo_events = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        for event_index, message in enumerate(track):
            absolute_tick += int(message.time)
            if message.is_meta and message.type == "set_tempo":
                tempo_events.append(
                    (absolute_tick, track_index, event_index, int(message.tempo))
                )
    tempo_events.sort()

    tempo = 500_000
    previous_tick = 0
    elapsed_seconds = 0.0
    cutoff_tick = None
    for event_tick, _track_index, _event_index, event_tempo in tempo_events:
        segment_seconds = mido.tick2second(
            event_tick - previous_tick,
            midi.ticks_per_beat,
            tempo,
        )
        if elapsed_seconds + segment_seconds >= duration:
            remaining = max(0.0, duration - elapsed_seconds)
            cutoff_tick = previous_tick + floor(
                mido.second2tick(remaining, midi.ticks_per_beat, tempo)
            )
            break
        elapsed_seconds += segment_seconds
        previous_tick = event_tick
        tempo = event_tempo
    if cutoff_tick is None:
        remaining = max(0.0, duration - elapsed_seconds)
        cutoff_tick = previous_tick + floor(
            mido.second2tick(remaining, midi.ticks_per_beat, tempo)
        )

    for track in midi.tracks:
        absolute_tick = 0
        kept_messages = []
        active_notes = defaultdict(int)
        for event_index, message in enumerate(track):
            absolute_tick += int(message.time)
            is_note_on = (
                not message.is_meta
                and message.type == "note_on"
                and message.velocity > 0
            )
            is_note_off = (
                not message.is_meta
                and (
                    message.type == "note_off"
                    or (message.type == "note_on" and message.velocity == 0)
                )
            )
            note_key = (
                (int(message.channel), int(message.note))
                if is_note_on or is_note_off
                else None
            )

            target_tick = absolute_tick
            keep = True
            if absolute_tick > cutoff_tick:
                if is_note_off and note_key is not None and active_notes[note_key] > 0:
                    target_tick = cutoff_tick
                    active_notes[note_key] -= 1
                elif not message.is_meta and message.type == "control_change":
                    target_tick = cutoff_tick
                elif message.is_meta and message.type == "end_of_track":
                    target_tick = cutoff_tick
                else:
                    keep = False
            elif is_note_on and note_key is not None:
                active_notes[note_key] += 1
            elif is_note_off and note_key is not None and active_notes[note_key] > 0:
                active_notes[note_key] -= 1

            if keep:
                kept_messages.append((target_tick, event_index, message))

        kept_messages.sort(key=lambda item: (item[0], item[1]))
        track.clear()
        previous_message_tick = 0
        for target_tick, _event_index, message in kept_messages:
            track.append(message.copy(time=target_tick - previous_message_tick))
            previous_message_tick = target_tick

    rewrite_path = unique_midi_temp_path(midi_path, "duration-clipped")
    try:
        midi.save(str(rewrite_path))
        validate_midi_output(rewrite_path, backend_label)
        os.replace(rewrite_path, midi_path)
    finally:
        remove_temporary_midi(rewrite_path)
    return midi_path


def publish_midi_output(
    temporary_path: str | Path,
    final_path: str | Path,
    backend_label: str,
) -> str:
    """Validate the new file, then atomically replace the requested final path."""
    temporary = validate_midi_output(temporary_path, backend_label)
    final = Path(final_path).resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final)
    return str(final)


def remove_temporary_midi(path: str | Path) -> None:
    """Remove only the unique temporary file created for the current attempt."""
    temporary = Path(path)
    try:
        temporary.unlink()
    except FileNotFoundError:
        return
