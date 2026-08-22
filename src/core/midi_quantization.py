"""Strict, opt-in musical-grid quantization for generated MIDI files."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido

from src.utils.midi_output import (
    publish_midi_output,
    remove_temporary_midi,
    unique_midi_temp_path,
    validate_midi_output,
)

MIDI_QUANTIZE_GRIDS: tuple[str, ...] = ("1/4", "1/8", "1/16", "1/32", "1/64")
DEFAULT_MIDI_QUANTIZE_GRID = "1/32"
MIDI_QUANTIZE_SCOPES: tuple[str, ...] = ("all_tracks", "selected_notes")
DEFAULT_MIDI_QUANTIZE_SCOPE = "all_tracks"
_GRID_DENOMINATORS = {grid: int(grid.split("/", 1)[1]) for grid in MIDI_QUANTIZE_GRIDS}


@dataclass(frozen=True)
class MidiQuantizationReport:
    """Verified result of one in-place MIDI quantization pass."""

    path: Path
    grid: str
    grid_ticks: int
    paired_note_count: int


def quantize_grid_ticks(ticks_per_beat: int, grid: str) -> int:
    """Return the integral tick width of one supported musical grid."""

    if (
        isinstance(ticks_per_beat, bool)
        or not isinstance(ticks_per_beat, int)
        or ticks_per_beat <= 0
    ):
        raise ValueError(f"MIDI ticks_per_beat must be a positive integer: {ticks_per_beat!r}")
    denominator = _GRID_DENOMINATORS.get(str(grid))
    if denominator is None:
        raise ValueError(f"Unsupported MIDI quantization grid: {grid!r}")
    return max(1, round(ticks_per_beat * 4 / denominator))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _message_payload(message: mido.Message | mido.MetaMessage) -> tuple[bool, Any]:
    payload = message.dict()
    payload.pop("time", None)
    return bool(message.is_meta), _freeze(payload)


def _event_fingerprints(
    midi: mido.MidiFile,
) -> tuple[tuple[Any, ...], Counter[tuple[Any, ...]]]:
    non_note_events: list[tuple[Any, ...]] = []
    note_events: Counter[tuple[Any, ...]] = Counter()
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            is_note = not message.is_meta and message.type in {"note_on", "note_off"}
            if is_note:
                note_events[(track_index, _message_payload(message))] += 1
            elif not (message.is_meta and message.type == "end_of_track"):
                non_note_events.append((track_index, absolute_tick, _message_payload(message)))
    return tuple(non_note_events), note_events


def _paired_note_ticks(
    midi: mido.MidiFile,
) -> list[tuple[int, tuple[int, int], int, int]]:
    pairs: list[tuple[int, tuple[int, int], int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        active: dict[tuple[int, int], list[int]] = defaultdict(list)
        for message in track:
            absolute_tick += int(message.time)
            if message.is_meta or message.type not in {"note_on", "note_off"}:
                continue
            key = (int(message.channel), int(message.note))
            is_note_on = message.type == "note_on" and int(message.velocity) > 0
            if is_note_on:
                active[key].append(absolute_tick)
                continue
            starts = active.get(key)
            if not starts:
                continue
            pairs.append((track_index, key, starts.pop(0), absolute_tick))
    return pairs


def _quantize_track(
    track: mido.MidiTrack,
    grid_ticks: int,
) -> tuple[mido.MidiTrack, int]:
    events: list[list[Any]] = []
    active: dict[tuple[int, int], list[int]] = defaultdict(list)
    end_of_track_ticks: list[int] = []
    absolute_tick = 0

    for sequence, message in enumerate(track):
        absolute_tick += int(message.time)
        if message.is_meta and message.type == "end_of_track":
            end_of_track_ticks.append(absolute_tick)
            continue
        events.append([absolute_tick, 1, sequence, message.copy(time=0)])

    paired_note_count = 0
    for event_index, event in enumerate(events):
        event_tick, _priority, _sequence, message = event
        if message.is_meta or message.type not in {"note_on", "note_off"}:
            continue
        key = (int(message.channel), int(message.note))
        is_note_on = message.type == "note_on" and int(message.velocity) > 0
        if is_note_on:
            active[key].append(event_index)
            continue
        starts = active.get(key)
        if not starts:
            continue
        start_index = starts.pop(0)
        start_tick = int(events[start_index][0])
        quantized_start = max(0, round(start_tick / grid_ticks) * grid_ticks)
        duration = max(0, int(event_tick) - start_tick)
        quantized_duration = max(
            grid_ticks,
            round(duration / grid_ticks) * grid_ticks,
        )
        events[start_index][0] = quantized_start
        event[0] = quantized_start + quantized_duration
        paired_note_count += 1

    final_event_tick = max((int(event[0]) for event in events), default=0)
    events.append(
        [
            max([final_event_tick, *end_of_track_ticks]),
            2,
            len(track),
            mido.MetaMessage("end_of_track", time=0),
        ]
    )
    events.sort(key=lambda event: (int(event[0]), int(event[1]), int(event[2])))

    rebuilt = mido.MidiTrack()
    previous_tick = 0
    for event_tick, _priority, _sequence, message in events:
        target_tick = int(event_tick)
        rebuilt.append(message.copy(time=target_tick - previous_tick))
        previous_tick = target_tick
    return rebuilt, paired_note_count


def quantize_midi_notes(
    midi_path: str | Path,
    grid: str = DEFAULT_MIDI_QUANTIZE_GRID,
    *,
    label: str = "MIDI note quantization",
) -> MidiQuantizationReport:
    """Quantize paired note starts and durations and atomically replace ``midi_path``.

    Tempo, meter, controller, program, marker, lyric, SysEx, and all other
    non-note events retain their original absolute ticks. An explicit request
    with no complete note pair fails instead of reporting a false success.
    """

    source_path = validate_midi_output(midi_path, label).resolve()
    source = mido.MidiFile(str(source_path))
    grid_ticks = quantize_grid_ticks(int(source.ticks_per_beat), grid)
    source_non_notes, source_note_events = _event_fingerprints(source)

    rebuilt_tracks: list[mido.MidiTrack] = []
    paired_note_count = 0
    for track in source.tracks:
        rebuilt, track_pairs = _quantize_track(track, grid_ticks)
        rebuilt_tracks.append(rebuilt)
        paired_note_count += track_pairs
    if paired_note_count <= 0:
        raise RuntimeError(f"{label} found no complete note pair in {source_path}")

    source.tracks = rebuilt_tracks
    temporary = unique_midi_temp_path(source_path, "note-quantized")
    try:
        source.save(str(temporary))
        verified = mido.MidiFile(str(temporary))
        if (
            verified.type != source.type
            or verified.ticks_per_beat != source.ticks_per_beat
            or len(verified.tracks) != len(source.tracks)
        ):
            raise RuntimeError(f"{label} changed the MIDI container contract: path={temporary}")

        verified_non_notes, verified_note_events = _event_fingerprints(verified)
        if verified_non_notes != source_non_notes:
            raise RuntimeError(
                f"{label} changed a non-note event or its absolute tick: path={temporary}"
            )
        if verified_note_events != source_note_events:
            raise RuntimeError(f"{label} changed note event content: path={temporary}")

        verified_pairs = _paired_note_ticks(verified)
        if len(verified_pairs) != paired_note_count:
            raise RuntimeError(
                f"{label} changed note pairing: expected={paired_note_count}, "
                f"actual={len(verified_pairs)}, path={temporary}"
            )
        invalid_pairs = [
            (track_index, key, start_tick, end_tick)
            for track_index, key, start_tick, end_tick in verified_pairs
            if start_tick % grid_ticks != 0
            or end_tick - start_tick < grid_ticks
            or (end_tick - start_tick) % grid_ticks != 0
        ]
        if invalid_pairs:
            raise RuntimeError(
                f"{label} verification found off-grid note pairs: "
                f"grid={grid!r}, examples={invalid_pairs[:3]!r}, path={temporary}"
            )

        expected_bytes = temporary.read_bytes()
        published = Path(publish_midi_output(temporary, source_path, label))
        if published.read_bytes() != expected_bytes:
            raise RuntimeError(
                f"{label} atomic publication changed the verified file: path={published}"
            )
        return MidiQuantizationReport(
            path=published,
            grid=str(grid),
            grid_ticks=grid_ticks,
            paired_note_count=paired_note_count,
        )
    finally:
        remove_temporary_midi(temporary)
