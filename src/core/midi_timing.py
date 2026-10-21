"""One reversible musical timeline for playback, editing and MIDI exports."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import mido

from src.core.midi_tempo import _tempo_sections, validated_midi_bpm
from src.utils.midi_output import (
    publish_midi_output,
    remove_temporary_midi,
    unique_midi_temp_path,
)

PROJECT_RATE_MARKER = "music-to-midi:playback_rate="


def project_rate_marker(midi):
    values = [
        float(message.text[len(PROJECT_RATE_MARKER) :])
        for track in midi.tracks
        for message in track
        if message.type == "marker" and message.text.startswith(PROJECT_RATE_MARKER)
    ]
    if not values:
        return None
    if any(
        not math.isfinite(value) or value <= 0 or abs(value - values[0]) > 1e-9
        for value in values
    ):
        raise ValueError("Invalid or conflicting MIDI project-speed markers")
    return values[0]


@dataclass(frozen=True)
class MidiTimeline:
    ticks_per_beat: int
    sections: tuple[tuple[int, float, int], ...]

    @classmethod
    def from_midi(cls, midi: mido.MidiFile, track: int = 0, reference_bpm=None):
        if midi.ticks_per_beat <= 0:
            raise ValueError("SMPTE MIDI timing is not supported")
        _, sections = _tempo_sections(midi, track)
        rate = project_rate_marker(midi) if reference_bpm is not None else None
        if rate is not None:
            sections = [
                (tick, seconds * rate, round(tempo * rate)) for tick, seconds, tempo in sections
            ]
        elif reference_bpm is not None and len({s[2] for s in sections}) == 1:
            sections = [(0, 0.0, mido.bpm2tempo(validated_midi_bpm(reference_bpm)))]
        return cls(int(midi.ticks_per_beat), tuple(sections))

    def seconds_at_tick(self, tick: float) -> float:
        index = max(0, bisect_right([s[0] for s in self.sections], tick) - 1)
        start, seconds, tempo = self.sections[index]
        return seconds + (float(tick) - start) * tempo / 1_000_000 / self.ticks_per_beat

    def tick_at_seconds(self, seconds: float) -> float:
        if not math.isfinite(float(seconds)):
            raise ValueError("MIDI timeline requires finite seconds")
        index = max(0, bisect_right([s[1] for s in self.sections], seconds) - 1)
        tick, start, tempo = self.sections[index]
        return tick + (float(seconds) - start) * 1_000_000 * self.ticks_per_beat / tempo

    def beats_at(self, seconds: float) -> float:
        return self.tick_at_seconds(seconds) / self.ticks_per_beat

    def seconds_at(self, beats: float) -> float:
        return self.seconds_at_tick(float(beats) * self.ticks_per_beat)

    def manifest(self) -> dict:
        return {"division": self.ticks_per_beat, "sections": [list(s) for s in self.sections]}


def reference_timelines(midi: mido.MidiFile, reference_bpm: float) -> tuple[MidiTimeline, ...]:
    return tuple(
        MidiTimeline.from_midi(midi, i, reference_bpm) for i in range(len(midi.tracks))
    )


def midi_timeline_manifest(path: str | Path, reference_bpm: float) -> dict:
    midi = mido.MidiFile(str(path))
    timelines = reference_timelines(midi, reference_bpm)
    return {"format": midi.type, "tracks": [t.manifest() for t in timelines]}


def write_project_tempo_map(
    source_path: str | Path,
    destination_path: str | Path,
    reference_bpm: float,
    target_bpm: float,
    *,
    label: str = "Project tempo-map export",
) -> Path:
    """Scale the entire reference tempo map, retaining every musical tick."""
    from src.core.midi_tempo import _bar_offset_value, non_tempo_event_tick_fingerprint

    reference = validated_midi_bpm(reference_bpm, "reference")
    target = validated_midi_bpm(target_bpm, "target")
    source = mido.MidiFile(str(source_path))
    timelines = reference_timelines(source, reference)
    output = mido.MidiFile(type=source.type, ticks_per_beat=source.ticks_per_beat)
    rate = target / reference
    for index, track in enumerate(source.tracks):
        events = []
        tick = 0
        for order, message in enumerate(track):
            tick += message.time
            if message.type == "end_of_track":
                continue
            if message.type == "set_tempo":
                continue
            if message.type == "marker" and message.text.startswith(PROJECT_RATE_MARKER):
                continue
            offset = _bar_offset_value(message) if message.is_meta else None
            if offset is not None:
                # A fixed manual-tempo source carries its marker at its existing speed.
                source_tempo = MidiTimeline.from_midi(source, index).sections[0][2]
                marker_scale = timelines[index].sections[0][2] / rate / source_tempo
                message = message.copy(
                    text=f"muscriptor:bar_offset={offset * marker_scale:.4f}"
                )
            events.append((int(tick), 1, order, message.copy(time=0)))
        if source.type == 2 or index == 0 or any(m.type == "set_tempo" for m in track):
            for order, (tempo_tick, _seconds, tempo) in enumerate(timelines[index].sections):
                scaled = round(tempo / rate)
                if not 1 <= scaled <= 0xFFFFFF:
                    raise ValueError("Scaled MIDI tempo exceeds the SMF range")
                events.append(
                    (tempo_tick, 0, order, mido.MetaMessage("set_tempo", tempo=scaled))
                )
        end_tick = max([int(tick), *(event[0] for event in events)])
        events.append((end_tick, 2, len(events), mido.MetaMessage("end_of_track")))
        if source.type == 2 or index == 0:
            events.append(
                (0, 1, -1, mido.MetaMessage("marker", text=f"{PROJECT_RATE_MARKER}{rate:.12g}"))
            )
        events.sort(key=lambda e: e[:3])
        rebuilt = mido.MidiTrack()
        previous = 0
        for tick, _priority, _order, message in events:
            rebuilt.append(message.copy(time=tick - previous))
            previous = tick
        output.tracks.append(rebuilt)
    destination = Path(destination_path).resolve()
    temporary = unique_midi_temp_path(destination, "project-map")
    try:
        output.save(str(temporary))
        checked = mido.MidiFile(str(temporary))

        # Markers intentionally change with playback speed; all other payloads and ticks stay exact.
        def fingerprint(midi):
            snapshot = mido.MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat)
            for track in midi.tracks:
                rebuilt, elapsed = mido.MidiTrack(), 0
                for message in track:
                    elapsed += message.time
                    if message.is_meta and _bar_offset_value(message) is not None:
                        continue
                    if message.type == "marker" and message.text.startswith(
                        PROJECT_RATE_MARKER
                    ):
                        continue
                    rebuilt.append(message.copy(time=elapsed))
                    elapsed = 0
                snapshot.tracks.append(rebuilt)
            return non_tempo_event_tick_fingerprint(snapshot)

        if fingerprint(checked) != fingerprint(source):
            raise RuntimeError("Project tempo-map export changed a non-tempo event")
        return Path(publish_midi_output(temporary, destination, label))
    finally:
        remove_temporary_midi(temporary)


def quantize_times(
    start: float,
    end: float,
    timeline: MidiTimeline,
    grid: str,
    *,
    strength: float = 1.0,
    triplet: bool = False,
) -> tuple[float, float]:
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Quantization strength must be between zero and one")
    if grid not in {"1/4", "1/8", "1/16", "1/32", "1/64"}:
        raise ValueError(f"Unsupported MIDI quantization grid: {grid!r}")
    denominator = int(grid.split("/")[1])
    if strength == 0:
        return start, end
    step = 4 / denominator * (2 / 3 if triplet else 1)
    onset, offset = timeline.beats_at(start), timeline.beats_at(end)
    snapped_start = max(0.0, round(onset / step) * step)
    snapped_duration = max(step, round((offset - onset) / step) * step)
    snapped_end = snapped_start + snapped_duration
    return (
        timeline.seconds_at(onset + strength * (snapped_start - onset)),
        timeline.seconds_at(offset + strength * (snapped_end - offset)),
    )


def detect_quantization_grid(
    notes: Iterable, timelines: tuple[MidiTimeline, ...], *, triplet=False
) -> str:
    notes = tuple(notes)
    if not notes:
        raise ValueError("Automatic grid requires notes")
    scores = []
    for grid in ("1/4", "1/8", "1/16", "1/32", "1/64"):
        step = 4 / int(grid[2:]) * (2 / 3 if triplet else 1)
        errors = []
        for note in notes:
            timeline = timelines[min(int(note.track_index), len(timelines) - 1)]
            for value in (timeline.beats_at(note.start), timeline.beats_at(note.end)):
                errors.append(abs(value / step - round(value / step)))
        score = sum(min(e, 0.5) for e in errors) / len(errors)
        scores.append((score, grid))
        if score <= 0.08:
            return grid
    return min(scores)[1]


def quantize_editor_notes(
    notes, timelines, grid="1/32", *, strength=1.0, triplet=False, indices=None, auto=False
):
    updated = list(notes)
    indices = tuple(range(len(updated))) if indices is None else tuple(indices)
    if auto:
        grid = detect_quantization_grid(
            (updated[i] for i in indices), timelines, triplet=triplet
        )
    for index in indices:
        note = updated[index]
        timeline = timelines[min(int(note.track_index), len(timelines) - 1)]
        start, end = quantize_times(
            note.start, note.end, timeline, grid, strength=strength, triplet=triplet
        )
        updated[index] = replace(note, start=start, end=max(start + 0.000001, end))
    return updated, grid
