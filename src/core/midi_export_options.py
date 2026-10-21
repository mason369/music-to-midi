"""Explicit, non-destructive grid correction and notation export options."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import mido

from src.core.midi_tempo import (
    _bar_offset_value,
    validated_midi_bpm,
    validated_midi_time_signature,
)
from src.core.midi_timing import PROJECT_RATE_MARKER, MidiTimeline
from src.utils.midi_output import (
    publish_midi_output,
    remove_temporary_midi,
    unique_midi_temp_path,
)


@dataclass(frozen=True)
class MidiExportOptions:
    bpm: float | None = None
    beats_per_bar: int | None = None
    beat_unit: int = 4
    first_downbeat: float | None = None
    beat_scale: float = 1.0
    fill_gaps: float = 0.0
    trim_leading_bars: bool = False
    smooth_tempo: bool = False

    def __post_init__(self):
        if self.bpm is not None:
            validated_midi_bpm(self.bpm, "grid correction")
        if self.beats_per_bar is not None:
            validated_midi_time_signature((self.beats_per_bar, self.beat_unit))
        if self.first_downbeat is not None and (
            not math.isfinite(self.first_downbeat) or self.first_downbeat < 0
        ):
            raise ValueError("First downbeat must be a finite non-negative source second")
        scale = float(self.beat_scale)
        if not math.isfinite(scale) or scale <= 0 or not math.log2(scale).is_integer():
            raise ValueError("Beat scale must be a positive power of two")
        if not math.isfinite(self.fill_gaps) or not 0 <= self.fill_gaps <= 4:
            raise ValueError("Gap filling must be between zero and four beats")


def _absolute_tracks(midi):
    result = []
    for track in midi.tracks:
        tick, events = 0, []
        for order, message in enumerate(track):
            tick += message.time
            events.append([int(tick), order, message.copy(time=0)])
        result.append(events)
    return result


def _from_events(source, tracks):
    result = mido.MidiFile(type=source.type, ticks_per_beat=source.ticks_per_beat)
    for events in tracks:
        events.sort(
            key=lambda event: (event[0], 0 if event[2].type == "set_tempo" else 1, event[1])
        )
        track, previous = mido.MidiTrack(), 0
        end = max((event[0] for event in events), default=0)
        for tick, _order, message in events:
            if message.type == "end_of_track":
                continue
            track.append(message.copy(time=tick - previous))
            previous = tick
        track.append(mido.MetaMessage("end_of_track", time=max(0, end - previous)))
        result.tracks.append(track)
    return result


def _smoothed_timeline(timeline, end_tick, bar_beats):
    """Fit a sparse map with a measured 25 ms bound; prefer bar anchors."""
    import numpy as np

    count = max(2, math.ceil(end_tick / timeline.ticks_per_beat) + 1)
    if count > 200_000:
        raise ValueError("Tempo smoothing requires fewer than 200,000 beats")
    beats = np.arange(count, dtype=float)
    original = np.array([timeline.seconds_at(float(beat)) for beat in beats])
    tolerance = min(0.025, float(np.diff(original).min()) / 16)
    smooth = original.copy()
    for index in range(1, count - 1):
        lo, hi = max(0, index - 4), min(count, index + 5)
        if hi - lo >= 3:
            fitted = float(np.polyval(np.polyfit(beats[lo:hi] - index, original[lo:hi], 2), 0))
            smooth[index] = np.clip(
                fitted, original[index] - tolerance * 0.48, original[index] + tolerance * 0.48
            )
    if (np.diff(smooth) <= 0).any():
        raise ValueError("Tempo smoothing cannot preserve a monotonic beat timeline")
    kept, pending = {0, count - 1}, [(0, count - 1)]
    while pending:
        start, end = pending.pop()
        if end <= start + 1:
            continue
        fitted = smooth[start] + (beats[start : end + 1] - start) * (
            smooth[end] - smooth[start]
        ) / (end - start)
        errors = abs(fitted - smooth[start : end + 1])
        if float(errors.max()) <= tolerance * 0.48:
            continue
        split = start + int(errors.argmax())
        candidates = [
            i for i in range(start + 1, end) if math.isclose(i % bar_beats, 0, abs_tol=1e-8)
        ]
        if candidates:
            candidate = min(candidates, key=lambda i: abs(i - split))
            if abs(candidate - split) <= bar_beats:
                split = candidate
        if split <= start or split >= end:
            raise ValueError("Tempo smoothing failed to find an interior beat anchor")
        kept.add(split)
        pending.extend(((start, split), (split, end)))
    ordered = sorted(kept)
    sections, seconds = [], 0.0
    for start, end in zip(ordered, ordered[1:]):
        tempo = round((smooth[end] - smooth[start]) * 1_000_000 / (end - start))
        if not 1 <= tempo <= 0xFFFFFF:
            raise ValueError("Smoothed MIDI tempo exceeds the SMF range")
        sections.append((start * timeline.ticks_per_beat, seconds, tempo))
        seconds += (end - start) * tempo / 1_000_000
    result = MidiTimeline(timeline.ticks_per_beat, tuple(sections))
    error = max(
        abs(result.seconds_at(float(beat)) - original[i]) for i, beat in enumerate(beats)
    )
    error = max(
        error,
        *(
            abs(result.seconds_at_tick(tick) - seconds)
            for tick, seconds, _tempo in timeline.sections
        ),
    )
    if error > tolerance:
        raise ValueError(f"Tempo smoothing exceeded its 25 ms bound: {error:.6f}s")
    return result


def apply_midi_export_options(
    source_path, destination_path, options: MidiExportOptions | dict
) -> Path:
    """Correct the grid without changing source-relative event playback times."""
    options = MidiExportOptions(**options) if isinstance(options, dict) else options
    source = mido.MidiFile(str(source_path))
    if not source.tracks:
        raise ValueError("MIDI export source has no tracks")
    source_tracks = _absolute_tracks(source)
    timelines = [MidiTimeline.from_midi(source, i) for i in range(len(source.tracks))]
    signature = next(
        (
            (m.numerator, m.denominator)
            for t in source.tracks
            for m in t
            if m.type == "time_signature"
        ),
        None,
    )
    if options.beats_per_bar is not None:
        signature = (options.beats_per_bar, options.beat_unit)
    elif signature is not None and options.beat_scale != 1:
        scaled = signature[0] * float(options.beat_scale)
        if scaled.is_integer() and 1 <= scaled <= 255:
            signature = (int(scaled), signature[1])
    if options.trim_leading_bars and signature is None:
        raise ValueError("Trimming leading bars requires a known or corrected time signature")
    bar = signature[0] * 4 / signature[1] if signature is not None else 1.0
    target_timelines = []
    end_tick = max((e[0] for t in source_tracks for e in t), default=0)
    for timeline in timelines:
        if options.bpm is not None:
            target = MidiTimeline(
                source.ticks_per_beat, ((0, 0.0, mido.bpm2tempo(options.bpm)),)
            )
        elif options.beat_scale != 1:
            tick_scale, seconds, previous, sections = float(options.beat_scale), 0.0, None, []
            for tick, _time, tempo in timeline.sections:
                scaled_tick, scaled_tempo = round(tick * tick_scale), round(tempo / tick_scale)
                if not 1 <= scaled_tempo <= 0xFFFFFF:
                    raise ValueError("Corrected MIDI tempo exceeds the SMF range")
                if previous is not None:
                    seconds += (
                        (scaled_tick - previous[0])
                        * previous[2]
                        / 1_000_000
                        / source.ticks_per_beat
                    )
                previous = (scaled_tick, seconds, scaled_tempo)
                sections.append(previous)
            target = MidiTimeline(source.ticks_per_beat, tuple(sections))
        else:
            target = timeline
        if options.smooth_tempo and len(target.sections) > 1:
            target = _smoothed_timeline(
                target, round(target.tick_at_seconds(timeline.seconds_at_tick(end_tick))), bar
            )
        target_timelines.append(target)

    tracks = []
    for index, events in enumerate(source_tracks):
        old, new = timelines[index], target_timelines[index]
        rebuilt = []
        for tick, order, message in events:
            if (
                message.type == "set_tempo"
                or (options.beats_per_bar is not None or options.beat_scale != 1)
                and message.type == "time_signature"
            ):
                continue
            if message.type == "marker" and message.text.startswith(PROJECT_RATE_MARKER):
                continue
            new_tick = max(0, round(new.tick_at_seconds(old.seconds_at_tick(tick))))
            rebuilt.append([new_tick, order, message])
        if source.type == 2 or index == 0 or any(e[2].type == "note_on" for e in events):
            rebuilt.extend(
                [tick, -100, mido.MetaMessage("set_tempo", tempo=tempo)]
                for tick, _seconds, tempo in new.sections
            )
        if (
            signature is not None
            and (options.beats_per_bar is not None or options.beat_scale != 1)
            and (source.type == 2 or index == 0)
        ):
            rebuilt.append(
                [
                    0,
                    -90,
                    mido.MetaMessage(
                        "time_signature", numerator=signature[0], denominator=signature[1]
                    ),
                ]
            )
        if source.type == 2 or index == 0:
            rebuilt.append([0, -98, mido.MetaMessage("marker", text=f"{PROJECT_RATE_MARKER}1")])
        tracks.append(rebuilt)

    for index, events in enumerate(tracks):
        timeline = target_timelines[index]
        shift = 0
        if options.first_downbeat is not None:
            old_offset = next(
                (
                    _bar_offset_value(e[2])
                    for e in events
                    if e[2].is_meta and _bar_offset_value(e[2]) is not None
                ),
                0.0,
            )
            origin_tick = timeline.tick_at_seconds(options.first_downbeat + old_offset)
            shift = round((-origin_tick) % (bar * source.ticks_per_beat))
        padding = shift * timeline.sections[0][2] / 1_000_000 / source.ticks_per_beat
        if shift:
            for event in events:
                event[0] += shift
            events.append(
                [0, -101, mido.MetaMessage("set_tempo", tempo=timeline.sections[0][2])]
            )
        markers = [e for e in events if e[2].is_meta and _bar_offset_value(e[2]) is not None]
        for event in markers:
            event[2] = event[2].copy(
                text=f"muscriptor:bar_offset={_bar_offset_value(event[2]) + padding:.4f}"
            )
        if padding and not markers and (source.type == 2 or index == 0):
            events.append(
                [
                    0,
                    -99,
                    mido.MetaMessage("marker", text=f"muscriptor:bar_offset={padding:.4f}"),
                ]
            )

    for events in tracks:
        active, pairs, onsets = defaultdict(deque), [], defaultdict(set)
        programs = defaultdict(int)
        for event in sorted(events, key=lambda e: (e[0], e[1])):
            tick, _order, msg = event
            if msg.type == "program_change":
                programs[msg.channel] = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                instrument = (msg.channel, programs[msg.channel])
                active[key].append((event, instrument))
                onsets[instrument].add(tick)
            elif msg.type in {"note_on", "note_off"} and active[(msg.channel, msg.note)]:
                start, instrument = active[(msg.channel, msg.note)].popleft()
                pairs.append((start, event, instrument))
        if options.fill_gaps:
            for start, end, instrument in pairs:
                if start[2].channel == 9:
                    continue
                next_onset = min((t for t in onsets[instrument] if t >= end[0]), default=None)
                if (
                    next_onset is not None
                    and 0 < next_onset - end[0] <= options.fill_gaps * source.ticks_per_beat
                ):
                    end[0] = next_onset

    if options.trim_leading_bars:
        first = min(
            (e[0] for t in tracks for e in t if e[2].type == "note_on" and e[2].velocity > 0),
            default=0,
        )
        trim_tick = int(first // (bar * source.ticks_per_beat) * bar * source.ticks_per_beat)
        staged = _from_events(source, tracks)
        for index, events in enumerate(tracks):
            timeline = MidiTimeline.from_midi(staged, index)
            removed_seconds = timeline.seconds_at_tick(trim_tick)
            old_marker = next(
                (
                    _bar_offset_value(e[2])
                    for e in events
                    if e[2].is_meta and _bar_offset_value(e[2]) is not None
                ),
                0.0,
            )
            for event in events:
                event[0] = max(0, event[0] - trim_tick)
            events[:] = [
                e for e in events if not (e[2].is_meta and _bar_offset_value(e[2]) is not None)
            ]
            if source.type == 2 or index == 0:
                events.append(
                    [
                        0,
                        -99,
                        mido.MetaMessage(
                            "marker",
                            text=f"muscriptor:bar_offset={old_marker - removed_seconds:.4f}",
                        ),
                    ]
                )

    output = _from_events(source, tracks)
    destination = Path(destination_path).resolve()
    temporary = unique_midi_temp_path(destination, "grid-export")
    try:
        output.save(str(temporary))
        checked = mido.MidiFile(str(temporary))
        if len(checked.tracks) != len(source.tracks):
            raise RuntimeError("MIDI grid export changed the track count")

        def payloads(midi):
            return Counter(
                (index, bytes(message.bytes()))
                for index, track in enumerate(midi.tracks)
                for message in track
                if message.type not in {"set_tempo", "time_signature", "end_of_track"}
                and not (message.is_meta and _bar_offset_value(message) is not None)
                and not (
                    message.type == "marker" and message.text.startswith(PROJECT_RATE_MARKER)
                )
            )

        if payloads(checked) != payloads(source):
            raise RuntimeError("MIDI grid export changed an event payload")
        return Path(publish_midi_output(temporary, destination, "MIDI grid export"))
    finally:
        remove_temporary_midi(temporary)
