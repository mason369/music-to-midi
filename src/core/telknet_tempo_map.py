"""Stable TelkNet tempo-map construction for normalized Beat This evidence."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

MIN_TEMPO_BPM = 30.0
MAX_TEMPO_BPM = 300.0
TEMPO_BOUNDARY_ABS_TOLERANCE = 1e-9
MIN_TEMPO_CHANGE_RATIO = 0.03
MIN_TEMPO_CHANGE_BPM = 2.0
MIN_SUSTAINED_ANCHORS = 2
MAX_VARIABLE_TEMPO_PHASE_ERROR_BEATS = 1.0 / 16.0


class BeatThisTempoMapError(RuntimeError):
    """Normalized beat evidence cannot produce a safe MIDI tempo map."""


def _validated_timeline(
    values: Sequence[float],
    *,
    label: str,
    minimum_count: int,
) -> list[float]:
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BeatThisTempoMapError(f"Beat This final0 returned an invalid {label}")
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise BeatThisTempoMapError(f"Beat This final0 returned an invalid {label}")
        if normalized and timestamp <= normalized[-1]:
            raise BeatThisTempoMapError(f"Beat This final0 returned a non-monotonic {label}")
        normalized.append(timestamp)
    if len(normalized) < minimum_count:
        raise BeatThisTempoMapError(f"Beat This final0 returned too few timestamps for {label}")
    return normalized


def _tempo_change_threshold(bpm: float) -> float:
    return max(MIN_TEMPO_CHANGE_BPM, abs(float(bpm)) * MIN_TEMPO_CHANGE_RATIO)


def _fold_subdivision_bpm(bpm: float, representative_bpm: float) -> float:
    """Canonicalize half/double-time tracker changes around the song pulse."""

    ratio = bpm / representative_bpm
    if 1.75 <= ratio <= 2.25 and bpm / 2.0 >= MIN_TEMPO_BPM:
        return bpm / 2.0
    if 0.44 <= ratio <= 0.57 and bpm * 2.0 <= MAX_TEMPO_BPM:
        return bpm * 2.0
    return bpm


@dataclass
class _TempoSection:
    start: float
    bpms: list[float]
    weights: list[float]

    @property
    def bpm(self) -> float:
        total = sum(self.weights)
        if total <= 0:
            return float(statistics.median(self.bpms))
        return sum(bpm * weight for bpm, weight in zip(self.bpms, self.weights)) / total


def _sectioned_tempo_map(
    observations: Sequence[tuple[float, float, float]],
    *,
    representative_bpm: float,
) -> tuple[tuple[float, float], ...]:
    """Turn local tempo evidence into persistent, hysteretic sections."""

    sections: list[_TempoSection] = []
    for start, raw_bpm, weight in observations:
        bpm = _fold_subdivision_bpm(float(raw_bpm), representative_bpm)
        if not sections or abs(bpm - sections[-1].bpm) >= _tempo_change_threshold(sections[-1].bpm):
            sections.append(_TempoSection(float(start), [bpm], [float(weight)]))
        else:
            sections[-1].bpms.append(bpm)
            sections[-1].weights.append(float(weight))

    # A one-anchor excursion is detector jitter, not a publishable tempo
    # section. Merge it with the musically closer neighbour and repeat because
    # the merge can expose another isolated section.
    while len(sections) > 1:
        short_index = next(
            (
                index
                for index, section in enumerate(sections)
                if len(section.bpms) < MIN_SUSTAINED_ANCHORS
            ),
            None,
        )
        if short_index is None:
            break
        section = sections[short_index]
        neighbours = [
            index for index in (short_index - 1, short_index + 1) if 0 <= index < len(sections)
        ]
        if not neighbours:
            break
        target_index = min(
            neighbours,
            key=lambda index: abs(sections[index].bpm - section.bpm),
        )
        target = sections[target_index]
        if target_index < short_index:
            target.bpms.extend(section.bpms)
            target.weights.extend(section.weights)
        else:
            target.start = section.start
            target.bpms = [*section.bpms, *target.bpms]
            target.weights = [*section.weights, *target.weights]
        del sections[short_index]

    # Re-merge neighbours whose weighted estimates converged after removing
    # short excursions.
    merged: list[_TempoSection] = []
    for section in sections:
        if merged and abs(section.bpm - merged[-1].bpm) < _tempo_change_threshold(merged[-1].bpm):
            merged[-1].bpms.extend(section.bpms)
            merged[-1].weights.extend(section.weights)
        else:
            merged.append(section)

    if not merged:
        return ((0.0, representative_bpm),)
    if len(merged) == 1:
        return ((0.0, representative_bpm),)

    events: list[tuple[float, float]] = []
    for index, section in enumerate(merged):
        bpm = section.bpm
        if index == 0:
            if abs(bpm - representative_bpm) < _tempo_change_threshold(representative_bpm):
                bpm = representative_bpm
            events.append((0.0, bpm))
            continue
        if abs(bpm - events[-1][1]) < _tempo_change_threshold(events[-1][1]):
            continue
        events.append((section.start, bpm))
    return tuple(events)


def _tempo_map_from_downbeats(
    downbeat_times: Sequence[float],
    *,
    beats_per_bar: int,
    representative_bpm: float,
) -> tuple[tuple[float, float], ...]:
    """Fit bar-aligned tempo sections while correcting missed downbeats."""

    downbeats = _validated_timeline(
        downbeat_times,
        label="downbeat timeline",
        minimum_count=3,
    )
    reference_bar_seconds = beats_per_bar * 60.0 / representative_bpm
    raw_bpms: list[float] = []
    starts: list[float] = []
    weights: list[float] = []
    for previous, current in zip(downbeats, downbeats[1:]):
        interval = current - previous
        represented_bars = max(1, round(interval / reference_bar_seconds))
        represented_beats = float(represented_bars * beats_per_bar)
        bpm = 60.0 * represented_beats / interval
        if not MIN_TEMPO_BPM <= bpm <= MAX_TEMPO_BPM:
            raise BeatThisTempoMapError("Beat This final0 returned an out-of-range downbeat tempo")
        starts.append(previous)
        raw_bpms.append(bpm)
        weights.append(represented_beats)

    observations = [
        (
            starts[index],
            float(statistics.median(raw_bpms[max(0, index - 1) : min(len(raw_bpms), index + 2)])),
            weights[index],
        )
        for index in range(len(raw_bpms))
    ]
    return _sectioned_tempo_map(
        observations,
        representative_bpm=representative_bpm,
    )


def _tempo_map_from_beat_times(
    beat_times: Sequence[float],
    *,
    representative_bpm: float,
) -> tuple[tuple[float, float], ...]:
    """Build a tempo map with a hard cumulative beat-phase error bound.

    A BPM-difference threshold is not a timing guarantee: even a persistent
    one-percent difference accumulates into whole beats on a long recording.
    Treat the normalized Beat This timeline as beat ordinal versus physical
    seconds and simplify that curve directly.  Every retained segment starts
    and ends on a detected beat; recursive splitting continues until no beat
    inside the segment differs from the published MIDI grid by more than one
    sixteenth of a beat.
    """

    normalized = _validated_timeline(
        beat_times,
        label="beat timeline",
        minimum_count=2,
    )
    if (
        isinstance(representative_bpm, bool)
        or not isinstance(representative_bpm, (int, float))
        or not math.isfinite(float(representative_bpm))
        or not MIN_TEMPO_BPM <= float(representative_bpm) <= MAX_TEMPO_BPM
    ):
        raise BeatThisTempoMapError("Beat This final0 returned an invalid representative tempo")

    anchors = {0, len(normalized) - 1}
    pending = [(0, len(normalized) - 1)]
    while pending:
        start, end = pending.pop()
        if end <= start + 1:
            continue
        elapsed = normalized[end] - normalized[start]
        beats = end - start
        if elapsed <= 0:
            raise BeatThisTempoMapError(
                "Beat This final0 returned a non-positive tempo-map segment"
            )
        beats_per_second = beats / elapsed
        error, split = max(
            (
                abs((normalized[index] - normalized[start]) * beats_per_second - (index - start)),
                index,
            )
            for index in range(start + 1, end)
        )
        if error > MAX_VARIABLE_TEMPO_PHASE_ERROR_BEATS:
            anchors.add(split)
            pending.append((start, split))
            pending.append((split, end))

    ordered = sorted(anchors)
    events: list[tuple[float, float]] = []
    for event_index, (start, end) in enumerate(zip(ordered, ordered[1:])):
        elapsed = normalized[end] - normalized[start]
        bpm = 60.0 * (end - start) / elapsed
        if (
            bpm < MIN_TEMPO_BPM - TEMPO_BOUNDARY_ABS_TOLERANCE
            or bpm > MAX_TEMPO_BPM + TEMPO_BOUNDARY_ABS_TOLERANCE
        ):
            raise BeatThisTempoMapError(
                "Beat This final0 returned an out-of-range phase-bounded tempo"
            )
        # Beat timestamps are decimal detector frames represented as binary
        # floats.  Accept only machine-rounding noise at the reviewed limits,
        # then serialize the exact 30/300 BPM boundary value.
        bpm = min(MAX_TEMPO_BPM, max(MIN_TEMPO_BPM, bpm))
        events.append((0.0 if event_index == 0 else normalized[start], bpm))
    return tuple(events)


def _tempo_map_position_beats(
    timestamp: float,
    tempo_map: Sequence[tuple[float, float]],
) -> float:
    elapsed_beats = 0.0
    previous_start = 0.0
    previous_bpm = float(tempo_map[0][1])
    for start, bpm in tempo_map[1:]:
        start = float(start)
        if timestamp <= start:
            return elapsed_beats + (timestamp - previous_start) * previous_bpm / 60.0
        elapsed_beats += (start - previous_start) * previous_bpm / 60.0
        previous_start = start
        previous_bpm = float(bpm)
    return elapsed_beats + (timestamp - previous_start) * previous_bpm / 60.0


def _tempo_map_max_phase_error(
    beat_times: Sequence[float],
    tempo_map: Sequence[tuple[float, float]],
) -> float:
    normalized = _validated_timeline(
        beat_times,
        label="beat timeline",
        minimum_count=2,
    )
    if not tempo_map or float(tempo_map[0][0]) != 0.0:
        raise BeatThisTempoMapError("Tempo map must start at source second zero")
    origin = _tempo_map_position_beats(normalized[0], tempo_map)
    return max(
        abs(_tempo_map_position_beats(timestamp, tempo_map) - origin - ordinal)
        for ordinal, timestamp in enumerate(normalized)
    )


def _tempo_map_phase_violation_count(
    beat_times: Sequence[float],
    tempo_map: Sequence[tuple[float, float]],
    *,
    maximum_error_beats: float,
) -> int:
    normalized = _validated_timeline(
        beat_times,
        label="beat timeline",
        minimum_count=2,
    )
    if not tempo_map or float(tempo_map[0][0]) != 0.0:
        raise BeatThisTempoMapError("Tempo map must start at source second zero")
    origin = _tempo_map_position_beats(normalized[0], tempo_map)
    return sum(
        abs(_tempo_map_position_beats(timestamp, tempo_map) - origin - ordinal)
        > maximum_error_beats
        for ordinal, timestamp in enumerate(normalized)
    )


@dataclass(frozen=True)
class AdaptiveTempoMap:
    events: tuple[tuple[float, float], ...]
    strategy: str
    maximum_phase_error_beats: float


def build_adaptive_tempo_map(
    beat_times: Sequence[float],
    downbeat_times: Sequence[float],
    *,
    beats_per_bar: int | None,
    representative_bpm: float,
) -> AdaptiveTempoMap:
    """Prefer a bar-aligned stable map only when it keeps the v10 phase bound.

    Reliable meter first gets the sectioned downbeat treatment described by
    the product UI. If that candidate cannot keep every accepted Beat This
    beat within 1/16 beat, the exact v10 phase-bounded beat segmentation is
    selected and reported explicitly as its strategy.
    """

    if beats_per_bar is not None and len(downbeat_times) >= 3:
        bar_events = _tempo_map_from_downbeats(
            downbeat_times,
            beats_per_bar=beats_per_bar,
            representative_bpm=representative_bpm,
        )
        bar_error = _tempo_map_max_phase_error(beat_times, bar_events)
        if bar_error <= MAX_VARIABLE_TEMPO_PHASE_ERROR_BEATS + 1e-8:
            return AdaptiveTempoMap(
                events=bar_events,
                strategy="bar_aligned_phase_validated",
                maximum_phase_error_beats=bar_error,
            )

    events = _tempo_map_from_beat_times(
        beat_times,
        representative_bpm=representative_bpm,
    )
    error = _tempo_map_max_phase_error(beat_times, events)
    if error > MAX_VARIABLE_TEMPO_PHASE_ERROR_BEATS + 1e-8:
        raise BeatThisTempoMapError(
            "Beat This adaptive tempo map exceeded the reviewed 1/16-beat phase bound"
        )
    return AdaptiveTempoMap(
        events=events,
        strategy="phase_bounded_beats",
        maximum_phase_error_beats=error,
    )


def tempo_map_max_phase_error(
    beat_times: Sequence[float],
    tempo_map: Sequence[tuple[float, float]],
) -> float:
    """Public verifier used by acceptance tests and MIDI serialization checks."""

    return _tempo_map_max_phase_error(beat_times, tempo_map)


__all__ = [
    "AdaptiveTempoMap",
    "BeatThisTempoMapError",
    "MAX_VARIABLE_TEMPO_PHASE_ERROR_BEATS",
    "build_adaptive_tempo_map",
    "tempo_map_max_phase_error",
]
