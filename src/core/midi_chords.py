"""Lossless ChordMini labels and audition voicings on the source-audio timeline.

Uses the 170-class vocabulary supported by TelkNet's chord tool. This module
only validates and presents model output; it never estimates harmony from MIDI.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

QUALITIES = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
    "minmaj7": (0, 3, 7, 11),
    "maj6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}
ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass(frozen=True)
class MidiChord:
    start: float
    end: float
    label: str
    pitches: tuple[int, ...]


def chord_pitches(label: str) -> tuple[int, ...]:
    if label in {"N", "X"}:
        return ()
    match = re.fullmatch(r"([A-G])([#b]*)(?::([^/]+))?(?:/([#b]*)([1-7]))?", label)
    if not match or (match[3] or "maj") not in QUALITIES:
        raise ValueError(f"Unsupported chord label: {label!r}")
    root = (ROOTS[match[1]] + match[2].count("#") - match[2].count("b")) % 12
    notes = {60 + root + interval for interval in QUALITIES[match[3] or "maj"]}
    if match[5]:
        bass = (0, 2, 4, 5, 7, 9, 11)[int(match[5]) - 1]
        bass += match[4].count("#") - match[4].count("b")
        notes.add(48 + (root + bass) % 12)
    return tuple(sorted(notes))


def read_chord_record(record: dict) -> tuple[MidiChord, ...]:
    """Reject wrong identities, gaps, truncation and fabricated model output."""
    from src.utils.chordmini_runtime import CHECKPOINT_SHA256, CHORDMINI_COMMIT, RECIPE_ID

    if (
        record.get("id") != "chordmini_btc"
        or record.get("recipe") != RECIPE_ID
        or record.get("source_revision") != CHORDMINI_COMMIT
        or record.get("checkpoint_sha256") != CHECKPOINT_SHA256
    ):
        raise ValueError("ChordMini model identity mismatch")
    duration, step = float(record["duration"]), float(record["frame_seconds"])
    if (
        not math.isfinite(duration)
        or not 0 < duration <= 14400
        or not math.isfinite(step)
        or not 0 < step < 1
    ):
        raise ValueError("Invalid chord duration or frame size")
    segments = record.get("segments")
    if not isinstance(segments, list) or not 0 < len(segments) <= 250000:
        raise ValueError("Missing or excessive chord segments")
    result = []
    previous = 0.0
    for segment in segments:
        start, end, label = float(segment["start"]), float(segment["end"]), segment["label"]
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or not math.isclose(start, previous, abs_tol=1e-6)
            or end > duration + step
            or not isinstance(label, str)
        ):
            raise ValueError("Invalid chord interval")
        pitches = chord_pitches(label)
        if start < duration:
            result.append(MidiChord(start, min(end, duration), label, pitches))
        previous = end
    if not result or duration - result[-1].end > step * 1.5:
        raise ValueError("Incomplete chord prediction")
    return tuple(result)


def split_chords_at_bars(chords, reference_bpm, time_signature=(4, 4)):
    """Split display rectangles only; never quantize native chord boundaries."""
    from src.core.midi_tempo import validated_midi_bpm, validated_midi_time_signature

    bpm = validated_midi_bpm(reference_bpm, "chord reference")
    numerator, denominator = validated_midi_time_signature(time_signature)
    bar = 60.0 / bpm * 4.0 / denominator * numerator
    result = []
    for chord in chords:
        start = chord.start
        boundary = (math.floor(start / bar + 1e-9) + 1) * bar
        while boundary < chord.end - 1e-9:
            result.append(replace(chord, start=start, end=boundary))
            start, boundary = boundary, boundary + bar
        result.append(replace(chord, start=start))
    return tuple(result)
