"""Validate fretted notation without deleting or changing transcribed notes.

Only the temporary MuseScore engraving document is modified. A string is a
physical resource: simultaneous notes, including held notes in other voices,
cannot share it. Unrepresentable parts remain available in standard notation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction

_DURATIONS = {
    "long": Fraction(4),
    "breve": Fraction(2),
    "whole": Fraction(1),
    "half": Fraction(1, 2),
    "quarter": Fraction(1, 4),
    "eighth": Fraction(1, 8),
    **{
        name: Fraction(1, value)
        for name, value in (
            ("16th", 16),
            ("32nd", 32),
            ("64th", 64),
            ("128th", 128),
            ("256th", 256),
            ("512th", 512),
            ("1024th", 1024),
        )
    },
}


@dataclass
class _TabNote:
    pitch: int
    start: Fraction
    end: Fraction
    measure: int
    beat: Fraction
    elements: list[ET.Element] = field(default_factory=list)
    string: int | None = None


def _measure_times(staff: ET.Element) -> list[tuple[Fraction, Fraction]]:
    result = []
    start, meter = Fraction(0), Fraction(1)
    for measure in staff.findall("Measure"):
        signature = measure.find("./voice/TimeSig")
        if signature is not None:
            meter = Fraction(int(signature.findtext("sigN")), int(signature.findtext("sigD")))
        length = Fraction(measure.get("len", str(meter)))
        result.append((start, length))
        start += length
    return result


def _staff_notes(staff: ET.Element, times: list[tuple[Fraction, Fraction]]) -> list[_TabNote]:
    notes: list[_TabNote] = []
    pending_ties: dict[tuple[int, int, Fraction], _TabNote] = {}
    measures = staff.findall("Measure")
    if len(measures) != len(times):
        raise ValueError("Tablature staff has a different measure count")
    for number, (measure, (start, length)) in enumerate(zip(measures, times), 1):
        for voice_index, voice in enumerate(measure.findall("voice")):
            position = Fraction(0)
            tuplets: list[Fraction] = []
            for item in voice:
                if item.tag == "Tuplet":
                    tuplets.append(
                        Fraction(
                            int(item.findtext("normalNotes")), int(item.findtext("actualNotes"))
                        )
                    )
                    continue
                if item.tag == "endTuplet":
                    tuplets.pop()
                    continue
                if item.tag == "location":
                    position += Fraction(item.findtext("fractions", "0"))
                    continue
                if item.tag not in {"Chord", "Rest"}:
                    continue
                kind = item.findtext("durationType")
                duration = (
                    Fraction(item.findtext("duration", str(length)))
                    if kind == "measure"
                    else _DURATIONS[kind]
                )
                dots = int(item.findtext("dots", "0"))
                duration *= 2 - Fraction(1, 2**dots)
                for ratio in tuplets:
                    duration *= ratio
                onset, end = start + position, start + position + duration
                for element in item.findall("Note"):
                    pitch = int(element.findtext("pitch"))
                    tie_stop = element.find("./Spanner[@type='Tie']/prev") is not None
                    if tie_stop:
                        note = pending_ties.pop((voice_index, pitch, onset), None)
                        if note is None:
                            raise ValueError(
                                f"Unresolved tablature tie: measure={number}, pitch={pitch}"
                            )
                        note.end = end
                        note.elements.append(element)
                    else:
                        note = _TabNote(pitch, onset, end, number, position * 4, [element])
                        notes.append(note)
                    if element.find("./Spanner[@type='Tie']/next") is not None:
                        pending_ties[(voice_index, pitch, end)] = note
                position += duration
            if tuplets:
                raise ValueError("Unclosed tablature tuplet")
    if pending_ties:
        raise ValueError("Unresolved tablature tie at end of score")
    return notes


def _match_strings(
    pitches: list[int], tuning: list[int], frets: int, occupied: set[int]
) -> list[int] | None:
    """Find a complete matching; a greedy first-free-string choice is insufficient."""
    candidates = [
        sorted(
            (
                s
                for s, base in enumerate(tuning)
                if s not in occupied and 0 <= pitch - base <= frets
            ),
            key=lambda s: (pitch - tuning[s], s),
        )
        for pitch in pitches
    ]
    if any(not values for values in candidates):
        return None
    assigned: dict[int, int] = {}

    def augment(index: int, visited: set[int]) -> bool:
        for string in candidates[index]:
            if string in visited:
                continue
            visited.add(string)
            if string not in assigned or augment(assigned[string], visited):
                assigned[string] = index
                return True
        return False

    for index in sorted(range(len(pitches)), key=lambda i: (len(candidates[i]), pitches[i])):
        if not augment(index, set()):
            return None
    result = [0] * len(pitches)
    for string, index in assigned.items():
        result[index] = string
    return result


def assign_score_tablature(score: ET.Element, part_indices: list[int]) -> list[dict]:
    """Assign strings to complete parts and report exact unresolved locations."""
    staves = score.findall("Staff")
    times = _measure_times(staves[0]) if staves else []
    reports, staff_index = [], 0
    for part_index, part in enumerate(score.findall("Part")):
        part_staves = staves[staff_index : staff_index + len(part.findall("Staff"))]
        staff_index += len(part.findall("Staff"))
        if part_index not in part_indices:
            continue
        instrument = part.find("Instrument")
        transpose = int(instrument.findtext("transposeChromatic", "0"))
        tuning = list(
            reversed([int(s.text) + transpose for s in instrument.findall("./StringData/string")])
        )
        frets = int(instrument.findtext("./StringData/frets", "24"))
        issues = []
        for staff in part_staves:
            notes = _staff_notes(staff, times)
            starts: dict[Fraction, list[_TabNote]] = {}
            for note in notes:
                starts.setdefault(note.start, []).append(note)
            active: list[_TabNote] = []
            for onset, new_notes in sorted(starts.items()):
                active = [n for n in active if n.end > onset]
                pitches = [n.pitch for n in new_notes]
                occupied = {n.string for n in active if n.string is not None}
                assignment = _match_strings(pitches, tuning, frets, occupied)
                if assignment is None:
                    sounding = [n.pitch for n in active] + pitches
                    reason = (
                        "polyphony_exceeds_strings"
                        if len(sounding) > len(tuning)
                        else "no_string_assignment"
                    )
                    issues.append(
                        {
                            "measure": new_notes[0].measure,
                            "beat": float(new_notes[0].beat),
                            "reason": reason,
                            "pitches": sounding,
                            "strings": len(tuning),
                        }
                    )
                else:
                    for note, string in zip(new_notes, assignment):
                        note.string = string
                active.extend(new_notes)
            if not issues:
                for note in notes:
                    for element in note.elements:
                        for name, value in (
                            ("string", note.string),
                            ("fret", note.pitch - tuning[note.string]),
                        ):
                            child = element.find(name)
                            if child is None:
                                child = ET.SubElement(element, name)
                            child.text = str(value)
        reports.append(
            {
                "part": part_index + 1,
                "name": part.findtext("trackName", ""),
                "status": "unrepresentable" if issues else "assigned",
                "issues": issues,
            }
        )
    return reports
