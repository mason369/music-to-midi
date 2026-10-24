"""Explicit triplet notation for MuseScore, independent of meter heuristics.

The import copy uses a 3/2 rhythmic coordinate transform: ternary subdivisions
become binary. MusicXML then restores the original clock and meter with explicit
3:2 groups, split at quarter beats and barlines. No model/MIDI artifact is edited.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import mido

_TYPES = {
    Fraction(8): "breve",
    Fraction(4): "whole",
    Fraction(2): "half",
    Fraction(1): "quarter",
    Fraction(1, 2): "eighth",
    **{
        Fraction(4, d): name
        for d, name in (
            (16, "16th"),
            (32, "32nd"),
            (64, "64th"),
            (128, "128th"),
            (256, "256th"),
            (512, "512th"),
            (1024, "1024th"),
        )
    },
}
_PIECES = sorted(
    (
        (value * (Fraction(3, 2) if dot else 1) * Fraction(2, 3), name, dot)
        for value, name in _TYPES.items()
        for dot in (True, False)
    ),
    reverse=True,
)
_NOTE_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "grace",
            "cue",
            "chord",
            "pitch",
            "unpitched",
            "rest",
            "duration",
            "tie",
            "instrument",
            "footnote",
            "level",
            "voice",
            "type",
            "dot",
            "accidental",
            "time-modification",
            "stem",
            "notehead",
            "notehead-text",
            "staff",
            "beam",
            "notations",
            "lyric",
            "play",
            "listen",
        )
    )
}


def prepare_triplet_import(source: Path, destination: Path) -> list[tuple[Fraction, int, int]]:
    midi = mido.MidiFile(source)
    if midi.ticks_per_beat > 16383:
        raise ValueError("Triplet notation requires MIDI PPQ at most 16383")
    signatures = {}
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            message.time *= 3
            if message.type == "time_signature":
                signatures[tick] = (message.numerator, message.denominator)
                fraction = Fraction(message.numerator * 3, message.denominator * 2)
                if fraction.numerator > 255:
                    raise ValueError("Time signature exceeds the MIDI notation import range")
                message.numerator, message.denominator = fraction.numerator, fraction.denominator
    if 0 not in signatures:
        signatures[0] = (4, 4)
        midi.tracks[0].insert(0, mido.MetaMessage("time_signature", numerator=3, denominator=2))
    original_ppq = midi.ticks_per_beat
    midi.ticks_per_beat *= 2
    midi.save(destination)
    return [
        (Fraction(tick, original_ppq), numerator, denominator)
        for tick, (numerator, denominator) in sorted(signatures.items())
    ]


def _fragment(
    item: ET.Element,
    onset: Fraction,
    end: Fraction,
    start: Fraction,
    amount: Fraction,
    kind: str,
    dotted: bool,
    divisions: int,
    base: Fraction,
) -> ET.Element:
    note = deepcopy(item)
    note.attrib.pop("default-x", None)
    note.attrib.pop("default-y", None)
    for tag in ("chord", "duration", "type", "dot", "beam", "time-modification", "tie"):
        for child in note.findall(tag):
            note.remove(child)
    notations = note.find("notations")
    if notations is None:
        notations = ET.SubElement(note, "notations")
    for tag in ("tuplet", "tied"):
        for child in notations.findall(tag):
            notations.remove(child)
    rest = note.find("rest")
    if rest is not None:
        rest.attrib.pop("measure", None)
    duration = amount * divisions
    if duration.denominator != 1:
        raise ValueError("Triplet duration is not representable at the MusicXML resolution")
    ET.SubElement(note, "duration").text = str(duration.numerator)
    ET.SubElement(note, "type").text = kind
    if dotted:
        ET.SubElement(note, "dot")
    modification = ET.SubElement(note, "time-modification")
    ET.SubElement(modification, "actual-notes").text = "3"
    ET.SubElement(modification, "normal-notes").text = "2"
    ET.SubElement(modification, "normal-type").text = _TYPES[base]
    if rest is None:
        ties = {tie.get("type") for tie in item.findall("tie")}
        if start > onset:
            ties.add("stop")
            for child in note.findall("accidental"):
                note.remove(child)
        if start + amount < end:
            ties.add("start")
        for kind in sorted(ties):
            ET.SubElement(note, "tie", type=kind)
            ET.SubElement(notations, "tied", type=kind)
    note[:] = sorted(note, key=lambda child: _NOTE_ORDER.get(child.tag, 99))
    return note


def _tuplet_boundary(note: ET.Element, kind: str, base: Fraction) -> None:
    tuplet = ET.SubElement(note.find("notations"), "tuplet", type=kind, number="1")
    if kind == "start":
        for name, count in (("actual", 3), ("normal", 2)):
            child = ET.SubElement(tuplet, f"tuplet-{name}")
            ET.SubElement(child, "tuplet-number").text = str(count)
            ET.SubElement(child, "tuplet-type").text = _TYPES[base]


def restore_triplet_musicxml(path: Path, signatures: list[tuple[Fraction, int, int]]) -> None:
    tree = ET.parse(path)
    for node in tree.findall(".//divisions"):
        node.text = str(int(node.text) * 3)
    for tag in ("duration", "offset"):
        for node in tree.findall(f".//{tag}"):
            node.text = str(int(node.text) * 2)
    for part in tree.getroot().findall("part"):
        # MIDI import merges repeated meters and equivalent transformed ratios
        # (e.g. original 3/4 and 6/8 both become 9/8). Match the source clock,
        # not the number of surviving MusicXML declarations.
        pending_signatures = {beat: (n, d) for beat, n, d in signatures}
        measure_start = Fraction(0)
        divisions, meter = 1, Fraction(4)
        for measure in part.findall("measure"):
            for attributes in measure.findall("attributes"):
                for node in attributes.findall("time"):
                    attributes.remove(node)
            if measure_start in pending_signatures:
                numerator, denominator = pending_signatures.pop(measure_start)
                attributes = measure.find("attributes")
                if attributes is None:
                    attributes = ET.Element("attributes")
                    measure.insert(0, attributes)
                node = ET.Element("time")
                ET.SubElement(node, "beats").text = str(numerator)
                ET.SubElement(node, "beat-type").text = str(denominator)
                insertion = next(
                    (
                        i
                        for i, child in enumerate(attributes)
                        if child.tag not in {"divisions", "key"}
                    ),
                    len(attributes),
                )
                attributes.insert(insertion, node)
            measure.attrib.pop("width", None)
            voices, metadata = {}, []
            cursor, previous = Fraction(0), Fraction(0)
            for item in measure:
                if item.tag == "attributes":
                    divisions = int(item.findtext("divisions", str(divisions)))
                    if item.find("time") is not None:
                        meter = Fraction(
                            int(item.findtext("time/beats")) * 4,
                            int(item.findtext("time/beat-type")),
                        )
                    metadata.append((cursor, item))
                elif item.tag in {"backup", "forward"}:
                    cursor += (-1 if item.tag == "backup" else 1) * Fraction(
                        int(item.findtext("duration")), divisions
                    )
                elif item.tag == "note":
                    duration = Fraction(int(item.findtext("duration")), divisions)
                    onset = previous if item.find("chord") is not None else cursor
                    if item.find("chord") is None:
                        previous, cursor = cursor, cursor + duration
                    voices.setdefault(item.findtext("voice", "1"), []).append(
                        (onset, onset + duration, item)
                    )
                else:
                    metadata.append((cursor, item))
            output_divisions = divisions * 4
            measure[:] = []
            scheduled = {}
            trailing = []
            for position, item in metadata:
                if item.tag == "attributes":
                    child = item.find("divisions")
                    if child is None:
                        child = ET.SubElement(item, "divisions")
                    child.text = str(output_divisions)
                elif item.tag == "direction":
                    offset = item.find("offset")
                    absolute = position + (
                        Fraction(int(offset.text), divisions) if offset is not None else 0
                    )
                    if offset is None:
                        offset = ET.SubElement(item, "offset")
                    offset.text = str(int(absolute * output_divisions))
                    position = Fraction(0)
                elif item.tag == "barline" and item.get("location", "right") == "right":
                    trailing.append(item)
                    continue
                scheduled.setdefault(position, []).append(item)
            if not any(item.tag == "attributes" for item in scheduled.get(Fraction(0), [])):
                attributes = ET.Element("attributes")
                ET.SubElement(attributes, "divisions").text = str(output_divisions)
                scheduled.setdefault(Fraction(0), []).insert(0, attributes)
            for voice_index, events in enumerate(voices.values()):
                if voice_index:
                    backup = ET.SubElement(measure, "backup")
                    ET.SubElement(backup, "duration").text = str(int(meter * output_divisions))
                group_start = Fraction(0)
                while group_start < meter:
                    # An irregular short bar can end with 3/4 or 7/8 of a
                    # quarter beat. Split that tail into binary spans so every
                    # explicit 3:2 group has a valid MusicXML normal-type.
                    span = Fraction(1)
                    while span > meter - group_start:
                        span /= 2
                    group_end = group_start + span
                    base = span / 2
                    boundaries = {group_start, group_end}
                    for onset, end, _ in events:
                        boundaries.update(p for p in (onset, end) if group_start < p < group_end)
                    points, group_leaders = sorted(boundaries), []
                    for start, end in zip(points, points[1:]):
                        if voice_index == 0:
                            measure.extend(scheduled.pop(start, []))
                        active = [(a, b, note) for a, b, note in events if a <= start < b]
                        if not active:
                            rest = ET.Element("note")
                            ET.SubElement(rest, "rest")
                            if events:
                                for tag in ("voice", "staff"):
                                    node = events[0][2].find(tag)
                                    if node is not None:
                                        rest.append(deepcopy(node))
                            active = [(start, end, rest)]
                        position = start
                        while position < end:
                            choice = next(
                                (piece for piece in _PIECES if piece[0] <= end - position), None
                            )
                            if choice is None:
                                raise ValueError(
                                    "Triplet notation contains a duration below 1/1024"
                                )
                            amount, kind, dotted = choice
                            for index, (onset, stop, item) in enumerate(active):
                                note = _fragment(
                                    item,
                                    onset,
                                    stop,
                                    position,
                                    amount,
                                    kind,
                                    dotted,
                                    output_divisions,
                                    base,
                                )
                                if index:
                                    note.insert(0, ET.Element("chord"))
                                else:
                                    group_leaders.append(note)
                                measure.append(note)
                            if voice_index == 0:
                                # Clef changes retain MIDI-tick positions, even
                                # when the notes use a finer binary import grid.
                                # Position metadata independently instead of
                                # splitting a note/rest into unnotatable pieces.
                                cursor = position + amount
                                for point in sorted(p for p in scheduled if position < p <= cursor):
                                    distance = (cursor - point) * output_divisions
                                    if distance.denominator != 1:
                                        raise ValueError(
                                            "Score metadata exceeds the MusicXML resolution"
                                        )
                                    if distance:
                                        backup = ET.SubElement(measure, "backup")
                                        ET.SubElement(backup, "duration").text = str(int(distance))
                                    measure.extend(scheduled.pop(point))
                                    if distance:
                                        forward = ET.SubElement(measure, "forward")
                                        ET.SubElement(forward, "duration").text = str(int(distance))
                            position += amount
                    if group_leaders:
                        _tuplet_boundary(group_leaders[0], "start", base)
                        _tuplet_boundary(group_leaders[-1], "stop", base)
                    group_start = group_end
            measure.extend(scheduled.get(meter, []))
            measure.extend(trailing)
            measure_start += meter
        if pending_signatures:
            raise ValueError("Source time-signature changes do not align with engraved measures")
    tree.write(path, encoding="UTF-8", xml_declaration=True)
