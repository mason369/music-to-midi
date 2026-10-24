"""Restore note ends removed by MuseScore's MIDI overlap cleanup.

MuseScore's MIDI importer clips overlapping repetitions even with duration
simplification disabled. MusicXML can express these notes in separate voices.
This operates on the engraving copy, before triplet-coordinate restoration.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import mido

_FINE_GRID = Fraction(1, 256)  # A 1/1024 whole note, in quarter-note units.
_TYPES = {
    Fraction(4): "whole",
    Fraction(2): "half",
    Fraction(1): "quarter",
    Fraction(1, 2): "eighth",
    **{
        Fraction(4, denominator): name
        for denominator, name in (
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
    [
        (length * ratio, name, dotted)
        for length, name in _TYPES.items()
        for ratio, dotted in ((Fraction(1), False), (Fraction(3, 2), True))
    ],
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


@dataclass
class _Note:
    element: ET.Element
    start: Fraction
    end: Fraction
    staff: int
    voice: int


@dataclass
class _Measure:
    element: ET.Element
    start: Fraction
    length: Fraction
    divisions: int
    events: list[tuple[Fraction, ET.Element]]


@dataclass
class _Part:
    measures: list[_Measure] = field(default_factory=list)
    notes: list[_Note] = field(default_factory=list)
    removed: set[int] = field(default_factory=set)


@dataclass
class _Chain:
    part: _Part
    notes: list[_Note]

    @property
    def start(self):
        return self.notes[0].start

    @property
    def end(self):
        return self.notes[-1].end


def _source_notes(path: Path):
    midi = mido.MidiFile(path)
    programs, pending, expected = {}, defaultdict(deque), defaultdict(list)
    tick = 0
    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        if message.type == "program_change":
            programs[message.channel] = message.program
        elif message.type == "note_on" and message.velocity:
            program = programs.get(message.channel, 0) if message.channel != 9 else -1
            pending[message.channel, message.note].append(
                (Fraction(tick, midi.ticks_per_beat), program)
            )
        elif message.type == "note_off" or message.type == "note_on" and not message.velocity:
            key = message.channel, message.note
            if pending[key]:
                start, program = pending[key].popleft()
                expected[program, message.note].append((start, Fraction(tick, midi.ticks_per_beat)))
    return expected


def _read_parts(tree: ET.ElementTree):
    definitions = {part.get("id"): part for part in tree.findall("./part-list/score-part")}
    parts, chains = [], defaultdict(list)
    pitch_classes = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)
    for element in tree.findall("part"):
        definition = definitions[element.get("id")]
        program = int(definition.findtext("./midi-instrument/midi-program", "1")) - 1
        drums = {
            node.get("id"): int(node.findtext("midi-unpitched")) - 1
            for node in definition.findall("midi-instrument")
            if node.find("midi-unpitched") is not None
        }
        part = _Part()
        start, divisions, transpose = Fraction(0), 1, 0
        pending = {}
        for measure in element.findall("measure"):
            cursor, previous, length = Fraction(0), Fraction(0), Fraction(0)
            events = []
            for item in measure:
                if item.tag == "attributes":
                    divisions = int(item.findtext("divisions", str(divisions)))
                    if item.find("transpose") is not None:
                        transpose = int(item.findtext("transpose/chromatic", "0")) + 12 * int(
                            item.findtext("transpose/octave-change", "0")
                        )
                if item.tag in ("backup", "forward"):
                    cursor += (-1 if item.tag == "backup" else 1) * Fraction(
                        int(item.findtext("duration")), divisions
                    )
                    length = max(length, cursor)
                    continue
                position = cursor
                if item.tag == "note":
                    duration = Fraction(int(item.findtext("duration", "0")), divisions)
                    position = previous if item.find("chord") is not None else cursor
                    if item.find("chord") is None:
                        previous, cursor = cursor, cursor + duration
                    length = max(length, position + duration, cursor)
                    note = _Note(
                        item,
                        start + position,
                        start + position + duration,
                        int(item.findtext("staff", "1")),
                        int(item.findtext("voice", "1")),
                    )
                    part.notes.append(note)
                    if item.find("rest") is None:
                        pitch = item.find("pitch")
                        if pitch is None:
                            value = drums[item.find("instrument").get("id")]
                        else:
                            value = (
                                (int(pitch.findtext("octave")) + 1) * 12
                                + pitch_classes[pitch.findtext("step")]
                                + int(pitch.findtext("alter", "0"))
                                + transpose
                            )
                        key = note.staff, note.voice, value, note.start
                        if any(t.get("type") == "stop" for t in item.findall("tie")):
                            chain = pending.pop(key)
                            chain.notes.append(note)
                        else:
                            chain = _Chain(part, [note])
                            chains[program if pitch is not None else -1, value].append(chain)
                        if any(t.get("type") == "start" for t in item.findall("tie")):
                            pending[note.staff, note.voice, value, note.end] = chain
                events.append((position, item))
            part.measures.append(_Measure(measure, start, length, divisions, events))
            start += length
        if pending:
            raise ValueError("Unresolved MusicXML tie while restoring note durations")
        parts.append(part)
    return parts, chains


def _free_voice(chain: _Chain, end: Fraction):
    part, original = chain.part, chain.notes[0]
    own = {id(note.element) for note in chain.notes}
    staves = [original.staff] + sorted({note.staff for note in part.notes} - {original.staff})
    for staff in staves:
        for voice in range(4 * (staff - 1) + 1, 4 * staff + 1):
            if not any(
                note.staff == staff
                and note.voice == voice
                and id(note.element) not in part.removed | own
                and note.element.find("rest") is None
                and note.start < end
                and note.end > chain.start
                for note in part.notes
            ):
                return staff, voice
    # A staff supports four voices. An additional staff retains real polyphony
    # when every existing voice is occupied, instead of truncating a note.
    staff = max(staves) + 1
    attributes = part.measures[0].element.find("attributes")
    staves_node = attributes.find("staves")
    if staves_node is None:
        staves_node = ET.SubElement(attributes, "staves")
    staves_node.text = str(staff)
    clef = next(
        (
            node
            for node in attributes.findall("clef")
            if int(node.get("number", "1")) == original.staff
        ),
        None,
    )
    clef = (
        deepcopy(clef)
        if clef is not None
        else ET.fromstring("<clef><sign>G</sign><line>2</line></clef>")
    )
    clef.set("number", str(staff))
    attributes.append(clef)
    return staff, 4 * (staff - 1) + 1


def _restore_chain(chain: _Chain, end: Fraction) -> None:
    part, start = chain.part, chain.start
    if not part.removed:
        # Clipping can remove the only fine subdivision in a part, leaving
        # divisions=1. Restore a sufficient exact resolution and declare it in
        # every measure so an inherited declaration cannot change later notes.
        for measure in part.measures:
            resolution = math.lcm(measure.divisions, _FINE_GRID.denominator)
            factor = resolution // measure.divisions
            for _, item in measure.events:
                for tag in ("duration", "offset"):
                    for node in item.findall(f".//{tag}"):
                        node.text = str(int(node.text) * factor)
            attributes = measure.element.find("attributes")
            if attributes is None:
                attributes = ET.Element("attributes")
                measure.events.insert(0, (Fraction(0), attributes))
            divisions = attributes.find("divisions")
            if divisions is None:
                divisions = ET.Element("divisions")
                attributes.insert(0, divisions)
            divisions.text = str(resolution)
            measure.divisions = resolution
    staff, voice = _free_voice(chain, end)
    part.removed.update(id(note.element) for note in chain.notes)
    for note in part.notes:
        if (
            note.staff == staff
            and note.voice == voice
            and note.element.find("rest") is not None
            and note.start < end
            and note.end > start
        ):
            part.removed.add(id(note.element))
    position = start
    for measure in part.measures:
        finish = min(end, measure.start + measure.length)
        if position >= finish or position < measure.start:
            continue
        while position < finish:
            choice = next((piece for piece in _PIECES if piece[0] <= finish - position), None)
            if choice is None:
                raise ValueError("Restored note duration is below the 1/1024 notation grid")
            amount, kind, dotted = choice
            node = deepcopy(chain.notes[0].element)
            for tag in (
                "chord",
                "duration",
                "type",
                "dot",
                "beam",
                "time-modification",
                "tie",
                "notations",
            ):
                for child in node.findall(tag):
                    node.remove(child)
            for attribute in ("default-x", "default-y"):
                node.attrib.pop(attribute, None)
            for tag, value in (("voice", voice), ("staff", staff)):
                child = node.find(tag)
                if child is None:
                    child = ET.SubElement(node, tag)
                child.text = str(value)
            duration = amount * measure.divisions
            if duration.denominator != 1:
                raise ValueError("MusicXML divisions cannot represent the restored duration")
            ET.SubElement(node, "duration").text = str(duration.numerator)
            ET.SubElement(node, "type").text = kind
            if dotted:
                ET.SubElement(node, "dot")
            if node.find("stem") is not None:
                node.find("stem").text = "up" if voice % 2 else "down"
            if position > start:
                for child in node.findall("accidental"):
                    node.remove(child)
            notations = ET.SubElement(node, "notations")
            for label, enabled in (
                ("stop", position > start),
                ("start", position + amount < end),
            ):
                if enabled:
                    ET.SubElement(node, "tie", type=label)
                    ET.SubElement(notations, "tied", type=label)
            node[:] = sorted(node, key=lambda child: _NOTE_ORDER.get(child.tag, 99))
            measure.events.append((position - measure.start, node))
            part.notes.append(_Note(node, position, position + amount, staff, voice))
            position += amount
    if position != end:
        raise ValueError("Restored note extends outside the engraved score")


def _write_part(part: _Part) -> None:
    for measure in part.measures:
        output = measure.element
        output[:] = []
        cursor, previous = Fraction(0), None

        def shift(target):
            nonlocal cursor
            if target != cursor:
                move = ET.SubElement(output, "forward" if target > cursor else "backup")
                ET.SubElement(move, "duration").text = str(
                    int(abs(target - cursor) * measure.divisions)
                )
                cursor = target

        for position, item in sorted(
            measure.events,
            key=lambda event: (
                0 if event[1].tag != "note" and event[0] == 0 else 1,
                event[0],
                int(event[1].findtext("voice", "0")),
            ),
        ):
            if id(item) in part.removed:
                continue
            if item.tag == "note":
                for child in item.findall("chord"):
                    item.remove(child)
                key = (
                    position,
                    item.findtext("voice", "1"),
                    item.findtext("staff", "1"),
                    item.findtext("duration"),
                )
                if key == previous and item.find("rest") is None:
                    item.insert(0, ET.Element("chord"))
                else:
                    shift(position)
                    cursor += Fraction(int(item.findtext("duration")), measure.divisions)
                previous = key if item.find("rest") is None else None
            else:
                shift(position)
                previous = None
            output.append(item)
        shift(measure.length)


def restore_note_durations(midi_path: Path, musicxml: Path, *, import_grid: Fraction) -> None:
    """Keep imported note starts and restore clipped ends in independent voices."""
    expected = _source_notes(midi_path)
    tree = ET.parse(musicxml)
    parts, chains = _read_parts(tree)
    if chains.keys() != expected.keys():
        raise ValueError("MuseScore changed the source instruments or pitches")
    for key, actual in chains.items():
        wanted = sorted(expected[key])
        actual = sorted(actual, key=lambda chain: (chain.start, chain.end))
        if len(actual) != len(wanted):
            raise ValueError(f"MuseScore changed the source note count for instrument/pitch {key}")
        for chain, (start, end) in zip(actual, wanted):
            if abs(chain.start - start) > import_grid:
                raise ValueError(f"MuseScore changed a note onset beyond its import grid: {key}")
            # Partial quantization can lie between notation ticks. This is the
            # same finite 1/1024 grid, not a second full-strength quantization.
            target_end = round(end / _FINE_GRID) * _FINE_GRID
            if abs(chain.end - target_end) <= _FINE_GRID:
                continue
            if target_end <= chain.start:
                raise ValueError("Source note is shorter than the notation resolution")
            _restore_chain(chain, target_end)
    for part in parts:
        if part.removed:
            _write_part(part)
    tree.write(musicxml, encoding="UTF-8", xml_declaration=True)
