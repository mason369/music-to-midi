"""Real MusicXML round trips must retain note ends, including repeated pitches."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from fractions import Fraction

import mido
import pytest

from src.core.sheet_music import _run_musescore, export_sheet_music_zip
from src.core.sheet_note_durations import restore_note_durations
from src.utils.musescore_runtime import get_musescore_executable


def _intervals(path):
    result = []
    for part in ET.parse(path).findall("part"):
        start, divisions, transpose = Fraction(0), 1, 0
        pending = {}
        for measure in part.findall("measure"):
            position, previous, length = Fraction(0), Fraction(0), Fraction(0)
            for node in measure:
                if node.tag == "attributes":
                    divisions = int(node.findtext("divisions", str(divisions)))
                    if node.find("transpose") is not None:
                        transpose = int(node.findtext("transpose/chromatic", "0")) + 12 * int(
                            node.findtext("transpose/octave-change", "0")
                        )
                elif node.tag in ("forward", "backup"):
                    position += (-1 if node.tag == "backup" else 1) * Fraction(
                        int(node.findtext("duration")), divisions
                    )
                elif node.tag == "note":
                    duration = Fraction(int(node.findtext("duration", "0")), divisions)
                    onset = previous if node.find("chord") is not None else position
                    if node.find("chord") is None:
                        previous, position = position, position + duration
                    length = max(length, onset + duration, position)
                    p = node.find("pitch")
                    if p is None:
                        continue
                    pitch = (
                        (int(p.findtext("octave")) + 1) * 12
                        + dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)[p.findtext("step")]
                        + int(p.findtext("alter", "0"))
                        + transpose
                    )
                    key = node.findtext("voice", "1"), node.findtext("staff", "1"), pitch
                    if any(t.get("type") == "stop" for t in node.findall("tie")):
                        original, expected_start = pending.pop(key)
                        assert expected_start == start + onset
                    else:
                        original = start + onset
                    if any(t.get("type") == "start" for t in node.findall("tie")):
                        pending[key] = original, start + onset + duration
                    else:
                        result.append((pitch, original, start + onset + duration))
                length = max(length, position)
            start += length
        assert not pending
    return Counter(result)


def test_restoration_recovers_subdivision_lost_with_the_overlap(tmp_path):
    source = tmp_path / "source.mid"
    midi = mido.MidiFile(ticks_per_beat=960)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("program_change", program=0),
                mido.Message("note_on", note=60, velocity=90),
                mido.Message("note_on", note=60, velocity=90, time=3840),
                mido.Message("note_off", note=60, time=120),
                mido.Message("note_off", note=60, time=840),
            ]
        )
    )
    midi.save(source)
    xml = tmp_path / "clipped.musicxml"
    xml.write_text(
        '<score-partwise><part-list><score-part id="P1"><midi-instrument id="I1">'
        "<midi-program>1</midi-program></midi-instrument></score-part></part-list>"
        '<part id="P1"><measure><attributes><divisions>1</divisions>'
        "<time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
        "<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration>"
        "<voice>1</voice><type>whole</type></note></measure><measure>"
        "<note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>"
        "<voice>1</voice><type>quarter</type></note><note><rest/><duration>3</duration>"
        "<voice>1</voice><type>half</type><dot/></note></measure></part></score-partwise>",
        encoding="utf-8",
    )
    restore_note_durations(source, xml, import_grid=Fraction(1, 16))
    assert _intervals(xml) == Counter(
        [(60, Fraction(0), Fraction(33, 8)), (60, Fraction(4), Fraction(5))]
    )


@pytest.mark.skipif(
    os.environ.get("MUSIC_TO_MIDI_TEST_MUSESCORE") != "1",
    reason="requires explicit real MuseScore acceptance",
)
@pytest.mark.parametrize("triplet", [False, True])
def test_actual_score_reopening_preserves_overlapping_repetitions_and_ties(tmp_path, triplet):
    ppq = 960
    midi = mido.MidiFile(ticks_per_beat=ppq)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("time_signature", numerator=4, denominator=4),
                mido.MetaMessage("set_tempo", tempo=500000),
            ]
        )
    )
    expected = []
    # Same-pitch overlap spans a barline, occurs within a bar, and uses a
    # transposing instrument. Ends deliberately differ from the next start.
    for channel, program, pitch, notes in (
        (0, 0, 60, [(0, 4480), (3840, 5120)]),
        (1, 33, 40, [(0, 2240), (1920, 3200)]),
        (2, 65, 69, [(320, 2880), (2560, 4160)]),
    ):
        track = mido.MidiTrack([mido.Message("program_change", channel=channel, program=program)])
        events = []
        for start, end in notes:
            # Both fixtures lie exactly on their explicit quantization grid.
            if not triplet:
                start, end = round(start / 120) * 120, round(end / 120) * 120
            expected.append((pitch, Fraction(start, ppq), Fraction(end, ppq)))
            events.extend(
                [
                    (start, mido.Message("note_on", channel=channel, note=pitch, velocity=90)),
                    (end, mido.Message("note_off", channel=channel, note=pitch)),
                ]
            )
        previous = 0
        for tick, message in sorted(
            events, key=lambda event: (event[0], event[1].type == "note_on")
        ):
            track.append(message.copy(time=tick - previous))
            previous = tick
        midi.tracks.append(track)
    source = tmp_path / "source.mid"
    midi.save(source)
    original = source.read_bytes()
    result = export_sheet_music_zip(
        source, tmp_path / "score.zip", quantize_grid="1/32", quantize_triplet=triplet
    )
    assert source.read_bytes() == original
    xml = tmp_path / "score.musicxml"
    with zipfile.ZipFile(result.path) as archive:
        xml.write_bytes(archive.read("score.musicxml"))
    assert _intervals(xml) == Counter(expected)
    reopened = tmp_path / "reopened.musicxml"
    completed = _run_musescore(
        get_musescore_executable(), ["-o", str(reopened), str(xml)], cancel_check=lambda: False
    )
    assert completed.returncode == 0, completed.stderr
    assert _intervals(reopened) == Counter(expected)
