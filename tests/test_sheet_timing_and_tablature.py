"""Notation acceptance covers real note timing, ties and physical strings."""

import collections
import io
import os
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction

import mido
import pytest

from src.core.sheet_music import _midi_import_options, _run_musescore, export_sheet_music_zip
from src.core.sheet_tablature import _match_strings, assign_score_tablature
from src.core.sheet_triplets import prepare_triplet_import, restore_triplet_musicxml
from src.utils.musescore_runtime import get_musescore_executable


def test_matching_uses_distinct_strings_when_greedy_fretting_conflicts():
    pitches = [45, 52, 57, 61, 64, 85]
    tuning = [64, 59, 55, 50, 45, 40]
    strings = _match_strings(pitches, tuning, 24, set())
    assert strings is not None and len(set(strings)) == 6
    assert all(0 <= pitch - tuning[string] <= 24 for pitch, string in zip(pitches, strings))
    assert _match_strings([45, 52, 54, 57, 61, 64, 66], tuning, 24, set()) is None
    assert _match_strings([39], tuning, 24, set()) is None


def _score(chords, *, transpose=0):
    tuning = "".join(f"<string>{n}</string>" for n in (40, 45, 50, 55, 59, 64))
    return ET.fromstring(
        f"<Score><Part><Staff/><trackName>Guitar</trackName><Instrument>"
        f"<transposeChromatic>{transpose}</transposeChromatic>"
        f"<StringData><frets>24</frets>{tuning}</StringData></Instrument></Part>"
        f"<Staff><Measure>{chords}</Measure></Staff></Score>"
    )


def _chord(pitches, duration="quarter", tie=""):
    return (
        f"<Chord><durationType>{duration}</durationType>"
        + "".join(f"<Note><pitch>{pitch}</pitch>{tie}</Note>" for pitch in pitches)
        + "</Chord>"
    )


def test_tablature_keeps_tied_notes_on_one_string_and_reserves_sustain():
    first = _chord([40], "half", '<Spanner type="Tie"><next/></Spanner>')
    second = _chord([40], "half", '<Spanner type="Tie"><prev/></Spanner>')
    score = _score(
        f'<voice>{first}{second}</voice><voice>{_chord([45,50,55,59,64], "whole")}</voice>'
    )
    report = assign_score_tablature(score, [0])
    assert report[0]["status"] == "assigned"
    notes = score.findall("./Staff/Measure/voice/Chord/Note")
    assert notes[0].findtext("string") == notes[1].findtext("string") == "5"
    assert len({n.findtext("string") for n in [notes[0], *notes[2:]]}) == 6
    assert [int(n.findtext("pitch")) for n in notes] == [40, 40, 45, 50, 55, 59, 64]


def test_tablature_reports_impossible_source_without_rewriting_notes():
    score = _score(f"<voice>{_chord([40,45,50,55,59,64,67])}</voice>")
    original = ET.tostring(score)
    report = assign_score_tablature(score, [0])
    assert ET.tostring(score) == original
    assert report[0]["status"] == "unrepresentable"
    assert report[0]["issues"] == [
        dict(
            measure=1,
            beat=0.0,
            reason="polyphony_exceeds_strings",
            pitches=[40, 45, 50, 55, 59, 64, 67],
            strings=6,
        )
    ]


def test_tablature_rejects_held_string_collision_and_handles_transposition():
    score = _score(
        f'<voice>{_chord([28], "whole")}</voice><voice><Rest><durationType>quarter</durationType></Rest>{_chord([29])}</voice>',
        transpose=-12,
    )
    report = assign_score_tablature(score, [0])
    assert report[0]["issues"][0]["reason"] == "no_string_assignment"
    assert report[0]["issues"][0]["beat"] == 1.0
    playable = _score(f"<voice>{_chord([28,33,38,43,47,52])}</voice>", transpose=-12)
    assert assign_score_tablature(playable, [0])[0]["status"] == "assigned"
    assert {n.text for n in playable.findall("./Staff/Measure/voice/Chord/Note/fret")} == {"0"}


@pytest.mark.parametrize(
    "grid,index", [("1/4", 0), ("1/8", 1), ("1/16", 2), ("1/32", 3), ("1/64", 4)]
)
def test_import_options_respect_grid_without_reestimating_performance(grid, index):
    options = ET.fromstring(_midi_import_options(grid, 1, False))
    assert options.findtext("QuantValue") == "4"
    assert options.findtext("HumanPerformance") == "false"
    assert options.findtext("Triplets") == "false"
    assert options.findtext("RecognizePickupBar") == "false"
    partial = ET.fromstring(_midi_import_options(grid, 0.5, True))
    assert partial.findtext("QuantValue") == "8"
    assert partial.findtext("Triplets") == "true"


def _xml_attacks(path):
    """Independent MusicXML cursor, including backup/chords/tied continuations."""
    found = []
    for part in ET.parse(path).getroot().findall("part"):
        start, divisions = Fraction(0), 1
        for measure in part.findall("measure"):
            position, previous, furthest = Fraction(0), Fraction(0), Fraction(0)
            for item in measure:
                if item.tag == "attributes":
                    divisions = int(item.findtext("divisions", str(divisions)))
                elif item.tag in {"backup", "forward"}:
                    position += (-1 if item.tag == "backup" else 1) * Fraction(
                        int(item.findtext("duration")), divisions
                    )
                elif item.tag == "note":
                    duration = Fraction(int(item.findtext("duration", "0")), divisions)
                    onset = previous if item.find("chord") is not None else position
                    if item.find("chord") is None:
                        previous = position
                        position += duration
                    furthest = max(furthest, onset + duration, position)
                    if item.find("pitch") is not None and not any(
                        t.get("type") == "stop" for t in item.findall("tie")
                    ):
                        p = item.find("pitch")
                        pitch = (
                            (int(p.findtext("octave")) + 1) * 12
                            + dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)[p.findtext("step")]
                            + int(p.findtext("alter", "0"))
                        )
                        found.append((pitch, start + onset))
            start += furthest
    return collections.Counter(found)


@pytest.fixture
def midnote_clef_score(tmp_path):
    path = tmp_path / "midnote-clefs.musicxml"
    path.write_text(
        '<score-partwise version="4.0"><part-list><score-part id="P1">'
        '<part-name>Piano</part-name></score-part></part-list><part id="P1">'
        '<measure number="1"><attributes><divisions>480</divisions>'
        "<time><beats>3</beats><beat-type>4</beat-type></time>"
        "<clef><sign>G</sign><line>2</line></clef></attributes>"
        "<note><pitch><step>E</step><octave>4</octave></pitch><duration>1440</duration>"
        "<voice>1</voice><type>half</type><dot/></note>"
        "<backup><duration>829</duration></backup>"
        "<attributes><clef><sign>F</sign><line>4</line></clef></attributes>"
        "<forward><duration>585</duration></forward>"
        "<attributes><clef><sign>G</sign><line>2</line></clef></attributes>"
        "<forward><duration>244</duration></forward></measure></part></score-partwise>",
        encoding="utf-8",
    )
    return path


def test_triplet_metadata_does_not_split_sustains_at_non_grid_ticks(midnote_clef_score):
    restore_triplet_musicxml(midnote_clef_score, [(Fraction(0), 2, 4)])
    cursor, divisions, clefs = Fraction(0), 1, []
    durations = []
    for item in ET.parse(midnote_clef_score).find("./part/measure"):
        if item.tag == "attributes":
            divisions = int(item.findtext("divisions", str(divisions)))
            if item.find("clef") is not None:
                clefs.append((cursor, item.findtext("clef/sign")))
        elif item.tag in {"forward", "backup"}:
            cursor += (-1 if item.tag == "backup" else 1) * Fraction(
                int(item.findtext("duration")), divisions
            )
        elif item.tag == "note":
            length = Fraction(int(item.findtext("duration")), divisions)
            durations.append(length)
            cursor += length
    assert clefs == [(0, "G"), (Fraction(611, 720), "F"), (Fraction(299, 180), "G")]
    assert durations == [1, 1] and cursor == 2
    assert _xml_attacks(midnote_clef_score) == {(64, Fraction(0)): 1}


@pytest.mark.skipif(
    os.environ.get("MUSIC_TO_MIDI_TEST_MUSESCORE") != "1",
    reason="requires explicit real MuseScore acceptance",
)
def test_real_triplet_midnote_clefs_preserve_pitch_and_sustain(midnote_clef_score):
    restore_triplet_musicxml(midnote_clef_score, [(Fraction(0), 2, 4)])
    output = midnote_clef_score.with_name("reopened.musicxml")
    process = _run_musescore(
        get_musescore_executable(),
        ["-o", str(output), str(midnote_clef_score)],
        cancel_check=lambda: False,
    )
    assert process.returncode == 0, process.stderr
    assert _xml_attacks(output) == {(64, Fraction(0)): 1}
    xml = ET.parse(output)
    divisions = int(xml.findtext("./part/measure/attributes/divisions"))
    durations = [
        Fraction(int(note.findtext("duration")), divisions)
        for note in xml.findall("./part/measure/note")
        if note.find("pitch") is not None
    ]
    assert sum(durations) == 2
    assert [node.findtext("sign") for node in xml.findall("./part/measure/attributes/clef")] == [
        "G",
        "F",
        "G",
    ]


@pytest.mark.skipif(
    os.environ.get("MUSIC_TO_MIDI_TEST_MUSESCORE") != "1",
    reason="requires explicit real MuseScore acceptance",
)
@pytest.mark.parametrize("grid", ["1/4", "1/8", "1/16", "1/32", "1/64"])
@pytest.mark.parametrize("triplet", [False, True])
@pytest.mark.parametrize("meter", [(4, 4), (6, 8), (9, 8), (5, 8), (7, 16), (7, 32), (11, 32)])
def test_real_musescore_preserves_quantized_note_starts(tmp_path, grid, triplet, meter):
    binary = get_musescore_executable()
    ppq = 960
    step = Fraction(4, int(grid[2:])) * (Fraction(2, 3) if triplet else 1)
    expected = [(60 + i % 9, step * i) for i in range(24)]
    midi = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack(
        [
            mido.Message("program_change", program=0),
            mido.MetaMessage("time_signature", numerator=meter[0], denominator=meter[1]),
            mido.MetaMessage("set_tempo", tempo=625000),
        ]
    )
    events = []
    for pitch, start in expected:
        events.extend(
            [
                (int(start * ppq), mido.Message("note_on", note=pitch, velocity=90)),
                (int((start + step) * ppq), mido.Message("note_off", note=pitch)),
            ]
        )
    previous = 0
    for tick, message in sorted(events, key=lambda e: (e[0], e[1].type == "note_on")):
        track.append(message.copy(time=tick - previous))
        previous = tick
    midi.tracks.append(track)
    source = tmp_path / "source.mid"
    midi.save(source)
    options = tmp_path / "import.xml"
    options.write_text(_midi_import_options(grid, 1, False), encoding="utf-8")
    output = tmp_path / "score.musicxml"
    if triplet:
        helper = tmp_path / "notation-import.mid"
        signatures = prepare_triplet_import(source, helper)
        source = helper
    process = _run_musescore(
        binary, ["-M", str(options), "-o", str(output), str(source)], cancel_check=lambda: False
    )
    assert process.returncode == 0, process.stderr
    if triplet:
        restore_triplet_musicxml(output, signatures)
        roundtrip = tmp_path / "roundtrip.musicxml"
        process = _run_musescore(
            binary, ["-o", str(roundtrip), str(output)], cancel_check=lambda: False
        )
        assert process.returncode == 0, process.stderr
        output = roundtrip
    assert _xml_attacks(output) == collections.Counter(expected)
    xml = ET.parse(output)
    assert xml.findtext("./part/measure/attributes/time/beats") == str(meter[0])
    assert xml.findtext("./part/measure/attributes/time/beat-type") == str(meter[1])


@pytest.mark.skipif(
    os.environ.get("MUSIC_TO_MIDI_TEST_MUSESCORE") != "1",
    reason="requires explicit real MuseScore acceptance",
)
@pytest.mark.parametrize("tempo_tick", [960, 307])
@pytest.mark.parametrize("next_meter", [(9, 8), (6, 8), (3, 4)])
def test_real_triplet_export_preserves_polyphony_tempo_meter_and_barline_ties(
    tmp_path, tempo_tick, next_meter
):
    midi = mido.MidiFile(ticks_per_beat=960)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("time_signature", numerator=6, denominator=8),
                mido.MetaMessage("set_tempo", tempo=480000),
                mido.MetaMessage("set_tempo", tempo=666667, time=tempo_tick),
                mido.MetaMessage(
                    "time_signature",
                    numerator=next_meter[0],
                    denominator=next_meter[1],
                    time=2880 - tempo_tick,
                ),
            ]
        )
    )
    source_notes = [
        [(60, 0, 1920), (64, 320, 640), (67, 640, 1920), (71, 2560, 3200), (72, 2880, 3520)],
        [(48, 0, 2880), (52, 2880, 4160)],
    ]
    expected = []
    for index, notes in enumerate(source_notes):
        track = mido.MidiTrack([mido.Message("program_change", program=0, channel=index)])
        events = []
        for pitch, start, end in notes:
            expected.append((pitch, Fraction(start, 960)))
            events.extend(
                [
                    (start, mido.Message("note_on", channel=index, note=pitch, velocity=90)),
                    (end, mido.Message("note_off", channel=index, note=pitch)),
                ]
            )
        previous = 0
        for tick, message in sorted(events, key=lambda x: (x[0], x[1].type == "note_on")):
            track.append(message.copy(time=tick - previous))
            previous = tick
        midi.tracks.append(track)
    source = tmp_path / "source.mid"
    midi.save(source)
    original = source.read_bytes()
    result = export_sheet_music_zip(
        source, tmp_path / "score.zip", quantize_grid="1/8", quantize_triplet=True
    )
    assert source.read_bytes() == original
    with zipfile.ZipFile(result.path) as archive:
        xml_path = tmp_path / "score.musicxml"
        xml_path.write_bytes(archive.read("score.musicxml"))
        exported = mido.MidiFile(file=io.BytesIO(archive.read("score.mid")))
        assert sum(m.type == "note_on" and m.velocity > 0 for t in exported.tracks for m in t) == 7
    assert _xml_attacks(xml_path) == collections.Counter(expected)
    xml = ET.parse(xml_path)
    meters = [
        (x.findtext("beats"), x.findtext("beat-type"))
        for x in xml.findall("./part/measure/attributes/time")
    ]
    assert set(meters) == {("6", "8"), tuple(map(str, next_meter))}
    tempos = {
        round(float(x.attrib["tempo"]), 3) for x in xml.findall(".//sound") if "tempo" in x.attrib
    }
    assert tempos == {125.0, 90.0}
    assert xml.findall('.//tie[@type="start"]')
    # Merely finding <sound tempo> is insufficient: MuseScore can ignore an
    # offset-only event when another tempo was already placed at that cursor.
    reopened = tmp_path / "reopened.mid"
    process = _run_musescore(
        get_musescore_executable(),
        ["-o", str(reopened), str(xml_path)],
        cancel_check=lambda: False,
    )
    assert process.returncode == 0, process.stderr
    midi = mido.MidiFile(reopened)
    tick, restored_tempos = 0, []
    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        if message.type == "set_tempo":
            restored_tempos.append((Fraction(tick, midi.ticks_per_beat), message.tempo))
    assert len(restored_tempos) == 2
    assert restored_tempos[0][0] == 0
    assert abs(restored_tempos[1][0] - Fraction(tempo_tick, 960)) <= Fraction(
        1, midi.ticks_per_beat
    )
    assert [round(mido.tempo2bpm(tempo), 3) for _, tempo in restored_tempos] == [125.0, 90.0]
