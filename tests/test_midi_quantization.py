from __future__ import annotations

from pathlib import Path

import mido
import pytest

from src.core.midi_quantization import (
    DEFAULT_MIDI_QUANTIZE_GRID,
    DEFAULT_MIDI_QUANTIZE_SCOPE,
    MIDI_QUANTIZE_GRIDS,
    MIDI_QUANTIZE_SCOPES,
    quantize_grid_ticks,
    quantize_midi_notes,
)


def _absolute_messages(path: Path):
    midi = mido.MidiFile(path)
    messages = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            messages.append((track_index, absolute_tick, message))
    return midi, messages


def test_supported_quantization_grids_have_one_shared_default():
    assert MIDI_QUANTIZE_GRIDS == ("1/4", "1/8", "1/16", "1/32", "1/64")
    assert DEFAULT_MIDI_QUANTIZE_GRID == "1/32"
    assert MIDI_QUANTIZE_SCOPES == ("all_tracks", "selected_notes")
    assert DEFAULT_MIDI_QUANTIZE_SCOPE == "all_tracks"
    assert [quantize_grid_ticks(480, grid) for grid in MIDI_QUANTIZE_GRIDS] == [
        480,
        240,
        120,
        60,
        30,
    ]


def test_quantize_midi_notes_snaps_paired_starts_and_durations_only(tmp_path: Path):
    path = tmp_path / "source.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=17))
    midi.tracks.append(conductor)
    notes = mido.MidiTrack()
    notes.append(mido.Message("program_change", channel=2, program=24, time=11))
    notes.append(mido.Message("control_change", channel=2, control=64, value=127, time=6))
    notes.append(mido.Message("note_on", channel=2, note=60, velocity=91, time=44))
    notes.append(mido.Message("control_change", channel=2, control=1, value=33, time=20))
    notes.append(mido.Message("note_off", channel=2, note=60, velocity=0, time=158))
    notes.append(mido.Message("note_on", channel=2, note=64, velocity=79, time=62))
    notes.append(mido.Message("note_off", channel=2, note=64, velocity=0, time=91))
    midi.tracks.append(notes)
    midi.save(path)

    report = quantize_midi_notes(path, "1/16")

    assert report.path == path.resolve()
    assert report.grid == "1/16"
    assert report.grid_ticks == 120
    assert report.paired_note_count == 2
    published, messages = _absolute_messages(path)
    assert published.type == 1
    assert published.ticks_per_beat == 480
    assert [
        (track_index, tick, message.type)
        for track_index, tick, message in messages
        if message.type in {"set_tempo", "time_signature", "program_change", "control_change"}
    ] == [
        (0, 0, "set_tempo"),
        (0, 17, "time_signature"),
        (1, 11, "program_change"),
        (1, 17, "control_change"),
        (1, 81, "control_change"),
    ]
    note_ticks = [
        (tick, message.type, int(message.note), int(message.velocity))
        for track_index, tick, message in messages
        if track_index == 1 and message.type in {"note_on", "note_off"}
    ]
    assert note_ticks == [
        (120, "note_on", 60, 91),
        (240, "note_off", 60, 0),
        (360, "note_on", 64, 79),
        (480, "note_off", 64, 0),
    ]


@pytest.mark.parametrize("grid", ["1/3", "1/128", "", "32"])
def test_invalid_quantization_grid_fails_without_changing_file(tmp_path: Path, grid: str):
    path = tmp_path / "invalid-grid.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=120))
    midi.tracks.append(track)
    midi.save(path)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="Unsupported MIDI quantization grid"):
        quantize_midi_notes(path, grid)

    assert path.read_bytes() == before


def test_explicit_quantization_without_a_complete_note_pair_fails(tmp_path: Path):
    path = tmp_path / "unpaired.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=90, time=7))
    midi.tracks.append(track)
    midi.save(path)
    before = path.read_bytes()

    with pytest.raises(RuntimeError, match="no complete note pair"):
        quantize_midi_notes(path, "1/32")

    assert path.read_bytes() == before
