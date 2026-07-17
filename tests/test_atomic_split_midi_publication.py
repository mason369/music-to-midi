from pathlib import Path

import pytest
from mido import Message, MetaMessage, MidiFile, MidiTrack

from src.core.midi_generator import MidiGenerator
from src.core.pipeline import MusicToMidiPipeline
from src.models.data_models import Config, NoteEvent


def _write_note_midi(path: Path, *, note: int, track_name: str) -> None:
    midi = MidiFile(type=1, ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("track_name", name=track_name, time=0))
    track.append(Message("note_on", note=note, velocity=90, channel=0, time=0))
    track.append(Message("note_off", note=note, velocity=0, channel=0, time=240))
    track.append(MetaMessage("end_of_track", time=0))
    midi.save(str(path))


def _note_on_count(path: Path) -> int:
    midi = MidiFile(str(path))
    return sum(
        1
        for track in midi.tracks
        for message in track
        if not message.is_meta and message.type == "note_on" and message.velocity > 0
    )


def _assert_no_midi_temps(directory: Path) -> None:
    assert list(directory.glob(".*.tmp.mid")) == []


@pytest.mark.parametrize("track_mode", ["multi_track", "single_track"])
def test_precise_generator_atomically_publishes_parseable_midi(tmp_path, track_mode):
    output_path = tmp_path / f"{track_mode}_stem.mid"
    generator = MidiGenerator(Config(midi_track_mode=track_mode))

    result = generator.generate_from_precise_instruments_v2(
        instrument_notes={
            0: [NoteEvent(pitch=60, start_time=0.0, end_time=0.5, velocity=90)],
        },
        drum_notes={},
        tempo=120.0,
        output_path=str(output_path),
        quality="best",
    )

    assert result == str(output_path)
    assert _note_on_count(output_path) == 1
    _assert_no_midi_temps(tmp_path)


@pytest.mark.parametrize("track_mode", ["multi_track", "single_track"])
def test_precise_generator_validation_failure_preserves_existing_final(
    monkeypatch,
    tmp_path,
    track_mode,
):
    output_path = tmp_path / f"{track_mode}_stem.mid"
    _write_note_midi(output_path, note=48, track_name="existing")
    existing_bytes = output_path.read_bytes()
    generator = MidiGenerator(Config(midi_track_mode=track_mode))

    def save_invalid_midi(_self, filename=None, file=None):
        assert filename is not None and file is None
        Path(filename).write_bytes(b"not a standard MIDI file")

    monkeypatch.setattr(MidiFile, "save", save_invalid_midi)

    with pytest.raises(RuntimeError, match="created an invalid MIDI output"):
        generator.generate_from_precise_instruments_v2(
            instrument_notes={
                0: [NoteEvent(pitch=60, start_time=0.0, end_time=0.5, velocity=90)],
            },
            drum_notes={},
            tempo=120.0,
            output_path=str(output_path),
            quality="best",
        )

    assert output_path.read_bytes() == existing_bytes
    assert _note_on_count(output_path) == 1
    _assert_no_midi_temps(tmp_path)


@pytest.mark.parametrize("track_mode", ["multi_track", "single_track"])
def test_precise_generator_interruption_preserves_existing_final(
    monkeypatch,
    tmp_path,
    track_mode,
):
    output_path = tmp_path / f"{track_mode}_stem.mid"
    _write_note_midi(output_path, note=48, track_name="existing")
    existing_bytes = output_path.read_bytes()
    generator = MidiGenerator(Config(midi_track_mode=track_mode))

    def interrupt_save(_self, filename=None, file=None):
        assert filename is not None and file is None
        Path(filename).write_bytes(b"partial")
        raise InterruptedError("injected cancellation")

    monkeypatch.setattr(MidiFile, "save", interrupt_save)

    with pytest.raises(InterruptedError, match="injected cancellation"):
        generator.generate_from_precise_instruments_v2(
            instrument_notes={
                0: [NoteEvent(pitch=60, start_time=0.0, end_time=0.5, velocity=90)],
            },
            drum_notes={},
            tempo=120.0,
            output_path=str(output_path),
            quality="best",
        )

    assert output_path.read_bytes() == existing_bytes
    assert _note_on_count(output_path) == 1
    _assert_no_midi_temps(tmp_path)


def _pipeline_for_merge() -> MusicToMidiPipeline:
    pipeline = object.__new__(MusicToMidiPipeline)
    pipeline.config = Config()
    pipeline._cancelled = False
    return pipeline


def test_merge_stem_midis_atomically_publishes_parseable_midi(tmp_path):
    first_path = tmp_path / "vocal.mid"
    second_path = tmp_path / "accompaniment.mid"
    output_path = tmp_path / "merged.mid"
    _write_note_midi(first_path, note=60, track_name="vocal")
    _write_note_midi(second_path, note=64, track_name="accompaniment")
    pipeline = _pipeline_for_merge()

    result = pipeline._merge_stem_midis(
        {"vocal": str(first_path), "accompaniment": str(second_path)},
        str(output_path),
    )

    assert result == str(output_path)
    assert _note_on_count(output_path) == 2
    _assert_no_midi_temps(tmp_path)


def test_merge_stem_midis_validation_failure_preserves_existing_final(
    monkeypatch,
    tmp_path,
):
    source_path = tmp_path / "vocal.mid"
    output_path = tmp_path / "merged.mid"
    _write_note_midi(source_path, note=60, track_name="vocal")
    _write_note_midi(output_path, note=48, track_name="existing")
    existing_bytes = output_path.read_bytes()
    pipeline = _pipeline_for_merge()

    def save_invalid_midi(_self, filename=None, file=None):
        assert filename is not None and file is None
        Path(filename).write_bytes(b"not a standard MIDI file")

    monkeypatch.setattr(MidiFile, "save", save_invalid_midi)

    with pytest.raises(RuntimeError, match="created an invalid MIDI output"):
        pipeline._merge_stem_midis(
            {"vocal": str(source_path)},
            str(output_path),
        )

    assert output_path.read_bytes() == existing_bytes
    assert _note_on_count(output_path) == 1
    _assert_no_midi_temps(tmp_path)


def test_merge_stem_midis_cancellation_preserves_existing_final_and_cleans_temp(
    monkeypatch,
    tmp_path,
):
    source_path = tmp_path / "vocal.mid"
    output_path = tmp_path / "merged.mid"
    _write_note_midi(source_path, note=60, track_name="vocal")
    _write_note_midi(output_path, note=48, track_name="existing")
    existing_bytes = output_path.read_bytes()
    pipeline = _pipeline_for_merge()
    original_save = MidiFile.save

    def save_then_cancel(self, filename=None, file=None):
        original_save(self, filename=filename, file=file)
        pipeline._cancelled = True

    monkeypatch.setattr(MidiFile, "save", save_then_cancel)

    with pytest.raises(InterruptedError, match="用户取消了处理"):
        pipeline._merge_stem_midis(
            {"vocal": str(source_path)},
            str(output_path),
        )

    assert output_path.read_bytes() == existing_bytes
    assert _note_on_count(output_path) == 1
    _assert_no_midi_temps(tmp_path)


def test_merge_stem_midis_preserves_every_source_track_message(tmp_path):
    first_path = tmp_path / "vocal.mid"
    second_path = tmp_path / "accompaniment.mid"
    output_path = tmp_path / "merged.mid"

    first = MidiFile(type=1, ticks_per_beat=480)
    first_conductor = MidiTrack()
    first_conductor.append(MetaMessage("marker", text="official metadata", time=0))
    first.tracks.append(first_conductor)
    first_track = MidiTrack()
    first_track.append(MetaMessage("track_name", name="rare instrument", time=0))
    first_track.append(Message("program_change", program=73, channel=0, time=0))
    first_track.append(Message("control_change", control=11, value=88, channel=0, time=0))
    first_track.append(Message("pitchwheel", pitch=123, channel=0, time=0))
    first_track.append(Message("note_on", note=60, velocity=90, channel=0, time=0))
    first_track.append(Message("note_on", note=60, velocity=75, channel=0, time=0))
    first_track.append(Message("note_off", note=60, velocity=0, channel=0, time=1))
    first_track.append(Message("note_off", note=60, velocity=0, channel=0, time=0))
    first.tracks.append(first_track)
    first.save(str(first_path))

    second = MidiFile(type=1, ticks_per_beat=480)
    second_track = MidiTrack()
    second_track.append(MetaMessage("track_name", name="controllers", time=0))
    second_track.append(Message("program_change", program=52, channel=1, time=0))
    second_track.append(Message("control_change", control=64, value=127, channel=1, time=9))
    second_track.append(Message("control_change", control=64, value=0, channel=1, time=11))
    second.tracks.append(second_track)
    second.save(str(second_path))

    expected_tracks = [
        list(track)
        for midi_path in (first_path, second_path)
        for track in MidiFile(str(midi_path)).tracks
    ]

    pipeline = _pipeline_for_merge()
    pipeline._merge_stem_midis(
        {"vocal": str(first_path), "accompaniment": str(second_path)},
        str(output_path),
    )

    merged = MidiFile(str(output_path))
    assert merged.ticks_per_beat == 480
    assert [list(track) for track in merged.tracks] == expected_tracks


def test_merge_stem_midis_rejects_time_division_mismatch_without_overwriting(tmp_path):
    first_path = tmp_path / "first.mid"
    second_path = tmp_path / "second.mid"
    output_path = tmp_path / "merged.mid"
    _write_note_midi(first_path, note=60, track_name="first")
    _write_note_midi(output_path, note=48, track_name="existing")
    existing_bytes = output_path.read_bytes()

    second = MidiFile(type=1, ticks_per_beat=960)
    second_track = MidiTrack()
    second_track.append(Message("note_on", note=64, velocity=90, time=0))
    second_track.append(Message("note_off", note=64, velocity=0, time=480))
    second.tracks.append(second_track)
    second.save(str(second_path))

    pipeline = _pipeline_for_merge()
    with pytest.raises(RuntimeError, match="ticks_per_beat"):
        pipeline._merge_stem_midis(
            {"first": str(first_path), "second": str(second_path)},
            str(output_path),
        )

    assert output_path.read_bytes() == existing_bytes
    _assert_no_midi_temps(tmp_path)
