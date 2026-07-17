from pathlib import Path

import mido
import pytest

from src.core.pipeline import MusicToMidiPipeline


def _write_semantic_midi(path: Path, *, include_tempo: bool = False) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    if include_tempo:
        conductor.append(mido.MetaMessage("set_tempo", tempo=600_000, time=0))
    conductor.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(conductor)

    track = mido.MidiTrack()
    track.extend(
        [
            mido.MetaMessage("track_name", name="rare official track", time=0),
            mido.Message("program_change", program=73, channel=0, time=0),
            mido.Message("control_change", control=11, value=87, channel=0, time=12),
            mido.Message("pitchwheel", pitch=321, channel=0, time=3),
            mido.Message("note_on", note=60, velocity=91, channel=0, time=1),
            mido.Message("note_off", note=60, velocity=0, channel=0, time=1),
            mido.Message("note_on", note=60, velocity=74, channel=0, time=0),
            mido.Message("note_off", note=60, velocity=0, channel=0, time=1),
            mido.MetaMessage("end_of_track", time=0),
        ]
    )
    midi.tracks.append(track)
    midi.save(path)


def test_telknet_tempo_alignment_preserves_every_non_tempo_message_and_seconds(tmp_path):
    midi_path = tmp_path / "backend.mid"
    _write_semantic_midi(midi_path)
    source = mido.MidiFile(midi_path)

    result = MusicToMidiPipeline._normalize_midi_tempo_metadata(
        str(midi_path),
        90.0,
    )

    assert result == str(midi_path.resolve())
    normalized = mido.MidiFile(midi_path)
    tempo_messages = [
        message
        for track in normalized.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    ]
    assert tempo_messages == [mido.MetaMessage("set_tempo", tempo=666_667, time=0)]
    assert len(normalized.tracks) == len(source.tracks)

    source_tempo = 500_000
    target_tempo = tempo_messages[0].tempo
    tolerance = mido.tick2second(1, normalized.ticks_per_beat, target_tempo)
    for source_track, target_track in zip(source.tracks, normalized.tracks):
        target_messages = [
            message
            for message in target_track
            if not (message.is_meta and message.type == "set_tempo")
        ]
        assert len(target_messages) == len(source_track)

        source_tick = 0
        target_tick = 0
        for source_message, target_message in zip(source_track, target_messages):
            source_tick += source_message.time
            target_tick += target_message.time
            assert source_message.copy(time=0) == target_message.copy(time=0)
            source_seconds = mido.tick2second(
                source_tick,
                source.ticks_per_beat,
                source_tempo,
            )
            target_seconds = mido.tick2second(
                target_tick,
                normalized.ticks_per_beat,
                target_tempo,
            )
            assert target_seconds == pytest.approx(source_seconds, abs=tolerance)


def test_telknet_tempo_alignment_leaves_existing_tempo_midi_byte_identical(tmp_path):
    midi_path = tmp_path / "already-tempo.mid"
    _write_semantic_midi(midi_path, include_tempo=True)
    original_bytes = midi_path.read_bytes()

    MusicToMidiPipeline._normalize_midi_tempo_metadata(str(midi_path), 90.0)

    assert midi_path.read_bytes() == original_bytes


@pytest.mark.parametrize("tempo", [0.0, -1.0, float("nan"), float("inf")])
def test_tempo_alignment_rejects_invalid_detected_bpm_without_overwriting(tmp_path, tempo):
    midi_path = tmp_path / "invalid-tempo.mid"
    _write_semantic_midi(midi_path)
    original_bytes = midi_path.read_bytes()

    with pytest.raises(RuntimeError, match="无效 MIDI 速度"):
        MusicToMidiPipeline._normalize_midi_tempo_metadata(str(midi_path), tempo)

    assert midi_path.read_bytes() == original_bytes
