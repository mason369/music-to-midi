from __future__ import annotations

from pathlib import Path

import mido
import pytest

from src.core.midi_tempo import (
    non_tempo_event_tick_fingerprint,
    non_tempo_event_time_fingerprint,
    read_muscriptor_bar_offset_seconds,
    rewrite_midi_tempo_for_project_speed,
    rewrite_midi_tempo_preserving_seconds,
    rewrite_midi_tempo_preserving_ticks,
    write_midi_time_signature_preserving_ticks,
)


def _assert_fingerprints_preserve_payloads_and_seconds(
    source: mido.MidiFile,
    exported: mido.MidiFile,
) -> None:
    source_tracks, source_resolutions = non_tempo_event_time_fingerprint(source)
    exported_tracks, exported_resolutions = non_tempo_event_time_fingerprint(exported)
    assert len(exported_tracks) == len(source_tracks)
    for track_index, (source_events, exported_events) in enumerate(
        zip(source_tracks, exported_tracks)
    ):
        assert [payload for payload, _seconds in exported_events] == [
            payload for payload, _seconds in source_events
        ]
        tolerance = source_resolutions[track_index] + exported_resolutions[track_index] + 1e-9
        assert [seconds for _payload, seconds in exported_events] == pytest.approx(
            [seconds for _payload, seconds in source_events],
            abs=tolerance,
        )


def test_constant_project_bpm_preserves_multitrack_event_seconds_and_payloads(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "tempo-map-source.mid"
    destination = tmp_path / "project-77.9.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=960)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    conductor.append(mido.MetaMessage("marker", text="intro", time=111))
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(89.5), time=849))
    conductor.append(mido.MetaMessage("time_signature", numerator=7, denominator=8, time=73))
    midi.tracks.append(conductor)
    notes = mido.MidiTrack()
    notes.append(mido.MetaMessage("track_name", name="MuScriptor", time=0))
    notes.append(mido.Message("program_change", channel=3, program=24, time=0))
    notes.append(mido.Message("control_change", channel=3, control=64, value=127, time=37))
    notes.append(mido.Message("note_on", channel=3, note=64, velocity=91, time=19))
    notes.append(mido.Message("pitchwheel", channel=3, pitch=317, time=211))
    notes.append(mido.Message("note_off", channel=3, note=64, velocity=0, time=853))
    notes.append(mido.Message("control_change", channel=3, control=64, value=0, time=5))
    notes.append(mido.Message("sysex", data=(0x7D, 0x12, 0x34), time=17))
    midi.tracks.append(notes)
    midi.save(source_path)

    published = rewrite_midi_tempo_preserving_seconds(
        source_path,
        destination,
        77.9,
    )

    assert published == destination.resolve()
    source = mido.MidiFile(source_path)
    exported = mido.MidiFile(destination)
    tempos = [
        message.tempo
        for track in exported.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    ]
    assert tempos == [mido.bpm2tempo(77.9)]
    _assert_fingerprints_preserve_payloads_and_seconds(source, exported)
    assert exported.length == pytest.approx(
        source.length,
        abs=60.0 / 77.9 / exported.ticks_per_beat,
    )


def test_type_two_tracks_receive_project_bpm_and_keep_independent_seconds(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "type-two-source.mid"
    destination = tmp_path / "type-two-project.mid"
    midi = mido.MidiFile(type=2, ticks_per_beat=480)
    first = mido.MidiTrack()
    first.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(100), time=0))
    first.append(mido.Message("note_on", note=60, velocity=90, time=37))
    first.append(mido.Message("note_off", note=60, velocity=0, time=480))
    midi.tracks.append(first)
    second = mido.MidiTrack()
    second.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(150), time=0))
    second.append(mido.Message("note_on", note=67, velocity=70, time=113))
    second.append(mido.Message("note_off", note=67, velocity=0, time=211))
    midi.tracks.append(second)
    midi.save(source_path)

    rewrite_midi_tempo_preserving_seconds(source_path, destination, 132.5)

    source = mido.MidiFile(source_path)
    exported = mido.MidiFile(destination)
    assert [
        mido.tempo2bpm(message.tempo)
        for track in exported.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    ] == pytest.approx([132.5, 132.5], abs=0.001)
    _assert_fingerprints_preserve_payloads_and_seconds(source, exported)


@pytest.mark.parametrize("invalid_bpm", [0.0, 1000.0, float("nan"), float("inf")])
def test_invalid_project_bpm_does_not_touch_existing_destination(
    tmp_path: Path,
    invalid_bpm: float,
) -> None:
    source = tmp_path / "valid-source.mid"
    destination = tmp_path / "existing.mid"
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage("end_of_track", time=0)]))
    midi.save(source)
    sentinel = b"existing-user-file"
    destination.write_bytes(sentinel)

    with pytest.raises(ValueError, match="Invalid MIDI BPM"):
        rewrite_midi_tempo_preserving_seconds(source, destination, invalid_bpm)

    assert destination.read_bytes() == sentinel
    assert list(tmp_path.glob(".*.tempo-preserving.*.tmp.mid")) == []


def test_tick_preserving_rewrite_changes_duration_without_moving_events(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "reference-100.mid"
    destination = tmp_path / "project-160.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(100), time=0))
    track.append(mido.MetaMessage("marker", text="off-grid", time=37))
    track.append(mido.Message("program_change", channel=2, program=24, time=0))
    track.append(mido.Message("note_on", channel=2, note=67, velocity=83, time=16))
    track.append(mido.Message("control_change", channel=2, control=64, value=127, time=211))
    track.append(mido.Message("note_off", channel=2, note=67, velocity=0, time=313))
    midi.tracks.append(track)
    midi.save(source_path)

    rewrite_midi_tempo_preserving_ticks(source_path, destination, 160)

    source = mido.MidiFile(source_path)
    exported = mido.MidiFile(destination)
    assert non_tempo_event_tick_fingerprint(exported) == non_tempo_event_tick_fingerprint(source)
    assert exported.length == pytest.approx(
        source.length * 100 / 160,
        abs=60 / 160 / exported.ticks_per_beat,
    )


def test_tick_preserving_project_speed_scales_muscriptor_bar_offset_marker(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "bar-aligned-120.mid"
    destination = tmp_path / "bar-aligned-60.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    conductor.append(
        mido.MetaMessage("marker", text="muscriptor:bar_offset=0.2500", time=0)
    )
    midi.tracks.append(conductor)
    notes = mido.MidiTrack()
    notes.append(mido.Message("note_on", note=60, velocity=100, time=240))
    notes.append(mido.Message("note_off", note=60, velocity=0, time=480))
    midi.tracks.append(notes)
    midi.save(source_path)

    rewrite_midi_tempo_preserving_ticks(source_path, destination, 60.0)

    exported = mido.MidiFile(destination)
    assert read_muscriptor_bar_offset_seconds(exported) == pytest.approx(0.5)
    assert [
        message.tempo
        for track in exported.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    ] == [mido.bpm2tempo(60.0)]

    def note_ticks(path: Path) -> list[tuple[str, int, int]]:
        result: list[tuple[str, int, int]] = []
        for track in mido.MidiFile(path).tracks:
            absolute_tick = 0
            for message in track:
                absolute_tick += int(message.time)
                if not message.is_meta and message.type in {"note_on", "note_off"}:
                    result.append((message.type, int(message.note), absolute_tick))
        return result

    assert note_ticks(destination) == note_ticks(source_path)
    assert exported.length == pytest.approx(
        mido.MidiFile(source_path).length * 2.0,
        abs=60 / 60 / exported.ticks_per_beat,
    )


def test_second_preserving_rewrite_keeps_muscriptor_bar_offset_marker(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "bar-offset-source.mid"
    destination = tmp_path / "bar-offset-seconds.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.MetaMessage("marker", text="muscriptor:bar_offset=0.3750", time=0))
    track.append(mido.Message("note_on", note=64, velocity=90, time=120))
    track.append(mido.Message("note_off", note=64, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(source_path)

    rewrite_midi_tempo_preserving_seconds(source_path, destination, 77.9)

    assert read_muscriptor_bar_offset_seconds(destination) == pytest.approx(0.375)


def test_conflicting_muscriptor_bar_offset_markers_fail_without_publication(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "conflicting-offsets.mid"
    destination = tmp_path / "existing.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    conductor.append(mido.MetaMessage("marker", text="muscriptor:bar_offset=0.1000", time=0))
    midi.tracks.append(conductor)
    second = mido.MidiTrack()
    second.append(mido.MetaMessage("marker", text="muscriptor:bar_offset=0.2000", time=0))
    midi.tracks.append(second)
    midi.save(source_path)
    sentinel = b"existing-user-file"
    destination.write_bytes(sentinel)

    with pytest.raises(RuntimeError, match="Conflicting MuScriptor bar-offset"):
        rewrite_midi_tempo_preserving_ticks(source_path, destination, 90.0)

    assert destination.read_bytes() == sentinel


def test_project_speed_rewrite_maps_seconds_to_reference_ticks_then_keeps_them(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "backend-placeholder.mid"
    reference_path = tmp_path / "reference-grid.mid"
    destination = tmp_path / "project-speed.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=960)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.Message("note_on", channel=0, note=64, velocity=91, time=73))
    track.append(mido.Message("pitchwheel", channel=0, pitch=317, time=211))
    track.append(mido.Message("note_off", channel=0, note=64, velocity=0, time=853))
    midi.tracks.append(track)
    midi.save(source_path)

    rewrite_midi_tempo_preserving_seconds(source_path, reference_path, 76.3)
    rewrite_midi_tempo_for_project_speed(source_path, destination, 76.3, 120)

    reference = mido.MidiFile(reference_path)
    exported = mido.MidiFile(destination)
    assert non_tempo_event_tick_fingerprint(exported) == non_tempo_event_tick_fingerprint(reference)
    assert exported.length == pytest.approx(
        reference.length * 76.3 / 120,
        abs=60 / 120 / exported.ticks_per_beat,
    )


def test_conductor_meter_rewrite_preserves_every_non_meter_event_tick(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "meter-source.mid"
    destination = tmp_path / "meter-6-8.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(77.9), time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    conductor.append(mido.MetaMessage("marker", text="keep", time=37))
    midi.tracks.append(conductor)
    notes = mido.MidiTrack()
    notes.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    notes.append(mido.Message("program_change", channel=2, program=24, time=0))
    notes.append(mido.Message("note_on", channel=2, note=64, velocity=91, time=53))
    notes.append(mido.Message("control_change", channel=2, control=64, value=127, time=211))
    notes.append(mido.Message("note_off", channel=2, note=64, velocity=0, time=853))
    midi.tracks.append(notes)
    midi.save(source_path)

    def non_meter_events(path: Path) -> list[list[tuple[bytes, int]]]:
        result: list[list[tuple[bytes, int]]] = []
        for track in mido.MidiFile(path).tracks:
            absolute_tick = 0
            events: list[tuple[bytes, int]] = []
            for message in track:
                absolute_tick += int(message.time)
                if message.is_meta and message.type in {"time_signature", "end_of_track"}:
                    continue
                payload = bytes(message.bytes()) if message.is_meta else bytes(message.bin())
                events.append((payload, absolute_tick))
            result.append(events)
        return result

    before = non_meter_events(source_path)
    write_midi_time_signature_preserving_ticks(
        source_path,
        destination,
        (6, 8),
    )
    exported = mido.MidiFile(destination)
    signatures = []
    for track_index, track in enumerate(exported.tracks):
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            if message.is_meta and message.type == "time_signature":
                signatures.append(
                    (
                        track_index,
                        absolute_tick,
                        message.numerator,
                        message.denominator,
                        message.clocks_per_click,
                    )
                )

    assert signatures == [(0, 0, 6, 8, 36)]
    assert non_meter_events(destination) == before
