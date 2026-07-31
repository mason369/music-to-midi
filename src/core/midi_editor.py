"""Validated publication of user-edited piano-roll notes.

Notes, tempo, and General MIDI program selection are editor-owned. Every other
event is passed through only when it already exists in the backend MIDI; the
editor never invents pedal, controller, marker, SysEx, or expression data.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import mido

from src.core.muscriptor_result_assets import MuscriptorRollNote
from src.models.data_models import MAX_MIDI_BPM, MIN_MIDI_BPM
from src.utils.midi_output import (
    publish_midi_output,
    remove_temporary_midi,
    unique_midi_temp_path,
)

_MIN_NOTE_SECONDS = 0.01


def _validated_bpm(value: float, label: str) -> float:
    bpm = float(value)
    if not math.isfinite(bpm) or not MIN_MIDI_BPM <= bpm <= MAX_MIDI_BPM:
        raise ValueError(f"Invalid {label} BPM for edited MIDI export: {value!r}")
    return bpm


def validate_editable_notes(
    notes: Iterable[MuscriptorRollNote],
    *,
    track_count: int,
) -> tuple[MuscriptorRollNote, ...]:
    """Return an immutable, fully validated editor note snapshot."""

    if track_count <= 0:
        raise ValueError(f"Edited MIDI source must have at least one track: {track_count}")
    normalized = tuple(notes)
    for index, note in enumerate(normalized):
        if (
            not note.instrument
            or not 0 <= int(note.pitch) <= 127
            or not 1 <= int(note.velocity) <= 127
            or not math.isfinite(float(note.start))
            or not math.isfinite(float(note.end))
            or float(note.start) < 0
            or float(note.end) - float(note.start) < _MIN_NOTE_SECONDS - 1e-9
            or not 0 <= int(note.program) <= 127
            or not 0 <= int(note.track_index) < track_count
            or not 0 <= int(note.channel) <= 15
            or (bool(note.is_drum) and int(note.channel) != 9)
            or (not bool(note.is_drum) and int(note.channel) == 9)
        ):
            raise ValueError(f"Invalid edited MIDI note at index {index}: {note!r}")
    return normalized


def _message_bytes(message: mido.Message | mido.MetaMessage) -> bytes:
    if message.is_meta:
        return bytes(message.bytes())
    return bytes(message.bin())


def _retained_event_fingerprint(
    midi: mido.MidiFile,
) -> list[list[tuple[bytes, int]]]:
    """Capture retained event payloads at their exact musical ticks."""

    retained: list[list[tuple[bytes, int]]] = []
    for track in midi.tracks:
        absolute_tick = 0
        track_events: list[tuple[bytes, int]] = []
        for message in track:
            absolute_tick += int(message.time)
            if message.type in {
                "note_on",
                "note_off",
                "program_change",
                "set_tempo",
                "end_of_track",
            }:
                continue
            track_events.append((_message_bytes(message), absolute_tick))
        retained.append(track_events)
    return retained


def _read_note_tick_fingerprint(
    midi: mido.MidiFile,
) -> Counter[tuple[int, int, int, int, int, int, int, bool]]:
    notes: Counter[tuple[int, int, int, int, int, int, int, bool]] = Counter()
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        programs = {channel: 0 for channel in range(16)}
        active: dict[tuple[int, int], list[tuple[int, int, int, bool]]] = defaultdict(list)
        for message in track:
            absolute_tick += int(message.time)
            if message.is_meta:
                continue
            if message.type == "program_change":
                programs[int(message.channel)] = int(message.program)
                continue
            if message.type not in {"note_on", "note_off"}:
                continue
            channel = int(message.channel)
            pitch = int(message.note)
            key = (channel, pitch)
            if message.type == "note_on" and int(message.velocity) > 0:
                active[key].append(
                    (
                        absolute_tick,
                        int(message.velocity),
                        programs[channel],
                        channel == 9,
                    )
                )
                continue
            if not active[key]:
                raise RuntimeError(
                    "Edited MIDI verification found an unmatched note-off: "
                    f"track={track_index}, channel={channel}, pitch={pitch}, tick={absolute_tick}"
                )
            start_tick, velocity, program, is_drum = active[key].pop(0)
            notes[
                (
                    track_index,
                    channel,
                    pitch,
                    velocity,
                    start_tick,
                    absolute_tick,
                    program,
                    is_drum,
                )
            ] += 1
        dangling = [key for key, values in active.items() if values]
        if dangling:
            raise RuntimeError(
                "Edited MIDI verification found unterminated notes: "
                f"track={track_index}, notes={dangling[:8]!r}"
            )
    return notes


def _note_tick_bounds(
    note: MuscriptorRollNote,
    *,
    ticks_per_beat: int,
    timeline_bpm: float,
) -> tuple[int, int]:
    ticks_per_second = ticks_per_beat * timeline_bpm / 60.0
    start_tick = max(0, round(float(note.start) * ticks_per_second))
    duration_ticks = max(1, round((float(note.end) - float(note.start)) * ticks_per_second))
    return start_tick, start_tick + duration_ticks


def _assign_export_channels(
    source_midi: mido.MidiFile,
    prepared_notes: list[tuple[int, MuscriptorRollNote, int, int, int, int]],
) -> tuple[
    list[tuple[int, MuscriptorRollNote, int, int, int, int, int]],
    dict[int, tuple[int, ...]],
]:
    """Use spare melodic channels when same-pitch notes overlap.

    MIDI note-off messages identify only channel and pitch. Two nested notes on
    the same channel/pitch therefore cannot retain their individual durations.
    A dedicated spare channel makes each overlapping voice unambiguous without
    moving, resizing, filtering, or otherwise changing the note.
    """

    reserved_channels = {9}
    for track in source_midi.tracks:
        for message in track:
            if not message.is_meta and hasattr(message, "channel"):
                reserved_channels.add(int(message.channel))
    reserved_channels.update(item[5] for item in prepared_notes)
    spare_channels = [channel for channel in range(16) if channel not in reserved_channels]

    assigned_by_index: dict[int, int] = {}
    auxiliary_by_source: dict[int, set[int]] = defaultdict(set)
    logical_lanes: dict[
        tuple[int, int],
        list[tuple[int, MuscriptorRollNote, int, int, int, int]],
    ] = defaultdict(list)
    for prepared in prepared_notes:
        logical_lanes[(prepared[4], prepared[5])].append(prepared)

    for (track_index, source_channel), lane_notes in logical_lanes.items():
        lane_notes.sort(key=lambda item: (item[2], item[3], item[0]))
        voice_channels = [source_channel]
        pitch_end_by_voice: list[dict[int, int]] = [dict()]
        for prepared in lane_notes:
            note_index, note, start_tick, end_tick, _track_index, _channel = prepared
            pitch = int(note.pitch)
            selected_voice = next(
                (
                    voice_index
                    for voice_index, pitch_ends in enumerate(pitch_end_by_voice)
                    if pitch_ends.get(pitch, -1) <= start_tick
                ),
                None,
            )
            if selected_voice is None:
                if note.is_drum:
                    raise ValueError(
                        "Edited MIDI cannot losslessly encode overlapping drum hits "
                        "with the same track/channel/pitch: "
                        f"track={track_index}, channel={source_channel}, "
                        f"pitch={pitch}, tick={start_tick}"
                    )
                if not spare_channels:
                    raise ValueError(
                        "Edited MIDI needs another melodic channel to preserve "
                        "overlapping same-pitch notes, but all 15 melodic channels "
                        "are already in use: "
                        f"track={track_index}, channel={source_channel}, "
                        f"pitch={pitch}, tick={start_tick}"
                    )
                auxiliary_channel = spare_channels.pop(0)
                voice_channels.append(auxiliary_channel)
                pitch_end_by_voice.append({})
                auxiliary_by_source[source_channel].add(auxiliary_channel)
                selected_voice = len(voice_channels) - 1
            assigned_channel = voice_channels[selected_voice]
            pitch_end_by_voice[selected_voice][pitch] = end_tick
            assigned_by_index[note_index] = assigned_channel

    assigned_notes = [(*prepared, assigned_by_index[prepared[0]]) for prepared in prepared_notes]
    return assigned_notes, {
        source_channel: tuple(sorted(channels))
        for source_channel, channels in auxiliary_by_source.items()
    }


def _note_fingerprint_delta(
    expected: Counter[tuple[int, int, int, int, int, int, int, bool]],
    actual: Counter[tuple[int, int, int, int, int, int, int, bool]],
) -> str:
    """Return bounded, actionable diagnostics instead of dumping every note."""

    missing = expected - actual
    extra = actual - expected

    def sample(
        values: Counter[tuple[int, int, int, int, int, int, int, bool]],
    ) -> list[tuple[tuple[int, int, int, int, int, int, int, bool], int]]:
        return list(values.items())[:4]

    return (
        f"expected_total={expected.total()}, actual_total={actual.total()}, "
        f"missing_total={missing.total()}, extra_total={extra.total()}, "
        f"missing_sample={sample(missing)!r}, extra_sample={sample(extra)!r}"
    )


def export_edited_midi(
    source_path: str | Path,
    destination_path: str | Path,
    notes: Iterable[MuscriptorRollNote],
    *,
    reference_bpm: float,
    target_bpm: float,
) -> Path:
    """Publish source-timeline edits at a project BPM that truly changes speed.

    Editor notes and retained source events are expressed on the detected/
    reference BPM tick grid. The requested target tempo is then written without
    moving those musical ticks, so playback duration scales by
    ``reference_bpm / target_bpm``. No note is snapped, filtered, or resized.
    """

    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    source_midi = mido.MidiFile(str(source))
    if not source_midi.tracks:
        raise RuntimeError(f"Edited MIDI source has no tracks: {source}")
    if source_midi.ticks_per_beat <= 0:
        raise RuntimeError(
            "SMPTE time-division MIDI is not supported by the piano-roll editor: "
            f"path={source}, division={source_midi.ticks_per_beat}"
        )
    reference = _validated_bpm(reference_bpm, "reference")
    export_bpm = _validated_bpm(target_bpm, "target")
    edited_notes = validate_editable_notes(notes, track_count=len(source_midi.tracks))
    tempo_us = mido.bpm2tempo(export_bpm)

    output = mido.MidiFile(
        type=source_midi.type,
        ticks_per_beat=source_midi.ticks_per_beat,
    )
    track_events: list[list[tuple[int, int, int, mido.Message | mido.MetaMessage]]] = [
        [] for _track in source_midi.tracks
    ]
    sequence = 0
    prepared_notes: list[tuple[int, MuscriptorRollNote, int, int, int, int]] = []
    for note_index, note in enumerate(edited_notes):
        start_tick, end_tick = _note_tick_bounds(
            note,
            ticks_per_beat=source_midi.ticks_per_beat,
            timeline_bpm=reference,
        )
        track_index = int(note.track_index)
        channel = 9 if note.is_drum else int(note.channel)
        prepared_notes.append((note_index, note, start_tick, end_tick, track_index, channel))

    assigned_notes, auxiliary_by_source = _assign_export_channels(
        source_midi,
        prepared_notes,
    )
    expected_retained: list[list[tuple[bytes, int]]] = [[] for _track in source_midi.tracks]
    for track_index, source_track in enumerate(source_midi.tracks):
        absolute_tick = 0
        for message in source_track:
            absolute_tick += int(message.time)
            if message.type in {
                "note_on",
                "note_off",
                "program_change",
                "set_tempo",
                "end_of_track",
            }:
                continue
            # The pipeline has already mapped every backend event onto the
            # detected/reference musical tick grid. Retained controller,
            # marker, pedal and SysEx events must therefore keep that exact
            # tick; converting through the source file's target-tempo seconds
            # here would apply the BPM ratio a second time.
            target_tick = absolute_tick
            retained_copies = [message.copy(time=0)]
            if not message.is_meta and hasattr(message, "channel"):
                retained_copies.extend(
                    message.copy(channel=channel, time=0)
                    for channel in auxiliary_by_source.get(int(message.channel), ())
                )
            for retained in retained_copies:
                sequence += 1
                track_events[track_index].append((target_tick, 0, sequence, retained))
                expected_retained[track_index].append((_message_bytes(retained), target_tick))

    tempo_tracks = range(len(source_midi.tracks)) if source_midi.type == 2 else (0,)
    for track_index in tempo_tracks:
        track_events[track_index].append(
            (0, -100, 0, mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        )

    expected_notes: Counter[tuple[int, int, int, int, int, int, int, bool]] = Counter()
    for (
        note_index,
        note,
        start_tick,
        end_tick,
        track_index,
        _source_channel,
        channel,
    ) in assigned_notes:
        track_events[track_index].append(
            (
                end_tick,
                10,
                note_index,
                mido.Message(
                    "note_off",
                    channel=channel,
                    note=int(note.pitch),
                    velocity=0,
                    time=0,
                ),
            )
        )
        track_events[track_index].append(
            (
                start_tick,
                30,
                note_index,
                mido.Message(
                    "note_on",
                    channel=channel,
                    note=int(note.pitch),
                    velocity=int(note.velocity),
                    time=0,
                ),
            )
        )
        expected_notes[
            (
                track_index,
                channel,
                int(note.pitch),
                int(note.velocity),
                start_tick,
                end_tick,
                int(note.program),
                bool(note.is_drum),
            )
        ] += 1

    melodic_lanes: dict[
        tuple[int, int],
        list[tuple[int, MuscriptorRollNote, int, int, int, int, int]],
    ] = defaultdict(list)
    for prepared in assigned_notes:
        if not prepared[1].is_drum:
            melodic_lanes[(prepared[4], prepared[6])].append(prepared)

    for (track_index, channel), lane_notes in melodic_lanes.items():
        lane_notes.sort(key=lambda item: (item[2], item[3], item[0]))
        active: list[tuple[int, int]] = []
        current_program: int | None = None
        cursor = 0
        while cursor < len(lane_notes):
            start_tick = lane_notes[cursor][2]
            active = [(end_tick, program) for end_tick, program in active if end_tick > start_tick]
            group_end = cursor
            starting: list[tuple[int, MuscriptorRollNote, int, int, int, int, int]] = []
            while group_end < len(lane_notes) and lane_notes[group_end][2] == start_tick:
                starting.append(lane_notes[group_end])
                group_end += 1
            programs = {program for _end_tick, program in active}
            programs.update(int(item[1].program) for item in starting)
            if len(programs) != 1:
                raise ValueError(
                    "Edited MIDI cannot assign overlapping instruments to one "
                    f"track/channel: track={track_index}, channel={channel}, "
                    f"tick={start_tick}, programs={sorted(programs)!r}"
                )
            desired_program = next(iter(programs))
            if current_program != desired_program:
                sequence += 1
                track_events[track_index].append(
                    (
                        0 if current_program is None else start_tick,
                        20,
                        sequence,
                        mido.Message(
                            "program_change",
                            channel=channel,
                            program=desired_program,
                            time=0,
                        ),
                    )
                )
                current_program = desired_program
            active.extend((item[3], int(item[1].program)) for item in starting)
            cursor = group_end

    for events in track_events:
        events.sort(key=lambda event: (event[0], event[1], event[2]))
        target_track = mido.MidiTrack()
        previous_tick = 0
        for absolute_tick, _priority, _sequence, message in events:
            target_track.append(message.copy(time=absolute_tick - previous_tick))
            previous_tick = absolute_tick
        target_track.append(mido.MetaMessage("end_of_track", time=0))
        output.tracks.append(target_track)

    temporary = unique_midi_temp_path(destination, "edited-result")
    try:
        output.save(str(temporary))
        verified = mido.MidiFile(str(temporary))
        published_tempos = [
            int(message.tempo)
            for track in verified.tracks
            for message in track
            if message.type == "set_tempo"
        ]
        expected_tempo_count = len(verified.tracks) if verified.type == 2 else 1
        if len(published_tempos) != expected_tempo_count or any(
            tempo != tempo_us for tempo in published_tempos
        ):
            raise RuntimeError(
                "Edited MIDI tempo verification failed: "
                f"path={temporary}, expected={tempo_us}, actual={published_tempos!r}"
            )
        actual_notes = _read_note_tick_fingerprint(verified)
        if actual_notes != expected_notes:
            raise RuntimeError(
                "Edited MIDI note verification failed: "
                f"path={temporary}, {_note_fingerprint_delta(expected_notes, actual_notes)}"
            )
        published_events = _retained_event_fingerprint(verified)
        if len(expected_retained) != len(published_events):
            raise RuntimeError(
                "Edited MIDI pass-through verification changed track count: " f"path={temporary}"
            )
        for track_index, (expected_track_events, published_track_events) in enumerate(
            zip(expected_retained, published_events)
        ):
            if len(expected_track_events) != len(published_track_events):
                raise RuntimeError(
                    "Edited MIDI pass-through verification changed event count: "
                    f"path={temporary}, track={track_index}, "
                    f"expected={len(expected_track_events)}, "
                    f"actual={len(published_track_events)}"
                )
            for event_index, (expected_event, published_event) in enumerate(
                zip(expected_track_events, published_track_events)
            ):
                if expected_event != published_event:
                    raise RuntimeError(
                        "Edited MIDI pass-through event verification failed: "
                        f"path={temporary}, track={track_index}, "
                        f"event={event_index}, expected={expected_event!r}, "
                        f"actual={published_event!r}"
                    )
        expected_bytes = temporary.read_bytes()
        published = Path(
            publish_midi_output(
                temporary,
                destination,
                "Edited MIDI export",
            )
        )
        if published.read_bytes() != expected_bytes:
            raise RuntimeError(
                "Edited MIDI atomic publication changed the verified file: " f"path={published}"
            )
        return published
    finally:
        remove_temporary_midi(temporary)


__all__ = ["export_edited_midi", "validate_editable_notes"]
