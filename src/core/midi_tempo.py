"""Strict MIDI tempo rewrites for both absolute-time and project-speed exports."""

from __future__ import annotations

from bisect import bisect_right
from math import isfinite
from pathlib import Path
from typing import Callable

import mido

from src.models.data_models import MAX_MIDI_BPM, MIN_MIDI_BPM
from src.utils.midi_output import (
    publish_midi_output,
    remove_temporary_midi,
    unique_midi_temp_path,
)

_DEFAULT_TEMPO_US = 500_000
_MUSCRIPTOR_BAR_OFFSET_PREFIX = "muscriptor:bar_offset="


def _bar_offset_value(message: mido.MetaMessage) -> float | None:
    if not (message.is_meta and message.type == "marker"):
        return None
    text = str(message.text)
    if not text.startswith(_MUSCRIPTOR_BAR_OFFSET_PREFIX):
        return None
    raw = text.removeprefix(_MUSCRIPTOR_BAR_OFFSET_PREFIX)
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid MuScriptor bar-offset marker: {text!r}") from exc
    if not isfinite(value) or value < 0.0:
        raise RuntimeError(f"Invalid MuScriptor bar-offset marker: {text!r}")
    return value


def read_muscriptor_bar_offset_seconds(source: str | Path | mido.MidiFile) -> float:
    """Read one strict e2bd0fc bar-alignment offset, or zero when absent."""

    midi = source if isinstance(source, mido.MidiFile) else mido.MidiFile(str(source))
    values = [
        value
        for track in midi.tracks
        for message in track
        if (value := _bar_offset_value(message)) is not None
    ]
    if not values:
        return 0.0
    if any(abs(value - values[0]) > 0.0001 for value in values[1:]):
        raise RuntimeError(f"Conflicting MuScriptor bar-offset markers: {values!r}")
    return float(values[0])


def _constant_source_bpm_for_bar_offset(midi: mido.MidiFile) -> float:
    tempos = {
        int(message.tempo)
        for track in midi.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    }
    if not tempos:
        tempos = {_DEFAULT_TEMPO_US}
    if len(tempos) != 1:
        raise RuntimeError(
            "MuScriptor bar alignment requires one constant source tempo before "
            f"project-speed rewriting, got {sorted(tempos)!r}"
        )
    return float(mido.tempo2bpm(next(iter(tempos))))


def _rescale_bar_offset_message(
    message: mido.Message | mido.MetaMessage,
    scale: float,
) -> mido.Message | mido.MetaMessage:
    value = _bar_offset_value(message) if message.is_meta else None
    if value is None:
        return message
    scaled = value * float(scale)
    if not isfinite(scaled) or scaled < 0.0:
        raise RuntimeError(
            f"Invalid scaled MuScriptor bar offset: source={value}, scale={scale}"
        )
    return message.copy(text=f"{_MUSCRIPTOR_BAR_OFFSET_PREFIX}{scaled:.4f}")


def _project_speed_tick_fingerprint(
    midi: mido.MidiFile,
    bar_offset_scale: float,
) -> list[list[tuple[bytes, int]]]:
    fingerprint: list[list[tuple[bytes, int]]] = []
    for track in midi.tracks:
        absolute_tick = 0
        events: list[tuple[bytes, int]] = []
        for message in track:
            absolute_tick += int(message.time)
            if message.is_meta and message.type in {"set_tempo", "end_of_track"}:
                continue
            expected = _rescale_bar_offset_message(message, bar_offset_scale)
            events.append((message_payload(expected), absolute_tick))
        fingerprint.append(events)
    return fingerprint


def validated_midi_bpm(value: float, label: str = "MIDI") -> float:
    """Return a finite supported BPM or fail explicitly."""

    bpm = float(value)
    if not isfinite(bpm) or not MIN_MIDI_BPM <= bpm <= MAX_MIDI_BPM:
        raise ValueError(
            f"Invalid {label} BPM: expected {MIN_MIDI_BPM:g}–" f"{MAX_MIDI_BPM:g}, got {value!r}"
        )
    return bpm


def _tempo_sections(
    midi: mido.MidiFile,
    track_index: int,
) -> tuple[list[int], list[tuple[int, float, int]]]:
    """Build tick/second/tempo sections for one track's effective timeline."""

    source_tracks = (
        ((track_index, midi.tracks[track_index]),)
        if midi.type == 2
        else tuple(enumerate(midi.tracks))
    )
    tempo_events: list[tuple[int, int, int, int]] = []
    for source_track_index, track in source_tracks:
        absolute_tick = 0
        for sequence, message in enumerate(track):
            absolute_tick += int(message.time)
            if message.is_meta and message.type == "set_tempo":
                tempo_events.append(
                    (
                        absolute_tick,
                        source_track_index,
                        sequence,
                        int(message.tempo),
                    )
                )

    collapsed: dict[int, int] = {}
    for tick, _source_track_index, _sequence, tempo_us in sorted(tempo_events):
        collapsed[tick] = tempo_us

    sections: list[tuple[int, float, int]] = [(0, 0.0, _DEFAULT_TEMPO_US)]
    for event_tick, event_tempo in sorted(collapsed.items()):
        previous_tick, previous_seconds, previous_tempo = sections[-1]
        if event_tick == previous_tick:
            sections[-1] = (event_tick, previous_seconds, event_tempo)
            continue
        event_seconds = previous_seconds + mido.tick2second(
            event_tick - previous_tick,
            midi.ticks_per_beat,
            previous_tempo,
        )
        sections.append((event_tick, event_seconds, event_tempo))

    return [tick for tick, _seconds, _tempo in sections], sections


def track_tick_to_seconds(
    midi: mido.MidiFile,
    track_index: int,
) -> tuple[Callable[[int], float], float]:
    """Return an absolute tick converter and the track's worst one-tick duration."""

    section_ticks, sections = _tempo_sections(midi, track_index)

    def convert(absolute_tick: int) -> float:
        index = bisect_right(section_ticks, int(absolute_tick)) - 1
        start_tick, start_seconds, tempo_us = sections[index]
        return float(
            start_seconds
            + mido.tick2second(
                int(absolute_tick) - start_tick,
                midi.ticks_per_beat,
                tempo_us,
            )
        )

    max_tick_seconds = max(
        tempo_us / 1_000_000.0 / midi.ticks_per_beat for _tick, _seconds, tempo_us in sections
    )
    return convert, max_tick_seconds


def message_payload(message: mido.Message | mido.MetaMessage) -> bytes:
    """Return the complete serialized message bytes without delta time."""

    return bytes(message.bytes()) if message.is_meta else bytes(message.bin())


def non_tempo_event_time_fingerprint(
    midi: mido.MidiFile,
) -> tuple[list[list[tuple[bytes, float]]], list[float]]:
    """Capture every non-tempo/non-EOT event and its absolute playback second."""

    fingerprint: list[list[tuple[bytes, float]]] = []
    resolutions: list[float] = []
    for track_index, track in enumerate(midi.tracks):
        tick_to_seconds, tick_resolution = track_tick_to_seconds(midi, track_index)
        resolutions.append(tick_resolution)
        absolute_tick = 0
        track_events: list[tuple[bytes, float]] = []
        for message in track:
            absolute_tick += int(message.time)
            if message.is_meta and message.type in {"set_tempo", "end_of_track"}:
                continue
            track_events.append((message_payload(message), tick_to_seconds(absolute_tick)))
        fingerprint.append(track_events)
    return fingerprint, resolutions


def non_tempo_event_tick_fingerprint(
    midi: mido.MidiFile,
) -> list[list[tuple[bytes, int]]]:
    """Capture every non-tempo/non-EOT event and its absolute tick."""

    fingerprint: list[list[tuple[bytes, int]]] = []
    for track in midi.tracks:
        absolute_tick = 0
        track_events: list[tuple[bytes, int]] = []
        for message in track:
            absolute_tick += int(message.time)
            if message.is_meta and message.type in {"set_tempo", "end_of_track"}:
                continue
            track_events.append((message_payload(message), absolute_tick))
        fingerprint.append(track_events)
    return fingerprint


def validated_midi_time_signature(
    value: tuple[int, int],
) -> tuple[int, int]:
    """Return a Standard MIDI File-compatible numerator and denominator."""

    if len(value) != 2:
        raise ValueError(f"Invalid MIDI time signature: {value!r}")
    numerator = int(value[0])
    denominator = int(value[1])
    if (
        numerator <= 0
        or numerator > 255
        or denominator <= 0
        or denominator > 128
        or denominator & (denominator - 1)
    ):
        raise ValueError(
            "Invalid MIDI time signature: expected numerator 1..255 and "
            f"power-of-two denominator 1..128, got {value!r}"
        )
    return numerator, denominator


def _non_time_signature_event_tick_fingerprint(
    midi: mido.MidiFile,
) -> list[list[tuple[bytes, int]]]:
    """Capture every non-time-signature/non-EOT event at its absolute tick."""

    fingerprint: list[list[tuple[bytes, int]]] = []
    for track in midi.tracks:
        absolute_tick = 0
        track_events: list[tuple[bytes, int]] = []
        for message in track:
            absolute_tick += int(message.time)
            if message.is_meta and message.type in {"time_signature", "end_of_track"}:
                continue
            track_events.append((message_payload(message), absolute_tick))
        fingerprint.append(track_events)
    return fingerprint


def write_midi_time_signature_preserving_ticks(
    source_path: str | Path,
    destination_path: str | Path,
    time_signature: tuple[int, int] | None,
    *,
    label: str = "MIDI conductor metadata export",
) -> Path:
    """Write one detected meter, or remove placeholder meters when unknown."""

    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    validated_signature = (
        validated_midi_time_signature(time_signature) if time_signature is not None else None
    )
    source_midi = mido.MidiFile(str(source))
    if not source_midi.tracks:
        raise RuntimeError(f"MIDI source has no tracks: {source}")
    if source_midi.ticks_per_beat <= 0:
        raise RuntimeError(
            "SMPTE time-division MIDI is not supported for conductor metadata: "
            f"path={source}, division={source_midi.ticks_per_beat}"
        )

    output = mido.MidiFile(
        type=source_midi.type,
        ticks_per_beat=source_midi.ticks_per_beat,
    )
    conductor_tracks = set(range(len(source_midi.tracks))) if source_midi.type == 2 else {0}
    clocks_per_click = 24
    if validated_signature is not None:
        numerator, denominator = validated_signature
        clocks_per_click = (
            36 if denominator == 8 and numerator > 3 and numerator % 3 == 0 else 24
        )

    for track_index, source_track in enumerate(source_midi.tracks):
        absolute_tick = 0
        events: list[tuple[int, int, int, mido.Message | mido.MetaMessage]] = []
        end_ticks: list[int] = []
        for sequence, message in enumerate(source_track):
            absolute_tick += int(message.time)
            if message.is_meta and message.type == "time_signature":
                continue
            if message.is_meta and message.type == "end_of_track":
                end_ticks.append(absolute_tick)
                continue
            events.append((absolute_tick, 1, sequence, message.copy(time=0)))

        if track_index in conductor_tracks and validated_signature is not None:
            numerator, denominator = validated_signature
            events.append(
                (
                    0,
                    0,
                    -1,
                    mido.MetaMessage(
                        "time_signature",
                        numerator=numerator,
                        denominator=denominator,
                        clocks_per_click=clocks_per_click,
                        notated_32nd_notes_per_beat=8,
                        time=0,
                    ),
                )
            )

        last_event_tick = max((tick for tick, _order, _sequence, _msg in events), default=0)
        end_tick = max([last_event_tick, *end_ticks])
        events.append(
            (
                end_tick,
                2,
                len(source_track),
                mido.MetaMessage("end_of_track", time=0),
            )
        )
        events.sort(key=lambda event: (event[0], event[1], event[2]))

        target_track = mido.MidiTrack()
        previous_tick = 0
        for target_tick, _order, _sequence, message in events:
            target_track.append(message.copy(time=target_tick - previous_tick))
            previous_tick = target_tick
        output.tracks.append(target_track)

    source_fingerprint = _non_time_signature_event_tick_fingerprint(source_midi)
    temporary = unique_midi_temp_path(destination, "conductor-metadata")
    try:
        output.save(str(temporary))
        verified = mido.MidiFile(str(temporary))
        published_signatures: list[tuple[int, int, int, int]] = []
        for track_index, track in enumerate(verified.tracks):
            absolute_tick = 0
            for message in track:
                absolute_tick += int(message.time)
                if message.is_meta and message.type == "time_signature":
                    published_signatures.append(
                        (
                            track_index,
                            absolute_tick,
                            int(message.numerator),
                            int(message.denominator),
                        )
                    )
        expected_signatures = []
        if validated_signature is not None:
            numerator, denominator = validated_signature
            expected_signatures = [
                (track_index, 0, numerator, denominator)
                for track_index in sorted(conductor_tracks)
            ]
        if published_signatures != expected_signatures:
            raise RuntimeError(
                "MIDI conductor metadata verification found incorrect time "
                f"signatures: path={temporary}, expected={expected_signatures!r}, "
                f"actual={published_signatures!r}"
            )

        output_fingerprint = _non_time_signature_event_tick_fingerprint(verified)
        if output_fingerprint != source_fingerprint:
            raise RuntimeError(
                "MIDI conductor metadata verification changed a non-meter event "
                f"or its absolute tick: path={temporary}"
            )
        expected_bytes = temporary.read_bytes()
        published = Path(publish_midi_output(temporary, destination, label))
        if published.read_bytes() != expected_bytes:
            raise RuntimeError(
                "MIDI conductor metadata atomic publication changed the verified "
                f"file: path={published}"
            )
        return published
    finally:
        remove_temporary_midi(temporary)


def rewrite_midi_tempo_preserving_seconds(
    source_path: str | Path,
    destination_path: str | Path,
    bpm: float,
    *,
    label: str = "Tempo-preserving MIDI export",
) -> Path:
    """Write one constant project BPM without changing event playback seconds."""

    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    export_bpm = validated_midi_bpm(bpm)
    source_midi = mido.MidiFile(str(source))
    if not source_midi.tracks:
        raise RuntimeError(f"MIDI source has no tracks: {source}")
    if source_midi.ticks_per_beat <= 0:
        raise RuntimeError(
            "SMPTE time-division MIDI is not supported for BPM rewriting: "
            f"path={source}, division={source_midi.ticks_per_beat}"
        )

    tempo_us = mido.bpm2tempo(export_bpm)
    output = mido.MidiFile(
        type=source_midi.type,
        ticks_per_beat=source_midi.ticks_per_beat,
    )
    tempo_tracks = set(range(len(source_midi.tracks))) if source_midi.type == 2 else {0}

    for track_index, source_track in enumerate(source_midi.tracks):
        tick_to_seconds, _tick_resolution = track_tick_to_seconds(
            source_midi,
            track_index,
        )
        absolute_tick = 0
        events: list[tuple[int, int, int, mido.Message | mido.MetaMessage]] = []
        end_seconds: list[float] = []
        for sequence, message in enumerate(source_track):
            absolute_tick += int(message.time)
            absolute_seconds = tick_to_seconds(absolute_tick)
            if message.is_meta and message.type == "set_tempo":
                continue
            if message.is_meta and message.type == "end_of_track":
                end_seconds.append(absolute_seconds)
                continue
            target_tick = max(
                0,
                int(
                    round(
                        mido.second2tick(
                            absolute_seconds,
                            source_midi.ticks_per_beat,
                            tempo_us,
                        )
                    )
                ),
            )
            events.append((target_tick, 1, sequence, message.copy(time=0)))

        if track_index in tempo_tracks:
            events.append(
                (
                    0,
                    0,
                    -1,
                    mido.MetaMessage("set_tempo", tempo=tempo_us, time=0),
                )
            )

        last_event_tick = max((tick for tick, _order, _sequence, _msg in events), default=0)
        for sequence, absolute_seconds in enumerate(end_seconds, start=len(source_track)):
            end_tick = max(
                last_event_tick,
                int(
                    round(
                        mido.second2tick(
                            absolute_seconds,
                            source_midi.ticks_per_beat,
                            tempo_us,
                        )
                    )
                ),
            )
            events.append(
                (
                    end_tick,
                    2,
                    sequence,
                    mido.MetaMessage("end_of_track", time=0),
                )
            )

        events.sort(key=lambda event: (event[0], event[1], event[2]))
        target_track = mido.MidiTrack()
        previous_tick = 0
        for target_tick, _order, _sequence, message in events:
            target_track.append(message.copy(time=target_tick - previous_tick))
            previous_tick = target_tick
        output.tracks.append(target_track)

    source_fingerprint, source_resolutions = non_tempo_event_time_fingerprint(source_midi)
    temporary = unique_midi_temp_path(destination, "tempo-preserving")
    try:
        output.save(str(temporary))
        verified = mido.MidiFile(str(temporary))
        published_tempos = [
            int(message.tempo)
            for track in verified.tracks
            for message in track
            if message.is_meta and message.type == "set_tempo"
        ]
        expected_tempo_count = len(verified.tracks) if verified.type == 2 else 1
        if len(published_tempos) != expected_tempo_count or any(
            value != tempo_us for value in published_tempos
        ):
            raise RuntimeError(
                "Tempo-preserving MIDI verification found incorrect tempo events: "
                f"path={temporary}, expected={tempo_us}, actual={published_tempos!r}"
            )

        output_fingerprint, output_resolutions = non_tempo_event_time_fingerprint(verified)
        if len(source_fingerprint) != len(output_fingerprint):
            raise RuntimeError(
                "Tempo-preserving MIDI verification changed the track count: " f"path={temporary}"
            )
        for track_index, (source_events, output_events) in enumerate(
            zip(source_fingerprint, output_fingerprint)
        ):
            if len(source_events) != len(output_events):
                raise RuntimeError(
                    "Tempo-preserving MIDI verification changed event count: "
                    f"path={temporary}, track={track_index}, "
                    f"source={len(source_events)}, output={len(output_events)}"
                )
            tolerance = source_resolutions[track_index] + output_resolutions[track_index] + 1e-9
            for event_index, (
                (source_payload, source_seconds),
                (output_payload, output_seconds),
            ) in enumerate(zip(source_events, output_events)):
                if (
                    source_payload != output_payload
                    or abs(source_seconds - output_seconds) > tolerance
                ):
                    raise RuntimeError(
                        "Tempo-preserving MIDI verification changed an event: "
                        f"path={temporary}, track={track_index}, "
                        f"event={event_index}, source_seconds={source_seconds:.9f}, "
                        f"output_seconds={output_seconds:.9f}, "
                        f"tolerance={tolerance:.9f}"
                    )
        expected_bytes = temporary.read_bytes()
        published = Path(publish_midi_output(temporary, destination, label))
        if published.read_bytes() != expected_bytes:
            raise RuntimeError(
                "Tempo-preserving MIDI atomic publication changed the verified file: "
                f"path={published}"
            )
        return published
    finally:
        remove_temporary_midi(temporary)


def rewrite_midi_tempo_preserving_ticks(
    source_path: str | Path,
    destination_path: str | Path,
    bpm: float,
    *,
    label: str = "Tick-preserving MIDI tempo export",
) -> Path:
    """Replace tempo while preserving every non-tempo event's absolute tick.

    This is the deliberate project-speed operation: musical beat positions stay
    unchanged while the constant tempo changes their real playback seconds.
    """

    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    export_bpm = validated_midi_bpm(bpm)
    source_midi = mido.MidiFile(str(source))
    if not source_midi.tracks:
        raise RuntimeError(f"MIDI source has no tracks: {source}")
    if source_midi.ticks_per_beat <= 0:
        raise RuntimeError(
            "SMPTE time-division MIDI is not supported for BPM rewriting: "
            f"path={source}, division={source_midi.ticks_per_beat}"
        )

    tempo_us = mido.bpm2tempo(export_bpm)
    bar_offset_seconds = read_muscriptor_bar_offset_seconds(source_midi)
    bar_offset_scale = 1.0
    if bar_offset_seconds > 0.0:
        bar_offset_scale = _constant_source_bpm_for_bar_offset(source_midi) / export_bpm
    output = mido.MidiFile(
        type=source_midi.type,
        ticks_per_beat=source_midi.ticks_per_beat,
    )
    tempo_tracks = set(range(len(source_midi.tracks))) if source_midi.type == 2 else {0}

    for track_index, source_track in enumerate(source_midi.tracks):
        absolute_tick = 0
        events: list[tuple[int, int, int, mido.Message | mido.MetaMessage]] = []
        end_ticks: list[int] = []
        for sequence, message in enumerate(source_track):
            absolute_tick += int(message.time)
            if message.is_meta and message.type == "set_tempo":
                continue
            if message.is_meta and message.type == "end_of_track":
                end_ticks.append(absolute_tick)
                continue
            rewritten = _rescale_bar_offset_message(message, bar_offset_scale)
            events.append((absolute_tick, 1, sequence, rewritten.copy(time=0)))

        if track_index in tempo_tracks:
            events.append(
                (
                    0,
                    0,
                    -1,
                    mido.MetaMessage("set_tempo", tempo=tempo_us, time=0),
                )
            )

        last_event_tick = max((tick for tick, _order, _sequence, _msg in events), default=0)
        end_tick = max([last_event_tick, *end_ticks])
        events.append(
            (
                end_tick,
                2,
                len(source_track),
                mido.MetaMessage("end_of_track", time=0),
            )
        )
        events.sort(key=lambda event: (event[0], event[1], event[2]))

        target_track = mido.MidiTrack()
        previous_tick = 0
        for target_tick, _order, _sequence, message in events:
            target_track.append(message.copy(time=target_tick - previous_tick))
            previous_tick = target_tick
        output.tracks.append(target_track)

    source_fingerprint = _project_speed_tick_fingerprint(source_midi, bar_offset_scale)
    temporary = unique_midi_temp_path(destination, "tempo-tick-preserving")
    try:
        output.save(str(temporary))
        verified = mido.MidiFile(str(temporary))
        published_tempos = [
            int(message.tempo)
            for track in verified.tracks
            for message in track
            if message.is_meta and message.type == "set_tempo"
        ]
        expected_tempo_count = len(verified.tracks) if verified.type == 2 else 1
        if len(published_tempos) != expected_tempo_count or any(
            value != tempo_us for value in published_tempos
        ):
            raise RuntimeError(
                "Tick-preserving MIDI verification found incorrect tempo events: "
                f"path={temporary}, expected={tempo_us}, actual={published_tempos!r}"
            )

        output_fingerprint = non_tempo_event_tick_fingerprint(verified)
        if output_fingerprint != source_fingerprint:
            raise RuntimeError(
                "Tick-preserving MIDI verification changed a non-tempo event or "
                f"its absolute tick: path={temporary}"
            )
        expected_bytes = temporary.read_bytes()
        published = Path(publish_midi_output(temporary, destination, label))
        if published.read_bytes() != expected_bytes:
            raise RuntimeError(
                "Tick-preserving MIDI atomic publication changed the verified file: "
                f"path={published}"
            )
        return published
    finally:
        remove_temporary_midi(temporary)


def rewrite_midi_tempo_for_project_speed(
    source_path: str | Path,
    destination_path: str | Path,
    reference_bpm: float,
    target_bpm: float,
    *,
    label: str = "Project-speed MIDI export",
) -> Path:
    """Map backend seconds onto reference beats, then apply target tempo.

    The first strict rewrite retains the backend's original event seconds while
    expressing them on the detected/reference BPM tick grid. The second retains
    those ticks and writes the requested project BPM, so the resulting duration
    scales by ``reference_bpm / target_bpm`` without changing event payloads or
    snapping notes to a grid.
    """

    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    reference = validated_midi_bpm(reference_bpm, "reference")
    target = validated_midi_bpm(target_bpm, "target")
    reference_tempo_midi = unique_midi_temp_path(destination, "reference-tempo")
    try:
        rewrite_midi_tempo_preserving_seconds(
            source,
            reference_tempo_midi,
            reference,
            label=f"{label} reference-timeline stage",
        )
        return rewrite_midi_tempo_preserving_ticks(
            reference_tempo_midi,
            destination,
            target,
            label=label,
        )
    finally:
        remove_temporary_midi(reference_tempo_midi)


__all__ = [
    "message_payload",
    "non_tempo_event_tick_fingerprint",
    "non_tempo_event_time_fingerprint",
    "read_muscriptor_bar_offset_seconds",
    "rewrite_midi_tempo_for_project_speed",
    "rewrite_midi_tempo_preserving_seconds",
    "rewrite_midi_tempo_preserving_ticks",
    "track_tick_to_seconds",
    "validated_midi_bpm",
    "validated_midi_time_signature",
    "write_midi_time_signature_preserving_ticks",
]
