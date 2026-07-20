"""Strict MuScriptor instrument-constraint checks for events and MIDI files."""

from __future__ import annotations

from pathlib import Path

from src.models.muscriptor_instruments import (
    MUSCRIPTOR_REPRESENTATIVE_PROGRAMS,
    validate_muscriptor_instruments,
)


def require_allowed_muscriptor_event_instrument(
    instrument: str,
    selected_instruments: list[str],
) -> None:
    """Reject a model event whose instrument was not explicitly selected."""

    if not selected_instruments:
        return
    if instrument not in selected_instruments:
        raise RuntimeError(
            "MuScriptor hard instrument constraint violation in model event stream: "
            f"instrument={instrument!r}, allowed={selected_instruments!r}. "
            "The MIDI output was not published."
        )


def validate_muscriptor_midi_constraint(
    midi_path: str | Path,
    selected_instruments: list[str],
) -> Path:
    """Reject any note whose final MIDI program/drum channel is not selected."""

    selected = validate_muscriptor_instruments(selected_instruments)
    path = Path(midi_path)
    if not selected:
        return path

    import mido

    allowed_programs = {
        MUSCRIPTOR_REPRESENTATIVE_PROGRAMS[name] for name in selected if name != "drums"
    }
    drums_allowed = "drums" in selected
    violations: list[str] = []
    midi = mido.MidiFile(str(path))
    for track_index, track in enumerate(midi.tracks):
        programs = {channel: 0 for channel in range(16)}
        absolute_tick = 0
        track_name = f"track {track_index}"
        for message in track:
            absolute_tick += int(message.time)
            if message.is_meta:
                if message.type == "track_name" and str(message.name).strip():
                    track_name = str(message.name)
                continue
            if message.type == "program_change":
                programs[int(message.channel)] = int(message.program)
                continue
            if message.type != "note_on" or int(message.velocity) <= 0:
                continue

            channel = int(message.channel)
            if channel == 9:
                if not drums_allowed:
                    violations.append(
                        f"{track_name}: drum note {int(message.note)} at tick {absolute_tick}"
                    )
            else:
                program = programs[channel]
                if program not in allowed_programs:
                    violations.append(
                        f"{track_name}: program {program}, channel {channel}, "
                        f"note {int(message.note)} at tick {absolute_tick}"
                    )
            if len(violations) >= 12:
                break
        if len(violations) >= 12:
            break

    if violations:
        raise RuntimeError(
            "MuScriptor hard instrument constraint violation in final MIDI; "
            "the file was not published. Allowed instruments: "
            f"{selected!r}. Violations: {'; '.join(violations)}"
        )
    return path
