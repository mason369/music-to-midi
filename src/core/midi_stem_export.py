"""Explicit instrument exports on the editor's existing musical tick grid."""

from __future__ import annotations

import re
import tempfile
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

from src.core.midi_editor import export_edited_midi
from src.core.midi_tempo import repeat_tempo_events_on_note_tracks
from src.core.muscriptor_result_assets import MuscriptorRollNote


def export_midi_selection(
    source: str | Path,
    destination: str | Path,
    notes: Iterable[MuscriptorRollNote],
    *,
    reference_bpm: float,
    target_bpm: float,
    repeat_tempo: bool = False,
) -> Path:
    """Retain controllers, meter and ticks while exporting an explicit note subset."""
    selected = tuple(notes)
    if not selected:
        raise ValueError("No audible MIDI instruments selected")
    result = export_edited_midi(
        source, destination, selected, reference_bpm=reference_bpm, target_bpm=target_bpm
    )
    if repeat_tempo:
        result = repeat_tempo_events_on_note_tracks(result, label="Instrument MIDI export")
    return Path(result)


def export_midi_stems_zip(
    source: str | Path,
    destination: str | Path,
    notes: Iterable[MuscriptorRollNote],
    *,
    reference_bpm: float,
    target_bpm: float,
    instrument_label: Callable[[str], str] = str,
    repeat_tempo: bool = False,
) -> tuple[Path, int]:
    """Atomically publish every current instrument, regardless of monitor mute/solo."""
    groups: dict[str, list[MuscriptorRollNote]] = {}
    for note in notes:
        groups.setdefault(note.instrument, []).append(note)
    if not groups:
        raise ValueError("No MIDI instruments to export")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".midi-stems-", dir=destination.parent) as staging:
        root = Path(staging)
        archive = root / "stems.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for index, (instrument, selected) in enumerate(groups.items(), 1):
                label = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", instrument_label(instrument))
                label = label.strip(" .")[:80] or f"instrument-{index}"
                name = f"{index:02d}-{label}-{target_bpm:.1f}BPM.mid"
                midi = export_midi_selection(
                    source,
                    root / name,
                    selected,
                    reference_bpm=reference_bpm,
                    target_bpm=target_bpm,
                    repeat_tempo=repeat_tempo,
                )
                output.write(midi, name)
        with zipfile.ZipFile(archive) as verification:
            if len(verification.namelist()) != len(groups) or verification.testzip() is not None:
                raise RuntimeError("MIDI stem archive verification failed")
        archive.replace(destination)
    return destination, len(groups)
