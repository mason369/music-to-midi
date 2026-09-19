"""Download individual instrument MIDIs without altering published result timing."""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from collections import defaultdict, deque
from pathlib import Path

import mido
from fastapi import HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from src.core.muscriptor_result_assets import read_midi_roll_notes
from src.i18n.translator import Translator
from src.models.gm_instruments import get_instrument_name
from src.models.muscriptor_instruments import muscriptor_instrument_label


def build_source_midi_archive(source: Path, destination: Path, language: str) -> int:
    """Filter paired note events only, preserving tempo maps and every event tick."""
    midi = mido.MidiFile(source)
    instruments = list(dict.fromkeys(note.instrument for note in read_midi_roll_notes(source)))
    if not instruments:
        raise ValueError("MIDI contains no instrument notes")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, instrument in enumerate(instruments, 1):
            output = mido.MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat)
            for source_track in midi.tracks:
                track = mido.MidiTrack()
                programs = defaultdict(int)
                active = defaultdict(deque)
                absolute_tick = previous_tick = 0
                for message in source_track:
                    absolute_tick += message.time
                    retain = True
                    if message.type == "program_change":
                        programs[message.channel] = message.program
                    elif message.type == "note_on" and message.velocity:
                        key = (
                            "drums"
                            if message.channel == 9
                            else f"gm:{programs[message.channel]:03d}"
                        )
                        retain = key == instrument
                        active[(message.channel, message.note)].append(retain)
                    elif message.type in {"note_on", "note_off"}:
                        queue = active[(message.channel, message.note)]
                        retain = queue.popleft() if queue else False
                    if retain:
                        track.append(message.copy(time=absolute_tick - previous_tick))
                        previous_tick = absolute_tick
                if any(active.values()):
                    raise ValueError("MIDI contains unmatched note-on events")
                output.tracks.append(track)
            data = io.BytesIO()
            output.save(file=data)
            # Parse before publication; the ZIP is a derivative, never the source.
            mido.MidiFile(file=io.BytesIO(data.getvalue()))
            label = (
                muscriptor_instrument_label("drums", language)
                if instrument == "drums"
                else get_instrument_name(int(instrument.split(":")[1]), language)
            )
            label = "".join("_" if char in '<>:"/\\|?*' else char for char in label)
            archive.writestr(f"{index:02d}-{label}.mid", data.getvalue())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("MIDI stem archive verification failed")
    return len(instruments)


def install_midi_export_routes(app, manager, *, prefix="/api/v1/jobs"):
    @app.get(prefix + "/{job_id}/midi-stems/{artifact_id}")
    def export_midi_stems(job_id: str, artifact_id: str, language: str = "zh_CN"):
        temporary = None
        try:
            if language not in Translator.AVAILABLE_LANGUAGES:
                raise ValueError("Unsupported language")
            with manager._lock:
                source, artifact = manager.resolve_artifact(job_id, artifact_id)
                if artifact.kind != "midi":
                    raise ValueError("Instrument MIDI export requires a MIDI result")
                temporary = tempfile.TemporaryDirectory(prefix="music-to-midi-stems-")
                root = Path(temporary.name)
                snapshot = root / "source.mid"
                shutil.copyfile(source, snapshot)
            destination = root / "midi-stems.zip"
            count = build_source_midi_archive(snapshot, destination, language)
            return FileResponse(
                destination,
                media_type="application/zip",
                filename="midi-stems.zip",
                headers={"X-MIDI-Instrument-Count": str(count)},
                background=BackgroundTask(temporary.cleanup),
            )
        except (KeyError, OSError, ValueError, RuntimeError) as exc:
            if temporary is not None:
                temporary.cleanup()
            code = 404 if isinstance(exc, (KeyError, FileNotFoundError)) else 422
            raise HTTPException(status_code=code, detail=str(exc)) from exc
