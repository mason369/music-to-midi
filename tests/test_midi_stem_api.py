"""Real MIDI ZIP responses for job and restored-project delivery surfaces."""

import io
import threading
import zipfile
from types import SimpleNamespace

import mido
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.web_api.midi_exports import install_midi_export_routes


def event_ticks(midi):
    events = []
    for index, track in enumerate(midi.tracks):
        tick = 0
        for message in track:
            tick += message.time
            if message.type != "end_of_track":
                events.append((index, tick, message.copy(time=0)))
    return events


@pytest.mark.parametrize("prefix", ["/api/v1/jobs", "/api/v1/project-results"])
@pytest.mark.parametrize("language", ["zh_CN", "en_US"])
def test_download_stem_archive_preserves_variable_tempo_and_note_pairs(tmp_path, prefix, language):
    source = tmp_path / "source.mid"
    midi = mido.MidiFile(type=1)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500000),
                mido.MetaMessage("time_signature", numerator=6, denominator=8),
                mido.MetaMessage("set_tempo", tempo=700000, time=480),
            ]
        )
    )
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("program_change", program=32, channel=0),
                mido.Message("control_change", control=64, value=127, time=120),
                mido.Message("note_on", note=40, velocity=100, time=120),
                mido.Message("program_change", program=33, time=120),
                mido.Message("note_on", note=43, velocity=95),
                mido.Message("note_off", note=40, time=480),
                mido.Message("note_off", note=43),
                mido.Message("control_change", control=64, value=0),
            ]
        )
    )
    midi.save(source)
    before = source.read_bytes()

    class Resolver:
        _lock = threading.RLock()

        def resolve_artifact(self, job_id, artifact_id):
            if job_id != "saved-result" or artifact_id != "midi":
                raise KeyError("Unknown result")
            return source, SimpleNamespace(kind="midi")

    app = FastAPI()
    install_midi_export_routes(app, Resolver(), prefix=prefix)
    with TestClient(app) as client:
        result = client.get(f"{prefix}/saved-result/midi-stems/midi?language={language}")
        assert result.status_code == 200
        assert result.headers["x-midi-instrument-count"] == "2"
        assert result.headers["content-type"] == "application/zip"
        assert client.get(f"{prefix}/other/midi-stems/midi").status_code == 404
        assert (
            client.get(f"{prefix}/saved-result/midi-stems/midi?language=invalid").status_code == 422
        )
    with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
        assert len(archive.namelist()) == 2 and archive.testzip() is None
        all_notes = []
        for name in archive.namelist():
            part = mido.MidiFile(file=io.BytesIO(archive.read(name)))
            actual = event_ticks(part)
            assert [e for e in actual if e[2].type not in {"note_on", "note_off"}] == [
                e for e in event_ticks(midi) if e[2].type not in {"note_on", "note_off"}
            ]
            notes = [
                (tick, msg.type, msg.note)
                for _, tick, msg in actual
                if msg.type in {"note_on", "note_off"}
            ]
            assert notes in [
                [(240, "note_on", 40), (840, "note_off", 40)],
                [(360, "note_on", 43), (840, "note_off", 43)],
            ]
            all_notes.extend(notes)
        assert len(all_notes) == 4
    assert source.read_bytes() == before
