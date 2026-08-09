import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.gui.web import edited_midi_preview
from src.gui.web.edited_midi_preview import EditedMidiPreviewRegistry


def _note_payload():
    return {
        "instrument": "gm:000",
        "pitch": 60,
        "velocity": 91,
        "start": 0.25,
        "end": 0.75,
        "program": 0,
        "is_drum": False,
        "track_index": 0,
        "channel": 0,
    }


def test_browser_edited_preview_uses_opaque_context_and_exact_shared_renderer(
    tmp_path: Path, monkeypatch
):
    request_dir = tmp_path / "request-test"
    request_dir.mkdir()
    source_midi = request_dir / "source.mid"
    source_audio = request_dir / "source.wav"
    source_midi.write_bytes(b"MThd-source")
    source_audio.write_bytes(b"RIFF-source")
    calls = []

    def fake_export(source, destination, notes, *, reference_bpm, target_bpm):
        calls.append((Path(source), tuple(notes), reference_bpm, target_bpm))
        destination = Path(destination)
        destination.write_bytes(b"MThd-edited")
        return destination

    def fake_prepare(midi_path, audio_path, output_dir, **kwargs):
        output_dir = Path(output_dir)
        transcription = output_dir / "transcription.wav"
        stereo = output_dir / "stereo.wav"
        piano = output_dir / "piano.wav"
        for path in (transcription, stereo, piano):
            path.write_bytes(b"RIFF-rendered")
        return SimpleNamespace(
            notes=(object(),),
            duration=2.5,
            transcription_wav=transcription,
            stereo_mix_wav=stereo,
            instrument_wavs={"gm:000": piano},
        )

    monkeypatch.setattr(edited_midi_preview, "export_edited_midi", fake_export)
    monkeypatch.setattr(edited_midi_preview, "prepare_midi_playback_assets", fake_prepare)
    registry = EditedMidiPreviewRegistry()
    token = registry.register(
        request_dir=request_dir,
        source_midi_path=source_midi,
        original_audio_path=source_audio,
        reference_bpm=117.9,
        muscriptor_groups=False,
    )

    payload = json.dumps({"token": token, "notes": [_note_payload()]})
    first = registry.render(payload)
    second = registry.render(payload)

    assert first == second
    assert len(calls) == 1
    assert calls[0][0] == source_midi.resolve()
    assert calls[0][2:] == (117.9, 117.9)
    assert first["noteCount"] == 1
    assert first["duration"] == 2.5
    assert set(first["instrumentUrls"]) == {"gm:000"}
    assert "edited-midi-previews" in first["transcriptionUrl"]
    assert (
        registry.require_matching(
            token,
            request_dir=request_dir,
            source_midi_path=source_midi,
            original_audio_path=source_audio,
        )
        == token
    )


def test_browser_edited_preview_rejects_unknown_or_cross_result_token(tmp_path: Path):
    request_dir = tmp_path / "request-test"
    request_dir.mkdir()
    source_midi = request_dir / "source.mid"
    source_audio = request_dir / "source.wav"
    source_midi.write_bytes(b"MThd-source")
    source_audio.write_bytes(b"RIFF-source")
    registry = EditedMidiPreviewRegistry()
    token = registry.register(
        request_dir=request_dir,
        source_midi_path=source_midi,
        original_audio_path=source_audio,
        reference_bpm=120.0,
        muscriptor_groups=True,
    )

    with pytest.raises(RuntimeError, match="invalid or expired"):
        registry.render('{"token":"unknown","notes":[]}')
    with pytest.raises(RuntimeError, match="does not belong"):
        registry.require_matching(
            token,
            request_dir=request_dir,
            source_midi_path=request_dir / "different.mid",
            original_audio_path=source_audio,
        )
