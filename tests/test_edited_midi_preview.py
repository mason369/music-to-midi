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
        duration_seconds=2.5,
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
        duration_seconds=1.0,
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


def test_browser_audio_export_uses_current_notes_selected_preset_and_cache(
    tmp_path: Path,
    monkeypatch,
):
    request_dir = tmp_path / "request-test"
    request_dir.mkdir()
    source_midi = request_dir / "source.mid"
    source_audio = request_dir / "source.wav"
    source_midi.write_bytes(b"MThd-source")
    source_audio.write_bytes(b"RIFF-source")
    exported_notes = []
    rendered = []

    def fake_export(source, destination, notes, *, reference_bpm, target_bpm):
        destination = Path(destination)
        destination.write_bytes(b"MThd-current-notes")
        exported_notes.append((Path(source), tuple(notes), reference_bpm, target_bpm))
        return destination

    def fake_render(
        midi_path,
        destination,
        preset_id,
        *,
        silence_duration_seconds,
    ):
        preset = edited_midi_preview.get_midi_audio_export_preset(preset_id)
        destination = Path(destination)
        destination.write_bytes(b"RIFF-verified")
        rendered.append((Path(midi_path), destination, preset_id, silence_duration_seconds))
        return SimpleNamespace(
            path=destination,
            preset=preset,
            frames=120_000,
            channels=2,
            peak=0.25,
        )

    monkeypatch.setattr(edited_midi_preview, "export_edited_midi", fake_export)
    monkeypatch.setattr(edited_midi_preview, "render_midi_audio_export", fake_render)
    registry = EditedMidiPreviewRegistry()
    token = registry.register(
        request_dir=request_dir,
        source_midi_path=source_midi,
        original_audio_path=source_audio,
        reference_bpm=117.9,
        muscriptor_groups=False,
        duration_seconds=2.5,
    )
    payload = json.dumps({"token": token, "notes": [_note_payload()], "preset": "pcm24_48000"})

    first = registry.export_audio(payload)
    second = registry.export_audio(payload)

    assert first == second
    assert len(exported_notes) == 1
    assert exported_notes[0][0] == source_midi.resolve()
    assert exported_notes[0][2:] == (117.9, 117.9)
    assert len(rendered) == 1
    assert rendered[0][2:] == ("pcm24_48000", 2.5)
    assert rendered[0][1].name == "render.wav"
    assert len(rendered[0][1].parts[-3]) == 43
    assert first["presetId"] == "pcm24_48000"
    assert first["bitDepth"] == 24
    assert first["sampleRate"] == 48_000
    assert first["subtype"] == "PCM_24"
    assert first["channels"] == 2
    assert first["frames"] == 120_000
    assert first["peak"] == 0.25
    assert "midi-wav" in first["url"]
    assert first["filename"] == "transcription-24bit-48kHz.wav"

    invalid = json.dumps({"token": token, "notes": [_note_payload()], "preset": "pcm32_192000"})
    with pytest.raises(ValueError, match="Unsupported MIDI audio export preset"):
        registry.export_audio(invalid)
