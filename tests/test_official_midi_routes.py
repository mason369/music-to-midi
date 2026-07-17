from pathlib import Path
from types import SimpleNamespace

import mido
import pytest

from src.core.pipeline import MusicToMidiPipeline
from src.models.data_models import BeatInfo, Config


def _write_valid_backend_midi(path: str) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.extend(
        [
            mido.MetaMessage("track_name", name="official", time=0),
            mido.Message("program_change", program=73, channel=0, time=0),
            mido.Message("control_change", control=11, value=87, channel=0, time=0),
            mido.Message("note_on", note=60, velocity=100, channel=0, time=0),
            mido.Message("note_off", note=60, velocity=0, channel=0, time=480),
        ]
    )
    midi.tracks.append(track)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def test_smart_yourmt3_uses_official_direct_midi_output(monkeypatch, tmp_path):
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    output_dir = tmp_path / "out"

    class FakeYourMT3:
        def __init__(self):
            self.calls = []

        def transcribe_to_midi(self, audio_path, output_path, progress_callback=None):
            self.calls.append((audio_path, output_path))
            _write_valid_backend_midi(output_path)
            if progress_callback:
                progress_callback(1.0, "YourMT3 complete")
            return output_path

    fake_yourmt3 = FakeYourMT3()
    pipeline = object.__new__(MusicToMidiPipeline)
    pipeline.config = Config(
        processing_mode="smart",
        transcription_backend="yourmt3",
        multi_instrument_model="yourmt3",
    )
    pipeline.yourmt3_transcriber = fake_yourmt3
    pipeline.miros_transcriber = SimpleNamespace(
        transcribe_to_midi=lambda **_kwargs: pytest.fail("YourMT3 route must not call MIROS")
    )
    pipeline.midi_generator = SimpleNamespace(
        generate_from_precise_instruments_v2=lambda **_kwargs: pytest.fail(
            "official YourMT3 route must not regenerate MIDI"
        )
    )
    pipeline._cancelled = False
    pipeline._progress_callback = None
    pipeline._translator = SimpleNamespace(t=lambda key, **_kwargs: key)

    monkeypatch.setattr(
        pipeline,
        "_detect_beat_or_raise",
        lambda _path: BeatInfo(bpm=96.0),
    )
    monkeypatch.setattr(pipeline, "_require_multi_instrument_available", lambda: None)
    monkeypatch.setattr(pipeline, "_cleanup_multi_instrument_backend", lambda: None)
    monkeypatch.setattr(MusicToMidiPipeline, "_count_midi_notes", staticmethod(lambda _path: 9))

    result = pipeline._process_smart(str(input_path), str(output_dir))

    assert fake_yourmt3.calls == [(str(input_path), str(output_dir / "song.mid"))]
    assert result.midi_path == str(output_dir / "song.mid")
    assert result.beat_info == BeatInfo(bpm=96.0)
    assert result.total_notes == 9
    messages = [message for track in mido.MidiFile(result.midi_path).tracks for message in track]
    assert any(message.type == "set_tempo" for message in messages)
    assert any(message.type == "program_change" and message.program == 73 for message in messages)


def test_smart_miros_uses_official_direct_midi_output(monkeypatch, tmp_path):
    input_path = tmp_path / "song.wav"
    input_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    output_dir = tmp_path / "out"

    class FakeMiros:
        def __init__(self):
            self.calls = []

        def transcribe_to_midi(self, audio_path, output_path, progress_callback=None):
            self.calls.append((audio_path, output_path))
            _write_valid_backend_midi(output_path)
            if progress_callback:
                progress_callback(1.0, "MIROS complete")
            return output_path

    fake_miros = FakeMiros()
    pipeline = object.__new__(MusicToMidiPipeline)
    pipeline.config = Config(
        processing_mode="smart",
        transcription_backend="miros",
        multi_instrument_model="miros",
    )
    pipeline.miros_transcriber = fake_miros
    pipeline.yourmt3_transcriber = SimpleNamespace(
        transcribe_to_midi=lambda **_kwargs: pytest.fail("MIROS route must not call YourMT3")
    )
    pipeline.midi_generator = SimpleNamespace(
        generate_from_precise_instruments_v2=lambda **_kwargs: pytest.fail(
            "official MIROS route must not regenerate MIDI"
        )
    )
    pipeline._cancelled = False
    pipeline._progress_callback = None
    pipeline._translator = SimpleNamespace(t=lambda key, **_kwargs: key)

    monkeypatch.setattr(
        pipeline,
        "_detect_beat_or_raise",
        lambda _path: BeatInfo(bpm=132.0),
    )
    monkeypatch.setattr(pipeline, "_require_multi_instrument_available", lambda: None)
    monkeypatch.setattr(pipeline, "_cleanup_multi_instrument_backend", lambda: None)
    monkeypatch.setattr(MusicToMidiPipeline, "_count_midi_notes", staticmethod(lambda _path: 7))

    result = pipeline._process_smart(str(input_path), str(output_dir))

    assert fake_miros.calls == [(str(input_path), str(output_dir / "song.mid"))]
    assert result.midi_path == str(output_dir / "song.mid")
    assert result.beat_info == BeatInfo(bpm=132.0)
    assert result.total_notes == 7
    messages = [message for track in mido.MidiFile(result.midi_path).tracks for message in track]
    assert any(message.type == "set_tempo" for message in messages)
    assert any(message.type == "control_change" and message.control == 11 for message in messages)
