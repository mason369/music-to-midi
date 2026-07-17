from pathlib import Path
from types import SimpleNamespace

import pytest
from mido import Message, MetaMessage, MidiFile, MidiTrack

from src.core.multi_stem_separator import STEM_KEYS
from src.core.pipeline import MusicToMidiPipeline
from src.models.data_models import BeatInfo, Config


def _write_backend_midi(path: Path, note: int = 60) -> None:
    midi = MidiFile(type=1, ticks_per_beat=480)
    conductor = MidiTrack()
    conductor.append(MetaMessage("end_of_track", time=0))
    midi.tracks.append(conductor)
    track = MidiTrack()
    track.append(MetaMessage("track_name", name="rare official track", time=0))
    track.append(Message("program_change", program=73, channel=0, time=0))
    track.append(Message("control_change", control=11, value=87, channel=0, time=0))
    track.append(Message("note_on", note=note, velocity=91, channel=0, time=0))
    track.append(Message("note_off", note=note, velocity=0, channel=0, time=1))
    track.append(MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(path))


class _RecordingTranscriber:
    def __init__(self):
        self.calls = []
        self.midi_paths = []

    def transcribe_to_midi(self, audio_path, output_path, progress_callback=None):
        path = Path(audio_path)
        assert path.is_file()
        self.calls.append(path)
        midi_path = Path(output_path)
        _write_backend_midi(midi_path, note=60 + len(self.calls) - 1)
        self.midi_paths.append(midi_path)
        if progress_callback:
            progress_callback(1.0, "transcribed")
        return str(midi_path)


def _run_six_stem_pipeline(monkeypatch, tmp_path, backend, save_separated_tracks):
    audio_path = tmp_path / "original_mix.wav"
    audio_path.write_bytes(b"mix")
    output_dir = tmp_path / "output"
    separator_calls = []
    separated_paths = {}

    class FakeSixStemSeparator:
        @staticmethod
        def is_available():
            return True

        def __init__(self, **_kwargs):
            pass

        def separate(self, audio_path, output_dir, progress_callback=None):
            separator_calls.append(Path(audio_path))
            output_root = Path(output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            for stem_name in STEM_KEYS:
                stem_path = output_root / f"original_mix_{stem_name}.wav"
                stem_path.write_bytes(stem_name.encode("utf-8"))
                separated_paths[stem_name] = str(stem_path)
            if progress_callback:
                progress_callback(1.0, "separated")
            return dict(separated_paths)

    config = Config(
        processing_mode="six_stem_split",
        transcription_backend=backend,
        multi_instrument_model=backend,
        save_separated_tracks=save_separated_tracks,
    )
    pipeline = MusicToMidiPipeline(config)
    selected = _RecordingTranscriber()
    not_selected = _RecordingTranscriber()
    if backend == "yourmt3":
        pipeline.yourmt3_transcriber = selected
        pipeline.miros_transcriber = not_selected
    else:
        pipeline.miros_transcriber = selected
        pipeline.yourmt3_transcriber = not_selected

    pipeline.midi_generator = SimpleNamespace(
        generate_from_precise_instruments_v2=lambda **_kwargs: pytest.fail(
            "split routes must keep the backend's official MIDI output"
        )
    )
    merged_inputs = []

    def fake_merge(midi_paths_by_track, output_path):
        merged_inputs.append(dict(midi_paths_by_track))
        path = Path(output_path)
        _write_backend_midi(path, note=72)
        return str(path)

    monkeypatch.setattr(
        "src.core.multi_stem_separator.SixStemSeparator",
        FakeSixStemSeparator,
    )
    monkeypatch.setattr(
        pipeline,
        "_detect_beat_or_raise",
        lambda *_args, **_kwargs: BeatInfo(bpm=120.0),
    )
    monkeypatch.setattr(pipeline, "_require_multi_instrument_available", lambda: None)
    monkeypatch.setattr(pipeline, "_cleanup_multi_instrument_backend", lambda: None)
    monkeypatch.setattr(pipeline, "_merge_stem_midis", fake_merge)

    result = pipeline._process_six_stem_split(str(audio_path), str(output_dir))
    return {
        "audio_path": audio_path,
        "separator_calls": separator_calls,
        "separated_paths": separated_paths,
        "selected": selected,
        "not_selected": not_selected,
        "merged_inputs": merged_inputs,
        "result": result,
    }


@pytest.mark.parametrize("backend", ["yourmt3", "miros"])
def test_six_stem_transcribes_each_real_stem_once_with_selected_backend(
    monkeypatch,
    tmp_path,
    backend,
):
    run = _run_six_stem_pipeline(
        monkeypatch,
        tmp_path,
        backend=backend,
        save_separated_tracks=True,
    )

    expected_stem_paths = [Path(run["separated_paths"][stem_name]) for stem_name in STEM_KEYS]
    assert run["separator_calls"] == [run["audio_path"]]
    assert run["selected"].calls == expected_stem_paths
    assert run["audio_path"] not in run["selected"].calls
    assert run["not_selected"].calls == []
    assert len(run["selected"].midi_paths) == len(STEM_KEYS)
    assert all(path.is_file() for path in run["selected"].midi_paths)
    for path in run["selected"].midi_paths:
        midi = MidiFile(str(path))
        messages = [message for track in midi.tracks for message in track]
        assert any(
            message.type == "program_change" and message.program == 73 for message in messages
        )
        assert any(
            message.type == "control_change" and message.control == 11 for message in messages
        )
        assert any(message.type == "note_off" and message.time == 1 for message in messages)
    assert list(run["merged_inputs"][0]) == list(STEM_KEYS)
    assert set(run["result"].stem_midi_paths) == set(STEM_KEYS)
    assert run["result"].total_notes == len(STEM_KEYS)
    assert run["result"].beat_info == BeatInfo(bpm=120.0)
    assert run["result"].separated_audio == run["separated_paths"]


def test_six_stem_deletes_real_wavs_only_after_all_midi_outputs_exist(
    monkeypatch,
    tmp_path,
):
    original_unlink = Path.unlink
    unlink_states = []
    run_state = {}

    def checked_unlink(path, *args, **kwargs):
        if path.suffix.lower() == ".wav" and "run" in run_state:
            run = run_state["run"]
            unlink_states.append(
                (
                    len(run["transcriber"].midi_paths),
                    all(midi_path.is_file() for midi_path in run["transcriber"].midi_paths),
                    bool(run["merged_inputs"]),
                )
            )
        return original_unlink(path, *args, **kwargs)

    # The pipeline reaches unlink during the helper call, so expose live state through
    # objects populated before finalization rather than replacing the cleanup method.
    monkeypatch.setattr(Path, "unlink", checked_unlink)

    # Capture the state object before cleanup by wrapping only the merge operation.
    audio_path = tmp_path / "original_mix.wav"
    audio_path.write_bytes(b"mix")
    output_dir = tmp_path / "output"
    separated_paths = {}

    class FakeSixStemSeparator:
        @staticmethod
        def is_available():
            return True

        def __init__(self, **_kwargs):
            pass

        def separate(self, **_kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            for stem_name in STEM_KEYS:
                path = output_dir / f"original_mix_{stem_name}.wav"
                path.write_bytes(b"stem")
                separated_paths[stem_name] = str(path)
            return dict(separated_paths)

    pipeline = MusicToMidiPipeline(
        Config(
            processing_mode="six_stem_split",
            transcription_backend="yourmt3",
            save_separated_tracks=False,
        )
    )
    transcriber = _RecordingTranscriber()
    merged_inputs = []
    pipeline.yourmt3_transcriber = transcriber
    pipeline.midi_generator = SimpleNamespace(
        generate_from_precise_instruments_v2=lambda **_kwargs: pytest.fail(
            "split routes must not regenerate backend MIDI"
        )
    )

    def fake_merge(midi_paths_by_track, output_path):
        merged_inputs.append(dict(midi_paths_by_track))
        path = Path(output_path)
        _write_backend_midi(path, note=72)
        run_state["run"] = {
            "transcriber": transcriber,
            "merged_inputs": merged_inputs,
        }
        return str(path)

    monkeypatch.setattr(
        "src.core.multi_stem_separator.SixStemSeparator",
        FakeSixStemSeparator,
    )
    monkeypatch.setattr(
        pipeline,
        "_detect_beat_or_raise",
        lambda *_args, **_kwargs: BeatInfo(bpm=120.0),
    )
    monkeypatch.setattr(pipeline, "_require_multi_instrument_available", lambda: None)
    monkeypatch.setattr(pipeline, "_cleanup_multi_instrument_backend", lambda: None)
    monkeypatch.setattr(pipeline, "_merge_stem_midis", fake_merge)

    result = pipeline._process_six_stem_split(str(audio_path), str(output_dir))

    assert result.separated_audio is None
    assert len(unlink_states) == len(STEM_KEYS)
    assert all(state == (len(STEM_KEYS), True, True) for state in unlink_states)
    assert all(not Path(path).exists() for path in separated_paths.values())
    assert all(path.is_file() for path in transcriber.midi_paths)
    assert Path(result.merged_midi_path).is_file()
