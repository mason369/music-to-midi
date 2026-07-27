from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import types
import wave
from pathlib import Path

import mido
import numpy as np
import pretty_midi
import pytest
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog, QStyle, QStyleOptionSlider

from src.core import muscriptor_result_assets
from src.core.manual_midi import (
    MIDI_ROUTE_MIROS,
    MIDI_ROUTE_MUSCRIPTOR,
    MIDI_ROUTE_MUSCRIPTOR_MEDIUM,
    MIDI_ROUTE_MUSCRIPTOR_SMALL,
    build_manual_midi_config,
)
from src.core.muscriptor_midi import validate_muscriptor_midi_constraint
from src.core.muscriptor_result_assets import (
    MuscriptorPreviewAssets,
    MuscriptorRollNote,
    prepare_muscriptor_preview_assets,
    read_midi_roll_notes,
    read_muscriptor_roll_notes,
)
from src.core.muscriptor_transcriber import MuscriptorTranscriber
from src.gui.web.muscriptor_result_runtime import (
    MUSCRIPTOR_RESULT_CSS,
    MUSCRIPTOR_RESULT_JS,
    build_muscriptor_result_html,
)
from src.gui.widgets.muscriptor_instrument_selector import (
    MuscriptorInstrumentSelector,
)
from src.gui.widgets.muscriptor_result import (
    MuscriptorResultWidget,
    _ChunkProgressEstimator,
    _PianoRollCanvas,
    _SmoothPlaybackClock,
    _export_midi_with_bpm,
)
from src.i18n.translator import t
from src.models.data_models import Config, MultiInstrumentModel, MuscriptorModel, ProcessingResult
from src.models.muscriptor_instruments import muscriptor_instrument_label
from src.utils import muscriptor_downloader
from src.utils.muscriptor_runtime import configure_muscriptor_kv_cache


def _midi_bytes(program: int = 0, *, drum: bool = False) -> bytes:
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    channel = 9 if drum else 0
    if not drum:
        track.append(mido.Message("program_change", channel=channel, program=program))
    track.append(mido.Message("note_on", channel=channel, note=60, velocity=90))
    track.append(mido.Message("note_off", channel=channel, note=60, velocity=0, time=120))
    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()


def _silent_wav(path: Path, seconds: float = 1.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(b"\x00\x00" * int(44_100 * seconds))
    return path


def _tone_wav(
    path: Path,
    seconds: float = 1.0,
    *,
    amplitude: float = 0.1,
    frequency: float = 440.0,
) -> Path:
    samples = int(round(44_100 * seconds))
    time_axis = np.arange(samples, dtype="float32") / 44_100.0
    audio = amplitude * np.sin(2.0 * np.pi * frequency * time_axis)
    sf.write(path, audio, 44_100)
    return path


def test_config_round_trip_preserves_canonical_muscriptor_constraint():
    config = Config(
        transcription_backend=MultiInstrumentModel.MUSCRIPTOR.value,
        multi_instrument_model=MultiInstrumentModel.MUSCRIPTOR.value,
        muscriptor_instruments=["acoustic_piano", "drums", "acoustic_piano"],
    )

    restored = Config.from_dict(config.to_dict())

    assert restored.get_effective_multi_instrument_model() == "muscriptor"
    assert restored.muscriptor_instruments == ["acoustic_piano", "drums"]


def test_all_muscriptor_sizes_have_distinct_pinned_artifacts_and_manual_routes():
    artifacts = muscriptor_downloader.MUSCRIPTOR_ARTIFACTS

    assert set(artifacts) == {"small", "medium", "large"}
    assert len({artifact.repo_id for artifact in artifacts.values()}) == 3
    assert len({artifact.revision for artifact in artifacts.values()}) == 3
    assert all(len(artifact.model_sha256) == 64 for artifact in artifacts.values())
    assert all(len(artifact.config_sha256) == 64 for artifact in artifacts.values())

    route_expectations = {
        MIDI_ROUTE_MUSCRIPTOR: MuscriptorModel.LARGE.value,
        MIDI_ROUTE_MUSCRIPTOR_MEDIUM: MuscriptorModel.MEDIUM.value,
        MIDI_ROUTE_MUSCRIPTOR_SMALL: MuscriptorModel.SMALL.value,
    }
    for route, expected_size in route_expectations.items():
        config = build_manual_midi_config(Config(), route)
        assert config.muscriptor_model == expected_size
        assert config.get_effective_multi_instrument_model() == "muscriptor"


def test_config_round_trip_preserves_model_size_and_authoritative_bpm():
    restored = Config.from_dict(
        Config(
            transcription_backend=MultiInstrumentModel.MUSCRIPTOR.value,
            muscriptor_model=MuscriptorModel.SMALL.value,
            custom_bpm=10.0,
        ).to_dict()
    )

    assert restored.muscriptor_model == "small"
    assert restored.custom_bpm == pytest.approx(10.0)


def test_muscriptor_checkpoint_hash_is_reused_only_while_snapshot_is_unchanged(
    tmp_path: Path,
    monkeypatch,
):
    weights = tmp_path / muscriptor_downloader.MUSCRIPTOR_MODEL_FILENAME
    config = tmp_path / muscriptor_downloader.MUSCRIPTOR_CONFIG_FILENAME
    weights.write_bytes(b"weights")
    config.write_bytes(b"config")
    paths = {
        muscriptor_downloader.MUSCRIPTOR_MODEL_FILENAME: weights,
        muscriptor_downloader.MUSCRIPTOR_CONFIG_FILENAME: config,
    }
    validations: list[Path] = []
    monkeypatch.setattr(
        muscriptor_downloader,
        "_cached_file",
        lambda _artifact, filename: paths.get(filename),
    )
    monkeypatch.setattr(
        muscriptor_downloader,
        "validate_file_identity",
        lambda path, **_kwargs: validations.append(Path(path)),
    )
    muscriptor_downloader._validate_exact_muscriptor_files.cache_clear()

    muscriptor_downloader.get_cached_muscriptor_paths(validate_hashes=True)
    muscriptor_downloader.get_cached_muscriptor_paths(validate_hashes=True)
    assert validations == [weights, config]

    current = weights.stat().st_mtime_ns
    os.utime(weights, ns=(current + 1_000_000_000, current + 1_000_000_000))
    muscriptor_downloader.get_cached_muscriptor_paths(validate_hashes=True)
    assert validations == [weights, config, weights, config]
    muscriptor_downloader._validate_exact_muscriptor_files.cache_clear()


def test_muscriptor_organ_label_uses_the_musical_instrument_term():
    assert muscriptor_instrument_label("organ", "zh_CN") == "风琴"
    assert muscriptor_instrument_label("organ", "en_US") == "organ"


def test_manual_route_owns_constraint_and_other_routes_clear_stale_selection():
    base = Config(muscriptor_instruments=["drums"])

    muscriptor = build_manual_midi_config(
        base,
        MIDI_ROUTE_MUSCRIPTOR,
        muscriptor_instruments=["acoustic_piano"],
    )
    miros = build_manual_midi_config(base, MIDI_ROUTE_MIROS)

    assert muscriptor.muscriptor_instruments == ["acoustic_piano"]
    assert miros.muscriptor_instruments == []


def test_final_midi_constraint_rejects_unselected_program_and_drums(tmp_path: Path):
    guitar = tmp_path / "guitar.mid"
    guitar.write_bytes(_midi_bytes(26))
    drums = tmp_path / "drums.mid"
    drums.write_bytes(_midi_bytes(drum=True))

    with pytest.raises(RuntimeError, match="program 26"):
        validate_muscriptor_midi_constraint(guitar, ["acoustic_piano"])
    with pytest.raises(RuntimeError, match="drum note"):
        validate_muscriptor_midi_constraint(drums, ["acoustic_piano"])


def test_final_midi_constraint_accepts_selected_program(tmp_path: Path):
    piano = tmp_path / "piano.mid"
    piano.write_bytes(_midi_bytes(0))

    assert validate_muscriptor_midi_constraint(piano, ["acoustic_piano"]) == piano


def _install_fake_event_module(monkeypatch):
    events = types.ModuleType("muscriptor.events")

    class ProgressEvent:
        def __init__(self, completed: int, total: int):
            self.completed = completed
            self.total = total

    class NoteStartEvent:
        def __init__(self, index: int, instrument: str, pitch: int, start_time: float):
            self.index = index
            self.instrument = instrument
            self.pitch = pitch
            self.start_time = start_time

    class NoteEndEvent:
        def __init__(self, start_event_index: int, start_event, end_time: float):
            self.start_event_index = start_event_index
            self.start_event = start_event
            self.end_time = end_time

    events.ProgressEvent = ProgressEvent
    events.NoteStartEvent = NoteStartEvent
    events.NoteEndEvent = NoteEndEvent
    package = types.ModuleType("muscriptor")
    package.events = events
    monkeypatch.setitem(sys.modules, "muscriptor", package)
    monkeypatch.setitem(sys.modules, "muscriptor.events", events)
    return ProgressEvent, NoteStartEvent, NoteEndEvent


def test_muscriptor_load_model_keeps_fp32_parameters_and_verified_fp16_autocast(
    tmp_path: Path,
    monkeypatch,
):
    import torch

    inner_model = torch.nn.Linear(4, 4)
    inner_model.autocast = types.SimpleNamespace(
        enabled=True,
        dtype=torch.float16,
    )
    loaded_model = types.SimpleNamespace(_model=inner_model)
    captured = {}

    class FakeTranscriptionModel:
        @classmethod
        def load_model(cls, *, weights_path, device):
            captured["weights_path"] = Path(weights_path)
            captured["device"] = device
            return loaded_model

    fake_package = types.ModuleType("muscriptor")
    fake_package.TranscriptionModel = FakeTranscriptionModel
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    precision_plan = types.SimpleNamespace(
        autocast=True,
        compute_dtype="float16",
        device="cuda:0",
        capabilities=types.SimpleNamespace(device_name="Test CUDA GPU"),
        torch_dtype=lambda _torch_module: torch.float16,
    )

    monkeypatch.setitem(sys.modules, "muscriptor", fake_package)
    monkeypatch.setattr(
        MuscriptorTranscriber,
        "_runtime_unavailable_reason",
        staticmethod(lambda: ""),
    )
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.get_cached_muscriptor_paths",
        lambda *_args, **_kwargs: (weights, config_path),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.select_inference_precision",
        lambda *_args, **_kwargs: precision_plan,
    )
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.configure_torch_precision",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.log_precision_plan",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.configure_muscriptor_kv_cache",
        lambda *_args, **_kwargs: 14,
    )

    transcriber = MuscriptorTranscriber(Config(use_gpu=True, gpu_device=0))

    assert transcriber.load_model() is loaded_model
    assert captured == {"weights_path": weights, "device": "cuda:0"}
    assert {parameter.dtype for parameter in inner_model.parameters()} == {torch.float32}
    assert transcriber._runtime_details == {
        "type": "runtime",
        "model": "MuScriptor-large",
        "device": "cuda:0",
        "gpu": "Test CUDA GPU",
        "compute_dtype": "float16",
        "kv_cache_dtype": "float16",
        "kv_cache_reused_layers": 14,
        "batch_size": 1,
        "prelude_forcing": True,
    }


def test_muscriptor_fp16_kv_cache_is_reused_without_changing_fp32_parameters():
    import torch
    from muscriptor.modules.streaming import init_states
    from muscriptor.modules.transformer import StreamingMultiheadAttention

    model = torch.nn.Sequential(
        StreamingMultiheadAttention(
            embed_dim=8,
            num_heads=2,
            device="cpu",
            dtype=torch.float32,
        )
    )

    assert (
        configure_muscriptor_kv_cache(
            model,
            compute_dtype="float16",
            torch_module=torch,
        )
        == 1
    )
    first = init_states(model, batch_size=1, sequence_length=32)
    second = init_states(model, batch_size=1, sequence_length=32)
    first_state = next(state for state in first.values() if "cache" in state)
    second_state = next(state for state in second.values() if "cache" in state)

    assert first_state["cache"].dtype == torch.float16
    assert first_state["cache"].data_ptr() == second_state["cache"].data_ptr()
    assert first_state["offset"].item() == 0
    assert second_state["offset"].item() == 0
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


def test_muscriptor_kv_cache_requires_verified_fp16_compute():
    import torch

    with pytest.raises(ValueError, match="requires verified FP16 compute"):
        configure_muscriptor_kv_cache(
            torch.nn.Linear(2, 2),
            compute_dtype="float32",
            torch_module=torch,
        )


def test_transcriber_passes_official_hard_mask_and_publishes_only_valid_midi(
    tmp_path: Path, monkeypatch
):
    ProgressEvent, NoteStartEvent, NoteEndEvent = _install_fake_event_module(monkeypatch)
    start = NoteStartEvent(0, "acoustic_piano", 60, 0.1)

    class FakeModel:
        def __init__(self):
            self.kwargs = None

        def transcribe(self, source, **kwargs):
            assert Path(source).is_file()
            self.kwargs = kwargs
            return iter([ProgressEvent(1, 1), start, NoteEndEvent(0, start, 0.8)])

        def events_to_midi_bytes(self, events):
            assert len(list(events)) == 3
            return _midi_bytes(0)

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    output = tmp_path / "output.mid"
    model = FakeModel()
    config = Config(
        use_gpu=False,
        transcription_backend="muscriptor",
        multi_instrument_model="muscriptor",
        muscriptor_instruments=["acoustic_piano"],
    )
    transcriber = MuscriptorTranscriber(config)
    transcriber._model = model

    assert transcriber.transcribe_to_midi(str(audio), str(output)) == str(output.resolve())
    assert output.is_file()
    assert model.kwargs == {
        "instruments": ["acoustic_piano"],
        "use_sampling": False,
        "batch_size": 1,
        "beam_size": 1,
        "prelude_forcing": True,
    }


def test_transcriber_emits_verified_runtime_details_and_uses_inference_mode(
    tmp_path: Path,
    monkeypatch,
):
    import torch

    ProgressEvent, _NoteStartEvent, _NoteEndEvent = _install_fake_event_module(monkeypatch)
    inference_states: list[bool] = []

    class FakeModel:
        def transcribe(self, _source, **_kwargs):
            inference_states.append(torch.is_inference_mode_enabled())
            return iter([ProgressEvent(1, 1)])

        def events_to_midi_bytes(self, events):
            assert len(list(events)) == 1
            return _midi_bytes()

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    output = tmp_path / "output.mid"
    transcriber = MuscriptorTranscriber(Config(use_gpu=False))
    transcriber._model = FakeModel()
    transcriber._runtime_details = {
        "type": "runtime",
        "model": "MuScriptor-small",
        "device": "cuda:0",
        "gpu": "Test GPU",
        "compute_dtype": "float16",
        "kv_cache_dtype": "float16",
        "kv_cache_reused_layers": 14,
        "batch_size": 1,
        "prelude_forcing": True,
    }
    received: list[dict[str, object]] = []
    transcriber.set_event_callback(received.append)

    transcriber.transcribe_to_midi(str(audio), str(output))

    assert inference_states == [True]
    assert received[0] == transcriber._runtime_details
    assert received[1] == {"type": "progress", "completed": 1, "total": 1}


def test_transcriber_refuses_backend_event_outside_selected_instruments(
    tmp_path: Path, monkeypatch
):
    _ProgressEvent, NoteStartEvent, _NoteEndEvent = _install_fake_event_module(monkeypatch)

    class ViolatingModel:
        def transcribe(self, _source, **_kwargs):
            return iter([NoteStartEvent(0, "clean_electric_guitar", 60, 0.0)])

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    output = tmp_path / "must-not-exist.mid"
    config = Config(
        use_gpu=False,
        transcription_backend="muscriptor",
        multi_instrument_model="muscriptor",
        muscriptor_instruments=["acoustic_piano"],
    )
    transcriber = MuscriptorTranscriber(config)
    transcriber._model = ViolatingModel()

    with pytest.raises(RuntimeError, match="constraint violation"):
        transcriber.transcribe_to_midi(str(audio), str(output))
    assert not output.exists()


def test_transcriber_batches_dense_note_events_before_each_progress_anchor(
    tmp_path: Path, monkeypatch
):
    ProgressEvent, NoteStartEvent, NoteEndEvent = _install_fake_event_module(monkeypatch)
    first = NoteStartEvent(0, "acoustic_piano", 60, 0.1)
    second = NoteStartEvent(1, "drums", 36, 0.2)

    class FakeModel:
        def transcribe(self, _source, **_kwargs):
            return iter(
                [
                    ProgressEvent(0, 2),
                    first,
                    NoteEndEvent(0, first, 0.8),
                    ProgressEvent(1, 2),
                    second,
                    NoteEndEvent(1, second, 0.3),
                    ProgressEvent(2, 2),
                ]
            )

        def events_to_midi_bytes(self, events):
            assert len(list(events)) == 7
            return _midi_bytes(0)

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    output = tmp_path / "output.mid"
    transcriber = MuscriptorTranscriber(Config(use_gpu=False))
    transcriber._model = FakeModel()
    received: list[dict[str, object]] = []
    transcriber.set_event_callback(received.append)

    transcriber.transcribe_to_midi(str(audio), str(output))

    assert [event["type"] for event in received] == [
        "progress",
        "note_batch",
        "progress",
        "note_batch",
        "progress",
    ]
    assert received[1]["notes"] == [
        {
            "index": 0,
            "instrument": "acoustic_piano",
            "pitch": 60,
            "start_time": 0.1,
            "end_time": 0.8,
            "program": 0,
            "is_drum": False,
        }
    ]
    assert received[3]["notes"][0]["instrument"] == "drums"


def test_roll_parser_uses_final_midi_programs_and_tempo(tmp_path: Path):
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    midi.tracks.append(tempo)
    piano = mido.MidiTrack()
    piano.append(mido.MetaMessage("track_name", name="acoustic piano", time=0))
    piano.append(mido.Message("program_change", program=0, channel=0, time=0))
    piano.append(mido.Message("note_on", note=64, velocity=77, channel=0, time=0))
    piano.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=480))
    midi.tracks.append(piano)
    path = tmp_path / "roll.mid"
    midi.save(path)

    notes = read_muscriptor_roll_notes(path)

    assert len(notes) == 1
    assert notes[0].instrument == "acoustic_piano"
    assert notes[0].pitch == 64
    assert notes[0].velocity == 77
    assert notes[0].start == pytest.approx(0.0)
    assert notes[0].end == pytest.approx(0.5)


def test_official_style_chunk_progress_interpolates_and_estimates_eta():
    estimator = _ChunkProgressEstimator(started_at=0.0)

    estimator.on_anchor(1, 10, now=4.0)
    assert estimator.fraction(now=4.0) == pytest.approx(0.1)
    assert estimator.eta_seconds(now=4.0) is None

    estimator.on_anchor(2, 10, now=6.0)
    assert estimator.fraction(now=7.0) == pytest.approx(0.25)
    assert estimator.eta_seconds(now=7.0) == pytest.approx(15.0)
    assert estimator.elapsed_seconds(now=7.0) == pytest.approx(7.0)


def test_chunk_progress_accepts_consecutive_anchors_in_same_clock_tick():
    estimator = _ChunkProgressEstimator(started_at=100.0)

    estimator.on_anchor(22, 42, now=101.0)
    estimator.on_anchor(23, 42, now=101.0)

    assert estimator.completed == 23
    assert estimator.total == 42
    assert estimator.ema_chunk_seconds is None
    assert estimator.fraction(now=101.0) == pytest.approx(23 / 42)
    assert estimator.eta_seconds(now=101.0) is None

    estimator.on_anchor(24, 42, now=101.25)

    assert estimator.ema_chunk_seconds == pytest.approx(0.25)
    assert estimator.eta_seconds(now=101.25) == pytest.approx(4.5)


def test_chunk_progress_still_rejects_a_clock_that_moves_backwards():
    estimator = _ChunkProgressEstimator(started_at=100.0)
    estimator.on_anchor(22, 42, now=101.0)

    with pytest.raises(ValueError, match="duration sample"):
        estimator.on_anchor(23, 42, now=100.9)


def test_playhead_clock_interpolates_media_samples_and_stops_at_bounded_lead():
    clock = _SmoothPlaybackClock(max_lead_ms=120.0)
    clock.reset(8_300, now=0.0)

    assert clock.sample(8_300, now=0.016) == pytest.approx(8_316)
    assert clock.sample(8_270, now=0.032) == pytest.approx(8_332)
    assert clock.sample(8_300, now=0.200) == pytest.approx(8_420)
    assert clock.sample(8_300, now=1.000) == pytest.approx(8_420)
    assert clock.sample(8_500, now=1.010) == pytest.approx(8_435)
    assert clock.sample(8_500, now=1.026) == pytest.approx(8_459)


def test_playhead_repaints_fractional_frame_motion(monkeypatch):
    app = QApplication.instance() or QApplication([])
    canvas = _PianoRollCanvas()
    canvas.set_pixels_per_second(46.0)
    updates: list[tuple[object, ...]] = []
    monkeypatch.setattr(canvas, "update", lambda *args: updates.append(args))

    canvas.set_position(0.005)

    assert updates
    assert canvas.position == pytest.approx(0.005)
    canvas.close()
    app.processEvents()


def test_preview_assets_render_real_per_instrument_sources(tmp_path: Path, monkeypatch):
    executable = tmp_path / "fluidsynth.exe"
    soundfont = tmp_path / "MuseScore_General.sf2"
    executable.write_bytes(b"runtime")
    soundfont.write_bytes(b"soundfont")
    rendered: list[tuple[Path, Path]] = []

    monkeypatch.setattr(
        muscriptor_result_assets,
        "get_fluidsynth_executable",
        lambda: executable,
    )
    monkeypatch.setattr(
        muscriptor_result_assets,
        "download_muscriptor_soundfont",
        lambda printer: soundfont,
    )

    def fake_synthesize(_executable, _soundfont, midi_path, output_path, cancel_check=None):
        assert cancel_check is not None and not cancel_check()
        assert Path(midi_path).is_file()
        _tone_wav(Path(output_path), 5.0, amplitude=0.01)
        rendered.append((Path(midi_path), Path(output_path)))

    monkeypatch.setattr(muscriptor_result_assets, "_synthesize", fake_synthesize)
    notes = (
        MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 0.5),
        MuscriptorRollNote("drums", 36, 100, 0.2, 0.4),
    )

    reference = _tone_wav(tmp_path / "reference.wav", 5.0, amplitude=0.2)
    assets = prepare_muscriptor_preview_assets(
        notes,
        5.0,
        tmp_path / "preview",
        reference_audio_path=reference,
    )

    assert isinstance(assets, MuscriptorPreviewAssets)
    assert assets.duration == pytest.approx(5.0)
    assert set(assets.instrument_wavs) == {"acoustic_piano", "drums"}
    assert assets.transcription_wav.is_file()
    assert sf.info(assets.transcription_wav).duration == pytest.approx(5.0, abs=1 / 44_100)
    live_mix, _sample_rate = sf.read(assets.transcription_wav, dtype="float32")
    reference_audio, _sample_rate = sf.read(reference, dtype="float32")
    assert np.sqrt(np.mean(live_mix**2)) == pytest.approx(
        np.sqrt(np.mean(reference_audio**2)),
        rel=0.03,
    )
    assert np.max(np.abs(live_mix)) <= 0.951
    assert assets.midi_gain_db > 10.0
    assert len(rendered) == 2
    assert all(
        path.is_file() and path.stat().st_size > 0 for path in assets.instrument_wavs.values()
    )


def test_preview_assets_round_up_non_frame_aligned_note_boundary(
    tmp_path: Path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    executable = tmp_path / "fluidsynth.exe"
    soundfont = tmp_path / "MuseScore_General.sf2"
    executable.write_bytes(b"runtime")
    soundfont.write_bytes(b"soundfont")
    monkeypatch.setattr(
        muscriptor_result_assets,
        "get_fluidsynth_executable",
        lambda: executable,
    )
    monkeypatch.setattr(
        muscriptor_result_assets,
        "download_muscriptor_soundfont",
        lambda printer: soundfont,
    )

    def fake_synthesize(_executable, _soundfont, _midi_path, output_path, cancel_check=None):
        assert cancel_check is not None and not cancel_check()
        _tone_wav(Path(output_path), 0.1, amplitude=0.01)

    monkeypatch.setattr(muscriptor_result_assets, "_synthesize", fake_synthesize)
    frontier = 1_000.25 / 44_100
    note = MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, frontier)
    reference = _tone_wav(tmp_path / "reference.wav", 0.1, amplitude=0.2)

    assets = prepare_muscriptor_preview_assets(
        (note,),
        frontier,
        tmp_path / "preview",
        reference_audio_path=reference,
    )
    info = sf.info(assets.transcription_wav)

    assert info.frames == 1_001
    assert assets.duration == info.frames / info.samplerate
    assert assets.duration >= note.end
    canvas = _PianoRollCanvas()
    try:
        canvas.set_notes(assets.notes, duration=assets.duration)
        assert canvas._notes == (note,)
    finally:
        canvas.close()
        app.processEvents()


def test_stream_preview_enables_playback_before_final_midi_exists(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 1.0)
    piano = _silent_wav(tmp_path / "acoustic_piano.wav", 1.0)
    note = MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 0.5)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    widget.show()
    app.processEvents()

    try:
        widget._on_preview_ready(
            1,
            MuscriptorPreviewAssets(
                notes=(note,),
                duration=1.0,
                transcription_wav=piano,
                instrument_wavs={"acoustic_piano": piano},
                midi_gain_db=0.0,
            ),
        )
        app.processEvents()

        assert widget._assets is None
        assert widget.midi_path == ""
        assert widget.play_button.isEnabled()
        assert widget.mix_slider.isEnabled()
        assert not widget.stereo_checkbox.isEnabled()
        assert set(widget._normal_sources) == {"acoustic_piano"}
        assert widget._normal_players == {}
        assert len(widget._active_playback_players()) == 2

        widget._toggle_playback()
        assert widget._playing
    finally:
        widget.shutdown()
        widget.close()


def test_stream_progress_queues_latest_completed_notes_and_exposes_timing(
    tmp_path: Path, monkeypatch
):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    app.processEvents()
    queued: list[tuple[int, int]] = []
    monkeypatch.setattr(
        widget, "_queue_preview", lambda completed, total: queued.append((completed, total))
    )

    try:
        widget.add_stream_event(
            {
                "type": "note_end",
                "instrument": "acoustic_piano",
                "pitch": 60,
                "start_time": 0.0,
                "end_time": 0.5,
            }
        )
        widget.add_stream_event({"type": "progress", "completed": 1, "total": 10})

        assert queued == [(1, 10)]
        assert widget.progress_bar.value() >= 100
        assert "1/10" in widget.status_label.text()
        assert " / " in widget.progress_label.text()
    finally:
        widget.shutdown()
        widget.close()


def test_stream_progress_does_not_crash_when_chunks_finish_in_same_clock_tick(
    tmp_path: Path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "same-tick-source.wav", 1.0)
    module = sys.modules[MuscriptorResultWidget.__module__]
    monkeypatch.setattr(module.time, "monotonic", lambda: 101.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    monkeypatch.setattr(widget, "_queue_preview", lambda *_args, **_kwargs: None)

    try:
        widget.add_stream_event({"type": "progress", "completed": 22, "total": 42})
        widget.add_stream_event({"type": "progress", "completed": 23, "total": 42})

        assert widget._progress_completed == 23
        assert widget._progress_total == 42
        assert widget._progress_estimator.ema_chunk_seconds is None
        assert "23/42" in widget.status_label.text()
    finally:
        widget.shutdown()
        widget.close()
        app.processEvents()


def test_ten_minute_large_model_hint_explains_verified_runtime(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "slow-large.wav", 1.0)
    widget = MuscriptorResultWidget(
        str(source),
        ["acoustic_piano"],
        backend_label="MuScriptor-large",
    )
    widget.add_stream_event(
        {
            "type": "runtime",
            "model": "MuScriptor-large",
            "device": "cuda:0",
            "gpu": "Test GPU",
            "compute_dtype": "float16",
            "batch_size": 1,
            "prelude_forcing": True,
        }
    )
    widget._progress_completed = 10
    widget._progress_total = 40
    widget._progress_estimator.on_anchor(10, 40, now=600.0)

    try:
        widget._update_slow_conversion_hint(elapsed=599.9, processed=50.0)
        assert widget.slow_hint_label.isHidden()

        widget._update_slow_conversion_hint(elapsed=600.0, processed=50.0)

        assert not widget.slow_hint_label.isHidden()
        hint = widget.slow_hint_label.text()
        assert "MuScriptor-large" in hint
        assert "Test GPU" in hint
        assert "10/40" in hint
        assert "Medium" in hint
        assert "Small" in hint

        widget.mark_cancelled()
        assert widget.slow_hint_label.isHidden()
    finally:
        widget.shutdown()
        widget.close()
        app.processEvents()


def test_ten_minute_cpu_hint_reports_that_cuda_is_not_active(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "slow-cpu.wav", 1.0)
    widget = MuscriptorResultWidget(
        str(source),
        ["acoustic_piano"],
        backend_label="MuScriptor-small",
    )
    widget.add_stream_event(
        {
            "type": "runtime",
            "model": "MuScriptor-small",
            "device": "cpu",
            "gpu": "CPU",
            "compute_dtype": "float32",
            "batch_size": 1,
            "prelude_forcing": True,
        }
    )
    widget._progress_completed = 2
    widget._progress_total = 40

    try:
        widget._update_slow_conversion_hint(elapsed=600.0, processed=10.0)

        assert not widget.slow_hint_label.isHidden()
        hint = widget.slow_hint_label.text()
        assert "cpu" in hint.lower()
        assert "CUDA" in hint
    finally:
        widget.shutdown()
        widget.close()
        app.processEvents()


def test_stream_preview_rendering_is_throttled_but_extends_near_playback_end(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "preview-throttle.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    widget._stream_notes = [MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 1.0)]
    widget._start_pending_preview = lambda: None

    try:
        widget._queue_preview(1, 100, frontier=5.0)
        assert widget._preview_generation == 1

        widget._queue_preview(2, 100, frontier=10.0)
        widget._queue_preview(3, 100, frontier=15.0)
        assert widget._preview_generation == 1

        widget._queue_preview(4, 100, frontier=20.0)
        assert widget._preview_generation == 2

        widget._preview_duration = 20.0
        widget._position_ms = 18_000
        widget._queue_preview(5, 100, frontier=25.0)
        assert widget._preview_generation == 3
        assert widget._preview_pending is not None
        assert widget._preview_pending[2] == pytest.approx(25.0)
    finally:
        widget.shutdown()
        widget.close()
        app.processEvents()


def test_finished_asset_worker_is_detached_before_result_reset(tmp_path: Path, monkeypatch):
    """A completed QThread must not remain as a deleted wrapper during song changes."""
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 0.1)
    midi = tmp_path / "result.mid"
    midi.write_bytes(_midi_bytes())
    created_workers: list[QThread] = []

    class ImmediateAssetWorker(QThread):
        progress = pyqtSignal(float, str)
        succeeded = pyqtSignal(object)
        failed = pyqtSignal(str)

        def __init__(self, *_args):
            super().__init__(_args[-1])
            created_workers.append(self)

        def run(self):
            pass

        def cancel(self):
            pass

    module = sys.modules[MuscriptorResultWidget.__module__]
    monkeypatch.setattr(module, "_AssetWorker", ImmediateAssetWorker)
    monkeypatch.setattr(module, "read_midi_roll_notes", lambda *_args, **_kwargs: ())

    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    widget.finalize_result(ProcessingResult(midi_path=str(midi)))
    worker = created_workers[0]
    assert worker.wait(2_000)
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()

    assert widget._asset_worker is None
    widget.shutdown()
    widget.shutdown()
    widget.close()


def test_generic_midi_parser_keeps_real_program_identity(tmp_path: Path):
    path = tmp_path / "guitar.mid"
    path.write_bytes(_midi_bytes(program=24))

    notes = read_midi_roll_notes(path)

    assert len(notes) == 1
    assert notes[0].instrument == "gm:024"
    assert notes[0].program == 24
    assert notes[0].is_drum is False


def test_generic_midi_parser_keeps_note_on_channel_state_until_matching_note_off(
    tmp_path: Path,
):
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("program_change", channel=0, program=24, time=0))
    track.append(mido.Message("note_on", channel=0, note=60, velocity=90, time=0))
    track.append(mido.Message("note_on", channel=9, note=36, velocity=100, time=120))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=120))
    track.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=120))
    path = tmp_path / "interleaved-channels.mid"
    midi.save(path)

    notes = read_midi_roll_notes(path)
    piano = next(note for note in notes if note.pitch == 60)
    drums = next(note for note in notes if note.pitch == 36)

    assert (piano.instrument, piano.program, piano.is_drum) == ("gm:024", 24, False)
    assert (drums.instrument, drums.program, drums.is_drum) == ("drums", 0, True)


def test_generic_snapshot_does_not_unlock_playback_before_rendered_audio(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 2.0)
    widget = MuscriptorResultWidget(
        str(source),
        [],
        backend_label="YourMT3+",
        muscriptor_groups=False,
    )
    queued = []
    widget._queue_preview = lambda completed, total, *, frontier=None: queued.append(
        (completed, total, frontier)
    )
    try:
        widget.add_stream_event(
            {
                "type": "snapshot",
                "backend": "YourMT3+",
                "completed": 8,
                "total": 16,
                "frontier_seconds": 1.0,
                "duration_seconds": 2.0,
                "notes": [
                    {
                        "instrument": "gm:024",
                        "program": 24,
                        "is_drum": False,
                        "pitch": 64,
                        "velocity": 100,
                        "start": 0.1,
                        "end": 0.8,
                    }
                ],
            }
        )
        app.processEvents()

        assert queued == [(8, 16, 1.0)]
        assert widget._stream_notes[0].program == 24
        assert widget._detected == ["gm:024"]
        assert widget.roll._notes == ()
        assert not widget.play_button.isEnabled()
        assert "YourMT3+" in widget.status_label.text()

        rendered = _silent_wav(tmp_path / "gm-024.wav", 1.0)
        widget._on_preview_ready(
            1,
            MuscriptorPreviewAssets(
                notes=tuple(widget._stream_notes),
                duration=1.0,
                transcription_wav=rendered,
                instrument_wavs={"gm:024": rendered},
                midi_gain_db=0.0,
            ),
        )
        app.processEvents()
        assert widget.play_button.isEnabled()
        assert widget.roll._notes == tuple(widget._stream_notes)
        assert widget.roll._duration == pytest.approx(1.0)
    finally:
        widget.shutdown()
        widget.close()


def test_new_preview_is_committed_only_after_current_playback_stops(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _tone_wav(tmp_path / "source.wav", 2.0, amplitude=0.1)
    first_audio = _tone_wav(tmp_path / "first.wav", 1.0, amplitude=0.1)
    second_audio = _tone_wav(tmp_path / "second.wav", 2.0, amplitude=0.1)
    first_note = MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 0.5)
    second_note = MuscriptorRollNote("acoustic_piano", 64, 100, 1.0, 1.5)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    try:
        widget._on_preview_ready(
            1,
            MuscriptorPreviewAssets(
                notes=(first_note,),
                duration=1.0,
                transcription_wav=first_audio,
                instrument_wavs={"acoustic_piano": first_audio},
                midi_gain_db=0.0,
            ),
        )
        widget._toggle_playback()
        app.processEvents()

        widget._on_preview_ready(
            2,
            MuscriptorPreviewAssets(
                notes=(first_note, second_note),
                duration=2.0,
                transcription_wav=second_audio,
                instrument_wavs={"acoustic_piano": second_audio},
                midi_gain_db=0.0,
            ),
        )

        assert widget._preview_duration == pytest.approx(1.0)
        assert widget.roll._notes == (first_note,)
        assert widget._deferred_preview is not None

        widget._toggle_playback()
        app.processEvents()

        assert not widget._playing
        assert widget._deferred_preview is None
        assert widget._preview_duration == pytest.approx(2.0)
        assert widget.roll._notes == (first_note, second_note)
        assert all(
            player.playbackState().name != "PlayingState"
            for player in widget._active_playback_players()
        )
    finally:
        widget.shutdown()
        widget.close()


def test_invalid_preview_commit_is_reported_without_escaping_qt_slot(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 1.0)
    rendered = _silent_wav(tmp_path / "preview.wav", 1.0)
    invalid_note = MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 1.01)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])

    try:
        widget._on_preview_ready(
            1,
            MuscriptorPreviewAssets(
                notes=(invalid_note,),
                duration=1.0,
                transcription_wav=rendered,
                instrument_wavs={"acoustic_piano": rendered},
                midi_gain_db=0.0,
            ),
        )
        app.processEvents()

        assert widget._preview_error == "Piano roll contains notes beyond playable audio"
        assert widget._preview_applied_generation == 0
        assert widget.roll._notes == ()
    finally:
        widget.shutdown()
        widget.close()


def test_invalid_deferred_preview_commit_is_reported_without_timer_exception(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 1.0)
    rendered = _silent_wav(tmp_path / "preview.wav", 1.0)
    invalid_note = MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 1.01)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])

    try:
        widget._playing = True
        widget._on_preview_ready(
            1,
            MuscriptorPreviewAssets(
                notes=(invalid_note,),
                duration=1.0,
                transcription_wav=rendered,
                instrument_wavs={"acoustic_piano": rendered},
                midi_gain_db=0.0,
            ),
        )
        assert widget._deferred_preview is not None

        widget._playing = False
        widget._apply_deferred_assets()
        app.processEvents()

        assert widget._deferred_preview is None
        assert widget._preview_error == "Piano roll contains notes beyond playable audio"
        assert widget._preview_applied_generation == 0
        assert widget.roll._notes == ()
    finally:
        widget._playing = False
        widget.shutdown()
        widget.close()


def test_invalid_final_asset_commit_is_reported_without_escaping_qt_slot(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 1.0)
    rendered = _silent_wav(tmp_path / "rendered.wav", 1.0)
    invalid_note = MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 1.01)
    assets = muscriptor_result_assets.MuscriptorPlaybackAssets(
        notes=(invalid_note,),
        duration=1.0,
        transcription_wav=rendered,
        live_transcription_wav=rendered,
        stereo_mix_wav=rendered,
        original_left_wav=rendered,
        transcription_right_wav=rendered,
        instrument_wavs={"acoustic_piano": rendered},
        instrument_right_wavs={"acoustic_piano": rendered},
        midi_gain_db=0.0,
    )
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])

    try:
        widget._on_assets_ready(assets)
        app.processEvents()

        assert widget._assets is None
        assert widget.roll._notes == ()
        assert not widget.play_button.isEnabled()
        assert not widget.stereo_checkbox.isEnabled()
    finally:
        widget.shutdown()
        widget.close()


def test_real_qmedia_preview_rollover_does_not_deadlock_event_loop(tmp_path: Path):
    source = _tone_wav(tmp_path / "source.wav", 2.0, amplitude=0.1)
    first = _tone_wav(tmp_path / "first.wav", 1.0, amplitude=0.1)
    second = _tone_wav(tmp_path / "second.wav", 2.0, amplitude=0.1)
    script = f"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from src.core.muscriptor_result_assets import MuscriptorPreviewAssets, MuscriptorRollNote
from src.gui.widgets.muscriptor_result import MuscriptorResultWidget

app = QApplication([])
note = MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 0.5)
widget = MuscriptorResultWidget({str(source)!r}, ["acoustic_piano"])
widget._on_preview_ready(
    1,
    MuscriptorPreviewAssets(
        notes=(note,),
        duration=1.0,
        transcription_wav={str(first)!r},
        instrument_wavs={{"acoustic_piano": {str(first)!r}}},
        midi_gain_db=0.0,
    ),
)
widget._toggle_playback()

def queue_second_preview():
    widget._on_preview_ready(
        2,
        MuscriptorPreviewAssets(
            notes=(note,),
            duration=2.0,
            transcription_wav={str(second)!r},
            instrument_wavs={{"acoustic_piano": {str(second)!r}}},
            midi_gain_db=0.0,
        ),
    )
    # Drive the preview boundary explicitly without seeking live multimedia
    # backends. QMediaPlayer startup timing depends on the host audio device,
    # which is unrelated to the rollover/deadlock path under test.
    widget._position_ms = 1_000

def verify_rollover():
    if widget._preview_applied_generation != 2:
        raise RuntimeError(
            f"preview rollover did not commit: {{widget._preview_applied_generation}}"
        )
    widget.shutdown()
    widget.close()
    app.quit()

QTimer.singleShot(200, queue_second_preview)
QTimer.singleShot(1800, verify_rollover)
app.exec()
"""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_reference_audio_decode_is_cached_by_file_identity(tmp_path: Path, monkeypatch):
    source = _tone_wav(tmp_path / "source.wav", 0.1)
    original_loader = muscriptor_result_assets._load_mono_44k
    decoded: list[Path] = []

    def recording_loader(path: Path):
        decoded.append(Path(path))
        return original_loader(Path(path))

    muscriptor_result_assets._load_reference_mono_44k_cached.cache_clear()
    monkeypatch.setattr(muscriptor_result_assets, "_load_mono_44k", recording_loader)

    first = muscriptor_result_assets._load_reference_mono_44k(source)
    second = muscriptor_result_assets._load_reference_mono_44k(source)

    assert first is second
    assert decoded == [source.resolve()]


def test_live_bus_scales_each_raw_stem_without_a_second_full_decode(tmp_path: Path, monkeypatch):
    source = _tone_wav(tmp_path / "source.wav", 0.2, amplitude=0.1)
    combined = _tone_wav(tmp_path / "combined.wav", 0.2, amplitude=0.04)
    stems = {
        "acoustic_piano": _tone_wav(tmp_path / "piano.wav", 0.2, amplitude=0.02),
        "drums": _tone_wav(tmp_path / "drums.wav", 0.2, amplitude=0.02),
    }
    original_loader = muscriptor_result_assets._load_mono_44k
    decoded: list[Path] = []

    def recording_loader(path: Path):
        decoded.append(Path(path).resolve())
        return original_loader(Path(path))

    muscriptor_result_assets._load_reference_mono_44k_cached.cache_clear()
    monkeypatch.setattr(muscriptor_result_assets, "_load_mono_44k", recording_loader)
    output_dir = tmp_path / "live"
    output_dir.mkdir()

    buses = muscriptor_result_assets._write_live_playback_buses(
        source,
        stems,
        output_dir,
        combined_source=combined,
        target_duration=0.2,
    )

    assert decoded.count(source.resolve()) == 1
    assert decoded.count(combined.resolve()) == 1
    assert decoded.count(stems["acoustic_piano"].resolve()) == 1
    assert decoded.count(stems["drums"].resolve()) == 1
    assert all(
        sf.info(path).frames == round(0.2 * 44_100) for path in buses.instrument_wavs.values()
    )


def test_live_bus_uses_original_transport_length_and_fades_synth_tail(tmp_path: Path):
    source = _tone_wav(tmp_path / "source.wav", 0.2, amplitude=0.1)
    combined = _tone_wav(tmp_path / "combined-tail.wav", 0.5, amplitude=0.08)
    stem = _tone_wav(tmp_path / "piano-tail.wav", 0.5, amplitude=0.04)
    output_dir = tmp_path / "transport-live"
    output_dir.mkdir()

    buses = muscriptor_result_assets._write_live_playback_buses(
        source,
        {"acoustic_piano": stem},
        output_dir,
        combined_source=combined,
    )

    source_frames = sf.info(source).frames
    assert sf.info(buses.transcription_wav).frames == source_frames
    assert sf.info(buses.instrument_wavs["acoustic_piano"]).frames == source_frames
    combined_audio, _sample_rate = sf.read(buses.transcription_wav, dtype="float32")
    stem_audio, _sample_rate = sf.read(
        buses.instrument_wavs["acoustic_piano"],
        dtype="float32",
    )
    assert abs(float(combined_audio[-1])) <= 1 / 32_768
    assert abs(float(stem_audio[-1])) <= 1 / 32_768


def test_final_playback_transport_covers_notes_beyond_source_audio(
    tmp_path: Path,
    monkeypatch,
):
    executable = tmp_path / "fluidsynth.exe"
    soundfont = tmp_path / "MuseScore_General.sf2"
    executable.write_bytes(b"runtime")
    soundfont.write_bytes(b"soundfont")
    monkeypatch.setattr(
        muscriptor_result_assets,
        "get_fluidsynth_executable",
        lambda: executable,
    )
    monkeypatch.setattr(
        muscriptor_result_assets,
        "download_muscriptor_soundfont",
        lambda printer: soundfont,
    )

    def fake_synthesize(_executable, _soundfont, _midi_path, output_path, cancel_check=None):
        assert cancel_check is not None and not cancel_check()
        _tone_wav(Path(output_path), 2.0, amplitude=0.02)

    monkeypatch.setattr(muscriptor_result_assets, "_synthesize", fake_synthesize)
    midi = mido.MidiFile(type=1, ticks_per_beat=1_000)
    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
    midi.tracks.append(tempo)
    piano = mido.MidiTrack()
    piano.append(mido.Message("program_change", program=0, channel=0, time=0))
    piano.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
    piano.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=1_500))
    midi.tracks.append(piano)
    midi_path = tmp_path / "beyond-source.mid"
    midi.save(midi_path)
    source = _tone_wav(tmp_path / "source.wav", 1.0, amplitude=0.1)

    assets = muscriptor_result_assets.prepare_midi_playback_assets(
        midi_path,
        source,
        tmp_path / "final-assets",
    )
    master = sf.info(assets.live_transcription_wav)

    assert assets.duration >= 1.5
    assert assets.duration == master.frames / master.samplerate
    for path in (
        assets.original_left_wav,
        assets.transcription_right_wav,
        *assets.instrument_wavs.values(),
        *assets.instrument_right_wavs.values(),
    ):
        info = sf.info(path)
        assert (info.frames, info.samplerate) == (master.frames, master.samplerate)


def test_default_midi_monitoring_uses_one_combined_bus_and_real_mutes_use_stems(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    source = _tone_wav(tmp_path / "source.wav", 1.0, amplitude=0.1)
    combined = _tone_wav(tmp_path / "midi-live.wav", 1.0, amplitude=0.1)
    instruments = [f"instrument_{index}" for index in range(6)]
    stems = {
        instrument: _tone_wav(
            tmp_path / f"{instrument}.wav",
            1.0,
            amplitude=0.02,
            frequency=220.0 + index * 20.0,
        )
        for index, instrument in enumerate(instruments)
    }
    note = MuscriptorRollNote(instruments[0], 60, 100, 0.0, 0.5)
    widget = MuscriptorResultWidget(str(source), instruments)
    try:
        widget._on_preview_ready(
            1,
            MuscriptorPreviewAssets(
                notes=(note,),
                duration=1.0,
                transcription_wav=combined,
                instrument_wavs=stems,
                midi_gain_db=0.0,
            ),
        )
        app.processEvents()

        assert len(widget._all_playback_players()) == 2
        assert len(widget._active_playback_players()) == 2
        assert widget._midi_normal is not None
        assert widget._midi_normal[1].volume() == pytest.approx(0.75)
        assert all(output.volume() == 0.0 for _player, output in widget._normal_players.values())

        widget._muted = {instruments[0]}
        widget._apply_mix()
        assert len(widget._all_playback_players()) == 8
        # The silent combined MIDI bus stays active as the authoritative
        # transport clock: MIDI + original + five audible instrument stems.
        assert len(widget._active_playback_players()) == 7
        assert widget._midi_normal[1].volume() == pytest.approx(0.0)
        assert widget._normal_players[instruments[0]][1].volume() == pytest.approx(0.0)
        assert widget._normal_players[instruments[1]][1].volume() == pytest.approx(0.75)
    finally:
        widget.shutdown()
        widget.close()


def test_playhead_tick_never_reseeks_audio_players(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    app.processEvents()

    class FakePlayer:
        def __init__(self, position: int):
            self._position = position
            self.set_positions: list[int] = []

        def position(self):
            return self._position

        def setPosition(self, position):
            self.set_positions.append(position)

        def mediaStatus(self):
            from PyQt6.QtMultimedia import QMediaPlayer

            return QMediaPlayer.MediaStatus.LoadedMedia

    master = FakePlayer(1_234)
    slave = FakePlayer(0)
    midi_pair = widget._midi_normal
    original_pair = widget._original_normal
    original_players = widget._players
    try:
        widget._midi_normal = (master, object())
        widget._original_normal = (slave, object())
        widget._players = [master, slave]
        widget._playing = True
        widget._finalizing = True
        widget._preview_duration = 0.0
        widget._set_playback_duration(10.0)

        widget._tick()

        assert widget._position_ms == 1_234
        assert widget.playback_slider.value() == 1_234
        assert slave.set_positions == [1_234]
    finally:
        widget._playing = False
        widget._midi_normal = midi_pair
        widget._original_normal = original_pair
        widget._players = original_players
        widget.shutdown()
        widget.close()


def test_end_of_media_pauses_every_player_without_automatic_rewind(tmp_path: Path):
    from PyQt6.QtMultimedia import QMediaPlayer

    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    app.processEvents()

    class FakePlayer:
        def __init__(self, position: int, duration: int, status):
            self._position = position
            self._duration = duration
            self._status = status
            self.pause_calls = 0
            self.set_positions: list[int] = []

        def position(self):
            return self._position

        def duration(self):
            return self._duration

        def setPosition(self, position):
            self.set_positions.append(position)

        def mediaStatus(self):
            return self._status

        def pause(self):
            self.pause_calls += 1

    master = FakePlayer(990, 1_000, QMediaPlayer.MediaStatus.EndOfMedia)
    midi = FakePlayer(950, 1_000, QMediaPlayer.MediaStatus.LoadedMedia)
    midi_pair = widget._midi_normal
    original_players = widget._players
    try:
        widget._midi_normal = (master, object())
        widget._players = [master, midi]
        widget._playing = True
        widget._finalizing = True
        widget._preview_duration = 0.0
        widget._position_ms = 990

        widget._tick()

        assert widget._playing is False
        assert widget._playback_finished is True
        assert widget._position_ms == 1_000
        assert master.pause_calls == 1
        assert midi.pause_calls == 1
        assert master.set_positions == []
        assert midi.set_positions == []
        assert widget.clock_label.text() == "1.0s"

        widget._tick()
        assert master.pause_calls == 1
        assert midi.pause_calls == 1
        assert master.set_positions == []
        assert midi.set_positions == []
    finally:
        widget._playing = False
        widget._midi_normal = midi_pair
        widget._players = original_players
        widget.shutdown()
        widget.close()


def test_stream_end_of_media_schedules_buffered_preview_application(tmp_path: Path):
    from PyQt6.QtMultimedia import QMediaPlayer

    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "stream-end-source.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    app.processEvents()

    class FakePlayer:
        def position(self):
            return 990

        def duration(self):
            return 990

        def mediaStatus(self):
            return QMediaPlayer.MediaStatus.EndOfMedia

        def pause(self):
            pass

    master = FakePlayer()
    midi_pair = widget._midi_normal
    original_players = widget._players
    scheduled: list[bool] = []
    try:
        widget._midi_normal = (master, object())
        widget._players = [master]
        widget._playing = True
        widget._finalizing = False
        widget._preview_duration = 1.0
        widget._position_ms = 990
        widget._schedule_deferred_assets = lambda: scheduled.append(True)

        widget._tick()

        assert widget._playing is False
        assert widget._playback_finished is True
        assert scheduled == [True]
    finally:
        widget._playing = False
        widget._midi_normal = midi_pair
        widget._players = original_players
        widget.shutdown()
        widget.close()


def test_deferred_preview_waits_for_media_backend_to_finish_pausing(tmp_path: Path):
    from PyQt6.QtMultimedia import QMediaPlayer

    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "pause-transition.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    app.processEvents()

    class FakePlayer:
        def __init__(self):
            self.state = QMediaPlayer.PlaybackState.PlayingState

        def playbackState(self):
            return self.state

    player = FakePlayer()
    original_players = widget._players
    applied: list[bool] = []
    try:
        widget._players = [player]
        widget._playing = False
        widget._apply_deferred_assets = lambda: applied.append(True)

        widget._schedule_deferred_assets()
        app.processEvents()

        assert applied == []
        assert widget._deferred_apply_scheduled is True

        player.state = QMediaPlayer.PlaybackState.PausedState
        widget._on_player_playback_state_changed(player.state)

        assert applied == [True]
        assert widget._deferred_apply_scheduled is False
    finally:
        widget._players = original_players
        widget.shutdown()
        widget.close()


def test_preview_replacement_silences_retired_midi_output_immediately(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 1.0)
    first_midi = _tone_wav(tmp_path / "first-midi.wav", 1.0)
    second_midi = _tone_wav(tmp_path / "second-midi.wav", 1.0, frequency=330.0)
    note = MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 0.5)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    try:
        widget._on_preview_ready(
            1,
            MuscriptorPreviewAssets(
                notes=(note,),
                duration=1.0,
                transcription_wav=first_midi,
                instrument_wavs={"acoustic_piano": first_midi},
                midi_gain_db=0.0,
            ),
        )
        app.processEvents()
        assert widget._midi_normal is not None
        retired_player, retired_output = widget._midi_normal
        retired_player.play()

        widget._on_preview_ready(
            2,
            MuscriptorPreviewAssets(
                notes=(note,),
                duration=1.0,
                transcription_wav=second_midi,
                instrument_wavs={"acoustic_piano": second_midi},
                midi_gain_db=0.0,
            ),
        )

        assert retired_output.isMuted() is True
    finally:
        app.processEvents()
        widget.shutdown()
        widget.close()


def test_follow_playhead_ignores_media_clock_rollback_and_explicit_seek_can_move_left(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    widget.resize(1000, 720)
    widget.show()
    widget.roll.set_notes(
        (MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 0.5),),
        duration=30.0,
    )
    app.processEvents()

    class FakePlayer:
        def __init__(self, positions: list[int]):
            self._positions = iter(positions)
            self._position = 0
            self.set_positions: list[int] = []

        def position(self):
            self._position = next(self._positions)
            return self._position

        def setPosition(self, position):
            self._position = position
            self.set_positions.append(position)

        def mediaStatus(self):
            from PyQt6.QtMultimedia import QMediaPlayer

            return QMediaPlayer.MediaStatus.LoadedMedia

    master = FakePlayer([8_300, 8_270, 8_340])
    midi_pair = widget._midi_normal
    original_players = widget._players
    try:
        widget._midi_normal = (master, object())
        widget._players = [master]
        widget._playing = True
        widget._finalizing = True
        widget._preview_duration = 30.0
        widget._playback_clock.reset(8_300)

        positions: list[int] = []
        scroll_values: list[int] = []
        for _ in range(3):
            widget._tick()
            positions.append(widget._position_ms)
            scroll_values.append(widget.roll_scroll.horizontalScrollBar().value())

        assert positions == [8_300, 8_300, 8_340]
        assert scroll_values == sorted(scroll_values)

        previous_scroll = scroll_values[-1]
        widget.seek(2.0)
        assert widget._position_ms == 2_000
        assert widget.roll_scroll.horizontalScrollBar().value() < previous_scroll
    finally:
        widget._playing = False
        widget._midi_normal = midi_pair
        widget._players = original_players
        widget.shutdown()
        widget.close()


def test_transport_slider_supports_absolute_click_and_continuous_drag_seek(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "transport-source.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    widget.resize(1000, 720)
    widget.show()
    widget.play_button.setEnabled(True)
    widget._set_playback_duration(20.0)
    app.processEvents()

    slider = widget.playback_slider
    emitted: list[int] = []
    slider.seek_requested.connect(emitted.append)
    y = slider.rect().center().y()

    try:
        click_x = round(slider.width() * 0.25)
        QTest.mouseClick(
            slider,
            Qt.MouseButton.LeftButton,
            pos=QPoint(click_x, y),
        )
        assert widget._position_ms == pytest.approx(5_000, abs=600)

        start_x = round(slider.width() * 0.2)
        middle_x = round(slider.width() * 0.55)
        end_x = round(slider.width() * 0.8)
        QTest.mousePress(
            slider,
            Qt.MouseButton.LeftButton,
            pos=QPoint(start_x, y),
        )
        QTest.mouseMove(slider, QPoint(middle_x, y), delay=1)
        assert widget._position_ms == pytest.approx(11_000, abs=600)
        QTest.mouseMove(slider, QPoint(end_x, y), delay=1)
        QTest.mouseRelease(
            slider,
            Qt.MouseButton.LeftButton,
            pos=QPoint(end_x, y),
        )

        assert widget._position_ms == pytest.approx(16_000, abs=600)
        assert slider.isSliderDown() is False
        assert len(emitted) >= 6

        committed_seeks: list[float] = []
        original_seek = widget.seek

        def recording_seek(seconds: float) -> None:
            committed_seeks.append(seconds)
            original_seek(seconds)

        widget.seek = recording_seek
        widget.pause = lambda: pytest.fail("scrubbing must not pause QMediaPlayer")
        widget._toggle_playback = lambda: pytest.fail("scrubbing must not restart QMediaPlayer")
        widget._playing = True
        QTest.mousePress(
            slider,
            Qt.MouseButton.LeftButton,
            pos=QPoint(start_x, y),
        )
        QTest.mouseMove(slider, QPoint(end_x, y), delay=1)
        assert committed_seeks == []
        assert widget._transport_scrubbing is True
        QTest.mouseRelease(
            slider,
            Qt.MouseButton.LeftButton,
            pos=QPoint(end_x, y),
        )
        assert committed_seeks == [pytest.approx(16.0, abs=0.6)]
        assert widget._transport_scrubbing is False
        assert widget._playing is True
    finally:
        widget.shutdown()
        widget.close()


def test_real_qmedia_players_can_seek_during_playback_without_deadlock(tmp_path: Path):
    source = _silent_wav(tmp_path / "real-player-scrub.wav", 2.0)
    script = f"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from PyQt6.QtCore import QPoint, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from src.gui.widgets.muscriptor_result import MuscriptorResultWidget

app = QApplication([])
source = Path({str(source)!r})
widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
widget._midi_normal = widget._make_player(source)
widget.play_button.setEnabled(True)
widget._set_playback_duration(2.0)
widget.resize(1000, 720)
widget.show()

def exercise():
    widget._toggle_playback()
    slider = widget.playback_slider
    target = QPoint(round(slider.width() * 0.6), slider.rect().center().y())
    QTest.mouseClick(slider, Qt.MouseButton.LeftButton, pos=target)
    QTimer.singleShot(250, verify)

def verify():
    if not widget._playing:
        raise RuntimeError("playback stopped after an in-flight slider click")
    if not 900 <= widget._position_ms <= 1500:
        raise RuntimeError(f"seek did not commit: {{widget._position_ms}}")
    if not 900 <= widget._midi_normal[0].position() <= 1500:
        raise RuntimeError(
            f"MIDI player did not seek: {{widget._midi_normal[0].position()}}"
        )
    widget.shutdown()
    widget.close()
    app.quit()

QTimer.singleShot(200, exercise)
QTimer.singleShot(3000, lambda: (_ for _ in ()).throw(RuntimeError("scrub timed out")))
app.exec()
"""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_source_and_target_bpm_are_bidirectionally_linked_and_not_overwritten(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "editable-bpm-playback.wav", 2.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    widget._midi_normal = widget._make_player(source)
    widget.play_button.setEnabled(True)
    widget._set_playback_duration(2.0)
    widget.set_bpm_context(117.9, 23.0)

    try:
        assert widget.bpm_spin.value() == pytest.approx(23.0)
        assert widget.speed_spin.value() == pytest.approx(
            round(23.0 / 117.9, 3),
            abs=0.0005,
        )
        assert widget.speed_spin.isVisible() is False

        widget.show()
        app.processEvents()
        assert widget.speed_spin.isVisible() is True

        widget._toggle_playback()
        widget.bpm_spin.setValue(132.5)
        app.processEvents()

        expected_rate = 132.5 / 117.9
        assert widget._playing is True
        assert widget._bpm_user_overridden is True
        assert widget.bpm_spin.value() == pytest.approx(132.5)
        assert widget.speed_spin.value() == pytest.approx(
            round(expected_rate, 3),
            abs=0.0005,
        )
        assert all(
            player.playbackRate() == pytest.approx(expected_rate)
            for player in widget._all_playback_players()
        )

        widget.set_detected_bpm(90.0)

        assert widget.bpm_spin.value() == pytest.approx(132.5)
        assert widget._detected_bpm == pytest.approx(117.9)
        assert widget.speed_spin.value() == pytest.approx(
            round(expected_rate, 3),
            abs=0.0005,
        )

        widget.speed_spin.setValue(0.75)
        app.processEvents()

        expected_bpm = round(117.9 * 0.75, 1)
        expected_rate = expected_bpm / 117.9
        assert widget.bpm_spin.value() == pytest.approx(expected_bpm)
        assert widget.speed_spin.value() == pytest.approx(
            round(expected_rate, 3),
            abs=0.0005,
        )
        assert all(
            player.playbackRate() == pytest.approx(expected_rate)
            for player in widget._all_playback_players()
        )
        late_player = widget._make_player(source)[0]
        assert late_player.playbackRate() == pytest.approx(expected_rate)
    finally:
        widget.shutdown()
        widget.close()


def test_midi_download_uses_editable_result_bpm_and_changes_playback_seconds(
    tmp_path: Path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    source_audio = _silent_wav(tmp_path / "download-bpm-source.wav", 1.0)
    source_midi = tmp_path / "source.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(117.9), time=0))
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(source_midi)

    destination = tmp_path / "downloaded.mid"
    widget = MuscriptorResultWidget(str(source_audio), ["acoustic_piano"])
    widget._midi_path = str(source_midi)
    widget.set_detected_bpm(117.9)
    widget.bpm_spin.setValue(132.5)
    assert widget.speed_spin.value() == pytest.approx(
        round(132.5 / 117.9, 3),
        abs=0.0005,
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "MIDI (*.mid)"),
    )

    try:
        widget._save_asset("midi")
    finally:
        widget.shutdown()
        widget.close()

    downloaded = mido.MidiFile(destination)
    tempo_messages = [
        message
        for midi_track in downloaded.tracks
        for message in midi_track
        if message.is_meta and message.type == "set_tempo"
    ]
    assert len(tempo_messages) == 1
    assert mido.tempo2bpm(tempo_messages[0].tempo) == pytest.approx(132.5, abs=0.001)
    downloaded_note = read_midi_roll_notes(destination)[0]
    source_note = read_midi_roll_notes(source_midi)[0]
    assert downloaded_note.start == pytest.approx(source_note.start, abs=1e-6)
    assert downloaded_note.end == pytest.approx(
        source_note.end * 117.9 / 132.5,
        abs=1e-6,
    )
    downloaded_note_off = next(
        message
        for message in downloaded.tracks[0]
        if not message.is_meta and message.type == "note_off"
    )
    source_note_off = next(
        message
        for message in mido.MidiFile(source_midi).tracks[0]
        if not message.is_meta and message.type == "note_off"
    )
    assert downloaded_note_off.time == source_note_off.time == 480
    assert mido.tempo2bpm(
        next(
            message.tempo
            for midi_track in mido.MidiFile(source_midi).tracks
            for message in midi_track
            if message.is_meta and message.type == "set_tempo"
        )
    ) == pytest.approx(117.9, abs=0.001)
    app.processEvents()


@pytest.mark.parametrize(
    ("editor", "typed_text", "expected_bpm"),
    [
        ("bpm", "BPM 60.0", 60.0),
        ("speed", "0.500x", 59.0),
    ],
)
def test_midi_download_commits_only_the_last_tempo_control_and_verifies_file(
    tmp_path: Path,
    monkeypatch,
    editor: str,
    typed_text: str,
    expected_bpm: float,
):
    app = QApplication.instance() or QApplication([])
    source_audio = _silent_wav(tmp_path / f"{editor}-source.wav", 1.0)
    source_midi = tmp_path / f"{editor}-source.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(117.9), time=0))
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(source_midi)
    source_duration = mido.MidiFile(source_midi).length

    destination = tmp_path / f"{editor}-downloaded.mid"
    widget = MuscriptorResultWidget(str(source_audio), ["acoustic_piano"])
    widget._midi_path = str(source_midi)
    widget.set_bpm_context(117.9, 117.9)
    control = widget.bpm_spin if editor == "bpm" else widget.speed_spin
    control.lineEdit().setText(typed_text)
    control.lineEdit().textEdited.emit(typed_text)
    dialog_defaults: list[str] = []

    def choose_destination(_parent, _title, default_path, _filter):
        dialog_defaults.append(default_path)
        return str(destination), "MIDI (*.mid)"

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        choose_destination,
    )

    try:
        widget.download_midi_action.trigger()
        assert dialog_defaults == [f"{source_midi.stem}_{expected_bpm:.1f}BPM.mid"]
        assert f"{expected_bpm:.1f}" in widget.playback_status_label.text()
        assert str(destination.resolve()) in widget.playback_status_label.text()
    finally:
        widget.shutdown()
        widget.close()

    downloaded = mido.MidiFile(destination)
    tempo_messages = [
        message
        for midi_track in downloaded.tracks
        for message in midi_track
        if message.is_meta and message.type == "set_tempo"
    ]
    assert len(tempo_messages) == 1
    assert mido.tempo2bpm(tempo_messages[0].tempo) == pytest.approx(
        expected_bpm,
        abs=0.001,
    )
    assert downloaded.length == pytest.approx(
        source_duration * 117.9 / expected_bpm,
        abs=1e-6,
    )
    app.processEvents()


def test_midi_download_aligns_note_onsets_without_changing_note_durations(
    tmp_path: Path,
):
    source_midi = tmp_path / "unaligned-source.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(117.9), time=0))
    midi.tracks.append(conductor)
    notes = mido.MidiTrack()
    notes.append(mido.Message("program_change", channel=0, program=0, time=0))
    notes.append(mido.Message("note_on", channel=0, note=60, velocity=90, time=37))
    notes.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=56))
    notes.append(mido.Message("note_on", channel=9, note=36, velocity=90, time=122))
    notes.append(mido.Message("note_off", channel=9, note=36, velocity=0, time=8))
    notes.append(mido.Message("note_on", channel=0, note=64, velocity=90, time=143))
    notes.append(mido.Message("note_off", channel=0, note=64, velocity=0, time=113))
    midi.tracks.append(notes)
    midi.save(source_midi)

    destination = tmp_path / "notation-compatible.mid"
    _export_midi_with_bpm(
        source_midi,
        destination,
        77.9,
        notation_compatible=True,
    )

    downloaded = mido.MidiFile(destination)
    tempo_messages = [
        message
        for track in downloaded.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    ]
    assert len(tempo_messages) == 1
    assert mido.tempo2bpm(tempo_messages[0].tempo) == pytest.approx(77.9, abs=0.001)
    assert downloaded.tracks[0][0].type == "set_tempo"
    pretty_midi_result = pretty_midi.PrettyMIDI(str(destination))
    tempo_times, tempo_values = pretty_midi_result.get_tempo_changes()
    assert tempo_times.tolist() == [0.0]
    assert tempo_values.tolist() == pytest.approx([77.9], abs=0.001)
    assert sum(len(instrument.notes) for instrument in pretty_midi_result.instruments) == 3

    grid_ticks = downloaded.ticks_per_beat // 6

    def note_timings(path: Path) -> list[tuple[int, int, int, int]]:
        timings: list[tuple[int, int, int, int]] = []
        for track in mido.MidiFile(path).tracks:
            absolute_tick = 0
            active: dict[tuple[int, int], list[int]] = {}
            for message in track:
                absolute_tick += message.time
                if message.is_meta:
                    continue
                if message.type == "note_on" and message.velocity > 0:
                    key = (message.channel, message.note)
                    active.setdefault(key, []).append(absolute_tick)
                elif message.type == "note_off" or (
                    message.type == "note_on" and message.velocity == 0
                ):
                    key = (message.channel, message.note)
                    starts = active.get(key)
                    if starts:
                        start = starts.pop(0)
                        timings.append(
                            (message.channel, message.note, start, absolute_tick - start)
                        )
        return timings

    source_timings = note_timings(source_midi)
    downloaded_timings = note_timings(destination)
    assert [(item[0], item[1], item[3]) for item in downloaded_timings] == [
        (item[0], item[1], item[3]) for item in source_timings
    ]
    assert all(item[2] % grid_ticks == 0 for item in downloaded_timings)
    assert all(
        abs(downloaded_item[2] - source_item[2]) <= grid_ticks // 2
        for source_item, downloaded_item in zip(source_timings, downloaded_timings)
    )


def _send_roll_wheel(widget: MuscriptorResultWidget, modifiers, *, delta: int) -> None:
    viewport = widget.roll_scroll.viewport()
    local_point = viewport.rect().center()
    event = QWheelEvent(
        QPointF(local_point),
        QPointF(viewport.mapToGlobal(local_point)),
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(viewport, event)


def test_follow_scroll_uses_fractional_motion_and_coarse_scrollbar_updates(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "smooth-follow.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    widget.resize(1200, 720)
    widget.show()
    widget.roll.set_notes(
        (MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 0.5),),
        duration=60.0,
    )
    app.processEvents()

    try:
        scrollbar = widget.roll_scroll.horizontalScrollBar()
        values: list[int] = []
        effective_offsets: list[float] = []
        for frame in range(120):
            position = 20.0 + frame * 0.016
            widget._follow_roll_to_position(position, allow_backward=False)
            target = min(
                scrollbar.maximum(),
                max(
                    scrollbar.minimum(),
                    widget.roll.x_for_time_float(position)
                    - widget.roll_scroll.viewport().width() / 2.0,
                ),
            )
            effective = scrollbar.value() + widget.roll.render_offset_px
            assert effective == pytest.approx(target, abs=0.02)
            assert widget.roll.x_for_time_float(position) - effective == pytest.approx(
                widget.roll_scroll.viewport().width() / 2.0,
                abs=0.02,
            )
            values.append(scrollbar.value())
            effective_offsets.append(effective)

        scrollbar_changes = sum(left != right for left, right in zip(values, values[1:]))
        assert scrollbar_changes < len(values) // 4
        assert effective_offsets == sorted(effective_offsets)
        assert any(offset % 1.0 for offset in effective_offsets)
    finally:
        widget.shutdown()
        widget.close()


def test_roll_wheel_shortcuts_zoom_at_cursor_and_scroll_horizontally(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "wheel-shortcuts.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    widget.resize(1200, 720)
    widget.show()
    widget.roll.set_notes(
        (MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 0.5),),
        duration=60.0,
    )
    app.processEvents()

    try:
        scrollbar = widget.roll_scroll.horizontalScrollBar()
        widget.follow_checkbox.setChecked(False)
        scrollbar.setValue(600)
        anchor_x = widget.roll_scroll.viewport().rect().center().x()
        anchor_time_before = widget.roll.time_for_x(scrollbar.value() + anchor_x)
        pixels_before = widget.roll.pixels_per_second

        _send_roll_wheel(
            widget,
            Qt.KeyboardModifier.ControlModifier,
            delta=120,
        )
        app.processEvents()

        anchor_time_after = widget.roll.time_for_x(scrollbar.value() + anchor_x)
        assert widget.roll.pixels_per_second > pixels_before
        assert anchor_time_after == pytest.approx(
            anchor_time_before,
            abs=1.0 / widget.roll.pixels_per_second,
        )
        assert not widget.follow_checkbox.isChecked()

        widget.follow_checkbox.setChecked(True)
        widget.roll.set_position(20.0)
        alt_pixels_before = widget.roll.pixels_per_second
        _send_roll_wheel(
            widget,
            Qt.KeyboardModifier.AltModifier,
            delta=120,
        )
        app.processEvents()
        assert widget.roll.pixels_per_second > alt_pixels_before
        assert widget.follow_checkbox.isChecked()
        effective_offset = scrollbar.value() + widget.roll.render_offset_px
        assert widget.roll.x_for_time_float(
            widget.roll.position
        ) - effective_offset == pytest.approx(
            widget.roll_scroll.viewport().width() / 2.0,
            abs=0.02,
        )

        horizontal_before = scrollbar.value()
        _send_roll_wheel(
            widget,
            Qt.KeyboardModifier.ShiftModifier,
            delta=-120,
        )
        app.processEvents()
        assert scrollbar.value() > horizontal_before
        assert not widget.follow_checkbox.isChecked()
        assert "QScrollBar::handle:horizontal" in widget.styleSheet()
    finally:
        widget.shutdown()
        widget.close()


def test_roll_horizontal_scrollbar_handle_drag_is_not_cancelled(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "scrollbar-drag.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    widget.resize(1200, 720)
    widget.show()
    widget.roll.set_notes(
        (MuscriptorRollNote("acoustic_piano", 60, 100, 0.0, 0.5),),
        duration=120.0,
    )
    app.processEvents()

    try:
        scrollbar = widget.roll_scroll.horizontalScrollBar()
        assert scrollbar.maximum() > 0
        scrollbar.setValue(scrollbar.maximum() // 4)
        app.processEvents()

        option = QStyleOptionSlider()
        option.initFrom(scrollbar)
        option.orientation = scrollbar.orientation()
        option.minimum = scrollbar.minimum()
        option.maximum = scrollbar.maximum()
        option.singleStep = scrollbar.singleStep()
        option.pageStep = scrollbar.pageStep()
        option.sliderPosition = scrollbar.sliderPosition()
        option.sliderValue = scrollbar.value()
        option.upsideDown = scrollbar.invertedAppearance()
        option.subControls = QStyle.SubControl.SC_All
        handle = scrollbar.style().subControlRect(
            QStyle.ComplexControl.CC_ScrollBar,
            option,
            QStyle.SubControl.SC_ScrollBarSlider,
            scrollbar,
        )
        start = handle.center()
        target = QPoint(min(scrollbar.width() - 5, start.x() + 200), start.y())
        before = scrollbar.value()

        QTest.mousePress(scrollbar, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(scrollbar, target, delay=5)
        QTest.mouseRelease(scrollbar, Qt.MouseButton.LeftButton, pos=target)
        app.processEvents()

        assert scrollbar.value() > before + scrollbar.maximum() // 10
        assert scrollbar.isSliderDown() is False
        assert not widget.follow_checkbox.isChecked()
    finally:
        widget.shutdown()
        widget.close()


def test_project_native_selector_controls_real_constraint_state():
    app = QApplication.instance() or QApplication([])
    selector = MuscriptorInstrumentSelector()
    selector.resize(760, 220)
    selector.show()
    app.processEvents()

    selector.set_selected_instruments(["acoustic_piano"])
    assert selector.selected_instruments() == ["acoustic_piano"]
    assert selector.clear_button.isEnabled()
    assert "acoustic_piano" not in selector._model._items

    selector.search_edit.setFocus()
    selector.search_edit.setText("")
    selector._show_completion_popup()
    app.processEvents()
    assert selector.completer.popup().isVisible()
    assert selector.completer.popup().width() >= selector.input_surface.width() - 2
    assert selector.completer.popup().height() <= selector.MAX_POPUP_HEIGHT
    assert selector.completer.popup().maximumHeight() <= selector.MAX_POPUP_HEIGHT
    assert "#17243d" in selector.styleSheet()
    assert "#101010" not in selector.styleSheet()

    selector.search_edit.setText("电钢琴")
    selector._add_exact_search_match()
    assert selector.selected_instruments() == ["acoustic_piano", "electric_piano"]

    QTest.mouseClick(
        selector._tags["acoustic_piano"].remove_button,
        Qt.MouseButton.LeftButton,
    )
    assert selector.selected_instruments() == ["electric_piano"]

    QTest.mouseClick(selector.clear_button, Qt.MouseButton.LeftButton)
    assert selector.selected_instruments() == []
    selector.close()


def test_selector_uses_filtered_completion_identity_for_multiple_choices():
    app = QApplication.instance() or QApplication([])
    selector = MuscriptorInstrumentSelector()
    selector.set_selected_instruments(["acoustic_piano"])
    selector.show()
    app.processEvents()

    drums_label = selector._label("drums")
    selector.completer.setCompletionPrefix(drums_label)
    completion_model = selector.completer.completionModel()
    drums_index = completion_model.index(0, 0)
    assert drums_index.data(Qt.ItemDataRole.DisplayRole) == drums_label
    selector._on_completion_index(drums_index)
    app.processEvents()

    voice_label = selector._label("voice")
    selector.completer.setCompletionPrefix(voice_label)
    voice_index = selector.completer.completionModel().index(0, 0)
    assert voice_index.data(Qt.ItemDataRole.DisplayRole) == voice_label
    selector._on_completion_index(voice_index)
    app.processEvents()

    assert selector.selected_instruments() == ["acoustic_piano", "drums", "voice"]
    selector.close()


def test_selector_real_popup_click_clears_completer_text_before_next_choice():
    app = QApplication.instance() or QApplication([])
    selector = MuscriptorInstrumentSelector()
    selector.resize(760, 220)
    selector.show()
    app.processEvents()

    selector.search_edit.setFocus()
    selector._show_completion_popup()
    app.processEvents()
    popup = selector.completer.popup()
    first = selector.completer.completionModel().index(0, 0)
    first_name = selector._model._items[first.row()]
    QTest.mouseClick(
        popup.viewport(),
        Qt.MouseButton.LeftButton,
        pos=popup.visualRect(first).center(),
    )
    app.processEvents()

    assert selector.selected_instruments() == [first_name]
    assert selector.search_edit.text() == ""
    assert selector.completer.completionPrefix() == ""
    assert selector.completer.completionModel().rowCount() > 0

    popup.hide()
    QTest.mouseClick(selector.search_edit, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert popup.isVisible()
    second = selector.completer.completionModel().index(0, 0)
    second_label = second.data(Qt.ItemDataRole.DisplayRole)
    QTest.mouseClick(
        popup.viewport(),
        Qt.MouseButton.LeftButton,
        pos=popup.visualRect(second).center(),
    )
    app.processEvents()

    assert len(selector.selected_instruments()) == 2
    assert selector._label(selector.selected_instruments()[1]) == second_label
    assert selector.search_edit.text() == ""
    selector.close()


def test_project_native_selector_stays_bounded_at_narrow_width():
    app = QApplication.instance() or QApplication([])
    selector = MuscriptorInstrumentSelector()
    selector.resize(320, 150)
    selector.set_selected_instruments(["acoustic_piano"])
    selector.show()
    app.processEvents()

    selector._show_completion_popup()
    app.processEvents()
    popup = selector.completer.popup()
    available = selector.screen().availableGeometry()

    assert selector.rect().contains(selector.clear_button.geometry().topRight())
    assert selector.rect().contains(selector.input_surface.geometry().topRight())
    assert popup.width() <= available.width() - 16
    assert popup.height() <= selector.MAX_POPUP_HEIGHT
    selector.close()


def test_muscriptor_web_surfaces_use_project_palette_not_official_demo_skin():
    assert "background:#17243d" in MUSCRIPTOR_RESULT_CSS
    assert "border-color:#4a9eff" in MUSCRIPTOR_RESULT_CSS
    assert "#101010" not in MUSCRIPTOR_RESULT_CSS
    assert "#31e6b2" not in MUSCRIPTOR_RESULT_CSS


def test_browser_midi_workbench_uses_virtualized_smooth_transport_and_shortcuts():
    assert "msr-roll-viewport" in MUSCRIPTOR_RESULT_CSS
    assert "ResizeObserver" in MUSCRIPTOR_RESULT_JS
    assert "this.position=this.m.duration;this.pause();return;" in MUSCRIPTOR_RESULT_JS
    assert "this.position=0;this.pause();return;" not in MUSCRIPTOR_RESULT_JS
    assert "e.ctrlKey||e.altKey" in MUSCRIPTOR_RESULT_JS
    assert "e.shiftKey" in MUSCRIPTOR_RESULT_JS
    assert 'CustomEvent("music-to-midi-playback-start"' in MUSCRIPTOR_RESULT_JS
    assert 'this.playhead.style.transform="translate3d(' in MUSCRIPTOR_RESULT_JS
    assert 'querySelectorAll(".msr-root:not([data-msr-init])")' in MUSCRIPTOR_RESULT_JS


def test_browser_midi_workbench_labels_generic_gm_programs():
    state = {
        "audio_path": "C:/tmp/source.wav",
        "midi_path": "C:/tmp/result.mid",
        "transcription_wav": "C:/tmp/result.wav",
        "stereo_mix_wav": "C:/tmp/stereo.wav",
        "instrument_wavs": {"gm:000": "C:/tmp/piano.wav"},
        "selected_instruments": ["gm:000"],
        "detected_instruments": ["gm:000"],
        "notes": [],
        "duration": 1.0,
        "backend_label": "YourMT3+",
        "source_track_name": "vocals",
    }

    markup = build_muscriptor_result_html(state, lambda key: key, "en_US")

    assert "Acoustic Grand Piano" in markup
    assert "YourMT3+" in markup
    assert "vocals" in markup


def test_linked_desktop_midi_detail_identifies_source_and_closes_independently(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "linked.wav", 0.1)
    widget = MuscriptorResultWidget(
        str(source),
        [],
        backend_label="MIROS",
        muscriptor_groups=False,
        source_track_name="vocals",
    )
    try:
        app.processEvents()
        assert widget.source_label.isHidden() is False
        assert "vocals" in widget.source_label.text()
        assert "MIROS" in widget.source_label.text()
        assert widget.another_button.text() == t("muscriptor_result.close_detail")
        changes = []
        widget.playing_changed.connect(changes.append)
        widget._playing = True
        widget.pause()
        assert changes == [False]
    finally:
        widget.shutdown()
        widget.close()


def test_muscriptor_desktop_result_uses_project_palette(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    widget = MuscriptorResultWidget(
        str(tmp_path / "audio.wav"),
        ["acoustic_piano"],
    )
    widget.show()
    app.processEvents()

    try:
        stylesheet = widget.styleSheet()
        assert "#17243d" in stylesheet
        assert "#4a9eff" in stylesheet
        assert "#0d1013" not in stylesheet
        assert "#31e6b2" not in stylesheet
    finally:
        widget.shutdown()
        widget.close()


def test_muscriptor_browser_workbench_javascript_is_syntactically_valid(tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node runtime is unavailable for MuScriptor JS validation")
    javascript = tmp_path / "muscriptor-result-runtime.js"
    javascript.write_text(MUSCRIPTOR_RESULT_JS, encoding="utf-8")

    completed = subprocess.run(
        [node, "--check", str(javascript)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode == 0, completed.stdout
