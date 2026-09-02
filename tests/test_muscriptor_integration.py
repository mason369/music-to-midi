from __future__ import annotations

import io
import json
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

from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog, QLabel, QStyle, QStyleOptionSlider

from src.core import muscriptor_result_assets
from src.core.manual_midi import (
    MIDI_ROUTE_MIROS,
    MIDI_ROUTE_MUSCRIPTOR,
    MIDI_ROUTE_MUSCRIPTOR_MEDIUM,
    MIDI_ROUTE_MUSCRIPTOR_SMALL,
    build_manual_midi_config,
)
from src.core.midi_editor import export_edited_midi
from src.core.midi_tempo import (
    non_tempo_event_tick_fingerprint,
    non_tempo_event_time_fingerprint,
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
    _COLORS,
    MUSCRIPTOR_RESULT_CSS,
    MUSCRIPTOR_RESULT_JS,
    build_muscriptor_result_html,
)
from src.gui.widgets.muscriptor_instrument_selector import (
    MuscriptorInstrumentSelector,
)
from src.gui.widgets.muscriptor_result import (
    _INSTRUMENT_COLORS,
    MuscriptorResultWidget,
    _ChunkProgressEstimator,
    _export_midi_at_project_speed,
    _export_midi_with_bpm,
    _PianoRollCanvas,
    _SmoothPlaybackClock,
)
from src.i18n.translator import t
from src.models.data_models import (
    BeatInfo,
    Config,
    MultiInstrumentModel,
    MuscriptorModel,
    MuscriptorProcessingChain,
    ProcessingResult,
)
from src.models.muscriptor_instruments import (
    infer_muscriptor_instruments_from_stem_name,
    muscriptor_instrument_label,
)
from src.utils import muscriptor_downloader


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
            custom_bpm=60.0,
        ).to_dict()
    )

    assert restored.muscriptor_model == "small"
    assert restored.custom_bpm == pytest.approx(60.0)


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


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "陌上踏青_guitar.wav",
            [
                "acoustic_guitar",
                "clean_electric_guitar",
                "distorted_electric_guitar",
            ],
        ),
        ("song_bass.wav", ["acoustic_bass", "electric_bass"]),
        ("vocals.wav", ["voice"]),
        ("mix_clean_electric_guitar.flac", ["clean_electric_guitar"]),
        ("guitar solo demo.wav", []),
        ("accompaniment.wav", []),
    ],
)
def test_standard_stem_names_infer_visible_muscriptor_hard_constraints(
    filename: str,
    expected: list[str],
):
    assert infer_muscriptor_instruments_from_stem_name(filename) == expected


def test_solo_and_instrument_row_keep_active_editor_instrument_synchronized(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    audio = _silent_wav(tmp_path / "source.wav")
    notes = (
        MuscriptorRollNote("acoustic_guitar", 60, 90, 0.0, 0.5, 24, False, 0, 0),
        MuscriptorRollNote(
            "clean_electric_guitar",
            64,
            90,
            0.5,
            1.0,
            26,
            False,
            1,
            1,
        ),
    )
    widget = MuscriptorResultWidget(str(audio), [])
    try:
        widget.set_bpm_context(120.0, 120.0)
        widget._detected = ["acoustic_guitar", "clean_electric_guitar"]
        widget.roll.set_notes(notes, duration=1.0)
        widget._begin_editor_session(notes, 1.0)
        widget._rebuild_instrument_rows()
        widget.roll.set_selected_index(0)
        before = widget._edited_notes

        widget._toggle_solo("clean_electric_guitar")
        app.processEvents()

        assert widget._active_edit_instrument == "clean_electric_guitar"
        assert widget.edit_instrument_combo.currentData() == "clean_electric_guitar"
        assert widget._soloed == "clean_electric_guitar"
        assert widget._muted == {"acoustic_guitar"}
        assert "#4a9eff" in widget._instrument_rows["clean_electric_guitar"].styleSheet()
        assert widget._edited_notes == before
    finally:
        widget.shutdown()
        widget.deleteLater()


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
    beats = types.ModuleType("muscriptor.utils.beats")

    class BeatGrid:
        def __init__(
            self,
            *,
            bpm: float,
            beats_per_bar: int | None,
            first_downbeat: float,
            beats=None,
            onset_delay=None,
        ):
            self.bpm = bpm
            self.beats_per_bar = beats_per_bar
            self.first_downbeat = first_downbeat
            self.beats = beats
            self.onset_delay = onset_delay

        def bar_offset(self, min_shift: float = 0.0) -> float:
            step = (self.beats_per_bar or 1) * 60.0 / self.bpm
            offset = (
                (step - self.first_downbeat % step) % step
                if self.beats_per_bar is not None
                else 0.0
            )
            while offset < min_shift:
                offset += step
            return offset

        def with_onset_delay(self, onsets):
            unique_onsets = {round(float(value), 3) for value in onsets}
            delay = 0.012 if len(unique_onsets) >= 40 else 0.0
            return BeatGrid(
                bpm=self.bpm,
                beats_per_bar=self.beats_per_bar,
                first_downbeat=self.first_downbeat,
                beats=self.beats,
                onset_delay=delay,
            )

    beats.BeatGrid = BeatGrid
    utils = types.ModuleType("muscriptor.utils")
    utils.beats = beats
    package = types.ModuleType("muscriptor")
    package.events = events
    package.utils = utils
    monkeypatch.setitem(sys.modules, "muscriptor", package)
    monkeypatch.setitem(sys.modules, "muscriptor.events", events)
    monkeypatch.setitem(sys.modules, "muscriptor.utils", utils)
    monkeypatch.setitem(sys.modules, "muscriptor.utils.beats", beats)
    return ProgressEvent, NoteStartEvent, NoteEndEvent


def _configure_test_beat_grid(transcriber: MuscriptorTranscriber) -> None:
    transcriber.set_beat_info(
        BeatInfo(
            bpm=120.0,
            beat_times=[0.0, 0.5, 1.0, 1.5],
            downbeats=[0.0],
            time_signature=(4, 4),
        )
    )


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

    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    def fake_memory_bounded_loader(weights_path, device, *, processing_chain):
        captured["weights_path"] = Path(weights_path)
        captured["device"] = device
        captured["processing_chain"] = processing_chain
        return loaded_model

    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.load_muscriptor_model_memory_bounded",
        fake_memory_bounded_loader,
    )
    monkeypatch.setattr(
        MuscriptorTranscriber,
        "_runtime_unavailable_reason",
        staticmethod(lambda: ""),
    )
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.get_cached_muscriptor_paths",
        lambda *_args, **_kwargs: (weights, config_path),
    )
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.get_device",
        lambda *_args, **_kwargs: "cuda:0",
    )
    monkeypatch.setattr(
        "src.core.muscriptor_transcriber.ensure_accelerator_runtime_compatibility",
        lambda _device: None,
    )
    transcriber = MuscriptorTranscriber(
        Config(
            use_gpu=True,
            gpu_device=0,
            muscriptor_processing_chain=MuscriptorProcessingChain.TELKNET.value,
        )
    )

    assert transcriber.load_model() is loaded_model
    assert captured == {
        "weights_path": weights,
        "device": "cuda:0",
        "processing_chain": MuscriptorProcessingChain.TELKNET.value,
    }
    assert {parameter.dtype for parameter in inner_model.parameters()} == {torch.float32}
    assert transcriber._runtime_details == {
        "type": "runtime",
        "model": "MuScriptor-large",
        "device": "cuda:0",
        "gpu": "cuda:0",
        "compute_dtype": "upstream",
        "kv_cache_dtype": "upstream",
        "weight_load_strategy": "safetensors_cpu_tensor_stream",
        "kv_cache_reused_layers": 0,
        "batch_size": 1,
        "prelude_forcing": True,
        "processing_chain": MuscriptorProcessingChain.TELKNET.value,
        "quality_mode": "official_v0.3.0+telknet_issue74_boundary_recovery_v1",
        "boundary_recovery": "telknet_issue74_single_program_v1",
        "window_seconds": 5.0,
        "no_eos_is_ok": True,
        "package_version": "0.3.0",
        "source_commit": "d73147e75e5b9b0c0a79ebe154587db4fd603e0c",
    }


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

        def events_to_midi_bytes(self, events, *, beat_grid=None):
            assert len(list(events)) == 3
            assert beat_grid.bpm == pytest.approx(120.0)
            assert beat_grid.beats_per_bar == 4
            assert beat_grid.first_downbeat == pytest.approx(0.0)
            assert beat_grid.beats.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.5])
            assert beat_grid.onset_delay is None
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
    transcriber._loaded_processing_chain = config.muscriptor_processing_chain
    _configure_test_beat_grid(transcriber)

    assert transcriber.transcribe_to_midi(str(audio), str(output)) == str(output.resolve())
    assert output.is_file()
    assert model.kwargs == {
        "instruments": ["acoustic_piano"],
        "use_sampling": False,
        "batch_size": 1,
        "beam_size": 1,
        "prelude_forcing": True,
    }


def test_pinned_muscriptor_eos_policy_matches_official_default():
    import inspect

    from muscriptor.transcription_model import TranscriptionModel

    parameter = inspect.signature(TranscriptionModel.transcribe).parameters["no_eos_is_ok"]
    assert parameter.default is True


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

        def events_to_midi_bytes(self, events, *, beat_grid=None):
            assert len(list(events)) == 1
            assert beat_grid is not None
            return _midi_bytes()

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    output = tmp_path / "output.mid"
    transcriber = MuscriptorTranscriber(
        Config(
            use_gpu=False,
            muscriptor_processing_chain=MuscriptorProcessingChain.TELKNET.value,
        )
    )
    transcriber._model = FakeModel()
    transcriber._loaded_processing_chain = MuscriptorProcessingChain.TELKNET.value
    _configure_test_beat_grid(transcriber)
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
    assert received[0]["onset_phase_correction"] == "pending"
    assert received[1] == {"type": "progress", "completed": 1, "total": 1}
    assert received[2] == transcriber._runtime_details
    assert received[2]["onset_phase_correction"] == "disabled_native_seconds"
    assert received[2]["onset_delay_seconds"] == pytest.approx(0.0)
    assert received[2]["onset_correction_applied"] is False
    assert received[2]["bar_offset_seconds"] == pytest.approx(0.0)
    assert received[2]["chunk_boundary_continuity"] == ("exact_5s_three_frame_note_continuity_v1")
    assert received[2]["boundary_notes_merged"] == 0


def test_transcriber_applies_official_onset_phase_before_midi_writer(
    tmp_path: Path,
    monkeypatch,
):
    ProgressEvent, NoteStartEvent, NoteEndEvent = _install_fake_event_module(monkeypatch)
    starts = [
        NoteStartEvent(index, "acoustic_piano", 60 + index % 12, index * 0.025)
        for index in range(40)
    ]
    events = [ProgressEvent(0, 1)]
    for start in starts:
        events.extend((start, NoteEndEvent(start.index, start, start.start_time + 0.1)))
    events.append(ProgressEvent(1, 1))

    class FakeModel:
        written_grid = None

        def transcribe(self, _source, **_kwargs):
            return iter(events)

        def events_to_midi_bytes(self, streamed, *, beat_grid=None):
            assert len(list(streamed)) == len(events)
            self.written_grid = beat_grid
            return _midi_bytes(0)

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    output = tmp_path / "output.mid"
    transcriber = MuscriptorTranscriber(
        Config(
            use_gpu=False,
            muscriptor_processing_chain=MuscriptorProcessingChain.TELKNET.value,
        )
    )
    transcriber._model = FakeModel()
    transcriber._loaded_processing_chain = MuscriptorProcessingChain.TELKNET.value
    _configure_test_beat_grid(transcriber)

    transcriber.transcribe_to_midi(str(audio), str(output))

    assert transcriber._model.written_grid.onset_delay is None
    assert transcriber.last_onset_delay_seconds == pytest.approx(0.0)
    assert transcriber.last_bar_offset_seconds == pytest.approx(0.0)


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
    transcriber._loaded_processing_chain = config.muscriptor_processing_chain
    _configure_test_beat_grid(transcriber)

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

        def events_to_midi_bytes(self, events, *, beat_grid=None):
            assert len(list(events)) == 7
            assert beat_grid is not None
            return _midi_bytes(0)

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    output = tmp_path / "output.mid"
    transcriber = MuscriptorTranscriber(Config(use_gpu=False))
    transcriber._model = FakeModel()
    transcriber._loaded_processing_chain = transcriber.config.muscriptor_processing_chain
    _configure_test_beat_grid(transcriber)
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


def test_playhead_clock_interpolation_tracks_non_unit_playback_rate():
    clock = _SmoothPlaybackClock(max_lead_ms=120.0)
    clock.reset(1_000, now=0.0)

    assert clock.sample(1_000, playback_rate=1.5, now=0.100) == pytest.approx(1_120)
    clock.reset(1_000, now=0.0)
    assert clock.sample(1_000, playback_rate=0.5, now=0.100) == pytest.approx(1_050)


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
        "validate_muscriptor_soundfont",
        lambda: soundfont,
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


def test_preview_audio_loader_preserves_44k_stereo_average(tmp_path: Path):
    frames = 441
    left = np.linspace(-0.8, 0.8, frames, dtype=np.float32)
    right = np.linspace(0.4, -0.4, frames, dtype=np.float32)
    source = tmp_path / "stereo.wav"
    sf.write(source, np.stack([left, right], axis=1), 44_100, subtype="FLOAT")

    loaded = muscriptor_result_assets._load_mono_44k(source)

    assert loaded.dtype == np.float32
    assert loaded.flags.c_contiguous
    assert loaded == pytest.approx((left + right) / 2.0, abs=1e-7)


def test_preview_audio_loader_uses_hq_resampling(tmp_path: Path, monkeypatch):
    source = tmp_path / "source-48k.wav"
    time_axis = np.arange(480, dtype=np.float32) / 48_000.0
    sf.write(source, np.sin(2.0 * np.pi * 440.0 * time_axis), 48_000)
    calls: list[tuple[int, int, str]] = []
    import soxr

    original_resample = soxr.resample

    def recording_resample(audio, input_rate, output_rate, *, quality):
        calls.append((input_rate, output_rate, quality))
        return original_resample(
            audio,
            input_rate,
            output_rate,
            quality=quality,
        )

    monkeypatch.setattr(soxr, "resample", recording_resample)

    loaded = muscriptor_result_assets._load_mono_44k(source)

    assert calls == [(48_000, 44_100, "HQ")]
    assert len(loaded) == 441
    assert loaded.dtype == np.float32


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
        "validate_muscriptor_soundfont",
        lambda: soundfont,
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
    aligned_source = _silent_wav(tmp_path / "original-live.wav", 1.0)
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
                original_wav=aligned_source,
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
        assert widget._playback_engine.is_configured
        assert widget._playback_engine.output_stream_count == 1
        assert widget._playback_engine.duration_seconds == pytest.approx(1.0)

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
        aligned_source = _silent_wav(tmp_path / "generic-original-live.wav", 1.0)
        widget._on_preview_ready(
            1,
            MuscriptorPreviewAssets(
                notes=tuple(widget._stream_notes),
                duration=1.0,
                transcription_wav=rendered,
                original_wav=aligned_source,
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
    first_original = _tone_wav(tmp_path / "first-original.wav", 1.0, amplitude=0.1)
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
                original_wav=first_original,
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
                original_wav=source,
                instrument_wavs={"acoustic_piano": second_audio},
                midi_gain_db=0.0,
            ),
        )

        assert widget._preview_duration == pytest.approx(1.0)
        assert widget.roll._notes == (first_note,)
        assert widget._deferred_preview is not None

        widget._toggle_playback()
        for _ in range(100):
            app.processEvents()
            if widget._deferred_preview is None:
                break
            QTest.qWait(1)

        assert not widget._playing
        assert widget._deferred_preview is None
        assert widget._preview_duration == pytest.approx(2.0)
        assert widget.roll._notes == (first_note, second_note)
        assert not widget._playback_engine.is_playing
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
                original_wav=source,
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
                original_wav=source,
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
        original_wav=source,
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


def test_real_synchronized_preview_rollover_does_not_deadlock_event_loop(tmp_path: Path):
    source = _tone_wav(tmp_path / "source.wav", 2.0, amplitude=0.1)
    first_original = _tone_wav(tmp_path / "first-original.wav", 1.0, amplitude=0.1)
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
        original_wav={str(first_original)!r},
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
            original_wav={str(source)!r},
            instrument_wavs={{"acoustic_piano": {str(second)!r}}},
            midi_gain_db=0.0,
        ),
    )
    # Drive the preview boundary explicitly. The synchronized sink owns one
    # transport, so rollover only waits for that stream to stop.
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


def test_bar_aligned_reference_audio_prepends_exact_verified_silence(tmp_path: Path):
    source = _tone_wav(tmp_path / "source.wav", 0.2, amplitude=0.1)
    output_dir = tmp_path / "bar-aligned"
    output_dir.mkdir()
    offset_seconds = 0.125

    aligned = muscriptor_result_assets._write_bar_aligned_reference_audio(
        source,
        output_dir,
        offset_seconds,
    )

    source_audio, source_rate = sf.read(source, dtype="float32")
    aligned_audio, aligned_rate = sf.read(aligned, dtype="float32")
    padding_frames = round(offset_seconds * 44_100)
    assert source_rate == aligned_rate == 44_100
    assert len(aligned_audio) == len(source_audio) + padding_frames
    assert float(np.max(np.abs(aligned_audio[:padding_frames]))) == 0.0
    assert aligned_audio[padding_frames:] == pytest.approx(source_audio, abs=1 / 32_768)


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
        "validate_muscriptor_soundfont",
        lambda: soundfont,
    )

    def fake_synthesize(_executable, _soundfont, _midi_path, output_path, cancel_check=None):
        assert cancel_check is not None and not cancel_check()
        _tone_wav(Path(output_path), 2.0, amplitude=0.02)

    monkeypatch.setattr(muscriptor_result_assets, "_synthesize", fake_synthesize)
    midi = mido.MidiFile(type=1, ticks_per_beat=1_000)
    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
    tempo.append(
        mido.MetaMessage(
            "marker",
            text="muscriptor:bar_offset=0.125000",
            time=0,
        )
    )
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
        assets.original_wav,
        assets.original_left_wav,
        assets.transcription_right_wav,
        *assets.instrument_wavs.values(),
        *assets.instrument_right_wavs.values(),
    ):
        info = sf.info(path)
        assert (info.frames, info.samplerate) == (master.frames, master.samplerate)
    original_bus, original_rate = sf.read(assets.original_wav, dtype="float32")
    source_audio, source_rate = sf.read(source, dtype="float32")
    padding_frames = round(0.125 * original_rate)
    assert original_rate == source_rate == 44_100
    assert float(np.max(np.abs(original_bus[:padding_frames]))) == 0.0
    assert original_bus[padding_frames : padding_frames + len(source_audio)] == pytest.approx(
        source_audio,
        abs=1 / 32_768,
    )
    assert float(np.max(np.abs(original_bus[padding_frames + len(source_audio) :]))) == 0.0


def test_empty_edited_midi_builds_exact_silence_without_invoking_fluidsynth(
    tmp_path: Path,
    monkeypatch,
):
    source = _tone_wav(tmp_path / "empty-edit-source.wav", 1.25, amplitude=0.1)
    midi_path = tmp_path / "empty-edited.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    midi.tracks.append(conductor)
    midi.save(midi_path)

    monkeypatch.setattr(
        muscriptor_result_assets,
        "get_fluidsynth_executable",
        lambda: pytest.fail("FluidSynth must not be invoked for an intentionally empty edit"),
    )
    monkeypatch.setattr(
        muscriptor_result_assets,
        "validate_muscriptor_soundfont",
        lambda **_kwargs: pytest.fail(
            "The SoundFont must not be loaded for an intentionally empty edit"
        ),
    )

    assets = muscriptor_result_assets.prepare_midi_playback_assets(
        midi_path,
        source,
        tmp_path / "empty-assets",
        allow_empty_notes=True,
    )

    rendered, sample_rate = sf.read(assets.live_transcription_wav, dtype="float32")
    assert assets.notes == ()
    assert assets.instrument_wavs == {}
    assert assets.instrument_right_wavs == {}
    assert assets.duration == pytest.approx(1.25)
    assert sample_rate == 44_100
    assert len(rendered) == round(1.25 * sample_rate)
    assert float(np.max(np.abs(rendered))) == 0.0


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
                original_wav=source,
                instrument_wavs=stems,
                midi_gain_db=0.0,
            ),
        )
        app.processEvents()

        assert widget._playback_engine.output_stream_count == 1
        assert widget._playback_engine._instrument_buses_loaded is False
        assert widget._playback_engine._source is not None
        assert widget._playback_engine._source._mix == pytest.approx(0.75)

        widget._muted = {instruments[0]}
        widget._apply_mix()
        assert widget._playback_engine.output_stream_count == 1
        assert widget._playback_engine._instrument_buses_loaded is True
        assert widget._playback_engine._source._muted == frozenset({instruments[0]})
    finally:
        widget.shutdown()
        widget.close()


def test_playhead_tick_never_reseeks_synchronized_audio_stream(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    app.processEvents()

    class FakeSynchronizedEngine:
        is_configured = True
        is_playing = True
        position_seconds = 1.234

        def __init__(self):
            self.seek_calls: list[float] = []

        def seek(self, seconds: float) -> None:
            self.seek_calls.append(seconds)

    engine = FakeSynchronizedEngine()
    original_engine = widget._playback_engine
    try:
        widget._playback_engine = engine
        widget._playing = True
        widget._finalizing = True
        widget._preview_duration = 0.0
        widget._set_playback_duration(10.0)

        widget._tick()

        assert widget._position_ms == 1_234
        assert widget.playback_slider.value() == 1_234
        assert engine.seek_calls == []
    finally:
        widget._playing = False
        widget._playback_engine = original_engine
        widget.shutdown()
        widget.close()


def test_transport_start_submits_one_synchronized_audio_stream(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "batch-start-source.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])

    class FakeSynchronizedEngine:
        is_configured = True
        is_playing = False
        position_seconds = 0.0

        def __init__(self):
            self.seek_calls: list[float] = []
            self.play_calls = 0

        def set_mix_state(self, **_kwargs) -> None:
            pass

        def seek(self, seconds: float) -> None:
            self.seek_calls.append(seconds)

        def play(self) -> None:
            self.play_calls += 1

    engine = FakeSynchronizedEngine()
    original_engine = widget._playback_engine
    try:
        widget._playback_engine = engine
        widget._position_ms = 750
        widget._toggle_playback()

        assert engine.seek_calls == [0.75]
        assert engine.play_calls == 1
        assert widget._playing is True
    finally:
        widget._playing = False
        widget._playback_engine = original_engine
        widget.shutdown()
        widget.close()
        app.processEvents()


def test_end_of_stream_stops_transport_without_automatic_rewind(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "source.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    app.processEvents()
    try:
        widget._playing = True
        widget._finalizing = True
        widget._preview_duration = 0.0
        widget._position_ms = 990
        widget._set_playback_duration(1.0)

        widget._on_synchronized_playback_finished()

        assert widget._playing is False
        assert widget._playback_finished is True
        assert widget._position_ms == 1_000
        assert widget.clock_label.text() == "1.0s"

        widget._tick()
        assert widget._position_ms == 1_000
    finally:
        widget._playing = False
        widget.shutdown()
        widget.close()


def test_stream_end_schedules_buffered_preview_application(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "stream-end-source.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    app.processEvents()
    scheduled: list[bool] = []
    try:
        widget._playing = True
        widget._finalizing = False
        widget._preview_duration = 1.0
        widget._position_ms = 990
        widget._set_playback_duration(1.0)
        widget._apply_deferred_after_playback_stop = lambda: scheduled.append(True)

        widget._on_synchronized_playback_finished()

        assert widget._playing is False
        assert widget._playback_finished is True
        assert scheduled == [True]
    finally:
        widget._playing = False
        widget.shutdown()
        widget.close()


def test_deferred_preview_waits_for_synchronized_sink_to_finish_pausing(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "pause-transition.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    app.processEvents()

    class FakeSynchronizedEngine:
        def __init__(self):
            self.is_playing = True

    engine = FakeSynchronizedEngine()
    original_engine = widget._playback_engine
    applied: list[bool] = []
    try:
        widget._playback_engine = engine
        widget._playing = False
        widget._apply_deferred_assets = lambda: applied.append(True)

        widget._schedule_deferred_assets()
        app.processEvents()

        assert applied == []
        assert widget._deferred_apply_scheduled is True

        engine.is_playing = False
        widget._try_apply_deferred_assets()

        assert applied == [True]
        assert widget._deferred_apply_scheduled is False
    finally:
        widget._playback_engine = original_engine
        widget.shutdown()
        widget.close()


def test_preview_replacement_closes_the_retired_synchronized_stream(tmp_path: Path):
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
                original_wav=source,
                instrument_wavs={"acoustic_piano": first_midi},
                midi_gain_db=0.0,
            ),
        )
        app.processEvents()
        retired_source = widget._playback_engine._source
        retired_sink = widget._playback_engine._sink
        assert retired_source is not None
        assert retired_sink is not None

        widget._on_preview_ready(
            2,
            MuscriptorPreviewAssets(
                notes=(note,),
                duration=1.0,
                transcription_wav=second_midi,
                original_wav=source,
                instrument_wavs={"acoustic_piano": second_midi},
                midi_gain_db=0.0,
            ),
        )

        assert retired_source.isOpen() is False
        assert widget._playback_engine._source is not retired_source
        assert widget._playback_engine._sink is not retired_sink
        assert widget._playback_engine.output_stream_count == 1
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        first_midi.unlink()
        assert not first_midi.exists()
    finally:
        app.processEvents()
        widget.shutdown()
        widget.close()


def test_follow_playhead_ignores_audio_clock_rollback_and_explicit_seek_can_move_left(
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

    class FakeSynchronizedEngine:
        is_configured = True
        is_playing = True

        def __init__(self, positions: list[float]):
            self._positions = iter(positions)
            self.seek_positions: list[float] = []

        @property
        def position_seconds(self) -> float:
            return next(self._positions)

        def seek(self, position: float) -> None:
            self.seek_positions.append(position)

    engine = FakeSynchronizedEngine([8.3, 8.27, 8.34])
    original_engine = widget._playback_engine
    try:
        widget._playback_engine = engine
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
        widget._playing = False
        widget.seek(2.0)
        assert widget._position_ms == 2_000
        assert engine.seek_positions == [2.0]
        assert widget.roll_scroll.horizontalScrollBar().value() < previous_scroll
    finally:
        widget._playing = False
        widget._playback_engine = original_engine
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
        widget.pause = lambda: pytest.fail("scrubbing must not pause the audio sink")
        widget._toggle_playback = lambda: pytest.fail("scrubbing must not restart the audio sink")
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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows audio-device regression")
def test_real_synchronized_stream_can_seek_during_playback_without_deadlock(tmp_path: Path):
    source = _silent_wav(tmp_path / "real-synchronized-scrub.wav", 2.0)
    script = f"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from src.core.muscriptor_result_assets import MuscriptorPreviewAssets, MuscriptorRollNote
from src.gui.widgets.muscriptor_result import MuscriptorResultWidget

app = QApplication([])
source = {str(source)!r}
note = MuscriptorRollNote("acoustic_piano", 60, 100, 0.25, 0.75)
widget = MuscriptorResultWidget(source, ["acoustic_piano"])
widget._on_preview_ready(
    1,
    MuscriptorPreviewAssets(
        notes=(note,),
        duration=2.0,
        transcription_wav=source,
        original_wav=source,
        instrument_wavs={{"acoustic_piano": source}},
        midi_gain_db=0.0,
    ),
)
failure = []

def run(action):
    try:
        action()
    except Exception as exc:
        failure.append(f"{{type(exc).__name__}}: {{exc}}")
        app.exit(2)

def start():
    widget._toggle_playback()

def pause_seek_resume():
    widget.pause()
    widget.seek(1.2)
    widget._toggle_playback()

def change_rate_and_rewind():
    widget.speed_spin.setValue(1.1)
    widget.seek(0.0)

def verify():
    if widget._playback_engine.output_stream_count != 1:
        failure.append("result playback created more than one output stream")
    if not widget._playing or not widget._playback_engine.is_playing:
        failure.append("single synchronized stream stopped after seek/rate controls")
    if not 0.05 <= widget._playback_engine.position_seconds <= 0.8:
        failure.append(
            f"rewind did not resume from the shared cursor: "
            f"{{widget._playback_engine.position_seconds}}"
        )
    widget.shutdown()
    widget.close()
    app.quit()

QTimer.singleShot(0, lambda: run(start))
QTimer.singleShot(120, lambda: run(pause_seek_resume))
QTimer.singleShot(260, lambda: run(change_rate_and_rewind))
QTimer.singleShot(650, lambda: run(verify))
app.exec()
if failure:
    raise RuntimeError("; ".join(failure))
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


def test_project_bpm_and_playback_speed_are_bidirectionally_linked(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "editable-bpm-playback.wav", 2.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    widget._configure_synchronized_playback(
        original_wav=source,
        midi_wav=source,
        instrument_wavs={},
        stereo_available=False,
    )
    widget.play_button.setEnabled(True)
    widget._set_playback_duration(2.0)
    widget.set_bpm_context(117.9, 23.0)

    try:
        assert widget.bpm_spin.value() == pytest.approx(23.0)
        assert widget.speed_spin.value() == pytest.approx(23.0 / 117.9, abs=0.001)
        assert widget.speed_spin.isVisible() is False

        widget.show()
        app.processEvents()
        assert widget.speed_spin.isVisible() is True

        widget.speed_spin.setValue(1.25)
        widget.set_bpm_context(117.9, 23.0)
        assert widget.speed_spin.value() == pytest.approx(1.25)
        assert widget.bpm_spin.value() == pytest.approx(147.4)
        assert widget._playback_engine._source is not None
        assert widget._playback_engine._source.playback_rate == pytest.approx(147.4 / 117.9)
        widget._toggle_playback()
        widget.bpm_spin.setValue(132.5)
        app.processEvents()

        assert widget._playing is True
        assert widget._bpm_user_overridden is True
        assert widget.bpm_spin.value() == pytest.approx(132.5)
        assert widget.speed_spin.value() == pytest.approx(132.5 / 117.9, abs=0.001)
        assert widget._playback_engine._source.playback_rate == pytest.approx(132.5 / 117.9)
        assert widget._editor_grid_seconds() == pytest.approx(60.0 / 117.9 * 4.0 / 32.0)

        widget.set_detected_bpm(90.0)

        assert widget.bpm_spin.value() == pytest.approx(132.5)
        assert widget._detected_bpm == pytest.approx(117.9)
        assert widget.speed_spin.value() == pytest.approx(132.5 / 117.9, abs=0.001)

        widget.speed_spin.setValue(0.75)
        app.processEvents()

        assert widget.bpm_spin.value() == pytest.approx(88.4)
        assert widget.speed_spin.value() == pytest.approx(0.75)
        assert widget._playback_engine._source.playback_rate == pytest.approx(88.4 / 117.9)
    finally:
        widget.shutdown()
        widget.close()


def test_midi_download_uses_project_bpm_and_truly_changes_playback_speed(
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
    assert widget.speed_spin.value() == pytest.approx(132.5 / 117.9, abs=0.001)
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
    duration_scale = 117.9 / 132.5
    assert downloaded_note.start == pytest.approx(
        source_note.start * duration_scale,
        abs=60.0 / 132.5 / downloaded.ticks_per_beat,
    )
    assert downloaded_note.end == pytest.approx(
        source_note.end * duration_scale,
        abs=60.0 / 132.5 / downloaded.ticks_per_beat,
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
    assert downloaded_note_off.bytes() == source_note_off.bytes()
    assert downloaded_note_off.time == source_note_off.time
    assert mido.tempo2bpm(
        next(
            message.tempo
            for midi_track in mido.MidiFile(source_midi).tracks
            for message in midi_track
            if message.is_meta and message.type == "set_tempo"
        )
    ) == pytest.approx(117.9, abs=0.001)
    app.processEvents()


def test_custom_bpm_result_is_not_applied_twice_to_preview_or_download(tmp_path: Path):
    """A pipeline result already carries target tempo on the reference tick grid."""

    pipeline_result = tmp_path / "pipeline-77.9.mid"
    preview = tmp_path / "preview-117.9.mid"
    download = tmp_path / "download-60.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(77.9), time=0))
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(pipeline_result)
    source_ticks = non_tempo_event_tick_fingerprint(mido.MidiFile(pipeline_result))

    _export_midi_with_bpm(pipeline_result, preview, 117.9)
    _export_midi_at_project_speed(pipeline_result, download, 117.9, 60.0)

    preview_midi = mido.MidiFile(preview)
    downloaded_midi = mido.MidiFile(download)
    assert non_tempo_event_tick_fingerprint(preview_midi) == source_ticks
    assert non_tempo_event_tick_fingerprint(downloaded_midi) == source_ticks
    assert preview_midi.length == pytest.approx(60.0 / 117.9, abs=0.001)
    assert downloaded_midi.length == pytest.approx(1.0, abs=0.001)


def test_edited_midi_export_replaces_notes_and_preserves_pedal_and_markers(
    tmp_path: Path,
):
    source = tmp_path / "editor-source.mid"
    destination = tmp_path / "editor-published.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    conductor.append(mido.MetaMessage("marker", text="keep-me", time=120))
    conductor.append(mido.Message("sysex", data=(0x7D, 0x01), time=0))
    midi.tracks.append(conductor)
    notes = mido.MidiTrack()
    notes.append(mido.MetaMessage("track_name", name="Piano", time=0))
    notes.append(mido.Message("program_change", channel=0, program=0, time=0))
    notes.append(mido.Message("control_change", channel=0, control=64, value=127, time=0))
    notes.append(mido.Message("note_on", channel=0, note=60, velocity=90, time=0))
    notes.append(mido.Message("control_change", channel=0, control=64, value=0, time=240))
    notes.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=240))
    midi.tracks.append(notes)
    midi.save(source)

    published = export_edited_midi(
        source,
        destination,
        (
            MuscriptorRollNote(
                instrument="gm:000",
                pitch=64,
                velocity=77,
                start=0.25,
                end=0.75,
                program=0,
                is_drum=False,
                track_index=1,
                channel=0,
            ),
        ),
        reference_bpm=120,
        target_bpm=90,
    )

    assert published == destination.resolve()
    edited = mido.MidiFile(published)
    assert [
        message.text for track in edited.tracks for message in track if message.type == "marker"
    ] == ["keep-me"]
    assert [
        (message.control, message.value)
        for track in edited.tracks
        for message in track
        if message.type == "control_change"
    ] == [(64, 127), (64, 0)]
    assert [
        tuple(message.data)
        for track in edited.tracks
        for message in track
        if message.type == "sysex"
    ] == [(0x7D, 0x01)]
    assert [
        mido.tempo2bpm(message.tempo)
        for track in edited.tracks
        for message in track
        if message.type == "set_tempo"
    ] == pytest.approx([90.0], abs=0.001)

    absolute_tick = 0
    note_events = []
    for message in edited.tracks[1]:
        absolute_tick += message.time
        if message.type in {"note_on", "note_off"}:
            note_events.append((message.type, message.note, message.velocity, absolute_tick))
    assert note_events == [
        ("note_on", 64, 77, 240),
        ("note_off", 64, 0, 720),
    ]
    source_times, _source_resolutions = non_tempo_event_time_fingerprint(mido.MidiFile(source))
    edited_times, edited_resolutions = non_tempo_event_time_fingerprint(edited)
    source_retained = [
        (payload, seconds)
        for payload, seconds in source_times[0] + source_times[1]
        if payload[0] not in {0x80, 0x90, 0xC0}
    ]
    edited_retained = [
        (payload, seconds)
        for payload, seconds in edited_times[0] + edited_times[1]
        if payload[0] not in {0x80, 0x90, 0xC0}
    ]
    assert [payload for payload, _seconds in edited_retained] == [
        payload for payload, _seconds in source_retained
    ]
    for (_payload, source_seconds), (_payload, edited_seconds) in zip(
        source_retained,
        edited_retained,
    ):
        assert edited_seconds == pytest.approx(
            source_seconds * 120.0 / 90.0,
            abs=max(edited_resolutions),
        )
    parsed = read_midi_roll_notes(published)
    assert len(parsed) == 1
    assert parsed[0].track_index == 1
    assert parsed[0].channel == 0


def test_edited_midi_export_preserves_nested_same_pitch_notes_on_spare_channel(
    tmp_path: Path,
):
    source = tmp_path / "nested-overlap-source.mid"
    destination = tmp_path / "nested-overlap-edited.mid"
    bpm = 117.9
    ticks_per_beat = 480
    ticks_per_second = ticks_per_beat * bpm / 60.0
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    midi.tracks.append(conductor)
    notes_track = mido.MidiTrack()
    notes_track.append(mido.Message("program_change", channel=0, program=0, time=0))
    notes_track.append(mido.Message("control_change", channel=0, control=64, value=127, time=0))
    midi.tracks.append(notes_track)
    midi.save(source)

    expected_ticks = [(100, 500), (200, 400)]
    edited_notes = tuple(
        MuscriptorRollNote(
            instrument="gm:000",
            pitch=62,
            velocity=100,
            start=start_tick / ticks_per_second,
            end=end_tick / ticks_per_second,
            program=0,
            track_index=1,
            channel=0,
        )
        for start_tick, end_tick in expected_ticks
    )

    export_edited_midi(
        source,
        destination,
        edited_notes,
        reference_bpm=bpm,
        target_bpm=bpm,
    )

    published = mido.MidiFile(destination)
    active: dict[tuple[int, int], tuple[int, int]] = {}
    actual_ticks: list[tuple[int, int, int]] = []
    for track in published.tracks:
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            if message.type == "note_on" and message.velocity > 0:
                active[(message.channel, message.note)] = (
                    absolute_tick,
                    message.velocity,
                )
            elif message.type in {"note_on", "note_off"}:
                start_tick, velocity = active.pop((message.channel, message.note))
                assert velocity == 100
                actual_ticks.append((start_tick, absolute_tick, message.channel))

    assert [(start, end) for start, end, _channel in sorted(actual_ticks)] == expected_ticks
    assert {channel for _start, _end, channel in actual_ticks} == {0, 1}
    assert [
        (message.channel, message.control, message.value)
        for track in published.tracks
        for message in track
        if message.type == "control_change"
    ] == [(0, 64, 127), (1, 64, 127)]

    professional_reader = pretty_midi.PrettyMIDI(str(destination))
    professional_notes = sorted(
        (note for instrument in professional_reader.instruments for note in instrument.notes),
        key=lambda note: (note.start, note.end),
    )
    assert len(professional_notes) == 2
    assert [
        (round(note.start * ticks_per_second), round(note.end * ticks_per_second))
        for note in professional_notes
    ] == expected_ticks


def test_edited_midi_export_does_not_invent_unsupported_performance_events(
    tmp_path: Path,
):
    source = tmp_path / "notes-only-source.mid"
    destination = tmp_path / "notes-only-edited.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=6,
            denominator=8,
            clocks_per_click=36,
            time=0,
        )
    )
    track.append(mido.Message("note_on", channel=0, note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(source)

    export_edited_midi(
        source,
        destination,
        (
            MuscriptorRollNote(
                instrument="gm:000",
                pitch=64,
                velocity=77,
                start=0.25,
                end=0.75,
                program=0,
                is_drum=False,
                track_index=0,
                channel=0,
            ),
        ),
        reference_bpm=120,
        target_bpm=120,
    )

    unsupported_types = {
        "aftertouch",
        "control_change",
        "marker",
        "pitchwheel",
        "polytouch",
        "sysex",
    }
    actual_types = {
        message.type for midi_track in mido.MidiFile(destination).tracks for message in midi_track
    }
    assert actual_types.isdisjoint(unsupported_types)


def test_edited_midi_export_rejects_invalid_drum_channel_without_publishing(
    tmp_path: Path,
):
    source = tmp_path / "invalid-editor-source.mid"
    destination = tmp_path / "must-not-exist.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack())
    midi.save(source)
    sentinel = b"existing user destination"
    destination.write_bytes(sentinel)

    with pytest.raises(ValueError, match="Invalid edited MIDI note"):
        export_edited_midi(
            source,
            destination,
            (
                MuscriptorRollNote(
                    instrument="drums",
                    pitch=36,
                    velocity=100,
                    start=0.0,
                    end=0.5,
                    is_drum=True,
                    track_index=0,
                    channel=0,
                ),
            ),
            reference_bpm=120,
            target_bpm=120,
        )

    assert destination.read_bytes() == sentinel


def test_edit_mode_note_selection_and_empty_click_never_request_transport_seek():
    app = QApplication.instance() or QApplication([])
    roll = _PianoRollCanvas()
    note = MuscriptorRollNote("acoustic_piano", 60, 90, 0.5, 1.0)
    roll.set_notes((note,), duration=2.0)
    roll.set_editable(True)
    roll.show()
    app.processEvents()
    seeks: list[float] = []
    roll.seek_requested.connect(seeks.append)

    note_x = round(roll.x_for_time_float(0.75))
    note_y = (108 - note.pitch) * 7 + 3
    QTest.mouseClick(
        roll,
        Qt.MouseButton.LeftButton,
        pos=QPoint(note_x, note_y),
    )
    assert roll.selected_indices == (0,)
    assert seeks == []

    empty_y = (108 - (note.pitch + 1)) * 7 + 3
    QTest.mouseClick(
        roll,
        Qt.MouseButton.LeftButton,
        pos=QPoint(note_x, empty_y),
    )
    assert roll.selected_indices == ()
    assert seeks == []
    roll.close()


def test_desktop_midi_editor_add_delete_undo_redo_and_reset(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    audio = _silent_wav(tmp_path / "editor-controls.wav", 1.0)
    original = MuscriptorRollNote(
        "acoustic_piano",
        60,
        90,
        0.0,
        0.5,
        program=0,
        is_drum=False,
        track_index=0,
        channel=0,
    )
    widget = MuscriptorResultWidget(str(audio), ["acoustic_piano"])
    try:
        widget.set_bpm_context(120.0, 120.0)
        widget._begin_editor_session((original,), 1.0)
        widget.edit_toggle.setChecked(True)
        widget._add_editor_note(0.5, 64)
        assert [note.pitch for note in widget._edited_notes] == [60, 64]
        assert widget.roll.selected_index == 1

        widget.edit_velocity_spin.setValue(73)
        assert widget._edited_notes[1].velocity == 73
        widget._delete_selected_editor_note()
        assert widget._edited_notes == (original,)

        widget._undo_editor_notes()
        assert [note.pitch for note in widget._edited_notes] == [60, 64]
        assert widget._edited_notes[1].velocity == 73
        widget._redo_editor_notes()
        assert widget._edited_notes == (original,)

        widget._undo_editor_notes()
        widget._reset_editor_notes()
        assert widget._edited_notes == (original,)
        assert widget.edit_reset_button.isEnabled() is False
        assert widget.edit_summary_label.text()
        assert widget.edit_summary_label.wordWrap() is False
        assert widget.edit_summary_label.minimumHeight() >= 32
        assert all(
            button.minimumHeight() >= 34
            for button in (
                widget.edit_toggle,
                widget.edit_add_button,
                widget.edit_delete_button,
                widget.edit_undo_button,
                widget.edit_redo_button,
                widget.edit_reset_button,
                widget.edit_select_all_button,
                widget.edit_cut_button,
                widget.edit_copy_button,
                widget.edit_paste_button,
                widget.edit_duplicate_button,
                widget.edit_quantize_button,
            )
        )
        app.processEvents()
    finally:
        widget.shutdown()
        widget.close()


def test_desktop_midi_edit_immediately_disables_stale_audio_until_current_render_applies(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    source_audio = _silent_wav(tmp_path / "edit-audio-source.wav", 1.0)
    original_render = _tone_wav(tmp_path / "original-render.wav", 1.0, amplitude=0.1)
    edited_render = _silent_wav(tmp_path / "edited-render.wav", 1.0)
    source_midi = tmp_path / "edit-audio-source.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.Message("note_on", channel=0, note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(source_midi)
    note = read_midi_roll_notes(source_midi)[0]
    original_assets = muscriptor_result_assets.MuscriptorPlaybackAssets(
        notes=(note,),
        duration=1.0,
        transcription_wav=original_render,
        live_transcription_wav=original_render,
        original_wav=source_audio,
        stereo_mix_wav=original_render,
        original_left_wav=source_audio,
        transcription_right_wav=original_render,
        instrument_wavs={note.instrument: original_render},
        instrument_right_wavs={note.instrument: original_render},
        midi_gain_db=0.0,
    )
    edited_assets = muscriptor_result_assets.MuscriptorPlaybackAssets(
        notes=(),
        duration=1.0,
        transcription_wav=edited_render,
        live_transcription_wav=edited_render,
        original_wav=source_audio,
        stereo_mix_wav=edited_render,
        original_left_wav=source_audio,
        transcription_right_wav=edited_render,
        instrument_wavs={},
        instrument_right_wavs={},
        midi_gain_db=0.0,
    )
    widget = MuscriptorResultWidget(str(source_audio), ["acoustic_piano"])
    try:
        widget._midi_path = str(source_midi)
        widget.set_detected_bpm(120.0)
        widget._apply_final_assets(original_assets)
        assert widget.play_button.isEnabled()
        assert widget._playback_engine._source is not None
        assert float(np.max(np.abs(widget._playback_engine._source._midi_mix))) > 0.0

        widget._record_editor_commit((note,), ())

        generation = widget._edit_asset_generation
        assert generation == 1
        assert not widget.play_button.isEnabled()
        assert not widget.playback_slider.isEnabled()
        assert widget._assets is original_assets

        edited_output_dir = tmp_path / "generation-000001"
        edited_output_dir.mkdir()
        shutil.copy2(source_midi, edited_output_dir / "edited-source-tempo.mid")
        widget._apply_editor_audio_assets(
            generation,
            edited_assets,
            output_dir=edited_output_dir,
            restored_original=False,
        )
        app.processEvents()

        assert widget._assets is edited_assets
        assert widget._edited_notes == ()
        assert widget._detected == []
        assert widget.play_button.isEnabled()
        assert widget.playback_slider.isEnabled()
        assert widget._playback_engine._source is not None
        assert float(np.max(np.abs(widget._playback_engine._source._midi_mix))) == 0.0
        with pytest.raises(RuntimeError, match="stale edited MIDI audio"):
            widget._apply_editor_audio_assets(
                generation - 1,
                original_assets,
                output_dir=None,
                restored_original=True,
            )
    finally:
        widget.shutdown()
        widget.close()


def test_desktop_editor_failure_keeps_full_diagnostic_in_log_but_bounds_status_label(
    tmp_path: Path,
    caplog,
):
    app = QApplication.instance() or QApplication([])
    source_audio = _silent_wav(tmp_path / "bounded-editor-error.wav", 1.0)
    widget = MuscriptorResultWidget(str(source_audio), ["acoustic_piano"])
    full_error = "Edited MIDI note verification failed: expected=Counter(" + "x" * 20_000 + ")"
    widget._edit_asset_generation = 7
    try:
        with caplog.at_level("ERROR"):
            widget._on_editor_audio_failed(7, full_error)
        visible = widget.playback_status_label.text()
        assert len(visible) < 500
        assert visible.endswith("…")
        assert "x" * 1_000 not in visible
        assert any(full_error in record.getMessage() for record in caplog.records)
    finally:
        widget.shutdown()
        widget.close()
        app.processEvents()


def test_edited_midi_download_publishes_current_notes_and_exact_77_9_bpm(
    tmp_path: Path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    assert QApplication.instance() is app
    source_audio = _silent_wav(tmp_path / "edited-download-source.wav", 2.0)
    source_midi = tmp_path / "edited-download-source.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(117.9), time=0))
    conductor.append(mido.MetaMessage("marker", text="retain-marker", time=0))
    midi.tracks.append(conductor)
    notes_track = mido.MidiTrack()
    notes_track.append(mido.Message("program_change", channel=0, program=0, time=0))
    notes_track.append(mido.Message("control_change", channel=0, control=64, value=127, time=0))
    notes_track.append(mido.Message("note_on", channel=0, note=60, velocity=91, time=37))
    notes_track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=443))
    notes_track.append(mido.Message("note_on", channel=0, note=67, velocity=82, time=480))
    notes_track.append(mido.Message("note_off", channel=0, note=67, velocity=0, time=480))
    notes_track.append(mido.Message("control_change", channel=0, control=64, value=0, time=0))
    midi.tracks.append(notes_track)
    midi.save(source_midi)

    original_notes = read_midi_roll_notes(source_midi)
    destination = tmp_path / "current-edit-77.9BPM.mid"
    widget = MuscriptorResultWidget(str(source_audio), ["acoustic_piano"])
    widget._midi_path = str(source_midi)
    widget.set_detected_bpm(117.9)
    widget._original_edit_notes = original_notes
    widget._edited_notes = original_notes[:1]
    widget.bpm_spin.setValue(77.9)
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
        message for track in downloaded.tracks for message in track if message.type == "set_tempo"
    ]
    assert len(tempo_messages) == 2
    assert {message.tempo for message in tempo_messages} == {mido.bpm2tempo(77.9)}
    assert mido.tempo2bpm(tempo_messages[0].tempo) == pytest.approx(77.9, abs=0.001)
    assert [
        (note.pitch, note.velocity, note.track_index, note.channel)
        for note in read_midi_roll_notes(destination)
    ] == [
        (
            original_notes[0].pitch,
            original_notes[0].velocity,
            original_notes[0].track_index,
            original_notes[0].channel,
        )
    ]
    assert [
        message.text for track in downloaded.tracks for message in track if message.type == "marker"
    ] == ["retain-marker"]
    assert [
        (message.control, message.value)
        for track in downloaded.tracks
        for message in track
        if message.type == "control_change"
    ] == [(64, 127), (64, 0)]
    # MuScriptor exports intentionally repeat the canonical tempo on note
    # tracks for MuseScore compatibility. pretty_midi warns about that
    # non-standard type-1 extension, so assert the warning as part of the
    # compatibility contract instead of leaking unclassified warning noise.
    with pytest.warns(
        RuntimeWarning,
        match="Tempo, Key or Time signature change events found on non-zero tracks",
    ):
        pretty_result = pretty_midi.PrettyMIDI(str(destination))
    tempo_times, tempo_values = pretty_result.get_tempo_changes()
    assert tempo_times.tolist() == [0.0]
    assert tempo_values.tolist() == pytest.approx([77.9], abs=0.001)


def test_desktop_midi_editor_uses_daw_multi_select_clipboard_and_nudge_shortcuts(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    audio = _silent_wav(tmp_path / "editor-shortcuts.wav", 4.0)
    notes = (
        MuscriptorRollNote(
            "acoustic_piano",
            60,
            90,
            0.13,
            0.63,
            program=0,
            is_drum=False,
            track_index=0,
            channel=0,
        ),
        MuscriptorRollNote(
            "acoustic_piano",
            64,
            80,
            1.13,
            1.63,
            program=0,
            is_drum=False,
            track_index=0,
            channel=0,
        ),
    )
    widget = MuscriptorResultWidget(str(audio), ["acoustic_piano"])
    widget.resize(1100, 900)
    widget.show()
    try:
        widget.set_bpm_context(120.0, 120.0)
        widget.roll.set_notes(notes, duration=4.0)
        widget._begin_editor_session(notes, 4.0)
        widget.edit_toggle.setChecked(True)
        widget.roll.setFocus()
        app.processEvents()

        QTest.keyClick(
            widget.roll,
            Qt.Key.Key_A,
            Qt.KeyboardModifier.ControlModifier,
        )
        assert widget.roll.selected_indices == (0, 1)

        QTest.keyClick(
            widget.roll,
            Qt.Key.Key_C,
            Qt.KeyboardModifier.ControlModifier,
        )
        assert len(widget._edit_clipboard) == 2

        QTest.keyClick(widget.roll, Qt.Key.Key_Right)
        assert [note.start for note in widget._edited_notes] == pytest.approx([0.1925, 1.1925])
        QTest.keyClick(
            widget.roll,
            Qt.Key.Key_Up,
            Qt.KeyboardModifier.ShiftModifier,
        )
        assert [note.pitch for note in widget._edited_notes] == [72, 76]

        QTest.keyClick(
            widget.roll,
            Qt.Key.Key_Q,
            Qt.KeyboardModifier.AltModifier,
        )
        assert [note.start for note in widget._edited_notes] == pytest.approx([0.1875, 1.1875])

        QTest.keyClick(
            widget.roll,
            Qt.Key.Key_B,
            Qt.KeyboardModifier.ControlModifier,
        )
        assert len(widget._edited_notes) == 4
        assert widget.roll.selected_indices == (2, 3)

        QTest.keyClick(
            widget.roll,
            Qt.Key.Key_X,
            Qt.KeyboardModifier.ControlModifier,
        )
        assert len(widget._edited_notes) == 2
        QTest.keyClick(
            widget.roll,
            Qt.Key.Key_V,
            Qt.KeyboardModifier.ControlModifier,
        )
        assert len(widget._edited_notes) == 4

        QTest.keyClick(
            widget.roll,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        assert len(widget._edited_notes) == 2
        app.processEvents()
    finally:
        widget.shutdown()
        widget.close()


def test_desktop_roll_and_legend_share_one_high_contrast_instrument_palette(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    widget = MuscriptorResultWidget(
        str(tmp_path / "colors.wav"),
        ["string_ensemble", "acoustic_piano"],
    )
    notes = (
        MuscriptorRollNote("acoustic_piano", 60, 90, 0.0, 0.5),
        MuscriptorRollNote("string_ensemble", 67, 90, 0.5, 1.0),
    )
    try:
        widget.roll.set_notes(notes, duration=1.0)
        widget._detected = ["acoustic_piano", "string_ensemble"]
        widget._rebuild_instrument_rows()

        assert _COLORS == _INSTRUMENT_COLORS
        assert _INSTRUMENT_COLORS[2] == "#7bd88f"
        assert widget.roll._colors["string_ensemble"].name() == "#4a9eff"
        assert widget.roll._colors["acoustic_piano"].name() == "#ff8d66"
        for index, instrument in enumerate(widget.selected_instruments):
            swatch = widget._instrument_rows[instrument].findChildren(QLabel)[0]
            assert _INSTRUMENT_COLORS[index] in swatch.styleSheet()
        app.processEvents()
    finally:
        widget.shutdown()
        widget.close()


def test_desktop_piano_roll_splitter_expands_the_note_view_upward(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    widget = MuscriptorResultWidget(str(tmp_path / "splitter.wav"), ["acoustic_piano"])
    widget.resize(1100, 1000)
    widget.show()
    app.processEvents()
    try:
        assert widget.result_splitter.widget(0) is widget.result_controls_panel
        assert widget.result_splitter.widget(1) is widget.roll_panel
        assert widget.result_splitter.isCollapsible(0)
        assert not widget.result_splitter.isCollapsible(1)
        assert widget.source_label.parentWidget() is widget.result_controls_panel
        assert widget.editor_panel.parentWidget() is widget.result_controls_panel

        before = widget.result_splitter.sizes()
        handle = widget.result_splitter.handle(1)
        start = handle.rect().center()
        finish = QPoint(start.x(), start.y() - widget.height())
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(handle, finish, delay=20)
        QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=finish)
        app.processEvents()
        maximized = widget.result_splitter.sizes()

        assert len(before) == len(maximized) == 2
        assert maximized[0] == 0
        assert maximized[1] > before[1]
        assert widget.result_controls_panel.height() == 0
        assert handle.geometry().top() == 0

        restore_start = handle.rect().center()
        restore_finish = QPoint(restore_start.x(), restore_start.y() + 260)
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=restore_start)
        QTest.mouseMove(handle, restore_finish, delay=20)
        QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=restore_finish)
        app.processEvents()
        restored = widget.result_splitter.sizes()

        assert restored[0] >= 240
        assert restored[1] < maximized[1]
        assert widget.result_controls_panel.height() >= 240
        assert widget.result_splitter.handle(1).toolTip() == t(
            "muscriptor_result.editor_resize_hint"
        )
    finally:
        widget.shutdown()
        widget.close()


def test_desktop_view_zoom_grows_notes_beyond_one_x_and_is_not_playback_speed(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "view-zoom.wav", 1.0)
    widget = MuscriptorResultWidget(str(source), ["acoustic_piano"])
    widget.resize(1200, 800)
    widget.show()
    widget.roll.set_notes(
        (MuscriptorRollNote("acoustic_piano", 60, 100, 1.0, 2.0),),
        duration=12.0,
    )
    widget.set_bpm_context(120.0, 120.0)
    app.processEvents()

    try:
        default_width = widget.roll.x_for_time_float(2.0) - widget.roll.x_for_time_float(1.0)
        default_pixels_per_second = widget.roll.pixels_per_second

        widget.speed_spin.setValue(1.25)
        app.processEvents()
        assert widget.roll.pixels_per_second == pytest.approx(default_pixels_per_second)
        assert (
            widget.roll.x_for_time_float(2.0) - widget.roll.x_for_time_float(1.0)
        ) == pytest.approx(default_width)

        assert widget.roll_zoom_spin.maximum() > 1.0
        widget.roll_zoom_spin.setValue(2.0)
        app.processEvents()
        doubled_width = widget.roll.x_for_time_float(2.0) - widget.roll.x_for_time_float(1.0)
        assert widget.roll_zoom_spin.value() == pytest.approx(2.0)
        assert widget.roll.pixels_per_second == pytest.approx(default_pixels_per_second * 2.0)
        assert doubled_width == pytest.approx(default_width * 2.0)
        assert widget.speed_label.text() == t("muscriptor_result.playback_speed_label")
        assert widget.roll_zoom_label.text() == t("muscriptor_result.editor_view_zoom")
    finally:
        widget.shutdown()
        widget.close()


def test_desktop_piano_roll_draws_beat_lines_downbeats_and_alternating_bars():
    app = QApplication.instance() or QApplication([])
    roll = _PianoRollCanvas()
    roll.set_notes((), duration=4.0)
    roll.set_beat_grid(
        (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
        (0.0, 1.0, 2.0, 3.0),
    )
    roll.show()
    app.processEvents()
    try:
        image = roll.grab().toImage()
        beat_x = round(roll.x_for_time_float(0.5))
        downbeat_x = round(roll.x_for_time_float(1.0))
        plain_x = round(roll.x_for_time_float(0.75))
        shaded_x = round(roll.x_for_time_float(1.75))
        y = 30
        beat_color = image.pixelColor(beat_x, y)
        downbeat_color = image.pixelColor(downbeat_x, y)
        plain_color = image.pixelColor(plain_x, y)
        shaded_color = image.pixelColor(shaded_x, y)
        assert beat_color != plain_color
        assert downbeat_color != beat_color
        assert shaded_color != plain_color
        assert roll._beat_times == pytest.approx((0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5))
        assert roll._downbeat_times == pytest.approx((0.0, 1.0, 2.0, 3.0))
    finally:
        roll.close()


def test_desktop_daw_grid_and_quantize_follow_reference_ticks_not_target_bpm(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    audio = _silent_wav(tmp_path / "reference-grid.wav", 4.0)
    note = MuscriptorRollNote(
        "acoustic_piano",
        60,
        90,
        0.14,
        0.61,
        program=0,
        is_drum=False,
        track_index=0,
        channel=0,
    )
    widget = MuscriptorResultWidget(str(audio), ["acoustic_piano"])
    widget.resize(1200, 800)
    widget.show()
    try:
        widget.set_bpm_context(120.0, 60.0, time_signature=(3, 4))
        widget.roll.set_notes((note,), duration=4.0)
        widget._begin_editor_session((note,), 4.0)
        widget.edit_toggle.setChecked(True)
        widget.roll.set_selected_index(0)
        assert [
            widget.edit_quantize_grid_combo.itemText(index)
            for index in range(widget.edit_quantize_grid_combo.count())
        ] == ["1/4", "1/8", "1/16", "1/32", "1/64"]
        assert widget.edit_quantize_grid_combo.currentText() == "1/32"
        assert widget._editor_grid_seconds() == pytest.approx(0.0625)
        widget.edit_quantize_button.click()
        app.processEvents()

        assert widget._edited_notes[0].start == pytest.approx(0.125)
        assert widget._edited_notes[0].end == pytest.approx(0.625)
        assert widget.roll._daw_reference_bpm == pytest.approx(120.0)
        assert widget.roll._daw_time_signature == (3, 4)

        quantized_note = widget._edited_notes[0]
        widget.edit_quantize_grid_combo.setCurrentText("1/8")
        app.processEvents()
        assert widget._editor_grid_seconds() == pytest.approx(0.25)
        assert widget._edited_notes[0] == quantized_note

        widget.bpm_spin.setValue(90.0)
        app.processEvents()
        assert widget._editor_grid_seconds() == pytest.approx(0.25)
        assert widget.roll._daw_reference_bpm == pytest.approx(120.0)

        image = widget.roll.grab().toImage()
        selected_grid_x = round(widget.roll.x_for_time_float(0.25))
        beat_x = round(widget.roll.x_for_time_float(0.5))
        bar_x = round(widget.roll.x_for_time_float(1.5))
        plain_x = round(widget.roll.x_for_time_float(0.2))
        y = 30

        def colors_around(x: int) -> set[str]:
            return {image.pixelColor(value, y).name() for value in range(x - 2, x + 3)}

        plain_color = image.pixelColor(plain_x, y).name()
        assert "#263d59" in colors_around(selected_grid_x)
        assert "#36506f" in colors_around(beat_x)
        assert "#78aee8" in colors_around(bar_x)
        assert plain_color not in {"#263d59", "#36506f", "#78aee8"}
    finally:
        widget.shutdown()


def test_desktop_quantize_scope_defaults_to_all_tracks_and_selected_is_explicit(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    audio = _silent_wav(tmp_path / "quantize-scope.wav", 4.0)
    notes = (
        MuscriptorRollNote(
            "acoustic_piano",
            60,
            90,
            0.14,
            0.61,
            program=0,
            is_drum=False,
            track_index=0,
            channel=0,
        ),
        MuscriptorRollNote(
            "electric_guitar_clean",
            64,
            80,
            0.19,
            0.62,
            program=27,
            is_drum=False,
            track_index=3,
            channel=2,
        ),
    )
    widget = MuscriptorResultWidget(
        str(audio),
        ["acoustic_piano", "electric_guitar_clean"],
    )
    try:
        widget.set_bpm_context(120.0, 120.0)
        widget.roll.set_notes(notes, duration=4.0)
        widget._begin_editor_session(notes, 4.0)
        widget.edit_toggle.setChecked(True)
        widget.roll.set_selected_indices((), primary=None)
        app.processEvents()

        assert [
            widget.edit_quantize_scope_combo.itemData(index)
            for index in range(widget.edit_quantize_scope_combo.count())
        ] == ["all_tracks", "selected_notes"]
        assert widget.edit_quantize_scope_combo.currentData() == "all_tracks"
        assert widget.edit_quantize_button.isEnabled()

        widget.edit_quantize_button.click()
        app.processEvents()
        assert widget._edited_notes[0].start == pytest.approx(0.125)
        assert widget._edited_notes[0].end == pytest.approx(0.625)
        assert widget._edited_notes[1].start == pytest.approx(0.1875)
        assert widget._edited_notes[1].end == pytest.approx(0.625)

        widget.edit_undo_button.click()
        app.processEvents()
        assert widget._edited_notes == notes
        widget.edit_quantize_scope_combo.setCurrentIndex(
            widget.edit_quantize_scope_combo.findData("selected_notes")
        )
        app.processEvents()
        assert not widget.edit_quantize_button.isEnabled()
        widget.roll.set_selected_index(0)
        app.processEvents()
        assert widget.edit_quantize_button.isEnabled()

        widget.edit_quantize_button.click()
        app.processEvents()
        assert widget._edited_notes[0].start == pytest.approx(0.125)
        assert widget._edited_notes[0].end == pytest.approx(0.625)
        assert widget._edited_notes[1] == notes[1]
    finally:
        widget.shutdown()
        widget.close()


@pytest.mark.parametrize(
    ("editor", "typed_text", "expected_bpm"),
    [
        ("bpm", "BPM 60.0", 60.0),
        ("speed", "0.500x", 59.0),
    ],
)
def test_midi_download_uses_bidirectionally_linked_bpm_or_speed(
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
    control.interpretText()
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
        expected_rate = expected_bpm / 117.9
        assert widget._result_playback_rate() == pytest.approx(expected_rate)
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
        abs=60.0 / expected_bpm / downloaded.ticks_per_beat,
    )
    app.processEvents()


def test_midi_download_preserves_reference_ticks_and_changes_real_duration(
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

    destination = tmp_path / "tempo-only.mid"
    _export_midi_at_project_speed(
        source_midi,
        destination,
        117.9,
        77.9,
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

    source_notes = read_midi_roll_notes(source_midi)
    downloaded_notes = read_midi_roll_notes(destination)
    assert [
        (note.track_index, note.channel, note.pitch, note.velocity) for note in downloaded_notes
    ] == [(note.track_index, note.channel, note.pitch, note.velocity) for note in source_notes]
    source_ticks = non_tempo_event_tick_fingerprint(mido.MidiFile(source_midi))
    downloaded_ticks = non_tempo_event_tick_fingerprint(downloaded)
    assert downloaded_ticks == source_ticks
    source_times, source_resolutions = non_tempo_event_time_fingerprint(mido.MidiFile(source_midi))
    downloaded_times, downloaded_resolutions = non_tempo_event_time_fingerprint(downloaded)
    assert len(downloaded_times) == len(source_times)
    for track_index, (source_events, downloaded_events) in enumerate(
        zip(source_times, downloaded_times)
    ):
        assert len(downloaded_events) == len(source_events)
        tolerance = source_resolutions[track_index] + downloaded_resolutions[track_index] + 1e-9
        for (source_payload, source_seconds), (
            downloaded_payload,
            downloaded_seconds,
        ) in zip(source_events, downloaded_events):
            assert downloaded_payload == source_payload
            assert downloaded_seconds == pytest.approx(
                source_seconds * 117.9 / 77.9,
                abs=tolerance,
            )
    assert downloaded.length == pytest.approx(
        mido.MidiFile(source_midi).length * 117.9 / 77.9,
        abs=2 * 60.0 / 77.9 / downloaded.ticks_per_beat,
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


def test_selector_returns_search_input_to_left_edge_after_last_tag_is_removed():
    app = QApplication.instance() or QApplication([])
    selector = MuscriptorInstrumentSelector()
    selector.resize(760, 220)
    selector.show()
    app.processEvents()

    empty_left = selector.search_edit.geometry().left()
    selector.set_selected_instruments(["acoustic_piano"])
    app.processEvents()
    assert selector.search_edit.geometry().left() > empty_left

    QTest.mouseClick(
        selector._tags["acoustic_piano"].remove_button,
        Qt.MouseButton.LeftButton,
    )
    app.processEvents()

    assert selector.selected_instruments() == []
    assert selector.search_edit.geometry().left() == empty_left
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
    assert "var startAt = context.currentTime + 0.02" in MUSCRIPTOR_RESULT_JS
    assert "source.start(startAt, self.position)" in MUSCRIPTOR_RESULT_JS
    assert 'this.playhead.style.transform="translate3d(' in MUSCRIPTOR_RESULT_JS
    assert 'querySelectorAll(".msr-root:not([data-msr-init])")' in MUSCRIPTOR_RESULT_JS


def test_browser_midi_editor_exposes_daw_commands_and_upward_resize_handle():
    assert "msr-resize-handle" in MUSCRIPTOR_RESULT_CSS
    assert "cursor:ns-resize" in MUSCRIPTOR_RESULT_CSS
    assert "startHeight + self.resizeDrag.startY - event.clientY" in MUSCRIPTOR_RESULT_JS
    assert 'key === "b" || key === "d"' in MUSCRIPTOR_RESULT_JS
    assert 'event.altKey && key === "q"' in MUSCRIPTOR_RESULT_JS
    assert 'command && key === "u"' in MUSCRIPTOR_RESULT_JS
    assert "ResultSession.prototype.selectAll" in MUSCRIPTOR_RESULT_JS
    assert "ResultSession.prototype.copySelected" in MUSCRIPTOR_RESULT_JS
    assert "ResultSession.prototype.pasteNotes" in MUSCRIPTOR_RESULT_JS
    assert "ResultSession.prototype.quantizeSelected" in MUSCRIPTOR_RESULT_JS
    assert 'this.quantizeScope === "all_tracks"' in MUSCRIPTOR_RESULT_JS
    assert "this.quantizeScopeSelect" in MUSCRIPTOR_RESULT_JS
    assert "ResultSession.prototype.transformSelected" in MUSCRIPTOR_RESULT_JS
    assert "ResultSession.prototype.setZoomRatio" in MUSCRIPTOR_RESULT_JS
    assert "ResultSession.prototype.setZoomPps" in MUSCRIPTOR_RESULT_JS
    assert "BASE_PPS * ratio" in MUSCRIPTOR_RESULT_JS
    assert "this.zoomInput.value = (this.pps / BASE_PPS).toFixed(2)" in MUSCRIPTOR_RESULT_JS
    assert "msr-zoom-input" in MUSCRIPTOR_RESULT_CSS


def test_browser_midi_workbench_labels_generic_gm_programs():
    state = {
        "audio_path": "C:/tmp/source.wav",
        "playback_audio_path": "C:/tmp/original-live.wav",
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
    assert "original-live.wav" in markup
    assert "source.wav" not in markup


def test_browser_midi_editor_manifest_and_runtime_preserve_full_note_identity():
    state = {
        "audio_path": "C:/tmp/source.wav",
        "playback_audio_path": "C:/tmp/original-live.wav",
        "midi_path": "C:/tmp/result.mid",
        "transcription_wav": "C:/tmp/result.wav",
        "stereo_mix_wav": "C:/tmp/stereo.wav",
        "instrument_wavs": {"gm:024": "C:/tmp/guitar.wav"},
        "selected_instruments": ["gm:024"],
        "detected_instruments": ["gm:024"],
        "notes": [
            {
                "instrument": "gm:024",
                "pitch": 64,
                "velocity": 91,
                "start": 0.25,
                "end": 0.75,
                "program": 24,
                "is_drum": False,
                "track_index": 2,
                "channel": 3,
            }
        ],
        "duration": 1.0,
        "reference_bpm": 117.9,
        "target_bpm": 132.5,
        "time_signature": (3, 4),
        "beat_times": [0.25, 0.75, 1.25],
        "downbeats": [0.25, 1.25],
        "repeat_tempo_per_note_track": True,
        "backend_label": "MIROS",
        "source_track_name": "guitar",
        "preview_api": "./api/render_edited_midi_preview",
        "preview_token": "opaque-preview-token",
    }

    markup = build_muscriptor_result_html(state, lambda key: key, "en_US")

    assert '"referenceBpm": 117.9' in markup
    assert '"targetBpm": 132.5' in markup
    assert '"timeSignature": [3, 4]' in markup
    assert '"beatTimes": [0.25, 0.75, 1.25]' in markup
    assert '"downbeats": [0.25, 1.25]' in markup
    assert '"repeatTempoPerNoteTrack": true' in markup
    assert '"quantizeGrids": ["1/4", "1/8", "1/16", "1/32", "1/64"]' in markup
    assert '"defaultQuantizeGrid": "1/32"' in markup
    assert '"quantizeScopes": ["all_tracks", "selected_notes"]' in markup
    assert '"defaultQuantizeScope": "all_tracks"' in markup
    assert '"previewApi": "./api/render_edited_midi_preview"' in markup
    assert '"previewToken": "opaque-preview-token"' in markup
    assert '"originalUrl": "/gradio_api/file=C%3A/tmp/original-live.wav"' in markup
    assert '"program": 24' in markup
    assert '"track_index": 2' in markup
    assert '"channel": 3' in markup
    assert "buildEditedSmf" in MUSCRIPTOR_RESULT_JS
    assert "pass-through event verification failed" in MUSCRIPTOR_RESULT_JS
    assert "this.undoStack" in MUSCRIPTOR_RESULT_JS
    assert "onPointerMove" in MUSCRIPTOR_RESULT_JS
    assert "downloadEditedMidi" in MUSCRIPTOR_RESULT_JS
    assert "scheduleEditedPreview" in MUSCRIPTOR_RESULT_JS
    assert "renderEditedPreview" in MUSCRIPTOR_RESULT_JS
    assert "Invalid Beat This grid in piano-roll manifest" in MUSCRIPTOR_RESULT_JS
    assert "repeatTempoPerNoteTrack" in MUSCRIPTOR_RESULT_JS
    assert "return 60 / referenceBpm * 4 / Number(match[1])" in MUSCRIPTOR_RESULT_JS
    assert "subdivisionSeconds = this.gridSeconds()" in MUSCRIPTOR_RESULT_JS
    assert "this.quantizeGridSelect" in MUSCRIPTOR_RESULT_JS
    assert "quantizedDuration = Math.max(grid" in MUSCRIPTOR_RESULT_JS
    assert 'painter.fillText((downbeatIndex + 1) + ".1"' in MUSCRIPTOR_RESULT_JS


def test_every_browser_midi_workbench_exposes_current_edit_sheet_export():
    state = {
        "audio_path": "C:/tmp/source.wav",
        "playback_audio_path": "C:/tmp/original-live.wav",
        "midi_path": "C:/tmp/result.mid",
        "transcription_wav": "C:/tmp/result.wav",
        "stereo_mix_wav": "C:/tmp/stereo.wav",
        "instrument_wavs": {"gm:000": "C:/tmp/piano.wav"},
        "selected_instruments": ["gm:000"],
        "detected_instruments": ["gm:000"],
        "notes": [],
        "duration": 1.0,
        "reference_bpm": 120.0,
        "target_bpm": 120.0,
        "time_signature": (4, 4),
        "backend_label": "YourMT3+",
        "sheet_api": "./api/render_sheet_music_export",
        "sheet_token": "opaque-sheet-token",
    }

    markup = build_muscriptor_result_html(state, lambda key: key, "en_US")

    assert '"sheetApi": "./api/render_sheet_music_export"' in markup
    assert '"sheetToken": "opaque-sheet-token"' in markup
    assert "muscriptor_result.download_sheet_music" in markup
    assert "ResultSession.prototype.downloadSheetMusic" in MUSCRIPTOR_RESULT_JS
    assert "fetchEditedMidiBytes" in MUSCRIPTOR_RESULT_JS
    assert "midi_base64" in MUSCRIPTOR_RESULT_JS
    assert "quantize_grid: self.quantizeGrid" in MUSCRIPTOR_RESULT_JS
    assert "self.m.sheetToken" in MUSCRIPTOR_RESULT_JS


def test_browser_midi_editor_builds_verified_smf_and_preserves_controller_events(
    tmp_path: Path,
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node runtime is unavailable for browser MIDI editor validation")
    source = tmp_path / "browser-editor-source.mid"
    destination = tmp_path / "browser-editor-output.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    # Browser receives the pipeline's target-tempo file; its non-tempo ticks
    # are already expressed on the 120 BPM reference grid.
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(90), time=0))
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=6,
            denominator=8,
            clocks_per_click=36,
            time=0,
        )
    )
    track.append(mido.MetaMessage("marker", text="browser-keeps-this", time=10))
    track.append(mido.Message("sysex", data=(0x7D, 0x02), time=0))
    track.append(mido.Message("program_change", channel=2, program=24, time=0))
    track.append(mido.Message("control_change", channel=2, control=64, value=127, time=40))
    track.append(mido.Message("note_on", channel=2, note=60, velocity=90, time=0))
    track.append(mido.Message("control_change", channel=2, control=64, value=0, time=200))
    track.append(mido.Message("note_off", channel=2, note=60, velocity=0, time=230))
    midi.tracks.append(track)
    midi.save(source)
    notes_path = tmp_path / "browser-editor-notes.json"
    notes_path.write_text(
        json.dumps(
            [
                {
                    "instrument": "gm:024",
                    "pitch": 67,
                    "velocity": 76,
                    "start": 0.0375,
                    "end": 0.8375,
                    "program": 24,
                    "is_drum": False,
                    "track_index": 0,
                    "channel": 2,
                },
                {
                    "instrument": "gm:024",
                    "pitch": 67,
                    "velocity": 76,
                    "start": 0.2875,
                    "end": 0.5375,
                    "program": 24,
                    "is_drum": False,
                    "track_index": 0,
                    "channel": 2,
                },
            ]
        ),
        encoding="utf-8",
    )
    harness = tmp_path / "browser-editor-harness.js"
    harness.write_text(
        "\n".join(
            [
                "global.window = {};",
                "global.document = {",
                "  documentElement: {},",
                '  readyState: "loading",',
                "  addEventListener: function () {},",
                "  querySelectorAll: function () { return []; }",
                "};",
                "global.MutationObserver = function () { this.observe = function () {}; };",
                MUSCRIPTOR_RESULT_JS,
                'const fs = require("fs");',
                "const source = fs.readFileSync(process.argv[2]);",
                "const sourceBuffer = source.buffer.slice(",
                "  source.byteOffset, source.byteOffset + source.byteLength",
                ");",
                'const notes = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));',
                "const output = window.musicToMidiMidiEditorRuntime.buildEditedSmf(",
                "  sourceBuffer, notes, 90, 120",
                ");",
                "fs.writeFileSync(process.argv[4], Buffer.from(output));",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [node, str(harness), str(source), str(notes_path), str(destination)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode == 0, completed.stdout
    edited = mido.MidiFile(destination)
    assert [
        message.text
        for midi_track in edited.tracks
        for message in midi_track
        if message.type == "marker"
    ] == ["browser-keeps-this"]
    assert [
        (message.channel, message.control, message.value)
        for midi_track in edited.tracks
        for message in midi_track
        if message.type == "control_change"
    ] == [
        (2, 64, 127),
        (0, 64, 127),
        (2, 64, 0),
        (0, 64, 0),
    ]
    assert [
        tuple(message.data)
        for midi_track in edited.tracks
        for message in midi_track
        if message.type == "sysex"
    ] == [(0x7D, 0x02)]
    assert [
        (message.numerator, message.denominator, message.clocks_per_click)
        for midi_track in edited.tracks
        for message in midi_track
        if message.type == "time_signature"
    ] == [(6, 8, 36)]
    absolute_tick = 0
    active_notes = {}
    note_intervals = []
    for message in edited.tracks[0]:
        absolute_tick += message.time
        if message.type == "note_on" and message.velocity > 0:
            active_notes[(message.channel, message.note)] = absolute_tick
        elif message.type in {"note_on", "note_off"}:
            note_intervals.append(
                (
                    active_notes.pop((message.channel, message.note)),
                    absolute_tick,
                    message.channel,
                )
            )
    assert sorted(note_intervals) == [(36, 804, 2), (276, 516, 0)]
    assert [
        mido.tempo2bpm(message.tempo)
        for midi_track in edited.tracks
        for message in midi_track
        if message.type == "set_tempo"
    ] == pytest.approx([90.0], abs=0.001)

    plain_source = tmp_path / "browser-notes-only-source.mid"
    plain_destination = tmp_path / "browser-notes-only-output.mid"
    plain_midi = mido.MidiFile(type=1, ticks_per_beat=480)
    plain_track = mido.MidiTrack()
    plain_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    plain_track.append(mido.Message("note_on", channel=2, note=60, velocity=90, time=0))
    plain_track.append(mido.Message("note_off", channel=2, note=60, velocity=0, time=480))
    plain_midi.tracks.append(plain_track)
    plain_midi.save(plain_source)
    plain_completed = subprocess.run(
        [
            node,
            str(harness),
            str(plain_source),
            str(notes_path),
            str(plain_destination),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert plain_completed.returncode == 0, plain_completed.stdout
    unsupported_types = {
        "aftertouch",
        "control_change",
        "marker",
        "pitchwheel",
        "polytouch",
        "sysex",
    }
    actual_types = {
        message.type
        for midi_track in mido.MidiFile(plain_destination).tracks
        for message in midi_track
    }
    assert actual_types.isdisjoint(unsupported_types)


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


def test_muscriptor_result_shutdown_unloads_edited_wav_before_temp_cleanup(
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "shutdown-source.wav", 0.2)
    widget = MuscriptorResultWidget(str(source), [], backend_label="MuScriptor-large")
    edit_root = widget._edit_asset_root
    generation_dir = edit_root / "generation-000001"
    generation_dir.mkdir()
    edited_wav = _silent_wav(generation_dir / "midi-live.wav", 0.2)
    widget._configure_synchronized_playback(
        original_wav=edited_wav,
        midi_wav=edited_wav,
        instrument_wavs={},
        stereo_available=False,
    )
    widget._playback_engine.play()
    QTest.qWait(100)
    app.processEvents()

    widget.shutdown()
    app.processEvents()

    assert not edit_root.exists()
    widget.close()


def test_muscriptor_shutdown_closes_synchronized_stream_before_deletion(tmp_path: Path):
    app = QApplication.instance() or QApplication([])
    source = _silent_wav(tmp_path / "retired-source.wav", 0.2)
    widget = MuscriptorResultWidget(str(source), [], backend_label="MuScriptor-large")
    preview_root = widget._preview_root
    generation_dir = preview_root / "generation-000001"
    generation_dir.mkdir()
    preview_wav = _silent_wav(generation_dir / "midi-live.wav", 0.2)
    widget._configure_synchronized_playback(
        original_wav=preview_wav,
        midi_wav=preview_wav,
        instrument_wavs={},
        stereo_available=False,
    )
    retired_source = widget._playback_engine._source
    assert retired_source is not None

    widget._dispose_dynamic_players()
    assert not widget._playback_engine.is_configured
    assert retired_source.isOpen() is False

    widget.shutdown()
    widget.close()
    app.processEvents()

    assert not preview_root.exists()


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


def test_browser_project_bpm_changes_linked_audition_rate(tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node runtime is unavailable for MuScriptor JS validation")
    harness = tmp_path / "browser-bpm-rate.js"
    harness.write_text(
        "\n".join(
            [
                "global.window = {};",
                "global.document = {",
                "  documentElement: {},",
                '  readyState: "loading",',
                "  addEventListener: function () {},",
                "  querySelectorAll: function () { return []; }",
                "};",
                "global.MutationObserver = function () { this.observe = function () {}; };",
                MUSCRIPTOR_RESULT_JS,
                "const runtime = window.musicToMidiMidiEditorRuntime;",
                "console.log(runtime.projectPlaybackRate(76.3, 120));",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [node, str(harness)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode == 0, completed.stdout
    assert float(completed.stdout.strip()) == pytest.approx(120.0 / 76.3)
