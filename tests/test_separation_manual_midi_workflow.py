# ruff: noqa: E402

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.core.pipeline import MusicToMidiPipeline
from src.gui.main_window import MainWindow
from src.gui.widgets.audio_track_mixer import (
    MIDI_ROUTE_MIROS,
    MIDI_ROUTE_PIANO_ARIA_AMT,
    MIDI_ROUTE_PIANO_BYTEDANCE_PEDAL,
    MIDI_ROUTE_PIANO_TRANSKUN,
    MIDI_ROUTE_PIANO_TRANSKUN_V2_AUG,
    MIDI_ROUTE_YOURMT3_PREFIX,
    _YOURMT3_MANUAL_MODELS,
)
from src.gui.workers.separation_worker import SeparationResult, SeparationWorker
from src.models.data_models import Config, MultiInstrumentModel, ProcessingMode


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize(
    ("mode", "expected_names"),
    [
        (ProcessingMode.VOCAL_SPLIT.value, ("vocals", "accompaniment")),
        (
            ProcessingMode.SIX_STEM_SPLIT.value,
            ("bass", "drums", "guitar", "piano", "vocals", "other"),
        ),
    ],
)
def test_separation_worker_outputs_only_real_wavs(
    qapp,
    tmp_path,
    monkeypatch,
    mode,
    expected_names,
):
    source_path = tmp_path / "source.wav"
    source_path.write_bytes(b"source-audio")
    output_dir = tmp_path / "separated"
    separator_calls = []

    class FakeVocalSeparator:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_model_available():
            return True

        def __init__(self, **_kwargs):
            pass

        def set_cancel_check(self, callback):
            self.cancel_check = callback

        def separate(self, audio_path, output_dir, progress_callback=None):
            separator_calls.append((Path(audio_path).resolve(), Path(output_dir).resolve()))
            paths = {}
            for name in expected_names:
                path = Path(output_dir) / f"source_{name}.wav"
                path.write_bytes(name.encode("utf-8"))
                paths[name] = str(path)
            if progress_callback:
                progress_callback(1.0, "separated")
            return paths

    class FakeSixStemSeparator:
        @staticmethod
        def is_available():
            return True

        def __init__(self, **_kwargs):
            pass

        def is_model_available(self):
            return True

        def separate(self, audio_path, output_dir, progress_callback=None):
            separator_calls.append((Path(audio_path).resolve(), Path(output_dir).resolve()))
            paths = {}
            for name in expected_names:
                path = Path(output_dir) / f"source_{name}.wav"
                path.write_bytes(name.encode("utf-8"))
                paths[name] = str(path)
            if progress_callback:
                progress_callback(1.0, "separated")
            return paths

    if mode == ProcessingMode.VOCAL_SPLIT.value:
        monkeypatch.setattr(
            "src.core.vocal_separator.VocalSeparator",
            FakeVocalSeparator,
        )
    else:
        monkeypatch.setattr(
            "src.core.multi_stem_separator.SixStemSeparator",
            FakeSixStemSeparator,
        )

    automatic_midi = mock.Mock(
        side_effect=AssertionError("separation must not invoke the MIDI pipeline")
    )
    monkeypatch.setattr(MusicToMidiPipeline, "process", automatic_midi)
    monkeypatch.setattr(
        "src.gui.workers.separation_worker.clear_gpu_memory",
        lambda: None,
    )

    results = []
    errors = []
    worker = SeparationWorker(
        str(source_path),
        str(output_dir),
        Config(processing_mode=mode, use_gpu=False),
    )
    worker.separation_finished.connect(results.append)
    worker.error_occurred.connect(errors.append)
    worker.run()

    assert errors == []
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, SeparationResult)
    assert result.mode == mode
    assert Path(result.source_path) == source_path.resolve()
    assert Path(result.output_dir) == output_dir.resolve()
    assert tuple(result.separated_audio) == expected_names
    assert separator_calls == [(source_path.resolve(), output_dir.resolve())]
    assert all(Path(path).is_file() for path in result.separated_audio.values())
    assert all(Path(path).suffix.lower() == ".wav" for path in result.separated_audio.values())
    assert list(output_dir.rglob("*.mid")) == []
    automatic_midi.assert_not_called()

    worker.deleteLater()
    qapp.processEvents()


_ROUTE_CONFIG_CASES = [
    *[
        (
            f"{MIDI_ROUTE_YOURMT3_PREFIX}{model.value}",
            ProcessingMode.SMART.value,
            MultiInstrumentModel.YOURMT3.value,
            model.value,
        )
        for model in _YOURMT3_MANUAL_MODELS
    ],
    (
        MIDI_ROUTE_MIROS,
        ProcessingMode.SMART.value,
        MultiInstrumentModel.MIROS.value,
        None,
    ),
    (
        MIDI_ROUTE_PIANO_TRANSKUN,
        ProcessingMode.PIANO_TRANSKUN.value,
        MultiInstrumentModel.MIROS.value,
        None,
    ),
    (
        MIDI_ROUTE_PIANO_TRANSKUN_V2_AUG,
        ProcessingMode.PIANO_TRANSKUN_V2_AUG.value,
        MultiInstrumentModel.MIROS.value,
        None,
    ),
    (
        MIDI_ROUTE_PIANO_ARIA_AMT,
        ProcessingMode.PIANO_ARIA_AMT.value,
        MultiInstrumentModel.MIROS.value,
        None,
    ),
    (
        MIDI_ROUTE_PIANO_BYTEDANCE_PEDAL,
        ProcessingMode.PIANO_BYTEDANCE_PEDAL.value,
        MultiInstrumentModel.MIROS.value,
        None,
    ),
]


@pytest.mark.parametrize(
    ("route", "expected_mode", "expected_backend", "expected_yourmt3_model"),
    _ROUTE_CONFIG_CASES,
)
def test_main_window_manual_route_builds_isolated_config(
    route,
    expected_mode,
    expected_backend,
    expected_yourmt3_model,
):
    base = Config(
        processing_mode=ProcessingMode.VOCAL_SPLIT.value,
        transcription_backend=MultiInstrumentModel.MIROS.value,
        multi_instrument_model=MultiInstrumentModel.MIROS.value,
        save_separated_tracks=False,
        vocal_split_merge_midi=True,
    )
    original = base.to_dict()

    manual = MainWindow._manual_midi_config(base, route)

    assert manual is not base
    assert manual.processing_mode == expected_mode
    assert manual.transcription_backend == expected_backend
    assert manual.multi_instrument_model == expected_backend
    if expected_yourmt3_model is not None:
        assert manual.yourmt3_model == expected_yourmt3_model
    assert manual.save_separated_tracks is True
    assert manual.vocal_split_merge_midi is False
    assert base.to_dict() == original


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class _RecordingWorker:
    created = []

    def __init__(self, audio_path, output_dir, config, parent=None):
        self.audio_path = str(audio_path)
        self.output_dir = str(output_dir)
        self.config = config
        self.parent = parent
        self.progress_updated = _Signal()
        self.processing_finished = _Signal()
        self.error_occurred = _Signal()
        self.finished = _Signal()
        self.running = False
        self.started = False
        self.deleted = False
        self.cancelled = False
        type(self).created.append(self)

    def start(self):
        self.started = True
        self.running = True

    def isRunning(self):
        return self.running

    def cancel(self):
        self.cancelled = True

    def deleteLater(self):
        self.deleted = True


class _RecordingMixer:
    def __init__(self, tracks):
        self.tracks = {name: Path(path).resolve() for name, path in tracks.items()}
        self.track_state_calls = []
        self.control_states = []
        self.running_calls = []
        self.failed_calls = []

    def track_state(self, track_name):
        self.track_state_calls.append(track_name)
        return SimpleNamespace(path=self.tracks[track_name])

    def set_midi_controls_enabled(self, enabled):
        self.control_states.append(bool(enabled))

    def set_track_midi_running(self, track_name, route):
        self.running_calls.append((track_name, route))

    def set_track_midi_failed(self, track_name, error):
        self.failed_calls.append((track_name, str(error)))


def test_main_window_starts_only_clicked_track_in_route_specific_directory(
    qapp,
    tmp_path,
):
    vocals = tmp_path / "source_vocals.wav"
    piano = tmp_path / "source_piano.wav"
    vocals.write_bytes(b"vocals")
    piano.write_bytes(b"piano")
    mixer = _RecordingMixer({"vocals": vocals, "piano": piano})
    _RecordingWorker.created = []

    with (
        mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None),
        mock.patch("src.gui.main_window.ProcessingWorker", _RecordingWorker),
    ):
        window = MainWindow(
            Config(
                transcription_backend=MultiInstrumentModel.YOURMT3.value,
                multi_instrument_model=MultiInstrumentModel.YOURMT3.value,
            )
        )
        window.audio_mixer = mixer
        try:
            window._start_track_midi_conversion(
                "vocals",
                str(vocals),
                MIDI_ROUTE_MIROS,
            )

            assert len(_RecordingWorker.created) == 1
            first = _RecordingWorker.created[0]
            first_output_dir = tmp_path / "midi" / MIDI_ROUTE_MIROS
            assert first.started is True
            assert Path(first.audio_path) == vocals.resolve()
            assert Path(first.output_dir) == first_output_dir.resolve()
            assert Path(first.config.output_dir) == first_output_dir.resolve()
            assert first.config.processing_mode == ProcessingMode.SMART.value
            assert first.config.transcription_backend == MultiInstrumentModel.MIROS.value
            assert mixer.track_state_calls == ["vocals"]
            assert mixer.running_calls == [("vocals", MIDI_ROUTE_MIROS)]

            window._start_track_midi_conversion(
                "piano",
                str(piano),
                MIDI_ROUTE_PIANO_TRANSKUN,
            )
            assert len(_RecordingWorker.created) == 1
            assert mixer.track_state_calls == ["vocals"]
            assert mixer.failed_calls[-1][0] == "piano"

            first.running = False
            window._on_worker_thread_finished(first)
            assert window.worker is None

            window._start_track_midi_conversion(
                "piano",
                str(piano),
                MIDI_ROUTE_PIANO_TRANSKUN,
            )
            assert len(_RecordingWorker.created) == 2
            second = _RecordingWorker.created[1]
            second_output_dir = tmp_path / "midi" / MIDI_ROUTE_PIANO_TRANSKUN
            assert second.started is True
            assert Path(second.audio_path) == piano.resolve()
            assert Path(second.output_dir) == second_output_dir.resolve()
            assert Path(second.config.output_dir) == second_output_dir.resolve()
            assert second.config.processing_mode == ProcessingMode.PIANO_TRANSKUN.value
            assert first_output_dir != second_output_dir
            assert mixer.track_state_calls == ["vocals", "piano"]
            assert mixer.running_calls[-1] == (
                "piano",
                MIDI_ROUTE_PIANO_TRANSKUN,
            )
        finally:
            window.worker = None
            window._manual_midi_context = None
            window.audio_mixer = None
            window.close()
            qapp.processEvents()
