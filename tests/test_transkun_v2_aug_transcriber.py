from __future__ import annotations

import queue
from pathlib import Path

import pytest
from mido import Message, MidiFile, MidiTrack

from src.core import transkun_v2_aug_transcriber as transcriber_module
from src.core.transkun_v2_aug_transcriber import TranskunV2AugTranscriber
from src.models.data_models import Config


def _write_valid_midi(path: Path) -> None:
    midi = MidiFile(type=1)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(Message("note_on", note=60, velocity=64, time=0))
    track.append(Message("note_off", note=60, velocity=0, time=240))
    midi.save(str(path))


class _FakeQueue:
    def __init__(self):
        self.results = []

    def put(self, value):
        self.results.append(value)

    def get(self, timeout=None):
        del timeout
        if not self.results:
            raise queue.Empty
        return self.results.pop(0)

    def close(self):
        return None

    def join_thread(self):
        return None


class _SynchronousProcess:
    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.exitcode = None
        self._alive = False

    def start(self):
        self._alive = True
        self.target(*self.args)
        self.exitcode = 0
        self._alive = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        del timeout

    def terminate(self):
        self._alive = False
        self.exitcode = -15


def _prepare_transcriber(tmp_path: Path, monkeypatch) -> TranskunV2AugTranscriber:
    model_dir = tmp_path / "checkpointMSimplerAug"
    model_dir.mkdir()
    (model_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    (model_dir / "model.conf").write_text("{}", encoding="utf-8")
    transcriber = TranskunV2AugTranscriber(
        Config(use_gpu=False),
        model_dir=model_dir,
    )
    monkeypatch.setattr(transcriber, "is_available", lambda: True)
    monkeypatch.setattr(transcriber, "get_unavailable_reason", lambda: "")
    monkeypatch.setattr(transcriber, "get_model_validation_error", lambda: "")
    monkeypatch.setattr(transcriber, "_resolve_runtime_device", lambda: "cpu")
    monkeypatch.setattr(transcriber_module.multiprocessing, "Queue", _FakeQueue)
    monkeypatch.setattr(
        transcriber_module.multiprocessing,
        "Process",
        _SynchronousProcess,
    )
    monkeypatch.setattr(transcriber_module, "clear_gpu_memory", lambda: None)
    return transcriber


def test_transcribe_passes_external_aug_checkpoint_and_config(tmp_path, monkeypatch):
    transcriber = _prepare_transcriber(tmp_path, monkeypatch)
    audio_path = tmp_path / "song.mp3"
    output_path = tmp_path / "out" / "song_piano_transkun_v2_aug.mid"
    audio_path.write_bytes(b"audio")
    calls = []

    def fake_worker(audio, output, checkpoint, config, device, result_queue):
        calls.append((audio, output, checkpoint, config, device))
        _write_valid_midi(Path(output))
        result_queue.put({"ok": output})

    monkeypatch.setattr(transcriber_module, "_transkun_worker", fake_worker)
    progress = []

    result = transcriber.transcribe(
        str(audio_path),
        str(output_path),
        progress_callback=lambda value, message: progress.append((value, message)),
    )

    assert result == str(output_path)
    MidiFile(str(output_path))
    assert len(calls) == 1
    audio, temporary_output, checkpoint, config, device = calls[0]
    assert audio == str(audio_path)
    temporary_output_path = Path(temporary_output)
    assert temporary_output_path.parent == output_path.parent
    assert temporary_output_path.name.startswith(".song_piano_transkun_v2_aug.transkun-v2-aug.")
    assert not temporary_output_path.exists()
    assert checkpoint == str(transcriber.model_dir / "checkpoint.pt")
    assert config == str(transcriber.model_dir / "model.conf")
    assert device == "cpu"
    assert [value for value, _message in progress] == [0.05, 0.30, 1.0]


def test_transcribe_reports_invalid_aug_model_without_starting_process(tmp_path, monkeypatch):
    audio_path = tmp_path / "song.mp3"
    audio_path.write_bytes(b"audio")
    transcriber = TranskunV2AugTranscriber(
        Config(use_gpu=False),
        model_dir=tmp_path / "missing-model",
    )
    monkeypatch.setattr(transcriber, "is_available", lambda: True)
    monkeypatch.setattr(transcriber, "get_unavailable_reason", lambda: "")
    monkeypatch.setattr(
        transcriber,
        "get_model_validation_error",
        lambda: "checkpoint SHA256 mismatch",
    )
    process = monkeypatch.setattr(
        transcriber_module.multiprocessing,
        "Process",
        lambda **_kwargs: pytest.fail("process must not start"),
    )
    del process

    with pytest.raises(RuntimeError, match="checkpoint SHA256 mismatch") as exc_info:
        transcriber.transcribe(str(audio_path), str(tmp_path / "out.mid"))

    assert "download_transkun_v2_aug_model.py" in str(exc_info.value)


def test_transcribe_surfaces_worker_failure(tmp_path, monkeypatch):
    transcriber = _prepare_transcriber(tmp_path, monkeypatch)
    audio_path = tmp_path / "song.mp3"
    audio_path.write_bytes(b"audio")

    def failing_worker(_audio, _output, _checkpoint, _config, _device, result_queue):
        result_queue.put({"error": "checkpoint schema mismatch"})

    monkeypatch.setattr(transcriber_module, "_transkun_worker", failing_worker)

    with pytest.raises(RuntimeError, match="checkpoint schema mismatch"):
        transcriber.transcribe(str(audio_path), str(tmp_path / "out.mid"))


def test_transcribe_rejects_missing_output_even_when_worker_reports_ok(tmp_path, monkeypatch):
    transcriber = _prepare_transcriber(tmp_path, monkeypatch)
    audio_path = tmp_path / "song.mp3"
    audio_path.write_bytes(b"audio")

    output_path = tmp_path / "out.mid"
    _write_valid_midi(output_path)
    stale_bytes = output_path.read_bytes()

    def silent_worker(_audio, output, _checkpoint, _config, _device, result_queue):
        result_queue.put({"ok": output})

    monkeypatch.setattr(transcriber_module, "_transkun_worker", silent_worker)

    with pytest.raises(RuntimeError, match="未生成 MIDI 输出"):
        transcriber.transcribe(str(audio_path), str(output_path))

    assert output_path.read_bytes() == stale_bytes
    assert list(tmp_path.glob(".out.transkun-v2-aug.*.tmp.mid")) == []
