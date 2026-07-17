import hashlib
import inspect
import queue
from pathlib import Path
from unittest.mock import patch

from mido import Message, MidiFile, MidiTrack

from src.core import transkun_transcriber as transkun_module
from src.core.transkun_transcriber import (
    TRANSKUN_CONF_SHA256,
    TRANSKUN_CONF_SIZE,
    TRANSKUN_PACKAGE_VERSION,
    TRANSKUN_WEIGHT_SHA256,
    TRANSKUN_WEIGHT_SIZE,
    TranskunTranscriber,
    _transkun_worker,
)


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

    def empty(self):
        return not self.results

    def get(self, timeout=None):
        del timeout
        if not self.results:
            raise queue.Empty
        return self.results.pop(0)


class _SynchronousProcess:
    def __init__(self, target, args):
        self.target = target
        self.args = args
        self._alive = False

    def start(self):
        self._alive = True
        self.target(*self.args)
        self._alive = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        del timeout

    def terminate(self):
        self._alive = False


def test_transkun_pinned_identity_constants_match_official_201_artifacts():
    assert TRANSKUN_PACKAGE_VERSION == "2.0.1"
    assert TRANSKUN_WEIGHT_SIZE == 56_408_978
    assert TRANSKUN_WEIGHT_SHA256 == (
        "50a80010effc2a59ffcd068a95cd2b29bd7f23a27a3515bc3ccd209c89a3d44c"
    )
    assert TRANSKUN_CONF_SIZE == 782
    assert TRANSKUN_CONF_SHA256 == (
        "d3d989214eb148230ee5df476d994dcde6af595904d3f968f1221d2e3bea5ac6"
    )


def test_transkun_rejects_a_different_installed_package_version():
    with (
        patch(
            "src.core.transkun_transcriber.importlib.util.find_spec",
            return_value=object(),
        ),
        patch(
            "src.core.transkun_transcriber.metadata.version",
            return_value="2.0.2",
        ),
    ):
        reason = TranskunTranscriber.get_unavailable_reason()

    assert "expected 2.0.1, got 2.0.2" in reason


def test_transkun_packaged_resources_require_exact_size_and_sha256(tmp_path):
    weight_payload = b"transkun-weight"
    conf_payload = b"transkun-conf"
    weight_path = tmp_path / "2.0.pt"
    conf_path = tmp_path / "2.0.conf"
    weight_path.write_bytes(weight_payload)
    conf_path.write_bytes(conf_payload)

    def get_resource(name: str) -> Path:
        return weight_path if name == "2.0.pt" else conf_path

    with (
        patch.object(
            TranskunTranscriber,
            "_get_packaged_resource",
            side_effect=get_resource,
        ),
        patch(
            "src.core.transkun_transcriber.TRANSKUN_WEIGHT_SIZE",
            len(weight_payload),
        ),
        patch(
            "src.core.transkun_transcriber.TRANSKUN_WEIGHT_SHA256",
            hashlib.sha256(weight_payload).hexdigest(),
        ),
        patch(
            "src.core.transkun_transcriber.TRANSKUN_CONF_SIZE",
            len(conf_payload),
        ),
        patch(
            "src.core.transkun_transcriber.TRANSKUN_CONF_SHA256",
            hashlib.sha256(conf_payload).hexdigest(),
        ),
    ):
        transcriber = TranskunTranscriber()
        assert transcriber.is_model_available()

        weight_path.write_bytes(weight_payload[:-1] + b"X")
        assert not transcriber.is_model_available()


def test_transkun_worker_loads_the_pinned_checkpoint_strictly():
    source = inspect.getsource(_transkun_worker)
    assert "model.load_state_dict(state_dict, strict=True)" in source
    assert "strict=False" not in source


def test_transkun_parent_uses_blocking_queue_read_with_timeout():
    source = inspect.getsource(TranskunTranscriber.transcribe)
    assert "result_queue.empty()" not in source
    assert "result_queue.get(timeout=2.0)" in source


def test_transkun_does_not_accept_a_stale_final_when_worker_writes_nothing(tmp_path):
    audio_path = tmp_path / "piano.wav"
    output_path = tmp_path / "out.mid"
    weight_path = tmp_path / "2.0.pt"
    conf_path = tmp_path / "2.0.conf"
    audio_path.write_bytes(b"audio")
    weight_path.write_bytes(b"weight")
    conf_path.write_text("{}", encoding="utf-8")
    _write_valid_midi(output_path)
    stale_bytes = output_path.read_bytes()

    def get_resource(name: str) -> Path:
        return weight_path if name == "2.0.pt" else conf_path

    def silent_worker(_audio, output, _weight, _conf, _device, result_queue):
        result_queue.put({"ok": output})

    transcriber = TranskunTranscriber()
    with (
        patch.object(
            transcriber,
            "get_unavailable_reason",
            return_value="",
        ),
        patch.object(
            transcriber,
            "is_model_available",
            return_value=True,
        ),
        patch.object(
            transcriber,
            "_get_packaged_resource",
            side_effect=get_resource,
        ),
        patch.object(
            transcriber,
            "_resolve_runtime_device",
            return_value="cpu",
        ),
        patch.object(
            transkun_module.multiprocessing,
            "Queue",
            _FakeQueue,
        ),
        patch.object(
            transkun_module.multiprocessing,
            "Process",
            _SynchronousProcess,
        ),
        patch.object(
            transkun_module,
            "_transkun_worker",
            silent_worker,
        ),
        patch.object(
            transkun_module,
            "clear_gpu_memory",
            return_value=None,
        ),
    ):
        try:
            transcriber.transcribe(str(audio_path), str(output_path))
        except RuntimeError as exc:
            assert "did not create a MIDI output" in str(exc)
        else:
            raise AssertionError("Expected missing current-run MIDI output to fail")

    assert output_path.read_bytes() == stale_bytes
    assert list(tmp_path.glob(".out.transkun.*.tmp.mid")) == []
