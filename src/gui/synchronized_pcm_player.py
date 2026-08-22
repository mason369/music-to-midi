"""Sample-synchronous PCM mixing through one Qt audio output stream."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import soundfile as sf
from PyQt6.QtCore import QIODevice, QObject, pyqtSignal
from PyQt6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices

_SAMPLE_RATE = 44_100
_CHANNEL_COUNT = 2
_BYTES_PER_OUTPUT_FRAME = _CHANNEL_COUNT * 2
_OUTPUT_BUFFER_FRAMES = 2_048


def _load_aligned_mono_pcm(
    path: str | Path,
    *,
    expected_frames: int | None = None,
) -> np.ndarray:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Synchronized playback PCM is missing: {source}")
    samples, sample_rate = sf.read(
        str(source),
        dtype="float32",
        always_2d=True,
    )
    if sample_rate != _SAMPLE_RATE:
        raise RuntimeError(
            "Synchronized playback PCM has an unsupported sample rate: "
            f"path={source}, expected={_SAMPLE_RATE}, actual={sample_rate}"
        )
    if samples.shape[1] != 1:
        raise RuntimeError(
            "Synchronized playback requires mono source buses before final mixing: "
            f"path={source}, channels={samples.shape[1]}"
        )
    if expected_frames is not None and samples.shape[0] != expected_frames:
        raise RuntimeError(
            "Synchronized playback buses do not share one frame axis: "
            f"path={source}, expected={expected_frames}, actual={samples.shape[0]}"
        )
    mono = np.ascontiguousarray(samples[:, 0], dtype="float32")
    if mono.size <= 0:
        raise RuntimeError(f"Synchronized playback PCM is empty: {source}")
    if not np.isfinite(mono).all():
        raise RuntimeError(f"Synchronized playback PCM contains non-finite samples: {source}")
    mono.setflags(write=False)
    return mono


class _AlignedPcmMixDevice(QIODevice):
    """Pull-mode device that mixes every bus at one shared fractional frame."""

    def __init__(
        self,
        original: np.ndarray,
        midi_mix: np.ndarray,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if original.ndim != 1 or midi_mix.ndim != 1:
            raise ValueError("Synchronized playback buses must be mono arrays")
        if original.shape != midi_mix.shape or original.size <= 0:
            raise ValueError(
                "Synchronized original and MIDI arrays must have one positive shared length"
            )
        self._original = original
        self._midi_mix = midi_mix
        self._instrument_buses: dict[str, np.ndarray] = {}
        self._position_frames = 0.0
        self._playback_rate = 1.0
        self._mix = 0.5
        self._stereo = False
        self._muted: frozenset[str] = frozenset()
        self._lock = threading.RLock()
        self.open(QIODevice.OpenModeFlag.ReadOnly)

    @property
    def frame_count(self) -> int:
        return int(self._original.size)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / _SAMPLE_RATE

    @property
    def position_frames(self) -> float:
        with self._lock:
            return float(self._position_frames)

    @property
    def playback_rate(self) -> float:
        with self._lock:
            return float(self._playback_rate)

    def set_position_frames(self, position_frames: float) -> None:
        position = float(position_frames)
        if not math.isfinite(position):
            raise ValueError(f"Invalid synchronized playback position: {position_frames!r}")
        with self._lock:
            self._position_frames = min(float(self.frame_count), max(0.0, position))

    def set_playback_rate(self, playback_rate: float) -> None:
        rate = float(playback_rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"Invalid synchronized playback rate: {playback_rate!r}")
        with self._lock:
            self._playback_rate = rate

    def set_instrument_buses(self, buses: Mapping[str, np.ndarray]) -> None:
        normalized: dict[str, np.ndarray] = {}
        for instrument, samples in buses.items():
            name = str(instrument).strip()
            if not name:
                raise ValueError("Synchronized playback instrument id cannot be empty")
            if samples.ndim != 1 or samples.size != self.frame_count:
                raise ValueError(
                    "Synchronized instrument bus has a different frame axis: "
                    f"instrument={name!r}, expected={self.frame_count}, actual={samples.size}"
                )
            normalized[name] = samples
        with self._lock:
            self._instrument_buses = normalized

    def set_mix_state(
        self,
        *,
        mix: float,
        stereo: bool,
        muted: frozenset[str],
    ) -> None:
        mix_value = float(mix)
        if not math.isfinite(mix_value) or not 0.0 <= mix_value <= 1.0:
            raise ValueError(f"Invalid synchronized playback mix: {mix!r}")
        muted_set = frozenset(str(item) for item in muted)
        with self._lock:
            if muted_set and not self._instrument_buses:
                raise RuntimeError(
                    "Per-instrument synchronized playback was requested before its buses loaded"
                )
            missing = muted_set - self._instrument_buses.keys()
            if missing:
                raise RuntimeError(
                    "Synchronized playback mute state references unknown instruments: "
                    f"{sorted(missing)!r}"
                )
            self._mix = mix_value
            self._stereo = bool(stereo)
            self._muted = muted_set

    def isSequential(self) -> bool:  # noqa: N802 - Qt virtual method name
        return True

    def bytesAvailable(self) -> int:  # noqa: N802 - Qt virtual method name
        with self._lock:
            remaining_source_frames = max(0.0, self.frame_count - self._position_frames)
            output_frames = math.ceil(remaining_source_frames / self._playback_rate)
        return output_frames * _BYTES_PER_OUTPUT_FRAME + super().bytesAvailable()

    def atEnd(self) -> bool:  # noqa: N802 - Qt virtual method name
        with self._lock:
            return self._position_frames >= self.frame_count

    @staticmethod
    def _sample_bus(
        bus: np.ndarray,
        base_indices: np.ndarray,
        next_indices: np.ndarray,
        fractions: np.ndarray,
    ) -> np.ndarray:
        return bus[base_indices] + (bus[next_indices] - bus[base_indices]) * fractions

    def _render_frames(self, requested_frames: int) -> bytes:
        if requested_frames <= 0:
            return b""
        with self._lock:
            remaining_source_frames = self.frame_count - self._position_frames
            if remaining_source_frames <= 0.0:
                return b""
            output_frames = min(
                int(requested_frames),
                int(math.ceil(remaining_source_frames / self._playback_rate)),
            )
            positions = self._position_frames + self._playback_rate * np.arange(
                output_frames,
                dtype="float64",
            )
            positions = np.minimum(positions, self.frame_count - 1)
            base_indices = np.floor(positions).astype(np.int64)
            next_indices = np.minimum(base_indices + 1, self.frame_count - 1)
            fractions = (positions - base_indices).astype("float32")

            original = self._sample_bus(
                self._original,
                base_indices,
                next_indices,
                fractions,
            )
            if self._muted:
                midi = np.zeros(output_frames, dtype="float32")
                for instrument, bus in self._instrument_buses.items():
                    if instrument not in self._muted:
                        midi += self._sample_bus(
                            bus,
                            base_indices,
                            next_indices,
                            fractions,
                        )
            else:
                midi = self._sample_bus(
                    self._midi_mix,
                    base_indices,
                    next_indices,
                    fractions,
                )

            if self._stereo:
                stereo = np.column_stack((original, midi))
            else:
                mono = original * (1.0 - self._mix) + midi * self._mix
                stereo = np.column_stack((mono, mono))
            pcm = np.rint(np.clip(stereo, -1.0, 1.0) * 32_767.0).astype("<i2")
            self._position_frames = min(
                float(self.frame_count),
                self._position_frames + output_frames * self._playback_rate,
            )
            return pcm.tobytes(order="C")

    def render_frames_for_test(self, requested_frames: int) -> bytes:
        """Render deterministic PCM without requiring a host audio device."""

        return self._render_frames(requested_frames)

    def readData(self, maxlen: int) -> bytes:  # noqa: N802 - Qt virtual method name
        return self._render_frames(int(maxlen) // _BYTES_PER_OUTPUT_FRAME)

    def writeData(self, _data: bytes) -> int:  # noqa: N802 - Qt virtual method name
        return -1


class SynchronizedPcmPlayer(QObject):
    """One output-device clock for original, MIDI mix, and instrument buses."""

    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._format = QAudioFormat()
        self._format.setSampleRate(_SAMPLE_RATE)
        self._format.setChannelCount(_CHANNEL_COUNT)
        self._format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        self._sink: QAudioSink | None = None
        self._source: _AlignedPcmMixDevice | None = None
        self._instrument_paths: dict[str, Path] = {}
        self._instrument_buses_loaded = False
        self._playing = False
        self._suspended = False
        self._started = False
        self._resetting = False
        self._finished_emitted = False
        self._segment_start_frame = 0.0
        self._processed_origin_us = 0
        self._frozen_position_frame = 0.0

    @property
    def is_configured(self) -> bool:
        return self._sink is not None and self._source is not None

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def duration_seconds(self) -> float:
        return self._source.duration_seconds if self._source is not None else 0.0

    @property
    def output_stream_count(self) -> int:
        return 1 if self.is_configured else 0

    def _create_sink(self) -> QAudioSink:
        output_device = QMediaDevices.defaultAudioOutput()
        if output_device.isNull():
            raise RuntimeError("No Windows audio output device is available for MIDI playback")
        if not output_device.isFormatSupported(self._format):
            raise RuntimeError(
                "The selected audio device does not support the required synchronized format: "
                f"device={output_device.description()!r}, format=44100 Hz stereo Int16"
            )
        sink = QAudioSink(output_device, self._format, self)
        sink.setBufferSize(self._format.bytesForFrames(_OUTPUT_BUFFER_FRAMES))
        sink.stateChanged.connect(self._on_sink_state_changed)
        return sink

    def configure(
        self,
        *,
        original_wav: str | Path,
        midi_wav: str | Path,
        instrument_wavs: Mapping[str, str | Path],
        playback_rate: float,
    ) -> None:
        original = _load_aligned_mono_pcm(original_wav)
        midi = _load_aligned_mono_pcm(midi_wav, expected_frames=original.size)
        source = _AlignedPcmMixDevice(original, midi, self)
        source.set_playback_rate(playback_rate)
        paths = {
            str(instrument): Path(path).resolve() for instrument, path in instrument_wavs.items()
        }
        if any(not instrument for instrument in paths):
            raise ValueError("Synchronized playback instrument id cannot be empty")
        sink = self._create_sink()

        self.shutdown()
        self._source = source
        self._sink = sink
        self._instrument_paths = paths
        self._instrument_buses_loaded = False
        self._playing = False
        self._suspended = False
        self._started = False
        self._finished_emitted = False
        self._segment_start_frame = 0.0
        self._processed_origin_us = 0
        self._frozen_position_frame = 0.0

    def ensure_instrument_buses(self) -> None:
        if self._source is None:
            raise RuntimeError("Synchronized playback is not configured")
        if self._instrument_buses_loaded:
            return
        expected_frames = self._source.frame_count
        loaded = {
            instrument: _load_aligned_mono_pcm(path, expected_frames=expected_frames)
            for instrument, path in self._instrument_paths.items()
        }
        self._source.set_instrument_buses(loaded)
        self._instrument_buses_loaded = True

    def set_mix_state(
        self,
        *,
        mix: float,
        stereo: bool,
        muted: frozenset[str],
    ) -> None:
        if self._source is None:
            return
        if muted:
            self.ensure_instrument_buses()
        self._source.set_mix_state(mix=mix, stereo=stereo, muted=muted)
        if self._suspended:
            self._discard_buffer_at_current_position()

    def _processed_position_frame(self) -> float:
        if self._source is None:
            return 0.0
        if self._sink is None or not self._started:
            return self._source.position_frames
        processed_us = max(0, int(self._sink.processedUSecs()) - self._processed_origin_us)
        position = self._segment_start_frame + (
            processed_us * _SAMPLE_RATE * self._source.playback_rate / 1_000_000.0
        )
        return min(float(self._source.frame_count), max(0.0, position))

    @property
    def position_seconds(self) -> float:
        if self._source is None:
            return 0.0
        if self._suspended:
            return self._frozen_position_frame / _SAMPLE_RATE
        return self._processed_position_frame() / _SAMPLE_RATE

    def _start_from_frame(self, position_frame: float) -> None:
        if self._source is None or self._sink is None:
            raise RuntimeError("Synchronized playback is not configured")
        self._source.set_position_frames(position_frame)
        self._segment_start_frame = self._source.position_frames
        self._processed_origin_us = int(self._sink.processedUSecs())
        self._finished_emitted = False
        self._started = True
        self._suspended = False
        self._playing = True
        self._sink.start(self._source)
        if self._sink.state() == QAudio.State.StoppedState:
            error = self._sink.error()
            self._playing = False
            self._started = False
            raise RuntimeError(
                "Unable to start synchronized audio output: "
                f"error={error.name}, device={QMediaDevices.defaultAudioOutput().description()!r}"
            )

    def play(self) -> None:
        if self._source is None or self._sink is None:
            raise RuntimeError("Synchronized playback is not configured")
        if self._playing:
            return
        if self.position_seconds >= self.duration_seconds - (0.5 / _SAMPLE_RATE):
            self.seek(0.0)
        if self._suspended and self._sink.state() == QAudio.State.SuspendedState:
            self._sink.resume()
            self._playing = True
            self._suspended = False
            return
        self._start_from_frame(self._frozen_position_frame)

    def pause(self) -> None:
        if not self._playing or self._sink is None:
            return
        self._sink.suspend()
        self._frozen_position_frame = self._processed_position_frame()
        self._playing = False
        self._suspended = self._sink.state() == QAudio.State.SuspendedState
        if not self._suspended:
            error = self._sink.error()
            raise RuntimeError(f"Unable to suspend synchronized audio output: error={error.name}")

    def _reset_sink(self) -> None:
        if self._sink is None:
            return
        self._resetting = True
        try:
            self._sink.reset()
        finally:
            self._resetting = False
        self._started = False
        self._suspended = False

    def _discard_buffer_at_current_position(self) -> None:
        position = (
            self._frozen_position_frame if self._suspended else self._processed_position_frame()
        )
        self._reset_sink()
        if self._source is not None:
            self._source.set_position_frames(position)
        self._frozen_position_frame = position

    def seek(self, seconds: float) -> None:
        if self._source is None:
            return
        position = float(seconds)
        if not math.isfinite(position):
            raise ValueError(f"Invalid synchronized playback seek: {seconds!r}")
        target_frame = min(
            float(self._source.frame_count),
            max(0.0, position * _SAMPLE_RATE),
        )
        resume = self._playing
        self._reset_sink()
        self._source.set_position_frames(target_frame)
        self._frozen_position_frame = target_frame
        self._playing = False
        self._finished_emitted = False
        if resume and target_frame < self._source.frame_count:
            self._start_from_frame(target_frame)

    def set_playback_rate(self, playback_rate: float) -> None:
        if self._source is None:
            return
        rate = float(playback_rate)
        if math.isclose(rate, self._source.playback_rate, rel_tol=0.0, abs_tol=1e-12):
            return
        position = self.position_seconds
        resume = self._playing
        self._reset_sink()
        self._source.set_playback_rate(rate)
        target_frame = position * _SAMPLE_RATE
        self._source.set_position_frames(target_frame)
        self._frozen_position_frame = target_frame
        self._playing = False
        if resume:
            self._start_from_frame(target_frame)

    def _on_sink_state_changed(self, state: QAudio.State) -> None:
        if self._resetting or self._sink is None or self._source is None:
            return
        if state == QAudio.State.StoppedState:
            error = self._sink.error()
            if error != QAudio.Error.NoError:
                self._playing = False
                self._suspended = False
                self.error_occurred.emit(
                    "Synchronized audio output stopped with an error: " f"{error.name}"
                )
            return
        if (
            state == QAudio.State.IdleState
            and self._source.atEnd()
            and self._processed_position_frame() >= self._source.frame_count - 1
            and not self._finished_emitted
        ):
            self._playing = False
            self._suspended = False
            self._frozen_position_frame = float(self._source.frame_count)
            self._finished_emitted = True
            self.finished.emit()

    def shutdown(self) -> None:
        self._playing = False
        self._suspended = False
        self._started = False
        if self._sink is not None:
            self._reset_sink()
            self._sink.deleteLater()
        if self._source is not None:
            self._source.close()
            self._source.deleteLater()
        self._sink = None
        self._source = None
        self._instrument_paths = {}
        self._instrument_buses_loaded = False
        self._finished_emitted = False
        self._segment_start_frame = 0.0
        self._processed_origin_us = 0
        self._frozen_position_frame = 0.0


__all__ = ["SynchronizedPcmPlayer"]
