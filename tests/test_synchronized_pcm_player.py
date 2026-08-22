from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtMultimedia import QAudio, QAudioFormat, QMediaDevices
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from src.gui.synchronized_pcm_player import (
    SynchronizedPcmPlayer,
    _AlignedPcmMixDevice,
    _load_aligned_mono_pcm,
)


def _decode_stereo_pcm(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<i2").reshape(-1, 2).astype("float32") / 32_767.0


def _write_mono_wav(path: Path, samples: np.ndarray) -> Path:
    sf.write(path, np.asarray(samples, dtype="float32"), 44_100, subtype="FLOAT")
    return path


def test_mix_device_routes_original_and_midi_from_the_same_frame_axis() -> None:
    original = np.array([0.0, 0.8, 0.2, 0.0], dtype="float32")
    midi = np.array([0.0, 0.4, 0.6, 0.0], dtype="float32")
    device = _AlignedPcmMixDevice(original, midi)

    device.set_position_frames(1)
    device.set_mix_state(mix=0.5, stereo=False, muted=frozenset())
    mixed = _decode_stereo_pcm(device.render_frames_for_test(2))

    np.testing.assert_allclose(mixed[:, 0], [0.6, 0.4], atol=1.0 / 32_767.0)
    np.testing.assert_array_equal(mixed[:, 0], mixed[:, 1])

    device.set_position_frames(1)
    device.set_mix_state(mix=0.5, stereo=True, muted=frozenset())
    stereo = _decode_stereo_pcm(device.render_frames_for_test(2))

    np.testing.assert_allclose(stereo[:, 0], [0.8, 0.2], atol=1.0 / 32_767.0)
    np.testing.assert_allclose(stereo[:, 1], [0.4, 0.6], atol=1.0 / 32_767.0)


def test_mix_device_seek_and_rate_apply_once_to_every_bus() -> None:
    original = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype="float32")
    midi = np.array([0.0, -0.2, -0.4, -0.6, -0.8], dtype="float32")
    device = _AlignedPcmMixDevice(original, midi)
    device.set_mix_state(mix=0.5, stereo=True, muted=frozenset())
    device.set_position_frames(1.5)
    device.set_playback_rate(0.5)

    stereo = _decode_stereo_pcm(device.render_frames_for_test(3))

    np.testing.assert_allclose(stereo[:, 0], [0.3, 0.4, 0.5], atol=1.0 / 32_767.0)
    np.testing.assert_allclose(stereo[:, 1], [-0.3, -0.4, -0.5], atol=1.0 / 32_767.0)
    assert device.position_frames == pytest.approx(3.0)


def test_per_instrument_mute_is_mixed_before_the_single_output_stream() -> None:
    zeros = np.zeros(4, dtype="float32")
    device = _AlignedPcmMixDevice(zeros, np.ones(4, dtype="float32"))
    device.set_instrument_buses(
        {
            "piano": np.full(4, 0.2, dtype="float32"),
            "guitar": np.full(4, 0.3, dtype="float32"),
        }
    )
    device.set_mix_state(mix=1.0, stereo=False, muted=frozenset({"piano"}))

    mixed = _decode_stereo_pcm(device.render_frames_for_test(2))

    np.testing.assert_allclose(mixed, 0.3, atol=1.0 / 32_767.0)


def test_aligned_pcm_loader_rejects_a_different_frame_axis(tmp_path: Path) -> None:
    reference = _write_mono_wav(tmp_path / "reference.wav", np.zeros(100, dtype="float32"))
    mismatched = _write_mono_wav(tmp_path / "mismatched.wav", np.zeros(99, dtype="float32"))

    assert _load_aligned_mono_pcm(reference).shape == (100,)
    with pytest.raises(RuntimeError, match="do not share one frame axis"):
        _load_aligned_mono_pcm(mismatched, expected_frames=100)


def test_exhausted_idle_source_finishes_even_when_backend_clock_lags() -> None:
    class ExhaustedSource:
        frame_count = 44_100
        playback_rate = 1.0

        @staticmethod
        def atEnd() -> bool:  # noqa: N802 - mirrors QIODevice
            return True

    class LaggingSink:
        @staticmethod
        def processedUSecs() -> int:  # noqa: N802 - mirrors QAudioSink
            return 0

    player = SynchronizedPcmPlayer()
    player._source = ExhaustedSource()
    player._sink = LaggingSink()
    player._playing = True
    finished: list[bool] = []
    player.finished.connect(lambda: finished.append(True))

    try:
        player._on_sink_state_changed(QAudio.State.IdleState)

        assert finished == [True]
        assert player.is_playing is False
        assert player._finished_emitted is True
        assert player._frozen_position_frame == 44_100.0
    finally:
        player._sink = None
        player._source = None


@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("MUSIC_TO_MIDI_RUN_AUDIO_DEVICE_TESTS") != "1",
    reason="requires an explicitly enabled Windows audio-device acceptance run",
)
def test_windows_audio_sink_survives_repeated_pause_seek_and_resume(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    audio_format = QAudioFormat()
    audio_format.setSampleRate(44_100)
    audio_format.setChannelCount(2)
    audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    device = QMediaDevices.defaultAudioOutput()
    if device.isNull() or not device.isFormatSupported(audio_format):
        pytest.skip("the current Windows output device cannot run 44.1 kHz stereo Int16")

    time_axis = np.arange(44_100 * 2, dtype="float32") / 44_100.0
    original = _write_mono_wav(
        tmp_path / "original.wav",
        0.01 * np.sin(2.0 * np.pi * 220.0 * time_axis),
    )
    midi = _write_mono_wav(
        tmp_path / "midi.wav",
        0.01 * np.sin(2.0 * np.pi * 440.0 * time_axis),
    )
    player = SynchronizedPcmPlayer()
    errors: list[str] = []
    player.error_occurred.connect(errors.append)
    player.configure(
        original_wav=original,
        midi_wav=midi,
        instrument_wavs={},
        playback_rate=1.0,
    )

    try:
        assert player.output_stream_count == 1
        player.play()
        QTest.qWait(120)
        assert player.is_playing
        assert player.position_seconds > 0.0

        for target in (0.75, 0.0, 1.25, 0.2, 1.6, 0.0):
            player.pause()
            assert not player.is_playing
            player.seek(target)
            assert player.position_seconds == pytest.approx(target, abs=1.0 / 44_100.0)
            player.play()
            QTest.qWait(45)
            assert player.is_playing
            assert player.output_stream_count == 1

        player.set_playback_rate(1.25)
        QTest.qWait(80)
        assert player.is_playing
        assert errors == []
    finally:
        player.shutdown()
        app.processEvents()
