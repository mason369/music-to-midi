"""Official audio semantics without requiring shared FFmpeg DLLs at runtime."""

import json
import subprocess
import sys
import types
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from src.core.aria_amt_transcriber import AriaAmtTranscriber


def test_long_stereo_input_matches_recorded_official_streamreader_samples(tmp_path):
    reference = json.loads(
        (Path(__file__).parent / "fixtures/aria_official_audio.json").read_text()
    )
    i = np.arange(reference["frames"], dtype=np.int64)
    pcm = np.stack(((i * 61 % 65536) - 32768, (i * 127 % 65536) - 32768), axis=1)
    audio = tmp_path / "钢琴 输入.wav"
    sf.write(audio, pcm.astype(np.int16), reference["sample_rate"], subtype="PCM_16")

    with patch("src.core.aria_amt_transcriber.load_audio_tensor") as whole_file_load:
        actual = list(AriaAmtTranscriber._iter_windows_wav_segments(audio, 16000, 30, 3))

    whole_file_load.assert_not_called()
    assert len(actual) == len(reference["segments"]) == 3
    for segment, expected in zip(actual, reference["segments"]):
        assert segment.dtype == torch.float32
        assert segment.shape == (480000,)
        np.testing.assert_allclose(
            segment[reference["indices"]].numpy(), expected, rtol=1e-6, atol=1e-7
        )
    assert torch.count_nonzero(actual[-1][176000:]) == 0


@pytest.mark.parametrize("duration", [0.5, 1.0])
@pytest.mark.parametrize("channels", [1, 2])
def test_short_inputs_keep_official_mean_then_sinc_resample(tmp_path, duration, channels):
    rate = 8000
    pcm = np.arange(int(rate * duration) * channels, dtype=np.int16).reshape(-1, channels)
    audio = tmp_path / "short.wav"
    sf.write(audio, pcm, rate, subtype="PCM_16")
    waveform = torch.from_numpy(pcm.astype(np.float32).T / 32768).mean(0)
    expected = torchaudio.functional.resample(waveform, rate, 16000)
    expected = torch.nn.functional.pad(expected, (0, 16000 - len(expected)))
    with patch("src.core.aria_amt_transcriber.subprocess.run") as ffmpeg:
        actual = list(AriaAmtTranscriber._iter_windows_wav_segments(audio, 16000, 1, 2))
    ffmpeg.assert_not_called()
    assert len(actual) == 1
    assert torch.equal(actual[0], expected)


@pytest.mark.parametrize("channels", [1, 2])
def test_long_same_rate_inputs_keep_channel_mean_and_tail(tmp_path, channels):
    rate = 16000
    pcm = np.arange(rate * 2 * channels, dtype=np.int16).reshape(-1, channels)
    audio = tmp_path / "same-rate.wav"
    sf.write(audio, pcm, rate, subtype="PCM_16")
    waveform = torch.from_numpy(pcm.astype(np.float32) / 32768).mean(1)
    actual = list(AriaAmtTranscriber._iter_windows_wav_segments(audio, rate, 1, 2))
    assert len(actual) == 4
    for index, segment in enumerate(actual):
        expected = waveform[index * 8000 : index * 8000 + 16000]
        expected = torch.nn.functional.pad(expected, (0, 16000 - len(expected)))
        assert torch.equal(segment, expected)


@pytest.mark.parametrize(
    "returncode,payload,message",
    [(1, b"", "decoder failed"), (0, b"", "incomplete float32"), (0, b"abc", "incomplete float32")],
)
def test_long_decode_failures_are_not_retried_with_different_preprocessing(
    tmp_path, returncode, payload, message
):
    audio = tmp_path / "failure.wav"
    sf.write(audio, np.zeros((16001, 2), dtype=np.int16), 16000, subtype="PCM_16")
    with (
        patch(
            "src.core.aria_amt_transcriber.subprocess.run",
            return_value=subprocess.CompletedProcess([], returncode, payload, b"decoder failed"),
        ) as ffmpeg,
        patch("src.core.aria_amt_transcriber.load_audio_tensor") as alternative,
    ):
        with pytest.raises(RuntimeError, match=message):
            list(AriaAmtTranscriber._iter_windows_wav_segments(audio, 16000, 1, 2))
    ffmpeg.assert_called_once()
    alternative.assert_not_called()


def test_audio_preparation_overlaps_model_loading_and_preserves_failure(tmp_path):
    audio_started = Event()
    model_started = Event()

    def read_windows(**kwargs):
        audio_started.set()
        assert model_started.wait(2), "model loading did not overlap audio preparation"
        yield torch.zeros(1)

    def load_model():
        model_started.set()
        assert audio_started.wait(2), "audio preparation did not overlap model loading"
        raise RuntimeError("MODEL_FAILURE_PROOF")

    audio_module = types.ModuleType("amt.audio")
    audio_module.AudioTransform = object
    config_module = types.ModuleType("amt.config")
    config_module.load_config = lambda: {"audio": {"sample_rate": 16000, "chunk_len": 30}}
    inference_module = types.ModuleType("amt.inference")
    inference_module.transcribe = types.SimpleNamespace(STRIDE_FACTOR=3)
    transcriber = AriaAmtTranscriber()
    with (
        patch.dict(
            sys.modules,
            {
                "amt.audio": audio_module,
                "amt.config": config_module,
                "amt.inference": inference_module,
            },
        ),
        patch("src.core.aria_amt_transcriber.prepare_aria_audio_assets"),
        patch.object(transcriber, "_resolve_runtime_device", return_value="cuda:0"),
        patch.object(transcriber, "_iter_windows_wav_segments", side_effect=read_windows),
        patch.object(transcriber, "_load_aria_model", side_effect=load_model),
    ):
        with pytest.raises(RuntimeError, match="MODEL_FAILURE_PROOF"):
            transcriber._run_transcription_windows_single_file(tmp_path / "input.wav", tmp_path)
    assert audio_started.is_set() and model_started.is_set()
