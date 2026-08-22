from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.utils.audio_utils import load_audio_tensor


def test_load_audio_tensor_returns_channels_first_float32(tmp_path: Path):
    path = tmp_path / "stereo.wav"
    samples = np.array(
        [[0.25, -0.25], [0.5, -0.5], [0.75, -0.75]],
        dtype=np.float32,
    )
    sf.write(path, samples, 16_000, subtype="FLOAT")

    waveform, sample_rate = load_audio_tensor(path)

    assert sample_rate == 16_000
    assert waveform.dtype == torch.float32
    assert waveform.shape == (2, 3)
    assert waveform.is_contiguous()
    assert torch.equal(waveform, torch.from_numpy(samples.T.copy()))


def test_load_audio_tensor_rejects_empty_audio(tmp_path: Path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.empty((0, 1), dtype=np.float32), 44_100)

    with pytest.raises(ValueError, match="音频文件为空"):
        load_audio_tensor(path)
