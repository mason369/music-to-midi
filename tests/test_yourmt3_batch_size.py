import pytest

from src.utils.gpu_utils import get_optimal_batch_size
import src.utils.gpu_utils as gpu_utils


def _high_cuda_profile():
    return {
        "tier": "high",
        "ram_gb": 32.0,
        "cpu_cores": 16,
        "has_gpu": True,
        "gpu_vram_gb": 16.0,
        "device": "cuda:0",
    }


def test_cuda_best_long_audio_uses_conservative_batch_size(monkeypatch):
    monkeypatch.delenv("MUSIC_TO_MIDI_YOURMT3_BATCH_SIZE", raising=False)
    monkeypatch.setattr(gpu_utils, "get_system_performance_profile", _high_cuda_profile)

    assert get_optimal_batch_size(157, "best", "cuda:0") == 4


def test_explicit_yourmt3_batch_size_override(monkeypatch):
    monkeypatch.setenv("MUSIC_TO_MIDI_YOURMT3_BATCH_SIZE", "2")

    assert get_optimal_batch_size(157, "best", "cuda:0") == 2


def test_invalid_yourmt3_batch_size_override_is_explicit_error(monkeypatch):
    monkeypatch.setenv("MUSIC_TO_MIDI_YOURMT3_BATCH_SIZE", "zero")

    with pytest.raises(ValueError, match="MUSIC_TO_MIDI_YOURMT3_BATCH_SIZE"):
        get_optimal_batch_size(157, "best", "cuda:0")
