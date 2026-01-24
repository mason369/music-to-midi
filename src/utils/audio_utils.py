"""
Utility functions for audio processing.
"""
import os
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


def load_audio(
    path: str,
    sr: int = 44100,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return as numpy array.

    Args:
        path: Path to audio file
        sr: Target sample rate
        mono: Whether to convert to mono

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    import librosa

    logger.info(f"Loading audio: {path}")
    y, loaded_sr = librosa.load(path, sr=sr, mono=mono)
    logger.info(f"Loaded audio: {len(y)/loaded_sr:.2f}s at {loaded_sr}Hz")

    return y, loaded_sr


def save_audio(
    path: str,
    audio: np.ndarray,
    sr: int = 44100
) -> None:
    """
    Save numpy array as audio file.

    Args:
        path: Output path
        audio: Audio data as numpy array
        sr: Sample rate
    """
    import soundfile as sf

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger.info(f"Saving audio: {path}")
    sf.write(path, audio, sr)


def get_audio_duration(path: str) -> float:
    """
    Get duration of audio file in seconds.

    Args:
        path: Path to audio file

    Returns:
        Duration in seconds
    """
    import librosa

    duration = librosa.get_duration(path=path)
    return duration


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.

    Args:
        audio: Audio data

    Returns:
        Normalized audio
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Audio data
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    import librosa

    if orig_sr == target_sr:
        return audio

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def get_supported_formats() -> list:
    """Get list of supported audio formats."""
    return [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"]


def is_supported_format(path: str) -> bool:
    """Check if file format is supported."""
    ext = Path(path).suffix.lower()
    return ext in get_supported_formats()


def convert_to_wav(input_path: str, output_path: str, sr: int = 44100) -> str:
    """
    Convert audio file to WAV format.

    Args:
        input_path: Input audio file path
        output_path: Output WAV file path
        sr: Target sample rate

    Returns:
        Path to converted file
    """
    audio, _ = load_audio(input_path, sr=sr, mono=False)
    save_audio(output_path, audio, sr)
    return output_path
