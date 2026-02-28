"""
音频处理工具函数
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
    加载音频文件并返回numpy数组

    参数:
        path: 音频文件路径
        sr: 目标采样率
        mono: 是否转换为单声道

    返回:
        (音频数据, 采样率) 元组
    """
    import librosa

    logger.info(f"正在加载音频: {path}")
    y, loaded_sr = librosa.load(path, sr=sr, mono=mono)

    if len(y) == 0:
        raise ValueError(f"音频文件为空或无法解码: {path}")

    logger.info(f"已加载音频: {len(y)/loaded_sr:.2f}秒 采样率{loaded_sr}Hz")

    return y, loaded_sr


def save_audio(
    path: str,
    audio: np.ndarray,
    sr: int = 44100
) -> None:
    """
    将numpy数组保存为音频文件

    参数:
        path: 输出路径
        audio: 音频数据numpy数组
        sr: 采样率
    """
    import soundfile as sf

    # 确保目录存在
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    logger.info(f"正在保存音频: {path}")
    sf.write(path, audio, sr)


def get_audio_duration(path: str) -> float:
    """
    获取音频文件时长（秒）

    参数:
        path: 音频文件路径

    返回:
        时长（秒）
    """
    import librosa

    duration = librosa.get_duration(path=path)
    return duration


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    将音频归一化到 [-1, 1] 范围

    参数:
        audio: 音频数据

    返回:
        归一化后的音频
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
    将音频重采样到目标采样率

    参数:
        audio: 音频数据
        orig_sr: 原始采样率
        target_sr: 目标采样率

    返回:
        重采样后的音频
    """
    import librosa

    if orig_sr == target_sr:
        return audio

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def get_supported_formats() -> list:
    """获取支持的音频格式列表"""
    return [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"]


def is_supported_format(path: str) -> bool:
    """检查文件格式是否支持"""
    ext = Path(path).suffix.lower()
    return ext in get_supported_formats()


def convert_to_wav(input_path: str, output_path: str, sr: int = 44100) -> str:
    """
    将音频文件转换为WAV格式

    参数:
        input_path: 输入音频文件路径
        output_path: 输出WAV文件路径
        sr: 目标采样率

    返回:
        转换后文件的路径
    """
    audio, _ = load_audio(input_path, sr=sr, mono=False)
    # librosa.load(mono=False) 返回 (channels, samples)
    # soundfile.write 期望 (samples,) 或 (samples, channels)
    if audio.ndim == 2:
        audio = audio.T
    save_audio(output_path, audio, sr)
    return output_path
