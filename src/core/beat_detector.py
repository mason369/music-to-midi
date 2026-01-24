"""
节拍检测模块 - 使用 librosa
"""
import logging
from typing import Optional, Callable
import numpy as np

from src.models.data_models import Config, BeatInfo

logger = logging.getLogger(__name__)


class BeatDetector:
    """
    使用 librosa 进行节拍和速度检测

    功能特点:
    - BPM 估算
    - 节拍位置检测
    - 下拍检测（可选）
    """

    def __init__(self, config: Config):
        """
        初始化节拍检测器

        参数:
            config: 应用配置
        """
        self.config = config

    def detect(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> BeatInfo:
        """
        从音频中检测节拍和速度

        参数:
            audio_path: 音频文件路径
            progress_callback: 可选的进度回调

        返回:
            包含速度和节拍时间的 BeatInfo
        """
        import librosa

        if progress_callback:
            progress_callback(0.0, "正在加载音频进行节拍检测...")

        logger.info(f"正在检测节拍: {audio_path}")

        # 加载音频
        y, sr = librosa.load(audio_path, sr=22050)

        if progress_callback:
            progress_callback(0.3, "正在分析速度...")

        # 估算速度
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # 处理标量和数组类型的速度
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        if progress_callback:
            progress_callback(0.6, "正在查找节拍位置...")

        # 将帧转换为时间
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        if progress_callback:
            progress_callback(0.8, "正在检测下拍...")

        # 尝试检测下拍
        downbeats = self._detect_downbeats(y, sr, beat_times)

        if progress_callback:
            progress_callback(1.0, f"BPM: {tempo:.1f}")

        beat_info = BeatInfo(
            bpm=tempo,
            beat_times=beat_times.tolist(),
            downbeats=downbeats,
            time_signature=(4, 4)
        )

        logger.info(f"检测到 BPM: {tempo:.1f}, {len(beat_times)} 个节拍")

        return beat_info

    def _detect_downbeats(
        self,
        y: np.ndarray,
        sr: int,
        beat_times: np.ndarray
    ) -> Optional[list]:
        """
        检测下拍（每小节的第一拍）

        参数:
            y: 音频信号
            sr: 采样率
            beat_times: 节拍时间

        返回:
            下拍时间列表或 None
        """
        try:
            import librosa

            # 计算起始强度
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # 获取节拍强度
            beat_frames = librosa.time_to_frames(beat_times, sr=sr)
            beat_frames = beat_frames[beat_frames < len(onset_env)]

            if len(beat_frames) < 4:
                return None

            beat_strengths = onset_env[beat_frames]

            # 查找强拍（潜在下拍）
            # 假设 4/4 拍号
            downbeats = []
            for i in range(0, len(beat_times), 4):
                if i < len(beat_times):
                    downbeats.append(beat_times[i])

            return downbeats

        except Exception as e:
            logger.warning(f"无法检测下拍: {e}")
            return None

    def estimate_tempo(self, audio_path: str) -> float:
        """
        快速速度估算

        参数:
            audio_path: 音频文件路径

        返回:
            估算的 BPM
        """
        import librosa

        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        if hasattr(tempo, '__len__'):
            return float(tempo[0]) if len(tempo) > 0 else 120.0
        return float(tempo)
