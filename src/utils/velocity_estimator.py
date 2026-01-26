"""
基于音频振幅的力度估算模块

从音频波形提取振幅包络，根据每个音符的起始时间估算 MIDI 力度。
用于恢复 YourMT3 模型在 ignore_velocity=True 模式下丢失的动态信息。
"""
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VelocityEstimator:
    """
    从音频振幅估算 MIDI 力度

    使用 RMS 振幅包络和百分位数归一化，将音频动态映射到 MIDI 力度范围。
    保持相对动态关系，而非使用绝对阈值。
    """

    def __init__(
        self,
        audio_path: str,
        velocity_range: Tuple[int, int] = (30, 110),
        hop_length: int = 512,
        sample_rate: int = 16000
    ):
        """
        初始化力度估算器

        参数:
            audio_path: 音频文件路径
            velocity_range: 力度范围 (min, max)，默认 30-110
            hop_length: RMS 计算的跳跃长度，默认 512
            sample_rate: 目标采样率，默认 16000
        """
        self.audio_path = audio_path
        self.velocity_range = velocity_range
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.envelope: Optional[np.ndarray] = None
        self.time_resolution: Optional[float] = None
        self._percentiles: Optional[Tuple[float, float]] = None
        self._is_computed = False

    def compute_envelope(self) -> bool:
        """
        计算 RMS 振幅包络

        返回:
            是否成功计算
        """
        if self._is_computed:
            return True

        try:
            import librosa

            logger.info(f"正在计算振幅包络: {self.audio_path}")

            # 加载音频
            y, sr = librosa.load(self.audio_path, sr=self.sample_rate, mono=True)

            # 计算 RMS 振幅包络
            # hop_length=512, sample_rate=16000 → ~32ms 分辨率
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

            # 计算时间分辨率
            self.time_resolution = self.hop_length / self.sample_rate

            # 存储包络
            self.envelope = rms

            # 计算百分位数（用于归一化）
            # 使用 5%-95% 百分位数避免极端值影响
            if len(rms) > 0:
                p5 = np.percentile(rms, 5)
                p95 = np.percentile(rms, 95)

                # 确保有足够的动态范围
                if p95 - p5 < 1e-6:
                    # 动态范围太小，使用 min-max
                    p5 = np.min(rms)
                    p95 = np.max(rms)

                self._percentiles = (p5, p95)
            else:
                self._percentiles = (0.0, 1.0)

            self._is_computed = True

            duration = len(y) / self.sample_rate
            logger.info(f"振幅包络计算完成: {len(rms)} 帧, {duration:.1f}秒, 分辨率 {self.time_resolution*1000:.1f}ms")
            logger.info(f"动态范围: {self._percentiles[0]:.4f} - {self._percentiles[1]:.4f}")

            return True

        except ImportError:
            logger.warning("librosa 未安装，无法计算振幅包络")
            return False
        except Exception as e:
            logger.error(f"计算振幅包络失败: {e}")
            return False

    def estimate_velocity(self, onset_time: float) -> int:
        """
        根据音符起始时间估算力度

        参数:
            onset_time: 音符起始时间（秒）

        返回:
            估算的 MIDI 力度值 (1-127)
        """
        # 确保包络已计算
        if not self._is_computed:
            if not self.compute_envelope():
                # 计算失败，返回默认力度
                return 80

        if self.envelope is None or self.time_resolution is None:
            return 80

        # 找到 onset_time 对应的包络帧
        frame_idx = int(onset_time / self.time_resolution)

        # 边界检查
        if frame_idx < 0:
            frame_idx = 0
        elif frame_idx >= len(self.envelope):
            frame_idx = len(self.envelope) - 1

        # 获取该帧的 RMS 值
        rms_value = self.envelope[frame_idx]

        # 使用百分位数归一化
        p5, p95 = self._percentiles

        if p95 - p5 < 1e-6:
            # 动态范围太小，返回中间力度
            normalized = 0.5
        else:
            # 归一化到 0-1 范围
            normalized = (rms_value - p5) / (p95 - p5)
            # 裁剪到 0-1
            normalized = np.clip(normalized, 0.0, 1.0)

        # 线性映射到力度范围
        vel_min, vel_max = self.velocity_range
        velocity = int(vel_min + normalized * (vel_max - vel_min))

        # 确保在有效范围内
        velocity = max(1, min(127, velocity))

        return velocity

    def estimate_velocities_batch(self, onset_times: list) -> list:
        """
        批量估算多个音符的力度

        参数:
            onset_times: 音符起始时间列表

        返回:
            力度值列表
        """
        # 确保包络已计算
        if not self._is_computed:
            if not self.compute_envelope():
                return [80] * len(onset_times)

        return [self.estimate_velocity(t) for t in onset_times]

    def get_envelope_stats(self) -> dict:
        """
        获取振幅包络统计信息

        返回:
            包含统计信息的字典
        """
        if not self._is_computed or self.envelope is None:
            return {}

        return {
            "frames": len(self.envelope),
            "time_resolution_ms": self.time_resolution * 1000 if self.time_resolution else 0,
            "rms_min": float(np.min(self.envelope)),
            "rms_max": float(np.max(self.envelope)),
            "rms_mean": float(np.mean(self.envelope)),
            "rms_std": float(np.std(self.envelope)),
            "percentile_5": float(self._percentiles[0]) if self._percentiles else 0,
            "percentile_95": float(self._percentiles[1]) if self._percentiles else 0,
            "velocity_range": self.velocity_range,
        }


def create_velocity_estimator(
    audio_path: str,
    velocity_range: Tuple[int, int] = (30, 110),
    precompute: bool = True
) -> Optional[VelocityEstimator]:
    """
    创建力度估算器的便捷函数

    参数:
        audio_path: 音频文件路径
        velocity_range: 力度范围
        precompute: 是否预先计算包络

    返回:
        VelocityEstimator 实例，失败时返回 None
    """
    try:
        estimator = VelocityEstimator(audio_path, velocity_range)

        if precompute:
            if not estimator.compute_envelope():
                logger.warning("预计算振幅包络失败，力度估算将使用默认值")
                return None

        return estimator

    except Exception as e:
        logger.error(f"创建力度估算器失败: {e}")
        return None
