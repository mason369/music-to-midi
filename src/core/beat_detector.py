"""
节拍检测模块 - 使用 librosa

多算法融合 BPM 检测：
- 使用多种检测方法（beat_track, tempo, onset_strength）
- 投票选择最可靠的结果
- 倍频/半频修正
"""
import logging
from typing import Optional, Callable, List, Tuple
import numpy as np

from src.models.data_models import Config, BeatInfo

logger = logging.getLogger(__name__)


class BeatDetector:
    """
    使用 librosa 进行节拍和速度检测

    功能特点:
    - 多算法 BPM 融合估算
    - 倍频/半频修正
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

        # 使用更高采样率以获得更精确的 BPM 检测
        y, sr = librosa.load(audio_path, sr=44100)

        if progress_callback:
            progress_callback(0.2, "正在使用多算法分析速度...")

        # 使用多算法检测 BPM
        tempo, all_tempos = self._detect_multi_method(y, sr)

        if progress_callback:
            progress_callback(0.5, "正在查找节拍位置...")

        # 使用检测到的 BPM 重新计算节拍位置
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, bpm=tempo)

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

        logger.info(f"多算法检测 BPM: {tempo:.1f}, 候选值: {all_tempos}")
        logger.info(f"检测到 {len(beat_times)} 个节拍")

        return beat_info

    def _detect_multi_method(
        self,
        y: np.ndarray,
        sr: int
    ) -> Tuple[float, List[float]]:
        """
        多算法融合 BPM 检测

        使用多种方法检测 BPM 并通过投票选择最可靠的结果。

        参数:
            y: 音频信号
            sr: 采样率

        返回:
            (最佳 BPM, 所有候选值列表)
        """
        import librosa

        all_tempos = []

        # 方法1: beat_track（默认方法）
        try:
            tempo1, _ = librosa.beat.beat_track(y=y, sr=sr)
            if hasattr(tempo1, '__len__'):
                tempo1 = float(tempo1[0]) if len(tempo1) > 0 else 120.0
            else:
                tempo1 = float(tempo1)
            all_tempos.append(tempo1)
            logger.debug(f"方法1 (beat_track): {tempo1:.1f} BPM")
        except Exception as e:
            logger.warning(f"方法1失败: {e}")

        # 方法2: tempo with onset envelope (median)
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # 使用 aggregate=None 获取所有候选值，然后手动计算中位数
            # 这样可以避免某些 librosa 版本中 aggregate 参数的兼容性问题
            tempo_candidates = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
            if hasattr(tempo_candidates, '__len__') and len(tempo_candidates) > 0:
                tempo2 = float(np.median(tempo_candidates))
            else:
                tempo2 = float(tempo_candidates)
            all_tempos.append(tempo2)
            logger.debug(f"方法2 (onset median): {tempo2:.1f} BPM")
        except Exception as e:
            logger.warning(f"方法2失败: {e}")

        # 方法3: tempo (mean aggregate)
        try:
            # 使用 aggregate=None 获取所有候选值，然后手动计算平均值
            tempo_candidates = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
            if hasattr(tempo_candidates, '__len__') and len(tempo_candidates) > 0:
                tempo3 = float(np.mean(tempo_candidates))
            else:
                tempo3 = float(tempo_candidates)
            all_tempos.append(tempo3)
            logger.debug(f"方法3 (tempo mean): {tempo3:.1f} BPM")
        except Exception as e:
            logger.warning(f"方法3失败: {e}")

        # 方法4: tempogram (通过傅里叶变换)
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            # 获取最大响应对应的 BPM
            tempo_freqs = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
            tempogram_mean = np.mean(tempogram, axis=1)
            # 只考虑 60-200 BPM 范围内的峰值
            valid_mask = (tempo_freqs >= 60) & (tempo_freqs <= 200)
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                best_idx = valid_indices[np.argmax(tempogram_mean[valid_indices])]
                tempo4 = float(tempo_freqs[best_idx])
                all_tempos.append(tempo4)
                logger.debug(f"方法4 (tempogram): {tempo4:.1f} BPM")
        except Exception as e:
            logger.warning(f"方法4失败: {e}")

        if not all_tempos:
            logger.warning("所有方法都失败，使用默认 120 BPM")
            return 120.0, [120.0]

        # 生成倍频候选（2x 和 0.5x）
        candidates = []
        for t in all_tempos:
            candidates.append(t)
            candidates.append(t * 2)
            candidates.append(t / 2)

        # 修正到合理范围 (60-200 BPM)
        candidates = [self._correct_octave_error(t) for t in candidates]

        # 统计候选值
        logger.debug(f"所有候选 BPM: {[f'{t:.1f}' for t in candidates]}")

        # 选择最佳 BPM：使用聚类投票
        best_tempo = self._vote_best_tempo(candidates, all_tempos)

        return best_tempo, [round(t, 1) for t in all_tempos]

    def _correct_octave_error(
        self,
        tempo: float,
        valid_range: Tuple[float, float] = (60.0, 200.0)
    ) -> float:
        """
        倍频误差修正

        将 BPM 修正到合理范围内（60-200）

        参数:
            tempo: 原始 BPM
            valid_range: 有效范围 (min, max)

        返回:
            修正后的 BPM
        """
        min_bpm, max_bpm = valid_range

        while tempo < min_bpm:
            tempo *= 2
        while tempo > max_bpm:
            tempo /= 2

        return tempo

    def _vote_best_tempo(
        self,
        candidates: List[float],
        original_tempos: List[float]
    ) -> float:
        """
        通过聚类投票选择最佳 BPM

        策略：
        1. 将相近的候选值聚类（阈值 8 BPM）
        2. 选择包含原始检测值最多的聚类
        3. 返回该聚类的中位数

        参数:
            candidates: 所有候选 BPM（包含倍频）
            original_tempos: 原始检测的 BPM 值

        返回:
            最佳 BPM
        """
        if not candidates:
            return 120.0

        # 聚类阈值（BPM 差值在此范围内视为同一组）
        cluster_threshold = 8.0

        # 按值排序
        sorted_candidates = sorted(candidates)

        # 聚类
        clusters: List[List[float]] = []
        current_cluster: List[float] = [sorted_candidates[0]]

        for tempo in sorted_candidates[1:]:
            if tempo - current_cluster[-1] <= cluster_threshold:
                current_cluster.append(tempo)
            else:
                clusters.append(current_cluster)
                current_cluster = [tempo]
        clusters.append(current_cluster)

        # 计算每个聚类包含多少原始检测值
        def count_original_in_cluster(cluster: List[float]) -> int:
            count = 0
            for orig in original_tempos:
                # 检查原始值或其倍频是否在聚类中
                for mult in [1.0, 2.0, 0.5]:
                    adjusted = self._correct_octave_error(orig * mult)
                    if any(abs(adjusted - c) <= cluster_threshold for c in cluster):
                        count += 1
                        break
            return count

        # 选择包含原始值最多的聚类
        best_cluster = max(clusters, key=lambda c: (count_original_in_cluster(c), len(c)))

        # 返回该聚类的中位数
        best_tempo = float(np.median(best_cluster))

        logger.debug(f"选择聚类: {[f'{t:.1f}' for t in best_cluster]}, 最佳 BPM: {best_tempo:.1f}")

        return best_tempo

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
        快速速度估算（使用多算法融合）

        参数:
            audio_path: 音频文件路径

        返回:
            估算的 BPM
        """
        import librosa

        # 只加载前 60 秒以加快速度
        y, sr = librosa.load(audio_path, sr=44100, duration=60)

        # 使用多算法检测
        tempo, _ = self._detect_multi_method(y, sr)

        logger.info(f"快速估算 BPM: {tempo:.1f}")
        return tempo
