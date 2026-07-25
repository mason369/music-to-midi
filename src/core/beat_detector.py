"""
节拍检测模块 - 使用 librosa

多算法融合 BPM 检测：
- 使用多种检测方法（beat_track, tempo, onset_strength）
- 投票选择最可靠的结果
- 倍频/半频修正
"""
import logging
from typing import Optional, Callable, List, Tuple
import math
import numpy as np

from src.i18n.translator import Translator
from src.models.data_models import Config, BeatInfo

logger = logging.getLogger(__name__)


def _librosa_tempo(librosa_module, **kwargs):
    rhythm = getattr(getattr(librosa_module, "feature", None), "rhythm", None)
    tempo_fn = getattr(rhythm, "tempo", None)
    if tempo_fn is None:
        tempo_fn = librosa_module.beat.tempo
    return tempo_fn(**kwargs)


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
        self._translator = Translator(getattr(config, "language", Translator.DEFAULT_LANGUAGE))

    def _pt(self, key: str, **kwargs) -> str:
        return self._translator.t(key, **kwargs)

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
            progress_callback(0.0, self._pt("progress.loading_audio_for_beat"))

        logger.info(f"正在检测节拍: {audio_path}")

        # 22050Hz 对节拍检测已足够，避免 44100Hz 浪费内存和计算
        logger.info("正在加载音频用于节拍检测 (sr=22050)...")
        y, sr = librosa.load(audio_path, sr=22050)
        logger.info(f"音频加载完成: {len(y)/sr:.1f}秒, {len(y)} 采样点")

        if progress_callback:
            progress_callback(0.2, self._pt("progress.analyzing_tempo"))

        # 使用多算法检测 BPM
        logger.info("开始多算法 BPM 检测...")
        tempo, all_tempos = self._detect_multi_method(y, sr)
        logger.info(f"多算法 BPM 检测完成: {tempo:.1f} BPM, 候选值: {all_tempos}")

        # 倍频校正：tempogram 类算法存在系统性半频/倍频误检（错误值会占
        # 多数候选而赢投票），以无先验跟拍的拍间隔中位数做校验
        interval_median = self._beat_interval_median(librosa, y, sr)
        corrected = self._resolve_octave_by_interval_median(tempo, interval_median)
        if corrected != tempo:
            logger.info(
                "BPM 倍频校正: %.1f -> %.1f（拍间隔中位数 %.1f）",
                tempo,
                corrected,
                interval_median,
            )
            tempo = corrected

        if progress_callback:
            progress_callback(0.5, self._pt("progress.finding_beats"))

        # 使用检测到的 BPM 重新计算节拍位置
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, bpm=tempo)

        # 将帧转换为时间
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        if progress_callback:
            progress_callback(0.8, self._pt("progress.detecting_downbeats"))

        # 尝试检测下拍并推断拍号
        downbeats, beats_per_bar = self._detect_downbeats(y, sr, beat_times)

        if progress_callback:
            progress_callback(1.0, f"BPM: {tempo:.1f}")

        # 变速 tempo map 默认关闭（config.enable_tempo_map）；开启时做高置信
        # 分段检测，恒速或检测失败返回空列表（退回全局单一 BPM）
        tempo_map = (
            self._detect_tempo_map(y, sr)
            if getattr(self.config, "enable_tempo_map", False)
            else []
        )
        if tempo_map:
            logger.info(
                "检测到变速 tempo map: %s",
                ", ".join(f"{t:.1f}s→{bpm:.1f}" for t, bpm in tempo_map),
            )

        beat_info = BeatInfo(
            bpm=tempo,
            beat_times=beat_times.tolist(),
            downbeats=downbeats,
            time_signature=(beats_per_bar, 4),
            tempo_map=tempo_map,
        )

        logger.info(f"多算法检测 BPM: {tempo:.1f}, 候选值: {all_tempos}")
        logger.info(f"检测到 {len(beat_times)} 个节拍")

        return beat_info

    _OCTAVE_CORRECT_MIN_RATIO = 1.9  # 严格倍频判定区间 [1.9, 2.1]
    _OCTAVE_CORRECT_MAX_RATIO = 2.1

    def _beat_interval_median(
        self,
        librosa_module,
        y: np.ndarray,
        sr: int,
    ) -> Optional[float]:
        """无先验 beat_track 的逐拍间隔中位数 BPM；跟拍不可靠时返回 None。"""
        try:
            onset_env = librosa_module.onset.onset_strength(y=y, sr=sr)
            _, beat_frames = librosa_module.beat.beat_track(
                onset_envelope=onset_env, sr=sr, trim=False
            )
            beat_times = librosa_module.frames_to_time(beat_frames, sr=sr)
            intervals = np.diff(beat_times)
            intervals = intervals[intervals > 0]
            if len(intervals) < 16:
                return None
            median_bpm = 60.0 / float(np.median(intervals))
            if not math.isfinite(median_bpm) or median_bpm <= 0:
                return None
            return self._correct_octave_error(median_bpm)
        except Exception as exc:
            logger.warning("拍间隔中位数估计失败: %s", exc)
            return None

    def _resolve_octave_by_interval_median(
        self,
        tempo: float,
        interval_median: Optional[float],
    ) -> float:
        """投票结果与拍间隔中位数呈严格 2x/0.5x 关系时，以拍间隔为准。

        仅倍频关系才替换；非倍频的合理分歧（如双峰歌 117.5/123.0）保持
        投票结果不动，避免用另一种"合理"替换"合理"。
        """
        if interval_median is None or tempo <= 0:
            return tempo
        ratio = interval_median / tempo
        if self._OCTAVE_CORRECT_MIN_RATIO <= ratio <= self._OCTAVE_CORRECT_MAX_RATIO:
            return interval_median
        if self._OCTAVE_CORRECT_MIN_RATIO <= 1.0 / ratio <= self._OCTAVE_CORRECT_MAX_RATIO:
            return interval_median
        return tempo

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
        try:
            import librosa
        except ModuleNotFoundError as exc:
            raise RuntimeError("所有 BPM 检测方法均失败，无法加载 librosa。") from exc

        all_tempos = []

        # 方法1: beat_track（默认方法）
        try:
            tempo1, _ = librosa.beat.beat_track(y=y, sr=sr)
            if hasattr(tempo1, '__len__'):
                if len(tempo1) == 0:
                    raise RuntimeError("beat_track 未返回 BPM")
                tempo1 = float(tempo1[0])
            else:
                tempo1 = float(tempo1)
            if not math.isfinite(tempo1) or tempo1 <= 0:
                raise RuntimeError(f"beat_track 返回无效 BPM: {tempo1}")
            all_tempos.append(tempo1)
            logger.debug(f"方法1 (beat_track): {tempo1:.1f} BPM")
        except Exception as e:
            logger.warning(f"方法1失败: {e}")

        # 方法2: tempo with onset envelope (median)
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            # 使用 aggregate=None 获取所有候选值，然后手动计算中位数
            # 这样可以避免某些 librosa 版本中 aggregate 参数的兼容性问题
            tempo_candidates = _librosa_tempo(
                librosa,
                onset_envelope=onset_env,
                sr=sr,
                aggregate=None,
            )
            if hasattr(tempo_candidates, '__len__') and len(tempo_candidates) > 0:
                tempo2 = float(np.median(tempo_candidates))
            else:
                tempo2 = float(tempo_candidates)
            if not math.isfinite(tempo2) or tempo2 <= 0:
                raise RuntimeError(f"tempo median 返回无效 BPM: {tempo2}")
            all_tempos.append(tempo2)
            logger.debug(f"方法2 (onset median): {tempo2:.1f} BPM")
        except Exception as e:
            logger.warning(f"方法2失败: {e}")

        # 方法3: tempo (mean aggregate)
        try:
            # 使用 aggregate=None 获取所有候选值，然后手动计算平均值
            tempo_candidates = _librosa_tempo(librosa, y=y, sr=sr, aggregate=None)
            if hasattr(tempo_candidates, '__len__') and len(tempo_candidates) > 0:
                tempo3 = float(np.mean(tempo_candidates))
            else:
                tempo3 = float(tempo_candidates)
            if not math.isfinite(tempo3) or tempo3 <= 0:
                raise RuntimeError(f"tempo mean 返回无效 BPM: {tempo3}")
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
            # 只考虑 40-240 BPM 范围内的峰值
            valid_mask = (tempo_freqs >= 40) & (tempo_freqs <= 240)
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                best_idx = valid_indices[np.argmax(tempogram_mean[valid_indices])]
                tempo4 = float(tempo_freqs[best_idx])
                if not math.isfinite(tempo4) or tempo4 <= 0:
                    raise RuntimeError(f"tempogram 返回无效 BPM: {tempo4}")
                all_tempos.append(tempo4)
                logger.debug(f"方法4 (tempogram): {tempo4:.1f} BPM")
        except Exception as e:
            logger.warning(f"方法4失败: {e}")

        if not all_tempos:
            raise RuntimeError("所有 BPM 检测方法均失败，无法生成可靠 tempo。")

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
        valid_range: Tuple[float, float] = (40.0, 240.0)
    ) -> float:
        """
        倍频误差修正

        将 BPM 修正到合理范围内（40-240）

        参数:
            tempo: 原始 BPM
            valid_range: 有效范围 (min, max)

        返回:
            修正后的 BPM
        """
        min_bpm, max_bpm = valid_range

        if not math.isfinite(float(tempo)) or tempo <= 0:
            raise RuntimeError(f"检测到无效 BPM: {tempo}")

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
            raise RuntimeError("BPM 候选列表为空，无法生成可靠 tempo。")

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
    ) -> Tuple[Optional[list], int]:
        """
        检测下拍（每小节的第一拍）

        通过比较不同拍号假设（2/4, 3/4, 4/4, 6/8）下的节拍强度周期性，
        自动推断拍号并标记下拍位置。

        参数:
            y: 音频信号
            sr: 采样率
            beat_times: 节拍时间

        返回:
            (下拍时间列表或 None, 每小节拍数)
        """
        try:
            import librosa

            if len(beat_times) < 4:
                return None, 4

            # 计算起始强度
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # 获取每个 beat 对应的 onset strength
            beat_frames = librosa.time_to_frames(beat_times, sr=sr)
            beat_frames = np.clip(beat_frames, 0, len(onset_env) - 1)

            beat_strengths = onset_env[beat_frames]

            # 自动推断每小节的拍数：对节拍强度序列做自相关分析
            # 候选拍号：2/4, 3/4, 4/4, 6/8
            candidates_bpb = [2, 3, 4, 6]
            best_bpb = 4  # 默认 4/4
            best_score = -1.0

            if len(beat_strengths) >= 8:
                # 归一化节拍强度
                bs_norm = beat_strengths - np.mean(beat_strengths)
                bs_std = np.std(bs_norm)
                if bs_std > 1e-6:
                    bs_norm = bs_norm / bs_std

                for bpb in candidates_bpb:
                    if len(beat_strengths) < bpb * 2:
                        continue
                    # 计算以 bpb 为周期的自相关值
                    n = len(bs_norm)
                    if bpb < n:
                        corr = np.mean(bs_norm[:n - bpb] * bs_norm[bpb:])
                    else:
                        corr = 0.0

                    # 同时计算下拍位置的平均强度（越高越好）
                    downbeat_strengths = []
                    for offset in range(bpb):
                        avg = np.mean(beat_strengths[offset::bpb])
                        downbeat_strengths.append(avg)
                    # 下拍强度对比：最强位置与平均值的差异
                    contrast = max(downbeat_strengths) - np.mean(downbeat_strengths)

                    # 综合评分：自相关 + 强度对比
                    score = corr * 0.6 + contrast * 0.4
                    if score > best_score:
                        best_score = score
                        best_bpb = bpb

            logger.debug(f"推断每小节拍数: {best_bpb}")

            # 在前 best_bpb 拍中找到 onset strength 最强的作为第一个下拍
            search_range = min(best_bpb, len(beat_strengths))
            first_downbeat_idx = int(np.argmax(beat_strengths[:search_range]))

            # 从第一个下拍开始，按推断的拍号间距标记下拍
            downbeats = []
            for i in range(first_downbeat_idx, len(beat_times), best_bpb):
                downbeats.append(beat_times[i])

            return downbeats, best_bpb

        except Exception as e:
            logger.warning(f"无法检测下拍: {e}")
            return None, 4

    def estimate_tempo(self, audio_path: str) -> float:
        """
        快速速度估算（使用多算法融合）

        参数:
            audio_path: 音频文件路径

        返回:
            估算的 BPM
        """
        import librosa

        # 只加载前 60 秒以加快速度（22050Hz 对节拍检测已足够）
        y, sr = librosa.load(audio_path, sr=22050, duration=60)

        # 使用多算法检测
        tempo, _ = self._detect_multi_method(y, sr)

        logger.info(f"快速估算 BPM: {tempo:.1f}")
        return tempo


    # ---- 变速 tempo map 检测 ----
    # 双引擎：
    # 1) beat 引擎：逐拍间隔（分辨率高，可捕捉约 ±3% 的段落级速度流动），
    #    要求 beat_track 跟拍覆盖充分；
    # 2) frame 引擎：逐帧 tempo 分箱分段（无需跟拍，覆盖 beat_track
    #    跟丢的突变型变速，但容忍阈值更高）。

    _TEMPO_MAP_MIN_SECONDS = 12.0   # 音频太短不做变速分析

    # beat 引擎参数
    _TEMPO_MAP_BEAT_MIN_BEATS = 32      # 拍数太少不做 beat 分析
    _TEMPO_MAP_BEAT_MIN_COVERAGE = 0.7  # 跟拍需覆盖至少 70% 时长
    _TEMPO_MAP_BEAT_SMOOTH = 8          # 滑动中位数窗口（拍）
    _TEMPO_MAP_BEAT_TOLERANCE = 0.02    # 段内允许的相对偏差
    _TEMPO_MAP_BEAT_MIN_RUN = 16        # 连续偏离达到该拍数才确认变速
    _TEMPO_MAP_BEAT_MAX_SECTIONS = 8    # 真变速通常只有几次变化
    _TEMPO_MAP_BEAT_MIN_DWELL = 10.0    # 每段最短驻留（秒）
    _TEMPO_MAP_BEAT_MIN_JUMP = 0.03     # 相邻段最小相对变化

    # frame 引擎参数（兜底，容忍更大抖动）
    _TEMPO_MAP_BIN_SECONDS = 0.5        # 逐帧 tempo 分箱宽度（秒）
    _TEMPO_MAP_SMOOTH_BINS = 5          # 分箱滑动中位数窗口（约 2.5 秒）
    _TEMPO_MAP_TOLERANCE = 0.05         # 段内允许的相对偏差
    _TEMPO_MAP_MIN_RUN = 4              # 连续偏离达到该分箱数（约 2 秒）才确认变速
    _TEMPO_MAP_MAX_SECTIONS = 6
    _TEMPO_MAP_MIN_SECTION_SECONDS = 8.0
    _TEMPO_MAP_MIN_SECTION_JUMP = 0.12

    def _detect_tempo_map(
        self,
        y: np.ndarray,
        sr: int,
    ) -> List[Tuple[float, float]]:
        """检测歌曲内部的速度变化，返回分段 tempo map。

        返回 [(秒, BPM), ...]，首点为 0 秒；恒速歌曲返回空列表。
        任何一步失败都返回空列表（退回恒速处理），绝不让变速分析影响主流程。
        """
        try:
            import librosa

            duration = len(y) / sr
            if duration < self._TEMPO_MAP_MIN_SECONDS:
                return []
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            sections = self._detect_beat_tempo_map(librosa, onset_env, sr, duration)
            if sections:
                return sections
            return self._detect_frame_tempo_map(librosa, onset_env, sr, duration)
        except Exception as exc:
            logger.warning("变速 tempo map 检测失败，按恒速处理: %s", exc)
            return []

    def _detect_beat_tempo_map(
        self,
        librosa,
        onset_env: np.ndarray,
        sr: int,
        duration: float,
    ) -> List[Tuple[float, float]]:
        """beat 引擎：逐拍间隔平滑分段，捕捉段落级速度流动。"""
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, trim=False
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if len(beat_times) < self._TEMPO_MAP_BEAT_MIN_BEATS:
            return []
        coverage = (beat_times[-1] - beat_times[0]) / duration if duration > 0 else 0.0
        if coverage < self._TEMPO_MAP_BEAT_MIN_COVERAGE:
            return []

        intervals = np.diff(beat_times)
        valid = intervals > 0
        if not np.any(valid):
            return []
        instant_bpms = 60.0 / intervals[valid]
        interval_end_times = beat_times[1:][valid]

        smoothed = self._sliding_median(instant_bpms, self._TEMPO_MAP_BEAT_SMOOTH)
        sections = self._segment_tempo_curve(
            smoothed,
            interval_end_times,
            tolerance=self._TEMPO_MAP_BEAT_TOLERANCE,
            min_run=self._TEMPO_MAP_BEAT_MIN_RUN,
        )
        return self._gate_tempo_map(
            sections,
            duration,
            max_sections=self._TEMPO_MAP_BEAT_MAX_SECTIONS,
            min_section_seconds=self._TEMPO_MAP_BEAT_MIN_DWELL,
            min_section_jump=self._TEMPO_MAP_BEAT_MIN_JUMP,
        )

    def _detect_frame_tempo_map(
        self,
        librosa,
        onset_env: np.ndarray,
        sr: int,
        duration: float,
    ) -> List[Tuple[float, float]]:
        """frame 引擎：逐帧 tempo 分箱分段，兜底突变型变速。"""
        frame_tempos = _librosa_tempo(
            librosa,
            onset_envelope=onset_env,
            sr=sr,
            aggregate=None,
        )
        frame_tempos = np.asarray(frame_tempos, dtype=float).ravel()
        if frame_tempos.size == 0:
            return []
        frame_times = librosa.frames_to_time(np.arange(frame_tempos.size), sr=sr)
        if len(frame_times) == 0:
            return []

        frame_tempos = np.array(
            [self._correct_octave_error(t) for t in frame_tempos], dtype=float
        )

        bin_sec = self._TEMPO_MAP_BIN_SECONDS
        n_bins = int(np.ceil(frame_times[-1] / bin_sec))
        bin_bpms: List[float] = []
        bin_end_times: List[float] = []
        for i in range(n_bins):
            mask = (frame_times >= i * bin_sec) & (frame_times < (i + 1) * bin_sec)
            if not np.any(mask):
                continue
            bin_bpms.append(float(np.median(frame_tempos[mask])))
            bin_end_times.append(float((i + 1) * bin_sec))
        if len(bin_bpms) < self._TEMPO_MAP_MIN_RUN * 2:
            return []

        smoothed = self._sliding_median(
            np.array(bin_bpms), self._TEMPO_MAP_SMOOTH_BINS
        )
        sections = self._segment_tempo_curve(
            smoothed,
            np.array(bin_end_times),
            tolerance=self._TEMPO_MAP_TOLERANCE,
            min_run=self._TEMPO_MAP_MIN_RUN,
        )
        return self._gate_tempo_map(
            sections,
            duration,
            max_sections=self._TEMPO_MAP_MAX_SECTIONS,
            min_section_seconds=self._TEMPO_MAP_MIN_SECTION_SECONDS,
            min_section_jump=self._TEMPO_MAP_MIN_SECTION_JUMP,
        )

    @staticmethod
    def _gate_tempo_map(
        sections: List[Tuple[float, float]],
        duration: float,
        *,
        max_sections: int = 6,
        min_section_seconds: float = 8.0,
        min_section_jump: float = 0.12,
    ) -> List[Tuple[float, float]]:
        """高置信门控：只有"少而显著"的分段才采纳为变速。

        检测信号在节奏密度变化时会产出倍频族抖动（如 70/92/144 快速
        交替），把它们写进 MIDI 比固定 BPM 更糟糕。因此要求：段数不超过
        上限、每段驻留不短于阈值、相邻段 BPM 相对差不小于阈值；任一不
        满足即退回恒速（返回空列表）。
        """
        if len(sections) < 2 or len(sections) > max_sections:
            return []
        for (_, bpm_a), (_, bpm_b) in zip(sections, sections[1:]):
            if abs(bpm_b - bpm_a) < min_section_jump * bpm_a:
                return []
        section_ends = [sec for sec, _ in sections[1:]] + [duration]
        for (start, _), end in zip(sections, section_ends):
            if end - start < min_section_seconds:
                return []
        return sections

    @staticmethod
    def _sliding_median(values: np.ndarray, window: int) -> np.ndarray:
        """滑动窗口中位数平滑，抑制逐拍检测抖动。"""
        half = window // 2
        smoothed = np.empty(len(values), dtype=float)
        for i in range(len(values)):
            lo = max(0, i - half)
            hi = min(len(values), i + half + 1)
            smoothed[i] = float(np.median(values[lo:hi]))
        return smoothed

    @staticmethod
    def _segment_tempo_curve(
        smoothed_bpms,
        interval_end_times,
        *,
        tolerance: float,
        min_run: int,
    ) -> List[Tuple[float, float]]:
        """把平滑后的逐区间 BPM 曲线切成若干恒定段。

        连续 min_run 个区间偏离当前段 BPM 超过 tolerance 才确认变速；
        相邻段 BPM 差异不显著的变化点会被丢弃。返回 [(秒, BPM), ...]，
        不足两段时返回空列表（恒速）。
        """
        n = len(smoothed_bpms)
        if n < min_run * 2:
            return []

        change_indices: List[int] = []
        current_bpm = float(np.median(smoothed_bpms[:min_run]))
        run_start: Optional[int] = None
        for i in range(n):
            if abs(float(smoothed_bpms[i]) - current_bpm) <= tolerance * current_bpm:
                run_start = None
                continue
            if run_start is None:
                run_start = i
            if i - run_start + 1 >= min_run:
                change_indices.append(run_start)
                current_bpm = float(np.median(smoothed_bpms[run_start : i + 1]))
                run_start = None

        if not change_indices:
            return []

        boundaries = [0] + change_indices + [n]
        segment_bpms = [
            float(np.median(smoothed_bpms[boundaries[k] : boundaries[k + 1]]))
            for k in range(len(boundaries) - 1)
        ]
        tempo_map: List[Tuple[float, float]] = [(0.0, segment_bpms[0])]
        for k, change_idx in enumerate(change_indices):
            new_bpm = segment_bpms[k + 1]
            last_bpm = tempo_map[-1][1]
            if abs(new_bpm - last_bpm) <= tolerance * last_bpm:
                continue  # 段间差异不显著，视为恒速抖动
            # 变化点记为新段第一个区间的起始拍时刻
            tempo_map.append((float(interval_end_times[change_idx - 1]), new_bpm))
        return tempo_map if len(tempo_map) >= 2 else []
