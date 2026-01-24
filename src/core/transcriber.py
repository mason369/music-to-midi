"""
音频转MIDI模块 - 使用 Basic Pitch
支持乐器类型特定的参数优化
"""
import logging
import os
import sys
from io import StringIO
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Union
from contextlib import contextmanager
import numpy as np

from src.models.data_models import Config, NoteEvent, TrackType, InstrumentType
from src.utils.gpu_utils import get_device, configure_tensorflow_gpu

logger = logging.getLogger(__name__)


@contextmanager
def suppress_stdout():
    """临时抑制标准输出"""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


class AudioTranscriber:
    """
    使用 Basic Pitch (Spotify) 进行音频转MIDI转写

    功能特点:
    - 多音音高检测
    - 乐器无关转写
    - 音高弯曲检测
    """

    # 标记是否已配置 TensorFlow GPU
    _tf_gpu_configured = False

    def __init__(self, config: Config):
        """
        初始化转写器

        参数:
            config: 应用配置
        """
        self.config = config
        self.onset_threshold = config.onset_threshold
        self.frame_threshold = config.frame_threshold
        self.min_note_length = config.min_note_length
        self.device = get_device(config.use_gpu, config.gpu_device)
        self._cancelled = False
        self._cancel_check_callback = None

        # 配置 TensorFlow GPU（仅首次）
        if not AudioTranscriber._tf_gpu_configured:
            if config.use_gpu:
                configure_tensorflow_gpu()
            AudioTranscriber._tf_gpu_configured = True

    def set_cancel_check(self, callback) -> None:
        """
        设置取消检查回调

        参数:
            callback: 返回 True 表示已取消的回调函数
        """
        self._cancel_check_callback = callback

    def cancel(self) -> None:
        """取消正在进行的处理"""
        self._cancelled = True
        logger.info("转写器：处理已取消")

    def reset_cancel(self) -> None:
        """重置取消标志"""
        self._cancelled = False

    def _check_cancelled(self) -> None:
        """检查是否已取消，如果是则抛出异常"""
        if self._cancelled:
            raise InterruptedError("转写处理已取消")
        if self._cancel_check_callback and self._cancel_check_callback():
            raise InterruptedError("转写处理已取消")

    def transcribe(
        self,
        audio_path: str,
        track_type: Union[TrackType, InstrumentType] = TrackType.OTHER,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[NoteEvent]:
        """
        将音频文件转写为MIDI音符事件

        参数:
            audio_path: 音频文件路径
            track_type: 轨道类型或乐器类型（影响处理参数）
            progress_callback: 可选的进度回调

        返回:
            NoteEvent 对象列表
        """
        # 统一转换为 InstrumentType
        instrument = self._to_instrument_type(track_type)
        type_name = instrument.value if isinstance(instrument, InstrumentType) else track_type.value

        logger.info(f"正在转写: {audio_path} (类型: {type_name})")

        # 检查取消
        self._check_cancelled()

        if progress_callback:
            progress_callback(0.0, f"正在转写 {type_name}...")

        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH

            # 检查取消
            self._check_cancelled()

            # 根据乐器类型调整阈值
            onset_thresh, frame_thresh = self._get_thresholds_for_instrument(instrument)

            # 获取乐器特定的频率范围
            min_freq, max_freq = self._get_frequency_range(instrument)

            # 获取乐器特定的最小音符长度
            min_note_len = self._get_min_note_length_for_instrument(instrument)

            logger.debug(f"转写参数: onset={onset_thresh:.2f}, frame={frame_thresh:.2f}")
            logger.debug(f"频率范围: {min_freq}Hz - {max_freq}Hz")
            logger.debug(f"最小音符长度: {min_note_len * 1000:.0f}ms")

            if progress_callback:
                progress_callback(0.3, "正在进行音高检测...")

            # 运行预测（抑制 basic_pitch 的英文输出）
            with suppress_stdout():
                model_output, midi_data, note_events = predict(
                    audio_path,
                    onset_threshold=onset_thresh,
                    frame_threshold=frame_thresh,
                    minimum_note_length=min_note_len,
                    minimum_frequency=min_freq,
                    maximum_frequency=max_freq,
                    multiple_pitch_bends=False,
                    melodia_trick=True
                )

            # 检查取消（预测完成后）
            self._check_cancelled()

            if progress_callback:
                progress_callback(0.8, "正在处理音符事件...")

            # 转换为我们的 NoteEvent 格式
            notes = self._process_note_events_v2(note_events, instrument)

            # 检查取消
            self._check_cancelled()

            if progress_callback:
                progress_callback(1.0, f"发现 {len(notes)} 个音符")

            logger.info(f"从 {type_name} 转写了 {len(notes)} 个音符")
            if notes:
                pitches = [n.pitch for n in notes]
                logger.debug(f"音高范围: MIDI {min(pitches)} - {max(pitches)}")
                logger.debug(f"时长范围: {notes[0].start_time:.2f}s - {notes[-1].end_time:.2f}s")
            return notes

        except ImportError as e:
            logger.error("Basic Pitch 未安装，请运行: pip install basic-pitch")
            raise ImportError("转写需要 Basic Pitch 库") from e

    def _to_instrument_type(
        self,
        track_type: Union[TrackType, InstrumentType]
    ) -> InstrumentType:
        """将 TrackType 转换为 InstrumentType"""
        if isinstance(track_type, InstrumentType):
            return track_type
        return InstrumentType.from_track_type(track_type)

    def _get_thresholds_for_instrument(
        self,
        instrument: InstrumentType
    ) -> Tuple[float, float]:
        """获取针对乐器类型优化的阈值"""
        thresholds = {
            # 钢琴：提高阈值减少碎片音符
            InstrumentType.PIANO: (0.50, 0.35),
            # 鼓：降低onset阈值（瞬态）
            InstrumentType.DRUMS: (0.55, 0.40),
            # 贝斯：适度提高阈值减少噪声
            InstrumentType.BASS: (0.45, 0.30),
            # 吉他：保持中等阈值
            InstrumentType.GUITAR: (0.50, 0.32),
            # 人声：保持中等阈值
            InstrumentType.VOCALS: (0.50, 0.30),
            # 弦乐：提高阈值减少噪声
            InstrumentType.STRINGS: (0.48, 0.32),
            # 其他：默认阈值
            InstrumentType.OTHER: (self.onset_threshold, self.frame_threshold),
        }
        return thresholds.get(instrument, (self.onset_threshold, self.frame_threshold))

    def _get_frequency_range(
        self,
        instrument: InstrumentType
    ) -> Tuple[Optional[float], Optional[float]]:
        """获取乐器的频率范围（Hz）"""
        # None 表示使用默认范围
        ranges = {
            # 钢琴：A0 (27.5 Hz) 到 C8 (4186 Hz)
            InstrumentType.PIANO: (27.5, 4200.0),
            # 鼓：不限制频率，因为鼓有广泛的频率范围
            InstrumentType.DRUMS: (None, None),
            # 贝斯：低频范围 E1 (41 Hz) 到 G4 (392 Hz)
            InstrumentType.BASS: (30.0, 500.0),
            # 吉他：E2 (82 Hz) 到 E6 (1319 Hz)
            InstrumentType.GUITAR: (80.0, 1400.0),
            # 人声：E2 (82 Hz) 到 C6 (1047 Hz)
            InstrumentType.VOCALS: (80.0, 1100.0),
            # 弦乐：较宽范围
            InstrumentType.STRINGS: (55.0, 2000.0),
            # 其他：不限制
            InstrumentType.OTHER: (None, None),
        }
        return ranges.get(instrument, (None, None))

    def _get_min_note_length_for_instrument(
        self,
        instrument: InstrumentType
    ) -> float:
        """获取乐器特定的最小音符长度（秒）"""
        # 不同乐器有不同的典型音符长度
        min_lengths = {
            InstrumentType.PIANO: 0.080,    # 80ms - 钢琴音符较短
            InstrumentType.DRUMS: 0.030,    # 30ms - 鼓的瞬态很短
            InstrumentType.BASS: 0.100,     # 100ms - 贝斯音符较长
            InstrumentType.GUITAR: 0.060,   # 60ms - 吉他适中
            InstrumentType.VOCALS: 0.100,   # 100ms - 人声较长
            InstrumentType.STRINGS: 0.120,  # 120ms - 弦乐持续音
            InstrumentType.OTHER: 0.058,    # 58ms - 默认
        }
        return min_lengths.get(instrument, self.min_note_length / 1000.0)

    def _get_complexity_params(self, track_count: int) -> dict:
        """
        根据轨道数量获取复杂度相关参数

        参数映射原理：
        - onset_threshold: 提高 → 减少检测到的音符起始（简化）
        - frame_threshold: 提高 → 过滤弱音符（简化）
        - min_note_length: 增加 → 过滤短音符/装饰音（简化）

        参数:
            track_count: 目标轨道数量（1-4）

        返回:
            包含转写参数的字典
        """
        params = {
            1: {  # 最简化：只保留核心旋律
                'onset_threshold': 0.65,
                'frame_threshold': 0.45,
                'min_note_length': 0.120,  # 120ms
            },
            2: {  # 中等：左右手分离
                'onset_threshold': 0.55,
                'frame_threshold': 0.38,
                'min_note_length': 0.090,  # 90ms
            },
            3: {  # 较高复杂度
                'onset_threshold': 0.50,
                'frame_threshold': 0.35,
                'min_note_length': 0.080,  # 80ms
            },
            4: {  # 完整细节
                'onset_threshold': 0.50,
                'frame_threshold': 0.35,
                'min_note_length': 0.080,  # 80ms
            },
        }
        return params.get(track_count, params[4])

    def transcribe_with_complexity(
        self,
        audio_path: str,
        instrument: InstrumentType,
        track_count: int,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[NoteEvent]:
        """
        根据轨道数量调整转写复杂度

        使用复杂度感知的参数进行转写，轨道数越少参数越严格，
        输出的音符越精简，更接近原曲的核心旋律。

        参数:
            audio_path: 音频文件路径
            instrument: 乐器类型
            track_count: 目标轨道数量（1-4）
            progress_callback: 可选的进度回调

        返回:
            转写的音符事件列表
        """
        type_name = instrument.value if isinstance(instrument, InstrumentType) else str(instrument)

        logger.info(f"正在转写（复杂度控制）: {audio_path} (类型: {type_name}, 轨道数: {track_count})")

        # 检查取消
        self._check_cancelled()

        if progress_callback:
            progress_callback(0.0, f"正在转写 {type_name}（{track_count}轨模式）...")

        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH

            # 检查取消
            self._check_cancelled()

            # 获取复杂度参数
            complexity_params = self._get_complexity_params(track_count)
            onset_thresh = complexity_params['onset_threshold']
            frame_thresh = complexity_params['frame_threshold']
            min_note_len = complexity_params['min_note_length']

            # 获取乐器特定的频率范围
            min_freq, max_freq = self._get_frequency_range(instrument)

            logger.debug(f"复杂度参数（{track_count}轨）: onset={onset_thresh:.2f}, "
                        f"frame={frame_thresh:.2f}, min_note={min_note_len*1000:.0f}ms")
            logger.debug(f"频率范围: {min_freq}Hz - {max_freq}Hz")

            if progress_callback:
                progress_callback(0.3, "正在进行音高检测...")

            # 运行预测（抑制 basic_pitch 的英文输出）
            with suppress_stdout():
                model_output, midi_data, note_events = predict(
                    audio_path,
                    onset_threshold=onset_thresh,
                    frame_threshold=frame_thresh,
                    minimum_note_length=min_note_len,
                    minimum_frequency=min_freq,
                    maximum_frequency=max_freq,
                    multiple_pitch_bends=False,
                    melodia_trick=True
                )

            # 检查取消（预测完成后）
            self._check_cancelled()

            if progress_callback:
                progress_callback(0.8, "正在处理音符事件...")

            # 转换为我们的 NoteEvent 格式
            notes = self._process_note_events_v2(note_events, instrument)

            # 检查取消
            self._check_cancelled()

            if progress_callback:
                progress_callback(1.0, f"发现 {len(notes)} 个音符")

            logger.info(f"从 {type_name} 转写了 {len(notes)} 个音符（{track_count}轨模式）")
            if notes:
                pitches = [n.pitch for n in notes]
                logger.debug(f"音高范围: MIDI {min(pitches)} - {max(pitches)}")
                logger.debug(f"时长范围: {notes[0].start_time:.2f}s - {notes[-1].end_time:.2f}s")
            return notes

        except ImportError as e:
            logger.error("Basic Pitch 未安装，请运行: pip install basic-pitch")
            raise ImportError("转写需要 Basic Pitch 库") from e

    def _get_thresholds(self, track_type: TrackType) -> Tuple[float, float]:
        """获取针对轨道类型优化的阈值（向后兼容）"""
        instrument = self._to_instrument_type(track_type)
        return self._get_thresholds_for_instrument(instrument)

    def _process_note_events_v2(
        self,
        note_events: List,
        instrument: InstrumentType
    ) -> List[NoteEvent]:
        """
        将原始音符事件处理为 NoteEvent 对象（新版）

        参数:
            note_events: Basic Pitch 的原始音符事件
            instrument: 用于力度调整的乐器类型

        返回:
            NoteEvent 对象列表
        """
        notes = []

        for start_time, end_time, pitch, velocity, pitch_bends in note_events:
            # 确保音高在有效的 MIDI 范围内
            midi_pitch = int(round(pitch))
            if midi_pitch < 0 or midi_pitch > 127:
                continue

            # 将力度归一化到 0-127
            midi_velocity = int(np.clip(velocity * 127, 1, 127))

            # 应用乐器特定的力度缩放
            midi_velocity = self._adjust_velocity_for_instrument(midi_velocity, instrument)

            note = NoteEvent(
                pitch=midi_pitch,
                start_time=float(start_time),
                end_time=float(end_time),
                velocity=midi_velocity
            )
            notes.append(note)

        # 按开始时间排序
        notes.sort(key=lambda n: n.start_time)

        return notes

    def _process_note_events(
        self,
        note_events: List,
        track_type: TrackType
    ) -> List[NoteEvent]:
        """
        将原始音符事件处理为 NoteEvent 对象（向后兼容）

        参数:
            note_events: Basic Pitch 的原始音符事件
            track_type: 用于力度调整的轨道类型

        返回:
            NoteEvent 对象列表
        """
        instrument = self._to_instrument_type(track_type)
        return self._process_note_events_v2(note_events, instrument)

    def _adjust_velocity_for_instrument(
        self,
        velocity: int,
        instrument: InstrumentType
    ) -> int:
        """根据乐器类型调整力度"""
        multipliers = {
            InstrumentType.PIANO: 1.0,     # 钢琴保持原始力度
            InstrumentType.DRUMS: 1.1,     # 鼓通常更响
            InstrumentType.BASS: 0.95,     # 贝斯通常更一致
            InstrumentType.GUITAR: 1.0,    # 吉他保持原始
            InstrumentType.VOCALS: 1.0,    # 人声有动态范围
            InstrumentType.STRINGS: 0.95,  # 弦乐较柔和
            InstrumentType.OTHER: 1.0,     # 其他保持原始
        }
        multiplier = multipliers.get(instrument, 1.0)
        adjusted = int(velocity * multiplier)
        return int(np.clip(adjusted, 1, 127))

    def _adjust_velocity(self, velocity: int, track_type: TrackType) -> int:
        """根据轨道类型调整力度（向后兼容）"""
        instrument = self._to_instrument_type(track_type)
        return self._adjust_velocity_for_instrument(velocity, instrument)

    def transcribe_to_midi(
        self,
        audio_path: str,
        output_path: str,
        track_type: Union[TrackType, InstrumentType] = TrackType.OTHER
    ) -> str:
        """
        将音频直接转写为 MIDI 文件

        参数:
            audio_path: 输入音频路径
            output_path: 输出 MIDI 路径
            track_type: 轨道类型或乐器类型

        返回:
            输出 MIDI 文件路径
        """
        from basic_pitch.inference import predict_and_save

        logger.info(f"正在转写为 MIDI: {audio_path} -> {output_path}")

        instrument = self._to_instrument_type(track_type)
        onset_thresh, frame_thresh = self._get_thresholds_for_instrument(instrument)

        # 抑制 basic_pitch 的英文输出
        with suppress_stdout():
            predict_and_save(
                [audio_path],
                Path(output_path).parent,
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                onset_threshold=onset_thresh,
                frame_threshold=frame_thresh,
                minimum_note_length=self.min_note_length
            )

        return output_path
