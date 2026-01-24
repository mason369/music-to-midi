"""
音频转MIDI模块 - 使用 Basic Pitch
支持乐器类型特定的参数优化
"""
import logging
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Union
import numpy as np

from src.models.data_models import Config, NoteEvent, TrackType, InstrumentType
from src.utils.gpu_utils import get_device

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    使用 Basic Pitch (Spotify) 进行音频转MIDI转写

    功能特点:
    - 多音音高检测
    - 乐器无关转写
    - 音高弯曲检测
    """

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

        if progress_callback:
            progress_callback(0.0, f"正在转写 {type_name}...")

        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH

            # 根据乐器类型调整阈值
            onset_thresh, frame_thresh = self._get_thresholds_for_instrument(instrument)

            # 获取乐器特定的频率范围
            min_freq, max_freq = self._get_frequency_range(instrument)

            if progress_callback:
                progress_callback(0.3, "正在进行音高检测...")

            # 运行预测
            model_output, midi_data, note_events = predict(
                audio_path,
                onset_threshold=onset_thresh,
                frame_threshold=frame_thresh,
                minimum_note_length=self.min_note_length,
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
                multiple_pitch_bends=False,
                melodia_trick=True
            )

            if progress_callback:
                progress_callback(0.8, "正在处理音符事件...")

            # 转换为我们的 NoteEvent 格式
            notes = self._process_note_events_v2(note_events, instrument)

            if progress_callback:
                progress_callback(1.0, f"发现 {len(notes)} 个音符")

            logger.info(f"从 {type_name} 转写了 {len(notes)} 个音符")
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
            # 钢琴：较低阈值以捕捉持续音符
            InstrumentType.PIANO: (0.45, 0.28),
            # 鼓：较高阈值（瞬态较多）
            InstrumentType.DRUMS: (0.6, 0.4),
            # 贝斯：较低阈值（持续音符，低频）
            InstrumentType.BASS: (0.4, 0.25),
            # 吉他：中等阈值
            InstrumentType.GUITAR: (0.5, 0.3),
            # 人声：中等阈值
            InstrumentType.VOCALS: (0.5, 0.3),
            # 弦乐：较低阈值（持续音符）
            InstrumentType.STRINGS: (0.45, 0.28),
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
