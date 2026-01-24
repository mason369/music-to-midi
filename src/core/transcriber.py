"""
音频转MIDI模块 - 使用 Basic Pitch
"""
import logging
from pathlib import Path
from typing import List, Optional, Callable, Tuple
import numpy as np

from src.models.data_models import Config, NoteEvent, TrackType
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
        track_type: TrackType = TrackType.OTHER,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[NoteEvent]:
        """
        将音频文件转写为MIDI音符事件

        参数:
            audio_path: 音频文件路径
            track_type: 轨道类型（影响处理参数）
            progress_callback: 可选的进度回调

        返回:
            NoteEvent 对象列表
        """
        logger.info(f"正在转写: {audio_path} (类型: {track_type.value})")

        if progress_callback:
            progress_callback(0.0, f"正在转写 {track_type.value}...")

        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH

            # 根据轨道类型调整阈值
            onset_thresh, frame_thresh = self._get_thresholds(track_type)

            if progress_callback:
                progress_callback(0.3, "正在进行音高检测...")

            # 运行预测
            model_output, midi_data, note_events = predict(
                audio_path,
                onset_threshold=onset_thresh,
                frame_threshold=frame_thresh,
                minimum_note_length=self.min_note_length,
                minimum_frequency=None,
                maximum_frequency=None,
                multiple_pitch_bends=False,
                melodia_trick=True
            )

            if progress_callback:
                progress_callback(0.8, "正在处理音符事件...")

            # 转换为我们的 NoteEvent 格式
            notes = self._process_note_events(note_events, track_type)

            if progress_callback:
                progress_callback(1.0, f"发现 {len(notes)} 个音符")

            logger.info(f"从 {track_type.value} 转写了 {len(notes)} 个音符")
            return notes

        except ImportError as e:
            logger.error("Basic Pitch 未安装，请运行: pip install basic-pitch")
            raise ImportError("转写需要 Basic Pitch 库") from e

    def _get_thresholds(self, track_type: TrackType) -> Tuple[float, float]:
        """获取针对轨道类型优化的阈值"""
        # 根据轨道特征调整阈值
        if track_type == TrackType.DRUMS:
            # 鼓使用较高阈值（瞬态较多）
            return 0.6, 0.4
        elif track_type == TrackType.BASS:
            # 贝斯使用较低阈值（持续音符）
            return 0.4, 0.25
        elif track_type == TrackType.VOCALS:
            # 人声使用中等阈值
            return 0.5, 0.3
        else:
            # 默认阈值
            return self.onset_threshold, self.frame_threshold

    def _process_note_events(
        self,
        note_events: List,
        track_type: TrackType
    ) -> List[NoteEvent]:
        """
        将原始音符事件处理为 NoteEvent 对象

        参数:
            note_events: Basic Pitch 的原始音符事件
            track_type: 用于力度调整的轨道类型

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

            # 应用轨道特定的力度缩放
            midi_velocity = self._adjust_velocity(midi_velocity, track_type)

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

    def _adjust_velocity(self, velocity: int, track_type: TrackType) -> int:
        """根据轨道类型调整力度"""
        if track_type == TrackType.DRUMS:
            # 鼓通常更响
            velocity = int(velocity * 1.1)
        elif track_type == TrackType.BASS:
            # 贝斯通常更一致
            velocity = int(velocity * 0.95)
        elif track_type == TrackType.VOCALS:
            # 人声有动态范围
            pass

        return int(np.clip(velocity, 1, 127))

    def transcribe_to_midi(
        self,
        audio_path: str,
        output_path: str,
        track_type: TrackType = TrackType.OTHER
    ) -> str:
        """
        将音频直接转写为 MIDI 文件

        参数:
            audio_path: 输入音频路径
            output_path: 输出 MIDI 路径
            track_type: 轨道类型

        返回:
            输出 MIDI 文件路径
        """
        from basic_pitch.inference import predict_and_save

        logger.info(f"正在转写为 MIDI: {audio_path} -> {output_path}")

        onset_thresh, frame_thresh = self._get_thresholds(track_type)

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
