"""
专业钢琴转写模块 - 使用 ByteDance Piano Transcription

相比 Basic Pitch，该模型针对钢琴优化，具有以下优势：
- 更精确的音符起止时间检测
- 踏板检测（延音踏板、柔音踏板）
- 更准确的力度检测
- 在 MAESTRO 数据集上 Frame AP 达到 0.9285
"""
import logging
import os
from pathlib import Path
from typing import List, Optional, Callable, Tuple

import numpy as np

from src.models.data_models import Config, NoteEvent, PedalEvent
from src.utils.gpu_utils import get_device

logger = logging.getLogger(__name__)


class PianoTranscriberPro:
    """
    使用 ByteDance Piano Transcription 模型的专业钢琴转写器

    参考: https://github.com/bytedance/piano_transcription

    功能特点:
    - 专门针对钢琴优化的深度学习模型
    - 支持延音踏板和柔音踏板检测
    - 更精确的音符时间和力度估计
    - 支持 GPU 加速
    """

    def __init__(self, config: Config):
        """
        初始化钢琴转写器

        参数:
            config: 应用配置
        """
        self.config = config
        self.device = get_device(config.use_gpu, config.gpu_device)
        self.transcriptor = None
        self._model_loaded = False
        self._cancelled = False
        self._cancel_check_callback = None

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
        logger.info("钢琴转写器：处理已取消")

    def reset_cancel(self) -> None:
        """重置取消标志"""
        self._cancelled = False

    def _check_cancelled(self) -> None:
        """检查是否已取消，如果是则抛出异常"""
        if self._cancelled:
            raise InterruptedError("钢琴转写处理已取消")
        if self._cancel_check_callback and self._cancel_check_callback():
            raise InterruptedError("钢琴转写处理已取消")

    def is_available(self) -> bool:
        """检查 ByteDance 模型是否可用"""
        try:
            from piano_transcription_inference import PianoTranscription
            return True
        except ImportError:
            return False

    def load_model(self) -> bool:
        """
        加载 ByteDance Piano Transcription 模型

        返回:
            True 如果加载成功，False 如果失败
        """
        if self._model_loaded and self.transcriptor is not None:
            return True

        logger.info("正在加载 ByteDance Piano Transcription 模型...")
        logger.info(f"使用设备: {self.device}")

        try:
            from piano_transcription_inference import PianoTranscription

            self.transcriptor = PianoTranscription(
                device=self.device,
                checkpoint_path=None  # 使用默认模型（自动下载）
            )

            self._model_loaded = True
            logger.info("ByteDance 钢琴模型加载完成")
            return True

        except ImportError:
            logger.warning(
                "piano_transcription_inference 未安装，"
                "请运行: pip install piano_transcription_inference"
            )
            return False
        except Exception as e:
            logger.error(f"加载 ByteDance 钢琴模型失败: {e}")
            return False

    def unload_model(self) -> None:
        """卸载模型以释放内存"""
        if self.transcriptor is not None:
            del self.transcriptor
            self.transcriptor = None
            self._model_loaded = False
            logger.info("ByteDance 钢琴模型已卸载")

    def transcribe(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[List[NoteEvent], List[PedalEvent]]:
        """
        转写钢琴音频

        参数:
            audio_path: 音频文件路径
            progress_callback: 可选的进度回调

        返回:
            (音符事件列表, 踏板事件列表) 元组
        """
        logger.info(f"正在使用 ByteDance 模型转写: {audio_path}")

        self._check_cancelled()

        if progress_callback:
            progress_callback(0.0, "正在加载钢琴模型...")

        if not self.load_model():
            logger.warning("ByteDance 模型不可用，返回空结果")
            return [], []

        self._check_cancelled()

        if progress_callback:
            progress_callback(0.1, "正在加载音频...")

        try:
            import librosa

            # 加载音频（模型期望 32000Hz 采样率）
            audio, sr = librosa.load(audio_path, sr=32000, mono=True)
            duration = len(audio) / sr
            logger.info(f"音频时长: {duration:.2f}秒")

            self._check_cancelled()

            if progress_callback:
                progress_callback(0.2, "正在进行钢琴转写...")

            # 运行转写
            # 注意：piano_transcription_inference 返回的是包含音符和踏板的字典
            transcribed_dict = self.transcriptor.transcribe(
                audio=audio,
                midi_path=None  # 不直接保存 MIDI
            )

            self._check_cancelled()

            if progress_callback:
                progress_callback(0.8, "正在处理音符和踏板...")

            # 提取音符
            notes = self._extract_notes(transcribed_dict)

            # 提取踏板事件
            pedals = self._extract_pedals(transcribed_dict)

            self._check_cancelled()

            if progress_callback:
                progress_callback(1.0, f"检测到 {len(notes)} 个音符, {len(pedals)} 个踏板事件")

            logger.info(f"转写完成: {len(notes)} 个音符, {len(pedals)} 个踏板事件")

            return notes, pedals

        except ImportError as e:
            logger.error(f"缺少依赖: {e}")
            raise ImportError("钢琴转写需要 librosa 和 piano_transcription_inference") from e

    def transcribe_to_notes_only(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[NoteEvent]:
        """
        仅转写为音符事件（不包含踏板）

        用于与 AudioTranscriber 接口兼容

        参数:
            audio_path: 音频文件路径
            progress_callback: 可选的进度回调

        返回:
            音符事件列表
        """
        notes, _ = self.transcribe(audio_path, progress_callback)
        return notes

    def _extract_notes(self, transcribed_dict: dict) -> List[NoteEvent]:
        """
        从转写结果中提取音符事件

        参数:
            transcribed_dict: 模型返回的转写字典

        返回:
            NoteEvent 列表
        """
        notes = []

        # 检查返回格式
        # piano_transcription_inference 返回的格式可能是:
        # - 直接包含 'est_note_events' 键
        # - 或者是 MIDI 对象需要解析
        est_notes = transcribed_dict.get('est_note_events', [])

        if not est_notes:
            # 尝试从其他可能的键获取
            est_notes = transcribed_dict.get('notes', [])

        for note_data in est_notes:
            # 格式: (onset_time, offset_time, pitch, velocity)
            if len(note_data) >= 4:
                onset, offset, pitch, velocity = note_data[:4]
            elif len(note_data) == 3:
                onset, offset, pitch = note_data
                velocity = 80  # 默认力度
            else:
                continue

            # 确保音高在有效范围内
            midi_pitch = int(round(pitch))
            if midi_pitch < 21 or midi_pitch > 108:  # 钢琴音域 A0-C8
                continue

            # 确保力度在有效范围内
            midi_velocity = int(np.clip(velocity * 127 if velocity <= 1 else velocity, 1, 127))

            notes.append(NoteEvent(
                pitch=midi_pitch,
                start_time=float(onset),
                end_time=float(offset),
                velocity=midi_velocity
            ))

        # 按开始时间排序
        notes.sort(key=lambda n: n.start_time)

        logger.debug(f"提取了 {len(notes)} 个音符")
        if notes:
            pitches = [n.pitch for n in notes]
            logger.debug(f"音高范围: MIDI {min(pitches)} - {max(pitches)}")

        return notes

    def _extract_pedals(self, transcribed_dict: dict) -> List[PedalEvent]:
        """
        从转写结果中提取踏板事件

        参数:
            transcribed_dict: 模型返回的转写字典

        返回:
            PedalEvent 列表
        """
        pedals = []

        # 提取延音踏板事件
        sustain_events = transcribed_dict.get('est_pedal_events', [])
        for pedal_data in sustain_events:
            if len(pedal_data) >= 2:
                onset, offset = pedal_data[:2]
                pedals.append(PedalEvent(
                    start_time=float(onset),
                    end_time=float(offset),
                    pedal_type="sustain"
                ))

        # 提取柔音踏板事件（如果有）
        soft_events = transcribed_dict.get('est_soft_pedal_events', [])
        for pedal_data in soft_events:
            if len(pedal_data) >= 2:
                onset, offset = pedal_data[:2]
                pedals.append(PedalEvent(
                    start_time=float(onset),
                    end_time=float(offset),
                    pedal_type="soft"
                ))

        # 按开始时间排序
        pedals.sort(key=lambda p: p.start_time)

        logger.debug(f"提取了 {len(pedals)} 个踏板事件")

        return pedals

    def transcribe_with_complexity(
        self,
        audio_path: str,
        track_count: int,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[List[NoteEvent], List[PedalEvent]]:
        """
        根据轨道数量调整转写复杂度

        与 AudioTranscriber.transcribe_with_complexity 类似，
        但 ByteDance 模型本身已经很精确，主要通过后处理调整复杂度

        参数:
            audio_path: 音频文件路径
            track_count: 目标轨道数量
            progress_callback: 可选的进度回调

        返回:
            (音符事件列表, 踏板事件列表) 元组
        """
        logger.info(f"转写（{track_count}轨模式）: {audio_path}")

        notes, pedals = self.transcribe(audio_path, progress_callback)

        # 根据轨道数量进行后处理过滤
        # 轨道数越少，过滤越严格
        if track_count <= 2:
            # 简化模式：过滤非常短的音符和非常弱的音符
            min_duration = 0.08  # 80ms
            min_velocity = 40

            original_count = len(notes)
            notes = [
                n for n in notes
                if n.duration >= min_duration and n.velocity >= min_velocity
            ]
            logger.info(f"简化过滤: {original_count} -> {len(notes)} 个音符")

        return notes, pedals
