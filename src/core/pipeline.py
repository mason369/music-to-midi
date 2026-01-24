"""
主处理流水线 - 协调所有模块
支持钢琴模式和智能识别模式
"""
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional, Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor

from src.models.data_models import (
    Config, Project, ProcessingResult, ProcessingProgress,
    ProcessingStage, Track, TrackType, NoteEvent, LyricEvent, BeatInfo,
    InstrumentType, TrackConfig, TrackLayout, ProcessingMode
)
from src.core.separator import SourceSeparator
from src.core.transcriber import AudioTranscriber
from src.core.lyrics_recognizer import LyricsRecognizer
from src.core.beat_detector import BeatDetector
from src.core.midi_generator import MidiGenerator
from src.core.instrument_classifier import InstrumentClassifier
from src.utils.gpu_utils import clear_gpu_memory

logger = logging.getLogger(__name__)


class MusicToMidiPipeline:
    """
    音乐转MIDI主处理流水线

    支持两种模式:
    - 钢琴模式: 跳过分离，直接将音频转写为多个钢琴轨道
    - 智能模式: 自动检测乐器，分离并转写

    协调:
    1. 音源分离 (Demucs)
    2. 乐器识别 (Demucs 6s + PANNs)
    3. 节拍检测 (librosa)
    4. 音频转写 (Basic Pitch)
    5. 歌词识别 (Whisper)
    6. MIDI生成 (mido)
    """

    def __init__(self, config: Config):
        """
        初始化流水线

        参数:
            config: 应用配置
        """
        self.config = config
        self.separator = SourceSeparator(config)
        self.transcriber = AudioTranscriber(config)
        self.lyrics_recognizer = LyricsRecognizer(config)
        self.beat_detector = BeatDetector(config)
        self.midi_generator = MidiGenerator(config)
        self.instrument_classifier = InstrumentClassifier(config)

        self._cancelled = False
        self._progress_callback: Optional[Callable[[ProcessingProgress], None]] = None

    def set_progress_callback(
        self,
        callback: Callable[[ProcessingProgress], None]
    ) -> None:
        """设置进度回调函数"""
        self._progress_callback = callback

    def cancel(self) -> None:
        """取消正在进行的处理"""
        self._cancelled = True
        logger.info("处理已取消")

    def _report_progress(
        self,
        stage: ProcessingStage,
        stage_progress: float,
        overall_progress: float,
        message: str
    ) -> None:
        """向回调报告进度"""
        if self._progress_callback:
            progress = ProcessingProgress(
                stage=stage,
                stage_progress=stage_progress,
                overall_progress=overall_progress,
                message=message
            )
            self._progress_callback(progress)

    def process(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        track_layout: Optional[TrackLayout] = None
    ) -> ProcessingResult:
        """
        通过完整流水线处理音频文件

        参数:
            audio_path: 输入音频文件路径
            output_dir: 输出目录
            progress_callback: 可选的进度回调
            track_layout: 可选的轨道布局（默认根据配置创建）

        返回:
            包含所有输出的 ProcessingResult
        """
        self._cancelled = False
        self._progress_callback = progress_callback
        start_time = time.time()

        input_name = Path(audio_path).stem
        stems_dir = os.path.join(output_dir, "stems")

        logger.info(f"开始流水线处理: {audio_path}")

        # 创建默认轨道布局
        if track_layout is None:
            mode = ProcessingMode(self.config.processing_mode)
            if mode == ProcessingMode.PIANO:
                track_layout = TrackLayout.default_piano(self.config.piano_track_count)
            else:
                # 智能模式：需要先进行分离和识别
                track_layout = None  # 稍后创建

        logger.info(f"处理模式: {track_layout.mode.value if track_layout else 'smart'}")

        # 阶段 1: 预处理
        self._report_progress(
            ProcessingStage.PREPROCESSING,
            0.0, 0.05,
            "正在准备音频..."
        )

        # 阶段 2: 音源分离
        if self._cancelled:
            raise InterruptedError("处理已取消")

        # 决定是否需要分离
        if track_layout and track_layout.mode == ProcessingMode.PIANO:
            # 钢琴模式：跳过分离
            self._report_progress(
                ProcessingStage.SEPARATION,
                1.0, 0.15,
                "钢琴模式：跳过音源分离"
            )
            stem_paths = self.separator.separate_v2(
                audio_path, stems_dir, track_layout
            )
        else:
            # 智能模式：进行6轨分离
            self._report_progress(
                ProcessingStage.SEPARATION,
                0.0, 0.10,
                "正在分离音源..."
            )

            stem_paths = self.separator.separate_6s(
                audio_path,
                stems_dir,
                lambda p, m: self._report_progress(
                    ProcessingStage.SEPARATION,
                    p, 0.10 + p * 0.25,
                    m
                )
            )

            # 释放分离器内存
            self.separator.unload_model()
            clear_gpu_memory()

            # 进行乐器识别并创建轨道布局
            self._report_progress(
                ProcessingStage.SEPARATION,
                0.9, 0.35,
                "正在识别乐器..."
            )

            track_layout = self.instrument_classifier.classify_audio(
                audio_path,
                stem_paths,
                lambda p, m: self._report_progress(
                    ProcessingStage.SEPARATION,
                    0.9 + p * 0.1, 0.35 + p * 0.05,
                    m
                )
            )

        # 阶段 3: 节拍检测
        self._report_progress(
            ProcessingStage.TRANSCRIPTION,
            0.0, 0.40,
            "正在检测节拍..."
        )

        if self._cancelled:
            raise InterruptedError("处理已取消")

        beat_info = self.beat_detector.detect(
            audio_path,
            lambda p, m: self._report_progress(
                ProcessingStage.TRANSCRIPTION,
                p * 0.2, 0.40 + p * 0.05,
                m
            )
        )

        # 阶段 4: 转写所有轨道
        tracks_notes: Dict[str, List[NoteEvent]] = {}
        enabled_tracks = track_layout.get_enabled_tracks()
        total_tracks = len(enabled_tracks)

        for i, track_config in enumerate(enabled_tracks):
            if self._cancelled:
                raise InterruptedError("处理已取消")

            base_progress = 0.45 + (i * 0.30 / max(total_tracks, 1))

            self._report_progress(
                ProcessingStage.TRANSCRIPTION,
                (i + 0.5) / total_tracks, base_progress,
                f"正在转写 {track_config.name}..."
            )

            # 获取该轨道对应的音频路径
            audio_for_track = stem_paths.get(track_config.id, audio_path)

            notes = self.transcriber.transcribe(
                audio_for_track,
                track_config.instrument,
                lambda p, m: self._report_progress(
                    ProcessingStage.TRANSCRIPTION,
                    (i + p) / total_tracks,
                    base_progress + p * (0.30 / max(total_tracks, 1)),
                    m
                )
            )

            tracks_notes[track_config.id] = notes
            logger.info(f"{track_config.name}: {len(notes)} 个音符")

        # 阶段 5: 歌词识别（仅智能模式且有人声轨道）
        lyrics: List[LyricEvent] = []

        vocals_stem = stem_paths.get("vocals")
        has_vocals_track = any(
            t.instrument == InstrumentType.VOCALS
            for t in enabled_tracks
        )

        if vocals_stem and has_vocals_track and self.config.embed_lyrics:
            if self._cancelled:
                raise InterruptedError("处理已取消")

            self._report_progress(
                ProcessingStage.LYRICS,
                0.0, 0.80,
                "正在识别歌词..."
            )

            try:
                lyrics = self.lyrics_recognizer.recognize(
                    vocals_stem,
                    lambda p, m: self._report_progress(
                        ProcessingStage.LYRICS,
                        p, 0.80 + p * 0.10,
                        m
                    )
                )
            except Exception as e:
                logger.warning(f"歌词识别失败: {e}")

            # 释放识别器内存
            self.lyrics_recognizer.unload_model()
            clear_gpu_memory()

        # 阶段 6: MIDI合成
        if self._cancelled:
            raise InterruptedError("处理已取消")

        self._report_progress(
            ProcessingStage.SYNTHESIS,
            0.0, 0.90,
            "正在生成MIDI..."
        )

        midi_path = os.path.join(output_dir, f"{input_name}.mid")

        self.midi_generator.generate_v2(
            track_layout=track_layout,
            tracks_notes=tracks_notes,
            tempo=beat_info.bpm,
            output_path=midi_path,
            lyrics=lyrics,
            embed_lyrics=self.config.embed_lyrics,
            apply_post_processing=True
        )

        # 如果需要则导出LRC
        lrc_path = None
        if lyrics and self.config.export_lrc:
            lrc_path = os.path.join(output_dir, f"{input_name}.lrc")
            self.midi_generator.export_lrc(
                lyrics=lyrics,
                output_path=lrc_path,
                title=input_name
            )

        # 完成
        processing_time = time.time() - start_time

        self._report_progress(
            ProcessingStage.COMPLETE,
            1.0, 1.0,
            f"完成! ({processing_time:.1f}秒)"
        )

        # 构建结果（使用旧格式保持兼容）
        result_tracks = []
        for track_config in enabled_tracks:
            notes = tracks_notes.get(track_config.id, [])
            # 尝试将 InstrumentType 映射回 TrackType（向后兼容）
            track_type = self._instrument_to_track_type(track_config.instrument)
            result_tracks.append(Track(
                type=track_type,
                audio_path=stem_paths.get(track_config.id, audio_path),
                notes=notes
            ))

        result = ProcessingResult(
            midi_path=midi_path,
            lrc_path=lrc_path,
            tracks=result_tracks,
            beat_info=beat_info,
            lyrics=lyrics,
            processing_time=processing_time
        )

        logger.info(f"流水线完成: {processing_time:.1f}秒")

        return result

    def _instrument_to_track_type(self, instrument: InstrumentType) -> TrackType:
        """将乐器类型映射到轨道类型（向后兼容）"""
        mapping = {
            InstrumentType.DRUMS: TrackType.DRUMS,
            InstrumentType.BASS: TrackType.BASS,
            InstrumentType.VOCALS: TrackType.VOCALS,
        }
        return mapping.get(instrument, TrackType.OTHER)

    async def process_async(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        track_layout: Optional[TrackLayout] = None
    ) -> ProcessingResult:
        """
        process方法的异步版本

        参数:
            audio_path: 输入音频文件路径
            output_dir: 输出目录
            progress_callback: 可选的进度回调
            track_layout: 可选的轨道布局

        返回:
            包含所有输出的 ProcessingResult
        """
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: self.process(audio_path, output_dir, progress_callback, track_layout)
            )

        return result
