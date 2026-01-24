"""
主处理流水线 - 协调所有模块
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
    ProcessingStage, Track, TrackType, NoteEvent, LyricEvent, BeatInfo
)
from src.core.separator import SourceSeparator
from src.core.transcriber import AudioTranscriber
from src.core.lyrics_recognizer import LyricsRecognizer
from src.core.beat_detector import BeatDetector
from src.core.midi_generator import MidiGenerator
from src.utils.gpu_utils import clear_gpu_memory

logger = logging.getLogger(__name__)


class MusicToMidiPipeline:
    """
    音乐转MIDI主处理流水线

    协调:
    1. 音源分离 (Demucs)
    2. 节拍检测 (librosa)
    3. 音频转写 (Basic Pitch)
    4. 歌词识别 (Whisper)
    5. MIDI生成 (mido)
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
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> ProcessingResult:
        """
        通过完整流水线处理音频文件

        参数:
            audio_path: 输入音频文件路径
            output_dir: 输出目录
            progress_callback: 可选的进度回调

        返回:
            包含所有输出的 ProcessingResult
        """
        self._cancelled = False
        self._progress_callback = progress_callback
        start_time = time.time()

        input_name = Path(audio_path).stem
        stems_dir = os.path.join(output_dir, "stems")

        logger.info(f"开始流水线处理: {audio_path}")

        # 阶段 1: 预处理
        self._report_progress(
            ProcessingStage.PREPROCESSING,
            0.0, 0.05,
            "正在准备音频..."
        )

        # 阶段 2: 音源分离
        self._report_progress(
            ProcessingStage.SEPARATION,
            0.0, 0.10,
            "正在分离音源..."
        )

        if self._cancelled:
            raise InterruptedError("处理已取消")

        stem_paths = self.separator.separate(
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

        # 阶段 3: 节拍检测（与转写并行开始）
        self._report_progress(
            ProcessingStage.TRANSCRIPTION,
            0.0, 0.35,
            "正在检测节拍..."
        )

        if self._cancelled:
            raise InterruptedError("处理已取消")

        beat_info = self.beat_detector.detect(
            audio_path,
            lambda p, m: self._report_progress(
                ProcessingStage.TRANSCRIPTION,
                p * 0.2, 0.35 + p * 0.05,
                m
            )
        )

        # 阶段 4: 转写所有分离轨道
        tracks: Dict[TrackType, List[NoteEvent]] = {}

        stem_types = [
            (TrackType.DRUMS, "drums"),
            (TrackType.BASS, "bass"),
            (TrackType.VOCALS, "vocals"),
            (TrackType.OTHER, "other")
        ]

        for i, (track_type, stem_name) in enumerate(stem_types):
            if self._cancelled:
                raise InterruptedError("处理已取消")

            if stem_name in stem_paths:
                base_progress = 0.40 + (i * 0.10)

                self._report_progress(
                    ProcessingStage.TRANSCRIPTION,
                    i / 4, base_progress,
                    f"正在转写 {stem_name}..."
                )

                notes = self.transcriber.transcribe(
                    stem_paths[stem_name],
                    track_type,
                    lambda p, m: self._report_progress(
                        ProcessingStage.TRANSCRIPTION,
                        (i + p) / 4,
                        base_progress + p * 0.10,
                        m
                    )
                )

                tracks[track_type] = notes
                logger.info(f"{stem_name}: {len(notes)} 个音符")

        # 阶段 5: 歌词识别
        lyrics: List[LyricEvent] = []

        if "vocals" in stem_paths and self.config.embed_lyrics:
            if self._cancelled:
                raise InterruptedError("处理已取消")

            self._report_progress(
                ProcessingStage.LYRICS,
                0.0, 0.80,
                "正在识别歌词..."
            )

            try:
                lyrics = self.lyrics_recognizer.recognize(
                    stem_paths["vocals"],
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

        self.midi_generator.generate(
            tracks=tracks,
            lyrics=lyrics,
            tempo=beat_info.bpm,
            output_path=midi_path,
            embed_lyrics=self.config.embed_lyrics
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

        # 构建结果
        result_tracks = []
        for track_type, notes in tracks.items():
            stem_name = track_type.value
            result_tracks.append(Track(
                type=track_type,
                audio_path=stem_paths.get(stem_name, ""),
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

    async def process_async(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> ProcessingResult:
        """
        process方法的异步版本

        参数:
            audio_path: 输入音频文件路径
            output_dir: 输出目录
            progress_callback: 可选的进度回调

        返回:
            包含所有输出的 ProcessingResult
        """
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: self.process(audio_path, output_dir, progress_callback)
            )

        return result
