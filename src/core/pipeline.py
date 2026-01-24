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
    InstrumentType, TrackConfig, TrackLayout, ProcessingMode, PedalEvent
)
from src.core.separator import SourceSeparator
from src.core.transcriber import AudioTranscriber
from src.core.piano_transcriber import PianoTranscriberPro
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
        self.piano_transcriber = PianoTranscriberPro(config)
        self.lyrics_recognizer = LyricsRecognizer(config)
        self.beat_detector = BeatDetector(config)
        self.midi_generator = MidiGenerator(config)
        self.instrument_classifier = InstrumentClassifier(config)

        self._cancelled = False
        self._progress_callback: Optional[Callable[[ProcessingProgress], None]] = None

        # 当前处理的踏板事件（钢琴模式使用）
        self._current_pedals: List[PedalEvent] = []

        # 设置子模块的取消检查回调
        self.separator.set_cancel_check(lambda: self._cancelled)
        self.transcriber.set_cancel_check(lambda: self._cancelled)
        self.piano_transcriber.set_cancel_check(lambda: self._cancelled)

    def set_progress_callback(
        self,
        callback: Callable[[ProcessingProgress], None]
    ) -> None:
        """设置进度回调函数"""
        self._progress_callback = callback

    def cancel(self) -> None:
        """取消正在进行的处理"""
        self._cancelled = True
        # 同时取消子模块
        self.separator.cancel()
        self.transcriber.cancel()
        self.piano_transcriber.cancel()
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
        # 重置子模块的取消标志
        self.separator.reset_cancel()
        self.transcriber.reset_cancel()
        self.piano_transcriber.reset_cancel()
        self._progress_callback = progress_callback
        start_time = time.time()

        input_name = Path(audio_path).stem
        stems_dir = os.path.join(output_dir, "stems")

        logger.info(f"开始流水线处理: {audio_path}")
        logger.debug(f"输出目录: {output_dir}")
        logger.debug(f"分离轨道目录: {stems_dir}")

        # 处理轨道布局
        # 如果 track_layout 为 None 或者轨道列表为空，需要自动检测
        need_auto_detect = (
            track_layout is None or
            len(track_layout.tracks) == 0  # 任何模式下空轨道都需要自动检测
        )

        if need_auto_detect:
            mode = track_layout.mode if track_layout else ProcessingMode(self.config.processing_mode)

            if mode == ProcessingMode.PIANO:
                # 钢琴模式：自动检测最佳轨道数
                self._report_progress(
                    ProcessingStage.PREPROCESSING,
                    0.0, 0.02,
                    "正在分析最佳轨道数..."
                )

                auto_track_count = self.instrument_classifier.analyze_piano_track_count(
                    audio_path,
                    progress_callback=lambda p, m: self._report_progress(
                        ProcessingStage.PREPROCESSING,
                        p, 0.02 + p * 0.03,
                        m
                    )
                )
                logger.info(f"自动检测钢琴轨道数: {auto_track_count}")
                track_layout = TrackLayout.default_piano(auto_track_count)
                logger.debug(f"创建钢琴轨道布局: {len(track_layout.tracks)} 个轨道")
            else:
                # 智能模式：保持模式标记，轨道稍后通过分离和识别创建
                track_layout = TrackLayout(mode=ProcessingMode.SMART, tracks=[])

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

            # 如果用户要求嵌入歌词，需要单独提取人声
            if self.config.embed_lyrics:
                vocals_path = self._extract_vocals_for_lyrics(audio_path, stems_dir)
                if vocals_path:
                    stem_paths["vocals"] = vocals_path
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

            # 将分离结果映射到轨道ID
            # stem_paths 格式: {"drums": "path", "vocals": "path", ...}
            # 需要转换为: {"vocals_1": "path", "drums_1": "path", ...}
            mapped_stem_paths = {}
            for track_config in track_layout.get_enabled_tracks():
                # 根据乐器类型找到对应的分离轨道
                stem_name = self.separator.get_stem_for_instrument(track_config.instrument)
                if stem_name in stem_paths:
                    mapped_stem_paths[track_config.id] = stem_paths[stem_name]
                    logger.info(f"映射轨道: {track_config.id} -> {stem_name}")
                else:
                    # 使用 other 或原始音频
                    mapped_stem_paths[track_config.id] = stem_paths.get("other", audio_path)
                    logger.info(f"映射轨道: {track_config.id} -> other (fallback)")

            # 保留原始 stem_paths 用于歌词识别（需要 "vocals" 键）
            # 同时更新为映射后的版本用于转写
            original_stem_paths = stem_paths.copy()
            stem_paths = mapped_stem_paths
            # 将原始 vocals 路径保留，用于后续歌词识别
            if "vocals" in original_stem_paths:
                stem_paths["vocals"] = original_stem_paths["vocals"]
                logger.info(f"保留人声路径用于歌词识别: {stem_paths['vocals']}")
            else:
                logger.warning("未找到人声轨道，将跳过歌词识别")

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

        if track_layout.mode == ProcessingMode.PIANO:
            # 钢琴模式：优先使用 ByteDance 模型，否则回退到 Basic Pitch
            self._report_progress(
                ProcessingStage.TRANSCRIPTION,
                0.0, 0.45,
                "正在转写钢琴..."
            )

            # 获取轨道数量用于复杂度控制
            num_tracks = len(enabled_tracks)

            # 重置踏板事件
            self._current_pedals = []

            # 尝试使用 ByteDance 钢琴模型
            if self.piano_transcriber.is_available():
                logger.info("使用 ByteDance Piano Transcription 模型")
                self._report_progress(
                    ProcessingStage.TRANSCRIPTION,
                    0.05, 0.46,
                    "使用专业钢琴模型转写..."
                )

                all_notes, pedals = self.piano_transcriber.transcribe_with_complexity(
                    audio_path,
                    track_count=num_tracks,
                    progress_callback=lambda p, m: self._report_progress(
                        ProcessingStage.TRANSCRIPTION,
                        p, 0.45 + p * 0.25,
                        m
                    )
                )

                # 保存踏板事件供 MIDI 生成使用
                self._current_pedals = pedals
                logger.info(f"检测到 {len(pedals)} 个踏板事件")
            else:
                # 回退到 Basic Pitch
                logger.info("ByteDance 模型不可用，使用 Basic Pitch")
                self._report_progress(
                    ProcessingStage.TRANSCRIPTION,
                    0.05, 0.46,
                    "使用 Basic Pitch 转写..."
                )

                all_notes = self.transcriber.transcribe_with_complexity(
                    audio_path,
                    InstrumentType.PIANO,
                    track_count=num_tracks,
                    progress_callback=lambda p, m: self._report_progress(
                        ProcessingStage.TRANSCRIPTION,
                        p, 0.45 + p * 0.25,
                        m
                    )
                )

            logger.info(f"转写完成：共 {len(all_notes)} 个音符（{num_tracks}轨模式）")

            # 智能分割音符到多个轨道
            self._report_progress(
                ProcessingStage.TRANSCRIPTION,
                0.9, 0.70,
                "正在分割音符到多个轨道..."
            )

            tracks_notes = self._split_notes_to_tracks(all_notes, enabled_tracks)

            for track_config in enabled_tracks:
                note_count = len(tracks_notes.get(track_config.id, []))
                logger.info(f"{track_config.name}: {note_count} 个音符")
        else:
            # 智能模式：为每个轨道分别转写
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

        # 阶段 5: 歌词识别
        # 歌词识别条件简化：只需要有人声音频且用户启用了歌词嵌入
        # 不再要求轨道布局中必须有人声轨道，这样用户可以生成纯乐器MIDI同时嵌入歌词（适用于卡拉OK场景）
        lyrics: List[LyricEvent] = []

        vocals_stem = stem_paths.get("vocals")

        if vocals_stem and self.config.embed_lyrics:
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

        # 确定轨道数量（用于钢琴模式的复杂度控制）
        piano_track_count = None
        if track_layout.mode == ProcessingMode.PIANO:
            piano_track_count = len(enabled_tracks)

        self.midi_generator.generate_v2(
            track_layout=track_layout,
            tracks_notes=tracks_notes,
            tempo=beat_info.bpm,
            output_path=midi_path,
            lyrics=lyrics,
            embed_lyrics=self.config.embed_lyrics,
            apply_post_processing=True,
            track_count=piano_track_count,
            pedals=self._current_pedals if track_layout.mode == ProcessingMode.PIANO else None
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

    def _extract_vocals_for_lyrics(
        self,
        audio_path: str,
        output_dir: str
    ) -> Optional[str]:
        """
        为歌词识别单独提取人声轨道（用于钢琴模式）

        参数:
            audio_path: 原始音频路径
            output_dir: 输出目录

        返回:
            人声音频文件路径，如果提取失败返回 None
        """
        try:
            self._report_progress(
                ProcessingStage.SEPARATION,
                0.5, 0.12,
                "正在提取人声用于歌词识别..."
            )

            # 使用默认4轨模型分离
            stem_paths = self.separator.separate(
                audio_path,
                output_dir,
                lambda p, m: self._report_progress(
                    ProcessingStage.SEPARATION,
                    0.5 + p * 0.3, 0.12 + p * 0.03,
                    m
                )
            )

            vocals_path = stem_paths.get("vocals")
            if vocals_path:
                logger.info(f"已提取人声用于歌词识别: {vocals_path}")
                return vocals_path
            else:
                logger.warning("未能提取人声轨道")
                return None

        except Exception as e:
            logger.warning(f"提取人声失败: {e}")
            return None

    def _split_notes_to_tracks(
        self,
        notes: List[NoteEvent],
        tracks: List[TrackConfig]
    ) -> Dict[str, List[NoteEvent]]:
        """
        将音符智能分割到多个轨道

        核心改进：
        1. 基于音高范围分割（而非音符数量）
        2. 和弦检测：同时发声的音符保持在一起
        3. 智能分割点：使用固定的音高边界

        分割策略：
        - 2轨：中央C (MIDI 60) 附近分割
        - 3轨：C3 (48) 和 C5 (72) 分割
        - 4轨：G2 (43)、C4 (60)、F5 (77) 分割

        参数:
            notes: 所有音符事件
            tracks: 轨道配置列表

        返回:
            轨道ID到音符列表的字典
        """
        if not notes or not tracks:
            return {t.id: [] for t in tracks}

        num_tracks = len(tracks)

        if num_tracks == 1:
            # 只有一个轨道，直接返回所有音符
            return {tracks[0].id: notes}

        # 定义分割点（MIDI音高）
        split_points = self._get_split_points(num_tracks, notes)

        logger.info(f"音高分割点: {split_points}")

        # 首先检测和弦（同时发声的音符组）
        chord_groups = self._detect_chords(notes)
        logger.info(f"检测到 {len(chord_groups)} 个和弦组")

        # 分配音符到轨道
        result = {t.id: [] for t in tracks}

        for chord in chord_groups:
            if len(chord) == 1:
                # 单个音符，直接按音高分配
                note = chord[0]
                track_idx = self._get_track_index_for_pitch(note.pitch, split_points)
                result[tracks[track_idx].id].append(note)
            else:
                # 和弦：根据和弦的中心音高决定主轨道
                # 但如果和弦跨越分割点，则分配到各自的轨道
                chord_pitches = [n.pitch for n in chord]
                min_pitch = min(chord_pitches)
                max_pitch = max(chord_pitches)

                # 检查和弦是否跨越分割点
                spans_multiple = False
                for sp in split_points:
                    if min_pitch < sp <= max_pitch:
                        spans_multiple = True
                        break

                if spans_multiple:
                    # 和弦跨越分割点，按音高分配到各轨道
                    for note in chord:
                        track_idx = self._get_track_index_for_pitch(note.pitch, split_points)
                        result[tracks[track_idx].id].append(note)
                else:
                    # 和弦在同一区域，保持在一起
                    center_pitch = (min_pitch + max_pitch) // 2
                    track_idx = self._get_track_index_for_pitch(center_pitch, split_points)
                    for note in chord:
                        result[tracks[track_idx].id].append(note)

        # 记录分配结果
        for track in tracks:
            logger.info(f"轨道 {track.id}: {len(result[track.id])} 个音符")

        return result

    def _get_split_points(
        self,
        num_tracks: int,
        notes: List[NoteEvent]
    ) -> List[int]:
        """
        获取音高分割点

        使用固定的音乐性分割点，基于钢琴音域

        参数:
            num_tracks: 轨道数
            notes: 音符列表（用于参考）

        返回:
            分割点列表（每个元素是MIDI音高值）
        """
        # 预定义的分割策略
        predefined_splits = {
            2: [60],           # 中央C，分为低音和高音
            3: [48, 72],       # C3 和 C5，分为低中高
            4: [43, 60, 77],   # G2、C4、F5
            5: [40, 52, 64, 76],  # 每个大约一个八度
            6: [36, 48, 60, 72, 84],  # 从C2到C6
        }

        if num_tracks in predefined_splits:
            return predefined_splits[num_tracks]

        # 对于其他数量，动态计算
        # 找出音高范围
        if notes:
            pitches = [n.pitch for n in notes]
            min_pitch = min(pitches)
            max_pitch = max(pitches)
        else:
            min_pitch = 21   # A0
            max_pitch = 108  # C8

        pitch_range = max_pitch - min_pitch
        step = pitch_range / num_tracks

        splits = []
        for i in range(1, num_tracks):
            split_pitch = int(min_pitch + i * step)
            splits.append(split_pitch)

        return splits

    def _detect_chords(
        self,
        notes: List[NoteEvent],
        time_tolerance: float = 0.030  # 30ms 视为同时开始
    ) -> List[List[NoteEvent]]:
        """
        检测和弦（同时发声的音符组）

        参数:
            notes: 音符列表
            time_tolerance: 时间容差（秒）

        返回:
            和弦组列表，每个元素是一组同时发声的音符
        """
        if not notes:
            return []

        # 按开始时间排序
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        chords = []
        current_chord = [sorted_notes[0]]

        for i in range(1, len(sorted_notes)):
            note = sorted_notes[i]
            # 检查是否与当前和弦的第一个音符同时开始
            if abs(note.start_time - current_chord[0].start_time) <= time_tolerance:
                current_chord.append(note)
            else:
                # 保存当前和弦，开始新的
                chords.append(current_chord)
                current_chord = [note]

        # 添加最后一个和弦
        if current_chord:
            chords.append(current_chord)

        return chords

    def _get_track_index_for_pitch(
        self,
        pitch: int,
        split_points: List[int]
    ) -> int:
        """
        根据音高确定应该分配到哪个轨道

        参数:
            pitch: MIDI音高
            split_points: 分割点列表

        返回:
            轨道索引
        """
        for i, sp in enumerate(split_points):
            if pitch < sp:
                return i
        return len(split_points)  # 最后一个轨道

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
