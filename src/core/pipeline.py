"""
主处理流水线 - 协调所有模块
支持钢琴模式和智能识别模式
"""
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from src.models.data_models import (
    Config, Project, ProcessingResult, ProcessingProgress,
    ProcessingStage, Track, TrackType, NoteEvent, BeatInfo,
    InstrumentType, TrackConfig, TrackLayout, ProcessingMode, PedalEvent,
    TranscriptionQuality
)
from src.core.separator import SourceSeparator
from src.core.transcriber import AudioTranscriber
from src.core.piano_transcriber import PianoTranscriberPro
from src.core.yourmt3_transcriber import YourMT3Transcriber
from src.core.beat_detector import BeatDetector
from src.core.midi_generator import MidiGenerator
from src.core.instrument_classifier import InstrumentClassifier
from src.utils.gpu_utils import clear_gpu_memory

logger = logging.getLogger(__name__)


class MusicToMidiPipeline:
    """
    音乐转MIDI主处理流水线

    支持两种模式:
    - 钢琴模式 (PIANO): 跳过分离，直接将音频转写为多个钢琴轨道
    - 智能模式 (SMART): 使用YourMT3+多乐器转写，动态识别乐器

    协调:
    1. 音源分离 (Demucs)
    2. 节拍检测 (librosa)
    3. 音频转写 (YourMT3+ / Basic Pitch / ByteDance Piano)
    4. 鼓层次化分离 (可选)
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
        self.piano_transcriber = PianoTranscriberPro(config)
        self.yourmt3_transcriber = YourMT3Transcriber(config)
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
        self.yourmt3_transcriber.set_cancel_check(lambda: self._cancelled)

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
        self.yourmt3_transcriber.cancel()
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
        self.yourmt3_transcriber.reset_cancel()
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
                # 智能模式：使用多乐器转写，轨道稍后由转写结果创建
                track_layout = TrackLayout(mode=ProcessingMode.SMART, tracks=[])
                logger.info("使用智能识别模式：多乐器转写")

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
            # 钢琴模式：根据轨道数决定分离策略
            num_tracks = len(track_layout.get_enabled_tracks())

            if num_tracks > 1:
                # 多轨钢琴模式：需要分离
                self._report_progress(
                    ProcessingStage.SEPARATION,
                    0.0, 0.10,
                    f"钢琴模式：正在分离音源（{num_tracks}轨）..."
                )

                stem_paths = self.separator.separate_v2(
                    audio_path, stems_dir, track_layout,
                    lambda p, m: self._report_progress(
                        ProcessingStage.SEPARATION,
                        p, 0.10 + p * 0.25,
                        m
                    )
                )

                # 释放分离器内存
                self.separator.unload_model()
                clear_gpu_memory()
            else:
                # 单轨钢琴模式：不分离
                self._report_progress(
                    ProcessingStage.SEPARATION,
                    1.0, 0.15,
                    "钢琴模式（1轨）：使用原始音频"
                )
                stem_paths = {"original": audio_path}

        elif track_layout and track_layout.mode == ProcessingMode.SMART:
            # 智能识别模式：跳过分离，使用多乐器转写
            self._report_progress(
                ProcessingStage.SEPARATION,
                1.0, 0.10,
                "智能识别模式：跳过音源分离"
            )
            stem_paths = {"original": audio_path}

        else:
            # 动态智能模式：先分析音频，根据检测到的乐器决定处理策略
            self._report_progress(
                ProcessingStage.SEPARATION,
                0.0, 0.05,
                "正在分析音频内容..."
            )

            # 第一步：使用 PANNs 预先分析音频，检测所有乐器
            logger.info("开始 PANNs 乐器预检测...")
            preliminary_layout = self.instrument_classifier.classify_audio(
                audio_path,
                stem_paths=None,  # 第一次分析不需要分离轨道
                progress_callback=lambda p, m: self._report_progress(
                    ProcessingStage.SEPARATION,
                    p * 0.15, 0.05 + p * 0.10,
                    f"AI识别: {m}"
                )
            )

            detected_instruments = [track.instrument for track in preliminary_layout.get_enabled_tracks()]
            logger.info(f"检测到 {len(detected_instruments)} 种乐器: {[inst.value for inst.value in detected_instruments]}")

            # 第二步：根据检测结果决定处理策略
            if len(detected_instruments) == 0:
                # 没有检测到乐器，使用默认 6 轨分离
                logger.warning("未检测到乐器，使用默认 6 轨分离")
                use_6s_separation = True
            elif len(detected_instruments) <= 6 and all(
                inst in [InstrumentType.DRUMS, InstrumentType.BASS, InstrumentType.GUITAR,
                        InstrumentType.PIANO, InstrumentType.VOCALS, InstrumentType.STRINGS,
                        InstrumentType.BRASS, InstrumentType.WOODWIND, InstrumentType.SYNTH,
                        InstrumentType.ORGAN]
                for inst in detected_instruments
            ):
                # 检测到的乐器适合用 Demucs 分离
                use_6s_separation = True
                logger.info(f"使用 Demucs 6轨分离处理 {len(detected_instruments)} 种乐器")
            else:
                # 检测到太多乐器或特殊乐器，应该使用多乐器转写
                use_6s_separation = False
                logger.info(f"检测到 {len(detected_instruments)} 种乐器，将尝试多乐器转写")

            # 第三步：执行相应的处理
            if use_6s_separation:
                # 使用 Demucs 6轨分离
                self._report_progress(
                    ProcessingStage.SEPARATION,
                    0.0, 0.15,
                    f"正在分离音源 (检测到{len(detected_instruments)}种乐器)..."
                )

                stem_paths = self.separator.separate_6s(
                    audio_path,
                    stems_dir,
                    lambda p, m: self._report_progress(
                        ProcessingStage.SEPARATION,
                        p, 0.15 + p * 0.25,
                        m
                    )
                )

                # 释放分离器内存
                self.separator.unload_model()
                clear_gpu_memory()

                # 使用分离结果细化识别
                self._report_progress(
                    ProcessingStage.SEPARATION,
                    0.9, 0.40,
                    "正在细化乐器识别..."
                )

                track_layout = self.instrument_classifier.classify_audio(
                    audio_path,
                    stem_paths,  # 传入分离结果进行细化
                    lambda p, m: self._report_progress(
                        ProcessingStage.SEPARATION,
                        0.9 + p * 0.1, 0.40 + p * 0.05,
                        m
                    )
                )
            else:
                # 不进行分离，直接使用预分析结果
                # 稍后会尝试 YourMT3+ 多乐器转写
                track_layout = preliminary_layout
                stem_paths = {"original": audio_path}

                self._report_progress(
                    ProcessingStage.SEPARATION,
                    1.0, 0.20,
                    f"跳过分离，将使用多乐器转写 ({len(detected_instruments)}种乐器)"
                )

                logger.info("设置为多乐器转写模式")
                # 将模式改为 SMART 以触发多乐器转写
                track_layout = TrackLayout(
                    mode=ProcessingMode.SMART,
                    tracks=preliminary_layout.tracks
                )

            # 将分离结果映射到轨道ID（仅在使用分离的情况下）
            if use_6s_separation and track_layout.mode != ProcessingMode.SMART:
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
            else:
                # 多乐器转写模式，不需要映射
                logger.info("多乐器转写模式，跳过轨道映射")
                # 保留 stem_paths = {"original": audio_path} 或已有的 vocals

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
            # 钢琴模式：根据分离轨道数量决定处理策略
            num_tracks = len(enabled_tracks)

            # 重置踏板事件
            self._current_pedals = []
            all_pedals = []

            if num_tracks == 1:
                # 单轨模式：直接转写原始音频
                self._report_progress(
                    ProcessingStage.TRANSCRIPTION,
                    0.0, 0.45,
                    "正在转写钢琴..."
                )

                if self.piano_transcriber.is_available():
                    logger.info("使用 ByteDance Piano Transcription 模型")
                    notes, pedals = self.piano_transcriber.transcribe(
                        audio_path,
                        lambda p, m: self._report_progress(
                            ProcessingStage.TRANSCRIPTION,
                            p, 0.45 + p * 0.25,
                            m
                        )
                    )
                    all_pedals.extend(pedals)
                else:
                    logger.info("ByteDance 模型不可用，使用 Basic Pitch")
                    notes = self.transcriber.transcribe(
                        audio_path,
                        InstrumentType.PIANO,
                        lambda p, m: self._report_progress(
                            ProcessingStage.TRANSCRIPTION,
                            p, 0.45 + p * 0.25,
                            m
                        )
                    )

                track_id = enabled_tracks[0].id
                tracks_notes[track_id] = notes
                logger.info(f"[诊断] 单轨模式转写完成: {len(notes)} 个音符")

            else:
                # 多轨模式：为每个分离轨道进行钢琴转写
                self._report_progress(
                    ProcessingStage.TRANSCRIPTION,
                    0.0, 0.35,
                    f"正在转写钢琴（{num_tracks}轨）..."
                )

                for i, track_config in enumerate(enabled_tracks):
                    if self._cancelled:
                        raise InterruptedError("处理已取消")

                    # 获取该轨道对应的分离音频
                    source_key = getattr(track_config, 'source', 'original')
                    audio_for_track = stem_paths.get(source_key, audio_path)

                    base_progress = 0.35 + (i * 0.35 / max(num_tracks, 1))

                    self._report_progress(
                        ProcessingStage.TRANSCRIPTION,
                        (i + 0.5) / num_tracks, base_progress,
                        f"正在转写 {track_config.name}..."
                    )

                    logger.info(f"[诊断] 转写轨道 {track_config.name}, 来源: {source_key}, 音频: {audio_for_track}")

                    # 使用钢琴转写器
                    if self.piano_transcriber.is_available():
                        notes, pedals = self.piano_transcriber.transcribe(
                            audio_for_track,
                            lambda p, m: self._report_progress(
                                ProcessingStage.TRANSCRIPTION,
                                (i + p) / num_tracks,
                                base_progress + p * (0.35 / max(num_tracks, 1)),
                                m
                            )
                        )
                        all_pedals.extend(pedals)
                    else:
                        # 回退到 Basic Pitch
                        notes = self.transcriber.transcribe(
                            audio_for_track,
                            InstrumentType.PIANO,
                            lambda p, m: self._report_progress(
                                ProcessingStage.TRANSCRIPTION,
                                (i + p) / num_tracks,
                                base_progress + p * (0.35 / max(num_tracks, 1)),
                                m
                            )
                        )

                    tracks_notes[track_config.id] = notes
                    logger.info(f"[诊断] {track_config.name}: {len(notes)} 个音符")

            # 合并踏板事件（去重）
            self._current_pedals = self._deduplicate_pedals(all_pedals)
            logger.info(f"检测到 {len(self._current_pedals)} 个踏板事件")

            # 汇总所有轨道音符数
            total_notes = sum(len(notes) for notes in tracks_notes.values())
            logger.info(f"[诊断] 钢琴模式转写完成：共 {total_notes} 个音符（{num_tracks}轨）")

            for track_config in enabled_tracks:
                note_count = len(tracks_notes.get(track_config.id, []))
                logger.info(f"{track_config.name}: {note_count} 个音符")

        elif track_layout.mode == ProcessingMode.SMART:
            # 智能识别模式：多乐器直接转写
            # 根据配置选择精确模式或标准模式
            use_precise = self.config.use_precise_instruments

            if use_precise and YourMT3Transcriber.is_available():
                # 使用极致精度模式：按 GM 程序号分组
                logger.info(f"使用极致精度模式 (质量: {self.config.transcription_quality})")

                instrument_notes_precise, drum_notes_precise = self._transcribe_precise(
                    audio_path,
                    lambda p, m: self._report_progress(
                        ProcessingStage.TRANSCRIPTION,
                        p, 0.45 + p * 0.30,
                        m
                    )
                )

                # 释放转写器内存
                self.yourmt3_transcriber.unload_model()
                clear_gpu_memory()

                if instrument_notes_precise or drum_notes_precise:
                    # 直接生成精确乐器 MIDI（跳过标准轨道布局）
                    self._report_progress(
                        ProcessingStage.SYNTHESIS,
                        0.0, 0.80,
                        "正在生成极致精度 MIDI..."
                    )

                    midi_path = os.path.join(output_dir, f"{input_name}.mid")

                    self.midi_generator.generate_from_precise_instruments_v2(
                        instrument_notes=instrument_notes_precise,
                        drum_notes=drum_notes_precise,
                        tempo=beat_info.bpm,
                        output_path=midi_path,
                        quality=self.config.transcription_quality
                    )

                    # 完成
                    processing_time = time.time() - start_time

                    self._report_progress(
                        ProcessingStage.COMPLETE,
                        1.0, 1.0,
                        f"完成! ({processing_time:.1f}秒)"
                    )

                    # 构建结果（精确模式）
                    result_tracks = []
                    # 将精确乐器转为轨道列表
                    from src.models.gm_instruments import get_instrument_name
                    for program, notes in instrument_notes_precise.items():
                        result_tracks.append(Track(
                            type=TrackType.OTHER,
                            audio_path=audio_path,
                            notes=notes
                        ))

                    result = ProcessingResult(
                        midi_path=midi_path,
                        tracks=result_tracks,
                        beat_info=beat_info,
                        processing_time=processing_time
                    )

                    logger.info(f"极致精度流水线完成: {processing_time:.1f}秒")
                    return result
                else:
                    logger.warning("极致精度转写无结果，回退到标准模式")

            # 标准模式或回退：使用 YourMT3+，回退到分离+Basic Pitch
            instrument_notes = self._transcribe_multi_instrument(
                audio_path,
                lambda p, m: self._report_progress(
                    ProcessingStage.TRANSCRIPTION,
                    p, 0.45 + p * 0.30,
                    m
                )
            )

            # 释放转写器内存
            self.yourmt3_transcriber.unload_model()
            clear_gpu_memory()

            if instrument_notes:
                # 从转写结果创建轨道布局
                self._report_progress(
                    ProcessingStage.TRANSCRIPTION,
                    0.9, 0.75,
                    "正在创建轨道布局..."
                )

                track_layout, tracks_notes = self._create_layout_from_yourmt3(instrument_notes)
                enabled_tracks = track_layout.get_enabled_tracks()

                # 如果启用了层次化鼓分离
                if self.config.separate_drum_voices and InstrumentType.DRUMS in instrument_notes:
                    self._report_progress(
                        ProcessingStage.TRANSCRIPTION,
                        0.92, 0.77,
                        "正在分离鼓声部..."
                    )
                    drum_notes = instrument_notes[InstrumentType.DRUMS]
                    drum_tracks = self._separate_drum_voices(drum_notes)
                    # 更新轨道布局
                    for drum_type, notes in drum_tracks.items():
                        track_id = f"{drum_type.value}_1"
                        tracks_notes[track_id] = notes
                        # 添加鼓轨道配置
                        track_layout.tracks.append(TrackConfig(
                            id=track_id,
                            instrument=drum_type,
                            name=drum_type.get_display_name(),
                            enabled=True,
                            midi_channel=9,
                            program=0
                        ))
                    # 移除原始鼓轨道
                    tracks_notes.pop("drums_1", None)
                    track_layout.tracks = [t for t in track_layout.tracks if t.id != "drums_1"]

                for inst_type, notes in instrument_notes.items():
                    logger.info(f"{inst_type.get_display_name()}: {len(notes)} 个音符")
            else:
                # 转写失败，使用分离+逐轨转写
                logger.warning("多乐器转写失败，回退到分离+逐轨转写")
                track_layout, tracks_notes, enabled_tracks = self._fallback_with_separation(
                    audio_path, stems_dir, beat_info
                )

        else:
            # 未知模式：使用分离+逐轨转写
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

        # 阶段 5: MIDI合成
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
            apply_post_processing=True,
            track_count=piano_track_count,
            pedals=self._current_pedals if track_layout.mode == ProcessingMode.PIANO else None
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
            tracks=result_tracks,
            beat_info=beat_info,
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

    def _create_layout_from_yourmt3(
        self,
        instrument_notes: Dict[InstrumentType, List[NoteEvent]]
    ) -> tuple:
        """
        从 YourMT3 转写结果创建轨道布局

        参数:
            instrument_notes: 乐器类型到音符列表的字典

        返回:
            (TrackLayout, tracks_notes字典) 元组
        """
        tracks = []
        tracks_notes = {}
        channel = 0

        for inst_type, notes in instrument_notes.items():
            if not notes:
                continue

            # 确定 MIDI 通道
            if inst_type == InstrumentType.DRUMS:
                midi_channel = 9
            else:
                midi_channel = channel
                channel += 1
                if channel == 9:
                    channel = 10  # 跳过鼓通道

            track_id = f"{inst_type.value}_1"
            tracks.append(TrackConfig(
                id=track_id,
                instrument=inst_type,
                name=inst_type.get_display_name(),
                enabled=True,
                midi_channel=midi_channel,
                program=inst_type.to_program_number()
            ))
            tracks_notes[track_id] = notes

        layout = TrackLayout(mode=ProcessingMode.SMART, tracks=tracks)
        return layout, tracks_notes

    def _transcribe_multi_instrument(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[InstrumentType, List[NoteEvent]]:
        """
        多乐器转写 - 使用 YourMT3-MoE（2025 SOTA）

        直接从混合音频识别多种乐器，无需先分离。

        参数:
            audio_path: 音频文件路径
            progress_callback: 进度回调

        返回:
            乐器类型到音符列表的字典
        """
        self._check_cancelled()

        # 使用 YourMT3-MoE (2025 AMT Challenge SOTA)
        if YourMT3Transcriber.is_available():
            if progress_callback:
                progress_callback(0.0, "使用 YourMT3-MoE 多乐器转写...")
            logger.info("使用 YourMT3-MoE 进行多乐器转写（2025 SOTA）")

            try:
                result = self.yourmt3_transcriber.transcribe_full_mix(
                    audio_path,
                    lambda p, m: progress_callback(p, m) if progress_callback else None
                )
                if result:
                    logger.info(f"YourMT3-MoE 转写成功: {len(result)} 种乐器")
                    return result
            except Exception as e:
                logger.warning(f"YourMT3-MoE 转写失败: {e}")

        # YourMT3 不可用，发出警告
        logger.warning("=" * 70)
        logger.warning("⚠ 多乐器转写引擎不可用")
        logger.warning("=" * 70)
        logger.warning("检测到音频可能包含多种乐器，但缺少多乐器转写模型。")
        logger.warning("当前系统只能输出最多 6 个 MIDI 轨道（Demucs 分离能力）。")
        logger.warning("")
        logger.warning("要支持 >6 种乐器的精确识别，请安装:")
        logger.warning("  • YourMT3+: bash install_yourmt3.sh")
        logger.warning("=" * 70)

        if progress_callback:
            progress_callback(1.0, "⚠ 多乐器引擎不可用，将回退到 6 轨模式")

        logger.warning("将回退到分离+逐轨转写（最多 6 轨）")
        return {}

    def _transcribe_precise(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[Dict[int, List[NoteEvent]], Dict[int, List[NoteEvent]]]:
        """
        极致精度转写 - 输出按 GM 程序号分组的音符

        使用 YourMT3-MoE（2025 SOTA）进行精确转写。
        根据 config.transcription_quality 选择质量模式。

        参数:
            audio_path: 音频文件路径
            progress_callback: 进度回调

        返回:
            (instrument_notes, drum_notes) 元组:
            - instrument_notes: GM 程序号(0-127)到音符列表的字典
            - drum_notes: 鼓音高(35-81)到音符列表的字典
        """
        self._check_cancelled()

        quality = self.config.transcription_quality

        # 使用 YourMT3-MoE (2025 SOTA, 128种精确乐器)
        if YourMT3Transcriber.is_available():
            if progress_callback:
                progress_callback(0.0, f"使用 YourMT3-MoE 极致精度转写 (质量: {quality})...")
            logger.info(f"使用 YourMT3-MoE 进行极致精度转写（128种GM乐器，质量: {quality}）")

            try:
                instrument_notes, drum_notes = self.yourmt3_transcriber.transcribe_precise(
                    audio_path,
                    quality=quality,
                    progress_callback=lambda p, m: progress_callback(p, m) if progress_callback else None
                )
                total_notes = sum(len(n) for n in instrument_notes.values()) + sum(len(n) for n in drum_notes.values())
                if total_notes > 0:
                    logger.info(f"YourMT3-MoE 极致精度转写成功: {len(instrument_notes)} 种乐器 + {len(drum_notes)} 种鼓")
                    return instrument_notes, drum_notes
            except Exception as e:
                logger.warning(f"YourMT3-MoE 极致精度转写失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # YourMT3 不可用，返回空结果
        if progress_callback:
            progress_callback(1.0, "极致精度转写引擎不可用")
        logger.warning("YourMT3+ 不可用，无法进行极致精度转写")
        return {}, {}

    def _check_cancelled(self) -> None:
        """检查是否已取消处理"""
        if self._cancelled:
            raise InterruptedError("处理已取消")

    def _deduplicate_pedals(self, pedals: List[PedalEvent]) -> List[PedalEvent]:
        """
        去除重复的踏板事件

        当从多个分离轨道合并踏板事件时，可能会有重复的踏板事件。
        此方法按时间和类型去重。

        参数:
            pedals: 踏板事件列表

        返回:
            去重后的踏板事件列表
        """
        if not pedals:
            return []

        # 按开始时间和类型排序
        pedals.sort(key=lambda p: (p.start_time, p.pedal_type))

        result = []
        for pedal in pedals:
            # 检查是否与最近的同类型踏板事件重叠
            is_duplicate = False
            for existing in result[-5:]:  # 只检查最近 5 个
                if (existing.pedal_type == pedal.pedal_type and
                    abs(existing.start_time - pedal.start_time) < 0.05):
                    is_duplicate = True
                    break

            if not is_duplicate:
                result.append(pedal)

        logger.debug(f"踏板去重: {len(pedals)} -> {len(result)}")
        return result

    def _fallback_with_separation(
        self,
        audio_path: str,
        stems_dir: str,
        beat_info: BeatInfo
    ) -> Tuple[TrackLayout, Dict[str, List[NoteEvent]], List[TrackConfig]]:
        """
        使用分离+逐轨转写作为最终备用方案

        当 YourMT3+ 完全不可用时，使用 Demucs 分离 + Basic Pitch 逐轨转写。

        参数:
            audio_path: 原始音频路径
            stems_dir: 分离轨道保存目录
            beat_info: 节拍信息

        返回:
            (TrackLayout, tracks_notes字典, enabled_tracks列表) 元组
        """
        logger.info("执行最终备用：分离+逐轨转写")

        # 执行6轨分离
        self._report_progress(
            ProcessingStage.SEPARATION,
            0.0, 0.60,
            "正在分离音源（备用模式）..."
        )

        try:
            stem_paths = self.separator.separate_6s(
                audio_path,
                stems_dir,
                lambda p, m: self._report_progress(
                    ProcessingStage.SEPARATION,
                    p, 0.60 + p * 0.10,
                    m
                )
            )
        except Exception as e:
            logger.warning(f"6轨分离失败: {e}，尝试4轨分离")
            stem_paths = self.separator.separate(
                audio_path,
                stems_dir,
                lambda p, m: self._report_progress(
                    ProcessingStage.SEPARATION,
                    p, 0.60 + p * 0.10,
                    m
                )
            )

        # 释放分离器内存
        self.separator.unload_model()
        clear_gpu_memory()

        # 进行乐器识别并创建轨道布局
        self._report_progress(
            ProcessingStage.SEPARATION,
            0.9, 0.70,
            "正在识别乐器..."
        )

        track_layout = self.instrument_classifier.classify_audio(
            audio_path,
            stem_paths,
            lambda p, m: self._report_progress(
                ProcessingStage.SEPARATION,
                0.9 + p * 0.1, 0.70 + p * 0.02,
                m
            )
        )

        # 转写各轨道
        tracks_notes: Dict[str, List[NoteEvent]] = {}
        enabled_tracks = track_layout.get_enabled_tracks()
        total_tracks = len(enabled_tracks)

        # stem名称到轨道ID的映射
        stem_track_map = {}
        for track_config in enabled_tracks:
            stem_name = self.separator.get_stem_for_instrument(track_config.instrument)
            if stem_name in stem_paths:
                stem_track_map[track_config.id] = stem_paths[stem_name]
            else:
                stem_track_map[track_config.id] = stem_paths.get("other", audio_path)

        for i, track_config in enumerate(enabled_tracks):
            if self._cancelled:
                raise InterruptedError("处理已取消")

            base_progress = 0.72 + (i * 0.18 / max(total_tracks, 1))

            self._report_progress(
                ProcessingStage.TRANSCRIPTION,
                (i + 0.5) / total_tracks, base_progress,
                f"正在转写 {track_config.name}..."
            )

            audio_for_track = stem_track_map.get(track_config.id, audio_path)

            notes = self.transcriber.transcribe(
                audio_for_track,
                track_config.instrument,
                lambda p, m: self._report_progress(
                    ProcessingStage.TRANSCRIPTION,
                    (i + p) / total_tracks,
                    base_progress + p * (0.18 / max(total_tracks, 1)),
                    m
                )
            )

            tracks_notes[track_config.id] = notes
            logger.info(f"备用转写 {track_config.name}: {len(notes)} 个音符")

        return track_layout, tracks_notes, enabled_tracks

    def _separate_drum_voices(
        self,
        drum_notes: List[NoteEvent]
    ) -> Dict[InstrumentType, List[NoteEvent]]:
        """
        将鼓音符按声部分离

        根据 MIDI 音高将鼓音符分类到不同的声部。

        参数:
            drum_notes: 鼓轨道的音符列表

        返回:
            鼓声部类型到音符列表的字典
        """
        # MIDI 鼓音高映射
        DRUM_MAPPING = {
            InstrumentType.KICK: {35, 36},           # 底鼓
            InstrumentType.SNARE: {38, 40, 37},      # 军鼓
            InstrumentType.HIHAT: {42, 44, 46},      # 踩镲
            InstrumentType.TOM: {41, 43, 45, 47, 48, 50},  # 嗵鼓
            InstrumentType.CYMBAL: {49, 51, 52, 55, 57},   # 镲片
            InstrumentType.RIDE: {51, 59, 53},       # 叮叮镲
        }

        result: Dict[InstrumentType, List[NoteEvent]] = {}

        for note in drum_notes:
            assigned = False
            for drum_type, pitches in DRUM_MAPPING.items():
                if note.pitch in pitches:
                    if drum_type not in result:
                        result[drum_type] = []
                    result[drum_type].append(note)
                    assigned = True
                    break

            # 未匹配的鼓音符放入 PERCUSSION
            if not assigned:
                if InstrumentType.PERCUSSION not in result:
                    result[InstrumentType.PERCUSSION] = []
                result[InstrumentType.PERCUSSION].append(note)

        # 按开始时间排序
        for notes in result.values():
            notes.sort(key=lambda n: n.start_time)

        return result

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
