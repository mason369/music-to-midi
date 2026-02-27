"""
主处理流水线 - 支持 YourMT3+ MoE 多乐器转写 和 人声分离+分别转写

处理模式：
1. SMART: YourMT3+ 直接对完整混音进行极致精度转写
2. VOCAL_SPLIT: Demucs 分离人声与伴奏，均使用 YourMT3+ 分别转写
"""
import logging
import time
from pathlib import Path
from typing import Optional, Callable

from src.models.data_models import (
    Config, ProcessingResult, ProcessingProgress,
    ProcessingStage, Track, TrackType, ProcessingMode
)
from src.core.yourmt3_transcriber import YourMT3Transcriber
from src.core.beat_detector import BeatDetector
from src.core.midi_generator import MidiGenerator
from src.utils.gpu_utils import clear_gpu_memory

logger = logging.getLogger(__name__)


class MusicToMidiPipeline:
    """
    音乐转MIDI主处理流水线

    处理模式：
        SMART: 使用 YourMT3+ MoE 直接对完整混音进行多乐器转写。
        VOCAL_SPLIT: Demucs 分离人声与伴奏，均使用 YourMT3+ 分别转写后输出两个 MIDI 文件。
    """

    def __init__(self, config: Config):
        self.config = config
        self.yourmt3_transcriber = YourMT3Transcriber(config)
        self.beat_detector = BeatDetector(config)
        self.midi_generator = MidiGenerator(config)

        self._cancelled = False
        self._progress_callback: Optional[Callable[[ProcessingProgress], None]] = None

        self.yourmt3_transcriber.set_cancel_check(lambda: self._cancelled)

    def set_progress_callback(self, callback: Callable[[ProcessingProgress], None]) -> None:
        self._progress_callback = callback

    def cancel(self) -> None:
        self._cancelled = True
        self.yourmt3_transcriber.cancel()
        logger.info("处理已取消")

    def _report(self, stage: ProcessingStage, sp: float, op: float, msg: str) -> None:
        if self._progress_callback:
            self._progress_callback(ProcessingProgress(
                stage=stage,
                stage_progress=sp,
                overall_progress=op,
                message=msg
            ))

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("用户取消了处理")

    def process(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
    ) -> ProcessingResult:
        """
        处理音频文件并输出 MIDI。

        根据 config.processing_mode 分发到对应的处理流程。
        """
        self._cancelled = False
        self._progress_callback = progress_callback

        mode = self.config.processing_mode
        if mode == "vocal_split":
            return self._process_vocal_split(audio_path, output_dir)
        else:
            return self._process_smart(audio_path, output_dir)

    def _process_smart(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """YourMT3+ 智能模式：直接对完整混音进行多乐器转写"""
        start_time = time.time()

        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(audio_path).stem
        midi_path = str(Path(output_dir) / f"{stem}.mid")

        logger.info(f"开始处理 (智能模式): {audio_path}")

        # ── 检查 YourMT3+ 是否可用 ──
        logger.info("正在检查 YourMT3+ 可用性...")
        if not YourMT3Transcriber.is_available():
            raise RuntimeError(
                "YourMT3+ 不可用。\n\n"
                "请先下载模型权重：\n"
                "  python download_sota_models.py\n\n"
                "详见 README.md 中的安装说明。"
            )
        logger.info("YourMT3+ 可用性检查通过")

        # ── 阶段1：预处理 / 节拍检测 ──
        self._report(ProcessingStage.PREPROCESSING, 0.0, 0.0, "正在分析音频...")
        self._check_cancelled()

        beat_info = None
        try:
            beat_info = self.beat_detector.detect(audio_path)
            tempo = beat_info.bpm if beat_info else 120.0
            logger.info(f"节拍检测完成: {tempo:.1f} BPM")
        except Exception as e:
            logger.warning(f"节拍检测失败，使用默认 120 BPM: {e}")
            tempo = 120.0

        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.1, f"BPM: {tempo:.1f}")
        self._check_cancelled()

        # ── 阶段2：YourMT3+ 极致精度转写 ──
        self._report(ProcessingStage.TRANSCRIPTION, 0.0, 0.1, "正在加载 YourMT3+ 模型...")
        logger.info("使用 YourMT3+ MoE 进行极致精度转写")

        quality = self.config.transcription_quality

        def _transcribe_cb(p: float, msg: str) -> None:
            overall = 0.1 + p * 0.75
            self._report(ProcessingStage.TRANSCRIPTION, p, overall, msg)

        try:
            instrument_notes, drum_notes = self.yourmt3_transcriber.transcribe_precise(
                audio_path=audio_path,
                quality=quality,
                progress_callback=_transcribe_cb,
            )
        except InterruptedError:
            raise
        except Exception as e:
            logger.error(f"YourMT3+ 转写失败: {e}", exc_info=True)
            raise RuntimeError(f"YourMT3+ 转写失败: {e}") from e
        finally:
            try:
                self.yourmt3_transcriber.unload_model()
            except Exception as e:
                logger.warning(f"模型卸载失败: {e}")
            try:
                clear_gpu_memory()
            except Exception as e:
                logger.warning(f"GPU内存清理失败: {e}")

        self._check_cancelled()

        total_notes = sum(len(n) for n in instrument_notes.values()) + \
                      sum(len(n) for n in drum_notes.values())
        logger.info(f"转写完成: {len(instrument_notes)} 种乐器, {total_notes} 个音符")

        self._report(
            ProcessingStage.TRANSCRIPTION, 1.0, 0.85,
            f"转写完成：{len(instrument_notes)} 种乐器，{total_notes} 个音符"
        )

        # ── 阶段3：生成 MIDI ──
        self._report(ProcessingStage.SYNTHESIS, 0.0, 0.85, "正在生成 MIDI 文件...")
        self._check_cancelled()
        logger.info("开始生成 MIDI 文件...")

        try:
            midi_path = self.midi_generator.generate_from_precise_instruments_v2(
                instrument_notes=instrument_notes,
                drum_notes=drum_notes,
                tempo=tempo,
                output_path=midi_path,
                quality=quality,
            )
        except Exception as e:
            logger.error(f"MIDI 生成失败: {e}", exc_info=True)
            raise RuntimeError(f"MIDI 生成失败: {e}") from e

        self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, "MIDI 文件已生成")

        # ── 完成 ──
        processing_time = time.time() - start_time
        self._report(ProcessingStage.COMPLETE, 1.0, 1.0,
                     f"处理完成，耗时 {processing_time:.1f} 秒")

        logger.info(f"处理完成: {midi_path} (耗时 {processing_time:.1f}s)")

        return ProcessingResult(
            midi_path=midi_path,
            tracks=[Track(type=TrackType.OTHER, audio_path=audio_path)],
            beat_info=beat_info,
            processing_time=processing_time,
            total_notes=total_notes,
        )

    def _process_vocal_split(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """人声分离模式：Demucs 分离 → YourMT3+ 分别转写伴奏和人声"""
        from src.core.vocal_separator import VocalSeparator

        start_time = time.time()

        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(audio_path).stem

        logger.info(f"开始处理 (人声分离模式): {audio_path}")

        # ── 检查依赖 ──
        logger.info("正在检查依赖: Demucs, YourMT3+...")
        if not VocalSeparator.is_available():
            raise RuntimeError(
                "Demucs 不可用。请安装: pip install demucs>=4.0.0"
            )
        if not YourMT3Transcriber.is_available():
            raise RuntimeError(
                "YourMT3+ 不可用。\n\n"
                "请先下载模型权重：\n"
                "  python download_sota_models.py\n\n"
                "详见 README.md 中的安装说明。"
            )
        logger.info("所有依赖检查通过")

        # ── 阶段1：预处理 / 节拍检测 (0-5%) ──
        self._report(ProcessingStage.PREPROCESSING, 0.0, 0.0, "正在分析音频...")
        self._check_cancelled()

        beat_info = None
        try:
            beat_info = self.beat_detector.detect(audio_path)
            tempo = beat_info.bpm if beat_info else 120.0
            logger.info(f"节拍检测完成: {tempo:.1f} BPM")
        except Exception as e:
            logger.warning(f"节拍检测失败，使用默认 120 BPM: {e}")
            tempo = 120.0

        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.05, f"BPM: {tempo:.1f}")
        self._check_cancelled()

        # ── 阶段2：Demucs 人声分离 (5-35%) ──
        self._report(ProcessingStage.SEPARATION, 0.0, 0.05, "正在分离人声与伴奏...")

        separator = VocalSeparator()
        separator.set_cancel_check(lambda: self._cancelled)

        def _sep_cb(p: float, msg: str) -> None:
            overall = 0.05 + p * 0.30
            self._report(ProcessingStage.SEPARATION, p, overall, msg)

        try:
            separated = separator.separate(
                audio_path=audio_path,
                output_dir=output_dir,
                progress_callback=_sep_cb,
            )
        except InterruptedError:
            raise
        except Exception as e:
            logger.error(f"人声分离失败: {e}", exc_info=True)
            raise RuntimeError(f"人声分离失败: {e}") from e

        if not separated or "vocals" not in separated or "no_vocals" not in separated:
            raise RuntimeError("人声分离失败: 输出不完整，缺少 vocals 或 no_vocals")

        vocals_path = separated["vocals"]
        accompaniment_path = separated["no_vocals"]
        logger.info(f"人声文件: {vocals_path}")
        logger.info(f"伴奏文件: {accompaniment_path}")

        self._report(ProcessingStage.SEPARATION, 1.0, 0.35, "人声分离完成")
        self._check_cancelled()

        # ── 阶段3：YourMT3+ 转写伴奏 (35-70%) ──
        self._report(ProcessingStage.TRANSCRIPTION, 0.0, 0.35, "正在加载 YourMT3+ 模型...")
        logger.info("使用 YourMT3+ 转写伴奏")

        quality = self.config.transcription_quality

        def _transcribe_cb(p: float, msg: str) -> None:
            overall = 0.35 + p * 0.25
            self._report(ProcessingStage.TRANSCRIPTION, p, overall, msg)

        try:
            instrument_notes, drum_notes = self.yourmt3_transcriber.transcribe_precise(
                audio_path=accompaniment_path,
                quality=quality,
                progress_callback=_transcribe_cb,
            )
        except InterruptedError:
            raise
        except Exception as e:
            logger.error(f"伴奏转写失败: {e}", exc_info=True)
            raise RuntimeError(f"伴奏转写失败: {e}") from e
        finally:
            try:
                self.yourmt3_transcriber.unload_model()
            except Exception as e:
                logger.warning(f"伴奏转写后模型卸载失败: {e}")
            try:
                clear_gpu_memory()
            except Exception as e:
                logger.warning(f"伴奏转写后GPU内存清理失败: {e}")

        self._check_cancelled()

        acc_total = sum(len(n) for n in instrument_notes.values()) + \
                    sum(len(n) for n in drum_notes.values())
        logger.info(f"伴奏转写完成: {len(instrument_notes)} 种乐器, {acc_total} 个音符")

        self._report(ProcessingStage.TRANSCRIPTION, 1.0, 0.60, f"伴奏转写完成：{acc_total} 个音符")

        # ── 阶段4：YourMT3+ 转写人声 (60-85%) ──
        self._report(ProcessingStage.VOCAL_TRANSCRIPTION, 0.0, 0.60, "正在用 YourMT3+ 转写人声...")
        logger.info(f"开始 YourMT3+ 人声转写: {vocals_path}")

        def _vocal_cb(p: float, msg: str) -> None:
            overall = 0.60 + p * 0.25
            self._report(ProcessingStage.VOCAL_TRANSCRIPTION, p, overall, msg)

        try:
            vocal_instrument_notes, vocal_drum_notes = self.yourmt3_transcriber.transcribe_precise(
                audio_path=vocals_path,
                quality=quality,
                progress_callback=_vocal_cb,
            )
        except InterruptedError:
            raise
        except Exception as e:
            logger.error(f"人声转写失败: {e}", exc_info=True)
            raise RuntimeError(f"人声转写失败: {e}") from e
        finally:
            try:
                self.yourmt3_transcriber.unload_model()
            except Exception as e:
                logger.warning(f"模型卸载失败: {e}")
            try:
                clear_gpu_memory()
            except Exception as e:
                logger.warning(f"GPU内存清理失败: {e}")

        vocal_total = sum(len(n) for n in vocal_instrument_notes.values()) + \
                      sum(len(n) for n in vocal_drum_notes.values())
        logger.info(f"人声转写完成: {len(vocal_instrument_notes)} 种乐器, {vocal_total} 个音符")

        self._report(ProcessingStage.VOCAL_TRANSCRIPTION, 1.0, 0.85,
                     f"人声转写完成：{vocal_total} 个音符")
        self._check_cancelled()

        # ── 阶段5：生成两个 MIDI 文件 (85-95%) ──
        self._report(ProcessingStage.SYNTHESIS, 0.0, 0.85, "正在生成 MIDI 文件...")
        logger.info("开始生成 MIDI 文件...")

        accompaniment_midi_path = str(Path(output_dir) / f"{stem}_accompaniment.mid")
        vocal_midi_path = str(Path(output_dir) / f"{stem}_vocal.mid")

        try:
            # 伴奏 MIDI
            accompaniment_midi_path = self.midi_generator.generate_from_precise_instruments_v2(
                instrument_notes=instrument_notes,
                drum_notes=drum_notes,
                tempo=tempo,
                output_path=accompaniment_midi_path,
                quality=quality,
            )

            # 人声 MIDI（同样使用 v2 生成器）
            vocal_midi_path = self.midi_generator.generate_from_precise_instruments_v2(
                instrument_notes=vocal_instrument_notes,
                drum_notes=vocal_drum_notes,
                tempo=tempo,
                output_path=vocal_midi_path,
                quality=quality,
            )

        except Exception as e:
            logger.error(f"MIDI 生成失败: {e}", exc_info=True)
            raise RuntimeError(f"MIDI 生成失败: {e}") from e

        self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, "MIDI 文件已生成")

        # ── 完成 ──
        processing_time = time.time() - start_time
        self._report(ProcessingStage.COMPLETE, 1.0, 1.0,
                     f"处理完成，耗时 {processing_time:.1f} 秒")

        logger.info(f"处理完成: 伴奏={accompaniment_midi_path}, 人声={vocal_midi_path} "
                    f"(耗时 {processing_time:.1f}s)")

        return ProcessingResult(
            midi_path=accompaniment_midi_path,
            tracks=[Track(type=TrackType.OTHER, audio_path=audio_path)],
            beat_info=beat_info,
            processing_time=processing_time,
            total_notes=acc_total + vocal_total,
            vocal_midi_path=vocal_midi_path,
            accompaniment_midi_path=accompaniment_midi_path,
            separated_audio=separated,
        )
