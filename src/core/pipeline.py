"""
主处理流水线 - 仅使用 YourMT3+ MoE 多乐器转写

唯一处理路径：YourMT3+ 直接对完整混音进行极致精度转写，
输出 128 种 GM 乐器的多轨道 MIDI，不做音源分离，无降级处理。
"""
import logging
import time
from pathlib import Path
from typing import Optional, Callable

from src.models.data_models import (
    Config, ProcessingResult, ProcessingProgress,
    ProcessingStage, Track, TrackType, TrackLayout, ProcessingMode
)
from src.core.yourmt3_transcriber import YourMT3Transcriber
from src.core.beat_detector import BeatDetector
from src.core.midi_generator import MidiGenerator
from src.utils.gpu_utils import clear_gpu_memory

logger = logging.getLogger(__name__)


class MusicToMidiPipeline:
    """
    音乐转MIDI主处理流水线

    唯一处理模式：
        使用 YourMT3+ MoE 直接对完整混音进行多乐器转写。
        精确识别 128 种 GM 乐器，无需音源分离，无降级备用方案。

    若 YourMT3+ 不可用（模型/代码库未安装），直接抛出错误。
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
        track_layout: Optional[TrackLayout] = None,
    ) -> ProcessingResult:
        """
        处理音频文件并输出 MIDI。

        参数:
            audio_path:        输入音频路径
            output_dir:        输出目录
            progress_callback: 进度回调
            track_layout:      轨道布局（保留参数兼容性，此处不使用）

        返回:
            ProcessingResult
        """
        self._cancelled = False
        self._progress_callback = progress_callback
        start_time = time.time()

        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(audio_path).stem
        midi_path = str(Path(output_dir) / f"{stem}.mid")

        logger.info(f"开始处理: {audio_path}")

        # ── 检查 YourMT3+ 是否可用 ──────────────────────────────────
        if not YourMT3Transcriber.is_available():
            raise RuntimeError(
                "YourMT3+ 不可用。\n\n"
                "请先安装 YourMT3+ 代码库和模型权重：\n"
                "  bash install_yourmt3_code.sh\n"
                "  python download_sota_models.py\n\n"
                "详见 README.md 中的安装说明。"
            )

        # ── 阶段1：预处理 / 节拍检测 ──────────────────────────────
        self._report(ProcessingStage.PREPROCESSING, 0.0, 0.0, "正在分析音频...")
        self._check_cancelled()

        try:
            beat_info = self.beat_detector.detect(audio_path)
            tempo = beat_info.bpm if beat_info else 120.0
            logger.info(f"节拍检测完成: {tempo:.1f} BPM")
        except Exception as e:
            logger.warning(f"节拍检测失败，使用默认 120 BPM: {e}")
            tempo = 120.0

        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.1, f"BPM: {tempo:.1f}")
        self._check_cancelled()

        # ── 阶段2：YourMT3+ 极致精度转写 ──────────────────────────
        self._report(ProcessingStage.TRANSCRIPTION, 0.0, 0.1, "正在加载 YourMT3+ 模型...")
        logger.info("使用 YourMT3+ MoE 进行极致精度转写")

        quality = self.config.transcription_quality  # "fast" / "balanced" / "best"

        def _transcribe_cb(p: float, msg: str) -> None:
            # 转写阶段占总进度 0.1 ~ 0.85
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
                clear_gpu_memory()
            except Exception:
                pass

        self._check_cancelled()

        total_notes = sum(len(n) for n in instrument_notes.values()) + \
                      sum(len(n) for n in drum_notes.values())
        logger.info(f"转写完成: {len(instrument_notes)} 种乐器, {total_notes} 个音符")

        self._report(
            ProcessingStage.TRANSCRIPTION, 1.0, 0.85,
            f"转写完成：{len(instrument_notes)} 种乐器，{total_notes} 个音符"
        )

        # ── 阶段3：生成 MIDI ────────────────────────────────────────
        self._report(ProcessingStage.SYNTHESIS, 0.0, 0.85, "正在生成 MIDI 文件...")
        self._check_cancelled()

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

        # ── 完成 ────────────────────────────────────────────────────
        processing_time = time.time() - start_time
        self._report(ProcessingStage.COMPLETE, 1.0, 1.0,
                     f"处理完成，耗时 {processing_time:.1f} 秒")

        logger.info(f"处理完成: {midi_path} (耗时 {processing_time:.1f}s)")

        return ProcessingResult(
            midi_path=midi_path,
            tracks=[Track(type=TrackType.OTHER, audio_path=audio_path)],
            processing_time=processing_time,
        )
