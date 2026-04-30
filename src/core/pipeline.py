"""
主处理流水线。

处理模式：
1. SMART: YourMT3+ 直接对完整混音进行极致精度转写
2. VOCAL_SPLIT: BS-RoFormer 分离人声与伴奏，均使用 YourMT3+ 分别转写（可选合并 MIDI）
"""
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Callable, Dict

from src.models.data_models import (
    Config, ProcessingResult, ProcessingProgress,
    ProcessingStage, Track, TrackType,
    MultiInstrumentModel,
)
from src.core.yourmt3_transcriber import YourMT3Transcriber
from src.core.beat_detector import BeatDetector
from src.core.midi_generator import MidiGenerator
from src.utils.gpu_utils import clear_gpu_memory
from src.utils.runtime_paths import get_ffmpeg_executable

logger = logging.getLogger(__name__)

try:
    from src.core.miros_transcriber import MirosTranscriber
    _MIROS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised through pipeline fallback behavior
    MirosTranscriber = None
    _MIROS_IMPORT_ERROR = exc


class _UnavailableMirosTranscriber:
    def __init__(self, import_error: Exception):
        self._import_error = import_error

    def set_cancel_check(self, _callback):
        return None

    def cancel(self):
        return None

    def unload_model(self):
        return None

    def is_available(self) -> bool:
        return False

    def get_unavailable_reason(self) -> str:
        return f"MIROS unavailable: {self._import_error}"


class MusicToMidiPipeline:
    """
    音乐转MIDI主处理流水线

    处理模式：
        SMART: 使用 YourMT3+ MoE 直接对完整混音进行多乐器转写。
        VOCAL_SPLIT: BS-RoFormer 分离人声与伴奏，均使用 YourMT3+ 分别转写。
                     可选额外输出一个“人声+伴奏合并 MIDI”。
    """

    def __init__(self, config: Config):
        self.config = config
        self.yourmt3_transcriber = YourMT3Transcriber(config)
        self.miros_transcriber = (
            MirosTranscriber(config)
            if MirosTranscriber is not None
            else _UnavailableMirosTranscriber(_MIROS_IMPORT_ERROR or RuntimeError("MIROS import failed"))
        )
        self.beat_detector = BeatDetector(config)
        self.midi_generator = MidiGenerator(config)

        self._cancelled = False
        self._progress_callback: Optional[Callable[[ProcessingProgress], None]] = None

        self.yourmt3_transcriber.set_cancel_check(lambda: self._cancelled)
        self.miros_transcriber.set_cancel_check(lambda: self._cancelled)

    def set_progress_callback(self, callback: Callable[[ProcessingProgress], None]) -> None:
        self._progress_callback = callback

    def cancel(self) -> None:
        self._cancelled = True
        self.yourmt3_transcriber.cancel()
        self.miros_transcriber.cancel()
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

    def _get_multi_instrument_model_name(self) -> str:
        backend = str(getattr(self.config, "transcription_backend", None) or "").strip().lower()
        valid_values = {model.value for model in MultiInstrumentModel}
        if backend in valid_values:
            return backend

        multi_model = str(getattr(self.config, "multi_instrument_model", None) or "").strip().lower()
        if multi_model in valid_values:
            return multi_model

        if any(candidate not in (None, "") for candidate in (backend, multi_model)):
            logger.warning(
                "Unknown multi-instrument model/backend (%s, %s); falling back to %s",
                getattr(self.config, "multi_instrument_model", None),
                getattr(self.config, "transcription_backend", None),
                MultiInstrumentModel.YOURMT3.value,
            )
        return MultiInstrumentModel.YOURMT3.value

    def _report_quality_behavior(self, mode_label: str) -> None:
        quality = getattr(self.config, "transcription_quality", "best")
        behavior = self.config.get_quality_behavior()

        if behavior.value == "configurable":
            logger.info(
                "Quality preset '%s' applies to the %s pass in %s mode.",
                quality,
                self._get_multi_instrument_label(),
                mode_label,
            )
            return

        logger.info(
            "Quality preset '%s' does not change inference in %s mode because the active path uses fixed checkpoint quality.",
            quality,
            mode_label,
        )

    def _get_multi_instrument_transcriber(self):
        model_name = self._get_multi_instrument_model_name()
        if model_name == MultiInstrumentModel.MIROS.value:
            return self.miros_transcriber
        return self.yourmt3_transcriber

    def _get_multi_instrument_label(self) -> str:
        model_name = self._get_multi_instrument_model_name()
        if model_name == MultiInstrumentModel.MIROS.value:
            return "MIROS"
        return "YourMT3+"

    def _require_multi_instrument_available(self) -> None:
        transcriber = self._get_multi_instrument_transcriber()
        is_available = getattr(transcriber, "is_available", None)
        if not callable(is_available) or is_available():
            return

        reason_getter = getattr(transcriber, "get_unavailable_reason", None)
        if callable(reason_getter):
            raise RuntimeError(reason_getter())

        raise RuntimeError(f"{self._get_multi_instrument_label()} 不可用")

    def _cleanup_multi_instrument_backend(self) -> None:
        transcriber = self._get_multi_instrument_transcriber()
        try:
            if hasattr(transcriber, "unload_model"):
                transcriber.unload_model()
        except Exception as exc:
            logger.warning("多乐器模型资源释放失败: %s", exc)
        try:
            clear_gpu_memory()
        except Exception as exc:
            logger.warning("GPU内存清理失败: %s", exc)

    @staticmethod
    def _ensure_wav(audio_path: str, output_dir: str) -> str:
        """非 WAV 格式自动转换为 WAV（44100Hz, PCM 16-bit），WAV 直接返回。"""
        audio_path = str(audio_path)
        if Path(audio_path).suffix.lower() == ".wav":
            return audio_path

        stem = Path(audio_path).stem
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        wav_path = str(Path(output_dir) / f"{stem}_converted.wav")

        # 优先使用 FFmpeg
        try:
            ffmpeg_exe = get_ffmpeg_executable()
            cmd = [
                ffmpeg_exe, "-y", "-i", audio_path,
                "-ar", "44100", "-ac", "2", "-sample_fmt", "s16",
                wav_path,
            ]
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=300,
            )
            if Path(wav_path).exists() and Path(wav_path).stat().st_size > 0:
                logger.info("已将 %s 转换为 WAV（FFmpeg）", Path(audio_path).name)
                return wav_path
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            logger.warning("FFmpeg 转换失败，尝试 librosa fallback: %s", exc)

        # Fallback: librosa + soundfile
        try:
            import librosa
            import soundfile as sf

            audio, sr = librosa.load(audio_path, sr=44100, mono=False)
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            sf.write(wav_path, audio.T, sr, subtype="PCM_16")
            logger.info("已将 %s 转换为 WAV（librosa）", Path(audio_path).name)
            return wav_path
        except Exception as exc:
            raise RuntimeError(
                f"无法将 {Path(audio_path).name} 转换为 WAV: {exc}\n"
                "请安装 FFmpeg 或确保 librosa/soundfile 可用。"
            ) from exc

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

        # 非 WAV 格式自动转换为 WAV
        audio_path = self._ensure_wav(audio_path, output_dir)

        mode = self.config.processing_mode
        if mode == "vocal_split":
            return self._process_vocal_split(audio_path, output_dir)
        return self._process_smart(audio_path, output_dir)

    @staticmethod
    def _has_note_messages(track) -> bool:
        return any(
            not message.is_meta and message.type in {"note_on", "note_off"}
            for message in track
        )

    def _merge_stem_midis(
        self,
        midi_paths_by_track: Dict[str, str],
        output_path: str,
        tempo: float,
    ) -> str:
        from mido import MidiFile, MidiTrack, MetaMessage, bpm2tempo

        merged = MidiFile(ticks_per_beat=self.config.ticks_per_beat)

        meta_track = MidiTrack()
        merged.tracks.append(meta_track)
        meta_track.append(MetaMessage("set_tempo", tempo=bpm2tempo(tempo), time=0))
        meta_track.append(
            MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0,
            )
        )
        meta_track.append(MetaMessage("end_of_track", time=0))

        for track_name, midi_path in midi_paths_by_track.items():
            src = MidiFile(midi_path)
            for src_track in src.tracks:
                if not self._has_note_messages(src_track):
                    continue
                dst_track = MidiTrack()
                has_track_name = False
                for message in src_track:
                    if message.is_meta and message.type == "track_name":
                        has_track_name = True
                    dst_track.append(message.copy(time=message.time))
                if not has_track_name:
                    dst_track.insert(0, MetaMessage("track_name", name=track_name, time=0))
                merged.tracks.append(dst_track)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged.save(output_path)
        return output_path

    def _process_smart(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """智能模式：直接对完整混音进行多乐器转写。"""
        start_time = time.time()

        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(Path(audio_path).name).stem
        midi_path = str(Path(output_dir) / f"{stem}.mid")
        transcriber = self._get_multi_instrument_transcriber()
        model_label = self._get_multi_instrument_label()

        logger.info(f"开始处理 (智能模式): {audio_path}")
        self._report_quality_behavior("smart")

        # ── 检查多乐器后端是否可用 ──
        logger.info("正在检查 %s 可用性...", model_label)
        self._require_multi_instrument_available()
        logger.info("%s 可用性检查通过", model_label)

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

        # ── 阶段2：多乐器后端极致精度转写 ──
        self._report(ProcessingStage.TRANSCRIPTION, 0.0, 0.1, f"正在加载 {model_label} 模型...")
        logger.info("使用 %s 进行极致精度转写", model_label)

        quality = self.config.transcription_quality

        def _transcribe_cb(p: float, msg: str) -> None:
            overall = 0.1 + p * 0.75
            self._report(ProcessingStage.TRANSCRIPTION, p, overall, msg)

        try:
            instrument_notes, drum_notes = transcriber.transcribe_precise(
                audio_path=audio_path,
                quality=quality,
                progress_callback=_transcribe_cb,
            )
        except InterruptedError:
            raise
        except Exception as e:
            logger.error("%s 转写失败: %s", model_label, e, exc_info=True)
            raise RuntimeError(f"{model_label} 转写失败: {e}") from e
        finally:
            self._cleanup_multi_instrument_backend()

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
        """人声分离模式：BS-RoFormer 分离 → 多乐器后端分别转写伴奏和人声"""
        from src.core.vocal_separator import VocalSeparator

        start_time = time.time()

        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(Path(audio_path).name).stem

        logger.info(f"开始处理 (人声分离模式): {audio_path}")
        transcriber = self._get_multi_instrument_transcriber()
        model_label = self._get_multi_instrument_label()
        self._report_quality_behavior("vocal_split")

        # ── 检查依赖 ──
        logger.info("正在检查依赖: BS-RoFormer, %s...", model_label)
        if not VocalSeparator.is_available():
            raise RuntimeError(
                "人声分离不可用。请安装: pip install audio-separator>=0.38.0"
            )
        self._require_multi_instrument_available()
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

        # ── 阶段2：BS-RoFormer 人声分离 (5-35%) ──
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

        if not os.path.exists(vocals_path) or not os.path.exists(accompaniment_path):
            raise RuntimeError(
                f"分离文件不存在: vocals={os.path.exists(vocals_path)}, "
                f"accompaniment={os.path.exists(accompaniment_path)}"
            )

        logger.info(f"人声文件: {vocals_path}")
        logger.info(f"伴奏文件: {accompaniment_path}")

        self._report(ProcessingStage.SEPARATION, 1.0, 0.35, "人声分离完成")
        self._check_cancelled()

        # ── 阶段3：多乐器后端转写伴奏 (35-70%) ──
        self._report(ProcessingStage.TRANSCRIPTION, 0.0, 0.35, f"正在加载 {model_label} 模型...")
        logger.info("使用 %s 转写伴奏", model_label)

        quality = self.config.transcription_quality

        def _transcribe_cb(p: float, msg: str) -> None:
            overall = 0.35 + p * 0.25
            self._report(ProcessingStage.TRANSCRIPTION, p, overall, msg)

        try:
            instrument_notes, drum_notes = transcriber.transcribe_precise(
                audio_path=accompaniment_path,
                quality=quality,
                progress_callback=_transcribe_cb,
            )
        except InterruptedError:
            raise
        except Exception as e:
            logger.error("%s 伴奏转写失败: %s", model_label, e, exc_info=True)
            raise RuntimeError(f"{model_label} 伴奏转写失败: {e}") from e
        finally:
            self._cleanup_multi_instrument_backend()

        self._check_cancelled()

        acc_total = sum(len(n) for n in instrument_notes.values()) + \
                    sum(len(n) for n in drum_notes.values())
        logger.info(f"伴奏转写完成: {len(instrument_notes)} 种乐器, {acc_total} 个音符")

        self._report(ProcessingStage.TRANSCRIPTION, 1.0, 0.60, f"伴奏转写完成：{acc_total} 个音符")

        # ── 阶段4：多乐器后端转写人声 (60-85%) ──
        self._report(ProcessingStage.VOCAL_TRANSCRIPTION, 0.0, 0.60, f"正在用 {model_label} 转写人声...")
        logger.info("开始 %s 人声转写: %s", model_label, vocals_path)

        def _vocal_cb(p: float, msg: str) -> None:
            overall = 0.60 + p * 0.25
            self._report(ProcessingStage.VOCAL_TRANSCRIPTION, p, overall, msg)

        try:
            vocal_instrument_notes, vocal_drum_notes = transcriber.transcribe_precise(
                audio_path=vocals_path,
                quality=quality,
                progress_callback=_vocal_cb,
            )
        except InterruptedError:
            raise
        except Exception as e:
            logger.error("%s 人声转写失败: %s", model_label, e, exc_info=True)
            raise RuntimeError(f"{model_label} 人声转写失败: {e}") from e
        finally:
            self._cleanup_multi_instrument_backend()

        # 人声 MIDI 只保留人声旋律，丢弃模型幻觉出的其他乐器和鼓
        # YourMT3 人声程序号: 100 (主旋律), 101 (和声); 也可能输出 GM 0 (钢琴)
        raw_inst_count = len(vocal_instrument_notes)
        raw_note_count = sum(len(n) for n in vocal_instrument_notes.values())
        raw_drum_count = sum(len(n) for n in vocal_drum_notes.values())

        # 优先使用 YourMT3 人声程序号 100/101，其次回退到 GM 0
        vocal_programs = [p for p in (100, 101, 0) if p in vocal_instrument_notes]
        if vocal_programs:
            # 合并所有匹配的人声音符，统一为 program 0（钢琴音色）
            merged_vocal_notes = []
            for vp in vocal_programs:
                merged_vocal_notes.extend(vocal_instrument_notes[vp])
            merged_vocal_notes.sort(key=lambda n: n.start_time)
            vocal_instrument_notes = {0: merged_vocal_notes}
        else:
            # 模型未识别出人声旋律，尝试使用音符最多的乐器作为人声
            if vocal_instrument_notes:
                best_program = max(vocal_instrument_notes, key=lambda k: len(vocal_instrument_notes[k]))
                logger.warning(
                    f"人声转写未产生人声/钢琴音符，回退使用 GM {best_program:03d} "
                    f"({len(vocal_instrument_notes[best_program])} 个音符) 作为人声旋律"
                )
                vocal_instrument_notes = {0: vocal_instrument_notes[best_program]}
            else:
                logger.warning("人声转写未产生任何音符，人声 MIDI 将为空")
                vocal_instrument_notes = {}
        vocal_drum_notes = {}

        kept = sum(len(n) for n in vocal_instrument_notes.values())
        logger.info(
            f"人声过滤: 保留人声 {kept} 个音符, "
            f"丢弃 {raw_inst_count - len(vocal_instrument_notes)} 种乐器 "
            f"({raw_note_count - kept} 个音符) + {raw_drum_count} 个鼓音符"
        )

        vocal_total = kept
        logger.info(f"人声转写完成: {len(vocal_instrument_notes)} 种乐器, {vocal_total} 个音符")

        self._report(ProcessingStage.VOCAL_TRANSCRIPTION, 1.0, 0.85,
                     f"人声转写完成：{vocal_total} 个音符")
        self._check_cancelled()

        # ── 阶段5：生成两个 MIDI 文件 (85-95%) ──
        self._check_cancelled()
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

            self._check_cancelled()

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

        merged_midi_path = None
        if getattr(self.config, "vocal_split_merge_midi", False):
            self._check_cancelled()
            self._report(ProcessingStage.SYNTHESIS, 0.85, 0.92, "正在合并伴奏+人声 MIDI...")
            merged_midi_path = str(Path(output_dir) / f"{stem}_vocal_accompaniment_merged.mid")
            merged_midi_path = self._merge_stem_midis(
                {
                    "accompaniment": accompaniment_midi_path,
                    "vocal": vocal_midi_path,
                },
                merged_midi_path,
                tempo,
            )
            self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, "MIDI 文件已生成（含合并）")
        else:
            self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, "MIDI 文件已生成")

        final_midi_path = merged_midi_path or accompaniment_midi_path

        # ── 完成 ──
        processing_time = time.time() - start_time
        self._report(ProcessingStage.COMPLETE, 1.0, 1.0,
                     f"处理完成，耗时 {processing_time:.1f} 秒")

        logger.info(
            f"处理完成: 伴奏={accompaniment_midi_path}, 人声={vocal_midi_path}, "
            f"合并={merged_midi_path or '未启用'} (耗时 {processing_time:.1f}s)"
        )

        return ProcessingResult(
            midi_path=final_midi_path,
            tracks=[Track(type=TrackType.OTHER, audio_path=audio_path)],
            beat_info=beat_info,
            processing_time=processing_time,
            total_notes=acc_total + vocal_total,
            vocal_midi_path=vocal_midi_path,
            accompaniment_midi_path=accompaniment_midi_path,
            separated_audio=separated,
            merged_midi_path=merged_midi_path,
        )
