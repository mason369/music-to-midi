"""
主处理流水线。

处理模式：
1. SMART: 所选 YourMT3+ / MIROS 后端直接转写完整混音
2. VOCAL_SPLIT: Leap XE + PolarFormer 分离后，用所选后端分别转写两路 stem
3. SIX_STEM_SPLIT: BS-RoFormer SW 分离六个 stem，再逐 stem 调用所选后端
4. PIANO_TRANSKUN: 官方 TransKun 2.0 钢琴转写
5. PIANO_TRANSKUN_V2_AUG: TransKun V2 Aug 钢琴与踏板转写
6. PIANO_ARIA_AMT: Aria-AMT 钢琴转写
7. PIANO_BYTEDANCE_PEDAL: ByteDance 钢琴与踏板转写
"""

import logging
import inspect
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Callable, Dict, List

from src.models.data_models import (
    Config,
    ProcessingResult,
    ProcessingProgress,
    ProcessingStage,
    Track,
    TrackType,
    ProcessingMode,
    MultiInstrumentModel,
    BeatInfo,
    NoteEvent,
)
from src.core.aria_amt_transcriber import AriaAmtTranscriber, ARIA_AMT_SOURCE_REQUIREMENT
from src.core.bytedance_piano_transcriber import (
    ByteDancePianoTranscriber,
    BYTEDANCE_PIANO_PACKAGE_VERSION,
)
from src.core.transkun_transcriber import TranskunTranscriber, TRANSKUN_PACKAGE_VERSION
from src.core.transkun_v2_aug_transcriber import TranskunV2AugTranscriber
from src.core.yourmt3_transcriber import YourMT3Transcriber
from src.core.beat_detector import BeatDetector
from src.core.midi_generator import MidiGenerator
from src.i18n.translator import Translator
from src.utils.gpu_utils import clear_gpu_memory
from src.utils.midi_output import (
    publish_midi_output,
    remove_temporary_midi,
    unique_midi_temp_path,
)
from src.utils.runtime_paths import get_ffmpeg_executable

logger = logging.getLogger(__name__)

try:
    from src.core.miros_transcriber import MirosTranscriber

    _MIROS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised through unavailable-backend tests
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
        SMART: 使用所选多乐器后端直接对完整混音进行转写。
        VOCAL_SPLIT: Leap XE + PolarFormer 分离人声与伴奏，均使用多乐器后端分别转写。
                     可选额外输出一个“人声+伴奏合并 MIDI”。
        SIX_STEM_SPLIT: BS-RoFormer SW 分离六个 stem，每个真实 stem 都独立调用所选后端。
        四个钢琴模式分别固定路由到 TransKun、TransKun V2 Aug、Aria-AMT 与 ByteDance Pedal。
    """

    def __init__(self, config: Config):
        self.config = config
        self.yourmt3_transcriber = YourMT3Transcriber(config)
        self.miros_transcriber = (
            MirosTranscriber(config)
            if MirosTranscriber is not None
            else _UnavailableMirosTranscriber(
                _MIROS_IMPORT_ERROR or RuntimeError("MIROS import failed")
            )
        )
        self.aria_amt_transcriber = AriaAmtTranscriber(
            language=getattr(config, "language", Translator.DEFAULT_LANGUAGE)
        )
        self.bytedance_piano_transcriber = ByteDancePianoTranscriber(config)
        self.transkun_transcriber = TranskunTranscriber(config)
        self.transkun_v2_aug_transcriber = TranskunV2AugTranscriber(config)
        self.beat_detector = BeatDetector(config)
        self.midi_generator = MidiGenerator(config)

        self._cancelled = False
        self._active_separator = None
        self._progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
        language = getattr(config, "language", Translator.DEFAULT_LANGUAGE)
        self._translator = Translator(language)

        self.yourmt3_transcriber.set_cancel_check(lambda: self._cancelled)
        self.miros_transcriber.set_cancel_check(lambda: self._cancelled)

    def set_progress_callback(self, callback: Callable[[ProcessingProgress], None]) -> None:
        self._progress_callback = callback

    def _pt(self, key: str, **kwargs) -> str:
        return self._translator.t(key, **kwargs)

    def cancel(self) -> None:
        self._cancelled = True
        separator = self._active_separator
        cancel_separator = getattr(separator, "cancel", None)
        if callable(cancel_separator):
            cancel_separator()
        self.yourmt3_transcriber.cancel()
        self.miros_transcriber.cancel()
        if hasattr(self.aria_amt_transcriber, "cancel"):
            self.aria_amt_transcriber.cancel()
        if hasattr(self.bytedance_piano_transcriber, "cancel"):
            self.bytedance_piano_transcriber.cancel()
        if hasattr(self.transkun_transcriber, "cancel"):
            self.transkun_transcriber.cancel()
        if hasattr(self.transkun_v2_aug_transcriber, "cancel"):
            self.transkun_v2_aug_transcriber.cancel()
        logger.info("已请求取消；等待活动后端安全停止")

    def _report(self, stage: ProcessingStage, sp: float, op: float, msg: str) -> None:
        logger.info(
            "Progress | stage=%s %.0f%% | overall=%.0f%% | %s",
            stage.value,
            max(0.0, min(1.0, sp)) * 100,
            max(0.0, min(1.0, op)) * 100,
            msg,
        )
        if self._progress_callback:
            self._progress_callback(
                ProcessingProgress(stage=stage, stage_progress=sp, overall_progress=op, message=msg)
            )

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("用户取消了处理")

    def _get_multi_instrument_model_name(self) -> str:
        model_name = self.config.get_effective_multi_instrument_model()
        valid_values = {model.value for model in MultiInstrumentModel}
        if model_name not in valid_values:
            raise ValueError(f"Unsupported multi-instrument backend: {model_name!r}")
        return model_name

    def _get_multi_instrument_transcriber(self):
        model_name = self._get_multi_instrument_model_name()
        if model_name == MultiInstrumentModel.MIROS.value:
            return self.miros_transcriber
        if model_name == MultiInstrumentModel.YOURMT3.value:
            return self.yourmt3_transcriber
        raise ValueError(f"Unsupported multi-instrument backend: {model_name!r}")

    def _get_multi_instrument_label(self) -> str:
        model_name = self._get_multi_instrument_model_name()
        if model_name == MultiInstrumentModel.MIROS.value:
            return "MIROS"
        if model_name == MultiInstrumentModel.YOURMT3.value:
            return "YourMT3+"
        raise ValueError(f"Unsupported multi-instrument backend: {model_name!r}")

    def _require_multi_instrument_available(self) -> None:
        transcriber = self._get_multi_instrument_transcriber()
        is_available = getattr(transcriber, "is_selected_model_available", None)
        if not callable(is_available):
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
    def _format_backend_error(model_label: str, action: str, exc: Exception) -> str:
        prefix = f"{model_label} {action}"
        message = str(exc)
        if message.startswith(prefix):
            return message
        return f"{prefix}: {message}"

    @staticmethod
    def _ensure_wav(audio_path: str, output_dir: str) -> str:
        """非 WAV 格式使用 FFmpeg 转换为 WAV（44100Hz, PCM 16-bit），WAV 直接返回。"""
        audio_path = str(audio_path)
        if Path(audio_path).suffix.lower() == ".wav":
            return audio_path

        stem = Path(audio_path).stem
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        wav_path = str(Path(output_dir) / f"{stem}.wav")

        ffmpeg_exe = get_ffmpeg_executable()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            audio_path,
            "-ar",
            "44100",
            "-ac",
            "2",
            "-sample_fmt",
            "s16",
            wav_path,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "FFmpeg 不可用，无法转换非 WAV 音频。\n"
                "请安装 FFmpeg 并确保 ffmpeg 可执行文件在 PATH 或打包资源中。"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            detail = f"\nFFmpeg stderr:\n{stderr}" if stderr else ""
            raise RuntimeError(
                f"FFmpeg 转换 {Path(audio_path).name} 失败，已停止。{detail}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"FFmpeg 转换 {Path(audio_path).name} 超时，已停止。") from exc
        except subprocess.SubprocessError as exc:
            raise RuntimeError(f"FFmpeg 转换 {Path(audio_path).name} 失败，已停止: {exc}") from exc

        if Path(wav_path).exists() and Path(wav_path).stat().st_size > 0:
            logger.info("已将 %s 转换为 WAV（FFmpeg）", Path(audio_path).name)
            return wav_path

        raise RuntimeError(f"FFmpeg 转换 {Path(audio_path).name} 未生成有效 WAV 文件，已停止。")

    @staticmethod
    def _format_file_size(path: str) -> str:
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        return f"{size_mb:.1f} MB"

    @classmethod
    def _format_output_file_list(cls, output_paths: Dict[str, str]) -> str:
        lines = []
        for name, path in sorted(output_paths.items()):
            size_text = cls._format_file_size(path) if Path(path).is_file() else "missing"
            lines.append(f"{name}: {path} ({size_text})")
        return "\n".join(lines)

    def _report_output_files(
        self,
        stage: ProcessingStage,
        stage_progress: float,
        overall_progress: float,
        title: str,
        output_paths: Dict[str, str],
    ) -> None:
        file_list = self._format_output_file_list(output_paths)
        logger.info("%s:\n%s", title, file_list)
        summary = ", ".join(
            f"{name}={Path(path).name}" for name, path in sorted(output_paths.items())
        )
        self._report(stage, stage_progress, overall_progress, f"{title}: {summary}")

    def _detect_beat_or_raise(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> BeatInfo:
        try:
            detect_params = inspect.signature(self.beat_detector.detect).parameters
            if progress_callback is not None and "progress_callback" in detect_params:
                beat_info = self.beat_detector.detect(
                    audio_path, progress_callback=progress_callback
                )
            else:
                beat_info = self.beat_detector.detect(audio_path)
        except Exception as exc:
            logger.error("节拍检测失败: %s", exc)
            raise RuntimeError(f"节拍检测失败，已停止: {exc}") from exc

        if beat_info is None:
            raise RuntimeError("节拍检测失败，检测器未返回 BPM，已停止。")

        logger.info("节拍检测完成: %.1f BPM", beat_info.bpm)
        return beat_info

    @staticmethod
    def _filter_vocal_melody_notes(
        instrument_notes,
        drum_notes,
        *,
        preserve_isolated_vocal_stem: bool = False,
    ):
        vocal_programs = [p for p in (100, 101, 0) if p in instrument_notes]
        if not vocal_programs:
            raw_note_count = sum(len(notes) for notes in instrument_notes.values())
            raw_drum_count = sum(len(notes) for notes in drum_notes.values())
            if preserve_isolated_vocal_stem and raw_note_count:
                merged_vocal_notes = [
                    NoteEvent(
                        pitch=note.pitch,
                        start_time=note.start_time,
                        end_time=note.end_time,
                        velocity=note.velocity,
                        program=0,
                    )
                    for notes in instrument_notes.values()
                    for note in notes
                ]
                merged_vocal_notes.sort(key=lambda note: note.start_time)
                logger.warning(
                    "人声 stem 转写未产生 program 100/101/0；"
                    "已保留 %s 个真实非鼓音符并归入 vocal 轨，丢弃 %s 个鼓音符。",
                    raw_note_count,
                    raw_drum_count,
                )
                return {0: merged_vocal_notes}, {}
            logger.warning(
                "人声转写未产生 program 100/101/0，输出空人声 MIDI；"
                "丢弃 %s 个非人声乐器音符和 %s 个鼓音符。",
                raw_note_count,
                raw_drum_count,
            )
            return {}, {}

        merged_vocal_notes = []
        for program in vocal_programs:
            merged_vocal_notes.extend(instrument_notes[program])
        merged_vocal_notes.sort(key=lambda note: note.start_time)
        return {0: merged_vocal_notes}, {}

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
        self._progress_callback = progress_callback
        # Cancellation is intentionally sticky for the lifetime of a pipeline
        # instance.  ProcessingWorker creates one pipeline per job, so clearing
        # the flag here would lose a Stop request delivered between start() and
        # the worker thread entering this method.
        self._check_cancelled()
        self.config.validate()
        self._check_cancelled()

        # 非 WAV 格式自动转换为 WAV
        audio_path = self._ensure_wav(audio_path, output_dir)
        self._check_cancelled()

        mode = self.config.processing_mode
        if mode == ProcessingMode.SMART.value:
            return self._process_smart(audio_path, output_dir)
        if mode == ProcessingMode.VOCAL_SPLIT.value:
            return self._process_vocal_split(audio_path, output_dir)
        if mode == ProcessingMode.SIX_STEM_SPLIT.value:
            return self._process_six_stem_split(audio_path, output_dir)
        if mode == ProcessingMode.PIANO_TRANSKUN.value:
            return self._process_piano_transkun(audio_path, output_dir)
        if mode == ProcessingMode.PIANO_TRANSKUN_V2_AUG.value:
            return self._process_piano_transkun_v2_aug(audio_path, output_dir)
        if mode == ProcessingMode.PIANO_ARIA_AMT.value:
            return self._process_piano_aria_amt(audio_path, output_dir)
        if mode == ProcessingMode.PIANO_BYTEDANCE_PEDAL.value:
            return self._process_piano_bytedance_pedal(audio_path, output_dir)
        raise ValueError(f"Unsupported processing mode: {mode!r}")

    @staticmethod
    def _has_note_messages(track) -> bool:
        return any(
            not message.is_meta and message.type in {"note_on", "note_off"} for message in track
        )

    @staticmethod
    def _normalize_midi_tempo_metadata(
        midi_path: str,
        tempo_bpm: float,
        *,
        force: bool = False,
    ) -> str:
        """Add detected tempo without applying any note-level post-processing.

        YourMT3 and MIROS encode absolute seconds against the implicit Standard
        MIDI default tempo (120 BPM) but do not emit a ``set_tempo`` message.
        Match the authorized TelkNet dev contract by inserting the detected tempo
        and recomputing delta ticks from absolute seconds.  Pitch, program,
        velocity, controller, pitch-wheel, and note messages are copied unchanged.
        """
        from math import isfinite

        from mido import MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick, tick2second

        path = Path(midi_path).resolve()
        source = MidiFile(str(path))
        has_tempo = any(
            message.is_meta and message.type == "set_tempo"
            for track in source.tracks
            for message in track
        )
        if has_tempo and not force:
            return str(path)

        bpm = float(tempo_bpm)
        if not isfinite(bpm) or bpm <= 0.0:
            raise RuntimeError(f"检测到无效 MIDI 速度: {tempo_bpm!r}")
        target_tempo = bpm2tempo(bpm)
        output = MidiFile(type=source.type, ticks_per_beat=source.ticks_per_beat)

        for track_index, source_track in enumerate(source.tracks):
            target_track = MidiTrack()
            if track_index == 0:
                target_track.append(MetaMessage("set_tempo", tempo=target_tempo, time=0))

            source_tempo = 500_000
            absolute_seconds = 0.0
            previous_target_tick = 0
            for message in source_track:
                if message.time:
                    absolute_seconds += tick2second(
                        message.time,
                        source.ticks_per_beat,
                        source_tempo,
                    )
                if message.is_meta and message.type == "set_tempo":
                    source_tempo = message.tempo
                    continue

                target_tick = int(
                    round(
                        second2tick(
                            absolute_seconds,
                            output.ticks_per_beat,
                            target_tempo,
                        )
                    )
                )
                if target_tick < previous_target_tick:
                    raise RuntimeError("MIDI tempo normalization produced non-monotonic ticks")
                target_track.append(message.copy(time=target_tick - previous_target_tick))
                previous_target_tick = target_tick
            output.tracks.append(target_track)

        temporary_path = unique_midi_temp_path(path, "tempo-normalized")
        try:
            output.save(str(temporary_path))
            publish_midi_output(
                temporary_path,
                path,
                "Tempo-normalized backend MIDI",
            )
        finally:
            remove_temporary_midi(temporary_path)
        return str(path)

    def _merge_stem_midis(
        self,
        midi_paths_by_track: Dict[str, str],
        output_path: str,
    ) -> str:
        """Combine backend-produced MIDI files without rewriting their events.

        Every source track is copied verbatim at the MIDI-message level.  The
        selected backend writes all stems with one time division; accepting a
        different ``ticks_per_beat`` here would silently change timing, so that
        contract violation is reported instead of being rescaled or hidden.
        """
        from mido import MidiFile, MidiTrack

        self._check_cancelled()
        if not midi_paths_by_track:
            raise RuntimeError("无法合并 MIDI：没有后端 MIDI 输入")

        source_midis = []
        ticks_per_beat = None
        for track_name, midi_path in midi_paths_by_track.items():
            try:
                src = MidiFile(midi_path)
            except Exception as exc:
                raise RuntimeError(
                    f"无法读取后端 MIDI: track={track_name}, path={midi_path}: {exc}"
                ) from exc
            if ticks_per_beat is None:
                ticks_per_beat = src.ticks_per_beat
            elif src.ticks_per_beat != ticks_per_beat:
                raise RuntimeError(
                    "无法无损合并 MIDI：源文件 ticks_per_beat 不一致 "
                    f"(expected={ticks_per_beat}, track={track_name}, "
                    f"actual={src.ticks_per_beat})"
                )
            source_midis.append((track_name, src))

        merged = MidiFile(type=1, ticks_per_beat=ticks_per_beat)
        for _track_name, src in source_midis:
            for src_track in src.tracks:
                dst_track = MidiTrack()
                for message in src_track:
                    dst_track.append(message.copy(time=message.time))
                merged.tracks.append(dst_track)

        temporary_output_path = unique_midi_temp_path(output_path, "stem-merge")
        try:
            merged.save(str(temporary_output_path))
            self._check_cancelled()
            publish_midi_output(
                temporary_output_path,
                output_path,
                "Merged stem MIDI",
            )
        finally:
            remove_temporary_midi(temporary_output_path)
        return output_path

    @staticmethod
    def _count_midi_notes(midi_path: str) -> int:
        path = Path(midi_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"MIDI 输出不存在或为空: {path.resolve()}")
        try:
            from mido import MidiFile

            midi = MidiFile(str(path))
            return sum(
                1
                for track in midi.tracks
                for message in track
                if (not message.is_meta and message.type == "note_on" and message.velocity > 0)
            )
        except Exception as exc:
            raise RuntimeError(f"无法读取 MIDI 输出 {path.resolve()}: {exc}") from exc

    def _finalize_separated_audio(
        self,
        separated_audio: Dict[str, str],
    ) -> Optional[Dict[str, str]]:
        """Honor save_separated_tracks after every requested MIDI has been written."""
        if self.config.save_separated_tracks:
            return dict(separated_audio)

        unique_paths = {Path(path).resolve() for path in separated_audio.values()}
        for path in sorted(unique_paths, key=str):
            if not path.is_file():
                raise RuntimeError(f"无法清理分离音频，文件不存在: {path}")
            try:
                path.unlink()
            except OSError as exc:
                raise RuntimeError(f"无法清理分离音频 {path}: {exc}") from exc
        return None

    def _process_six_stem_split(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """
        六声部分离模式：
        1. BS-RoFormer SW 分离六个 stem（用于 WAV 输出）
        2. 所选 YourMT3+ / MIROS 后端分别转写每个 stem
        3. 输出六个 stem MIDI 和一个合并 MIDI
        """
        from src.core.multi_stem_separator import SixStemSeparator, STEM_KEYS

        start_time = time.time()
        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        input_stem = Path(audio_path).stem

        logger.info(f"开始处理(六声部分离模式): {audio_path}")

        if not SixStemSeparator.is_available():
            raise RuntimeError(
                "六声部分离不可用，请安装: pip install audio-separator==0.44.1 --no-deps"
            )
        self._require_multi_instrument_available()
        transcriber = self._get_multi_instrument_transcriber()
        model_label = self._get_multi_instrument_label()

        self._report(
            ProcessingStage.PREPROCESSING,
            0.0,
            0.0,
            self._pt("progress.preparing_six_stem_separator"),
        )
        self._check_cancelled()
        beat_info = self._detect_beat_or_raise(audio_path)
        tempo = beat_info.bpm
        self._report(
            ProcessingStage.PREPROCESSING,
            1.0,
            0.05,
            f"BPM: {tempo:.1f}",
        )

        self._report(
            ProcessingStage.SEPARATION, 0.0, 0.05, self._pt("progress.six_stem_separating")
        )
        separator_device = "cuda:0" if self.config.use_gpu else "cpu"
        separator = SixStemSeparator(
            language=getattr(self.config, "language", Translator.DEFAULT_LANGUAGE),
            target_device=separator_device,
        )

        def _sep_cb(p: float, msg: str) -> None:
            overall = 0.05 + p * 0.25
            self._report(ProcessingStage.SEPARATION, p, overall, msg)

        separated = separator.separate(
            audio_path=audio_path,
            output_dir=output_dir,
            progress_callback=_sep_cb,
        )

        missing_stems = [stem for stem in STEM_KEYS if stem not in separated]
        if missing_stems:
            raise RuntimeError(f"六声部分离输出不完整，缺少: {missing_stems}")

        self._report_output_files(
            ProcessingStage.SEPARATION,
            1.0,
            0.30,
            self._pt("progress.six_stem_wav_outputs"),
            separated,
        )
        self._report(ProcessingStage.SEPARATION, 1.0, 0.30, self._pt("progress.six_stem_complete"))
        self._check_cancelled()

        stem_midi_paths: Dict[str, str] = {}
        total_notes = 0
        total_stems = len(STEM_KEYS)
        per_stem_span = 0.60 / total_stems

        logger.info(
            "使用 %s 分别转写 %s 个分离 stem（不会对原混音做一次性路由）",
            model_label,
            total_stems,
        )
        try:
            for stem_index, stem_name in enumerate(STEM_KEYS):
                self._check_cancelled()
                stem_audio_path = str(separated[stem_name])
                if not Path(stem_audio_path).is_file():
                    raise RuntimeError(
                        f"六声部分离输出不存在: stem={stem_name}, path={stem_audio_path}"
                    )

                segment_start = 0.30 + stem_index * per_stem_span
                self._report(
                    ProcessingStage.TRANSCRIPTION,
                    stem_index / total_stems,
                    segment_start,
                    self._pt(
                        "progress.running_backend",
                        backend=f"{model_label} [{stem_name}]",
                    ),
                )

                def _stem_transcribe_cb(
                    p: float,
                    msg: str,
                    *,
                    _stem_index: int = stem_index,
                    _stem_name: str = stem_name,
                ) -> None:
                    normalized = max(0.0, min(1.0, p))
                    overall = 0.30 + _stem_index * per_stem_span + normalized * per_stem_span * 0.85
                    self._report(
                        ProcessingStage.TRANSCRIPTION,
                        (_stem_index + normalized) / total_stems,
                        overall,
                        f"{_stem_name}: {msg}",
                    )

                stem_midi_path = str(Path(output_dir) / f"{input_stem}_{stem_name}.mid")
                try:
                    stem_midi_paths[stem_name] = transcriber.transcribe_to_midi(
                        audio_path=stem_audio_path,
                        output_path=stem_midi_path,
                        progress_callback=_stem_transcribe_cb,
                    )
                    stem_midi_paths[stem_name] = self._normalize_midi_tempo_metadata(
                        stem_midi_paths[stem_name],
                        tempo,
                    )
                except InterruptedError:
                    raise
                except Exception as exc:
                    logger.error(
                        "%s 转写 stem %s 失败: %s",
                        model_label,
                        stem_name,
                        exc,
                        exc_info=True,
                    )
                    raise RuntimeError(
                        self._format_backend_error(
                            model_label,
                            f"转写 stem {stem_name} 失败",
                            exc,
                        )
                    ) from exc

                stem_note_count = self._count_midi_notes(stem_midi_paths[stem_name])
                total_notes += stem_note_count
                synthesis_progress = segment_start + per_stem_span * 0.90
                self._report(
                    ProcessingStage.SYNTHESIS,
                    (stem_index + 1) / total_stems,
                    synthesis_progress,
                    self._pt(
                        "progress.generating_stem_midi",
                        stem=stem_name,
                        note_count=stem_note_count,
                    ),
                )
        finally:
            self._cleanup_multi_instrument_backend()

        self._report(
            ProcessingStage.TRANSCRIPTION,
            1.0,
            0.90,
            self._pt(
                "progress.transcription_complete",
                instrument_count=len(STEM_KEYS),
                note_count=total_notes,
            ),
        )
        self._check_cancelled()

        self._check_cancelled()
        merged_midi_path = str(Path(output_dir) / f"{input_stem}_all_stems_merged.mid")
        self._report(ProcessingStage.SYNTHESIS, 0.95, 0.93, self._pt("progress.merging_stem_midi"))
        merged_midi_path = self._merge_stem_midis(stem_midi_paths, merged_midi_path)
        self._report(
            ProcessingStage.SYNTHESIS, 1.0, 0.97, self._pt("progress.six_stem_midi_complete")
        )

        retained_separated_audio = self._finalize_separated_audio(separated)

        processing_time = time.time() - start_time
        self._report(
            ProcessingStage.COMPLETE,
            1.0,
            1.0,
            self._pt("progress.complete_elapsed", seconds=f"{processing_time:.1f}"),
        )

        return ProcessingResult(
            midi_path=merged_midi_path,
            tracks=[Track(type=TrackType.OTHER, audio_path=audio_path)],
            beat_info=beat_info,
            processing_time=processing_time,
            total_notes=total_notes,
            separated_audio=retained_separated_audio,
            stem_midi_paths=stem_midi_paths,
            merged_midi_path=merged_midi_path,
        )

    def _process_specialized_piano(
        self,
        audio_path: str,
        output_dir: str,
        *,
        transcriber,
        mode_label: str,
        output_suffix: str,
        install_hint: str,
        model_hint: str,
    ) -> ProcessingResult:
        start_time = time.time()

        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(Path(audio_path).name).stem
        midi_path = str(Path(output_dir) / f"{stem}_{output_suffix}.mid")

        logger.info("开始处理 (%s 钢琴模式): %s", mode_label, audio_path)

        if not transcriber.is_available():
            raise RuntimeError(install_hint)
        if not transcriber.is_model_available():
            raise RuntimeError(model_hint)

        self._report(
            ProcessingStage.PREPROCESSING, 0.0, 0.0, self._pt("progress.analyzing_piano_audio")
        )
        self._check_cancelled()

        def _beat_cb(p: float, msg: str) -> None:
            self._report(ProcessingStage.PREPROCESSING, p, p * 0.10, msg)

        beat_info = self._detect_beat_or_raise(audio_path, progress_callback=_beat_cb)
        tempo = beat_info.bpm

        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.10, f"BPM: {tempo:.1f}")
        self._check_cancelled()

        self._report(
            ProcessingStage.TRANSCRIPTION,
            0.0,
            0.10,
            self._pt("progress.running_backend", backend=mode_label),
        )

        def _piano_cb(progress: float, msg: str) -> None:
            overall = 0.10 + progress * 0.75
            self._report(ProcessingStage.TRANSCRIPTION, progress, overall, msg)

        midi_path = transcriber.transcribe(
            audio_path=audio_path,
            output_path=midi_path,
            progress_callback=_piano_cb,
        )

        self._check_cancelled()
        self._report(
            ProcessingStage.SYNTHESIS, 1.0, 0.95, self._pt("progress.piano_midi_generated")
        )

        processing_time = time.time() - start_time
        note_count = self._count_midi_notes(midi_path)
        self._report(
            ProcessingStage.COMPLETE,
            1.0,
            1.0,
            self._pt("progress.complete_elapsed", seconds=f"{processing_time:.1f}"),
        )

        return ProcessingResult(
            midi_path=midi_path,
            tracks=[Track(type=TrackType.OTHER, audio_path=audio_path)],
            beat_info=beat_info,
            processing_time=processing_time,
            total_notes=note_count,
        )

    def _process_piano_transkun(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """TransKun 钢琴专用模式。"""
        return self._process_specialized_piano(
            audio_path=audio_path,
            output_dir=output_dir,
            transcriber=self.transkun_transcriber,
            mode_label="TransKun",
            output_suffix="piano_transkun",
            install_hint=(
                "TransKun 不可用，请先安装：\n"
                f"  python -m pip install transkun=={TRANSKUN_PACKAGE_VERSION}"
            ),
            model_hint=(
                "TransKun 预训练资源缺失或安装不完整，请执行：\n"
                "  python -m pip install --force-reinstall "
                f"transkun=={TRANSKUN_PACKAGE_VERSION}"
            ),
        )

    def _process_piano_transkun_v2_aug(
        self,
        audio_path: str,
        output_dir: str,
    ) -> ProcessingResult:
        """TransKun V2 Aug 钢琴专用模式。"""
        return self._process_specialized_piano(
            audio_path=audio_path,
            output_dir=output_dir,
            transcriber=self.transkun_v2_aug_transcriber,
            mode_label="TransKun V2 Aug",
            output_suffix="piano_transkun_v2_aug",
            install_hint=(
                "TransKun V2 Aug 运行时不可用，请先安装：\n"
                f"  python -m pip install transkun=={TRANSKUN_PACKAGE_VERSION}"
            ),
            model_hint=(
                "TransKun V2 Aug 官方 checkpoint 缺失或校验失败，请执行：\n"
                "  python download_transkun_v2_aug_model.py"
            ),
        )

    def _process_piano_aria_amt(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """Aria-AMT 钢琴专用模式。"""
        return self._process_specialized_piano(
            audio_path=audio_path,
            output_dir=output_dir,
            transcriber=self.aria_amt_transcriber,
            mode_label="Aria-AMT",
            output_suffix="piano_aria",
            install_hint=(
                "Aria-AMT 不可用，请先安装：\n"
                f'  python -m pip install "{ARIA_AMT_SOURCE_REQUIREMENT}"'
            ),
            model_hint=(
                "Aria-AMT 模型权重缺失，请先下载：\n" "  python download_aria_amt_model.py"
            ),
        )

    def _process_piano_bytedance_pedal(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """ByteDance 带踏板钢琴专用模式。"""
        return self._process_specialized_piano(
            audio_path=audio_path,
            output_dir=output_dir,
            transcriber=self.bytedance_piano_transcriber,
            mode_label="ByteDance Piano",
            output_suffix="piano_bytedance_pedal",
            install_hint=(
                "ByteDance Piano 不可用，请先安装：\n"
                "  python -m pip install "
                f"piano-transcription-inference=={BYTEDANCE_PIANO_PACKAGE_VERSION} "
                '"torchlibrosa>=0.1.0,<0.2"'
            ),
            model_hint=(
                "ByteDance Piano checkpoint 缺失或不完整，请先下载：\n"
                "  python download_bytedance_piano_model.py"
            ),
        )

    def _process_yourmt3_official(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """Run the official YourMT3 writer, then add TelkNet-aligned tempo metadata."""
        start_time = time.time()
        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(Path(audio_path).name).stem
        midi_path = str(Path(output_dir) / f"{stem}.mid")
        self._require_multi_instrument_available()

        self._report(
            ProcessingStage.PREPROCESSING,
            0.0,
            0.0,
            self._pt("progress.analyzing_audio"),
        )
        beat_info = self._detect_beat_or_raise(audio_path)
        tempo = beat_info.bpm
        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.1, f"BPM: {tempo:.1f}")
        self._check_cancelled()
        self._report(
            ProcessingStage.TRANSCRIPTION,
            0.0,
            0.1,
            self._pt("progress.loading_model", model="YourMT3+"),
        )
        self._check_cancelled()

        def _yourmt3_cb(p: float, msg: str) -> None:
            overall = 0.1 + p * 0.85
            self._report(ProcessingStage.TRANSCRIPTION, p, overall, msg)

        try:
            midi_path = self.yourmt3_transcriber.transcribe_to_midi(
                audio_path=audio_path,
                output_path=midi_path,
                progress_callback=_yourmt3_cb,
            )
            midi_path = self._normalize_midi_tempo_metadata(midi_path, tempo)
        except InterruptedError:
            raise
        except Exception as e:
            logger.error("YourMT3+ transcription failed: %s", e, exc_info=True)
            raise RuntimeError(self._format_backend_error("YourMT3+", "转写失败", e)) from e
        finally:
            self._cleanup_multi_instrument_backend()

        self._check_cancelled()
        total_notes = self._count_midi_notes(midi_path)
        self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, self._pt("progress.midi_generated"))

        processing_time = time.time() - start_time
        self._report(
            ProcessingStage.COMPLETE,
            1.0,
            1.0,
            self._pt("progress.complete_elapsed", seconds=f"{processing_time:.1f}"),
        )

        return ProcessingResult(
            midi_path=midi_path,
            tracks=[Track(type=TrackType.OTHER, audio_path=audio_path)],
            beat_info=beat_info,
            processing_time=processing_time,
            total_notes=total_notes,
        )

    def _process_miros_official(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """Run the official MIROS writer, then add TelkNet-aligned tempo metadata."""
        start_time = time.time()
        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(Path(audio_path).name).stem
        midi_path = str(Path(output_dir) / f"{stem}.mid")
        self._require_multi_instrument_available()

        self._report(
            ProcessingStage.PREPROCESSING,
            0.0,
            0.0,
            self._pt("progress.analyzing_audio"),
        )
        beat_info = self._detect_beat_or_raise(audio_path)
        tempo = beat_info.bpm
        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.1, f"BPM: {tempo:.1f}")
        self._check_cancelled()
        self._report(
            ProcessingStage.TRANSCRIPTION,
            0.0,
            0.1,
            self._pt("progress.loading_model", model="MIROS"),
        )
        self._check_cancelled()

        def _miros_cb(p: float, msg: str) -> None:
            overall = 0.1 + p * 0.85
            self._report(ProcessingStage.TRANSCRIPTION, p, overall, msg)

        try:
            midi_path = self.miros_transcriber.transcribe_to_midi(
                audio_path=audio_path,
                output_path=midi_path,
                progress_callback=_miros_cb,
            )
            midi_path = self._normalize_midi_tempo_metadata(midi_path, tempo)
        except InterruptedError:
            raise
        except Exception as e:
            logger.error("MIROS 转写失败: %s", e, exc_info=True)
            raise RuntimeError(self._format_backend_error("MIROS", "转写失败", e)) from e
        finally:
            self._cleanup_multi_instrument_backend()

        self._check_cancelled()
        total_notes = self._count_midi_notes(midi_path)
        self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, self._pt("progress.midi_generated"))

        processing_time = time.time() - start_time
        self._report(
            ProcessingStage.COMPLETE,
            1.0,
            1.0,
            self._pt("progress.complete_elapsed", seconds=f"{processing_time:.1f}"),
        )

        return ProcessingResult(
            midi_path=midi_path,
            tracks=[Track(type=TrackType.OTHER, audio_path=audio_path)],
            beat_info=beat_info,
            processing_time=processing_time,
            total_notes=total_notes,
        )

    def _process_smart(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """智能模式：直接对完整混音进行多乐器转写。"""
        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        model_name = self._get_multi_instrument_model_name()
        if model_name == MultiInstrumentModel.MIROS.value:
            return self._process_miros_official(audio_path, output_dir)
        if model_name == MultiInstrumentModel.YOURMT3.value:
            return self._process_yourmt3_official(audio_path, output_dir)
        raise ValueError(f"Unsupported multi-instrument backend: {model_name!r}")

    def _process_vocal_split(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """人声分离模式：Leap XE + PolarFormer → 分别转写伴奏和人声。"""
        from src.core.vocal_separator import VocalSeparator

        start_time = time.time()

        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(Path(audio_path).name).stem

        logger.info(f"开始处理 (人声分离模式): {audio_path}")
        transcriber = self._get_multi_instrument_transcriber()
        model_label = self._get_multi_instrument_label()

        # ── 检查依赖 ──
        logger.info("正在检查依赖: Leap XE vocals + PolarFormer accompaniment, %s...", model_label)
        if not VocalSeparator.is_available():
            raise RuntimeError(
                "人声分离不可用。请安装: pip install audio-separator==0.44.1 --no-deps"
            )
        if not VocalSeparator.is_model_available():
            raise RuntimeError(
                "人声分离模型未下载。请运行 python download_sota_models.py，"
                "或分别运行 python download_vocal_model.py 和 python download_accompaniment_model.py 后重试。"
            )
        self._require_multi_instrument_available()
        logger.info("所有依赖检查通过")

        # ── 阶段1：预处理 / 节拍检测 (0-5%) ──
        self._report(ProcessingStage.PREPROCESSING, 0.0, 0.0, self._pt("progress.analyzing_audio"))
        self._check_cancelled()
        beat_info = self._detect_beat_or_raise(audio_path)
        tempo = beat_info.bpm
        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.05, f"BPM: {tempo:.1f}")

        # ── 阶段2：Leap XE + PolarFormer 人声/伴奏分离 (5-35%) ──
        self._report(
            ProcessingStage.SEPARATION,
            0.0,
            0.05,
            self._pt("progress.separating_vocals_accompaniment"),
        )

        separator_device = "cuda:0" if self.config.use_gpu else "cpu"
        separator = VocalSeparator(
            language=getattr(self.config, "language", Translator.DEFAULT_LANGUAGE),
            primary_device=separator_device,
            accompaniment_device=separator_device,
        )
        separator.set_cancel_check(lambda: self._cancelled)

        def _sep_cb(p: float, msg: str) -> None:
            overall = 0.05 + p * 0.30
            self._report(ProcessingStage.SEPARATION, p, overall, msg)

        self._active_separator = separator
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
        finally:
            if self._active_separator is separator:
                self._active_separator = None

        if not separated or "vocals" not in separated or "accompaniment" not in separated:
            raise RuntimeError("人声分离失败: 输出不完整，缺少 vocals 或 accompaniment")

        vocals_path = separated["vocals"]
        accompaniment_path = separated["accompaniment"]
        separated = {
            "vocals": vocals_path,
            "accompaniment": accompaniment_path,
        }

        if not os.path.exists(vocals_path) or not os.path.exists(accompaniment_path):
            raise RuntimeError(
                f"分离文件不存在: vocals={os.path.exists(vocals_path)}, "
                f"accompaniment={os.path.exists(accompaniment_path)}"
            )

        logger.info(f"人声文件: {vocals_path}")
        logger.info(f"伴奏文件: {accompaniment_path}")

        self._report_output_files(
            ProcessingStage.SEPARATION,
            1.0,
            0.35,
            self._pt("progress.vocal_wav_outputs"),
            separated,
        )
        self._report(
            ProcessingStage.SEPARATION, 1.0, 0.35, self._pt("progress.vocal_separation_complete")
        )
        self._check_cancelled()

        # ── 阶段3：多乐器后端转写伴奏 (35-70%) ──
        self._report(
            ProcessingStage.TRANSCRIPTION,
            0.0,
            0.35,
            self._pt("progress.loading_model", model=model_label),
        )
        logger.info("使用 %s 转写伴奏", model_label)

        accompaniment_midi_path = str(Path(output_dir) / f"{stem}_accompaniment.mid")
        vocal_midi_path = str(Path(output_dir) / f"{stem}_vocal.mid")

        def _transcribe_cb(p: float, msg: str) -> None:
            overall = 0.35 + p * 0.25
            self._report(ProcessingStage.TRANSCRIPTION, p, overall, msg)

        try:
            accompaniment_midi_path = transcriber.transcribe_to_midi(
                audio_path=accompaniment_path,
                output_path=accompaniment_midi_path,
                progress_callback=_transcribe_cb,
            )
            accompaniment_midi_path = self._normalize_midi_tempo_metadata(
                accompaniment_midi_path,
                tempo,
            )
        except InterruptedError:
            self._cleanup_multi_instrument_backend()
            raise
        except Exception as e:
            self._cleanup_multi_instrument_backend()
            logger.error("%s 伴奏转写失败: %s", model_label, e, exc_info=True)
            raise RuntimeError(self._format_backend_error(model_label, "伴奏转写失败", e)) from e
        self._check_cancelled()
        acc_total = self._count_midi_notes(accompaniment_midi_path)
        logger.info("伴奏官方 MIDI 转写完成: %s 个音符", acc_total)

        self._report(
            ProcessingStage.TRANSCRIPTION,
            1.0,
            0.60,
            self._pt("progress.accompaniment_transcription_complete", note_count=acc_total),
        )

        # ── 阶段4：多乐器后端转写人声 (60-85%) ──
        self._report(
            ProcessingStage.VOCAL_TRANSCRIPTION,
            0.0,
            0.60,
            self._pt("progress.transcribing_vocals_with", model=model_label),
        )
        logger.info("开始 %s 人声转写: %s", model_label, vocals_path)

        def _vocal_cb(p: float, msg: str) -> None:
            overall = 0.60 + p * 0.25
            self._report(ProcessingStage.VOCAL_TRANSCRIPTION, p, overall, msg)

        try:
            vocal_midi_path = transcriber.transcribe_to_midi(
                audio_path=vocals_path,
                output_path=vocal_midi_path,
                progress_callback=_vocal_cb,
            )
            vocal_midi_path = self._normalize_midi_tempo_metadata(
                vocal_midi_path,
                tempo,
            )
        except InterruptedError:
            raise
        except Exception as e:
            logger.error("%s 人声转写失败: %s", model_label, e, exc_info=True)
            raise RuntimeError(self._format_backend_error(model_label, "人声转写失败", e)) from e
        finally:
            self._cleanup_multi_instrument_backend()

        vocal_total = self._count_midi_notes(vocal_midi_path)
        logger.info("人声官方 MIDI 转写完成: %s 个音符", vocal_total)

        self._report(
            ProcessingStage.VOCAL_TRANSCRIPTION,
            1.0,
            0.85,
            self._pt("progress.vocal_transcription_complete", note_count=vocal_total),
        )
        self._check_cancelled()

        # ── 阶段5：生成两个 MIDI 文件 (85-95%) ──
        self._check_cancelled()
        self._report(ProcessingStage.SYNTHESIS, 0.0, 0.85, self._pt("progress.generating_midi"))
        logger.info("后端官方音符已保留；仅补 tempo 元数据，未执行音符过滤或 NoteEvent 重建")

        merged_midi_path = None
        if getattr(self.config, "vocal_split_merge_midi", False):
            self._check_cancelled()
            self._report(
                ProcessingStage.SYNTHESIS,
                0.85,
                0.92,
                self._pt("progress.merging_vocal_accompaniment_midi"),
            )
            merged_midi_path = str(Path(output_dir) / f"{stem}_vocal_accompaniment_merged.mid")
            merged_midi_path = self._merge_stem_midis(
                {
                    "accompaniment": accompaniment_midi_path,
                    "vocal": vocal_midi_path,
                },
                merged_midi_path,
            )
            self._report(
                ProcessingStage.SYNTHESIS, 1.0, 0.95, self._pt("progress.midi_generated_with_merge")
            )
        else:
            self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, self._pt("progress.midi_generated"))

        final_midi_path = merged_midi_path or accompaniment_midi_path
        retained_separated_audio = self._finalize_separated_audio(separated)

        # ── 完成 ──
        processing_time = time.time() - start_time
        self._report(
            ProcessingStage.COMPLETE,
            1.0,
            1.0,
            self._pt("progress.complete_elapsed", seconds=f"{processing_time:.1f}"),
        )

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
            separated_audio=retained_separated_audio,
            merged_midi_path=merged_midi_path,
        )
