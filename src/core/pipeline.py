"""
主处理流水线。

处理模式：
1. SMART: YourMT3+ 直接对完整混音进行极致精度转写
2. VOCAL_SPLIT: BS-RoFormer 分离人声与伴奏后分别转写（可选合并 MIDI）
3. SIX_STEM_SPLIT: BS-RoFormer SW 六声部分离后分别转写并合并 MIDI
4. PIANO_TRANSKUN / PIANO_ARIA_AMT: 钢琴专用转写
"""
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Callable, Dict, List

from src.models.data_models import (
    Config, ProcessingResult, ProcessingProgress,
    ProcessingStage, Track, TrackType, ProcessingMode,
    FIXED_TRANSCRIPTION_QUALITY, MultiInstrumentModel, TranscriptionBackend,
    BeatInfo, NoteEvent,
)
from src.core.aria_amt_transcriber import AriaAmtTranscriber
from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber
from src.core.transkun_transcriber import TranskunTranscriber
from src.core.yourmt3_transcriber import YourMT3Transcriber
from src.core.beat_detector import BeatDetector
from src.core.midi_generator import MidiGenerator
from src.i18n.translator import Translator
from src.utils.gpu_utils import clear_gpu_memory
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
        self.aria_amt_transcriber = AriaAmtTranscriber(
            language=getattr(config, "language", Translator.DEFAULT_LANGUAGE)
        )
        self.bytedance_piano_transcriber = ByteDancePianoTranscriber(config)
        self.transkun_transcriber = TranskunTranscriber(config)
        self.beat_detector = BeatDetector(config)
        self.midi_generator = MidiGenerator(config)

        self._cancelled = False
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
        self.yourmt3_transcriber.cancel()
        self.miros_transcriber.cancel()
        if hasattr(self.aria_amt_transcriber, "cancel"):
            self.aria_amt_transcriber.cancel()
        if hasattr(self.bytedance_piano_transcriber, "cancel"):
            self.bytedance_piano_transcriber.cancel()
        if hasattr(self.transkun_transcriber, "cancel"):
            self.transkun_transcriber.cancel()
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
        if backend == TranscriptionBackend.ARIA_AMT.value:
            return MultiInstrumentModel.YOURMT3.value

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

    def _get_preferred_transcription_backend_name(self) -> str:
        candidates = [
            getattr(self.config, "transcription_backend", None),
            getattr(self.config, "multi_instrument_model", None),
        ]
        valid_values = {backend.value for backend in TranscriptionBackend}

        for candidate in candidates:
            configured = str(candidate or "").strip().lower()
            if configured in valid_values:
                return configured

        if any(candidate not in (None, "") for candidate in candidates):
            logger.warning(
                "Unknown preferred transcription backend (%s, %s); falling back to %s",
                getattr(self.config, "transcription_backend", None),
                getattr(self.config, "multi_instrument_model", None),
                TranscriptionBackend.ARIA_AMT.value,
            )
        return TranscriptionBackend.ARIA_AMT.value

    def _prefers_aria_amt(self) -> bool:
        return self._get_preferred_transcription_backend_name() == TranscriptionBackend.ARIA_AMT.value

    def _report_general_backend_routing(self, mode_label: str) -> None:
        if not self._prefers_aria_amt():
            return
        logger.info(
            "Preferred backend is Aria-AMT, but %s mode still uses %s for the full multi-instrument pass. "
            "Aria-AMT remains reserved for dedicated piano mode and the separated piano stem in six-stem mode.",
            mode_label,
            self._get_multi_instrument_label(),
        )

    def _can_use_aria_amt(self) -> bool:
        try:
            return (
                self.aria_amt_transcriber.is_available()
                and self.aria_amt_transcriber.is_model_available()
            )
        except Exception as exc:
            logger.debug("Failed to verify Aria-AMT availability: %s", exc)
            return False

    def _maybe_transcribe_piano_stem_with_aria(
        self,
        piano_audio_path: Optional[str],
        output_dir: str,
    ) -> Optional[str]:
        if not piano_audio_path or not self._prefers_aria_amt():
            return None
        if not self._can_use_aria_amt():
            raise RuntimeError(
                "Aria-AMT 被选为钢琴 stem 后端，但当前不可用或模型权重缺失。"
                "请安装 aria-amt 并运行 python download_aria_amt_model.py。"
            )

        piano_audio = str(piano_audio_path)
        midi_path = str(Path(output_dir) / f"{Path(piano_audio).stem}.mid")
        logger.info("Using Aria-AMT for separated piano stem: %s", piano_audio)
        return self.aria_amt_transcriber.transcribe(
            audio_path=piano_audio,
            output_path=midi_path,
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
        wav_path = str(Path(output_dir) / f"{stem}_converted.wav")

        ffmpeg_exe = get_ffmpeg_executable()
        cmd = [
            ffmpeg_exe, "-y", "-i", audio_path,
            "-ar", "44100", "-ac", "2", "-sample_fmt", "s16",
            wav_path,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
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
            raise RuntimeError(
                f"FFmpeg 转换 {Path(audio_path).name} 超时，已停止。"
            ) from exc
        except subprocess.SubprocessError as exc:
            raise RuntimeError(
                f"FFmpeg 转换 {Path(audio_path).name} 失败，已停止: {exc}"
            ) from exc

        if Path(wav_path).exists() and Path(wav_path).stat().st_size > 0:
            logger.info("已将 %s 转换为 WAV（FFmpeg）", Path(audio_path).name)
            return wav_path

        raise RuntimeError(
            f"FFmpeg 转换 {Path(audio_path).name} 未生成有效 WAV 文件，已停止。"
        )

    def _detect_beat_or_raise(self, audio_path: str) -> BeatInfo:
        try:
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
        self._cancelled = False
        self._progress_callback = progress_callback

        # 非 WAV 格式自动转换为 WAV
        audio_path = self._ensure_wav(audio_path, output_dir)

        mode = self.config.processing_mode
        if mode == "vocal_split":
            return self._process_vocal_split(audio_path, output_dir)
        if mode == "six_stem_split":
            return self._process_six_stem_split(audio_path, output_dir)
        if mode == "piano_transkun":
            return self._process_piano_transkun(audio_path, output_dir)
        if mode == "piano_aria_amt":
            return self._process_piano_aria_amt(audio_path, output_dir)
        if mode == "piano_bytedance_pedal":
            return self._process_piano_bytedance_pedal(audio_path, output_dir)
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

    @staticmethod
    def _count_midi_notes(midi_path: str) -> int:
        try:
            from mido import MidiFile

            midi = MidiFile(midi_path)
            return sum(
                1
                for track in midi.tracks
                for message in track
                if (not message.is_meta and message.type == "note_on" and message.velocity > 0)
            )
        except Exception:
            return 0

    def _apply_vocal_harmony_split(
        self,
        separated: Dict[str, str],
        selected_stems: List[str],
        output_dir: str,
    ) -> tuple[Dict[str, str], List[str]]:
        if not getattr(self.config, "six_stem_split_vocal_harmony", False):
            return separated, selected_stems

        if "vocals" not in separated or "vocals" not in selected_stems:
            return separated, selected_stems

        from src.core.vocal_harmony_separator import VocalHarmonySeparator

        if not VocalHarmonySeparator.is_available():
            raise RuntimeError(
                "已启用主唱/和声分离，但 audio-separator 不可用。"
                "请安装 audio-separator 后重试。"
            )

        if not VocalHarmonySeparator.is_model_available():
            raise RuntimeError(
                "已启用主唱/和声分离，但模型未下载。"
                "请运行 python download_vocal_harmony_model.py 后重试。"
            )

        self._report(ProcessingStage.SEPARATION, 0.85, 0.27, self._pt("progress.separating_lead_harmony"))
        harmony_separator = VocalHarmonySeparator(
            language=getattr(self.config, "language", Translator.DEFAULT_LANGUAGE)
        )
        vocal_split = harmony_separator.separate(
            audio_path=separated["vocals"],
            output_dir=output_dir,
        )

        merged_separated = dict(separated)
        merged_separated.update(vocal_split)

        expanded_stems: List[str] = []
        for stem in selected_stems:
            if stem == "vocals":
                expanded_stems.extend(["lead_vocals", "harmony_vocals"])
            else:
                expanded_stems.append(stem)

        return merged_separated, expanded_stems

    def _process_six_stem_split(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """
        六声部分离模式：
        1. BS-RoFormer SW 分离六个 stem（用于 WAV 输出）
        2. 当前多乐器后端对完整混音做一次推理
        3. 按 GM 乐器族将转写结果拆分到对应 stem MIDI
        """
        from src.core.multi_stem_separator import SixStemSeparator, STEM_KEYS
        from src.models.gm_instruments import get_instrument_family, GMFamily

        start_time = time.time()
        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        input_stem = Path(audio_path).stem

        logger.info(f"开始处理(六声部分离模式): {audio_path}")

        if not SixStemSeparator.is_available():
            raise RuntimeError("六声部分离不可用，请安装: pip install audio-separator>=0.38.0")
        self._require_multi_instrument_available()
        transcriber = self._get_multi_instrument_transcriber()
        model_label = self._get_multi_instrument_label()
        self._report_general_backend_routing("six_stem_split")

        self._report(ProcessingStage.PREPROCESSING, 0.0, 0.0, self._pt("progress.analyzing_audio"))
        self._check_cancelled()

        beat_info = self._detect_beat_or_raise(audio_path)
        tempo = beat_info.bpm

        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.05, f"BPM: {tempo:.1f}")
        self._check_cancelled()

        self._report(ProcessingStage.SEPARATION, 0.0, 0.05, self._pt("progress.six_stem_separating"))
        separator = SixStemSeparator(
            language=getattr(self.config, "language", Translator.DEFAULT_LANGUAGE)
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

        self._report(ProcessingStage.SEPARATION, 1.0, 0.30, self._pt("progress.six_stem_complete"))
        self._check_cancelled()

        self._report(
            ProcessingStage.TRANSCRIPTION,
            0.0,
            0.30,
            self._pt("progress.loading_model", model=model_label),
        )
        logger.info("使用 %s 对完整混音进行一次性多乐器转写", model_label)

        quality = FIXED_TRANSCRIPTION_QUALITY

        def _transcribe_cb(p: float, msg: str) -> None:
            overall = 0.30 + p * 0.45
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
            raise RuntimeError(
                self._format_backend_error(model_label, "转写失败", e)
            ) from e
        finally:
            self._cleanup_multi_instrument_backend()

        total_notes = sum(len(n) for n in instrument_notes.values()) + sum(
            len(n) for n in drum_notes.values()
        )
        self._report(
            ProcessingStage.TRANSCRIPTION,
            1.0,
            0.75,
            self._pt(
                "progress.transcription_complete",
                instrument_count=len(instrument_notes),
                note_count=total_notes,
            ),
        )
        self._check_cancelled()

        self._report(ProcessingStage.SYNTHESIS, 0.0, 0.75, self._pt("progress.splitting_midi_by_family"))

        family_to_stem: Dict[GMFamily, str] = {
            GMFamily.PIANO: "piano",
            GMFamily.CHROMATIC: "other",
            GMFamily.ORGAN: "other",
            GMFamily.GUITAR: "guitar",
            GMFamily.BASS: "bass",
            GMFamily.STRINGS: "other",
            GMFamily.ENSEMBLE: "vocals",
            GMFamily.BRASS: "other",
            GMFamily.REED: "other",
            GMFamily.PIPE: "other",
            GMFamily.SYNTH_LEAD: "other",
            GMFamily.SYNTH_PAD: "other",
            GMFamily.SYNTH_EFFECTS: "other",
            GMFamily.ETHNIC: "other",
            GMFamily.PERCUSSIVE: "other",
            GMFamily.SOUND_EFFECTS: "other",
        }

        stem_instrument: Dict[str, Dict[int, List]] = {s: {} for s in STEM_KEYS}
        stem_drums: Dict[str, Dict[int, List]] = {s: {} for s in STEM_KEYS}

        for program, notes in instrument_notes.items():
            family = get_instrument_family(program)
            stem = family_to_stem.get(family, "other") if family else "other"
            stem_instrument[stem][program] = notes
        stem_drums["drums"] = drum_notes

        requested_stems = [
            str(stem).lower()
            for stem in (getattr(self.config, "six_stem_targets", []) or [])
        ]
        selected_stems = [
            stem for stem in STEM_KEYS if not requested_stems or stem in requested_stems
        ] or list(STEM_KEYS)

        separated, selected_stems = self._apply_vocal_harmony_split(
            separated=separated,
            selected_stems=selected_stems,
            output_dir=output_dir,
        )
        self._check_cancelled()

        for expanded_key in ("lead_vocals", "harmony_vocals"):
            if expanded_key in selected_stems and expanded_key not in stem_instrument:
                stem_instrument[expanded_key] = dict(stem_instrument.get("vocals", {}))
                stem_drums[expanded_key] = dict(stem_drums.get("vocals", {}))

        stem_midi_paths: Dict[str, str] = {}
        total_selected = len(selected_stems)

        for stem_index, stem_name in enumerate(selected_stems):
            self._check_cancelled()

            if stem_name == "piano":
                aria_piano_midi_path = self._maybe_transcribe_piano_stem_with_aria(
                    separated.get("piano"),
                    output_dir,
                )
                if aria_piano_midi_path is not None:
                    stem_midi_paths[stem_name] = aria_piano_midi_path
                    logger.info("[piano] Using Aria-AMT specialized piano stem MIDI")
                    continue

            s_instr = stem_instrument.get(stem_name, {})
            s_drum = stem_drums.get(stem_name, {})
            stem_note_count = sum(len(v) for v in s_instr.values()) + sum(
                len(v) for v in s_drum.values()
            )

            if stem_note_count == 0:
                logger.info(f"[{stem_name}] 无音符，跳过 MIDI 生成")
                continue

            stem_midi_path = str(Path(output_dir) / f"{input_stem}_{stem_name}.mid")
            progress = 0.75 + (stem_index / total_selected) * 0.18
            self._report(
                ProcessingStage.SYNTHESIS,
                stem_index / total_selected,
                progress,
                self._pt(
                    "progress.generating_stem_midi",
                    stem=stem_name,
                    note_count=stem_note_count,
                ),
            )

            stem_midi_path = self.midi_generator.generate_from_precise_instruments_v2(
                instrument_notes=s_instr,
                drum_notes=s_drum,
                tempo=tempo,
                output_path=stem_midi_path,
                quality=quality,
            )
            stem_midi_paths[stem_name] = stem_midi_path

        self._check_cancelled()
        merged_suffix = (
            "all_stems_merged"
            if not requested_stems and not getattr(self.config, "six_stem_split_vocal_harmony", False)
            else "selected_stems_merged"
        )
        merged_midi_path = str(Path(output_dir) / f"{input_stem}_{merged_suffix}.mid")
        self._report(ProcessingStage.SYNTHESIS, 0.95, 0.93, self._pt("progress.merging_stem_midi"))
        merged_midi_path = self._merge_stem_midis(stem_midi_paths, merged_midi_path, tempo)
        self._report(ProcessingStage.SYNTHESIS, 1.0, 0.97, self._pt("progress.six_stem_midi_complete"))

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
            separated_audio=separated,
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

        self._report(ProcessingStage.PREPROCESSING, 0.0, 0.0, self._pt("progress.analyzing_piano_audio"))
        self._check_cancelled()

        beat_info = self._detect_beat_or_raise(audio_path)
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
        self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, self._pt("progress.piano_midi_generated"))

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
        """Transkun 钢琴专用模式。"""
        return self._process_specialized_piano(
            audio_path=audio_path,
            output_dir=output_dir,
            transcriber=self.transkun_transcriber,
            mode_label="Transkun",
            output_suffix="piano_transkun",
            install_hint=(
                "Transkun 不可用，请先安装：\n"
                "  python -m pip install transkun"
            ),
            model_hint=(
                "Transkun 预训练资源缺失或安装不完整，请执行：\n"
                "  python -m pip install --force-reinstall transkun"
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
                "  python -m pip install git+https://github.com/EleutherAI/aria-amt.git"
            ),
            model_hint=(
                "Aria-AMT 模型权重缺失，请先下载：\n"
                "  python download_aria_amt_model.py"
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
                "  python -m pip install piano-transcription-inference torchlibrosa"
            ),
            model_hint=(
                "ByteDance Piano checkpoint 缺失或不完整，请先下载：\n"
                "  python download_bytedance_piano_model.py"
            ),
        )

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
        self._report_general_backend_routing("smart")

        logger.info(f"开始处理 (智能模式): {audio_path}")

        # ── 检查多乐器后端是否可用 ──
        logger.info("正在检查 %s 可用性...", model_label)
        self._require_multi_instrument_available()
        logger.info("%s 可用性检查通过", model_label)

        # ── 阶段1：预处理 / 节拍检测 ──
        self._report(ProcessingStage.PREPROCESSING, 0.0, 0.0, self._pt("progress.analyzing_audio"))
        self._check_cancelled()

        beat_info = self._detect_beat_or_raise(audio_path)
        tempo = beat_info.bpm

        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.1, f"BPM: {tempo:.1f}")
        self._check_cancelled()

        # ── 阶段2：多乐器后端极致精度转写 ──
        self._report(
            ProcessingStage.TRANSCRIPTION,
            0.0,
            0.1,
            self._pt("progress.loading_model", model=model_label),
        )
        logger.info("使用 %s 进行极致精度转写", model_label)

        quality = FIXED_TRANSCRIPTION_QUALITY

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
            raise RuntimeError(
                self._format_backend_error(model_label, "转写失败", e)
            ) from e
        finally:
            self._cleanup_multi_instrument_backend()

        self._check_cancelled()

        total_notes = sum(len(n) for n in instrument_notes.values()) + \
                      sum(len(n) for n in drum_notes.values())
        logger.info(f"转写完成: {len(instrument_notes)} 种乐器, {total_notes} 个音符")

        self._report(
            ProcessingStage.TRANSCRIPTION, 1.0, 0.85,
            self._pt(
                "progress.transcription_complete",
                instrument_count=len(instrument_notes),
                note_count=total_notes,
            )
        )

        # ── 阶段3：生成 MIDI ──
        self._report(ProcessingStage.SYNTHESIS, 0.0, 0.85, self._pt("progress.generating_midi"))
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

        self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, self._pt("progress.midi_generated"))

        # ── 完成 ──
        processing_time = time.time() - start_time
        self._report(
            ProcessingStage.COMPLETE,
            1.0,
            1.0,
            self._pt("progress.complete_elapsed", seconds=f"{processing_time:.1f}"),
        )

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

        # ── 检查依赖 ──
        logger.info("正在检查依赖: BS-RoFormer, %s...", model_label)
        if not VocalSeparator.is_available():
            raise RuntimeError(
                "人声分离不可用。请安装: pip install audio-separator>=0.38.0"
            )
        self._require_multi_instrument_available()
        logger.info("所有依赖检查通过")

        # ── 阶段1：预处理 / 节拍检测 (0-5%) ──
        self._report(ProcessingStage.PREPROCESSING, 0.0, 0.0, self._pt("progress.analyzing_audio"))
        self._check_cancelled()

        beat_info = self._detect_beat_or_raise(audio_path)
        tempo = beat_info.bpm

        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.05, f"BPM: {tempo:.1f}")
        self._check_cancelled()

        # ── 阶段2：BS-RoFormer 人声分离 (5-35%) ──
        self._report(ProcessingStage.SEPARATION, 0.0, 0.05, self._pt("progress.separating_vocals_accompaniment"))

        separator = VocalSeparator(
            language=getattr(self.config, "language", Translator.DEFAULT_LANGUAGE)
        )
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

        self._report(ProcessingStage.SEPARATION, 1.0, 0.35, self._pt("progress.vocal_separation_complete"))
        self._check_cancelled()

        # ── 阶段3：多乐器后端转写伴奏 (35-70%) ──
        self._report(
            ProcessingStage.TRANSCRIPTION,
            0.0,
            0.35,
            self._pt("progress.loading_model", model=model_label),
        )
        logger.info("使用 %s 转写伴奏", model_label)

        quality = FIXED_TRANSCRIPTION_QUALITY

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
            raise RuntimeError(
                self._format_backend_error(model_label, "伴奏转写失败", e)
            ) from e
        finally:
            self._cleanup_multi_instrument_backend()

        self._check_cancelled()

        acc_total = sum(len(n) for n in instrument_notes.values()) + \
                    sum(len(n) for n in drum_notes.values())
        logger.info(f"伴奏转写完成: {len(instrument_notes)} 种乐器, {acc_total} 个音符")

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
            vocal_instrument_notes, vocal_drum_notes = transcriber.transcribe_precise(
                audio_path=vocals_path,
                quality=quality,
                progress_callback=_vocal_cb,
            )
        except InterruptedError:
            raise
        except Exception as e:
            logger.error("%s 人声转写失败: %s", model_label, e, exc_info=True)
            raise RuntimeError(
                self._format_backend_error(model_label, "人声转写失败", e)
            ) from e
        finally:
            self._cleanup_multi_instrument_backend()

        raw_inst_count = len(vocal_instrument_notes)
        raw_note_count = sum(len(n) for n in vocal_instrument_notes.values())
        raw_drum_count = sum(len(n) for n in vocal_drum_notes.values())

        vocal_instrument_notes, vocal_drum_notes = self._filter_vocal_melody_notes(
            vocal_instrument_notes,
            vocal_drum_notes,
            preserve_isolated_vocal_stem=True,
        )

        kept = sum(len(n) for n in vocal_instrument_notes.values())
        logger.info(
            f"人声过滤: 保留人声 {kept} 个音符, "
            f"丢弃 {raw_inst_count - len(vocal_instrument_notes)} 种乐器 "
            f"({raw_note_count - kept} 个音符) + {raw_drum_count} 个鼓音符"
        )

        vocal_total = kept
        logger.info(f"人声转写完成: {len(vocal_instrument_notes)} 种乐器, {vocal_total} 个音符")

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
            self._report(ProcessingStage.SYNTHESIS, 0.85, 0.92, self._pt("progress.merging_vocal_accompaniment_midi"))
            merged_midi_path = str(Path(output_dir) / f"{stem}_vocal_accompaniment_merged.mid")
            merged_midi_path = self._merge_stem_midis(
                {
                    "accompaniment": accompaniment_midi_path,
                    "vocal": vocal_midi_path,
                },
                merged_midi_path,
                tempo,
            )
            self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, self._pt("progress.midi_generated_with_merge"))
        else:
            self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, self._pt("progress.midi_generated"))

        final_midi_path = merged_midi_path or accompaniment_midi_path

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
            separated_audio=separated,
            merged_midi_path=merged_midi_path,
        )
