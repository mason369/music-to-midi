"""
主处理流水线。

处理模式：
1. SMART: YourMT3+ 直接对完整混音进行极致精度转写
2. VOCAL_SPLIT: BS-RoFormer 分离人声与伴奏，均使用 YourMT3+ 分别转写（可选合并 MIDI）
3. SIX_STEM_SPLIT: BS-RoFormer SW 分离六个 stem，分别转写并合并 MIDI
4. PIANO_ARIA_AMT: Aria-AMT 钢琴专用转写
"""
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Callable, Dict, List

from src.models.data_models import (
    Config, ProcessingResult, ProcessingProgress,
    ProcessingStage, Track, TrackType, ProcessingMode
)
from src.core.aria_amt_transcriber import AriaAmtTranscriber
from src.core.yourmt3_transcriber import YourMT3Transcriber
from src.core.beat_detector import BeatDetector
from src.core.midi_generator import MidiGenerator
from src.utils.gpu_utils import clear_gpu_memory
from src.utils.runtime_paths import get_ffmpeg_executable

logger = logging.getLogger(__name__)


class MusicToMidiPipeline:
    """
    音乐转MIDI主处理流水线

    处理模式：
        SMART: 使用 YourMT3+ MoE 直接对完整混音进行多乐器转写。
        VOCAL_SPLIT: BS-RoFormer 分离人声与伴奏，均使用 YourMT3+ 分别转写。
                     可选额外输出一个“人声+伴奏合并 MIDI”。
        SIX_STEM_SPLIT: BS-RoFormer SW 分离六个 stem，可选仅转写选中 stem。
    """

    def __init__(self, config: Config):
        self.config = config
        self.yourmt3_transcriber = YourMT3Transcriber(config)
        self.aria_amt_transcriber = AriaAmtTranscriber()
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

    @staticmethod
    def _require_yourmt3_available() -> None:
        if YourMT3Transcriber.is_available():
            return

        reason_getter = getattr(YourMT3Transcriber, "get_unavailable_reason", None)
        if callable(reason_getter):
            raise RuntimeError(reason_getter())

        raise RuntimeError(
            "YourMT3+ 不可用。\n\n"
            "请先下载模型权重：\n"
            "  python download_sota_models.py\n\n"
            "详见 README.md 中的安装说明。"
        )

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
        if mode == "six_stem_split":
            return self._process_six_stem_split(audio_path, output_dir)
        if mode == "piano_aria_amt":
            return self._process_piano_aria_amt(audio_path, output_dir)
        else:
            return self._process_smart(audio_path, output_dir)

    @staticmethod
    def _has_note_messages(track) -> bool:
        return any(
            not message.is_meta and message.type in {"note_on", "note_off"}
            for message in track
        )

    def _merge_stem_midis(
        self,
        stem_midi_paths: Dict[str, str],
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

        for stem_name, midi_path in stem_midi_paths.items():
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
                    dst_track.insert(0, MetaMessage("track_name", name=stem_name, time=0))
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
            logger.warning("主唱/和声分离不可用，回退为 vocals 单 stem 转写")
            self._report(ProcessingStage.SEPARATION, 0.90, 0.28,
                         "⚠ 主唱/和声分离库不可用，回退为 vocals 单 stem 转写")
            return separated, selected_stems

        if not VocalHarmonySeparator.is_model_available():
            logger.warning("主唱/和声分离模型未下载，回退为 vocals 单 stem 转写")
            self._report(ProcessingStage.SEPARATION, 0.90, 0.28,
                         "⚠ 主唱/和声分离模型未下载，请运行 python download_vocal_harmony_model.py")
            return separated, selected_stems

        self._report(ProcessingStage.SEPARATION, 0.85, 0.27, "正在分离主唱与和声...")
        harmony_separator = VocalHarmonySeparator()
        try:
            vocal_split = harmony_separator.separate(
                audio_path=separated["vocals"],
                output_dir=output_dir,
            )
        except Exception as exc:
            logger.warning("主唱/和声分离失败，回退为 vocals 单 stem: %s", exc)
            self._report(ProcessingStage.SEPARATION, 0.90, 0.28,
                         f"⚠ 主唱/和声分离失败（{exc}），回退为 vocals 单 stem 转写")
            return separated, selected_stems

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
        六声部分离模式（修正架构）：
        1. BS-RoFormer SW 分离六个 stem（仅用于 WAV 输出）
        2. YourMT3+ 对 **完整混音** 做一次推理（模型设计如此）
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
        self._require_yourmt3_available()

        # ── 阶段0：节拍检测 ──
        self._report(ProcessingStage.PREPROCESSING, 0.0, 0.0, "正在分析音频...")
        self._check_cancelled()

        beat_info = None
        try:
            beat_info = self.beat_detector.detect(audio_path)
            tempo = beat_info.bpm if beat_info else 120.0
        except Exception as exc:
            logger.warning(f"节拍检测失败，使用默认 120 BPM: {exc}")
            tempo = 120.0

        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.05, f"BPM: {tempo:.1f}")
        self._check_cancelled()

        # ── 阶段1：六声部音源分离（WAV 输出用） ──
        self._report(ProcessingStage.SEPARATION, 0.0, 0.05, "正在进行六声部分离...")
        separator = SixStemSeparator()

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

        self._report(ProcessingStage.SEPARATION, 1.0, 0.30, "六声部分离完成")
        self._check_cancelled()

        # ── 阶段2：YourMT3+ 对完整混音做一次转写 ──
        self._report(ProcessingStage.TRANSCRIPTION, 0.0, 0.30, "正在加载 YourMT3+ 模型...")
        logger.info("使用 YourMT3+ 对完整混音进行一次性多乐器转写")

        quality = self.config.transcription_quality

        def _transcribe_cb(p: float, msg: str) -> None:
            overall = 0.30 + p * 0.45
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
            except Exception:
                pass
            try:
                clear_gpu_memory()
            except Exception:
                pass

        total_notes = sum(len(n) for n in instrument_notes.values()) + \
                      sum(len(n) for n in drum_notes.values())
        logger.info(
            f"全混音转写完成: {len(instrument_notes)} 种乐器, "
            f"{len(drum_notes)} 种鼓音色, 共 {total_notes} 个音符"
        )
        self._report(
            ProcessingStage.TRANSCRIPTION, 1.0, 0.75,
            f"转写完成：{len(instrument_notes)} 种乐器，{total_notes} 个音符",
        )
        self._check_cancelled()

        # ── 阶段3：按 GM 乐器族拆分到 stem ──
        self._report(ProcessingStage.SYNTHESIS, 0.0, 0.75, "正在按乐器族拆分 MIDI...")

        # GM 乐器族 → stem 名称映射
        _FAMILY_TO_STEM: Dict[GMFamily, str] = {
            GMFamily.PIANO: "piano",
            GMFamily.CHROMATIC: "other",
            GMFamily.ORGAN: "other",
            GMFamily.GUITAR: "guitar",
            GMFamily.BASS: "bass",
            GMFamily.STRINGS: "other",
            GMFamily.ENSEMBLE: "vocals",   # 含人声合唱
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

        # 按 stem 分桶
        stem_instrument: Dict[str, Dict[int, List]] = {s: {} for s in STEM_KEYS}
        stem_drums: Dict[str, Dict[int, List]] = {s: {} for s in STEM_KEYS}

        for program, notes in instrument_notes.items():
            family = get_instrument_family(program)
            stem = _FAMILY_TO_STEM.get(family, "other") if family else "other"
            stem_instrument[stem][program] = notes

        # 鼓音符全部归入 drums stem
        stem_drums["drums"] = drum_notes

        # 确定要输出的 stem
        requested_stems = [
            str(stem).lower()
            for stem in (getattr(self.config, "six_stem_targets", []) or [])
        ]
        selected_stems = []
        for stem in STEM_KEYS:
            if not requested_stems or stem in requested_stems:
                selected_stems.append(stem)
        if not selected_stems:
            selected_stems = list(STEM_KEYS)

        # 人声和声分离（如果启用）
        separated, selected_stems = self._apply_vocal_harmony_split(
            separated=separated,
            selected_stems=selected_stems,
            output_dir=output_dir,
        )
        self._check_cancelled()

        # 如果 vocal harmony split 将 "vocals" 拆成了 "lead_vocals"/"harmony_vocals"，
        # 需要把 stem_instrument["vocals"] 的 MIDI 音符复制到新 stem 名下
        # （YourMT3+ 无法区分主唱与和声，两者共享同一份 MIDI）
        for expanded_key in ("lead_vocals", "harmony_vocals"):
            if expanded_key in selected_stems and expanded_key not in stem_instrument:
                stem_instrument[expanded_key] = dict(stem_instrument.get("vocals", {}))
                stem_drums[expanded_key] = dict(stem_drums.get("vocals", {}))

        # ── 阶段4：为每个 stem 生成 MIDI ──
        stem_midi_paths: Dict[str, str] = {}
        total_selected = len(selected_stems)

        for stem_index, stem_name in enumerate(selected_stems):
            self._check_cancelled()

            s_instr = stem_instrument.get(stem_name, {})
            s_drum = stem_drums.get(stem_name, {})
            stem_note_count = sum(len(v) for v in s_instr.values()) + \
                              sum(len(v) for v in s_drum.values())

            if stem_note_count == 0:
                logger.info(f"[{stem_name}] 无音符，跳过 MIDI 生成")
                continue

            stem_midi_path = str(Path(output_dir) / f"{input_stem}_{stem_name}.mid")
            progress = 0.75 + (stem_index / total_selected) * 0.18
            self._report(
                ProcessingStage.SYNTHESIS,
                stem_index / total_selected,
                progress,
                f"正在生成 {stem_name} MIDI ({stem_note_count} 个音符)...",
            )

            logger.info(
                f"[{stem_name}] 乐器: {len(s_instr)} 种, "
                f"鼓: {len(s_drum)} 种, 共 {stem_note_count} 个音符"
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
        self._report(ProcessingStage.SYNTHESIS, 0.95, 0.93, "正在合并 stem MIDI...")
        merged_midi_path = self._merge_stem_midis(stem_midi_paths, merged_midi_path, tempo)
        self._report(ProcessingStage.SYNTHESIS, 1.0, 0.97, "六声部 MIDI 生成完成")

        processing_time = time.time() - start_time
        self._report(
            ProcessingStage.COMPLETE,
            1.0,
            1.0,
            f"处理完成，耗时 {processing_time:.1f} 秒",
        )

        logger.info(
            f"六声部分离处理完成: merged={merged_midi_path}, stems={len(stem_midi_paths)}"
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

    def _process_piano_aria_amt(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """Aria-AMT 钢琴专用模式。"""
        start_time = time.time()

        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(Path(audio_path).name).stem
        midi_path = str(Path(output_dir) / f"{stem}_piano_aria.mid")

        logger.info("开始处理 (Aria-AMT 钢琴模式): %s", audio_path)

        if not self.aria_amt_transcriber.is_available():
            raise RuntimeError(
                "Aria-AMT 不可用，请先安装：\n"
                "  python -m pip install git+https://github.com/EleutherAI/aria-amt.git"
            )
        if not self.aria_amt_transcriber.is_model_available():
            raise RuntimeError(
                "Aria-AMT 模型权重缺失，请先下载：\n"
                "  python download_aria_amt_model.py"
            )

        self._report(ProcessingStage.PREPROCESSING, 0.0, 0.0, "正在分析钢琴音频...")
        self._check_cancelled()

        beat_info = None
        try:
            beat_info = self.beat_detector.detect(audio_path)
            tempo = beat_info.bpm if beat_info else 120.0
        except Exception as exc:
            logger.warning("节拍检测失败，使用默认 120 BPM: %s", exc)
            tempo = 120.0

        self._report(ProcessingStage.PREPROCESSING, 1.0, 0.10, f"BPM: {tempo:.1f}")
        self._check_cancelled()

        self._report(ProcessingStage.TRANSCRIPTION, 0.0, 0.10, "正在运行 Aria-AMT...")

        def _aria_cb(progress: float, msg: str) -> None:
            overall = 0.10 + progress * 0.75
            self._report(ProcessingStage.TRANSCRIPTION, progress, overall, msg)

        midi_path = self.aria_amt_transcriber.transcribe(
            audio_path=audio_path,
            output_path=midi_path,
            progress_callback=_aria_cb,
        )

        self._check_cancelled()
        self._report(ProcessingStage.SYNTHESIS, 1.0, 0.95, "钢琴 MIDI 文件已生成")

        processing_time = time.time() - start_time
        note_count = self._count_midi_notes(midi_path)
        self._report(ProcessingStage.COMPLETE, 1.0, 1.0, f"处理完成，耗时 {processing_time:.1f} 秒")

        return ProcessingResult(
            midi_path=midi_path,
            tracks=[Track(type=TrackType.OTHER, audio_path=audio_path)],
            beat_info=beat_info,
            processing_time=processing_time,
            total_notes=note_count,
        )

    def _process_smart(self, audio_path: str, output_dir: str) -> ProcessingResult:
        """YourMT3+ 智能模式：直接对完整混音进行多乐器转写"""
        start_time = time.time()

        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(Path(audio_path).name).stem
        midi_path = str(Path(output_dir) / f"{stem}.mid")

        logger.info(f"开始处理 (智能模式): {audio_path}")

        # ── 检查 YourMT3+ 是否可用 ──
        logger.info("正在检查 YourMT3+ 可用性...")
        self._require_yourmt3_available()
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
        """人声分离模式：BS-RoFormer 分离 → YourMT3+ 分别转写伴奏和人声"""
        from src.core.vocal_separator import VocalSeparator

        start_time = time.time()

        audio_path = str(audio_path)
        output_dir = str(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(Path(audio_path).name).stem

        logger.info(f"开始处理 (人声分离模式): {audio_path}")

        # ── 检查依赖 ──
        logger.info("正在检查依赖: BS-RoFormer, YourMT3+...")
        if not VocalSeparator.is_available():
            raise RuntimeError(
                "人声分离不可用。请安装: pip install audio-separator>=0.38.0"
            )
        self._require_yourmt3_available()
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
