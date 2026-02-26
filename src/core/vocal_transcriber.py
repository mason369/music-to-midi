"""
人声转写模块 - 使用 CREPE 音高追踪将人声音频转为 MIDI 音符
"""
import logging
from typing import Optional, Callable, List

import numpy as np

from src.models.data_models import NoteEvent

logger = logging.getLogger(__name__)

# 人声 MIDI 音色：Acoustic Grand Piano (GM program 0)
VOCAL_PROGRAM = 0


class VocalTranscriber:
    """
    使用 torchcrepe 对人声音频进行 F0 音高追踪，
    然后通过音符分割算法将连续帧转为离散 MIDI 音符。
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            from src.utils.gpu_utils import get_device
            device = get_device()
        self._device = device
        self._cancelled = False
        self._cancel_check: Optional[Callable[[], bool]] = None

    def set_cancel_check(self, fn: Callable[[], bool]) -> None:
        self._cancel_check = fn

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("用户取消了处理")
        if self._cancel_check and self._cancel_check():
            raise InterruptedError("用户取消了处理")

    def cancel(self) -> None:
        self._cancelled = True

    @staticmethod
    def is_available() -> bool:
        try:
            import torchcrepe
            return True
        except ImportError:
            return False

    def transcribe(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        confidence_threshold: float = 0.6,
        min_note_duration: float = 0.05,
        pitch_jump_threshold: float = 1.0,
    ) -> List[NoteEvent]:
        """
        对人声音频进行音高追踪并转为 MIDI 音符。

        参数:
            audio_path: 人声 WAV 文件路径
            progress_callback: 进度回调 (progress 0-1, message)
            confidence_threshold: 置信度阈值，低于此值视为无声
            min_note_duration: 最短音符时长（秒），过滤噪音
            pitch_jump_threshold: 音高跳变阈值（半音），超过则分割新音符

        返回:
            List[NoteEvent]，program=52 (Choir Aahs)
        """
        import torch
        import torchcrepe
        import soundfile as sf
        from src.utils.gpu_utils import clear_gpu_memory, get_available_memory_gb, get_optimal_thread_count

        # 确保 CPU 模式下充分利用多核
        if self._device == "cpu" or self._device.startswith("cpu"):
            optimal_threads = get_optimal_thread_count()
            torch.set_num_threads(optimal_threads)
            logger.info(f"CREPE CPU 线程数: {optimal_threads}")

        self._cancelled = False

        if progress_callback:
            progress_callback(0.0, "正在加载人声音频...")

        # 加载音频
        audio_np, sr = sf.read(audio_path, dtype="float32")

        # 转为单声道
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)

        # torchcrepe 需要 16kHz
        if sr != 16000:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
            sr = 16000

        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).to(self._device)

        self._check_cancelled()

        if progress_callback:
            progress_callback(0.2, "正在进行 CREPE 音高追踪...")

        logger.info(f"CREPE 音高追踪: {audio_path}, 设备: {self._device}")

        # 10ms hop_length → 每帧 0.01 秒
        hop_length = 160  # 16000 * 0.01

        # 根据可用内存动态计算 CREPE batch_size
        available_mem = get_available_memory_gb()
        if available_mem >= 4:
            crepe_batch_size = 2048
        elif available_mem >= 2:
            crepe_batch_size = 1024
        else:
            crepe_batch_size = 512
        logger.info(f"CREPE batch_size={crepe_batch_size}（可用内存={available_mem:.1f}GB）")

        try:
            pitch, periodicity = torchcrepe.predict(
                audio_tensor,
                sample_rate=sr,
                hop_length=hop_length,
                fmin=50,
                fmax=1100,
                model="full",
                return_periodicity=True,
                batch_size=crepe_batch_size,
                device=self._device,
            )
        except Exception as e:
            logger.error(f"CREPE 音高追踪失败: {e}")
            raise RuntimeError(f"CREPE 音高追踪失败: {e}") from e

        self._check_cancelled()

        if progress_callback:
            progress_callback(0.7, "正在分割音符...")

        # 转为 numpy
        pitch_np = pitch.squeeze().cpu().numpy()       # Hz per frame
        conf_np = periodicity.squeeze().cpu().numpy()   # confidence per frame

        # 帧时间分辨率
        frame_duration = hop_length / sr  # 0.01s

        # 频率转 MIDI 音高（连续值）
        # MIDI = 69 + 12 * log2(f / 440)
        # 注意：0 Hz 帧会产生 -inf，需要一并处理
        with np.errstate(divide="ignore", invalid="ignore"):
            midi_pitch = 69.0 + 12.0 * np.log2(pitch_np / 440.0)
        midi_pitch = np.nan_to_num(midi_pitch, nan=0.0, posinf=127.0, neginf=0.0)

        # 标记有声帧
        voiced = conf_np >= confidence_threshold

        # 平滑置信度：填补因人声分离不纯净导致的短暂置信度下降（≤80ms 的间隙）
        gap_fill_frames = int(0.08 / frame_duration)  # 80ms
        voiced = self._fill_short_gaps(voiced, gap_fill_frames)

        # 音符分割
        notes = self._segment_notes(
            midi_pitch=midi_pitch,
            voiced=voiced,
            frame_duration=frame_duration,
            pitch_jump_threshold=pitch_jump_threshold,
            min_note_duration=min_note_duration,
            audio_np=audio_np,
            sr=sr,
            hop_length=hop_length,
        )

        # 释放显存
        del audio_tensor, pitch, periodicity
        clear_gpu_memory()

        if progress_callback:
            progress_callback(1.0, f"人声转写完成：{len(notes)} 个音符")

        logger.info(f"人声转写完成: {len(notes)} 个音符")
        return notes

    def _segment_notes(
        self,
        midi_pitch: "np.ndarray",
        voiced: "np.ndarray",
        frame_duration: float,
        pitch_jump_threshold: float,
        min_note_duration: float,
        audio_np: Optional["np.ndarray"],
        sr: int,
        hop_length: int,
    ) -> List[NoteEvent]:
        """
        将逐帧 F0 数据分割为离散音符。

        分割条件：
        1. 有声 → 无声 或 无声 → 有声
        2. 相邻帧音高跳变 > pitch_jump_threshold 半音
        """
        notes: List[NoteEvent] = []
        n_frames = len(midi_pitch)

        if n_frames == 0:
            return notes

        # 预计算每帧 RMS 能量（用于估算 velocity）
        rms_per_frame = None
        if audio_np is not None:
            rms_per_frame = self._compute_frame_rms(audio_np, hop_length, n_frames)

        # 扫描帧，分割音符段
        in_note = False
        note_start_frame = 0
        note_pitches: List[float] = []

        for i in range(n_frames):
            if voiced[i]:
                if not in_note:
                    # 新音符开始
                    in_note = True
                    note_start_frame = i
                    note_pitches = [midi_pitch[i]]
                else:
                    # 检查音高跳变
                    jump = abs(midi_pitch[i] - midi_pitch[i - 1])
                    if jump > pitch_jump_threshold:
                        # 结束当前音符，开始新音符
                        note = self._make_note(
                            note_start_frame, i, note_pitches,
                            frame_duration, min_note_duration,
                            rms_per_frame,
                        )
                        if note:
                            notes.append(note)
                        note_start_frame = i
                        note_pitches = [midi_pitch[i]]
                    else:
                        note_pitches.append(midi_pitch[i])
            else:
                if in_note:
                    # 音符结束
                    note = self._make_note(
                        note_start_frame, i, note_pitches,
                        frame_duration, min_note_duration,
                        rms_per_frame,
                    )
                    if note:
                        notes.append(note)
                    in_note = False
                    note_pitches = []

        # 处理末尾未关闭的音符
        if in_note and note_pitches:
            note = self._make_note(
                note_start_frame, n_frames, note_pitches,
                frame_duration, min_note_duration,
                rms_per_frame,
            )
            if note:
                notes.append(note)

        # 合并相邻同音高音符（间隔 ≤ 100ms 且音高相同视为同一音符）
        notes = self._merge_nearby_notes(notes, max_gap=0.10)

        return notes

    @staticmethod
    def _fill_short_gaps(voiced: "np.ndarray", max_gap_frames: int) -> "np.ndarray":
        """填补短暂的无声间隙，避免人声分离不纯净导致的断音"""
        result = voiced.copy()
        n = len(result)
        i = 0
        while i < n:
            if not result[i]:
                # 找到无声段的结束位置
                j = i
                while j < n and not result[j]:
                    j += 1
                gap_len = j - i
                # 如果间隙足够短且两侧都有声，则填补
                if gap_len <= max_gap_frames and i > 0 and j < n:
                    result[i:j] = True
                i = j
            else:
                i += 1
        return result

    @staticmethod
    def _merge_nearby_notes(notes: List["NoteEvent"], max_gap: float = 0.10) -> List["NoteEvent"]:
        """合并相邻的同音高音符，消除因短暂中断产生的断音"""
        if len(notes) <= 1:
            return notes

        merged = [notes[0]]
        for note in notes[1:]:
            prev = merged[-1]
            gap = note.start_time - prev.end_time
            # 同音高且间隔很短 → 合并
            if note.pitch == prev.pitch and gap <= max_gap:
                # 扩展前一个音符的结束时间，取较大的 velocity
                merged[-1] = NoteEvent(
                    pitch=prev.pitch,
                    start_time=prev.start_time,
                    end_time=note.end_time,
                    velocity=max(prev.velocity, note.velocity),
                    program=prev.program,
                )
            else:
                merged.append(note)
        return merged

    def _make_note(
        self,
        start_frame: int,
        end_frame: int,
        pitches: List[float],
        frame_duration: float,
        min_note_duration: float,
        rms_per_frame: Optional["np.ndarray"],
    ) -> Optional[NoteEvent]:
        """从帧范围和音高列表创建一个 NoteEvent"""
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration
        duration = end_time - start_time

        if duration < min_note_duration:
            return None

        # 音高 = 帧内 MIDI 音高的中位数，四舍五入
        median_pitch = int(round(float(np.median(pitches))))
        median_pitch = max(0, min(127, median_pitch))

        # velocity 从 RMS 能量估算
        if rms_per_frame is not None and len(rms_per_frame) > start_frame:
            seg = rms_per_frame[start_frame:end_frame]
            if len(seg) > 0:
                mean_rms = float(np.mean(seg))
                # 映射 RMS 到 velocity (1-127)
                # 典型人声 RMS 范围 0.01 ~ 0.3
                velocity = int(np.clip(mean_rms / 0.25 * 100, 30, 127))
            else:
                velocity = 80
        else:
            velocity = 80

        return NoteEvent(
            pitch=median_pitch,
            start_time=start_time,
            end_time=end_time,
            velocity=velocity,
            program=VOCAL_PROGRAM,
        )

    @staticmethod
    def _compute_frame_rms(
        audio: "np.ndarray",
        hop_length: int,
        n_frames: int,
    ) -> "np.ndarray":
        """计算每帧的 RMS 能量"""
        rms = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_length
            end = start + hop_length
            if end > len(audio):
                end = len(audio)
            if start >= len(audio):
                break
            seg = audio[start:end]
            rms[i] = float(np.sqrt(np.mean(seg ** 2)))
        return rms
