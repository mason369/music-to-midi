"""
音乐转MIDI应用的数据模型
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path


class TrackType(Enum):
    """音源分离后的音轨类型"""
    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    OTHER = "other"


class ProcessingStage(Enum):
    """处理流水线中的阶段"""
    PREPROCESSING = "preprocessing"
    SEPARATION = "separation"
    TRANSCRIPTION = "transcription"
    LYRICS = "lyrics"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"


@dataclass
class NoteEvent:
    """表示一个MIDI音符事件"""
    pitch: int           # MIDI音高 (0-127)
    start_time: float    # 开始时间（秒）
    end_time: float      # 结束时间（秒）
    velocity: int = 80   # 音符力度 (0-127)

    @property
    def duration(self) -> float:
        """音符持续时间（秒）"""
        return self.end_time - self.start_time


@dataclass
class LyricEvent:
    """表示带时间戳的歌词事件"""
    text: str            # 歌词文本（单词/音节）
    start_time: float    # 开始时间（秒）
    end_time: float      # 结束时间（秒）
    confidence: float = 1.0  # 识别置信度


@dataclass
class BeatInfo:
    """节拍和速度信息"""
    bpm: float                                # 每分钟节拍数
    beat_times: List[float] = field(default_factory=list)  # 所有节拍时间
    downbeats: Optional[List[float]] = None  # 重拍时间
    time_signature: tuple = (4, 4)           # 拍号（分子/分母）


@dataclass
class Track:
    """表示一个分离的音轨"""
    type: TrackType
    audio_path: str              # 分离音频文件路径
    notes: List[NoteEvent] = field(default_factory=list)

    @property
    def note_count(self) -> int:
        return len(self.notes)


@dataclass
class ProcessingProgress:
    """处理过程中的进度信息"""
    stage: ProcessingStage       # 当前处理阶段
    stage_progress: float        # 当前阶段进度 (0-1)
    overall_progress: float      # 总体进度 (0-1)
    message: str                 # 状态消息

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "stage_progress": self.stage_progress,
            "overall_progress": self.overall_progress,
            "message": self.message
        }


@dataclass
class ProcessingResult:
    """完整处理流水线的结果"""
    midi_path: str                           # 输出MIDI文件路径
    lrc_path: Optional[str]                  # LRC歌词文件路径
    tracks: List[Track] = field(default_factory=list)
    beat_info: Optional[BeatInfo] = None
    lyrics: List[LyricEvent] = field(default_factory=list)
    processing_time: float = 0.0             # 总处理时间（秒）

    @property
    def has_lyrics(self) -> bool:
        return len(self.lyrics) > 0


@dataclass
class Config:
    """应用配置"""
    # 常规设置
    language: str = "zh_CN"
    theme: str = "dark"

    # 处理设置
    use_gpu: bool = True
    gpu_device: int = 0
    segment_size: float = 7.8  # Demucs分段大小（内存优化）

    # Whisper设置
    whisper_model: str = "medium"  # tiny, base, small, medium, large
    lyrics_language: Optional[str] = None  # None = 自动检测

    # MIDI设置
    ticks_per_beat: int = 480
    default_velocity: int = 80
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_length: int = 58  # 毫秒

    # 输出设置
    output_dir: str = ""
    save_separated_tracks: bool = True
    export_lrc: bool = True
    embed_lyrics: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "theme": self.theme,
            "use_gpu": self.use_gpu,
            "gpu_device": self.gpu_device,
            "segment_size": self.segment_size,
            "whisper_model": self.whisper_model,
            "lyrics_language": self.lyrics_language,
            "ticks_per_beat": self.ticks_per_beat,
            "default_velocity": self.default_velocity,
            "onset_threshold": self.onset_threshold,
            "frame_threshold": self.frame_threshold,
            "min_note_length": self.min_note_length,
            "output_dir": self.output_dir,
            "save_separated_tracks": self.save_separated_tracks,
            "export_lrc": self.export_lrc,
            "embed_lyrics": self.embed_lyrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class Project:
    """表示一个转换项目"""
    input_path: str                          # 输入音频文件路径
    output_dir: str                          # 输出目录
    config: Config = field(default_factory=Config)
    result: Optional[ProcessingResult] = None

    @property
    def input_filename(self) -> str:
        return Path(self.input_path).stem

    @property
    def is_complete(self) -> bool:
        return self.result is not None
