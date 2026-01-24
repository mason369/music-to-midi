"""
音乐转MIDI应用的数据模型
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path


class TrackType(Enum):
    """音源分离后的音轨类型 (已弃用，请使用 InstrumentType)"""
    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    OTHER = "other"


class InstrumentType(Enum):
    """乐器类型枚举"""
    PIANO = "piano"
    DRUMS = "drums"
    BASS = "bass"
    GUITAR = "guitar"
    VOCALS = "vocals"
    STRINGS = "strings"
    BRASS = "brass"       # 铜管乐器（小号、长号等）
    WOODWIND = "woodwind" # 木管乐器（长笛、萨克斯等）
    SYNTH = "synth"       # 合成器
    ORGAN = "organ"       # 风琴
    HARP = "harp"         # 竖琴
    OTHER = "other"

    @classmethod
    def from_track_type(cls, track_type: TrackType) -> "InstrumentType":
        """从旧的 TrackType 转换"""
        mapping = {
            TrackType.DRUMS: cls.DRUMS,
            TrackType.BASS: cls.BASS,
            TrackType.VOCALS: cls.VOCALS,
            TrackType.OTHER: cls.OTHER,
        }
        return mapping.get(track_type, cls.OTHER)

    def to_program_number(self) -> int:
        """获取 General MIDI 音色编号"""
        programs = {
            InstrumentType.PIANO: 0,      # Acoustic Grand Piano
            InstrumentType.DRUMS: 0,      # 鼓使用通道10，不需要音色
            InstrumentType.BASS: 32,      # Acoustic Bass
            InstrumentType.GUITAR: 24,    # Acoustic Guitar (nylon)
            InstrumentType.VOCALS: 52,    # Choir Aahs
            InstrumentType.STRINGS: 48,   # String Ensemble 1
            InstrumentType.BRASS: 56,     # Trumpet
            InstrumentType.WOODWIND: 73,  # Flute
            InstrumentType.SYNTH: 80,     # Lead 1 (square)
            InstrumentType.ORGAN: 16,     # Drawbar Organ
            InstrumentType.HARP: 46,      # Orchestral Harp
            InstrumentType.OTHER: 0,      # 默认钢琴
        }
        return programs.get(self, 0)

    def get_display_name(self, lang: str = "zh_CN") -> str:
        """获取显示名称"""
        names_zh = {
            InstrumentType.PIANO: "钢琴",
            InstrumentType.DRUMS: "鼓",
            InstrumentType.BASS: "贝斯",
            InstrumentType.GUITAR: "吉他",
            InstrumentType.VOCALS: "人声",
            InstrumentType.STRINGS: "弦乐",
            InstrumentType.BRASS: "铜管",
            InstrumentType.WOODWIND: "木管",
            InstrumentType.SYNTH: "合成器",
            InstrumentType.ORGAN: "风琴",
            InstrumentType.HARP: "竖琴",
            InstrumentType.OTHER: "其他",
        }
        names_en = {
            InstrumentType.PIANO: "Piano",
            InstrumentType.DRUMS: "Drums",
            InstrumentType.BASS: "Bass",
            InstrumentType.GUITAR: "Guitar",
            InstrumentType.VOCALS: "Vocals",
            InstrumentType.STRINGS: "Strings",
            InstrumentType.BRASS: "Brass",
            InstrumentType.WOODWIND: "Woodwind",
            InstrumentType.SYNTH: "Synth",
            InstrumentType.ORGAN: "Organ",
            InstrumentType.HARP: "Harp",
            InstrumentType.OTHER: "Other",
        }
        if lang.startswith("zh"):
            return names_zh.get(self, self.value)
        return names_en.get(self, self.value)

    def get_stem_source(self) -> str:
        """获取该乐器对应的 Demucs 分离轨道来源"""
        # Demucs 6s 只有这些轨道：drums, bass, vocals, guitar, piano, other
        # 其他乐器都来自 other 轨道
        direct_stems = {
            InstrumentType.DRUMS: "drums",
            InstrumentType.BASS: "bass",
            InstrumentType.VOCALS: "vocals",
            InstrumentType.GUITAR: "guitar",
            InstrumentType.PIANO: "piano",
        }
        return direct_stems.get(self, "other")


class ProcessingMode(Enum):
    """处理模式枚举"""
    PIANO = "piano"   # 默认：钢琴模式（跳过分离）
    SMART = "smart"   # 智能识别模式（自动检测乐器）


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
class PedalEvent:
    """表示一个踏板事件（延音踏板或柔音踏板）"""
    start_time: float    # 踩下时间（秒）
    end_time: float      # 抬起时间（秒）
    pedal_type: str = "sustain"  # "sustain" (CC64) 或 "soft" (CC67)

    @property
    def duration(self) -> float:
        """踏板持续时间（秒）"""
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
class TrackConfig:
    """单个轨道的配置"""
    id: str                         # 轨道ID，如 "piano_1", "drums"
    instrument: InstrumentType      # 乐器类型
    name: str                       # 显示名称
    enabled: bool = True            # 是否启用
    midi_channel: int = 0           # MIDI通道 (0-15)
    program: int = 0                # General MIDI 音色编号

    def __post_init__(self):
        # 自动设置默认音色编号
        if self.program == 0 and self.instrument != InstrumentType.PIANO:
            self.program = self.instrument.to_program_number()
        # 鼓轨道固定使用通道9（GM标准）
        if self.instrument == InstrumentType.DRUMS:
            self.midi_channel = 9


@dataclass
class TrackLayout:
    """轨道布局配置"""
    mode: ProcessingMode
    tracks: List[TrackConfig] = field(default_factory=list)

    @classmethod
    def default_piano(cls, count: int = 2) -> "TrackLayout":
        """创建默认的钢琴模式轨道布局"""
        count = max(1, min(count, 8))  # 限制在 1-8 之间
        tracks = []
        for i in range(count):
            tracks.append(TrackConfig(
                id=f"piano_{i + 1}",
                instrument=InstrumentType.PIANO,
                name=f"钢琴 {i + 1}",
                enabled=True,
                midi_channel=i,
                program=0  # Acoustic Grand Piano
            ))
        return cls(mode=ProcessingMode.PIANO, tracks=tracks)

    @classmethod
    def from_detected_instruments(
        cls,
        instruments: List[InstrumentType]
    ) -> "TrackLayout":
        """从检测到的乐器列表创建轨道布局"""
        tracks = []
        channel = 0
        for i, inst in enumerate(instruments):
            if inst == InstrumentType.DRUMS:
                midi_channel = 9  # 鼓固定使用通道9
            else:
                midi_channel = channel
                channel += 1
                if channel == 9:
                    channel = 10  # 跳过鼓通道

            tracks.append(TrackConfig(
                id=f"{inst.value}_{i + 1}",
                instrument=inst,
                name=inst.get_display_name(),
                enabled=True,
                midi_channel=midi_channel,
                program=inst.to_program_number()
            ))
        return cls(mode=ProcessingMode.SMART, tracks=tracks)

    def get_enabled_tracks(self) -> List[TrackConfig]:
        """获取所有启用的轨道"""
        return [t for t in self.tracks if t.enabled]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "mode": self.mode.value,
            "tracks": [
                {
                    "id": t.id,
                    "instrument": t.instrument.value,
                    "name": t.name,
                    "enabled": t.enabled,
                    "midi_channel": t.midi_channel,
                    "program": t.program,
                }
                for t in self.tracks
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackLayout":
        """从字典创建"""
        mode = ProcessingMode(data.get("mode", "piano"))
        tracks = []
        for t in data.get("tracks", []):
            tracks.append(TrackConfig(
                id=t["id"],
                instrument=InstrumentType(t["instrument"]),
                name=t["name"],
                enabled=t.get("enabled", True),
                midi_channel=t.get("midi_channel", 0),
                program=t.get("program", 0),
            ))
        return cls(mode=mode, tracks=tracks)


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

    # 轨道系统设置
    processing_mode: str = "piano"   # "piano" 或 "smart"
    piano_track_count: int = 2       # 钢琴轨道数量 (1-8)

    # Whisper设置
    whisper_model: str = "medium"  # tiny, base, small, medium, large
    lyrics_language: Optional[str] = None  # None = 自动检测

    # MIDI设置
    ticks_per_beat: int = 480
    default_velocity: int = 80
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_length: int = 58  # 毫秒

    # MIDI后处理设置
    quantize_notes: bool = True       # 音符量化
    quantize_grid: str = "1/32"       # 量化网格：从1/16改为1/32，更精细
    remove_duplicates: bool = True    # 去除重复音符
    velocity_smoothing: bool = True   # 力度平滑
    max_polyphony: int = 25           # 最大复音数：从10提高到25，更好支持钢琴

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
            "processing_mode": self.processing_mode,
            "piano_track_count": self.piano_track_count,
            "whisper_model": self.whisper_model,
            "lyrics_language": self.lyrics_language,
            "ticks_per_beat": self.ticks_per_beat,
            "default_velocity": self.default_velocity,
            "onset_threshold": self.onset_threshold,
            "frame_threshold": self.frame_threshold,
            "min_note_length": self.min_note_length,
            "quantize_notes": self.quantize_notes,
            "quantize_grid": self.quantize_grid,
            "remove_duplicates": self.remove_duplicates,
            "velocity_smoothing": self.velocity_smoothing,
            "max_polyphony": self.max_polyphony,
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
