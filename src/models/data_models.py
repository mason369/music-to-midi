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
    # 新增乐器类型（YourMT3+ 支持）
    PERCUSSION = "percussion"  # 非套鼓打击乐
    CHOIR = "choir"            # 合唱（区别于独唱）
    LEAD_SYNTH = "lead_synth"  # 主奏合成器
    PAD_SYNTH = "pad_synth"    # 铺底合成器
    # 层次化鼓分离
    KICK = "kick"              # 底鼓
    SNARE = "snare"            # 军鼓
    HIHAT = "hihat"            # 踩镲
    TOM = "tom"                # 嗵鼓
    CYMBAL = "cymbal"          # 镲片
    RIDE = "ride"              # 叮叮镲
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
            InstrumentType.PERCUSSION: 0, # 打击乐使用通道10
            InstrumentType.CHOIR: 52,     # Choir Aahs
            InstrumentType.LEAD_SYNTH: 80,# Lead 1
            InstrumentType.PAD_SYNTH: 88, # Pad 1 (new age)
            InstrumentType.KICK: 0,       # 底鼓（通道10）
            InstrumentType.SNARE: 0,      # 军鼓（通道10）
            InstrumentType.HIHAT: 0,      # 踩镲（通道10）
            InstrumentType.TOM: 0,        # 嗵鼓（通道10）
            InstrumentType.CYMBAL: 0,     # 镲片（通道10）
            InstrumentType.RIDE: 0,       # 叮叮镲（通道10）
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
            InstrumentType.PERCUSSION: "打击乐",
            InstrumentType.CHOIR: "合唱",
            InstrumentType.LEAD_SYNTH: "主奏合成器",
            InstrumentType.PAD_SYNTH: "铺底合成器",
            InstrumentType.KICK: "底鼓",
            InstrumentType.SNARE: "军鼓",
            InstrumentType.HIHAT: "踩镲",
            InstrumentType.TOM: "嗵鼓",
            InstrumentType.CYMBAL: "镲片",
            InstrumentType.RIDE: "叮叮镲",
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
            InstrumentType.PERCUSSION: "Percussion",
            InstrumentType.CHOIR: "Choir",
            InstrumentType.LEAD_SYNTH: "Lead Synth",
            InstrumentType.PAD_SYNTH: "Pad Synth",
            InstrumentType.KICK: "Kick",
            InstrumentType.SNARE: "Snare",
            InstrumentType.HIHAT: "Hi-Hat",
            InstrumentType.TOM: "Tom",
            InstrumentType.CYMBAL: "Cymbal",
            InstrumentType.RIDE: "Ride",
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
    PIANO = "piano"         # 钢琴模式（跳过分离，直接转写为钢琴轨道）
    SMART = "smart"         # 智能识别模式（多乐器转写，动态识别乐器）


class TranscriptionQuality(Enum):
    """转写质量模式枚举"""
    FAST = "fast"           # 快速模式：无后处理，最快速度
    BALANCED = "balanced"   # 平衡模式：轻量后处理，平衡质量和速度
    BEST = "best"           # 极致质量模式：最小后处理，保留最多细节


class ProcessingStage(Enum):
    """处理流水线中的阶段"""
    PREPROCESSING = "preprocessing"
    SEPARATION = "separation"
    TRANSCRIPTION = "transcription"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"


@dataclass
class NoteEvent:
    """表示一个MIDI音符事件"""
    pitch: int           # MIDI音高 (0-127)
    start_time: float    # 开始时间（秒）
    end_time: float      # 结束时间（秒）
    velocity: int = 80   # 音符力度 (0-127)
    program: int = 0     # GM 程序号 (0-127)，用于精确乐器识别

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
    source: str = "original"        # 分离轨道来源 (original, vocals, accompaniment, guitar, other)

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
        """创建钢琴模式轨道布局（支持1-4轨道自适应分离）

        轨道分配策略：
        - 1 轨道：不分离，全部为钢琴声部
        - 2 轨道：分离为钢琴（伴奏）+ 钢琴（人声）
        - 3 轨道：分离为钢琴（伴奏）+ 钢琴（人声）+ 钢琴（其他）
        - 4 轨道：分离为钢琴（伴奏）+ 钢琴（人声）+ 钢琴（吉他）+ 钢琴（其他）
        """
        count = max(1, min(count, 4))  # 限制为 1-4

        PIANO_TRACK_TEMPLATES = {
            1: [
                {"id": "piano_full", "name": "钢琴", "source": "original"}
            ],
            2: [
                {"id": "piano_accompaniment", "name": "钢琴（伴奏）", "source": "accompaniment"},
                {"id": "piano_vocals", "name": "钢琴（人声）", "source": "vocals"}
            ],
            3: [
                {"id": "piano_accompaniment", "name": "钢琴（伴奏）", "source": "accompaniment"},
                {"id": "piano_vocals", "name": "钢琴（人声）", "source": "vocals"},
                {"id": "piano_other", "name": "钢琴（其他）", "source": "other"}
            ],
            4: [
                {"id": "piano_accompaniment", "name": "钢琴（伴奏）", "source": "accompaniment"},
                {"id": "piano_vocals", "name": "钢琴（人声）", "source": "vocals"},
                {"id": "piano_guitar", "name": "钢琴（吉他）", "source": "guitar"},
                {"id": "piano_other", "name": "钢琴（其他）", "source": "other"}
            ]
        }

        template = PIANO_TRACK_TEMPLATES[count]
        tracks = []
        for i, t in enumerate(template):
            tracks.append(TrackConfig(
                id=t["id"],
                instrument=InstrumentType.PIANO,
                name=t["name"],
                enabled=True,
                midi_channel=i,
                program=0,  # Acoustic Grand Piano
                source=t["source"]
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
                    "source": t.source,
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
                source=t.get("source", "original"),
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
    tracks: List[Track] = field(default_factory=list)
    beat_info: Optional[BeatInfo] = None
    processing_time: float = 0.0             # 总处理时间（秒）


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

    # 转写引擎设置
    transcriber_engine: str = "auto"  # "auto", "yourmt3", "basic_pitch"
    use_direct_transcription: bool = False  # True = 跳过分离，直接YourMT3+
    transcription_quality: str = "best"      # "fast", "balanced", "best" - 默认使用最佳质量
    use_precise_instruments: bool = True     # 使用精确 GM 程序号（128种乐器）
    preserve_all_notes: bool = True          # 保留所有音符（不做激进后处理）
    ultra_quality_mode: bool = True          # 极致质量模式（忽略性能，最大化质量）

    # 鼓设置
    separate_drum_voices: bool = False  # 层次化鼓输出（底鼓、军鼓、踩镲等）

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
    max_polyphony: int = 40           # 最大复音数：从25提高到40，更好支持钢琴
    aggressive_post_processing: bool = False  # False = 轻量后处理（保留更多音符），True = 激进后处理（更简化）

    # 力度估算设置（从音频振幅恢复动态）
    estimate_velocity_from_audio: bool = True   # 启用基于振幅的力度估算
    velocity_range_min: int = 30                # 最小力度
    velocity_range_max: int = 110               # 最大力度

    # 稀疏轨道过滤设置
    filter_sparse_tracks: bool = True           # 启用稀疏轨道过滤
    sparse_track_strategy: str = "merge_family" # 处理策略: "discard", "merge_family", "merge_other", "keep"
    min_track_note_count: int = 20              # 最小音符数
    min_track_notes_per_minute: float = 5.0     # 最小音符密度

    # 输出设置
    output_dir: str = ""
    save_separated_tracks: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "theme": self.theme,
            "use_gpu": self.use_gpu,
            "gpu_device": self.gpu_device,
            "segment_size": self.segment_size,
            "processing_mode": self.processing_mode,
            "piano_track_count": self.piano_track_count,
            "transcriber_engine": self.transcriber_engine,
            "use_direct_transcription": self.use_direct_transcription,
            "transcription_quality": self.transcription_quality,
            "use_precise_instruments": self.use_precise_instruments,
            "preserve_all_notes": self.preserve_all_notes,
            "ultra_quality_mode": self.ultra_quality_mode,
            "separate_drum_voices": self.separate_drum_voices,
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
            "aggressive_post_processing": self.aggressive_post_processing,
            "estimate_velocity_from_audio": self.estimate_velocity_from_audio,
            "velocity_range_min": self.velocity_range_min,
            "velocity_range_max": self.velocity_range_max,
            "filter_sparse_tracks": self.filter_sparse_tracks,
            "sparse_track_strategy": self.sparse_track_strategy,
            "min_track_note_count": self.min_track_note_count,
            "min_track_notes_per_minute": self.min_track_notes_per_minute,
            "output_dir": self.output_dir,
            "save_separated_tracks": self.save_separated_tracks
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
