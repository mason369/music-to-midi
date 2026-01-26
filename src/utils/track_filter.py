"""
稀疏轨道过滤模块

处理只有少量音符的错误乐器轨道，通过合并或丢弃来优化输出。
"""
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.models.data_models import NoteEvent

logger = logging.getLogger(__name__)


class SparseTrackStrategy(Enum):
    """稀疏轨道处理策略"""
    DISCARD = "discard"           # 丢弃稀疏轨道
    MERGE_FAMILY = "merge_family" # 合并到同类乐器家族
    MERGE_OTHER = "merge_other"   # 合并到"其他"轨道
    KEEP = "keep"                 # 保持原样


# GM 乐器家族定义 (程序号范围)
GM_FAMILIES = {
    "Piano": (0, 7),           # 钢琴
    "Chromatic": (8, 15),      # 半音阶打击乐
    "Organ": (16, 23),         # 风琴
    "Guitar": (24, 31),        # 吉他
    "Bass": (32, 39),          # 贝斯
    "Strings": (40, 55),       # 弦乐 (包括合奏)
    "Brass": (56, 63),         # 铜管
    "Reed": (64, 71),          # 簧管
    "Pipe": (72, 79),          # 哨笛
    "Synth Lead": (80, 87),    # 主奏合成器
    "Synth Pad": (88, 95),     # 铺底合成器
    "Synth Effects": (96, 103),# 合成器效果
    "Ethnic": (104, 111),      # 民族乐器
    "Percussive": (112, 119),  # 打击乐
    "Sound Effects": (120, 127),# 音效
}


def get_instrument_family(program: int) -> str:
    """
    获取 GM 程序号对应的乐器家族

    参数:
        program: GM 程序号 (0-127)

    返回:
        乐器家族名称
    """
    for family_name, (start, end) in GM_FAMILIES.items():
        if start <= program <= end:
            return family_name
    return "Other"


def is_sparse_track(
    notes: List[NoteEvent],
    min_note_count: int = 20,
    min_notes_per_minute: float = 5.0,
    audio_duration: Optional[float] = None
) -> Tuple[bool, str]:
    """
    判断轨道是否为稀疏轨道

    参数:
        notes: 音符列表
        min_note_count: 最小音符数阈值
        min_notes_per_minute: 最小音符密度阈值 (音符/分钟)
        audio_duration: 音频时长（秒），如果为 None 则从音符推断

    返回:
        (是否稀疏, 原因)
    """
    if not notes:
        return True, "empty"

    note_count = len(notes)

    # 条件1：音符数太少
    if note_count < min_note_count:
        return True, f"note_count={note_count}<{min_note_count}"

    # 计算音频时长
    if audio_duration is None:
        # 从音符推断时长
        if notes:
            audio_duration = max(n.end_time for n in notes)
        else:
            audio_duration = 0

    # 条件2：音符密度太低
    if audio_duration > 0:
        notes_per_minute = note_count / (audio_duration / 60)
        if notes_per_minute < min_notes_per_minute:
            return True, f"density={notes_per_minute:.1f}<{min_notes_per_minute}/min"

    return False, ""


def handle_sparse_tracks(
    instrument_notes: Dict[int, List[NoteEvent]],
    strategy: SparseTrackStrategy = SparseTrackStrategy.MERGE_FAMILY,
    min_note_count: int = 20,
    min_notes_per_minute: float = 5.0,
    audio_duration: Optional[float] = None
) -> Dict[int, List[NoteEvent]]:
    """
    处理稀疏轨道

    参数:
        instrument_notes: GM程序号到音符列表的字典
        strategy: 处理策略
        min_note_count: 最小音符数阈值
        min_notes_per_minute: 最小音符密度阈值
        audio_duration: 音频时长（秒）

    返回:
        处理后的字典
    """
    if strategy == SparseTrackStrategy.KEEP:
        return instrument_notes

    # 识别稀疏轨道
    sparse_tracks = {}
    normal_tracks = {}

    for program, notes in instrument_notes.items():
        is_sparse, reason = is_sparse_track(
            notes, min_note_count, min_notes_per_minute, audio_duration
        )

        if is_sparse:
            sparse_tracks[program] = (notes, reason)
            logger.info(f"识别稀疏轨道: GM {program} ({len(notes)} 音符, {reason})")
        else:
            normal_tracks[program] = notes

    if not sparse_tracks:
        logger.info("没有发现稀疏轨道")
        return instrument_notes

    logger.info(f"发现 {len(sparse_tracks)} 个稀疏轨道，策略: {strategy.value}")

    # 根据策略处理
    if strategy == SparseTrackStrategy.DISCARD:
        # 直接丢弃稀疏轨道
        discarded_notes = sum(len(notes) for notes, _ in sparse_tracks.values())
        logger.info(f"丢弃 {len(sparse_tracks)} 个稀疏轨道，共 {discarded_notes} 个音符")
        return normal_tracks

    elif strategy == SparseTrackStrategy.MERGE_FAMILY:
        # 合并到同一家族中音符最多的轨道
        return _merge_to_family(sparse_tracks, normal_tracks)

    elif strategy == SparseTrackStrategy.MERGE_OTHER:
        # 合并到 "其他" 轨道 (program 0 或最大的)
        return _merge_to_other(sparse_tracks, normal_tracks)

    return normal_tracks


def _merge_to_family(
    sparse_tracks: Dict[int, Tuple[List[NoteEvent], str]],
    normal_tracks: Dict[int, List[NoteEvent]]
) -> Dict[int, List[NoteEvent]]:
    """
    将稀疏轨道合并到同一家族中音符最多的轨道
    """
    result = {k: list(v) for k, v in normal_tracks.items()}  # 深拷贝

    # 按家族分组正常轨道
    family_tracks = defaultdict(list)
    for program, notes in normal_tracks.items():
        family = get_instrument_family(program)
        family_tracks[family].append((program, len(notes)))

    # 找到每个家族中音符最多的轨道
    family_main_track = {}
    for family, tracks in family_tracks.items():
        if tracks:
            main_program = max(tracks, key=lambda x: x[1])[0]
            family_main_track[family] = main_program

    # 处理每个稀疏轨道
    merged_count = 0
    for program, (notes, reason) in sparse_tracks.items():
        family = get_instrument_family(program)

        if family in family_main_track:
            # 合并到同家族的主轨道
            target_program = family_main_track[family]
            if target_program not in result:
                result[target_program] = []
            result[target_program].extend(notes)
            merged_count += len(notes)
            logger.info(f"合并 GM {program} ({len(notes)} 音符) -> GM {target_program} ({family})")
        else:
            # 没有同家族的正常轨道，尝试找最近的家族
            # 或者创建新轨道
            if notes:
                # 找到音符最多的正常轨道
                if result:
                    target_program = max(result.keys(), key=lambda p: len(result[p]))
                    result[target_program].extend(notes)
                    merged_count += len(notes)
                    logger.info(f"合并 GM {program} ({len(notes)} 音符) -> GM {target_program} (无同族)")
                else:
                    # 没有正常轨道，保留这个稀疏轨道
                    result[program] = notes
                    logger.info(f"保留 GM {program} ({len(notes)} 音符) (无可合并目标)")

    # 对合并后的轨道按时间排序
    for program in result:
        result[program].sort(key=lambda n: n.start_time)

    logger.info(f"家族合并完成: 合并 {merged_count} 个音符到 {len(result)} 个轨道")

    return result


def _merge_to_other(
    sparse_tracks: Dict[int, Tuple[List[NoteEvent], str]],
    normal_tracks: Dict[int, List[NoteEvent]]
) -> Dict[int, List[NoteEvent]]:
    """
    将稀疏轨道合并到 "其他" 轨道
    """
    result = {k: list(v) for k, v in normal_tracks.items()}

    # 收集所有稀疏轨道的音符
    all_sparse_notes = []
    for program, (notes, reason) in sparse_tracks.items():
        all_sparse_notes.extend(notes)
        logger.info(f"收集 GM {program} ({len(notes)} 音符) 到其他轨道")

    if not all_sparse_notes:
        return result

    # 找到或创建 "其他" 轨道
    # 优先使用 program 0 (钢琴) 或已存在的最大程序号
    if 0 in result:
        target_program = 0
    elif result:
        target_program = max(result.keys())
    else:
        target_program = 0
        result[target_program] = []

    # 合并
    result[target_program].extend(all_sparse_notes)
    result[target_program].sort(key=lambda n: n.start_time)

    logger.info(f"合并 {len(all_sparse_notes)} 个音符到 GM {target_program}")

    return result


def filter_sparse_drum_tracks(
    drum_notes: Dict[int, List[NoteEvent]],
    min_note_count: int = 10,
    audio_duration: Optional[float] = None
) -> Dict[int, List[NoteEvent]]:
    """
    过滤稀疏的鼓轨道

    鼓轨道按音高分组，稀疏的音高会被合并到最近的常用鼓音高。

    参数:
        drum_notes: 鼓音高到音符列表的字典
        min_note_count: 最小音符数阈值
        audio_duration: 音频时长

    返回:
        过滤后的字典
    """
    if not drum_notes:
        return drum_notes

    # 常用鼓音高映射 (GM 标准)
    COMMON_DRUM_PITCHES = {
        35: "Acoustic Bass Drum",
        36: "Bass Drum 1",
        38: "Acoustic Snare",
        40: "Electric Snare",
        42: "Closed Hi-Hat",
        44: "Pedal Hi-Hat",
        46: "Open Hi-Hat",
        49: "Crash Cymbal 1",
        51: "Ride Cymbal 1",
        45: "Low Tom",
        47: "Low-Mid Tom",
        48: "Hi-Mid Tom",
        50: "High Tom",
    }

    # 识别稀疏鼓轨道
    sparse_drums = {}
    normal_drums = {}

    for pitch, notes in drum_notes.items():
        if len(notes) < min_note_count:
            sparse_drums[pitch] = notes
        else:
            normal_drums[pitch] = notes

    if not sparse_drums:
        return drum_notes

    logger.info(f"发现 {len(sparse_drums)} 个稀疏鼓音高")

    # 合并稀疏鼓到最近的常用音高
    result = {k: list(v) for k, v in normal_drums.items()}

    for pitch, notes in sparse_drums.items():
        # 找到最近的常用鼓音高
        if normal_drums:
            target_pitch = min(normal_drums.keys(), key=lambda p: abs(p - pitch))
        elif COMMON_DRUM_PITCHES:
            # 找到最近的标准鼓音高
            target_pitch = min(COMMON_DRUM_PITCHES.keys(), key=lambda p: abs(p - pitch))
            if target_pitch not in result:
                result[target_pitch] = []
        else:
            target_pitch = 36  # 默认底鼓
            if target_pitch not in result:
                result[target_pitch] = []

        result[target_pitch].extend(notes)
        logger.debug(f"合并鼓音高 {pitch} ({len(notes)} 音符) -> {target_pitch}")

    # 排序
    for pitch in result:
        result[pitch].sort(key=lambda n: n.start_time)

    return result


def get_track_statistics(
    instrument_notes: Dict[int, List[NoteEvent]],
    audio_duration: Optional[float] = None
) -> List[dict]:
    """
    获取轨道统计信息

    参数:
        instrument_notes: GM程序号到音符列表的字典
        audio_duration: 音频时长

    返回:
        统计信息列表
    """
    from src.models.gm_instruments import get_instrument_name

    stats = []

    for program, notes in sorted(instrument_notes.items()):
        if not notes:
            continue

        note_count = len(notes)

        # 计算时长
        if audio_duration is None:
            duration = max(n.end_time for n in notes) if notes else 0
        else:
            duration = audio_duration

        # 计算密度
        notes_per_minute = note_count / (duration / 60) if duration > 0 else 0

        # 判断是否稀疏
        is_sparse, reason = is_sparse_track(notes, audio_duration=audio_duration)

        stats.append({
            "program": program,
            "name": get_instrument_name(program),
            "family": get_instrument_family(program),
            "note_count": note_count,
            "duration": duration,
            "notes_per_minute": notes_per_minute,
            "is_sparse": is_sparse,
            "sparse_reason": reason,
        })

    return stats
