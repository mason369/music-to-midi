"""
MIDI生成模块 - 支持歌词嵌入和后处理优化
"""
import logging
import os
from typing import List, Dict, Optional
from copy import deepcopy
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from src.models.data_models import (
    Config, NoteEvent, LyricEvent, TrackType,
    InstrumentType, TrackConfig, TrackLayout, PedalEvent
)

logger = logging.getLogger(__name__)


class MidiGenerator:
    """
    MIDI 文件生成器，支持多轨道和歌词嵌入

    功能特点:
    - 多轨道 MIDI 创建
    - 歌词元事件嵌入 (0xFF 0x05)
    - LRC 文件导出
    - 可配置速度和乐器
    """

    # 不同轨道类型的 MIDI 通道映射
    CHANNEL_MAP = {
        TrackType.DRUMS: 9,     # GM 标准鼓通道
        TrackType.BASS: 0,
        TrackType.VOCALS: 1,
        TrackType.OTHER: 2
    }

    # 不同轨道类型的 GM 音色编号
    PROGRAM_MAP = {
        TrackType.DRUMS: 0,      # 鼓组不需要音色变更
        TrackType.BASS: 33,      # 电贝斯（指弹）
        TrackType.VOCALS: 52,    # 合唱
        TrackType.OTHER: 0       # 原声大钢琴
    }

    # 轨道名称 (使用ASCII兼容名称，因为MIDI元数据使用latin-1编码)
    TRACK_NAMES = {
        TrackType.DRUMS: "Drums",
        TrackType.BASS: "Bass",
        TrackType.VOCALS: "Vocals",
        TrackType.OTHER: "Other"
    }

    def __init__(self, config: Config):
        """
        初始化 MIDI 生成器

        参数:
            config: 应用配置
        """
        self.config = config
        self.ticks_per_beat = config.ticks_per_beat

    def generate(
        self,
        tracks: Dict[TrackType, List[NoteEvent]],
        lyrics: List[LyricEvent],
        tempo: float,
        output_path: str,
        embed_lyrics: bool = True
    ) -> str:
        """
        生成带可选歌词的多轨道 MIDI 文件

        参数:
            tracks: 轨道类型到音符事件的字典
            lyrics: 带时间戳的歌词事件列表
            tempo: BPM
            output_path: 输出 MIDI 文件路径
            embed_lyrics: 是否将歌词作为元事件嵌入

        返回:
            生成的 MIDI 文件路径
        """
        logger.info(f"正在生成 MIDI: {output_path}")

        # 创建 MIDI 文件
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)

        # 轨道 0: 速度和歌词（指挥轨道）
        meta_track = MidiTrack()
        mid.tracks.append(meta_track)
        meta_track.name = "Conductor"

        # 设置速度
        tempo_value = mido.bpm2tempo(tempo)
        meta_track.append(MetaMessage('set_tempo', tempo=tempo_value, time=0))

        # 设置拍号 (4/4)
        meta_track.append(MetaMessage(
            'time_signature',
            numerator=4,
            denominator=4,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0
        ))

        # 将歌词添加到指挥轨道
        if embed_lyrics and lyrics:
            self._add_lyrics_events(meta_track, lyrics, tempo)

        # 轨道结束
        meta_track.append(MetaMessage('end_of_track', time=0))

        # 为每个音源创建轨道
        for track_type, notes in tracks.items():
            if notes:
                track = self._create_track(track_type, notes, tempo)
                mid.tracks.append(track)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # 保存 MIDI 文件
        mid.save(output_path)
        logger.info(f"MIDI 已保存: {output_path}")

        return output_path

    def _add_lyrics_events(
        self,
        track: MidiTrack,
        lyrics: List[LyricEvent],
        tempo: float
    ) -> None:
        """
        将歌词添加为 MIDI 元事件

        参数:
            track: 要添加歌词的 MIDI 轨道
            lyrics: 歌词事件列表
            tempo: 用于时间转换的 BPM
        """
        logger.info(f"正在嵌入 {len(lyrics)} 个歌词事件")

        current_tick = 0

        for lyric in sorted(lyrics, key=lambda l: l.start_time):
            # 将时间转换为 ticks
            tick = self._time_to_ticks(lyric.start_time, tempo)
            delta = max(0, tick - current_tick)

            # 添加歌词元事件
            try:
                track.append(MetaMessage(
                    'lyrics',
                    text=lyric.text,
                    time=delta
                ))
                current_tick = tick
            except Exception as e:
                logger.warning(f"无法添加歌词 '{lyric.text}': {e}")

    def _create_track(
        self,
        track_type: TrackType,
        notes: List[NoteEvent],
        tempo: float
    ) -> MidiTrack:
        """
        创建带音符的 MIDI 轨道

        参数:
            track_type: 轨道类型
            notes: 音符事件
            tempo: BPM

        返回:
            带音符的 MidiTrack
        """
        track = MidiTrack()
        track.name = self.TRACK_NAMES.get(track_type, "Track")

        channel = self.CHANNEL_MAP.get(track_type, 0)
        program = self.PROGRAM_MAP.get(track_type, 0)

        # 音色变更（鼓组除外）
        if track_type != TrackType.DRUMS:
            track.append(Message(
                'program_change',
                channel=channel,
                program=program,
                time=0
            ))

        # 按开始时间排序音符
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        # 创建音符开/关事件
        events = []
        for note in sorted_notes:
            start_tick = self._time_to_ticks(note.start_time, tempo)
            end_tick = self._time_to_ticks(note.end_time, tempo)

            events.append({
                'type': 'note_on',
                'tick': start_tick,
                'note': note.pitch,
                'velocity': note.velocity,
                'channel': channel
            })
            events.append({
                'type': 'note_off',
                'tick': end_tick,
                'note': note.pitch,
                'velocity': 0,
                'channel': channel
            })

        # 按 tick 排序事件
        events.sort(key=lambda e: (e['tick'], e['type'] == 'note_on'))

        # 添加带增量时间的事件
        current_tick = 0
        for event in events:
            delta = max(0, event['tick'] - current_tick)

            if event['type'] == 'note_on':
                track.append(Message(
                    'note_on',
                    note=event['note'],
                    velocity=event['velocity'],
                    channel=event['channel'],
                    time=delta
                ))
            else:
                track.append(Message(
                    'note_off',
                    note=event['note'],
                    velocity=0,
                    channel=event['channel'],
                    time=delta
                ))

            current_tick = event['tick']

        # 轨道结束
        track.append(MetaMessage('end_of_track', time=0))

        logger.info(f"已创建 {track_type.value} 轨道，包含 {len(sorted_notes)} 个音符")

        return track

    def _time_to_ticks(self, time_seconds: float, tempo: float) -> int:
        """
        将秒转换为 MIDI ticks

        参数:
            time_seconds: 时间（秒）
            tempo: BPM

        返回:
            MIDI ticks
        """
        # ticks = time * (ticks_per_beat * bpm / 60)
        ticks = int(time_seconds * self.ticks_per_beat * tempo / 60)
        return ticks

    def _ticks_to_time(self, ticks: int, tempo: float) -> float:
        """
        将 MIDI ticks 转换为秒

        参数:
            ticks: MIDI ticks
            tempo: BPM

        返回:
            时间（秒）
        """
        return ticks * 60 / (self.ticks_per_beat * tempo)

    # ==================== 后处理方法 ====================

    def _quantize_notes(
        self,
        notes: List[NoteEvent],
        tempo: float,
        grid: str = "1/16"
    ) -> List[NoteEvent]:
        """
        将音符量化到指定的网格

        参数:
            notes: 音符列表
            tempo: BPM
            grid: 量化网格 ("1/4", "1/8", "1/16", "1/32")

        返回:
            量化后的音符列表
        """
        if not notes:
            return notes

        # 解析网格值
        grid_map = {
            "1/4": 1.0,
            "1/8": 0.5,
            "1/16": 0.25,
            "1/32": 0.125
        }
        beats_per_grid = grid_map.get(grid, 0.25)

        # 计算网格时间（秒）
        grid_time = beats_per_grid * 60 / tempo

        quantized = []
        for note in notes:
            # 量化开始时间
            quantized_start = round(note.start_time / grid_time) * grid_time

            # 量化时长（确保至少有一个网格单位）
            duration = note.end_time - note.start_time
            quantized_duration = max(grid_time, round(duration / grid_time) * grid_time)

            quantized.append(NoteEvent(
                pitch=note.pitch,
                start_time=quantized_start,
                end_time=quantized_start + quantized_duration,
                velocity=note.velocity
            ))

        logger.debug(f"已量化 {len(notes)} 个音符到 {grid} 网格")
        return quantized

    def _remove_duplicate_notes(
        self,
        notes: List[NoteEvent],
        time_threshold: float = 0.025  # 从0.01提高到0.025秒
    ) -> List[NoteEvent]:
        """
        去除重叠的重复音符

        参数:
            notes: 音符列表
            time_threshold: 判断重复的时间阈值（秒）

        返回:
            去重后的音符列表
        """
        if not notes:
            return notes

        # 按开始时间和音高排序
        sorted_notes = sorted(notes, key=lambda n: (n.start_time, n.pitch))

        result = []
        for note in sorted_notes:
            is_duplicate = False
            for existing in result:
                # 检查是否是相同音高且时间重叠
                if (existing.pitch == note.pitch and
                    abs(existing.start_time - note.start_time) < time_threshold):
                    # 保留较长的音符
                    if note.duration > existing.duration:
                        result.remove(existing)
                    else:
                        is_duplicate = True
                    break

            if not is_duplicate:
                result.append(note)

        removed_count = len(notes) - len(result)
        if removed_count > 0:
            logger.debug(f"已移除 {removed_count} 个重复音符")

        return result

    def _smooth_velocity(
        self,
        notes: List[NoteEvent],
        window_size: int = 5,  # 从3提高到5，更平滑的力度曲线
        min_velocity: int = 20
    ) -> List[NoteEvent]:
        """
        平滑音符力度

        参数:
            notes: 音符列表
            window_size: 平滑窗口大小
            min_velocity: 最小力度值（低于此值的音符将被过滤）

        返回:
            力度平滑后的音符列表
        """
        if not notes or len(notes) < window_size:
            return [n for n in notes if n.velocity >= min_velocity]

        # 按开始时间排序
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        # 过滤低力度音符
        filtered_notes = [n for n in sorted_notes if n.velocity >= min_velocity]

        if len(filtered_notes) < window_size:
            return filtered_notes

        # 使用移动平均平滑力度
        result = []
        for i, note in enumerate(filtered_notes):
            # 计算窗口范围
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(filtered_notes), i + window_size // 2 + 1)

            # 计算窗口内的平均力度
            window_velocities = [filtered_notes[j].velocity for j in range(start_idx, end_idx)]
            avg_velocity = int(sum(window_velocities) / len(window_velocities))

            # 混合原始力度和平均力度（保留一定的动态特征）
            smoothed_velocity = int(note.velocity * 0.6 + avg_velocity * 0.4)
            smoothed_velocity = max(min_velocity, min(127, smoothed_velocity))

            result.append(NoteEvent(
                pitch=note.pitch,
                start_time=note.start_time,
                end_time=note.end_time,
                velocity=smoothed_velocity
            ))

        removed_count = len(notes) - len(result)
        if removed_count > 0:
            logger.debug(f"已过滤 {removed_count} 个低力度音符")

        return result

    def _limit_polyphony(
        self,
        notes: List[NoteEvent],
        max_polyphony: int = 25  # 从10提高到25，更好支持钢琴
    ) -> List[NoteEvent]:
        """
        限制最大复音数

        参数:
            notes: 音符列表
            max_polyphony: 最大同时发声的音符数

        返回:
            限制复音后的音符列表
        """
        if not notes or max_polyphony <= 0:
            return notes

        # 按开始时间排序
        sorted_notes = sorted(notes, key=lambda n: (n.start_time, -n.velocity))

        result = []
        for note in sorted_notes:
            # 计算在该音符开始时还在发声的音符数
            active_notes = sum(
                1 for n in result
                if n.start_time <= note.start_time < n.end_time
            )

            if active_notes < max_polyphony:
                result.append(note)

        removed_count = len(notes) - len(result)
        if removed_count > 0:
            logger.debug(f"已移除 {removed_count} 个超出复音限制的音符")

        return result

    def _merge_close_notes(
        self,
        notes: List[NoteEvent],
        gap_threshold: float = 0.050  # 50ms 间隔阈值
    ) -> List[NoteEvent]:
        """
        合并间隔很近的同音高音符

        当两个相同音高的音符间隔小于阈值时，合并为一个音符

        参数:
            notes: 音符列表
            gap_threshold: 合并间隔阈值（秒）

        返回:
            合并后的音符列表
        """
        if not notes:
            return notes

        # 按音高分组
        from collections import defaultdict
        notes_by_pitch: Dict[int, List[NoteEvent]] = defaultdict(list)
        for note in notes:
            notes_by_pitch[note.pitch].append(note)

        merged_notes = []
        merge_count = 0

        for pitch, pitch_notes in notes_by_pitch.items():
            # 按开始时间排序
            sorted_notes = sorted(pitch_notes, key=lambda n: n.start_time)

            i = 0
            while i < len(sorted_notes):
                current = sorted_notes[i]
                merged_end = current.end_time
                merged_velocity = current.velocity
                velocity_count = 1

                # 检查后续音符是否应该合并
                j = i + 1
                while j < len(sorted_notes):
                    next_note = sorted_notes[j]
                    gap = next_note.start_time - merged_end

                    if gap <= gap_threshold:
                        # 合并：扩展结束时间，平均力度
                        merged_end = max(merged_end, next_note.end_time)
                        merged_velocity += next_note.velocity
                        velocity_count += 1
                        merge_count += 1
                        j += 1
                    else:
                        break

                # 创建合并后的音符
                merged_notes.append(NoteEvent(
                    pitch=pitch,
                    start_time=current.start_time,
                    end_time=merged_end,
                    velocity=merged_velocity // velocity_count
                ))

                i = j

        if merge_count > 0:
            logger.debug(f"已合并 {merge_count} 个碎片音符")

        # 按开始时间排序返回
        return sorted(merged_notes, key=lambda n: n.start_time)

    def _smooth_vibrato(
        self,
        notes: List[NoteEvent],
        vibrato_range: int = 1,  # 允许的音高波动范围（半音）
        time_tolerance: float = 0.100  # 100ms 时间容差
    ) -> List[NoteEvent]:
        """
        平滑颤音导致的音高波动

        当连续的音符在很短时间内有小幅音高变化时，统一到主音高

        参数:
            notes: 音符列表
            vibrato_range: 视为颤音的音高范围（半音数）
            time_tolerance: 时间容差（秒）

        返回:
            平滑后的音符列表
        """
        if len(notes) < 2:
            return notes

        # 按开始时间排序
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        result = []
        smoothed_count = 0

        i = 0
        while i < len(sorted_notes):
            current = sorted_notes[i]

            # 收集可能是颤音的音符组
            vibrato_group = [current]
            j = i + 1

            while j < len(sorted_notes):
                next_note = sorted_notes[j]

                # 检查时间是否接近
                time_gap = next_note.start_time - vibrato_group[-1].end_time
                if time_gap > time_tolerance:
                    break

                # 检查音高是否在颤音范围内
                pitch_diff = abs(next_note.pitch - current.pitch)
                if pitch_diff > vibrato_range:
                    break

                vibrato_group.append(next_note)
                j += 1

            if len(vibrato_group) > 1:
                # 找到最常见的音高（主音高）
                from collections import Counter
                pitch_counts = Counter(n.pitch for n in vibrato_group)
                main_pitch = pitch_counts.most_common(1)[0][0]

                # 将所有音符统一到主音高
                for note in vibrato_group:
                    if note.pitch != main_pitch:
                        smoothed_count += 1
                    result.append(NoteEvent(
                        pitch=main_pitch,
                        start_time=note.start_time,
                        end_time=note.end_time,
                        velocity=note.velocity
                    ))
            else:
                result.append(current)

            i = j

        if smoothed_count > 0:
            logger.debug(f"已平滑 {smoothed_count} 个颤音音符")

        return result

    def _normalize_velocity(
        self,
        notes: List[NoteEvent],
        target_mean: int = 80,
        target_std: int = 20
    ) -> List[NoteEvent]:
        """
        力度归一化到目标分布

        将力度值调整到更自然的分布

        参数:
            notes: 音符列表
            target_mean: 目标平均力度
            target_std: 目标力度标准差

        返回:
            归一化后的音符列表
        """
        if len(notes) < 2:
            return notes

        import numpy as np

        # 计算当前力度统计
        velocities = np.array([n.velocity for n in notes])
        current_mean = np.mean(velocities)
        current_std = np.std(velocities)

        if current_std < 1:
            # 力度几乎没有变化，使用目标均值
            result = []
            for note in notes:
                result.append(NoteEvent(
                    pitch=note.pitch,
                    start_time=note.start_time,
                    end_time=note.end_time,
                    velocity=target_mean
                ))
            return result

        # Z-score 归一化后映射到目标分布
        result = []
        for note in notes:
            # 标准化
            z_score = (note.velocity - current_mean) / current_std
            # 映射到目标分布
            new_velocity = int(target_mean + z_score * target_std)
            # 限制在有效范围
            new_velocity = max(1, min(127, new_velocity))

            result.append(NoteEvent(
                pitch=note.pitch,
                start_time=note.start_time,
                end_time=note.end_time,
                velocity=new_velocity
            ))

        logger.debug(f"力度归一化: {current_mean:.0f}±{current_std:.0f} -> {target_mean}±{target_std}")

        return result

    def post_process_notes(
        self,
        notes: List[NoteEvent],
        tempo: float
    ) -> List[NoteEvent]:
        """
        对音符列表应用所有后处理

        处理流程:
        1. 颤音平滑 - 消除音高波动
        2. 音符合并 - 合并碎片音符
        3. 量化 - 对齐到网格
        4. 去重 - 移除重复音符
        5. 力度平滑 - 平滑力度变化
        6. 力度归一化 - 标准化力度分布
        7. 复音限制 - 限制同时发声数

        参数:
            notes: 音符列表
            tempo: BPM

        返回:
            后处理后的音符列表
        """
        if not notes:
            return notes

        processed = deepcopy(notes)
        initial_count = len(processed)

        # 1. 颤音平滑
        processed = self._smooth_vibrato(processed)

        # 2. 音符合并
        processed = self._merge_close_notes(processed)

        # 3. 量化
        if self.config.quantize_notes:
            processed = self._quantize_notes(processed, tempo, self.config.quantize_grid)

        # 4. 去重
        if self.config.remove_duplicates:
            processed = self._remove_duplicate_notes(processed)

        # 5. 力度平滑
        if self.config.velocity_smoothing:
            processed = self._smooth_velocity(processed)

        # 6. 力度归一化
        processed = self._normalize_velocity(processed)

        # 7. 限制复音
        if self.config.max_polyphony > 0:
            processed = self._limit_polyphony(processed, self.config.max_polyphony)

        logger.info(f"后处理: {initial_count} -> {len(processed)} 个音符")
        return processed

    def _get_post_process_params(self, track_count: int) -> dict:
        """
        根据轨道数量获取后处理参数

        原理：
        - max_polyphony: 降低 → 减少同时发声数（简化）
        - min_velocity: 提高 → 过滤弱音（简化）
        - gap_threshold: 增加 → 更多合并（简化）
        - quantize_grid: 粗网格 → 节奏更规整（简化）

        参数:
            track_count: 目标轨道数量（1-4）

        返回:
            包含后处理参数的字典
        """
        params = {
            1: {
                'max_polyphony': 4,      # 单旋律+和弦
                'min_velocity': 45,       # 只保留强音
                'gap_threshold': 0.150,   # 大幅合并
                'quantize_grid': '1/8',   # 粗网格
            },
            2: {
                'max_polyphony': 8,       # 每手4个音
                'min_velocity': 35,
                'gap_threshold': 0.100,
                'quantize_grid': '1/16',
            },
            3: {
                'max_polyphony': 15,
                'min_velocity': 28,
                'gap_threshold': 0.075,
                'quantize_grid': '1/32',
            },
            4: {
                'max_polyphony': 25,
                'min_velocity': 20,
                'gap_threshold': 0.050,
                'quantize_grid': '1/32',
            },
        }
        return params.get(track_count, params[4])

    def post_process_notes_with_complexity(
        self,
        notes: List[NoteEvent],
        tempo: float,
        track_count: int
    ) -> List[NoteEvent]:
        """
        根据轨道数量应用不同级别的后处理

        使用复杂度感知的参数进行后处理，轨道数越少参数越严格，
        输出的音符越精简，更适合人工演奏。

        参数:
            notes: 音符列表
            tempo: BPM
            track_count: 目标轨道数量（1-4）

        返回:
            后处理后的音符列表
        """
        if not notes:
            return notes

        # 获取复杂度参数
        params = self._get_post_process_params(track_count)
        logger.info(f"后处理参数（{track_count}轨）: polyphony={params['max_polyphony']}, "
                   f"min_vel={params['min_velocity']}, gap={params['gap_threshold']*1000:.0f}ms, "
                   f"grid={params['quantize_grid']}")

        processed = deepcopy(notes)
        initial_count = len(processed)

        # 1. 颤音平滑
        processed = self._smooth_vibrato(processed)

        # 2. 音符合并（使用复杂度参数）
        processed = self._merge_close_notes(processed, gap_threshold=params['gap_threshold'])

        # 3. 量化（使用复杂度参数的网格）
        processed = self._quantize_notes(processed, tempo, params['quantize_grid'])

        # 4. 去重
        processed = self._remove_duplicate_notes(processed)

        # 5. 力度平滑（使用复杂度参数的最小力度）
        processed = self._smooth_velocity(processed, min_velocity=params['min_velocity'])

        # 6. 力度归一化
        processed = self._normalize_velocity(processed)

        # 7. 限制复音（使用复杂度参数）
        processed = self._limit_polyphony(processed, max_polyphony=params['max_polyphony'])

        logger.info(f"复杂度感知后处理（{track_count}轨）: {initial_count} -> {len(processed)} 个音符")
        return processed

    # ==================== 新版生成方法 ====================

    def generate_v2(
        self,
        track_layout: TrackLayout,
        tracks_notes: Dict[str, List[NoteEvent]],
        tempo: float,
        output_path: str,
        lyrics: Optional[List[LyricEvent]] = None,
        embed_lyrics: bool = True,
        apply_post_processing: bool = True,
        track_count: Optional[int] = None,
        pedals: Optional[List[PedalEvent]] = None
    ) -> str:
        """
        使用新的轨道布局系统生成 MIDI 文件

        参数:
            track_layout: 轨道布局配置
            tracks_notes: 轨道ID到音符事件的字典
            tempo: BPM
            output_path: 输出 MIDI 文件路径
            lyrics: 带时间戳的歌词事件列表（可选）
            embed_lyrics: 是否将歌词作为元事件嵌入
            apply_post_processing: 是否应用后处理优化
            track_count: 轨道数量，用于复杂度感知后处理（可选，钢琴模式时使用）
            pedals: 踏板事件列表（可选，钢琴模式使用）

        返回:
            生成的 MIDI 文件路径
        """
        logger.info(f"正在生成 MIDI (v2): {output_path}")
        logger.info(f"轨道布局: {len(track_layout.tracks)} 个轨道")

        # 创建 MIDI 文件
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)

        # 轨道 0: 速度和歌词（指挥轨道）
        meta_track = MidiTrack()
        mid.tracks.append(meta_track)
        meta_track.name = "Conductor"

        # 设置速度
        tempo_value = mido.bpm2tempo(tempo)
        meta_track.append(MetaMessage('set_tempo', tempo=tempo_value, time=0))

        # 设置拍号 (4/4)
        meta_track.append(MetaMessage(
            'time_signature',
            numerator=4,
            denominator=4,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0
        ))

        # 将歌词添加到指挥轨道
        if embed_lyrics and lyrics:
            self._add_lyrics_events(meta_track, lyrics, tempo)

        # 轨道结束
        meta_track.append(MetaMessage('end_of_track', time=0))

        # 为每个启用的轨道创建 MIDI 轨道
        for track_config in track_layout.get_enabled_tracks():
            notes = tracks_notes.get(track_config.id, [])

            if not notes:
                logger.warning(f"轨道 {track_config.name} 没有音符，跳过")
                continue

            # 应用后处理
            if apply_post_processing:
                if track_count is not None and track_count > 0:
                    # 使用复杂度感知后处理
                    notes = self.post_process_notes_with_complexity(notes, tempo, track_count)
                else:
                    # 使用默认后处理
                    notes = self.post_process_notes(notes, tempo)

            # 创建 MIDI 轨道
            midi_track = self._create_track_v2(track_config, notes, tempo)
            mid.tracks.append(midi_track)

        # 添加踏板事件（钢琴模式）
        if pedals:
            self._add_pedal_events(mid, pedals, tempo)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # 保存 MIDI 文件
        mid.save(output_path)
        logger.info(f"MIDI 已保存: {output_path}")

        return output_path

    def _create_track_v2(
        self,
        track_config: TrackConfig,
        notes: List[NoteEvent],
        tempo: float
    ) -> MidiTrack:
        """
        使用轨道配置创建 MIDI 轨道

        参数:
            track_config: 轨道配置
            notes: 音符事件
            tempo: BPM

        返回:
            带音符的 MidiTrack
        """
        track = MidiTrack()
        # 使用 ASCII 兼容的轨道名（MIDI 元数据使用 latin-1 编码）
        track.name = track_config.id

        channel = track_config.midi_channel
        program = track_config.program

        # 音色变更（鼓组除外）
        if track_config.instrument != InstrumentType.DRUMS:
            track.append(Message(
                'program_change',
                channel=channel,
                program=program,
                time=0
            ))

        # 按开始时间排序音符
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        # 创建音符开/关事件
        events = []
        for note in sorted_notes:
            start_tick = self._time_to_ticks(note.start_time, tempo)
            end_tick = self._time_to_ticks(note.end_time, tempo)

            events.append({
                'type': 'note_on',
                'tick': start_tick,
                'note': note.pitch,
                'velocity': note.velocity,
                'channel': channel
            })
            events.append({
                'type': 'note_off',
                'tick': end_tick,
                'note': note.pitch,
                'velocity': 0,
                'channel': channel
            })

        # 按 tick 排序事件
        events.sort(key=lambda e: (e['tick'], e['type'] == 'note_on'))

        # 添加带增量时间的事件
        current_tick = 0
        for event in events:
            delta = max(0, event['tick'] - current_tick)

            if event['type'] == 'note_on':
                track.append(Message(
                    'note_on',
                    note=event['note'],
                    velocity=event['velocity'],
                    channel=event['channel'],
                    time=delta
                ))
            else:
                track.append(Message(
                    'note_off',
                    note=event['note'],
                    velocity=0,
                    channel=event['channel'],
                    time=delta
                ))

            current_tick = event['tick']

        # 轨道结束
        track.append(MetaMessage('end_of_track', time=0))

        logger.info(f"已创建 {track_config.name} 轨道，包含 {len(sorted_notes)} 个音符")

        return track

    def _add_pedal_events(
        self,
        midi_file: MidiFile,
        pedals: List[PedalEvent],
        tempo: float
    ) -> None:
        """
        将踏板事件添加到 MIDI 文件

        踏板使用 Control Change 消息:
        - CC64 (Sustain Pedal): 延音踏板
        - CC67 (Soft Pedal): 柔音踏板

        参数:
            midi_file: MIDI 文件对象
            pedals: 踏板事件列表
            tempo: BPM
        """
        if not pedals:
            return

        logger.info(f"正在添加 {len(pedals)} 个踏板事件")

        # 创建踏板轨道
        pedal_track = MidiTrack()
        pedal_track.name = "Pedals"
        midi_file.tracks.append(pedal_track)

        # 按类型分组踏板事件
        sustain_pedals = [p for p in pedals if p.pedal_type == "sustain"]
        soft_pedals = [p for p in pedals if p.pedal_type == "soft"]

        # 创建所有踏板事件
        events = []

        # 延音踏板事件 (CC64)
        for pedal in sustain_pedals:
            start_tick = self._time_to_ticks(pedal.start_time, tempo)
            end_tick = self._time_to_ticks(pedal.end_time, tempo)

            # 踩下踏板 (value 127)
            events.append({
                'tick': start_tick,
                'type': 'pedal_on',
                'cc': 64,
                'value': 127,
                'channel': 0
            })
            # 释放踏板 (value 0)
            events.append({
                'tick': end_tick,
                'type': 'pedal_off',
                'cc': 64,
                'value': 0,
                'channel': 0
            })

        # 柔音踏板事件 (CC67)
        for pedal in soft_pedals:
            start_tick = self._time_to_ticks(pedal.start_time, tempo)
            end_tick = self._time_to_ticks(pedal.end_time, tempo)

            events.append({
                'tick': start_tick,
                'type': 'pedal_on',
                'cc': 67,
                'value': 127,
                'channel': 0
            })
            events.append({
                'tick': end_tick,
                'type': 'pedal_off',
                'cc': 67,
                'value': 0,
                'channel': 0
            })

        # 按时间排序事件
        events.sort(key=lambda e: (e['tick'], e['type'] == 'pedal_on'))

        # 写入事件
        current_tick = 0
        for event in events:
            delta = max(0, event['tick'] - current_tick)

            pedal_track.append(Message(
                'control_change',
                channel=event['channel'],
                control=event['cc'],
                value=event['value'],
                time=delta
            ))

            current_tick = event['tick']

        # 轨道结束
        pedal_track.append(MetaMessage('end_of_track', time=0))

        sustain_count = len(sustain_pedals)
        soft_count = len(soft_pedals)
        logger.info(f"已添加踏板轨道: {sustain_count} 个延音踏板, {soft_count} 个柔音踏板")

    def export_lrc(
        self,
        lyrics: List[LyricEvent],
        output_path: str,
        title: str = "",
        artist: str = ""
    ) -> str:
        """
        将歌词导出为 LRC 格式

        参数:
            lyrics: 歌词事件列表
            output_path: 输出 LRC 文件路径
            title: 歌曲标题（可选）
            artist: 艺术家名称（可选）

        返回:
            LRC 文件路径
        """
        logger.info(f"正在导出 LRC: {output_path}")

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入元数据
            if title:
                f.write(f"[ti:{title}]\n")
            if artist:
                f.write(f"[ar:{artist}]\n")
            f.write("[by:音乐转MIDI转换器]\n")
            f.write("\n")

            # 按行分组歌词
            lines = self._group_lyrics_by_line(lyrics)

            # 写入歌词
            for line_start, line_text in lines:
                minutes = int(line_start // 60)
                seconds = line_start % 60
                f.write(f"[{minutes:02d}:{seconds:05.2f}]{line_text}\n")

        logger.info(f"LRC 已保存: {output_path}")
        return output_path

    def _group_lyrics_by_line(
        self,
        lyrics: List[LyricEvent],
        gap_threshold: float = 1.5
    ) -> List[tuple]:
        """
        根据时间间隔将歌词分组为行

        参数:
            lyrics: 歌词事件列表
            gap_threshold: 开始新行的时间间隔

        返回:
            (开始时间, 行文本) 元组列表
        """
        if not lyrics:
            return []

        sorted_lyrics = sorted(lyrics, key=lambda l: l.start_time)
        lines = []
        current_line_start = sorted_lyrics[0].start_time
        current_line_words = []

        for i, lyric in enumerate(sorted_lyrics):
            # 检查是否应该开始新行
            if i > 0:
                gap = lyric.start_time - sorted_lyrics[i-1].end_time
                if gap > gap_threshold:
                    # 保存当前行
                    if current_line_words:
                        lines.append((current_line_start, " ".join(current_line_words)))
                    current_line_start = lyric.start_time
                    current_line_words = []

            current_line_words.append(lyric.text)

        # 添加最后一行
        if current_line_words:
            lines.append((current_line_start, " ".join(current_line_words)))

        return lines
