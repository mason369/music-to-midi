"""
MIDI生成模块 - 支持歌词嵌入和后处理优化
"""
import logging
import os
from typing import List, Dict, Optional, Tuple
from copy import deepcopy
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from src.models.data_models import (
    Config, NoteEvent, TrackType,
    InstrumentType, TrackConfig, TrackLayout, PedalEvent
)

logger = logging.getLogger(__name__)


class MidiGenerator:
    """
    MIDI 文件生成器，支持多轨道

    功能特点:
    - 多轨道 MIDI 创建
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
        tempo: float,
        output_path: str
    ) -> str:
        """
        生成多轨道 MIDI 文件

        参数:
            tracks: 轨道类型到音符事件的字典
            tempo: BPM
            output_path: 输出 MIDI 文件路径

        返回:
            生成的 MIDI 文件路径
        """
        logger.info(f"正在生成 MIDI: {output_path}")

        # 创建 MIDI 文件
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)

        # 轨道 0: 速度（指挥轨道）
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
        min_velocity: int = 5  # 从20降低到5，保留更多细节音符
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
        max_polyphony: int = 40,  # 从25提高到40，更好支持钢琴等和声丰富的乐器
        instrument: Optional[InstrumentType] = None
    ) -> List[NoteEvent]:
        """
        限制最大复音数

        参数:
            notes: 音符列表
            max_polyphony: 最大同时发声的音符数
            instrument: 可选的乐器类型，用于选择合适的复音限制

        返回:
            限制复音后的音符列表
        """
        if not notes or max_polyphony <= 0:
            return notes

        # 乐器特定的复音限制
        polyphony_limits = {
            InstrumentType.PIANO: 40,      # 钢琴需要更高复音（双手演奏+踏板延音）
            InstrumentType.GUITAR: 8,      # 吉他6弦+泛音
            InstrumentType.STRINGS: 30,    # 弦乐合奏
            InstrumentType.SYNTH: 20,      # 合成器
            InstrumentType.BASS: 4,        # 贝斯单音为主
            InstrumentType.DRUMS: 16,      # 鼓组多元素
            InstrumentType.ORGAN: 20,      # 风琴
            InstrumentType.BRASS: 10,      # 铜管
            InstrumentType.WOODWIND: 8,    # 木管
            InstrumentType.HARP: 15,       # 竖琴
            InstrumentType.CHOIR: 20,      # 合唱
            InstrumentType.LEAD_SYNTH: 6,  # 主奏合成器
            InstrumentType.PAD_SYNTH: 12,  # 铺底合成器
            InstrumentType.PERCUSSION: 12, # 打击乐
        }

        # 如果提供了乐器类型，使用对应的限制
        if instrument and instrument in polyphony_limits:
            max_polyphony = polyphony_limits[instrument]

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
        gap_threshold: float = 0.010  # 10ms 间隔阈值（只合并真正的碎片，YourMT3层面已做去重）
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

    def post_process_minimal(
        self,
        notes: List[NoteEvent],
        tempo: float,
        is_drum: bool = False
    ) -> List[NoteEvent]:
        """
        最小化后处理 - 极致质量模式

        只移除明显的错误，保留所有细节:
        1. 移除 duration < 10ms 的极短音符（噪音）- 鼓除外
        2. 移除 velocity = 0 的无效音符
        3. 不做量化、合并、复音限制

        参数:
            notes: 音符列表
            tempo: BPM
            is_drum: 是否为鼓轨道（鼓不过滤短音符，因为鼓击天然很短）

        返回:
            后处理后的音符列表
        """
        if not notes:
            return notes

        initial_count = len(notes)

        # 只过滤明显的错误
        processed = []
        for note in notes:
            # 移除无效音符
            if note.velocity <= 0:
                continue

            # 移除极短音符（< 10ms，可能是噪音）
            # 注意：鼓音符不过滤短时长，因为踩镲、军鼓等天然持续时间极短
            if not is_drum and note.duration < 0.010:
                continue

            # 确保音高在有效范围
            if note.pitch < 0 or note.pitch > 127:
                continue

            processed.append(note)

        removed_count = initial_count - len(processed)
        if removed_count > 0:
            logger.info(f"最小化后处理: 移除 {removed_count} 个无效音符 (保留 {len(processed)}/{initial_count} = {len(processed)/initial_count:.1%})")
        else:
            logger.info(f"最小化后处理: 保留全部 {len(processed)} 个音符")

        return processed

    def post_process_by_quality(
        self,
        notes: List[NoteEvent],
        tempo: float,
        quality: str = "balanced",
        instrument: Optional[InstrumentType] = None,
        is_drum: bool = False
    ) -> List[NoteEvent]:
        """
        根据质量模式选择后处理策略

        参数:
            notes: 音符列表
            tempo: BPM
            quality: 质量模式 ("fast", "balanced", "best")
            instrument: 可选的乐器类型
            is_drum: 是否为鼓轨道（鼓不过滤短音符）

        返回:
            后处理后的音符列表
        """
        if not notes:
            return notes

        initial_count = len(notes)

        if quality == "fast":
            # 快速模式：无后处理
            logger.info(f"快速模式: 跳过后处理 ({initial_count} 个音符)")
            return notes

        elif quality == "best":
            # 极致质量模式：最小化后处理
            return self.post_process_minimal(notes, tempo, is_drum=is_drum)

        else:  # balanced
            # 平衡模式：轻量后处理
            processed = deepcopy(notes)

            # 1. 移除无效音符（鼓不过滤短音符）
            if is_drum:
                processed = [n for n in processed if n.velocity > 0]
            else:
                processed = [n for n in processed if n.velocity > 0 and n.duration >= 0.005]

            # 2. 轻度去重（只移除完全重叠的）
            processed = self._remove_duplicate_notes(processed, time_threshold=0.015)

            # 3. 轻度力度平滑（保留动态）
            if len(processed) > 5:
                processed = self._smooth_velocity(processed, window_size=3, min_velocity=1)

            # 4. 适度复音限制（针对乐器类型）
            if instrument:
                processed = self._limit_polyphony(processed, instrument=instrument)

            logger.info(f"平衡模式后处理: {initial_count} -> {len(processed)} 个音符")
            return processed

    def _get_post_process_params(self, track_count: int) -> dict:
        """
        根据轨道数量获取后处理参数

        原理：
        - max_polyphony: 降低 → 减少同时发声数（简化）
        - min_velocity: 提高 → 过滤弱音（简化）
        - gap_threshold: 增加 → 更多合并（简化）
        - quantize_grid: 粗网格 → 节奏更规整（简化）

        注意：这些参数已经过优化，减少后处理损失

        参数:
            track_count: 目标轨道数量（1-4）

        返回:
            包含后处理参数的字典
        """
        params = {
            1: {
                'max_polyphony': 6,       # 单旋律+和弦（从4提高）
                'min_velocity': 25,       # 从45降低，保留更多音符
                'gap_threshold': 0.040,   # 从80ms降低到40ms（YourMT3已去重）
                'quantize_grid': '1/16',  # 从1/8细化到1/16
            },
            2: {
                'max_polyphony': 12,      # 每手6个音（从8提高）
                'min_velocity': 15,       # 从35降低
                'gap_threshold': 0.025,   # 从50ms降低到25ms
                'quantize_grid': '1/16',
            },
            3: {
                'max_polyphony': 25,      # 从15提高
                'min_velocity': 10,       # 从28降低
                'gap_threshold': 0.015,   # 从30ms降低到15ms
                'quantize_grid': '1/32',
            },
            4: {
                'max_polyphony': 40,      # 从25提高到40
                'min_velocity': 5,        # 从20降低到5
                'gap_threshold': 0.010,   # 从20ms降低到10ms
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

        # 轨道 0: 速度（指挥轨道）
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

    def generate_from_precise_instruments(
        self,
        program_notes: Dict[int, List[NoteEvent]],
        tempo: float,
        output_path: str
    ) -> str:
        """
        从精确 GM 程序号分组的音符生成 MIDI

        每个 GM 程序号 (0-127) 对应一个独立轨道，实现精确的乐器识别。

        参数:
            program_notes: GM程序号到音符列表的字典
            tempo: BPM
            output_path: 输出 MIDI 文件路径

        返回:
            输出 MIDI 文件路径
        """
        from src.models.gm_instruments import get_instrument_name

        logger.info(f"正在生成精确乐器 MIDI: {output_path}")
        logger.info(f"包含 {len(program_notes)} 种精确乐器")

        # 创建 MIDI 文件
        midi = MidiFile(type=1, ticks_per_beat=self.ticks_per_beat)

        # 创建主轨道（速度和时间签名）
        main_track = MidiTrack()
        midi.tracks.append(main_track)

        # 添加速度
        tempo_microseconds = int(60_000_000 / tempo)
        main_track.append(MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))

        # 添加时间签名
        main_track.append(MetaMessage(
            'time_signature',
            numerator=4, denominator=4,
            clocks_per_click=24, notated_32nd_notes_per_beat=8,
            time=0
        ))

        main_track.append(MetaMessage('end_of_track', time=0))

        # 为每个 GM 程序号创建轨道
        channel = 0
        for program, notes in sorted(program_notes.items()):
            if not notes:
                continue

            # 鼓组使用通道 9
            is_drums = (program >= 112 and program <= 119) or (program == 47)  # 打击乐器家族或定音鼓
            if is_drums:
                midi_channel = 9
            else:
                midi_channel = channel
                channel += 1
                if channel == 9:
                    channel = 10  # 跳过鼓通道
                if channel > 15:
                    logger.warning(f"MIDI 通道已用尽，跳过程序 {program}")
                    continue

            # 获取乐器名称
            inst_name = get_instrument_name(program, "zh_CN")
            track_name = f"{program:03d}_{inst_name}"

            # 创建轨道
            track = MidiTrack()
            midi.tracks.append(track)

            # 轨道名称（使用 ASCII）
            ascii_name = f"GM{program:03d}"
            track.append(MetaMessage('track_name', name=ascii_name, time=0))

            # 音色变更（鼓组除外）
            if not is_drums:
                track.append(Message(
                    'program_change',
                    channel=midi_channel,
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
                    'channel': midi_channel
                })
                events.append({
                    'type': 'note_off',
                    'tick': end_tick,
                    'note': note.pitch,
                    'velocity': 0,
                    'channel': midi_channel
                })

            # 按 tick 排序
            events.sort(key=lambda e: (e['tick'], e['type'] == 'note_on'))

            # 写入事件
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

            track.append(MetaMessage('end_of_track', time=0))

            logger.info(f"轨道 GM{program:03d} ({inst_name}): {len(notes)} 个音符，通道 {midi_channel}")

        # 保存文件
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        midi.save(output_path)

        logger.info(f"精确乐器 MIDI 已保存: {output_path}")
        return output_path

    def generate_from_precise_instruments_v2(
        self,
        instrument_notes: Dict[int, List[NoteEvent]],
        drum_notes: Dict[int, List[NoteEvent]],
        tempo: float,
        output_path: str,
        quality: str = "best"
    ) -> str:
        """
        从精确 GM 程序号分组的音符生成高质量 MIDI（v2）

        增强功能:
        - 支持质量模式选择
        - 智能通道分配（超过15种乐器时合并同族）
        - 人声特殊处理（program 100/101）
        - 保留所有音符，不会因通道限制丢失

        参数:
            instrument_notes: GM程序号(0-127)到音符列表的字典
            drum_notes: 鼓音高(35-81)到音符列表的字典
            tempo: BPM
            output_path: 输出 MIDI 文件路径
            quality: 质量模式 ("fast", "balanced", "best")

        返回:
            输出 MIDI 文件路径
        """
        from src.models.gm_instruments import get_instrument_name, get_instrument_family, GMFamily

        total_instrument_notes = sum(len(notes) for notes in instrument_notes.values())
        total_drum_notes = sum(len(notes) for notes in drum_notes.values())

        logger.info(f"正在生成极致精度 MIDI: {output_path}")
        logger.info(f"质量模式: {quality}")
        logger.info(f"乐器: {len(instrument_notes)} 种，共 {total_instrument_notes} 个音符")
        logger.info(f"鼓: {len(drum_notes)} 种音高，共 {total_drum_notes} 个音符")

        # 创建 MIDI 文件
        midi = MidiFile(type=1, ticks_per_beat=self.ticks_per_beat)

        # 创建主轨道（速度和时间签名）
        main_track = MidiTrack()
        midi.tracks.append(main_track)

        # 添加速度
        tempo_microseconds = int(60_000_000 / tempo)
        main_track.append(MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))

        # 添加时间签名
        main_track.append(MetaMessage(
            'time_signature',
            numerator=4, denominator=4,
            clocks_per_click=24, notated_32nd_notes_per_beat=8,
            time=0
        ))

        main_track.append(MetaMessage('end_of_track', time=0))

        # 智能通道分配策略
        # MIDI 有 16 个通道，通道 9 (索引) 专用于鼓
        # 所以最多 15 个乐器通道
        MAX_INSTRUMENT_CHANNELS = 15

        # 分离人声轨道（YourMT3 使用 program 100/101 表示人声）
        # 这些不是标准 GM 程序号，需要特殊处理
        SINGING_PROGRAMS = {100, 101}  # FX 5 (brightness) 和 FX 6 (goblins) 被 YourMT3 用于人声

        singing_notes = {}
        normal_instruments = {}

        for program, notes in instrument_notes.items():
            if program in SINGING_PROGRAMS:
                singing_notes[program] = notes
            else:
                normal_instruments[program] = notes

        # 按音符数量排序乐器（优先分配给音符多的）
        sorted_instruments = sorted(
            normal_instruments.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        # 如果乐器超过限制，需要智能合并
        if len(sorted_instruments) > MAX_INSTRUMENT_CHANNELS:
            logger.info(f"乐器数量 ({len(sorted_instruments)}) 超过通道限制 ({MAX_INSTRUMENT_CHANNELS})，启用智能合并")
            sorted_instruments = self._merge_instruments_for_channels(
                sorted_instruments, MAX_INSTRUMENT_CHANNELS
            )

        # 统计
        initial_total = total_instrument_notes + total_drum_notes
        final_total = 0

        # 通道分配
        channel_index = 0
        used_channels = []

        # 为每个乐器创建轨道
        for program, notes in sorted_instruments:
            if not notes:
                continue

            # 分配 MIDI 通道（跳过通道 9）
            if channel_index == 9:
                channel_index = 10
            if channel_index >= 16:
                logger.error(f"通道分配错误：程序 {program}")
                continue

            midi_channel = channel_index
            used_channels.append(midi_channel)
            channel_index += 1

            # 根据质量模式进行后处理
            if quality == "best":
                # 最佳质量：最小化后处理
                processed_notes = self.post_process_minimal(notes, tempo)
            else:
                processed_notes = self.post_process_by_quality(notes, tempo, quality)

            final_total += len(processed_notes)

            if not processed_notes:
                continue

            # 获取乐器名称
            inst_name = get_instrument_name(program, "zh_CN")
            track_name = f"GM{program:03d}"

            # 创建轨道
            track = MidiTrack()
            midi.tracks.append(track)

            track.append(MetaMessage('track_name', name=track_name, time=0))

            # 音色变更
            track.append(Message(
                'program_change',
                channel=midi_channel,
                program=program,
                time=0
            ))

            # 写入音符
            self._write_notes_to_track(track, processed_notes, midi_channel, tempo)

            track.append(MetaMessage('end_of_track', time=0))

            logger.info(f"轨道 {track_name} ({inst_name}): {len(processed_notes)} 个音符，通道 {midi_channel}")

        # 处理人声轨道
        if singing_notes:
            total_singing_notes = sum(len(n) for n in singing_notes.values())
            logger.info(f"检测到人声轨道: {len(singing_notes)} 个程序, {total_singing_notes} 个音符")

            # 合并所有人声到一个轨道
            all_singing_notes = []
            for notes in singing_notes.values():
                all_singing_notes.extend(notes)
            all_singing_notes.sort(key=lambda n: n.start_time)

            if all_singing_notes:
                # 为人声分配通道
                if channel_index == 9:
                    channel_index = 10
                if channel_index < 16:
                    midi_channel = channel_index
                    channel_index += 1

                    # 后处理
                    if quality == "best":
                        processed_notes = self.post_process_minimal(all_singing_notes, tempo)
                    else:
                        processed_notes = self.post_process_by_quality(all_singing_notes, tempo, quality)

                    final_total += len(processed_notes)

                    # 创建人声轨道
                    singing_track = MidiTrack()
                    midi.tracks.append(singing_track)

                    singing_track.append(MetaMessage('track_name', name="Vocals", time=0))

                    # 使用 Choir Aahs (52) 作为人声音色
                    singing_track.append(Message(
                        'program_change',
                        channel=midi_channel,
                        program=52,  # Choir Aahs
                        time=0
                    ))

                    self._write_notes_to_track(singing_track, processed_notes, midi_channel, tempo)
                    singing_track.append(MetaMessage('end_of_track', time=0))

                    logger.info(f"人声轨道: {len(processed_notes)} 个音符，通道 {midi_channel}，音色 Choir Aahs")
                else:
                    logger.warning(f"无法分配人声通道，跳过 {len(all_singing_notes)} 个音符")

        # 处理鼓轨道
        if drum_notes:
            # 合并所有鼓音符到一个轨道
            all_drum_notes = []
            for pitch, notes in drum_notes.items():
                all_drum_notes.extend(notes)
            all_drum_notes.sort(key=lambda n: n.start_time)

            if all_drum_notes:
                # 后处理 - 鼓轨道使用 is_drum=True，不过滤短音符
                if quality == "best":
                    processed_drums = self.post_process_minimal(all_drum_notes, tempo, is_drum=True)
                else:
                    processed_drums = self.post_process_by_quality(all_drum_notes, tempo, quality, is_drum=True)

                final_total += len(processed_drums)

                if processed_drums:
                    drum_track = MidiTrack()
                    midi.tracks.append(drum_track)

                    drum_track.append(MetaMessage('track_name', name="Drums", time=0))

                    # 鼓不需要音色变更，使用通道 9
                    self._write_notes_to_track(drum_track, processed_drums, 9, tempo)

                    drum_track.append(MetaMessage('end_of_track', time=0))

                    logger.info(f"鼓轨道: {len(processed_drums)} 个音符，通道 9")

        # 保存文件
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        midi.save(output_path)

        # 输出保留率统计
        retention_rate = final_total / max(initial_total, 1)
        logger.info(f"极致精度 MIDI 已保存: {output_path}")
        logger.info(f"音符保留率: {final_total}/{initial_total} = {retention_rate:.1%}")
        logger.info(f"轨道总数: {len(midi.tracks)} (含指挥轨道)")

        return output_path

    def _merge_instruments_for_channels(
        self,
        sorted_instruments: List[tuple],
        max_channels: int
    ) -> List[tuple]:
        """
        当乐器数量超过通道限制时，智能合并同族乐器

        策略:
        1. 保留音符数最多的乐器
        2. 将音符少的同族乐器合并到族代表乐器

        参数:
            sorted_instruments: 按音符数量排序的 (program, notes) 列表
            max_channels: 最大通道数

        返回:
            合并后的乐器列表
        """
        from src.models.gm_instruments import get_instrument_family, GMFamily

        if len(sorted_instruments) <= max_channels:
            return sorted_instruments

        # 按乐器族分组
        family_groups = {}
        for program, notes in sorted_instruments:
            family = get_instrument_family(program)
            if family is None:
                family = GMFamily.SOUND_EFFECTS  # 默认

            if family not in family_groups:
                family_groups[family] = []
            family_groups[family].append((program, notes))

        # 每个族选择一个代表（音符最多的）
        result = []
        merged_count = 0

        for family, instruments in family_groups.items():
            if not instruments:
                continue

            # 按音符数量排序
            instruments.sort(key=lambda x: len(x[1]), reverse=True)

            # 代表乐器（音符最多）
            main_program, main_notes = instruments[0]
            merged_notes = list(main_notes)

            # 合并其他乐器的音符
            for program, notes in instruments[1:]:
                merged_notes.extend(notes)
                merged_count += 1

            # 按时间排序
            merged_notes.sort(key=lambda n: n.start_time)

            result.append((main_program, merged_notes))

        # 按音符数量排序
        result.sort(key=lambda x: len(x[1]), reverse=True)

        # 如果仍然超过限制，截取前 max_channels 个
        if len(result) > max_channels:
            # 将多余的合并到 "其他" 类别
            extra = result[max_channels:]
            result = result[:max_channels]

            # 将多余的音符合并到最后一个轨道
            if extra:
                extra_notes = []
                for _, notes in extra:
                    extra_notes.extend(notes)

                if extra_notes:
                    extra_notes.sort(key=lambda n: n.start_time)
                    # 添加到最后一个轨道或创建新轨道
                    last_program, last_notes = result[-1]
                    last_notes.extend(extra_notes)
                    last_notes.sort(key=lambda n: n.start_time)
                    result[-1] = (last_program, last_notes)

                logger.info(f"合并了 {len(extra)} 个额外轨道到其他乐器")

        logger.info(f"智能合并: {len(sorted_instruments)} -> {len(result)} 个轨道 (合并 {merged_count} 个同族乐器)")

        return result

    def _write_notes_to_track(
        self,
        track: MidiTrack,
        notes: List[NoteEvent],
        channel: int,
        tempo: float
    ) -> None:
        """
        将音符写入 MIDI 轨道

        参数:
            track: MIDI 轨道
            notes: 音符列表
            channel: MIDI 通道
            tempo: BPM
        """
        # 按开始时间排序
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

        # 按 tick 排序
        events.sort(key=lambda e: (e['tick'], e['type'] == 'note_on'))

        # 写入事件
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
