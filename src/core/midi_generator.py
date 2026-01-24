"""
MIDI生成模块 - 支持歌词嵌入
"""
import logging
import os
from typing import List, Dict, Optional
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from src.models.data_models import Config, NoteEvent, LyricEvent, TrackType

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

    # 轨道名称
    TRACK_NAMES = {
        TrackType.DRUMS: "鼓",
        TrackType.BASS: "贝斯",
        TrackType.VOCALS: "人声",
        TrackType.OTHER: "其他乐器"
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
        meta_track.name = "指挥轨道"

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
        track.name = self.TRACK_NAMES.get(track_type, "轨道")

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
