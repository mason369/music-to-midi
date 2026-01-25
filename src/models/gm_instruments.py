"""
General MIDI (GM) 乐器定义和映射

提供完整的 128 种 GM 乐器定义，按照 16 个乐器族分类。
支持 YourMT3 等高级转写器进行精确的乐器识别。

参考: https://www.midi.org/specifications/midi1-specifications/m1-v4-2-1-midi-1-0-detailed-specification
"""

from enum import Enum
from typing import Dict, List, Optional
from src.models.data_models import InstrumentType


class GMFamily(Enum):
    """GM 乐器族（16 类）"""
    PIANO = 0           # 钢琴
    CHROMATIC = 1       # 半音阶打击乐器（木琴、钟琴等）
    ORGAN = 2           # 风琴
    GUITAR = 3          # 吉他
    BASS = 4            # 贝斯
    STRINGS = 5         # 弦乐
    ENSEMBLE = 6        # 合奏
    BRASS = 7           # 铜管
    REED = 8            # 簧管
    PIPE = 9            # 哨笛
    SYNTH_LEAD = 10     # 合成主音
    SYNTH_PAD = 11      # 合成铺底
    SYNTH_EFFECTS = 12  # 合成效果
    ETHNIC = 13         # 民族乐器
    PERCUSSIVE = 14     # 打击乐（非鼓）
    SOUND_EFFECTS = 15  # 音效


class GMInstrument:
    """GM 乐器定义"""
    def __init__(self, program: int, name_en: str, name_zh: str, family: GMFamily):
        self.program = program  # MIDI Program Number (0-127)
        self.name_en = name_en
        self.name_zh = name_zh
        self.family = family


# 完整的 128 种 GM 乐器定义 (列表形式，用于遍历)
_GM_INSTRUMENTS_LIST: List[GMInstrument] = [
    # Piano (0-7)
    GMInstrument(0, "Acoustic Grand Piano", "大钢琴", GMFamily.PIANO),
    GMInstrument(1, "Bright Acoustic Piano", "明亮钢琴", GMFamily.PIANO),
    GMInstrument(2, "Electric Grand Piano", "电钢琴", GMFamily.PIANO),
    GMInstrument(3, "Honky-tonk Piano", "酒吧钢琴", GMFamily.PIANO),
    GMInstrument(4, "Electric Piano 1", "电子钢琴1", GMFamily.PIANO),
    GMInstrument(5, "Electric Piano 2", "电子钢琴2", GMFamily.PIANO),
    GMInstrument(6, "Harpsichord", "拨弦古钢琴", GMFamily.PIANO),
    GMInstrument(7, "Clavinet", "克拉维内特琴", GMFamily.PIANO),

    # Chromatic Percussion (8-15)
    GMInstrument(8, "Celesta", "钢片琴", GMFamily.CHROMATIC),
    GMInstrument(9, "Glockenspiel", "钟琴", GMFamily.CHROMATIC),
    GMInstrument(10, "Music Box", "八音盒", GMFamily.CHROMATIC),
    GMInstrument(11, "Vibraphone", "颤音琴", GMFamily.CHROMATIC),
    GMInstrument(12, "Marimba", "马林巴琴", GMFamily.CHROMATIC),
    GMInstrument(13, "Xylophone", "木琴", GMFamily.CHROMATIC),
    GMInstrument(14, "Tubular Bells", "管钟", GMFamily.CHROMATIC),
    GMInstrument(15, "Dulcimer", "扬琴", GMFamily.CHROMATIC),

    # Organ (16-23)
    GMInstrument(16, "Drawbar Organ", "拉杆风琴", GMFamily.ORGAN),
    GMInstrument(17, "Percussive Organ", "打击风琴", GMFamily.ORGAN),
    GMInstrument(18, "Rock Organ", "摇滚风琴", GMFamily.ORGAN),
    GMInstrument(19, "Church Organ", "教堂风琴", GMFamily.ORGAN),
    GMInstrument(20, "Reed Organ", "簧风琴", GMFamily.ORGAN),
    GMInstrument(21, "Accordion", "手风琴", GMFamily.ORGAN),
    GMInstrument(22, "Harmonica", "口琴", GMFamily.ORGAN),
    GMInstrument(23, "Tango Accordion", "探戈手风琴", GMFamily.ORGAN),

    # Guitar (24-31)
    GMInstrument(24, "Acoustic Guitar (nylon)", "尼龙弦吉他", GMFamily.GUITAR),
    GMInstrument(25, "Acoustic Guitar (steel)", "钢弦吉他", GMFamily.GUITAR),
    GMInstrument(26, "Electric Guitar (jazz)", "爵士电吉他", GMFamily.GUITAR),
    GMInstrument(27, "Electric Guitar (clean)", "清音电吉他", GMFamily.GUITAR),
    GMInstrument(28, "Electric Guitar (muted)", "闷音电吉他", GMFamily.GUITAR),
    GMInstrument(29, "Overdriven Guitar", "过载吉他", GMFamily.GUITAR),
    GMInstrument(30, "Distortion Guitar", "失真吉他", GMFamily.GUITAR),
    GMInstrument(31, "Guitar Harmonics", "吉他泛音", GMFamily.GUITAR),

    # Bass (32-39)
    GMInstrument(32, "Acoustic Bass", "原声贝斯", GMFamily.BASS),
    GMInstrument(33, "Electric Bass (finger)", "指弹电贝斯", GMFamily.BASS),
    GMInstrument(34, "Electric Bass (pick)", "拨片电贝斯", GMFamily.BASS),
    GMInstrument(35, "Fretless Bass", "无品贝斯", GMFamily.BASS),
    GMInstrument(36, "Slap Bass 1", "拍击贝斯1", GMFamily.BASS),
    GMInstrument(37, "Slap Bass 2", "拍击贝斯2", GMFamily.BASS),
    GMInstrument(38, "Synth Bass 1", "合成贝斯1", GMFamily.BASS),
    GMInstrument(39, "Synth Bass 2", "合成贝斯2", GMFamily.BASS),

    # Strings (40-47)
    GMInstrument(40, "Violin", "小提琴", GMFamily.STRINGS),
    GMInstrument(41, "Viola", "中提琴", GMFamily.STRINGS),
    GMInstrument(42, "Cello", "大提琴", GMFamily.STRINGS),
    GMInstrument(43, "Contrabass", "低音提琴", GMFamily.STRINGS),
    GMInstrument(44, "Tremolo Strings", "弦乐震音", GMFamily.STRINGS),
    GMInstrument(45, "Pizzicato Strings", "弦乐拨奏", GMFamily.STRINGS),
    GMInstrument(46, "Orchestral Harp", "竖琴", GMFamily.STRINGS),
    GMInstrument(47, "Timpani", "定音鼓", GMFamily.STRINGS),

    # Ensemble (48-55)
    GMInstrument(48, "String Ensemble 1", "弦乐合奏1", GMFamily.ENSEMBLE),
    GMInstrument(49, "String Ensemble 2", "弦乐合奏2", GMFamily.ENSEMBLE),
    GMInstrument(50, "Synth Strings 1", "合成弦乐1", GMFamily.ENSEMBLE),
    GMInstrument(51, "Synth Strings 2", "合成弦乐2", GMFamily.ENSEMBLE),
    GMInstrument(52, "Choir Aahs", "人声合唱", GMFamily.ENSEMBLE),
    GMInstrument(53, "Voice Oohs", "人声", GMFamily.ENSEMBLE),
    GMInstrument(54, "Synth Voice", "合成人声", GMFamily.ENSEMBLE),
    GMInstrument(55, "Orchestra Hit", "管弦乐打击", GMFamily.ENSEMBLE),

    # Brass (56-63)
    GMInstrument(56, "Trumpet", "小号", GMFamily.BRASS),
    GMInstrument(57, "Trombone", "长号", GMFamily.BRASS),
    GMInstrument(58, "Tuba", "大号", GMFamily.BRASS),
    GMInstrument(59, "Muted Trumpet", "弱音小号", GMFamily.BRASS),
    GMInstrument(60, "French Horn", "圆号", GMFamily.BRASS),
    GMInstrument(61, "Brass Section", "铜管组", GMFamily.BRASS),
    GMInstrument(62, "Synth Brass 1", "合成铜管1", GMFamily.BRASS),
    GMInstrument(63, "Synth Brass 2", "合成铜管2", GMFamily.BRASS),

    # Reed (64-71)
    GMInstrument(64, "Soprano Sax", "高音萨克斯", GMFamily.REED),
    GMInstrument(65, "Alto Sax", "中音萨克斯", GMFamily.REED),
    GMInstrument(66, "Tenor Sax", "次中音萨克斯", GMFamily.REED),
    GMInstrument(67, "Baritone Sax", "上低音萨克斯", GMFamily.REED),
    GMInstrument(68, "Oboe", "双簧管", GMFamily.REED),
    GMInstrument(69, "English Horn", "英国管", GMFamily.REED),
    GMInstrument(70, "Bassoon", "巴松管", GMFamily.REED),
    GMInstrument(71, "Clarinet", "单簧管", GMFamily.REED),

    # Pipe (72-79)
    GMInstrument(72, "Piccolo", "短笛", GMFamily.PIPE),
    GMInstrument(73, "Flute", "长笛", GMFamily.PIPE),
    GMInstrument(74, "Recorder", "竖笛", GMFamily.PIPE),
    GMInstrument(75, "Pan Flute", "排箫", GMFamily.PIPE),
    GMInstrument(76, "Blown Bottle", "吹瓶", GMFamily.PIPE),
    GMInstrument(77, "Shakuhachi", "尺八", GMFamily.PIPE),
    GMInstrument(78, "Whistle", "口哨", GMFamily.PIPE),
    GMInstrument(79, "Ocarina", "陶笛", GMFamily.PIPE),

    # Synth Lead (80-87)
    GMInstrument(80, "Lead 1 (square)", "方波主音", GMFamily.SYNTH_LEAD),
    GMInstrument(81, "Lead 2 (sawtooth)", "锯齿波主音", GMFamily.SYNTH_LEAD),
    GMInstrument(82, "Lead 3 (calliope)", "汽笛风琴主音", GMFamily.SYNTH_LEAD),
    GMInstrument(83, "Lead 4 (chiff)", "吹管主音", GMFamily.SYNTH_LEAD),
    GMInstrument(84, "Lead 5 (charang)", "吉他主音", GMFamily.SYNTH_LEAD),
    GMInstrument(85, "Lead 6 (voice)", "人声主音", GMFamily.SYNTH_LEAD),
    GMInstrument(86, "Lead 7 (fifths)", "五度主音", GMFamily.SYNTH_LEAD),
    GMInstrument(87, "Lead 8 (bass + lead)", "贝斯主音", GMFamily.SYNTH_LEAD),

    # Synth Pad (88-95)
    GMInstrument(88, "Pad 1 (new age)", "新世纪铺底", GMFamily.SYNTH_PAD),
    GMInstrument(89, "Pad 2 (warm)", "温暖铺底", GMFamily.SYNTH_PAD),
    GMInstrument(90, "Pad 3 (polysynth)", "复音合成铺底", GMFamily.SYNTH_PAD),
    GMInstrument(91, "Pad 4 (choir)", "合唱铺底", GMFamily.SYNTH_PAD),
    GMInstrument(92, "Pad 5 (bowed)", "弓弦铺底", GMFamily.SYNTH_PAD),
    GMInstrument(93, "Pad 6 (metallic)", "金属铺底", GMFamily.SYNTH_PAD),
    GMInstrument(94, "Pad 7 (halo)", "光环铺底", GMFamily.SYNTH_PAD),
    GMInstrument(95, "Pad 8 (sweep)", "扫频铺底", GMFamily.SYNTH_PAD),

    # Synth Effects (96-103)
    GMInstrument(96, "FX 1 (rain)", "雨声", GMFamily.SYNTH_EFFECTS),
    GMInstrument(97, "FX 2 (soundtrack)", "原声带", GMFamily.SYNTH_EFFECTS),
    GMInstrument(98, "FX 3 (crystal)", "水晶", GMFamily.SYNTH_EFFECTS),
    GMInstrument(99, "FX 4 (atmosphere)", "氛围", GMFamily.SYNTH_EFFECTS),
    GMInstrument(100, "FX 5 (brightness)", "明亮", GMFamily.SYNTH_EFFECTS),
    GMInstrument(101, "FX 6 (goblins)", "鬼魅", GMFamily.SYNTH_EFFECTS),
    GMInstrument(102, "FX 7 (echoes)", "回声", GMFamily.SYNTH_EFFECTS),
    GMInstrument(103, "FX 8 (sci-fi)", "科幻", GMFamily.SYNTH_EFFECTS),

    # Ethnic (104-111)
    GMInstrument(104, "Sitar", "西塔尔琴", GMFamily.ETHNIC),
    GMInstrument(105, "Banjo", "班卓琴", GMFamily.ETHNIC),
    GMInstrument(106, "Shamisen", "三味线", GMFamily.ETHNIC),
    GMInstrument(107, "Koto", "古筝", GMFamily.ETHNIC),
    GMInstrument(108, "Kalimba", "卡林巴琴", GMFamily.ETHNIC),
    GMInstrument(109, "Bag pipe", "风笛", GMFamily.ETHNIC),
    GMInstrument(110, "Fiddle", "小提琴", GMFamily.ETHNIC),
    GMInstrument(111, "Shanai", "唢呐", GMFamily.ETHNIC),

    # Percussive (112-119)
    GMInstrument(112, "Tinkle Bell", "铃铛", GMFamily.PERCUSSIVE),
    GMInstrument(113, "Agogo", "阿哥哥鼓", GMFamily.PERCUSSIVE),
    GMInstrument(114, "Steel Drums", "钢鼓", GMFamily.PERCUSSIVE),
    GMInstrument(115, "Woodblock", "木鱼", GMFamily.PERCUSSIVE),
    GMInstrument(116, "Taiko Drum", "太鼓", GMFamily.PERCUSSIVE),
    GMInstrument(117, "Melodic Tom", "旋律嗵鼓", GMFamily.PERCUSSIVE),
    GMInstrument(118, "Synth Drum", "合成鼓", GMFamily.PERCUSSIVE),
    GMInstrument(119, "Reverse Cymbal", "反镲", GMFamily.PERCUSSIVE),

    # Sound Effects (120-127)
    GMInstrument(120, "Guitar Fret Noise", "吉他品噪音", GMFamily.SOUND_EFFECTS),
    GMInstrument(121, "Breath Noise", "呼吸声", GMFamily.SOUND_EFFECTS),
    GMInstrument(122, "Seashore", "海浪", GMFamily.SOUND_EFFECTS),
    GMInstrument(123, "Bird Tweet", "鸟鸣", GMFamily.SOUND_EFFECTS),
    GMInstrument(124, "Telephone Ring", "电话铃", GMFamily.SOUND_EFFECTS),
    GMInstrument(125, "Helicopter", "直升机", GMFamily.SOUND_EFFECTS),
    GMInstrument(126, "Applause", "掌声", GMFamily.SOUND_EFFECTS),
    GMInstrument(127, "Gunshot", "枪声", GMFamily.SOUND_EFFECTS),
]


# 创建快速查找字典（供外部使用）
_PROGRAM_TO_INSTRUMENT: Dict[int, GMInstrument] = {
    inst.program: inst for inst in _GM_INSTRUMENTS_LIST
}

# GM_INSTRUMENTS 作为字典导出，支持 `program in GM_INSTRUMENTS` 语法
GM_INSTRUMENTS: Dict[int, GMInstrument] = _PROGRAM_TO_INSTRUMENT


# YourMT3 扩展程序号定义 (超出 GM 标准 0-127 范围)
# 这些是 YourMT3 特有的，用于表示标准 GM 不支持的乐器类型
YOURMT3_EXTENDED_PROGRAMS = {
    100: ("Singing Voice", "人声(主旋律)"),
    101: ("Singing Voice (chorus)", "人声(和声)"),
    128: ("Drums", "鼓组"),  # YourMT3 内部用于标记鼓
}


def get_instrument_name(program: int, language: str = "zh") -> str:
    """
    根据 MIDI Program Number 获取乐器名称

    支持标准 GM (0-127) 和 YourMT3 扩展程序号 (100, 101 为人声)

    Args:
        program: MIDI Program Number (0-127 标准, 100/101 为 YourMT3 人声)
        language: 语言 ('zh' 或 'en')

    Returns:
        乐器名称
    """
    # 检查 YourMT3 扩展程序号
    if program in YOURMT3_EXTENDED_PROGRAMS:
        names = YOURMT3_EXTENDED_PROGRAMS[program]
        return names[1] if language.startswith("zh") else names[0]

    if program < 0 or program > 127:
        return "未知乐器" if language.startswith("zh") else "Unknown"

    inst = _PROGRAM_TO_INSTRUMENT.get(program)
    if inst is None:
        return f"Program {program}"

    return inst.name_zh if language.startswith("zh") else inst.name_en


def get_instrument_family(program: int) -> Optional[GMFamily]:
    """
    根据 MIDI Program Number 获取乐器族

    支持标准 GM (0-127) 和 YourMT3 扩展程序号

    Args:
        program: MIDI Program Number (0-127 标准, 100/101 为人声)

    Returns:
        乐器族枚举，如果无效返回 None
    """
    # YourMT3 人声程序号特殊处理
    if program in (100, 101):
        return GMFamily.ENSEMBLE  # 人声归类到 Ensemble (合唱)

    if program < 0 or program > 127:
        return None

    inst = _PROGRAM_TO_INSTRUMENT.get(program)
    return inst.family if inst else None


def family_to_simple_type(family: GMFamily) -> InstrumentType:
    """
    将 GM 乐器族映射到简化的乐器类型

    Args:
        family: GM 乐器族

    Returns:
        InstrumentType 枚举值
    """
    # 映射表
    mapping = {
        GMFamily.PIANO: InstrumentType.PIANO,
        GMFamily.CHROMATIC: InstrumentType.PERCUSSION,
        GMFamily.ORGAN: InstrumentType.ORGAN,
        GMFamily.GUITAR: InstrumentType.GUITAR,
        GMFamily.BASS: InstrumentType.BASS,
        GMFamily.STRINGS: InstrumentType.STRINGS,
        GMFamily.ENSEMBLE: InstrumentType.STRINGS,
        GMFamily.BRASS: InstrumentType.BRASS,
        GMFamily.REED: InstrumentType.WOODWIND,
        GMFamily.PIPE: InstrumentType.WOODWIND,
        GMFamily.SYNTH_LEAD: InstrumentType.LEAD_SYNTH,
        GMFamily.SYNTH_PAD: InstrumentType.PAD_SYNTH,
        GMFamily.SYNTH_EFFECTS: InstrumentType.SYNTH,
        GMFamily.ETHNIC: InstrumentType.OTHER,
        GMFamily.PERCUSSIVE: InstrumentType.PERCUSSION,
        GMFamily.SOUND_EFFECTS: InstrumentType.OTHER,
    }
    return mapping.get(family, InstrumentType.OTHER)


def program_to_simple_type(program: int, is_drum_channel: bool = False) -> InstrumentType:
    """
    将 MIDI Program Number 映射到简化的乐器类型

    Args:
        program: MIDI Program Number (0-127)
        is_drum_channel: 是否为鼓通道（MIDI 通道 10）

    Returns:
        简化的乐器类型
    """
    if is_drum_channel:
        return InstrumentType.DRUMS

    family = get_instrument_family(program)
    if family is None:
        return InstrumentType.OTHER

    return family_to_simple_type(family)
