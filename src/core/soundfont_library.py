"""Validated, immutable SoundFont selections for audio rendering only.

The transcription model and the user's MIDI are never modified. Preset names
come from the bank's RIFF directory; FluidSynth remains the actual synthesizer.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import mido

MAX_SOUNDFONT_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class SoundFontPreset:
    bank: int
    program: int
    name: str

    def to_dict(self) -> dict:
        return {"bank": self.bank, "program": self.program, "name": self.name}


@dataclass(frozen=True)
class SoundFontLibrary:
    path: Path
    name: str
    sha256: str
    presets: tuple[SoundFontPreset, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.sha256,
            "name": self.name,
            "presets": [preset.to_dict() for preset in self.presets],
        }


@dataclass(frozen=True)
class SoundFontSelection:
    library: SoundFontLibrary
    # source program, source is_drum, destination bank, destination program
    assignments: tuple[tuple[int, bool, int, int], ...] = ()

    def cache_identity(self) -> dict:
        return {"library": self.library.sha256, "assignments": self.assignments}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_soundfont(path: str | Path, *, name: str | None = None) -> SoundFontLibrary:
    """Read bounded SF2/SF3 preset metadata and reject malformed RIFF files."""

    source = Path(path).resolve(strict=True)
    display_name = name or source.name
    if Path(display_name).suffix.lower() not in {".sf2", ".sf3"}:
        raise ValueError("音色库必须是 SF2 或 SF3 文件")
    size = source.stat().st_size
    if not 12 < size <= MAX_SOUNDFONT_BYTES:
        raise ValueError("音色库为空、损坏或超过 1 GiB 大小限制")
    chunks: dict[tuple[bytes, bytes], tuple[int, int]] = {}
    with source.open("rb") as stream:
        header = stream.read(12)
        if header[:4] != b"RIFF" or header[8:] != b"sfbk":
            raise ValueError("文件不是有效的 SoundFont RIFF/sfbk 音色库")
        if struct.unpack_from("<I", header, 4)[0] + 8 != size:
            raise ValueError("音色库长度与 RIFF 声明不符，文件可能被截断")

        def scan(start: int, end: int, parent: bytes = b"") -> None:
            position = start
            while position < end:
                if position + 8 > end:
                    raise ValueError("音色库包含不完整的数据块")
                stream.seek(position)
                tag, length = struct.unpack("<4sI", stream.read(8))
                body = position + 8
                limit = body + length
                if limit + (length & 1) > end:
                    raise ValueError("音色库数据块越界")
                if tag == b"LIST":
                    if parent or length < 4:
                        raise ValueError("音色库 LIST 结构无效")
                    group = stream.read(4)
                    scan(body + 4, limit, group)
                else:
                    key = (parent, tag)
                    if key in chunks:
                        raise ValueError("音色库包含重复的数据块")
                    chunks[key] = (body, length)
                position = limit + (length & 1)

        scan(12, size)
        required = {
            (b"INFO", b"ifil"): 4,
            (b"sdta", b"smpl"): 1,
            (b"pdta", b"phdr"): 38,
            (b"pdta", b"pbag"): 4,
            (b"pdta", b"pmod"): 10,
            (b"pdta", b"pgen"): 4,
            (b"pdta", b"inst"): 22,
            (b"pdta", b"ibag"): 4,
            (b"pdta", b"imod"): 10,
            (b"pdta", b"igen"): 4,
            (b"pdta", b"shdr"): 46,
        }
        for key, stride in required.items():
            if key not in chunks or chunks[key][1] < stride or chunks[key][1] % stride:
                raise ValueError(f"音色库缺少或损坏 {key[1].decode()} 数据块")
        stream.seek(chunks[(b"INFO", b"ifil")][0])
        major, _minor = struct.unpack("<HH", stream.read(4))
        if major not in {2, 3}:
            raise ValueError(f"不支持 SoundFont 版本 {major}")
        start, length = chunks[(b"pdta", b"phdr")]
        if length < 76:
            raise ValueError("音色库没有可用预设")
        stream.seek(start)
        presets = []
        seen = set()
        previous_bag = 0
        bag_count = chunks[(b"pdta", b"pbag")][1] // 4
        for index in range(length // 38):
            raw_name, program, bank, bag, _lib, _genre, _morph = struct.unpack(
                "<20sHHHIII", stream.read(38)
            )
            if bag < previous_bag or bag >= bag_count:
                raise ValueError("音色库预设区域索引无效")
            previous_bag = bag
            if index == length // 38 - 1:
                # MuseScore's SF3 writer leaves the terminal name empty.
                # The terminal bag offset, not a display name, delimits presets.
                if bag != bag_count - 1:
                    raise ValueError("音色库预设结束记录的区域索引无效")
                continue
            if program > 127 or bank > 16383 or (bank, program) in seen:
                raise ValueError("音色库预设编号无效或重复")
            seen.add((bank, program))
            label = raw_name.split(b"\0", 1)[0].decode("utf-8", errors="replace")
            label = "".join(character for character in label if character.isprintable()).strip()
            presets.append(SoundFontPreset(bank, program, label or f"{bank}:{program}"))
    return SoundFontLibrary(source, display_name, _sha256(source), tuple(presets))


def import_soundfont(path: str | Path, directory: str | Path) -> SoundFontLibrary:
    """Keep a content-addressed copy so later edits to the source cannot alter playback."""

    library = read_soundfont(path)
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (library.sha256 + Path(library.name).suffix.lower())
    if not destination.exists():
        import tempfile

        with tempfile.NamedTemporaryFile(dir=root, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(library.path, temporary)
            if _sha256(temporary) != library.sha256:
                raise RuntimeError("复制过程中音色库发生变化，请重新导入")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    if _sha256(destination) != library.sha256:
        raise RuntimeError("已导入的音色库校验失败，请重新导入")
    return SoundFontLibrary(destination, library.name, library.sha256, library.presets)


def default_soundfont() -> SoundFontLibrary:
    from src.utils.muscriptor_soundfont_downloader import (
        MUSCRIPTOR_SF2_FILENAME,
        validate_muscriptor_soundfont,
    )

    return read_soundfont(validate_muscriptor_soundfont(), name=MUSCRIPTOR_SF2_FILENAME)


def soundfont_selection(library: SoundFontLibrary, assignments: object) -> SoundFontSelection:
    if not isinstance(assignments, list) or len(assignments) > 256:
        raise ValueError("音色分配必须是最多 256 项的列表")
    available = {(preset.bank, preset.program) for preset in library.presets}
    result = []
    seen = set()
    for item in assignments:
        if not isinstance(item, dict) or set(item) != {
            "source_program",
            "is_drum",
            "bank",
            "program",
        }:
            raise ValueError("音色分配字段无效")
        source, drum, bank, program = (
            item[key] for key in ("source_program", "is_drum", "bank", "program")
        )
        if (
            any(type(value) is not int for value in (source, bank, program))
            or type(drum) is not bool
        ):
            raise ValueError("音色分配编号必须为整数，鼓组标记必须为布尔值")
        if not 0 <= source <= 127 or (bank, program) not in available:
            raise ValueError(f"音色库不包含所选预设：bank={bank}, program={program}")
        if drum != (bank == 128):
            raise ValueError("鼓轨只能选择 bank 128 鼓组预设；旋律轨请选择旋律预设")
        if (source, drum) in seen:
            raise ValueError("同一来源乐器存在重复音色分配")
        seen.add((source, drum))
        result.append((source, drum, bank, program))
    return SoundFontSelection(library, tuple(sorted(result)))


def write_soundfont_render_midi(
    source: Path, destination: Path, selection: SoundFontSelection
) -> Path:
    """Translate only bank/program selection, retaining all other absolute ticks.

    Input banks follow the existing GS playback contract. The private rendering
    MIDI uses FluidSynth's MMA bank mode to address every explicit bank exactly.
    """

    if _sha256(selection.library.path) != selection.library.sha256:
        raise RuntimeError("音色库在导入后发生变化，已停止渲染")
    mapping = {
        (program, drum): (bank, preset) for program, drum, bank, preset in selection.assignments
    }
    available = {(preset.bank, preset.program) for preset in selection.library.presets}
    midi = mido.MidiFile(str(source))
    if midi.type == 2:
        raise ValueError("自定义音色不支持异步 Type 2 MIDI")
    output = mido.MidiFile(type=0, ticks_per_beat=midi.ticks_per_beat)
    track = mido.MidiTrack()
    output.tracks.append(track)
    programs = [0] * 16
    banks = [0] * 16
    selected: dict[int, tuple[int, int]] = {}
    tick = 0
    previous_tick = 0

    def append(message: mido.Message) -> None:
        nonlocal previous_tick
        track.append(message.copy(time=tick - previous_tick))
        previous_tick = tick

    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        if message.type == "sysex":
            raise ValueError("此 MIDI 包含 SysEx 音源控制，自定义音色渲染无法保证正确映射")
        if message.type == "program_change":
            programs[message.channel] = message.program
            selected.pop(message.channel, None)
            continue
        if message.type == "control_change" and message.control in {0, 32}:
            if message.control == 0:
                banks[message.channel] = message.value
            selected.pop(message.channel, None)
            continue
        if message.type == "note_on" and message.velocity > 0:
            channel = message.channel
            drum = channel == 9
            preset = mapping.get(
                (programs[channel], drum), (128 if drum else banks[channel], programs[channel])
            )
            if preset not in available:
                raise ValueError(
                    f"音色库 {selection.library.name} 缺少 bank={preset[0]}, program={preset[1]}；"
                    "请为该轨道明确选择可用预设"
                )
            if selected.get(channel) != preset:
                bank, program = preset
                append(
                    mido.Message("control_change", channel=channel, control=0, value=bank // 128)
                )
                append(
                    mido.Message("control_change", channel=channel, control=32, value=bank % 128)
                )
                append(mido.Message("program_change", channel=channel, program=program))
                selected[channel] = preset
        append(message)
    output.save(str(destination))
    return destination
