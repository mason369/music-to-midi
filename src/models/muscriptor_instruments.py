"""Canonical MuScriptor instrument groups shared by every frontend.

The identifiers and representative General MIDI programs mirror the public
MuScriptor frontend/API contract at the pinned source revision.  Display names
are localized here; only the canonical identifiers are ever sent to the model.
"""

from __future__ import annotations

from collections.abc import Iterable

MUSCRIPTOR_INSTRUMENTS: tuple[str, ...] = (
    "acoustic_piano",
    "electric_piano",
    "chromatic_percussion",
    "organ",
    "acoustic_guitar",
    "clean_electric_guitar",
    "distorted_electric_guitar",
    "acoustic_bass",
    "electric_bass",
    "violin",
    "viola",
    "cello",
    "contrabass",
    "orchestral_harp",
    "timpani",
    "string_ensemble",
    "synth_strings",
    "voice",
    "orchestra_hit",
    "trumpet",
    "trombone",
    "tuba",
    "french_horn",
    "brass_section",
    "soprano_and_alto_sax",
    "tenor_sax",
    "baritone_sax",
    "oboe",
    "english_horn",
    "bassoon",
    "clarinet",
    "flutes",
    "synth_lead",
    "synth_pad",
    "drums",
)


# MuScriptor serializes every non-drum group using the first GM program in
# that group's official MT3_FULL_PLUS mapping.
MUSCRIPTOR_REPRESENTATIVE_PROGRAMS: dict[str, int] = {
    "acoustic_piano": 0,
    "electric_piano": 2,
    "chromatic_percussion": 8,
    "organ": 16,
    "acoustic_guitar": 24,
    "clean_electric_guitar": 26,
    "distorted_electric_guitar": 29,
    "acoustic_bass": 32,
    "electric_bass": 33,
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "orchestral_harp": 46,
    "timpani": 47,
    "string_ensemble": 48,
    "synth_strings": 50,
    "voice": 52,
    "orchestra_hit": 55,
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "french_horn": 60,
    "brass_section": 61,
    "soprano_and_alto_sax": 64,
    "tenor_sax": 66,
    "baritone_sax": 67,
    "oboe": 68,
    "english_horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    "flutes": 72,
    "synth_lead": 80,
    "synth_pad": 88,
}


MUSCRIPTOR_ZH_LABELS: dict[str, str] = {
    "acoustic_piano": "原声钢琴",
    "electric_piano": "电钢琴",
    "chromatic_percussion": "半音阶打击乐",
    "organ": "风琴",
    "acoustic_guitar": "原声吉他",
    "clean_electric_guitar": "干净的电吉他",
    "distorted_electric_guitar": "失真电吉他",
    "acoustic_bass": "原声贝斯",
    "electric_bass": "电贝斯",
    "violin": "小提琴",
    "viola": "中提琴",
    "cello": "大提琴",
    "contrabass": "低音提琴",
    "orchestral_harp": "管弦乐竖琴",
    "timpani": "定音鼓",
    "string_ensemble": "弦乐合奏",
    "synth_strings": "合成弦乐",
    "voice": "人声",
    "orchestra_hit": "管弦乐击奏",
    "trumpet": "小号",
    "trombone": "长号",
    "tuba": "大号",
    "french_horn": "圆号",
    "brass_section": "铜管乐组",
    "soprano_and_alto_sax": "高音及中音萨克斯",
    "tenor_sax": "次中音萨克斯",
    "baritone_sax": "上低音萨克斯",
    "oboe": "双簧管",
    "english_horn": "英国管",
    "bassoon": "巴松管",
    "clarinet": "单簧管",
    "flutes": "长笛组",
    "synth_lead": "合成主音",
    "synth_pad": "合成铺底",
    "drums": "鼓组",
}


def validate_muscriptor_instruments(values: Iterable[str] | None) -> list[str]:
    """Return unique canonical instrument identifiers or raise explicitly."""

    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise ValueError("muscriptor_instruments must be a list of canonical names")

    valid = set(MUSCRIPTOR_INSTRUMENTS)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value).strip().lower()
        if name not in valid:
            raise ValueError(
                f"invalid MuScriptor instrument {value!r}; expected one of "
                f"{list(MUSCRIPTOR_INSTRUMENTS)!r}"
            )
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return normalized


def muscriptor_instrument_label(name: str, language: str = "en_US") -> str:
    """Return a localized display label without changing the canonical id."""

    canonical = validate_muscriptor_instruments([name])[0]
    if str(language).lower().startswith("zh"):
        return MUSCRIPTOR_ZH_LABELS[canonical]
    return canonical.replace("_", " ")
