import json
import os
from pathlib import Path
from string import Formatter

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtTest import QSignalSpy
from PyQt6.QtWidgets import QApplication

from src.gui.widgets.audio_track_mixer import _AudioTrackRow, midi_route_label
from src.i18n.translator import set_language

EXPECTED_YOURMT3_ROUTES = (
    ("yourmt3:ymt3_plus", "YMT3+"),
    ("yourmt3:yptf_single_nops", "YPTF+Single (noPS)"),
    ("yourmt3:yptf_multi_ps", "YPTF+Multi (PS)"),
    ("yourmt3:yptf_moe_multi_nops", "YPTF.MoE+Multi (noPS)"),
    ("yourmt3:yptf_moe_multi_ps", "YPTF.MoE+Multi (PS)"),
)

EXPECTED_ROUTE_COPY = {
    "zh_CN": {
        "multi_group": "多乐器转写",
        "yourmt3_group": "YourMT3+（多乐器）",
        "piano_group": "钢琴专用转写",
        "other_routes": (
            ("miros", "MIROS (MusicFM)（多乐器）"),
            ("muscriptor", "MuScriptor-large（多乐器，支持乐器硬约束）"),
            ("piano_transkun", "TransKun V2（钢琴）"),
            ("piano_transkun_v2_aug", "TransKun V2 Aug（钢琴）"),
            ("piano_aria_amt", "Aria-AMT（钢琴）"),
            ("piano_bytedance_pedal", "ByteDance Pedal（钢琴）"),
        ),
    },
    "en_US": {
        "multi_group": "Multi-Instrument Transcription",
        "yourmt3_group": "YourMT3+ (Multi-Instrument)",
        "piano_group": "Dedicated Piano Transcription",
        "other_routes": (
            ("miros", "MIROS (MusicFM) (Multi-Instrument)"),
            ("muscriptor", "MuScriptor-large (Multi-Instrument, hard constraints)"),
            ("piano_transkun", "TransKun V2 (Piano)"),
            ("piano_transkun_v2_aug", "TransKun V2 Aug (Piano)"),
            ("piano_aria_amt", "Aria-AMT (Piano)"),
            ("piano_bytedance_pedal", "ByteDance Pedal (Piano)"),
        ),
    },
}

EXPECTED_MANUAL_MIDI_KEYS = {
    "convert",
    "enable",
    "select_model",
    "start",
    "open",
    "multi_instrument",
    "piano",
    "not_selected",
    "model_required",
    "selected",
    "converting",
    "converting_short",
    "complete",
    "failed",
    "cancelled",
    "busy",
    "models.yourmt3",
    "models.miros",
    "models.muscriptor",
    "models.piano_transkun",
    "models.piano_transkun_v2_aug",
    "models.piano_aria_amt",
    "models.piano_bytedance_pedal",
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def restore_default_language():
    yield
    set_language("zh_CN")


def _flatten_strings(data, prefix=""):
    flattened = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_strings(value, full_key))
        else:
            assert isinstance(value, str), full_key
            flattened[full_key] = value
    return flattened


def _format_fields(template):
    return {
        field_name
        for _literal, field_name, _format_spec, _conversion in Formatter().parse(template)
        if field_name
    }


@pytest.mark.parametrize("language", ["zh_CN", "en_US"])
def test_per_track_midi_menu_lists_every_route_with_localized_labels(
    qapp,
    tmp_path,
    language,
):
    set_language(language)
    audio_path = tmp_path / "vocals.wav"
    audio_path.write_bytes(b"audio")
    row = _AudioTrackRow("vocals", audio_path, "#ff70a6")
    row.show()
    qapp.processEvents()

    try:
        expected_copy = EXPECTED_ROUTE_COPY[language]
        selector = row.midi_model_selector
        assert row.convert_midi_button.menu() is None
        assert selector.count() == 12
        assert selector.itemData(0) == ""
        assert selector.currentIndex() == 0
        assert row.midi_enabled_checkbox.isChecked() is False
        assert row.convert_midi_button.isEnabled() is False
        assert row.midi_status_label.isVisibleTo(row)

        expected_options = [
            (
                route,
                f"{expected_copy['multi_group']} · YourMT3+ · {checkpoint_label}",
            )
            for route, checkpoint_label in EXPECTED_YOURMT3_ROUTES
        ]
        expected_options.extend(
            (
                route,
                f"{expected_copy['multi_group']} · {localized_label}",
            )
            for route, localized_label in expected_copy["other_routes"][:2]
        )
        expected_options.extend(
            (
                route,
                f"{expected_copy['piano_group']} · {localized_label}",
            )
            for route, localized_label in expected_copy["other_routes"][2:]
        )
        assert [
            (selector.itemData(index), selector.itemText(index))
            for index in range(1, selector.count())
        ] == expected_options

        all_routes = [selector.itemData(index) for index in range(1, selector.count())]
        assert all_routes == [route for route, _label in EXPECTED_YOURMT3_ROUTES] + [
            route for route, _label in expected_copy["other_routes"]
        ]
        assert len(all_routes) == 11
        assert len(set(all_routes)) == 11

        conversions = QSignalSpy(row.midi_conversion_requested)
        row.midi_enabled_checkbox.setChecked(True)
        assert selector.isEnabled() is True
        selector.setCurrentIndex(selector.findData("miros"))
        qapp.processEvents()
        assert len(conversions) == 0
        assert row.convert_midi_button.isEnabled() is True
        assert midi_route_label("miros") in row.midi_status_label.text()

        row.convert_midi_button.click()
        qapp.processEvents()
        assert len(conversions) == 1
        assert conversions[0][0] == "miros"

        for route, checkpoint_label in EXPECTED_YOURMT3_ROUTES:
            assert midi_route_label(route) == f"YourMT3+ · {checkpoint_label}"
        for route, localized_label in expected_copy["other_routes"]:
            assert midi_route_label(route) == localized_label
    finally:
        row.close()
        row.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    ("track_name", "expected_name"),
    (
        ("source", "原始音频"),
        ("bass", "贝斯"),
        ("drums", "鼓"),
        ("guitar", "吉他"),
        ("piano", "钢琴"),
        ("vocals", "人声"),
        ("accompaniment", "伴奏"),
        ("other", "其他"),
        ("reference take", "reference take"),
    ),
)
def test_every_wav_row_keeps_its_track_name_and_midi_controls_visible(
    qapp,
    tmp_path,
    track_name,
    expected_name,
):
    set_language("zh_CN")
    audio_path = tmp_path / f"{track_name.replace(' ', '_')}.wav"
    audio_path.write_bytes(b"audio")
    row = _AudioTrackRow(track_name, audio_path, "#5eb1ff")
    row.resize(900, row.sizeHint().height())
    row.show()
    qapp.processEvents()

    try:
        assert row.name_label.text() == f"♪  {expected_name}"
        assert row.name_label.isVisibleTo(row)
        assert row.name_label.geometry().width() > 0
        assert row.midi_enabled_checkbox.isVisibleTo(row)
        assert row.midi_model_selector.isVisibleTo(row)
        assert row.convert_midi_button.isVisibleTo(row)
        assert row.midi_status_label.isVisibleTo(row)
        assert "未转换" in row.midi_status_label.text()
    finally:
        row.close()
        row.deleteLater()
        qapp.processEvents()


def test_per_track_midi_status_machine_is_explicit_and_failure_detail_is_bounded(
    qapp,
    tmp_path,
):
    set_language("zh_CN")
    audio_path = tmp_path / "piano.wav"
    audio_path.write_bytes(b"audio")
    midi_path = tmp_path / "piano.mid"
    midi_path.write_bytes(b"midi")
    row = _AudioTrackRow("piano", audio_path, "#c89bff")
    row.resize(900, row.sizeHint().height())
    row.show()
    qapp.processEvents()

    try:
        assert "未转换" in row.midi_status_label.text()
        row.midi_enabled_checkbox.setChecked(True)
        assert "选择转写模型" in row.midi_status_label.text()
        route_index = row.midi_model_selector.findData("miros")
        row.midi_model_selector.setCurrentIndex(route_index)
        assert "点击“开始转换”" in row.midi_status_label.text()

        row.set_midi_conversion_running("miros")
        assert "正在转 MIDI" in row.midi_status_label.text()
        assert row.midi_enabled_checkbox.isEnabled() is False
        assert row.midi_model_selector.isEnabled() is False
        assert row.convert_midi_button.isEnabled() is False

        row.set_midi_conversion_succeeded("miros", str(midi_path))
        assert "MIDI 已生成" in row.midi_status_label.text()
        assert row.open_midi_button.isVisibleTo(row)
        assert row.open_midi_button.text() == "打开文件夹"

        full_error = "\n".join(
            (
                "MIROS 转写失败:",
                'File "adapter_utils.py", line 171, in forward',
                "torch.OutOfMemoryError: CUDA out of memory while allocating 28 MiB",
            )
        )
        row.set_midi_conversion_failed(full_error)
        assert "CUDA out of memory" in row.midi_status_label.text()
        assert "adapter_utils.py" not in row.midi_status_label.text()
        assert full_error == row.midi_status_label.toolTip()

        row.set_midi_conversion_cancelled()
        assert "已取消" in row.midi_status_label.text()
    finally:
        row.close()
        row.deleteLater()
        qapp.processEvents()


def test_manual_midi_translation_keys_and_placeholders_match_between_languages():
    repo_root = Path(__file__).resolve().parents[1]
    catalogs = {
        language: json.loads(
            (repo_root / "src" / "i18n" / f"{language}.json").read_text(encoding="utf-8")
        )
        for language in ("zh_CN", "en_US")
    }
    flattened = {
        language: _flatten_strings(catalog["dialogs"]["complete"]["audio_tracks"]["manual_midi"])
        for language, catalog in catalogs.items()
    }

    assert set(flattened["zh_CN"]) == set(flattened["en_US"])
    assert EXPECTED_MANUAL_MIDI_KEYS <= set(flattened["zh_CN"])
    assert {
        key.removeprefix("models.") for key in flattened["zh_CN"] if key.startswith("models.")
    } == {
        "yourmt3",
        "miros",
        "muscriptor",
        "piano_transkun",
        "piano_transkun_v2_aug",
        "piano_aria_amt",
        "piano_bytedance_pedal",
    }

    for key in flattened["zh_CN"]:
        assert flattened["zh_CN"][key].strip()
        assert flattened["en_US"][key].strip()
        assert _format_fields(flattened["zh_CN"][key]) == _format_fields(flattened["en_US"][key])
