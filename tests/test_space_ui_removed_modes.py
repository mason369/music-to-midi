from pathlib import Path


def test_space_ui_uses_shared_i18n_labels():
    source_text = Path("space/app.py").read_text(encoding="utf-8")

    for expected_text in (
        "from src.i18n.translator import Translator",
        "SPACE_LANGUAGE",
        "SPACE_TRANSLATOR",
        "space.ui.audio_input",
        "space.status.complete_header",
        "space.ui.footer_powered_by",
    ):
        assert expected_text in source_text


def test_space_ui_exposes_restored_modes_and_dependencies():
    source_text = Path("space/app.py").read_text(encoding="utf-8")
    zh_text = Path("src/i18n/zh_CN.json").read_text(encoding="utf-8")
    requirements_text = Path("space/requirements.txt").read_text(encoding="utf-8")
    searchable_text = source_text + "\n" + zh_text

    for restored_text in (
        "六声部分离 + 分别转写",
        "钢琴专用转写 (Transkun)",
        "钢琴专用转写 (Aria-AMT)",
        "钢琴专用转写 (ByteDance Pedal)",
        "six_stem_split",
        "piano_transkun",
        "piano_aria_amt",
        "piano_bytedance_pedal",
        "ensure_aria_amt_weights",
        "download_aria_amt_model",
    ):
        assert restored_text in searchable_text

    assert "aria-amt" in requirements_text
    assert "piano-transcription-inference" in requirements_text

    for package_name in (
        "einops",
        "smart-open",
        "pretty-midi",
        "soxr",
        "mido",
        "soundfile",
    ):
        assert package_name in requirements_text
