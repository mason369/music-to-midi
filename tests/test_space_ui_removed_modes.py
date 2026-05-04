from pathlib import Path


def test_space_ui_exposes_restored_modes_and_dependencies():
    source_text = Path("space/app.py").read_text(encoding="utf-8")
    requirements_text = Path("space/requirements.txt").read_text(encoding="utf-8")

    for restored_text in (
        "六声部分离 + 分别转写",
        "钢琴专用转写 (Transkun)",
        "钢琴专用转写 (Aria-AMT)",
        "six_stem_split",
        "piano_transkun",
        "piano_aria_amt",
        "ensure_aria_amt_weights",
        "download_aria_amt_model",
    ):
        assert restored_text in source_text

    assert "aria-amt" in requirements_text
