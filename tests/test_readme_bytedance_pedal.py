from pathlib import Path


def test_readme_documents_bytedance_pedal_selection_guidance_and_model_class():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "### ByteDance Pedal",
        "## 钢琴后端选择建议",
        "#### 当前已集成后端总览",
        "ByteDance Pedal | 钢琴专精 / 踏板感知",
        "#### 钢琴同类模型质量对比",
        "需要踏板 CC64",
        "踏板 onset F1",
        "96.72% / 91.86%",
        "Transkun V2",
        "Aria-AMT",
        "YourMT3+ / MIROS 属于多乐器后端",
        "钢琴专精模型对比",
        "piano_bytedance_pedal",
        "download_bytedance_piano_model.py",
    ):
        assert expected in readme


def test_english_readme_documents_bytedance_pedal_model_comparisons():
    readme = Path("docs/README.md").read_text(encoding="utf-8")

    for expected in (
        "### ByteDance Pedal",
        "## Piano Backend Selection Guide",
        "#### Integrated Backend Overview",
        "ByteDance Pedal | Piano-specialized / pedal-aware",
        "#### Piano Model Quality Comparison",
        "needs sustain pedal CC64",
        "pedal onset F1",
        "96.72% / 91.86%",
        "Transkun V2",
        "Aria-AMT",
        "YourMT3+ / MIROS are multi-instrument backends",
        "piano_bytedance_pedal",
        "download_bytedance_piano_model.py",
    ):
        assert expected in readme
