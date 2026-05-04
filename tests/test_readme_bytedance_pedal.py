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


def test_readme_documents_official_yourmt3_mode_and_delivery_sync_status():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "## 入口与依赖同步状态",
        "官方 YourMT3 demo 暴露的五种 checkpoint / 架构模式",
        "download_sota_models.py",
        "run.ps1",
        "install.ps1",
        ".github/workflows/build.yml",
        ".github/workflows/release.yml",
        "colab_notebook.ipynb",
        "YourMT3+ 官方 checkpoint 模式",
        "旧官方模式默认最多 1",
        "MoE 模式默认最多 2",
        "MUSIC_TO_MIDI_YOURMT3_BATCH_SIZE",
        "python download_vocal_harmony_model.py",
        "python download_bytedance_piano_model.py",
        "python download_miros_model.py",
    ):
        assert expected in readme


def test_english_readme_documents_official_yourmt3_mode_and_delivery_sync_status():
    readme = Path("docs/README.md").read_text(encoding="utf-8")

    for expected in (
        "## Entry And Dependency Sync Status",
        "five checkpoint / architecture modes exposed by the official YourMT3 demo",
        "download_sota_models.py",
        "run.ps1",
        "install.ps1",
        ".github/workflows/build.yml",
        ".github/workflows/release.yml",
        "colab_notebook.ipynb",
        "YourMT3+ Official Checkpoint Modes",
        "older official modes default to 1",
        "MoE modes default to 2",
        "MUSIC_TO_MIDI_YOURMT3_BATCH_SIZE",
        "python download_vocal_harmony_model.py",
        "python download_bytedance_piano_model.py",
        "python download_miros_model.py",
    ):
        assert expected in readme
