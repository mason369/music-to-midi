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
        "TransKun V2",
        "Aria-AMT",
        "YourMT3+ / MuScriptor / MIROS 属于多乐器后端",
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
        "PIANO_TRANSKUN",
        "Aria-AMT",
        "YourMT3+ / MuScriptor / MIROS are multi-instrument backends",
        "piano_bytedance_pedal",
        "download_bytedance_piano_model.py",
    ):
        assert expected in readme


def test_readme_documents_official_yourmt3_mode_and_delivery_sync_status():
    readme = Path("README.md").read_text(encoding="utf-8")

    for expected in (
        "## 入口与依赖同步状态",
        "七种处理工作流",
        "官方 YourMT3 demo 暴露的五种 checkpoint / 架构模式",
        "download_sota_models.py",
        "run.ps1",
        "install.ps1",
        ".github/workflows/build.yml",
        ".github/workflows/release.yml",
        "colab_notebook.ipynb",
        "YourMT3+ 官方 checkpoint 模式",
        "官方 `update_config`",
        "inference_file(bsz=8)",
        "环境变量不再改写这条官方路线的 batch",
        "python download_vocal_model.py",
        "python download_accompaniment_model.py",
        "python download_transkun_v2_aug_model.py",
        "python download_bytedance_piano_model.py",
        "python download_miros_model.py",
        "本项目默认使用 **YPTF.MoE+Multi (noPS)**",
        "Slakh `multi_f = 0.7398`",
        "YPTF.MoE+Multi (PS) | 8 专家 | 有 | 可选 pitch-shift MoE checkpoint",
        "### 当前人声分离模型：Leap XE vocals + PolarFormer accompaniment",
        "`BS-Rofo-SW-Fixed.ckpt` -> 六条 WAV",
        "每条 WAV 独立选择 11 条转写路线",
        "`PIANO_TRANSKUN_V2_AUG`",
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
        "official `update_config` path",
        "inference_file(bsz=8)",
        "environment variables no longer alter the batch size",
        "python download_bytedance_piano_model.py",
        "python download_miros_model.py",
        "Project default and official Hugging Face Space default",
        "Slakh `multi_f = 0.7398`",
        "Optional pitch-shift MoE checkpoint",
    ):
        assert expected in readme


def test_readmes_match_current_processing_routes_and_packaged_paths():
    zh_readme = Path("README.md").read_text(encoding="utf-8")
    en_readme = Path("docs/README.md").read_text(encoding="utf-8")
    docs_zh_readme = Path("docs/README_zh.md").read_text(encoding="utf-8")
    space_readme = Path("space/README.md").read_text(encoding="utf-8")
    combined = "\n".join([zh_readme, en_readme, docs_zh_readme, space_readme])
    aligned_routes = "\n".join([zh_readme, space_readme])

    for expected in (
        "song_piano_aria.mid",
        "song_piano_transkun_v2_aug.mid",
        "BS-Rofo-SW-Fixed.ckpt",
        "Leap XE 90-band",
        "PolarFormer accompaniment",
        "每条 WAV 独立选择 11 条转写路线",
        "PIANO_TRANSKUN_V2_AUG",
        "download_transkun_v2_aug_model.py",
        "models/bytedance_piano",
        "MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR",
    ):
        assert expected in aligned_routes

    for mode in (
        "SMART",
        "VOCAL_SPLIT",
        "SIX_STEM_SPLIT",
        "PIANO_TRANSKUN",
        "PIANO_TRANSKUN_V2_AUG",
        "PIANO_ARIA_AMT",
        "PIANO_BYTEDANCE_PEDAL",
    ):
        assert mode in zh_readme

    for stale_route in (
        "one full-mix multi-instrument transcription",
        "route notes to stem MIDI by GM family",
        "按 GM 乐器族分配到 stem MIDI",
        "RoFormer `vocal_rvc` / `karaoke`",
    ):
        assert stale_route not in aligned_routes

    for stale in (
        "song_piano_aria_amt.mid",
        "_piano_aria_amt.mid",
        "fast / balanced / best",
        "torch==2.4.0",
        "https://download.pytorch.org/whl/cu121",
        "https://download.pytorch.org/whl/cu118",
        "CUDA 11.8:",
        "首次使用会检查 checkpoint",
        "YourMT3+ 多乐器转写",
        "selected_stems_merged",
        "vocal_harmony_separator.py",
        "only selected stems are transcribed",
        "Experimental lead/harmony model download",
    ):
        assert stale not in combined
