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

    assert "tempfile.gettempdir()" in source_text
    assert 'LOG_FILE = os.path.join(APP_TEMP_DIR, "midi_process.log")' in source_text
    assert 'LOG_FILE = "/tmp/midi_process.log"' not in source_text
    assert "_normalize_json_schema_bool_nodes" in source_text
    assert "Component.api_info = _patched_component_api_info" in source_text
    assert 'if __name__ == "__main__":' in source_text
    assert '    demo.launch(server_name="0.0.0.0")' in source_text


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
    assert "audio-separator==0.44.1" in requirements_text
    assert "gradio==4.44.1" in requirements_text
    assert "fastapi==0.115.2" in requirements_text
    assert "starlette==0.40.0" in requirements_text
    assert "piano-transcription-inference" in requirements_text
    assert "matplotlib" in requirements_text

    for package_name in (
        "einops",
        "smart-open",
        "pretty-midi",
        "soxr",
        "mido",
        "soundfile",
    ):
        assert package_name in requirements_text


def test_space_uses_current_model_download_entrypoints_without_masking_failures():
    source_text = Path("space/app.py").read_text(encoding="utf-8")

    assert "ensure_yourmt3_code()" in source_text
    assert "download_official_yourmt3_models" in source_text
    assert "ensure_multistem_weights" in source_text
    assert "ensure_vocal_split_weights" in source_text
    assert "download_multistem_model" in source_text
    assert "download_vocal_model" in source_text
    assert "download_vocal_harmony_model" in source_text
    assert "download_ultimate_moe" not in source_text
    assert "will retry on first use" not in source_text
    assert "Aria-AMT downloader unavailable" not in source_text
    assert "Aria-AMT checkpoint download failed" not in source_text
    assert "ByteDance Piano downloader unavailable" not in source_text
    assert "ByteDance Piano checkpoint download failed" not in source_text


def test_space_ui_does_not_expose_removed_six_stem_experimental_controls():
    source_text = Path("space/app.py").read_text(encoding="utf-8")
    zh_text = Path("src/i18n/zh_CN.json").read_text(encoding="utf-8")
    en_text = Path("src/i18n/en_US.json").read_text(encoding="utf-8")
    combined = "\n".join([source_text, zh_text, en_text])

    for stale in (
        "six_stem_only_selected",
        "six_stem_targets",
        "six_stem_targets_info",
        "six_stem_vocal_harmony",
        "Only transcribe selected stems",
        "仅转写选中的 stem",
        "Split vocals into lead + harmony",
    ):
        assert stale not in combined
