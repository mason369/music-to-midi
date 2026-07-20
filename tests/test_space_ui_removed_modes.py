import ast
import json
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
    assert (
        "    demo.launch(" 'server_name="0.0.0.0", allowed_paths=[str(SPACE_OUTPUT_INSTANCE)]' ")"
    ) in source_text


def test_space_copy_describes_all_outputs_and_current_telknet_alignment():
    zh = json.loads(Path("src/i18n/zh_CN.json").read_text(encoding="utf-8"))
    en = json.loads(Path("src/i18n/en_US.json").read_text(encoding="utf-8"))

    assert zh["space"]["ui"]["download_section"] == "下载输出文件"
    assert en["space"]["ui"]["download_section"] == "Download Output Files"
    for catalog in (zh, en):
        vocal_info = catalog["space"]["mode"]["vocal_split_info"]
        six_stem_info = catalog["space"]["mode"]["six_stem_split_info"]
        assert "WAV" in vocal_info
        assert "MIDI" in vocal_info
        assert "WAV" in six_stem_info
        assert "MIDI" in six_stem_info

        product_copy = "\n".join(
            [
                *catalog["main"]["mode"].values(),
                *catalog["main"]["engine"].values(),
                *catalog["space"]["mode"].values(),
            ]
        ).lower()
        for banned_phrase in (
            "telknet",
            "对齐",
            "落后",
            "不声称",
            "逐行",
            "challenge-sota",
            "source parity",
            "line for line",
            "line-for-line",
            "website contract",
        ):
            assert banned_phrase not in product_copy


def test_space_ui_exposes_restored_modes_and_dependencies():
    source_text = Path("space/app.py").read_text(encoding="utf-8")
    zh_text = Path("src/i18n/zh_CN.json").read_text(encoding="utf-8")
    requirements_text = Path("space/requirements.txt").read_text(encoding="utf-8")
    searchable_text = source_text + "\n" + zh_text

    for restored_text in (
        "六声部分离",
        "钢琴专用转写 (TransKun)",
        "钢琴专用转写 (Aria-AMT)",
        "钢琴专用转写 (ByteDance Pedal)",
        "six_stem_split",
        "piano_transkun",
        "piano_transkun_v2_aug",
        "piano_aria_amt",
        "piano_bytedance_pedal",
        "ensure_aria_amt_weights",
        "ensure_transkun_v2_aug_weights",
        "download_aria_amt_model",
        "download_transkun_v2_aug_model",
    ):
        assert restored_text in searchable_text

    assert "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/" in source_text
    assert "a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip" in source_text
    assert "aria-amt" not in requirements_text
    assert "audio-separator==0.44.1" in requirements_text
    assert "gradio==4.44.1" in requirements_text
    assert "fastapi==0.115.2" in requirements_text
    assert "starlette==0.40.0" in requirements_text
    assert "piano-transcription-inference==0.0.6" in requirements_text
    assert "transkun==2.0.1" in requirements_text
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
    assert "ensure_model_weights" in source_text
    assert "from src.utils.yourmt3_downloader import download_model, get_model_path" in source_text
    assert "ensure_multistem_weights" in source_text
    assert "ensure_vocal_split_weights" in source_text
    assert "download_multistem_model" in source_text
    assert "download_vocal_model" in source_text
    assert "download_accompaniment_model" in source_text
    assert "download_vocal_harmony_model" not in source_text
    assert "download_ultimate_moe" not in source_text
    assert "will retry on first use" not in source_text
    assert "Aria-AMT downloader unavailable" not in source_text
    assert "Aria-AMT checkpoint download failed" not in source_text
    assert "ByteDance Piano downloader unavailable" not in source_text
    assert "ByteDance Piano checkpoint download failed" not in source_text


def test_space_forces_and_verifies_the_pinned_aria_source_identity():
    source_text = Path("space/app.py").read_text(encoding="utf-8")

    assert '"--force-reinstall"' in source_text
    assert "get_aria_amt_runtime_unavailable_reason" in source_text
    assert "source identity validation failed" in source_text


def test_space_backend_checkpoint_and_aug_contract_is_wired_end_to_end():
    source_text = Path("space/app.py").read_text(encoding="utf-8")
    module = ast.parse(source_text)

    functions = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    convert_impl = functions["_convert_impl"]
    build_config = functions["_build_space_request_config"]
    prepare_models = functions["_prepare_request_models"]
    convert_arg_names = [argument.arg for argument in convert_impl.args.args]
    self_source = ast.get_source_segment(source_text, convert_impl)
    config_source = ast.get_source_segment(source_text, build_config)
    prepare_source = ast.get_source_segment(source_text, prepare_models)
    mode_controls_source = ast.get_source_segment(
        source_text,
        functions["update_mode_controls"],
    )
    backend_controls_source = ast.get_source_segment(
        source_text,
        functions["update_backend_controls"],
    )

    assert convert_arg_names[:4] == [
        "audio_path",
        "mode",
        "transcription_backend",
        "yourmt3_model",
    ]
    assert "config.transcription_backend = transcription_backend" in config_source
    assert "config.multi_instrument_model = transcription_backend" in config_source
    assert "config.yourmt3_model = yourmt3_model" in config_source
    assert "config.validate()" in config_source
    assert "ProcessingMode.PIANO_TRANSKUN_V2_AUG.value" in prepare_source
    assert "ensure_transkun_v2_aug_weights()" in prepare_source
    assert "ensure_model_weights(config.yourmt3_model)" in prepare_source
    assert "ensure_miros_weights()" in prepare_source
    assert "ensure_model_weights" not in self_source

    for control_source in (mode_controls_source, backend_controls_source):
        assert "mode not in MODE_IDS" in control_source
        assert "Unsupported processing mode" in control_source
        assert "Unsupported multi-instrument backend" in control_source

    assert "transcription_backend = gr.Radio(" in source_text
    assert "yourmt3_model = gr.Dropdown(" in source_text
    assert "inputs=[mode_radio, transcription_backend]" in source_text
    assert "transcription_backend," in source_text
    assert "yourmt3_model," in source_text


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


def test_space_split_modes_stop_after_wav_and_only_start_midi_from_row_button():
    source_text = Path("space/app.py").read_text(encoding="utf-8")
    module = ast.parse(source_text)
    functions = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    direct_source = ast.get_source_segment(source_text, functions["_convert_impl"])
    separation_source = ast.get_source_segment(source_text, functions["_separate_impl"])
    selection_source = ast.get_source_segment(
        source_text,
        functions["_track_control_updates"],
    )
    per_track_source = ast.get_source_segment(
        source_text,
        functions["_convert_one_track"],
    )

    assert "if mode in SPLIT_MODE_IDS" in direct_source
    assert "MusicToMidiPipeline" in direct_source
    assert "AudioSeparationService(" not in direct_source
    assert "AudioSeparationService" in separation_source
    assert "MusicToMidiPipeline" not in separation_source
    assert "_build_track_state" in separation_source
    assert "track_state" in separation_source

    assert "_prepare_request_models" not in selection_source
    assert "_convert_manual_midi_on_gpu" not in selection_source
    assert "_prepare_request_models" in per_track_source
    assert "_convert_manual_midi_on_gpu" in per_track_source

    assert "midi_enabled.change(" in source_text
    assert "midi_route.change(" in source_text
    assert source_text.count("fn=_track_control_updates") == 2
    assert "start_midi.click(" in source_text
    assert "fn=_convert_one_track" in source_text
    assert "vocal_split_merge_midi = gr.Checkbox(" not in source_text
    assert "save_separated_tracks = gr.Checkbox(" not in source_text


def test_space_track_workbench_uses_shared_browser_mixer_and_ten_shared_routes():
    source_text = Path("space/app.py").read_text(encoding="utf-8")
    manual_source = Path("src/core/manual_midi.py").read_text(encoding="utf-8")

    assert "@gr.render(inputs=[track_state, mode_radio])" in source_text
    assert "def render_track_workbench(current_state, selected_mode):" in source_text
    assert "selected_mode not in SPLIT_MODE_IDS" in source_text

    # The workbench renders real waveforms through the shared browser mixer
    # runtime instead of per-track Gradio audio players.
    assert 'build_track_mixer_html(state["tracks"], st)' in source_text
    assert "head=LOG_POLL_HEAD + mixer_head() + muscriptor_result_head()" in source_text
    assert "from src.gui.web.track_mixer_runtime import (" in source_text
    assert "waveform_options=gr.WaveformOptions(" not in source_text
    assert "key=f\"waveform-{track['id']}\"" not in source_text
    assert 'sources=["upload"]' in source_text

    # Per-track removal mirrors the desktop mixer row contract.
    assert "def _remove_track(track_state, track_id):" in source_text
    assert 'st("dialogs.complete.audio_tracks.remove")' in source_text
    assert "fn=_remove_track" in source_text

    assert "MANUAL_MIDI_ROUTES" in source_text
    assert "MANUAL_MIDI_ROUTE_CHOICES" in source_text
    assert "build_manual_midi_config" in source_text
    assert "manual_midi_output_dir" in source_text
    assert "len(MANUAL_MIDI_ROUTE_CHOICES) != 11" in source_text
    assert "YOURMT3_MANUAL_MODELS" in manual_source
    assert "MIDI_ROUTE_MIROS" in manual_source
    assert "MIDI_ROUTE_MUSCRIPTOR" in manual_source
    assert "MIDI_ROUTE_PIANO_TRANSKUN" in manual_source
    assert "MIDI_ROUTE_PIANO_TRANSKUN_V2_AUG" in manual_source
    assert "MIDI_ROUTE_PIANO_ARIA_AMT" in manual_source
    assert "MIDI_ROUTE_PIANO_BYTEDANCE_PEDAL" in manual_source


def test_space_track_state_is_request_owned_and_added_audio_is_copied():
    source_text = Path("space/app.py").read_text(encoding="utf-8")
    module = ast.parse(source_text)
    functions = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    normalize_source = ast.get_source_segment(
        source_text,
        functions["_normalize_track_state"],
    )
    add_source = ast.get_source_segment(source_text, functions["_add_audio_tracks"])
    manual_source = ast.get_source_segment(
        source_text,
        functions["_convert_manual_midi_impl"],
    )

    assert "_require_active_request_dir" in normalize_source
    assert "_require_owned_request_file" in normalize_source
    assert "midi_path" in normalize_source
    assert "shutil.copy2" in add_source
    assert 'request_root / "added_tracks"' in add_source
    assert "_require_owned_request_file" in add_source
    assert "_require_owned_request_output_dir" in manual_source
    assert "_validate_processing_outputs(result, config, request_root)" in manual_source


def test_space_all_gpu_jobs_share_one_serial_concurrency_queue():
    source_text = Path("space/app.py").read_text(encoding="utf-8")

    assert 'GPU_CONCURRENCY_ID = "music-to-midi-gpu"' in source_text
    assert source_text.count("concurrency_id=GPU_CONCURRENCY_ID") == 2
    assert "@spaces.GPU(duration=_estimate_zerogpu_duration" in source_text
    assert "@spaces.GPU(duration=_estimate_manual_zerogpu_duration" in source_text
    assert 'setattr(_convert_one_track, "zerogpu", None)' in source_text
    assert "concurrency_limit=1" in source_text


def test_space_readme_describes_wav_only_split_and_explicit_per_track_midi():
    readme = Path("space/README.md").read_text(encoding="utf-8")

    assert "两个分离模式只先生成 WAV" in readme
    assert "选择复选框或模型不会开始推理" in readme
    assert "十一个明确路线" in readme
    assert "models:" in readme
    assert "MuScriptor/muscriptor-large" in readme
    assert "mimbres/YourMT3" in readme
    assert "minzwon/MusicFM" in readme
    assert "顶部 `license: mit` **只表示本 Space 自有应用代码使用 MIT**" in readme
    assert "必须在模型页接受条款" in readme
    assert "`SMART` + MuScriptor Large" in readme
    assert "约 0.833 秒" in readme
    assert "开始分离" in readme
    assert "不会自动生成 vocal、accompaniment 或 merged MIDI" in readme
    assert "不会自动生成六个 stem MIDI 或 merged MIDI" in readme
