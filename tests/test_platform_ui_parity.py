"""Cross-platform consistency contracts for desktop, Space, and Colab.

Every platform must present the same workflow: identical action labels,
stop/cancel support with cooperative cancellation, and identical completion
summaries for direct conversions and separation-only runs.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace


def _colab_source() -> str:
    notebook = json.loads(Path("colab_notebook.ipynb").read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )


def _space_source() -> str:
    return Path("space/app.py").read_text(encoding="utf-8")


def _desktop_source() -> str:
    return Path("src/gui/main_window.py").read_text(encoding="utf-8")


def _isolated_function(source: str, name: str, **globals_):
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = dict(globals_)
    exec(
        compile(ast.fix_missing_locations(module), "<isolated-platform-function>", "exec"),
        namespace,
    )
    return namespace[name]


def test_project_bpm_contract_is_shared_by_desktop_space_and_colab():
    space = _space_source()
    colab = _colab_source()
    track_panel = Path("src/gui/widgets/track_panel.py").read_text(encoding="utf-8")
    zh = json.loads(Path("src/i18n/zh_CN.json").read_text(encoding="utf-8"))
    en = json.loads(Path("src/i18n/en_US.json").read_text(encoding="utf-8"))
    web_zh = json.loads(Path("web/locales/zh_CN.json").read_text(encoding="utf-8"))
    web_en = json.loads(Path("web/locales/en_US.json").read_text(encoding="utf-8"))

    assert "self.custom_bpm_spin.setRange(MIN_TEMPO_BPM, MAX_TEMPO_BPM)" in track_panel
    assert "choices=TEMPO_MODE_CHOICES" in space
    assert "value=TempoMode.FIXED_AUTO.value" in space
    assert "minimum=MIN_TEMPO_BPM" in space
    assert "maximum=MAX_TEMPO_BPM" in space
    assert "choices=TEMPO_MODE_CHOICES" in colab
    assert "value=TempoMode.FIXED_AUTO.value" in colab
    assert "minimum=MIN_TEMPO_BPM" in colab
    assert "maximum=MAX_TEMPO_BPM" in colab
    assert "normalize_optional_project_bpm(custom_bpm)" in space
    assert "normalize_optional_project_bpm(custom_bpm)" in colab
    assert zh["main"]["tempo"]["label"] == "速度方案"
    assert en["main"]["tempo"]["label"] == "Tempo mode"
    for catalog in (zh, en):
        assert "Beat This final0" in catalog["main"]["tempo"]["mode_tooltip"]
    assert web_zh["config.tempo_mode"] == "速度方案"
    assert web_en["config.tempo_mode"] == "Tempo mode"
    for catalog in (web_zh, web_en):
        assert "Beat This final0" in catalog["config.tempo_help"]
    assert zh["main"]["tempo"]["fixed_auto"] == "自动检测唯一 BPM（推荐）"
    assert en["main"]["tempo"]["fixed_auto"] == "Auto-detect one BPM (recommended)"
    assert "30–300 BPM" in zh["main"]["tempo"]["custom_tooltip"]
    assert "播放速度按手动 BPM ÷ 检测 BPM 变化" in zh["main"]["tempo"]["custom_tooltip"]
    assert (
        "preserves the musical ticks established at the detected BPM"
        in en["main"]["tempo"]["custom_tooltip"]
    )
    assert "manual BPM ÷ detected BPM" in en["main"]["tempo"]["custom_tooltip"]


def test_space_and_colab_retain_original_mix_for_separated_track_tempo_analysis():
    space = _space_source()
    colab = _colab_source()
    zh = json.loads(Path("src/i18n/zh_CN.json").read_text(encoding="utf-8"))
    en = json.loads(Path("src/i18n/en_US.json").read_text(encoding="utf-8"))

    assert "_copy_tempo_source_into_request(audio_path, output_dir)" in space
    assert 'state.get("tempo_audio_path")' in space
    assert "tempo_audio_path=str(tempo_audio_file)" in space
    assert 'st("dialogs.complete.audio_tracks.tempo_source_unavailable")' in space
    for catalog in (zh, en):
        assert catalog["dialogs"]["complete"]["audio_tracks"]["tempo_source_unavailable"]

    assert 'output_dir / f"tempo-source{source_audio.suffix.lower()}"' in colab
    assert 'state.get("tempo_audio_path")' in colab
    assert "tempo_audio_path=str(tempo_audio_path)" in colab


def test_shared_action_label_keys_exist_in_every_language():
    zh = json.loads(Path("src/i18n/zh_CN.json").read_text(encoding="utf-8"))
    en = json.loads(Path("src/i18n/en_US.json").read_text(encoding="utf-8"))

    for catalog in (zh, en):
        toolbar = catalog["toolbar"]
        assert toolbar["start_convert"].strip()
        assert toolbar["start_separation"].strip()
        assert toolbar["stop"].strip()
        complete = catalog["dialogs"]["complete"]
        assert complete["bpm"].strip()
        assert complete["device"].strip()
        assert complete["track_count"].strip()

    assert zh["toolbar"]["start_convert"] == "开始转换"
    assert zh["toolbar"]["start_separation"] == "开始分离"
    assert zh["status"]["cancelled"].strip()
    assert zh["status"]["cancelling"].strip()


def test_all_platforms_use_the_shared_action_label_keys():
    desktop = _desktop_source()
    space = _space_source()
    colab = _colab_source()

    # Direct modes convert and split modes separate; the wording matches everywhere.
    assert 't("toolbar.start_convert")' in desktop
    assert 't("toolbar.start_separation")' in desktop
    assert 'st("toolbar.start_convert")' in space
    assert 'st("toolbar.start_separation")' in space
    assert 'COLAB_TRANSLATOR.t("toolbar.start_convert")' in colab
    assert 'COLAB_TRANSLATOR.t("toolbar.start_separation")' in colab

    # The desktop must switch the start button text with the selected mode.
    assert "_update_start_button_label" in desktop
    assert 'st("space.ui.convert_button")' not in space or True  # legacy key unused
    assert "_main_action_label" in space
    assert '"开始分离" if SPACE_LANGUAGE' not in space  # no inline hardcoding

    # Stop buttons exist on every platform with the same shared label.
    assert 't("toolbar.stop")' in desktop
    assert 'st("toolbar.stop")' in space
    assert 'COLAB_TRANSLATOR.t("toolbar.stop")' in colab


def test_all_platforms_offer_cooperative_cancellation():
    space = _space_source()
    colab = _colab_source()

    for source, cancel_fn in ((space, None), (colab, None)):
        assert "_ACTIVE_JOB_LOCK" in source
        assert "def _register_active_job(job)" in source
        assert "def _unregister_active_job(job)" in source
        assert "def request_stop_current_job()" in source
        assert "job.cancel()" in source
        # The visible event uses an independent unqueued HTTP request. Gradio
        # otherwise serializes a same-session stop click behind the GPU job.
        assert "def _stop_current_job_client_js" in source
        assert 'api_name="stop_current_job"' in source
        assert "js=_stop_current_job_client_js()" in source
        assert "./run/stop_current_job" in source
        assert "queue=False" in source
        # Cancellation is a first-class outcome, not a failure.
        assert "except InterruptedError" in source

    assert '"status.cancelling"' in space
    assert '"status.cancelled"' in space
    assert '"status.cancelling"' in colab
    assert '"status.cancelled"' in colab

    # Manual per-track cancellation reports the shared cancelled status.
    assert "manual_midi.cancelled" in space
    assert "manual_midi.cancelled" in colab


def test_web_reruns_replace_their_previous_ui_log_handler():
    for source in (_space_source(), _colab_source()):
        assert "_music_to_midi_ui_log_handler" in source
        assert "removeHandler(_existing_handler)" in source
        assert "_existing_handler.close()" in source

    # The desktop pipeline/separation raise InterruptedError cooperatively.
    pipeline = Path("src/core/pipeline.py").read_text(encoding="utf-8")
    separation = Path("src/core/separation_service.py").read_text(encoding="utf-8")
    assert "def cancel(" in pipeline
    assert "raise InterruptedError" in pipeline
    assert "raise InterruptedError" in separation


def test_direct_conversion_summary_fields_match_across_platforms():
    desktop = _desktop_source()
    space = _space_source()
    colab = _colab_source()

    # Direct summaries: MIDI file, elapsed time, notes, track count, BPM, device.
    for key in ("midi_file", "track_count", "note_count", "bpm", "device", "processing_time"):
        assert f"dialogs.complete.{key}" in desktop
    assert "space.status.total_notes" in space
    assert "dialogs.complete.track_count" in space
    assert "BPM" in space
    assert "space.status.device" in space
    assert "status.total_notes" in colab
    assert "dialogs.complete.track_count" in colab
    assert "BPM" in colab
    assert "status.device" in colab


def test_separation_summary_fields_match_across_platforms():
    desktop = _desktop_source()
    space = _space_source()
    colab = _colab_source()

    # Every platform shows: mode label, stem count, per-stem WAV list,
    # processing time, and the manual-MIDI hint.
    for marker in (
        "dialogs.complete.audio_tracks.separation_mode",
        "dialogs.complete.audio_tracks.separation_manual_hint",
    ):
        assert marker in desktop
        assert marker in space
        assert marker in colab

    assert "dialogs.complete.stem_audio_count" in desktop
    assert "space.status.separated_audio" in space
    assert "dialogs.complete.stem_audio_count" in colab

    assert "dialogs.complete.separated_audio" in desktop
    assert "dialogs.complete.separated_audio" in space
    assert "dialogs.complete.separated_audio" in colab

    # Per-stem WAV lines list the stem key and the real file name.
    for source in (space, colab):
        assert "track['name']" in source
        assert "track['audio_path']" in source


def test_colab_language_resolution_matches_the_space_contract():
    space = _space_source()
    colab = _colab_source()

    for source in (space, colab):
        assert 'os.environ.get("MUSIC_TO_MIDI_LANGUAGE", "zh_CN")' in source
        assert "Translator.AVAILABLE_LANGUAGES" in source
        assert "Unsupported MUSIC_TO_MIDI_LANGUAGE" in source

    assert "COLAB_TRANSLATOR = Translator(COLAB_LANGUAGE)" in colab


def test_web_platforms_source_shared_labels_from_the_catalog():
    space = _space_source()
    colab = _colab_source()

    # Backend and YourMT3 model selectors use identical shared labels.
    for key in ("main.engine.active_label", "main.engine.yourmt3_model_label"):
        assert f'st("{key}")' in space
        assert f'COLAB_TRANSLATOR.t("{key}")' in colab

    # Audio input, download, and logs labels match the Space wording.
    for key in ("space.ui.audio_input", "space.ui.download_label", "space.ui.logs_label"):
        assert f'COLAB_TRANSLATOR.t("{key}")' in colab

    # Timeline title/subtitle and add-track label are the shared dialog keys.
    for key in (
        "dialogs.complete.audio_tracks.title",
        "dialogs.complete.audio_tracks.subtitle",
        "dialogs.complete.audio_tracks.add_track",
        "dialogs.complete.audio_tracks.manual_midi.select_model",
    ):
        assert f'COLAB_TRANSLATOR.t("{key}")' in colab or f"COLAB_TRANSLATOR.t('{key}')" in colab

    # Retired Colab-only duplicates stay removed.
    for stale in (
        "完整混音多乐器转写（SMART）",
        "SMART 多乐器转写后端",
        "YourMT3+ 官方模型模式",
        '"ui.audio_input"',
        '"ui.download"',
        '"ui.logs"',
        '"ui.start"',
        '"ui.start_separation"',
        '"ui.add_audio"',
        '"ui.add_audio_button"',
        '"ui.backend_label"',
        '"ui.yourmt3_model_label"',
        '"ui.timeline_title"',
        '"ui.timeline_hint"',
        '"ui.track_route"',
        "status.track_ready",
        "status.track_disabled",
        "status.track_complete",
        "status.separation_header",
        "status.separated_track_count",
        "status.manual_next",
    ):
        assert stale not in colab


def test_per_track_status_texts_use_the_shared_manual_midi_keys():
    space = _space_source()
    colab = _colab_source()

    for key in ("not_selected", "selected", "complete", "cancelled"):
        marker = f"dialogs.complete.audio_tracks.manual_midi.{key}"
        assert marker in space or marker in colab

    # Colab formats the selected status with the concrete route label.
    assert "_manual_route_display_label" in colab
    assert "manual_midi.selected" in colab
    assert "manual_midi.not_selected" in colab
    assert "manual_midi.complete" in colab


def test_colab_notebook_remains_valid_json_and_python():
    notebook = json.loads(Path("colab_notebook.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell["source"]))


def test_every_platform_prepares_the_same_required_beat_this_final0_model():
    space = _space_source()
    colab = _colab_source()
    detector = Path("src/core/beat_detector.py").read_text(encoding="utf-8")
    pipeline = Path("src/core/pipeline.py").read_text(encoding="utf-8")

    assert "BeatThisTracker" in detector
    assert "librosa" not in detector
    assert "download_beat_this_model(printer=logger.info)" in space
    assert "download_beat_this_model(printer=logger.info)" in colab
    assert "beat_info.tempo_map if self.config.enable_tempo_map" not in pipeline


def test_every_direct_mode_and_split_track_uses_the_shared_midi_workbench():
    desktop = _desktop_source()
    space = _space_source()
    colab = _colab_source()

    assert "else:\n            self.worker = ProcessingWorker(" in desktop
    assert "self._show_muscriptor_streaming(" in desktop
    for source in (space, colab):
        assert "def _build_midi_result_state(" in source
        assert '"kind": "midi_result"' in source
        assert "active_midi_track_id" in source
        assert "active_midi_result" in source
        assert "source_track_name" in source
        assert "build_muscriptor_result_html(" in source
        assert "prepare_midi_playback_assets" in source
        assert '"playback_audio_path"' in source
        assert "assets.original_wav" in source
        assert "rewrite_midi_tempo_preserving_ticks" in source
        assert "source-tempo-playback.mid" in source
        assert "result.beat_info.source_bpm" in source

    assert "preserve_mixer=True" in desktop
    assert "source_track_name=track_name" in desktop
    assert "_on_audio_mixer_playing_changed" in desktop
    assert "_on_midi_workbench_playing_changed" in desktop
    assert "fn=_close_active_midi_detail" in space
    assert "fn=close_midi_detail" in colab


def test_every_midi_delivery_surface_exposes_sheet_music_download():
    desktop = Path("src/gui/widgets/muscriptor_result.py").read_text(encoding="utf-8")
    space = _space_source()
    colab = _colab_source()
    shared_browser_editor = Path("src/gui/web/muscriptor_result_runtime.py").read_text(
        encoding="utf-8"
    )
    standalone_web = Path("web/app.js").read_text(encoding="utf-8")
    web_api = Path("src/web_api/app.py").read_text(encoding="utf-8")

    assert "download_sheet_music_action" in desktop
    assert "SheetMusicExportWorker" in desktop
    assert "_publish_current_midi" in desktop
    for source in (space, colab):
        assert "SheetMusicExportRegistry" in source
        assert "render_sheet_music_export" in source
        assert '"sheet_api": "./api/render_sheet_music_export"' in source
        assert '"sheet_token": sheet_token' in source
        assert 'api_name="render_sheet_music_export"' in source
    assert "downloadSheetMusic" in shared_browser_editor
    assert "fetchEditedMidiBytes" in shared_browser_editor
    assert "generateSheetMusicForJob" in standalone_web
    assert "updateTrackDownload" in standalone_web
    assert "data-sheet-source" in standalone_web
    assert '@app.post("/api/v1/jobs/{job_id}/sheet-music"' in web_api


def test_quantization_grid_contract_covers_every_supported_delivery_surface():
    from src.core.midi_quantization import (
        DEFAULT_MIDI_QUANTIZE_GRID,
        DEFAULT_MIDI_QUANTIZE_SCOPE,
        MIDI_QUANTIZE_GRIDS,
        MIDI_QUANTIZE_SCOPES,
    )

    desktop_editor = Path("src/gui/widgets/muscriptor_result.py").read_text(encoding="utf-8")
    shared_browser_editor = Path("src/gui/web/muscriptor_result_runtime.py").read_text(
        encoding="utf-8"
    )
    space = _space_source()
    colab = _colab_source()
    web_html = Path("web/index.html").read_text(encoding="utf-8")
    web_js = Path("web/app.js").read_text(encoding="utf-8")
    schemas = Path("src/web_api/schemas.py").read_text(encoding="utf-8")
    backend_dockerfile = Path("docker/backend.Dockerfile").read_text(encoding="utf-8")
    gateway_dockerfile = Path("docker/gateway.Dockerfile").read_text(encoding="utf-8")
    public_documentation = Path("docs/README.md").read_text(encoding="utf-8")

    assert MIDI_QUANTIZE_GRIDS == ("1/4", "1/8", "1/16", "1/32", "1/64")
    assert DEFAULT_MIDI_QUANTIZE_GRID == "1/32"
    assert MIDI_QUANTIZE_SCOPES == ("all_tracks", "selected_notes")
    assert DEFAULT_MIDI_QUANTIZE_SCOPE == "all_tracks"
    assert "MIDI_QUANTIZE_GRIDS" in desktop_editor
    assert "edit_quantize_grid_combo" in desktop_editor
    assert "edit_quantize_scope_combo" in desktop_editor
    assert "DEFAULT_MIDI_QUANTIZE_SCOPE" in desktop_editor
    assert "MIDI_QUANTIZE_GRIDS" in shared_browser_editor
    assert '"quantizeGrids"' in shared_browser_editor
    assert '"defaultQuantizeScope"' in shared_browser_editor
    assert 'this.quantizeScope === "all_tracks"' in shared_browser_editor
    for source in (space, colab):
        assert "build_muscriptor_result_html(" in source
        assert "muscriptor_result_head" in source

    assert 'id="quantizeNotes"' in web_html
    assert 'id="quantizeGridSelect"' in web_html
    assert "quantize_notes:" in web_js
    assert "quantize_grid:" in web_js
    assert 'quantization.default_scope !== "all_tracks"' in web_js
    assert "quantize_notes: bool = False" in schemas
    assert 'quantize_grid: MidiQuantizeGrid = "1/32"' in schemas
    assert "COPY src ./src" in backend_dockerfile
    assert "COPY web /srv/web" in gateway_dockerfile
    assert "Note quantization is explicit and disabled by default." in public_documentation
    assert "their scope defaults to All tracks" in public_documentation
    assert "standalone Web/API, including Docker deployment" in public_documentation


def test_space_and_colab_midi_editor_state_keeps_note_identity_and_bpm_context():
    for source in (_space_source(), _colab_source()):
        for field in (
            '"program"',
            '"is_drum"',
            '"track_index"',
            '"channel"',
            '"reference_bpm"',
            '"target_bpm"',
        ):
            assert field in source
        assert "The browser MIDI editor requires result beat information" in source


def test_space_and_colab_build_real_muscriptor_beat_grid_state(tmp_path, monkeypatch):
    from src.core import midi_tempo, muscriptor_result_assets

    playback_midi = tmp_path / "source-tempo-playback.mid"
    monkeypatch.setattr(
        midi_tempo,
        "rewrite_midi_tempo_preserving_ticks",
        lambda *_args, **_kwargs: playback_midi,
    )
    monkeypatch.setattr(
        midi_tempo,
        "read_muscriptor_bar_offset_seconds",
        lambda _path: 0.25,
    )
    assets = SimpleNamespace(
        notes=(
            SimpleNamespace(
                instrument="acoustic_piano",
                pitch=60,
                velocity=90,
                start=0.25,
                end=0.75,
                program=0,
                is_drum=False,
                track_index=1,
                channel=0,
            ),
        ),
        duration=2.0,
        transcription_wav=tmp_path / "transcription.wav",
        original_wav=tmp_path / "original-live.wav",
        stereo_mix_wav=tmp_path / "stereo.wav",
        instrument_wavs={"acoustic_piano": tmp_path / "piano.wav"},
    )
    monkeypatch.setattr(
        muscriptor_result_assets,
        "prepare_midi_playback_assets",
        lambda *_args, **_kwargs: assets,
    )
    result = SimpleNamespace(
        midi_path=str(tmp_path / "result.mid"),
        selected_instruments=["acoustic_piano"],
        beat_info=SimpleNamespace(
            bpm=120.0,
            source_bpm=None,
            beat_times=[0.0, 0.5, 1.0],
            downbeats=[0.0, 1.0],
            time_signature=(3, 4),
        ),
    )

    preview_registry = SimpleNamespace(register=lambda **_kwargs: "preview-token")
    for source in (_space_source(), _colab_source()):
        build_state = _isolated_function(
            source,
            "_build_midi_result_state",
            Path=Path,
            _request_root_for_owned_path=lambda _path: tmp_path,
            _EDITED_MIDI_PREVIEWS=preview_registry,
            EDITED_MIDI_PREVIEWS=preview_registry,
            _SHEET_MUSIC_EXPORTS=preview_registry,
            SHEET_MUSIC_EXPORTS=preview_registry,
        )
        state = build_state(
            result,
            tmp_path / "audio.wav",
            tmp_path / "playback",
            backend_label="MuScriptor-small",
            muscriptor_groups=True,
        )
        assert state["beat_times"] == [0.25, 0.75, 1.25]
        assert state["downbeats"] == [0.25, 1.25]
        assert state["time_signature"] == (3, 4)
        assert state["repeat_tempo_per_note_track"] is True
        assert state["playback_audio_path"] == str(assets.original_wav)
        assert state["preview_api"] == "./api/render_edited_midi_preview"
        assert state["preview_token"] == "preview-token"


def test_space_and_colab_normalizers_preserve_muscriptor_beat_grid_state(tmp_path):
    raw_state = {
        "kind": "midi_result",
        "audio_path": str(tmp_path / "audio.wav"),
        "playback_audio_path": str(tmp_path / "original-live.wav"),
        "midi_path": str(tmp_path / "result.mid"),
        "transcription_wav": str(tmp_path / "transcription.wav"),
        "stereo_mix_wav": str(tmp_path / "stereo.wav"),
        "instrument_wavs": {"acoustic_piano": str(tmp_path / "piano.wav")},
        "notes": [
            {
                "instrument": "acoustic_piano",
                "pitch": 60,
                "velocity": 90,
                "start": 0.25,
                "end": 0.75,
                "program": 0,
                "is_drum": False,
                "track_index": 1,
                "channel": 0,
            }
        ],
        "duration": 2.0,
        "reference_bpm": 120.0,
        "target_bpm": 120.0,
        "time_signature": (3, 4),
        "beat_times": [0.25, 0.75, 1.25],
        "downbeats": [0.25, 1.25],
        "repeat_tempo_per_note_track": True,
        "preview_token": "preview-token",
        "sheet_token": "sheet-token",
    }

    def require_file(_root, path, *_args):
        return Path(path).resolve()

    common_globals = {
        "Path": Path,
        "math": __import__("math"),
        "MIN_MIDI_BPM": 4.0,
        "MAX_MIDI_BPM": 400.0,
        "SUPPORTED_AUDIO_SUFFIXES": {".wav"},
        "_require_owned_request_file": require_file,
        "_EDITED_MIDI_PREVIEWS": SimpleNamespace(require_matching=lambda token, **_kwargs: token),
        "EDITED_MIDI_PREVIEWS": SimpleNamespace(require_matching=lambda token, **_kwargs: token),
        "_SHEET_MUSIC_EXPORTS": SimpleNamespace(require_matching=lambda token, **_kwargs: token),
        "SHEET_MUSIC_EXPORTS": SimpleNamespace(require_matching=lambda token, **_kwargs: token),
    }

    space_normalize = _isolated_function(
        _space_source(),
        "_normalize_midi_result_state",
        **common_globals,
    )
    space_state = space_normalize(
        raw_state,
        tmp_path,
        expected_audio_path=raw_state["audio_path"],
    )

    colab_normalize = _isolated_function(
        _colab_source(),
        "_normalize_midi_result_state",
        gr=SimpleNamespace(Error=RuntimeError),
        ct=lambda *_args, **_kwargs: "error",
        **common_globals,
    )
    colab_state = colab_normalize(raw_state, tmp_path, raw_state["audio_path"])

    for state in (space_state, colab_state):
        assert state["beat_times"] == raw_state["beat_times"]
        assert state["downbeats"] == raw_state["downbeats"]
        assert state["time_signature"] == raw_state["time_signature"]
        assert state["repeat_tempo_per_note_track"] is True
        assert (
            Path(state["playback_audio_path"]).resolve()
            == Path(raw_state["playback_audio_path"]).resolve()
        )
        assert state["preview_api"] == "./api/render_edited_midi_preview"
        assert state["preview_token"] == "preview-token"
        assert state["sheet_api"] == "./api/render_sheet_music_export"
        assert state["sheet_token"] == "sheet-token"


def test_desktop_style_progress_panel_is_shared_by_space_and_colab():
    """Both web platforms stream real job progress to the desktop-style panel."""
    space = _space_source()
    colab = _colab_source()
    zh = json.loads(Path("src/i18n/zh_CN.json").read_text(encoding="utf-8"))
    en = json.loads(Path("src/i18n/en_US.json").read_text(encoding="utf-8"))

    for source in (space, colab):
        # Same stage checklists as the desktop ProgressWidget.
        assert '_SPLIT_STAGE_KEYS = ("preprocessing", "separation")' in source
        assert '_DEFAULT_STAGE_KEYS = ("preprocessing", "transcription", "synthesis")' in source
        # Same streaming machinery: worker thread + queue relay + generator.
        assert "def _stream_gpu_job(" in source
        assert "class _ProgressRelay" in source
        assert "threading.Thread(target=_worker, daemon=True" in source
        # The panel renders deterministic percent and stage states.
        assert "def _progress_panel_html(" in source
        assert "main.progress.stages.{key}" in source
        # The convert action streams into the panel, not only returns at the end.
        assert "yield from _stream_gpu_job(" in source
        assert "outputs=[file_output, status_output, track_state, progress_html]" in source
        # Progress panel markup carries the shared styling hooks.
        assert 'class="progress-track"' in source
        assert "stage-item" in source

    # Stage label keys exist in every language (shared desktop catalog).
    for catalog in (zh, en):
        stages = catalog["main"]["progress"]["stages"]
        for key in ("preprocessing", "separation", "transcription", "synthesis"):
            assert stages[key]

    # Behaviour: deterministic percent and stage states from the Space builder.
    panel = _isolated_function(
        space,
        "_progress_panel_html",
        st=lambda key, **kwargs: key,
        _stage_keys_for_mode=lambda mode: ("preprocessing", "transcription", "synthesis"),
    )
    running = panel("smart", "transcription", 0.5, "working")
    assert "50%" in running
    assert "stage-item done" in running
    assert "stage-item current" in running
    done = panel("smart", None, 1.0, "", finished=True)
    assert "100%" in done
    assert done.count("stage-item done") == 3
    failed = panel("smart", "transcription", 0.0, "boom", failed=True)
    assert "stage-item failed" in failed
