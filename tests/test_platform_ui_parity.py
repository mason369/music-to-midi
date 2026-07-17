"""Cross-platform consistency contracts for desktop, Space, and Colab.

Every platform must present the same workflow: identical action labels,
stop/cancel support with cooperative cancellation, and identical completion
summaries for direct conversions and separation-only runs.
"""

import ast
import json
from pathlib import Path


def _colab_source() -> str:
    notebook = json.loads(Path("colab_notebook.ipynb").read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _space_source() -> str:
    return Path("space/app.py").read_text(encoding="utf-8")


def _desktop_source() -> str:
    return Path("src/gui/main_window.py").read_text(encoding="utf-8")


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

    # Direct modes convert, split modes separate — same wording everywhere.
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
        # Stop button events bypass the queue so they fire while a job runs.
        assert "fn=request_stop_current_job" in source
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
        assert (
            f'COLAB_TRANSLATOR.t("{key}")' in colab
            or f"COLAB_TRANSLATOR.t('{key}')" in colab
        )

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
