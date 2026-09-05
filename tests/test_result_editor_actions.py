"""Real Qt events: toolbar contracts, boundary inputs and deterministic chaos."""

import math
import os
import random

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from src.core.muscriptor_result_assets import MuscriptorRollNote
from src.gui.widgets.muscriptor_result import MuscriptorResultWidget
from src.i18n.translator import set_language

COMMANDS = (
    "edit_add_button",
    "edit_delete_button",
    "edit_undo_button",
    "edit_redo_button",
    "edit_reset_button",
    "edit_select_all_button",
    "edit_cut_button",
    "edit_copy_button",
    "edit_paste_button",
    "edit_duplicate_button",
    "edit_quantize_button",
)


@pytest.fixture
def editor(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = MuscriptorResultWidget(
        str(tmp_path / "controls.wav"), ["acoustic_piano", "acoustic_guitar"]
    )
    # These tests isolate UI state from audio synthesis; the real render/export
    # integration tests exercise the worker and MIDI event preservation separately.
    renders = []
    monkeypatch.setattr(
        widget, "_queue_editor_audio_render", lambda before, after: renders.append(after)
    )
    widget.set_bpm_context(128.4, 128.4)
    notes = tuple(
        MuscriptorRollNote(
            "acoustic_piano" if i % 2 == 0 else "acoustic_guitar",
            60 + i,
            70 + i,
            0.113 + i * 0.33,
            0.287 + i * 0.33,
            0 if i % 2 == 0 else 24,
            False,
            i % 2,
            i % 2,
        )
        for i in range(12)
    )
    widget.roll.set_notes(notes, duration=8)
    widget._begin_editor_session(notes, 8)
    widget.resize(1280, 950)
    widget.show()
    app.processEvents()
    try:
        yield widget, renders
    finally:
        widget.shutdown()
        widget.close()
        app.processEvents()


def click(control):
    QTest.mouseClick(control, Qt.MouseButton.LeftButton)


def assert_invariants(w):
    assert tuple(w.roll._notes) == w._edited_notes
    assert all(0 <= i < len(w._edited_notes) for i in w.roll.selected_indices)
    for note in w._edited_notes:
        assert math.isfinite(note.start) and math.isfinite(note.end)
        assert 0 <= note.start < note.end <= w._edit_duration + 1e-9
        assert 21 <= note.pitch <= 108 and 1 <= note.velocity <= 127
    assert len(w._edit_undo) <= 100
    assert 4 <= w.bpm_spin.value() <= 400


def test_temp_roots_are_canonical_and_cleanup_keeps_owned_audio(editor, tmp_path):
    w, _ = editor
    for root in (w._preview_root, w._edit_asset_root, w._audio_export_root, w._sheet_export_root):
        assert root == root.resolve()
    active = w._edit_asset_root / "generation-000001"
    deferred = w._edit_asset_root / "generation-000002"
    stale = w._edit_asset_root / "generation-000003"
    for directory in (active, deferred, stale):
        directory.mkdir()
        (directory / "owned.wav").write_bytes(b"owned audio fixture")
    w._active_edit_asset_dir = active
    w._deferred_editor_assets = (2, object())
    w._remove_editor_audio_directory(active)
    w._remove_editor_audio_directory(deferred)
    w._remove_editor_audio_directory(stale)
    assert (active / "owned.wav").exists() and (deferred / "owned.wav").exists()
    assert not stale.exists()
    with pytest.raises(RuntimeError):
        w._remove_editor_audio_directory(w._edit_asset_root)
    with pytest.raises(RuntimeError):
        w._remove_editor_audio_directory(tmp_path)
    # Superseded / unowned output is eventually reclaimable.
    w._deferred_editor_assets = None
    w._remove_editor_audio_directory(deferred)
    assert not deferred.exists()


@pytest.mark.parametrize("language", ["zh_CN", "en_US"])
def test_no_bpm_placeholder_then_real_120_and_manual_changes(tmp_path, language):
    app = QApplication.instance() or QApplication([])
    set_language(language)
    w = MuscriptorResultWidget(str(tmp_path / "unknown.wav"), [])
    try:
        w.show()
        app.processEvents()
        assert w._detected_bpm is None
        assert not w.bpm_spin.isVisible() and not w.speed_spin.isVisible()
        assert "120" not in w.tempo_status_label.text()
        assert w.tempo_status_label.text() and w.tempo_status_label.isVisible()
        w.set_bpm_context(120, 120)
        assert w.bpm_spin.isVisible() and "120.0" in w.tempo_status_label.text()
        changes = []
        w.tempo_changed.connect(lambda: changes.append(w.bpm_spin.value()))
        w.bpm_spin.setValue(89.3)
        assert changes == [89.3]
        assert "120.0" in w.tempo_status_label.text()  # Detection is not overwritten.
    finally:
        w.shutdown()
        w.close()


@pytest.mark.parametrize("source", [4, 4.1, 60.1, 117.9, 128.4, 399.9, 400])
def test_speed_spin_both_endpoints_never_create_invalid_bpm(editor, source):
    w, _ = editor
    w.set_bpm_context(source, source)
    for value in (w.speed_spin.maximum(), w.speed_spin.minimum()):
        assert 4 <= source * value <= 400
        w.speed_spin.setValue(value)
        assert 4 <= w.bpm_spin.value() <= 400


def test_all_edit_buttons_disabled_until_explicit_edit_action(editor):
    w, renders = editor
    before = w._edited_notes
    for name in COMMANDS:
        control = getattr(w, name)
        assert not control.isEnabled()
        click(control)
    assert w._edited_notes == before and renders == []
    click(w.edit_toggle)
    click(w.edit_select_all_button)
    click(w.edit_copy_button)
    assert w._edit_clipboard == before
    click(w.edit_cut_button)
    assert not w._edited_notes
    click(w.edit_paste_button)
    assert len(w._edited_notes) == len(before)
    click(w.edit_undo_button)
    assert not w._edited_notes
    click(w.edit_redo_button)
    assert len(w._edited_notes) == len(before)
    click(w.edit_reset_button)
    assert w._edited_notes == before
    click(w.edit_add_button)
    assert len(w._edited_notes) == len(before) + 1
    click(w.edit_delete_button)
    assert w._edited_notes == before
    click(w.edit_select_all_button)
    click(w.edit_duplicate_button)
    assert len(w._edited_notes) == len(before) * 2
    assert_invariants(w)


@pytest.mark.parametrize("grid", ["1/4", "1/8", "1/16", "1/32", "1/64"])
@pytest.mark.parametrize("scope", ["all_tracks", "selected_notes"])
def test_every_quantization_grid_scope_and_history(editor, grid, scope):
    w, renders = editor
    click(w.edit_toggle)
    before = w._edited_notes
    w.roll.set_selected_indices((0, 3, 8), primary=3)
    w.edit_quantize_grid_combo.setCurrentText(grid)
    w.edit_quantize_scope_combo.setCurrentIndex(w.edit_quantize_scope_combo.findData(scope))
    assert w._edited_notes == before and renders == []
    click(w.edit_quantize_button)
    after = w._edited_notes
    assert before != after
    if scope == "selected_notes":
        assert all(after[i] == before[i] for i in range(len(before)) if i not in (0, 3, 8))
    assert w.roll.selected_indices == (0, 3, 8)
    history, generation = len(w._edit_undo), len(renders)
    click(w.edit_quantize_button)
    assert w._edited_notes == after
    assert (len(w._edit_undo), len(renders)) == (history, generation)
    click(w.edit_undo_button)
    assert w._edited_notes == before
    click(w.edit_redo_button)
    assert w._edited_notes == after
    assert_invariants(w)


def test_noop_snapshot_keeps_paint_cache_and_selection(editor, monkeypatch):
    w, renders = editor
    click(w.edit_toggle)
    calls = []
    monkeypatch.setattr(w.roll, "set_notes", lambda *args, **kwargs: calls.append(args))
    w._apply_editor_snapshot(
        w._edited_notes, selected_indices=(1, 4), selected_index=4, record=True
    )
    assert w.roll.selected_indices == (1, 4) and w.roll.selected_index == 4
    assert calls == [] and renders == []


def test_result_summary_tracks_current_project_bpm_and_malformed_context_is_rejected(
    tmp_path, monkeypatch
):
    from src.gui.main_window import MainWindow
    from src.models.data_models import BeatInfo, Config, ProcessingResult, Track, TrackType

    monkeypatch.setattr(MainWindow, "_start_gpu_detection", lambda self: None)
    monkeypatch.setattr(MuscriptorResultWidget, "finalize_result", lambda self, result: None)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(Config(language="zh_CN", output_dir=str(tmp_path)))
    try:
        window._show_muscriptor_result(
            ProcessingResult(
                midi_path=str(tmp_path / "fixture.mid"),
                tracks=[Track(TrackType.OTHER, str(tmp_path / "fixture.wav"))],
                beat_info=BeatInfo(128.4),
                total_notes=12,
            ),
            reveal=True,
        )
        w = window.muscriptor_result_widget
        assert "128.4" in window.result_info_label.text()
        for invalid in (float("nan"), float("inf"), -120, 0, 401):
            with pytest.raises(ValueError):
                w.set_bpm_context(invalid, 90)
        assert w._detected_bpm == 128.4
        w.bpm_spin.setValue(89.3)
        assert "89.3" in window.result_info_label.text()
        assert "128.4" not in window.result_info_label.text()
        assert "128.4" in w.tempo_status_label.text()
        app.processEvents()
    finally:
        window.close()


def test_scroll_over_any_value_control_does_not_edit(editor):
    w, renders = editor
    click(w.edit_toggle)
    click(w.edit_select_all_button)
    controls = (
        w.bpm_spin,
        w.speed_spin,
        w.roll_zoom_spin,
        w.edit_velocity_spin,
        w.edit_instrument_combo,
        w.edit_quantize_scope_combo,
        w.edit_quantize_grid_combo,
        w.mix_slider,
    )
    before = w._edited_notes
    for control in controls:
        value = control.value() if hasattr(control, "value") else control.currentIndex()
        control.setFocus()
        event = QWheelEvent(
            QPointF(5, 5),
            QPointF(5, 5),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(control, event)
        assert (control.value() if hasattr(control, "value") else control.currentIndex()) == value
    assert w._edited_notes == before and renders == []


def test_selection_paint_culls_offscreen_notes_without_losing_duplicates(editor):
    w, _ = editor
    note = MuscriptorRollNote("acoustic_piano", 60, 90, 0, 10000, 0, False, 0, 0)
    notes = (
        note,
        note,
        *(
            MuscriptorRollNote("acoustic_piano", 61, 90, i, i + 0.1, 0, False, 0, 0)
            for i in range(10000)
        ),
    )
    w.roll.set_notes(notes, duration=10000, selected_indices=range(len(notes)))
    visible = w.roll._visible_selected_indices(
        w.roll.x_for_time_float(500), w.roll.x_for_time_float(503)
    )
    assert visible == (0, 1, 502, 503, 504, 505)
    assert len(w.roll.selected_indices) == 10002
    assert w.roll._notes == notes


@pytest.mark.parametrize("seed", [20260906, 731, 4090])
def test_600_unexpected_actions_preserve_state_and_exact_reset(editor, seed):
    w, _ = editor
    rng = random.Random(seed)
    original = w._edited_notes
    names = (*COMMANDS, "edit_toggle", "follow_checkbox")
    for i in range(600):
        action = rng.randrange(8)
        if action < 3:
            name = rng.choice(names)
            # Keep test input bounded; the product imposes no artificial note cap.
            if len(w._edited_notes) > 256:
                name = "edit_reset_button"
            click(getattr(w, name))
        elif action == 3:
            w.edit_quantize_grid_combo.setCurrentIndex(rng.randrange(5))
            w.edit_quantize_scope_combo.setCurrentIndex(rng.randrange(2))
        elif action == 4:
            w.seek(rng.uniform(-4, 12))
        elif action == 5:
            w.bpm_spin.setValue(rng.uniform(4, 400))
        elif action == 6:
            w.edit_velocity_spin.setValue(rng.randrange(1, 128))
        else:
            w.roll_zoom_spin.setValue(rng.uniform(0.5, 4))
        assert_invariants(w)
        if i % 30 == 0:
            QApplication.processEvents()
    if not w.edit_toggle.isChecked():
        click(w.edit_toggle)
    click(w.edit_reset_button)
    assert w._edited_notes == original
