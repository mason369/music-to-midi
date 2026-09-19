"""Regression coverage for instrument exports, group solo, transport and harmony."""

import io
import json
import os
import zipfile
from collections import Counter
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mido
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog

from src.core.midi_chords import MidiChord, chord_pitches, split_chords_at_bars
from src.core.midi_stem_export import export_midi_selection, export_midi_stems_zip
from src.core.muscriptor_result_assets import MuscriptorRollNote, read_midi_roll_notes
from src.gui.widgets.muscriptor_result import MuscriptorResultWidget
from src.i18n.translator import set_language
from tests.test_browser_editor_interactions import run_javascript


@pytest.fixture
def source_midi(tmp_path):
    midi = mido.MidiFile(type=1)
    meta = mido.MidiTrack(
        [
            mido.MetaMessage("set_tempo", tempo=500000),
            mido.MetaMessage("time_signature", numerator=3, denominator=4),
            mido.MetaMessage("marker", text="preserve me", time=240),
        ]
    )
    midi.tracks.append(meta)
    for index, program in enumerate((0, 32, 33, 0)):
        channel = index if index < 3 else 9
        track = mido.MidiTrack(
            [
                mido.MetaMessage("track_name", name=f"instrument {index}"),
                mido.Message("program_change", program=program, channel=channel),
                mido.Message("control_change", control=64, value=127, channel=channel, time=120),
                mido.Message(
                    "note_on", note=48 + index, velocity=80 + index, channel=channel, time=120
                ),
                mido.Message("pitchwheel", pitch=600, channel=channel, time=240),
                mido.Message("note_off", note=48 + index, channel=channel, time=480),
                mido.Message("control_change", control=64, value=0, channel=channel, time=120),
            ]
        )
        midi.tracks.append(track)
    path = tmp_path / "source.mid"
    midi.save(path)
    return path


def test_batch_midi_archive_contains_every_instrument_and_preserves_ticks(source_midi, tmp_path):
    original = source_midi.read_bytes()
    notes = read_midi_roll_notes(source_midi)
    notes = (replace(notes[0], pitch=65, velocity=113),) + notes[1:]
    destination, count = export_midi_stems_zip(
        source_midi,
        tmp_path / "所有声部.zip",
        notes,
        reference_bpm=120,
        target_bpm=90,
        instrument_label=lambda _: "重复/名称",
    )
    assert count == 4
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert len(set(archive.namelist())) == 4
        found = []
        for name in archive.namelist():
            midi = mido.MidiFile(file=io.BytesIO(archive.read(name)))
            assert midi.ticks_per_beat == 480
            assert all("/" not in name and "\\" not in name for name in archive.namelist())
            found.extend(
                msg.note for track in midi.tracks for msg in track if msg.type == "note_on"
            )
            assert sum(msg.type == "note_on" for track in midi.tracks for msg in track) == 1
            assert [
                msg.tempo for track in midi.tracks for msg in track if msg.type == "set_tempo"
            ] == [mido.bpm2tempo(90)]
            for track in midi.tracks:
                tick = 0
                for message in track:
                    tick += message.time
                    if message.type == "note_on":
                        assert tick == 240
                    elif message.type == "pitchwheel":
                        assert tick == 480 and message.pitch == 600
                    elif message.type == "control_change":
                        assert (tick, message.value) in {(120, 127), (1080, 0)}
    assert Counter(found) == Counter(note.pitch for note in notes)
    assert source_midi.read_bytes() == original


def test_two_basses_can_export_together_and_empty_selection_never_overwrites(source_midi, tmp_path):
    notes = read_midi_roll_notes(source_midi)
    selected = tuple(note for note in notes if note.program in (32, 33))
    destination = tmp_path / "basses.mid"
    export_midi_selection(source_midi, destination, selected, reference_bpm=120, target_bpm=120)
    assert {note.program for note in read_midi_roll_notes(destination)} == {32, 33}
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="No audible"):
        export_midi_selection(source_midi, destination, (), reference_bpm=120, target_bpm=120)
    assert destination.read_bytes() == before


def chord_notes(pitches, start=0, end=2, drum=False):
    return tuple(
        MuscriptorRollNote("drums" if drum else "gm:000", pitch, 90, start, end, 0, drum)
        for pitch in pitches
    )


@pytest.mark.parametrize(
    "pitches,label",
    [
        ((60, 64, 67), "C"),
        ((69, 72, 76), "A:min"),
        ((60, 64, 67, 70), "C:7"),
        ((60, 63, 66), "C:dim"),
    ],
)
def test_chord_voicings_preserve_model_labels_and_bar_boundaries(pitches, label):
    assert chord_pitches(label) == pitches
    chords = split_chords_at_bars((MidiChord(0, 4, label, pitches),), 120)
    assert [(c.start, c.end, c.label) for c in chords] == [(0, 2, label), (2, 4, label)]
    assert len(chords[0].pitches) >= 3


def test_chord_display_keeps_native_boundaries_and_silent_classes():
    assert chord_pitches("N") == chord_pitches("X") == ()
    raw = (MidiChord(0.03, 3.97, "C:maj7", chord_pitches("C:maj7")),)
    chords = split_chords_at_bars(raw, 120, (3, 4))
    assert [(c.start, c.end) for c in chords] == [(0.03, 1.5), (1.5, 3.0), (3.0, 3.97)]
    assert all(c.pitches == (60, 64, 67, 71) for c in chords)


@pytest.fixture
def result_widget(source_midi):
    app = QApplication.instance() or QApplication([])
    widget = MuscriptorResultWidget(
        str(source_midi.with_suffix(".wav")), [], muscriptor_groups=False
    )
    notes = read_midi_roll_notes(source_midi)
    widget._midi_path = str(source_midi)
    widget.set_bpm_context(120, 120, time_signature=(3, 4))
    widget._detected = list(dict.fromkeys(note.instrument for note in notes))
    widget.roll.set_notes(notes, duration=6)
    widget._begin_editor_session(notes, 6)
    widget._rebuild_instrument_rows()
    widget.resize(1200, 800)
    widget.show()
    app.processEvents()
    yield widget
    widget.shutdown()
    widget.close()
    app.processEvents()


def test_default_audibility_and_multi_solo_preserve_manual_mutes(result_widget):
    w = result_widget
    assert w._effective_muted() == set()
    assert not w.roll._muted
    assert all(not row.mute_button.isChecked() for row in w._instrument_rows.values())
    QTest.mouseClick(w._instrument_rows["gm:032"].solo_button, Qt.MouseButton.LeftButton)
    QTest.mouseClick(w._instrument_rows["gm:033"].solo_button, Qt.MouseButton.LeftButton)
    assert w._soloed == {"gm:032", "gm:033"}
    assert w._effective_muted() == {"gm:000", "drums"}
    assert w.roll._muted == w._effective_muted()
    QTest.mouseClick(w._instrument_rows["gm:000"].mute_button, Qt.MouseButton.LeftButton)
    w._toggle_solo("gm:032")
    w._toggle_solo("gm:033")
    assert w._muted == w._effective_muted() == {"gm:000"}


@pytest.mark.parametrize("language", ["zh_CN", "en_US"])
def test_export_actions_snapshot_real_selection_and_all_stems(
    result_widget, tmp_path, monkeypatch, language
):
    set_language(language)
    w = result_widget
    w.update_translations()
    w._toggle_solo("gm:032")
    w._toggle_solo("gm:033")
    destination = tmp_path / "selected.mid"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(destination), ""))
    w.download_audible_midi_action.trigger()
    assert {note.program for note in read_midi_roll_notes(destination)} == {32, 33}
    destination = tmp_path / "all.zip"
    w.download_midi_stems_action.trigger()
    with zipfile.ZipFile(destination) as archive:
        assert len(archive.namelist()) == 4
    assert "muscriptor_result." not in w.playback_status_label.text()
    set_language("zh_CN")


def test_space_shortcut_and_stop_rewind_do_not_intercept_numeric_input(result_widget, monkeypatch):
    w = result_widget
    toggles = []
    monkeypatch.setattr(w, "_toggle_playback", lambda: toggles.append(True))
    w.play_button.setEnabled(True)
    w.roll.setFocus()
    QTest.keyClick(w.roll, Qt.Key.Key_Space)
    assert toggles == [True]
    w.bpm_spin.setFocus()
    QTest.keyClick(w.bpm_spin, Qt.Key.Key_Space)
    assert toggles == [True]
    w.roll.setFocus()
    w.seek(2)
    QTest.keyClick(w.roll, Qt.Key.Key_Space, Qt.KeyboardModifier.ShiftModifier)
    assert w._position_ms == 0 and w.roll.position == 0 and w.playback_slider.value() == 0


def test_chord_lane_alignment_after_zoom_scroll_and_bpm_change(result_widget):
    w = result_widget
    w.roll.set_notes(chord_notes((60, 64, 67), end=6), duration=6)
    w.chord_lane._source_chords = (MidiChord(0, 6, "C", (60, 64, 67)),)
    w.chord_lane.refresh()
    assert w.chord_lane.chords
    before = w.chord_lane.chords
    w.bpm_spin.setValue(90)
    assert w.chord_lane.chords == before
    w.roll.set_pixels_per_second(180)
    w.roll_scroll.horizontalScrollBar().setValue(100)
    w.roll.set_render_offset(3)
    expected = (
        w.chord_lane.mapFromGlobal(w.roll_scroll.viewport().mapToGlobal(QPoint())).x()
        + w.roll.x_for_time_float(1.5)
        - 103
    )
    assert w.chord_lane.x_for_time(1.5) == pytest.approx(expected)
    chosen = []
    w.chord_lane.audition_requested.disconnect(w._audition_chord)
    w.chord_lane.audition_requested.connect(chosen.append)
    chord = w.chord_lane.chords[1]
    QTest.mouseClick(
        w.chord_lane,
        Qt.MouseButton.LeftButton,
        pos=w.chord_lane.chord_rect(chord).center().toPoint(),
    )
    assert chosen == [chord]


def test_browser_group_solo_and_stop_cancel_pending_playback(tmp_path):
    run_javascript(
        tmp_path,
        r"""
const s=session(); s.solo=new Set(); s.muted=new Set();
s.m.instruments=[{id:'piano',detected:true},{id:'bass1',detected:true},{id:'bass2',detected:true}];
s.activateInstrument=function(){};s.syncRows=function(){};
assert.equal(s.audible('piano'),true);
s.toggleSolo('bass1');s.toggleSolo('bass2');
assert.equal(s.audible('piano'),false);assert.equal(s.audible('bass1'),true);assert.equal(s.audible('bass2'),true);
s.toggleMute('piano');s.toggleSolo('bass1');s.toggleSolo('bass2');
assert.equal(s.audible('piano'),false);assert.deepEqual([...s.muted],['piano']);
s.position=3;const start=s.start();s.stop();pending.shift().resolve();await start;
assert.equal(s.playing,false);assert.equal(s.position,0);
""",
    )


def test_browser_zip_is_readable_and_utf8_safe(tmp_path):
    path = tmp_path / "browser.zip"
    run_javascript(
        tmp_path,
        r"""
const zip=window.musicToMidiMidiEditorRuntime.buildMidiZip([
 {name:'01-电贝司.mid',bytes:new Uint8Array([0,1,2,3,255])},
 {name:'02-原声贝司.mid',bytes:new Uint8Array([4,5,6])}
]);
require('node:fs').writeFileSync(PATH,Buffer.from(await zip.arrayBuffer()));
""".replace("PATH", json.dumps(str(path))),
    )
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ["01-电贝司.mid", "02-原声贝司.mid"]
        assert archive.read("01-电贝司.mid") == bytes((0, 1, 2, 3, 255))


def test_chord_status_follows_language_and_end_without_clobbering_export(result_widget):
    from PyQt6.QtMultimedia import QMediaPlayer

    from src.i18n.translator import t

    w = result_widget
    try:
        w._on_chord_ready()
        set_language("en_US")
        w.update_translations()
        assert w.playback_status_label.text() == t("muscriptor_result.chord_playing")
        w.chord_auditioner.player.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.EndOfMedia)
        assert w.playback_status_label.text() == t("muscriptor_result.final_audio_ready")
        assert w._chord_status is None
        w._on_chord_ready()
        w.stop()
        assert w.playback_status_label.text() == t("muscriptor_result.final_audio_ready")
        w._on_chord_ready()
        w.playback_status_label.setText("Saved export")
        w.stop()
        assert w.playback_status_label.text() == "Saved export"
    finally:
        set_language("zh_CN")


def test_closing_result_releases_chord_audio_before_removing_cache(result_widget):
    import wave

    from PyQt6.QtCore import QUrl

    w = result_widget
    root = w._preview_root
    source = root / "chords" / "closing.wav"
    with wave.open(str(source), "wb") as output:
        output.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        output.writeframes(b"\x10\x00" * 44100)
    w.chord_auditioner.player.setSource(QUrl.fromLocalFile(str(source)))
    w.chord_auditioner.player.play()
    QTest.qWait(100)
    w.shutdown()
    assert not root.exists()
