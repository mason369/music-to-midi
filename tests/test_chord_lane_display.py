"""Chord notation, visible highlighting and timeline hit testing in real Qt widgets."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QEvent, QObject, QPoint, QPointF, Qt
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from src.core.midi_chords import QUALITIES, MidiChord, chord_pitches
from src.gui.widgets.midi_chord_lane import MidiChordLane, _chord_presentation
from src.gui.widgets.muscriptor_result import _PianoRollCanvas
from src.i18n.translator import Translator, set_language, t


@pytest.fixture
def lane(qapp):
    window = QWidget()
    layout = QVBoxLayout(window)
    scroll = QScrollArea()
    roll = _PianoRollCanvas()
    scroll.setWidget(roll)
    roll.set_daw_grid(120, (4, 4))
    chord_lane = MidiChordLane(roll, scroll)
    layout.addWidget(chord_lane)
    layout.addWidget(scroll)
    chord_lane._source_chords = tuple(
        MidiChord(start, end, label, chord_pitches(label))
        for start, end, label in [(0, 1, "C:maj7"), (1, 3, "D:min7"), (3, 4, "N"), (4, 5, "X")]
    )
    chord_lane.refresh()
    window.resize(780, 400)
    window.show()
    QTest.qWait(100)
    yield chord_lane
    chord_lane.shutdown()
    window.close()
    set_language("zh_CN")
    qapp.processEvents()


@pytest.mark.parametrize(
    "label,expected",
    [
        ("C", "C"),
        ("C:maj", "C"),
        ("C#:min7", "C♯m7"),
        ("Bb:hdim7", "B♭m7♭5"),
        ("F:sus4", "Fsus4"),
        ("A:minmaj7", "Am(maj7)"),
        ("C:maj7/3", "Cmaj7/E"),
        ("Bb:min/5", "B♭m/F"),
        ("F#:maj7/#5", "F♯maj7/C♯♯"),
        ("C:min/b3", "Cm/E♭"),
        ("N", "N"),
        ("X", "X"),
    ],
)
def test_compact_notation_preserves_quality_and_spelled_bass(label, expected):
    assert _chord_presentation(label)[0] == expected


def test_all_170_classes_remain_distinct_and_root_colors_are_stable():
    roots = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
    symbols = {
        _chord_presentation(f"{root}:{quality}")[0] for root in roots for quality in QUALITIES
    }
    assert len(symbols | {"N", "X"}) == 170
    for root in roots:
        assert len({_chord_presentation(f"{root}:{quality}")[1] for quality in QUALITIES}) == 1
    assert _chord_presentation("C#:maj7")[1] == _chord_presentation("Db:min")[1]


def test_playhead_repaints_current_chord_when_follow_is_off(lane):
    class PaintCounter(QObject):
        count = 0

        def eventFilter(self, watched, event):  # noqa: N802
            if event.type() == QEvent.Type.Paint:
                self.count += 1
            return False

    recorder = PaintCounter(lane)
    lane.installEventFilter(recorder)
    source = lane._source_chords
    scrollbar = lane.scroll.horizontalScrollBar().value()
    lane.roll.set_position(0.2)
    QTest.qWait(5)
    count = recorder.count
    before = lane.grab().toImage()
    lane.roll.set_position(1.2)
    QTest.qWait(5)
    assert recorder.count > count
    assert lane.current_chord().label == "D:min7"
    assert lane.scroll.horizontalScrollBar().value() == scrollbar
    after = lane.grab().toImage()
    # Compare fill pixels away from text and playhead; the old block must dim.
    point = lane.chord_rect(lane.chords[0]).bottomRight() - QPointF(10, 5)
    ratio = lane.devicePixelRatioF()
    pixel = QPoint(round(point.x() * ratio), round(point.y() * ratio))
    assert before.pixelColor(pixel).lightness() > after.pixelColor(pixel).lightness()
    assert lane._source_chords == source


def test_current_chord_uses_half_open_intervals_and_respects_lead_in(lane):
    for position, label in [(0, "C:maj7"), (1, "D:min7"), (2, "D:min7"), (3, "N"), (4, "X")]:
        lane.roll.set_position(position)
        assert lane.current_chord().label == label
    lane.roll.set_position(5)
    assert lane.current_chord() is None
    lane.source_offset = 1.5
    lane.refresh()
    lane.roll.set_position(1)
    assert lane.current_chord() is None
    lane.roll.set_position(1.5)
    assert lane.current_chord().label == "C:maj7"


@pytest.mark.parametrize("language", tuple(Translator.AVAILABLE_LANGUAGES))
def test_full_hover_details_and_non_audible_regions_in_every_locale(lane, language):
    set_language(language)
    lane.update_translations()
    emitted = QSignalSpy(lane.audition_requested)
    for chord in lane.chords:
        point = lane.chord_rect(chord).center().toPoint()
        QTest.mouseMove(lane, point)
        assert chord.label in lane.toolTip()
        assert f"{chord.start:.2f}" in lane.toolTip()
        assert f"{chord.end:.2f}" in lane.toolTip()
        assert "muscriptor_result." not in lane.toolTip()
        QTest.mouseClick(lane, Qt.MouseButton.LeftButton, pos=point)
        if not chord.pitches:
            key = "chord_none" if chord.label == "N" else "chord_unknown"
            assert t("muscriptor_result." + key) in lane.toolTip()
    assert [item[0] for item in emitted] == [chord for chord in lane.chords if chord.pitches]


def test_subpixel_chords_never_cover_the_next_interval(lane):
    tiny = MidiChord(0, 0.002, "C", (60, 64, 67))
    rect = lane.chord_rect(tiny)
    assert 0 < rect.width() < 1
    assert rect.right() <= lane.x_for_time(tiny.end)
    lane._source_chords = (tiny, MidiChord(0.002, 3, "D", (62, 66, 69)))
    lane.refresh()
    lane.grab()  # Exercise the subpixel paint path on a real widget.
    rect = lane.chord_rect(tiny)
    assert lane.chord_at(QPointF(lane.x_for_time(0.003), rect.center().y())).label == "D"


def test_every_short_chord_has_a_full_nonoverlapping_clickable_label(lane):
    from PyQt6.QtGui import QFont, QFontMetrics

    labels = ("C#:min7", "F:sus4", "Bb:hdim7", "D#:maj7", "G:min7")
    lane._source_chords = tuple(
        MidiChord(index * 0.05, (index + 1) * 0.05, label, chord_pitches(label))
        for index, label in enumerate(labels)
    )
    lane.refresh()
    font = QFont(lane.font())
    font.setWeight(QFont.Weight.DemiBold)
    metrics = QFontMetrics(font)
    assert {chord.label for chord, _ in lane._callouts} == set(labels)
    for index, (chord, rect) in enumerate(lane._callouts):
        assert rect.width() - 12 >= metrics.horizontalAdvance(_chord_presentation(chord.label)[0])
        assert lane.chord_at(rect.center()) == chord
        assert rect.bottom() < lane.chord_rect(chord).top()
        assert all(not rect.intersects(other) for _, other in lane._callouts[index + 1 :])


def test_visible_range_keeps_partially_scrolled_chords(lane):
    lane.roll.set_pixels_per_second(300)
    lane.scroll.horizontalScrollBar().setValue(550)
    lane.roll.set_render_offset(0.75)
    visible = lane._visible_chords()
    assert visible[0].start == 1
    assert lane.chord_rect(visible[0]).left() < 0 < lane.chord_rect(visible[0]).right()
    assert all(lane.chord_rect(chord).left() <= lane.width() for chord in visible)


@pytest.mark.parametrize("language", tuple(Translator.AVAILABLE_LANGUAGES))
def test_midi_hover_and_playback_feedback_preserve_notes_and_tile_cache(lane, language):
    from src.core.muscriptor_result_assets import MuscriptorRollNote

    set_language(language)
    notes = (
        MuscriptorRollNote("gm:000", 84, 91, 0, 3, 0, False, 0, 0),
        MuscriptorRollNote("gm:032", 86, 72, 1, 2, 32, False, 1, 1),
    )
    roll = lane.roll
    roll.set_notes(notes, duration=6)
    tile = roll._static_tile(0)
    changed = QSignalSpy(roll.edit_committed)
    roll.set_position(1.2)
    assert roll._active_note_indices() == {0, 1}
    roll.set_position(2)
    assert roll._active_note_indices() == {0}
    roll.set_position(3)
    assert not roll._active_note_indices()
    point = roll._note_rect(notes[0]).center().toPoint()
    QTest.mouseMove(roll, point)
    QTest.qWait(5)
    assert roll._hovered_index == 0
    assert "C6" in roll.toolTip() and "91" in roll.toolTip()
    assert "0.00–3.00" in roll.toolTip()
    assert "muscriptor_result." not in roll.toolTip()
    assert roll._static_tile(0).cacheKey() == tile.cacheKey()
    assert roll.notes == notes and len(changed) == 0
    roll.set_instrument_muted("gm:032", True)
    roll.set_position(1.2)
    assert roll._active_note_indices() == {0}
    QTest.mouseMove(roll, QPoint(10, 10))
    assert roll._hovered_index is None and not roll.toolTip()
