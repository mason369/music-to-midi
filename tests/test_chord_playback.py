"""Audible chord gain, native timing, cancellation and persistent bilingual controls."""

import json
import os
import string
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf
from PyQt6.QtCore import QObject, QSettings, Qt, pyqtSignal
from PyQt6.QtTest import QTest

from src.core import chord_audio
from src.core.chord_audio import CHORD_SAMPLE_RATE as SR
from src.core.midi_chords import MidiChord
from src.gui import chord_monitor
from src.gui.synchronized_pcm_player import _AlignedPcmMixDevice
from src.gui.widgets import chord_playback_controls
from src.gui.widgets.chord_playback_controls import ChordPlaybackControls
from src.gui.widgets.midi_chord_lane import ChordAuditioner, _ChordRenderWorker
from src.gui.widgets.muscriptor_result import MuscriptorResultWidget
from src.i18n.translator import Translator, set_language, t

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CHORDS = (
    MidiChord(0.2, 6.2, "C:maj", (60, 64, 67)),
    MidiChord(6.2, 7, "N", ()),
    MidiChord(7, 8, "G:maj", (55, 59, 62)),
)


@pytest.fixture
def preferences(tmp_path, monkeypatch):
    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(
        chord_playback_controls,
        "chord_settings",
        lambda: QSettings(str(path), QSettings.Format.IniFormat),
    )
    return path


def decoded(payload):
    return np.frombuffer(payload, dtype="<i2").reshape(-1, 2) / 32767.0


def test_native_chord_midi_preserves_long_intervals_and_lead_in():
    midi, intervals = chord_audio.chord_midi(CHORDS, 9 * SR, offset=0.5)
    assert intervals == [(round(0.7 * SR), round(6.7 * SR)), (round(7.5 * SR), round(8.5 * SR))]
    absolute, on, off = 0, [], []
    for message in midi.tracks[0]:
        absolute += message.time
        if message.type == "note_on":
            on.append((absolute, message.note, message.velocity))
        if message.type == "note_off":
            off.append((absolute, message.note))
    assert [item[0] for item in on] == [round(0.7 * SR)] * 3 + [round(7.5 * SR)] * 3
    assert [item[0] for item in off] == [round(6.7 * SR)] * 3 + [round(8.5 * SR)] * 3
    assert {item[2] for item in on} == {82}
    assert midi.length == pytest.approx(9)


def test_native_boundary_releases_shared_pitches_before_the_next_chord():
    chords = (MidiChord(0, 1, "C", (60, 64)), MidiChord(1, 2, "Am", (60, 69)))
    midi, _ = chord_audio.chord_midi(chords, SR * 2)
    tick, at_boundary = 0, []
    for message in midi.tracks[0]:
        tick += message.time
        if tick == SR:
            at_boundary.append(message.type)
    assert at_boundary == ["note_off", "note_off", "note_on", "note_on"]


@pytest.mark.parametrize("offset", [float("nan"), float("inf")])
def test_reject_invalid_timing(offset):
    with pytest.raises(ValueError):
        chord_audio.chord_midi(CHORDS, SR, offset)


def test_overlap_is_rejected_and_outside_audio_is_clipped():
    with pytest.raises(ValueError, match="non-overlapping"):
        chord_audio.chord_midi(CHORDS + (MidiChord(7.5, 9, "C", (60,)),), 10 * SR)
    _, intervals = chord_audio.chord_midi(CHORDS, SR, offset=-0.5)
    assert intervals == [(0, SR)]


def test_render_calibrates_stereo_and_keeps_unknown_intervals_silent(tmp_path, monkeypatch):
    def synth(source, destination, *_args, **_kwargs):
        length = round(mido.MidiFile(source).length * SR) + SR
        waveform = 0.015 * np.sin(2 * np.pi * 220 * np.arange(length) / SR)
        sf.write(destination, np.column_stack((waveform, waveform * 0.5)), SR, subtype="FLOAT")

    monkeypatch.setattr(chord_audio, "render_midi_audio_export", synth)
    pcm = chord_audio.render_chord_pcm(CHORDS, 9 * SR, tmp_path, offset=0.5)
    assert pcm.shape == (9 * SR, 2)
    assert not pcm.flags.writeable
    assert not np.any(pcm[: round(0.7 * SR)])
    assert not np.any(pcm[round(6.7 * SR) : round(7.5 * SR)])
    assert not np.any(pcm[round(8.5 * SR) :])
    assert np.max(np.abs(pcm)) > 0.1
    assert np.max(np.abs(pcm * 2)) <= 0.95 + 1e-6
    np.testing.assert_allclose(pcm[:, 0], pcm[:, 1] * 2, atol=1e-6)
    assert list(tmp_path.iterdir()) == []


def test_silent_soundfont_fails_and_cancellation_cleans_up(tmp_path, monkeypatch):
    def synth(_source, destination, *_args, **_kwargs):
        sf.write(destination, np.zeros((9 * SR, 2)), SR)

    monkeypatch.setattr(chord_audio, "render_midi_audio_export", synth)
    with pytest.raises(RuntimeError, match="silent"):
        chord_audio.render_chord_pcm(CHORDS, 9 * SR, tmp_path)
    with pytest.raises(InterruptedError):
        chord_audio.render_chord_pcm(CHORDS, 9 * SR, tmp_path, cancel_check=lambda: True)
    assert list(tmp_path.iterdir()) == []


def test_unknown_chords_never_call_the_synthesizer(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("N/X must not synthesize notes")

    monkeypatch.setattr(chord_audio, "render_midi_audio_export", forbidden)
    pcm = chord_audio.render_chord_pcm((MidiChord(0, 1, "X", ()),), SR, tmp_path)
    assert not pcm.any()


@pytest.mark.parametrize("volume", [0, 50, 100, 200])
def test_chord_gain_is_independent_of_original_midi_mix(volume):
    frames = SR
    device = _AlignedPcmMixDevice(np.full(frames, 0.1), np.full(frames, 0.2))
    device.set_chord_bus(np.full((frames, 2), 0.1))
    device.set_chord_volume(volume)
    device.set_mix_state(mix=0, stereo=False, muted=frozenset())
    device.render_frames_for_test(1000)
    pcm = decoded(device.render_frames_for_test(256))
    np.testing.assert_allclose(pcm, 0.1 + 0.1 * volume / 100, atol=1 / 32767)
    device.set_mix_state(mix=1, stereo=False, muted=frozenset())
    pcm = decoded(device.render_frames_for_test(256))
    np.testing.assert_allclose(pcm, 0.2 + 0.1 * volume / 100, atol=1 / 32767)


@pytest.mark.parametrize("rate", [0.5, 1, 1.75, 3])
def test_chord_seek_and_speed_share_the_exact_sample_axis(rate):
    frames = SR
    bus = np.arange(frames) / frames * 0.2
    device = _AlignedPcmMixDevice(np.zeros(frames), np.zeros(frames))
    device.set_chord_bus(np.column_stack((bus, -bus)))
    device.set_chord_volume(100)
    device.render_frames_for_test(1000)
    device.set_position_frames(2000.5)
    device.set_playback_rate(rate)
    pcm = decoded(device.render_frames_for_test(128))
    expected = (2000.5 + rate * np.arange(128)) / frames * 0.2
    np.testing.assert_allclose(pcm[:, 0], expected, atol=1 / 32767)
    np.testing.assert_allclose(pcm[:, 1], -expected, atol=1 / 32767)
    assert device.position_frames == pytest.approx(2000.5 + rate * 128)


def test_gain_ramps_and_muting_preserve_the_song_bit_for_bit():
    original = np.full(SR, 0.15)
    device = _AlignedPcmMixDevice(original, original)
    device.set_chord_bus(np.full((SR, 2), 0.1))
    device.set_chord_volume(200)
    attack = decoded(device.render_frames_for_test(1000))[:, 0]
    assert np.max(np.abs(np.diff(attack))) < 0.001
    device.set_chord_volume(0)
    release = decoded(device.render_frames_for_test(1000))[:, 0]
    assert np.max(np.abs(np.diff(release))) < 0.001
    assert release[-1] == pytest.approx(round(0.15 * 32767) / 32767)
    expected = _AlignedPcmMixDevice(original, original).render_frames_for_test(400)
    assert device.render_frames_for_test(400) == expected


def test_peak_protection_is_stereo_linked_and_independent_of_callback_block_size():
    def player():
        signal = 0.85 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
        device = _AlignedPcmMixDevice(signal, signal * 0.5)
        device.set_mix_state(mix=0.5, stereo=True, muted=frozenset())
        device.set_chord_bus(np.column_stack((signal * 0.5, signal * 0.25)))
        device.set_chord_volume(200)
        return device

    whole = decoded(player().render_frames_for_test(8000))
    partial = player()
    chunks = decoded(b"".join(partial.render_frames_for_test(n) for n in (7, 123, 901, 2048, 4921)))
    np.testing.assert_allclose(whole, chunks, atol=1 / 32767)
    assert np.max(np.abs(whole)) <= 0.98002
    np.testing.assert_allclose(whole[:, 0], whole[:, 1] * 2, atol=2 / 32767)


def test_enabling_peak_protection_does_not_step_the_loud_original():
    signal = np.full(SR, 0.8, dtype="float32")
    device = _AlignedPcmMixDevice(signal, signal)
    device.set_chord_bus(np.full((SR, 2), 0.4, dtype="float32"))
    before = decoded(device.render_frames_for_test(1000))
    device.set_chord_volume(200)
    enabled = decoded(device.render_frames_for_test(2000))
    joined = np.concatenate((before[-1:], enabled))[:, 0]
    assert np.max(np.abs(np.diff(joined))) < 0.003
    assert np.max(np.abs(enabled)) <= 0.98002
    device.set_chord_volume(0)
    released = decoded(device.render_frames_for_test(3000))
    assert np.max(np.abs(np.diff(np.concatenate((enabled[-1:], released))[:, 0]))) < 0.003
    expected = _AlignedPcmMixDevice(signal, signal).render_frames_for_test(1000)
    assert device.render_frames_for_test(1000) == expected


@pytest.mark.parametrize("volume", [-1, 201, float("nan"), float("inf")])
def test_invalid_gain_is_rejected(volume):
    player = _AlignedPcmMixDevice(np.zeros(100), np.zeros(100))
    with pytest.raises(ValueError):
        player.set_chord_volume(volume)


@pytest.mark.parametrize("language", tuple(Translator.AVAILABLE_LANGUAGES))
@pytest.mark.parametrize("width", [320, 640])
def test_real_controls_wrap_translate_and_persist(qtbot, preferences, language, width):
    set_language(language)
    controls = ChordPlaybackControls()
    qtbot.addWidget(controls)
    assert not controls.follow.isChecked()
    assert controls.value.value() == 100
    controls.resize(width, 180)
    controls.show()
    QTest.qWait(20)
    QTest.mouseClick(controls.follow, Qt.MouseButton.LeftButton)
    controls.value.setFocus()
    controls.value.selectAll()
    QTest.keyClicks(controls.value, "175")
    QTest.keyClick(controls.value, Qt.Key.Key_Return)
    assert controls.slider.value() == 175
    assert controls.follow.isChecked()
    assert controls.follow.text() == t("muscriptor_result.chord_follow")
    controls.set_state("failed", "Broken SoundFont")
    assert controls.retry.isVisible()
    assert controls.status.toolTip() == "Broken SoundFont"
    for child in (controls.follow, controls.volume_label, controls.slider, controls.value):
        left = child.mapTo(controls, child.rect().topLeft()).x()
        assert 0 <= left < controls.width()
        assert left + child.width() <= controls.width()
    controls.save()
    restored = ChordPlaybackControls()
    qtbot.addWidget(restored)
    assert restored.follow.isChecked()
    assert restored.value.value() == 175


class FakeWorker(QObject):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()
    instances = []

    def __init__(self, *_args):
        super().__init__()
        self.interrupted = False
        self.instances.append(self)

    def start(self):
        pass

    def requestInterruption(self):
        self.interrupted = True

    def isInterruptionRequested(self):
        return self.interrupted

    def wait(self):
        return True


class FakePlayer:
    duration_seconds = 9

    def __init__(self):
        self.bus, self.volume = None, 0

    def set_chord_bus(self, samples):
        self.bus = samples

    def set_chord_volume(self, volume):
        self.volume = volume


def test_late_render_cannot_play_after_disable_and_new_selection(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(chord_monitor, "_ChordMonitorWorker", FakeWorker)
    player = FakePlayer()
    monitor = chord_monitor.ChordMonitor(player, tmp_path)
    monitor.configure(CHORDS, 0, None)
    assert monitor.worker is None
    monitor.set_controls(True, 150)
    old = monitor.worker
    monitor.set_controls(False, 150)
    old.ready.emit(np.ones((9 * SR, 2)))
    assert player.bus is None and player.volume == 0
    monitor.set_controls(True, 150)
    old.finished.emit()
    new = monitor.worker
    assert new is not old
    samples = np.zeros((9 * SR, 2))
    new.ready.emit(samples)
    assert player.bus is samples and player.volume == 150
    new.finished.emit()
    monitor.configure(CHORDS, 0.5, None)
    assert player.bus is None and player.volume == 0
    current = monitor.worker
    monitor.shutdown()
    current.ready.emit(samples)
    assert player.bus is None and player.volume == 0


def test_failed_render_requires_explicit_retry(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(chord_monitor, "_ChordMonitorWorker", FakeWorker)
    monitor = chord_monitor.ChordMonitor(FakePlayer(), tmp_path)
    states = []
    monitor.state_changed.connect(lambda *values: states.append(values))
    monitor.configure(CHORDS, 0, None)
    monitor.set_controls(True, 100)
    old = monitor.worker
    old.failed.emit("SoundFont missing")
    old.finished.emit()
    assert states[-1] == ("failed", "SoundFont missing")
    monitor.set_controls(True, 125)
    assert monitor.worker is None
    monitor.retry()
    assert monitor.worker is not None
    assert states[-1] == ("loading", "")
    monitor.shutdown()


def test_retry_before_failed_thread_finishes_starts_the_new_request(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(chord_monitor, "_ChordMonitorWorker", FakeWorker)
    monitor = chord_monitor.ChordMonitor(FakePlayer(), tmp_path)
    monitor.configure(CHORDS, 0, None)
    monitor.set_controls(True, 100)
    old = monitor.worker
    old.failed.emit("render failed")
    monitor.retry()
    assert old.isInterruptionRequested()
    old.finished.emit()
    assert monitor.worker is not None and monitor.worker is not old
    monitor.shutdown()


def test_no_chords_does_not_create_a_useless_worker(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(chord_monitor, "_ChordMonitorWorker", FakeWorker)
    monitor = chord_monitor.ChordMonitor(FakePlayer(), tmp_path)
    states = []
    monitor.state_changed.connect(lambda *values: states.append(values))
    monitor.configure((MidiChord(0, 9, "N", ()),), 0, None)
    monitor.set_controls(True, 100)
    assert monitor.worker is None
    assert states[-1] == ("empty", "")
    monitor.shutdown()


def test_widget_mute_does_not_pause_song_and_soundfont_change_cancels_audition(
    qtbot, preferences, tmp_path, monkeypatch
):
    widget = MuscriptorResultWidget(str(tmp_path / "source.wav"), [], muscriptor_groups=False)
    qtbot.addWidget(widget)
    calls = []
    monkeypatch.setattr(widget, "pause", lambda: calls.append("pause"))
    monkeypatch.setattr(widget.chord_auditioner, "play", lambda *args: calls.append(args))
    monkeypatch.setattr(widget.chord_auditioner, "stop", lambda: calls.append("stop"))
    monkeypatch.setattr(widget, "_queue_editor_audio_render", lambda *_args, **_kwargs: None)
    try:
        widget.chord_controls.value.setValue(0)
        calls.clear()
        widget._audition_chord(CHORDS[0])
        assert calls == []
        widget.chord_controls.value.setValue(150)
        widget._audition_chord(CHORDS[0])
        assert calls[0] == "pause"
        assert calls[1][1] == 4.0
        widget._apply_soundfont(None)
        assert calls[-1] == "stop"
    finally:
        widget.shutdown()


def test_audition_volume_actually_uses_the_boosted_render(qtbot, tmp_path, monkeypatch):
    base = np.full((SR, 2), 0.1, dtype="float32")
    monkeypatch.setattr("src.gui.widgets.midi_chord_lane.render_chord_pcm", lambda *_a, **_kw: base)
    destination = tmp_path / "audition.wav"
    worker = _ChordRenderWorker(CHORDS[0], 1, destination, None, None)
    errors = []
    worker.failed.connect(errors.append)
    worker.run()
    assert errors == []
    pcm, rate = sf.read(destination)
    assert rate == SR
    np.testing.assert_allclose(pcm, 0.2, atol=1e-6)
    audition = ChordAuditioner(tmp_path)
    try:
        audition.set_volume(100)
        QTest.qWait(50)
        assert audition.output.volume() == pytest.approx(0.5)
        audition.set_volume(200)
        QTest.qWait(50)
        assert audition.output.volume() == pytest.approx(1.0)
        audition.set_volume(0)
        QTest.qWait(50)
        assert audition.output.volume() == pytest.approx(0.0)
    finally:
        audition.shutdown()


def test_every_supported_locale_has_matching_chord_keys_and_placeholders():
    catalogs = {
        code: json.loads((ROOT / "src/i18n" / f"{code}.json").read_text(encoding="utf-8"))[
            "muscriptor_result"
        ]
        for code in Translator.AVAILABLE_LANGUAGES
    }
    reference = catalogs[Translator.DEFAULT_LANGUAGE]
    keys = {key for key in reference if key.startswith("chord_")}
    formatter = string.Formatter()

    def fields(text):
        return {field for _, field, _, _ in formatter.parse(text) if field}

    for code, catalog in catalogs.items():
        assert {key for key in catalog if key.startswith("chord_")} == keys
        for key in keys:
            assert catalog[key].strip(), (code, key)
            assert fields(catalog[key]) == fields(reference[key]), (code, key)


@pytest.mark.skipif(
    os.environ.get("MUSIC_TO_MIDI_RUN_AUDIO_DEVICE_TESTS") != "1",
    reason="requires the real FluidSynth/SoundFont runtime and audio output device",
)
@pytest.mark.parametrize("language", tuple(Translator.AVAILABLE_LANGUAGES))
def test_real_soundfont_chords_follow_windows_transport_and_boost(
    qtbot, tmp_path, preferences, language
):
    from src.core.muscriptor_result_assets import read_midi_roll_notes
    from src.utils.fluidsynth_runtime import get_fluidsynth_executable
    from src.utils.muscriptor_soundfont_downloader import validate_muscriptor_soundfont

    assert get_fluidsynth_executable().is_file()
    assert validate_muscriptor_soundfont().is_file()
    set_language(language)
    original = tmp_path / "original.wav"
    midi = tmp_path / "midi.wav"
    time = np.arange(9 * SR) / SR
    sf.write(original, 0.015 * np.sin(2 * np.pi * 110 * time), SR)
    sf.write(midi, np.zeros(9 * SR), SR)
    before = (original.read_bytes(), midi.read_bytes())
    chord_file = tmp_path / "source.mid"
    chord_audio.chord_midi(CHORDS, 9 * SR)[0].save(chord_file)
    widget = MuscriptorResultWidget(str(original), [], muscriptor_groups=False)
    qtbot.addWidget(widget)
    widget.chord_lane._source_chords = CHORDS
    widget.set_bpm_context(120, 120)
    widget.roll.set_notes(read_midi_roll_notes(chord_file), duration=9)
    widget.chord_lane.refresh()
    widget._configure_synchronized_playback(
        original_wav=original, midi_wav=midi, instrument_wavs={}, stereo_available=True
    )
    widget.playback_slider.setRange(0, 9000)
    widget.play_button.setEnabled(True)
    widget.resize(1100, 760)
    widget.show()
    errors = []
    widget._playback_engine.error_occurred.connect(errors.append)
    try:
        QTest.mouseClick(widget.chord_controls.follow, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: widget.chord_monitor.samples is not None, timeout=60_000)
        bus = widget.chord_monitor.samples
        assert np.max(np.abs(bus)) > 0.05
        assert np.max(np.abs(bus)) <= 0.47501
        assert not bus[round(6.2 * SR) : 7 * SR].any()
        assert widget.grab().save(str(tmp_path / f"chords-{language}.png"))
        assert widget._playback_engine.output_stream_count == 1
        QTest.mouseClick(widget.play_button, Qt.MouseButton.LeftButton)
        QTest.qWait(100)
        assert widget._playing
        for target, bpm in ((2, 90), (7.1, 180), (0.1, 120)):
            widget.seek(target)
            widget.bpm_spin.setValue(bpm)
            QTest.qWait(60)
            assert widget._playback_engine.is_playing
            assert widget._playback_engine.output_stream_count == 1
            assert widget._playback_engine.position_seconds >= target
        widget.chord_controls.value.setValue(200)
        assert widget._playback_engine._source._chord_target_gain == 2
        # Device callbacks consume the gain ramp; 80 ms may precede the next
        # Windows buffer refill after seek. Observe actual consumption instead.
        qtbot.waitUntil(lambda: widget._playback_engine._source._chord_gain == 2, timeout=1000)
        assert widget._playback_engine._source._chord_gain == 2
        assert widget.grab().save(str(tmp_path / f"chords-playing-{language}.png"))
        widget.chord_controls.value.setValue(0)
        assert widget._playback_engine._source._chord_target_gain == 0
        qtbot.waitUntil(lambda: widget._playback_engine._source._chord_gain == 0, timeout=1000)
        assert widget._playback_engine._source._chord_gain == 0
        assert widget._playing
        widget.chord_controls.value.setValue(150)
        widget._audition_chord(CHORDS[0])
        assert not widget._playing
        qtbot.waitUntil(lambda: widget.chord_auditioner._cache_path is not None, timeout=60_000)
        QTest.qWait(50)
        assert widget.chord_auditioner.output.volume() == pytest.approx(0.75)
        widget.stop()
        assert widget._playback_engine.position_seconds == 0
        assert not errors
        assert original.is_file() and midi.is_file()
        assert before == (original.read_bytes(), midi.read_bytes())
    finally:
        widget.shutdown()
