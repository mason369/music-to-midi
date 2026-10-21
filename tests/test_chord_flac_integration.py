"""Opt-in real CUDA, Qt and SoundFont acceptance for the FLAC failure/retry path."""

import os

import numpy as np
import pytest
import soundfile as sf

from src.core import chord_analysis, chord_audio
from src.core.midi_chords import read_chord_record
from src.i18n.translator import Translator, get_translator, set_language, t

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.skipif(
    os.environ.get("MUSIC_TO_MIDI_RUN_CHORD_MODEL_TESTS") != "1",
    reason="requires the pinned ChordMini CUDA runtime, FluidSynth and audio output device",
)


@pytest.fixture(scope="module")
def real_reference(tmp_path_factory):
    folder = tmp_path_factory.mktemp("real-flac-chords")
    rate = 48000
    timeline = np.arange(4 * rate) / rate
    parts = []
    for pitches in ((261.625565, 329.627557, 391.995436), (195.997718, 246.941651, 293.664768)):
        tone = sum(
            np.sin(2 * np.pi * pitch * timeline) + 0.3 * np.sin(4 * np.pi * pitch * timeline)
            for pitch in pitches
        )
        tone *= 0.055 * np.minimum(1, timeline / 0.02) * np.exp(-timeline / 3)
        parts.append(tone)
    mono = np.concatenate(parts)
    samples = np.column_stack((mono, mono * 0.91))
    original = folder / "reference.wav"
    source = folder / "tagged.flac"
    sf.write(original, samples, rate, subtype="PCM_24")
    sf.write(source, samples, rate, subtype="PCM_24")
    source_bytes = source.read_bytes() + b"TAG" + b"test song".ljust(125, b"\0")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(chord_analysis, "get_runtime_data_dir", lambda: folder / "runtime")
        record = chord_analysis.analyze_audio_chords(original)
    assert any(chord.pitches for chord in read_chord_record(record))
    return original, source_bytes, record


@pytest.mark.parametrize("locale", tuple(Translator.AVAILABLE_LANGUAGES))
def test_real_model_retry_recovers_tagged_flac_and_drives_audible_chord_lane(
    qtbot, tmp_path, monkeypatch, real_reference, locale
):
    from PyQt6.QtCore import QPoint, QSettings, Qt
    from PyQt6.QtTest import QTest
    from scipy.signal import resample_poly

    from src.core.muscriptor_result_assets import read_midi_roll_notes
    from src.gui.widgets import chord_playback_controls
    from src.gui.widgets.muscriptor_result import MuscriptorResultWidget

    previous = get_translator().get_language()
    set_language(locale)
    original, source_bytes, expected = real_reference
    source = tmp_path / "待识别原音.flac"
    damaged = bytearray(source_bytes)
    damaged[26] ^= 0x40
    source.write_bytes(damaged)
    monkeypatch.setattr(chord_analysis, "get_runtime_data_dir", lambda: tmp_path / "runtime")
    monkeypatch.setattr(
        chord_playback_controls,
        "chord_settings",
        lambda: QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat),
    )
    frames = 8 * chord_audio.CHORD_SAMPLE_RATE
    playback = tmp_path / "playback-original.wav"
    reference_samples, _ = sf.read(original, always_2d=True)
    sf.write(
        playback,
        resample_poly(reference_samples.mean(axis=1), 147, 160),
        chord_audio.CHORD_SAMPLE_RATE,
    )
    silent = tmp_path / "midi.wav"
    sf.write(silent, np.zeros(frames), chord_audio.CHORD_SAMPLE_RATE)
    midi = tmp_path / "recognized-chords.mid"
    chord_audio.chord_midi(read_chord_record(expected), frames)[0].save(midi)
    widget = MuscriptorResultWidget(str(source), [], muscriptor_groups=False)
    qtbot.addWidget(widget)
    widget.set_bpm_context(120, 120)
    widget.roll.set_notes(read_midi_roll_notes(midi), duration=8)
    widget._configure_synchronized_playback(
        original_wav=playback, midi_wav=silent, instrument_wavs={}, stereo_available=True
    )
    widget.playback_slider.setRange(0, 8000)
    widget.play_button.setEnabled(True)
    widget.resize(1100, 800)
    widget.show()
    errors = []
    widget._playback_engine.error_occurred.connect(errors.append)
    try:
        widget.chord_lane.analyze_audio(source)
        qtbot.waitUntil(lambda: widget.chord_lane.worker is None, timeout=30000)
        assert widget.chord_lane.state == "chord_analysis_failed"
        assert widget.chord_lane.error == t("muscriptor_result.chord_flac_checksum_mismatch")
        assert widget.chord_controls.status.text() == t(
            "muscriptor_result.chord_monitor_analysis_failed"
        )
        assert widget.grab().save(str(tmp_path / f"flac-failed-{locale}.png"))
        assert not list((tmp_path / "runtime/chords").rglob("*.json"))
        source.write_bytes(source_bytes)
        QTest.mouseClick(widget.chord_lane, Qt.MouseButton.LeftButton, pos=QPoint(30, 15))
        qtbot.waitUntil(lambda: widget.chord_lane.record is not None, timeout=240000)
        assert not widget.chord_lane.error
        assert widget.chord_lane.record["segments"] == expected["segments"]
        assert widget.chord_lane.record["input_audio"] == dict(
            sample_rate=48000, channels=2, frames=384000, encoding="pcm_s32le"
        )
        assert widget.chord_lane.chords
        assert widget.chord_lane.state != "chord_analysis_failed"
        QTest.mouseClick(widget.chord_controls.follow, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: widget.chord_monitor.samples is not None, timeout=60000)
        assert np.max(np.abs(widget.chord_monitor.samples)) > 0.01
        widget.chord_controls.value.setValue(200)
        QTest.mouseClick(widget.play_button, Qt.MouseButton.LeftButton)
        QTest.qWait(150)
        assert widget._playing
        assert widget._playback_engine._source._chord_gain == 2
        assert widget._playback_engine.output_stream_count == 1
        widget.pause()
        widget.chord_lane._relayout()
        visible = next(chord for chord in widget.chord_lane.chords if chord.pitches)
        QTest.mouseClick(
            widget.chord_lane,
            Qt.MouseButton.LeftButton,
            pos=widget.chord_lane.chord_rect(visible).center().toPoint(),
        )
        qtbot.waitUntil(lambda: widget.chord_auditioner._cache_path is not None, timeout=60000)
        assert widget.chord_auditioner.output.volume() == pytest.approx(1)
        assert widget.chord_controls.status.text() == t("muscriptor_result.chord_monitor_ready")
        assert widget.grab().save(str(tmp_path / f"flac-recovered-{locale}.png"))
        assert source.read_bytes() == source_bytes
        assert not errors
    finally:
        widget.shutdown()
        set_language(previous)
