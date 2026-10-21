"""Timing regressions use real SMFs, not textual implementation assertions."""

import base64
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

from src.core.midi_editor import export_edited_midi
from src.core.midi_export_options import MidiExportOptions, apply_midi_export_options
from src.core.midi_quantization import quantize_midi_notes
from src.core.midi_tempo import read_muscriptor_bar_offset_seconds, track_tick_to_seconds
from src.core.midi_timing import (
    MidiTimeline,
    quantize_editor_notes,
    reference_timelines,
    write_project_tempo_map,
)
from src.core.muscriptor_result_assets import (
    _write_bar_aligned_reference_audio,
    read_midi_roll_notes,
)


@pytest.fixture
def variable_midi(tmp_path):
    source = tmp_path / "variable.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500000),
                mido.MetaMessage("time_signature", numerator=4, denominator=4),
                mido.MetaMessage("set_tempo", tempo=1000000, time=1920),
                mido.MetaMessage("end_of_track", time=2880),
            ]
        )
    )
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("program_change", program=0),
                mido.Message("control_change", control=64, value=127, time=2300),
                mido.Message("sysex", data=(1, 2, 3), time=20),
                mido.Message("note_on", note=60, velocity=97, time=80),
                mido.Message("note_off", note=60, time=180),
                mido.Message("note_on", note=64, velocity=88, time=60),
                mido.Message("note_off", note=64, time=240),
                mido.Message("control_change", control=64, value=0, time=60),
            ]
        )
    )
    midi.save(source)
    return source


def event_seconds(path):
    midi = mido.MidiFile(str(path))
    result = []
    for index, track in enumerate(midi.tracks):
        seconds_at, _ = track_tick_to_seconds(midi, index)
        tick = 0
        for message in track:
            tick += message.time
            if not message.is_meta:
                result.append((index, message.copy(time=0).dict(), seconds_at(tick)))
    return result


def assert_same_performance(left, right, rate=1):
    before, after = event_seconds(left), event_seconds(right)
    assert len(before) == len(after)
    for old, new in zip(before, after):
        assert old[:2] == new[:2]
        assert old[2] / rate == pytest.approx(new[2], abs=0.003)


def test_timeline_is_reversible_across_tempo_boundaries(variable_midi):
    timeline = MidiTimeline.from_midi(mido.MidiFile(variable_midi))
    for time in (0, 0.1, 1.999, 2, 2.001, 3.14, 10):
        assert timeline.seconds_at(timeline.beats_at(time)) == pytest.approx(time)


def test_reference_preview_and_project_speed_keep_the_map(variable_midi, tmp_path):
    normal, fast = tmp_path / "normal.mid", tmp_path / "fast.mid"
    write_project_tempo_map(variable_midi, normal, 120, 120)
    write_project_tempo_map(variable_midi, fast, 120, 240)
    assert_same_performance(variable_midi, normal)
    assert_same_performance(variable_midi, fast, rate=2)
    assert {
        m.tempo for tr in mido.MidiFile(normal).tracks for m in tr if m.type == "set_tempo"
    } == {500000, 1000000}


def test_edited_midi_preserves_variable_tempo_and_controller_ticks(variable_midi, tmp_path):
    notes = read_midi_roll_notes(variable_midi)
    output = export_edited_midi(
        variable_midi, tmp_path / "edited.mid", notes, reference_bpm=120, target_bpm=120
    )
    assert_same_performance(variable_midi, output)
    output = export_edited_midi(
        variable_midi, tmp_path / "fast-edit.mid", notes, reference_bpm=120, target_bpm=240
    )
    assert_same_performance(variable_midi, output, rate=2)


@pytest.mark.parametrize("strength", [0, 0.5, 1])
@pytest.mark.parametrize("triplet", [False, True])
def test_editor_and_file_quantization_share_the_variable_grid(
    variable_midi, tmp_path, strength, triplet
):
    parsed = mido.MidiFile(variable_midi)
    notes = read_midi_roll_notes(variable_midi)
    notes = tuple(replace(n, start=n.start + 0.14, end=n.end + 0.14) for n in notes)
    source = export_edited_midi(
        variable_midi, tmp_path / "offset.mid", notes, reference_bpm=120, target_bpm=120
    )
    exact_notes = read_midi_roll_notes(source)
    expected, _ = quantize_editor_notes(
        exact_notes,
        reference_timelines(parsed, 120),
        "1/16",
        strength=strength,
        triplet=triplet,
    )
    quantize_midi_notes(source, "1/16", strength=strength, triplet=triplet)
    actual = read_midi_roll_notes(source)
    for left, right in zip(expected, actual):
        assert left.start == pytest.approx(right.start, abs=0.003)
        assert left.end == pytest.approx(right.end, abs=0.003)


@pytest.mark.parametrize(
    "options",
    [
        MidiExportOptions(bpm=90),
        MidiExportOptions(beat_scale=2),
        MidiExportOptions(beats_per_bar=3, beat_unit=4),
    ],
)
def test_grid_corrections_keep_event_playback_seconds(variable_midi, tmp_path, options):
    result = apply_midi_export_options(variable_midi, tmp_path / "corrected.mid", options)
    assert_same_performance(variable_midi, result)


def test_downbeat_correction_aligns_bars_and_original_audio(variable_midi, tmp_path):
    result = apply_midi_export_options(
        variable_midi, tmp_path / "downbeat.mid", MidiExportOptions(first_downbeat=1.25)
    )
    offset = read_muscriptor_bar_offset_seconds(result)
    old, new = event_seconds(variable_midi), event_seconds(result)
    for before, after in zip(old, new):
        assert before[2] == pytest.approx(after[2] - offset, abs=0.003)
    timeline = MidiTimeline.from_midi(mido.MidiFile(result))
    assert timeline.beats_at(1.25 + offset) % 4 == pytest.approx(0, abs=0.003)


def test_gap_filling_and_trim_are_explicit_and_preserve_source(variable_midi, tmp_path):
    original = variable_midi.read_bytes()
    filled = apply_midi_export_options(
        variable_midi, tmp_path / "filled.mid", MidiExportOptions(fill_gaps=0.25)
    )
    notes = read_midi_roll_notes(filled)
    assert notes[0].end == pytest.approx(notes[1].start)
    trimmed = apply_midi_export_options(
        variable_midi, tmp_path / "trimmed.mid", MidiExportOptions(trim_leading_bars=True)
    )
    assert read_muscriptor_bar_offset_seconds(trimmed) == pytest.approx(-2)
    assert read_midi_roll_notes(trimmed)[0].start == pytest.approx(1)
    assert variable_midi.read_bytes() == original


def test_gap_fill_follows_releases_in_overlapping_chords(tmp_path):
    source = tmp_path / "chord.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500000),
                mido.Message("note_on", note=60, velocity=90),
                mido.Message("note_on", note=64, velocity=90, time=240),
                mido.Message("note_off", note=60, time=240),
                mido.Message("note_off", note=64),
                mido.Message("note_on", note=67, velocity=90, time=60),
                mido.Message("note_off", note=67, time=120),
            ]
        )
    )
    midi.save(source)
    output = apply_midi_export_options(
        source, tmp_path / "filled.mid", MidiExportOptions(fill_gaps=0.25)
    )
    notes = read_midi_roll_notes(output)
    assert notes[0].end == notes[1].end == pytest.approx(notes[2].start)


def test_signed_source_offset_crops_real_pcm(tmp_path):
    source = tmp_path / "original.wav"
    samples = np.sin(np.arange(44100 * 3) * 0.01).astype(np.float32) * 0.1
    sf.write(source, samples, 44100, subtype="PCM_16")
    destination = _write_bar_aligned_reference_audio(source, tmp_path, -1.0)
    actual, rate = sf.read(destination)
    original, _ = sf.read(source)
    assert rate == 44100
    np.testing.assert_array_equal(actual, original[44100:])


def test_optional_smoothing_keeps_notes_in_seconds(variable_midi, tmp_path):
    result = apply_midi_export_options(
        variable_midi, tmp_path / "smooth.mid", MidiExportOptions(smooth_tempo=True)
    )
    assert_same_performance(variable_midi, result)


def test_api_settings_validate_new_options_without_runtime_upgrade():
    from src.web_api.schemas import InferenceOptions

    options = InferenceOptions(
        quantize_strength=0.5,
        quantize_triplet=True,
        midi_export={"beat_scale": 2, "first_downbeat": 0.75},
    )
    assert options.midi_export.beat_scale == 2.0
    with pytest.raises(ValueError):
        InferenceOptions(quantize_strength=float("nan"))
    with pytest.raises(ValueError):
        InferenceOptions(midi_export={"beat_scale": 3})


def test_grid_correction_keeps_an_explicit_project_speed(variable_midi, tmp_path):
    from src.web_api.engine import InferenceEngine

    source = tmp_path / "manual.mid"
    fixed = apply_midi_export_options(variable_midi, source, MidiExportOptions(bpm=120))
    sped = write_project_tempo_map(fixed, tmp_path / "manual-speed.mid", 120, 180)
    expected = event_seconds(sped)
    report = InferenceEngine._apply_requested_quantization(
        sped,
        {"midi_export": {"bpm": 90, "beats_per_bar": 3}},
        reference_bpm=120,
        target_bpm=180,
    )
    assert report["reference_bpm"] == 90
    assert report["target_bpm"] == 135
    for before, after in zip(expected, event_seconds(sped)):
        assert before[:2] == after[:2]
        assert before[2] == pytest.approx(after[2], abs=0.003)


def test_edited_project_speed_scales_a_signed_original_audio_offset(variable_midi, tmp_path):
    trimmed = apply_midi_export_options(
        variable_midi,
        tmp_path / "trimmed.mid",
        MidiExportOptions(trim_leading_bars=True),
    )
    output = export_edited_midi(
        trimmed,
        tmp_path / "fast-trimmed.mid",
        read_midi_roll_notes(trimmed),
        reference_bpm=120,
        target_bpm=240,
    )
    assert read_muscriptor_bar_offset_seconds(output) == pytest.approx(-1)


@pytest.mark.parametrize("resolution", [100, 101, 480])
def test_fractional_triplet_steps_do_not_accumulate_clock_error(
    variable_midi, tmp_path, resolution
):
    midi = mido.MidiFile(variable_midi)
    midi.ticks_per_beat = resolution
    source = tmp_path / "fractional.mid"
    midi.save(source)
    notes = read_midi_roll_notes(source)
    timelines = reference_timelines(midi, 120)
    expected, _ = quantize_editor_notes(notes, timelines, "1/64", triplet=True)
    report = quantize_midi_notes(source, "1/64", triplet=True)
    assert report.grid_step_ticks == pytest.approx(resolution / 24)
    actual = read_midi_roll_notes(source)
    for left, right in zip(expected, actual):
        timeline = timelines[left.track_index]
        assert round(timeline.tick_at_seconds(left.start)) == round(
            timeline.tick_at_seconds(right.start)
        )
        assert round(timeline.tick_at_seconds(left.end)) == round(
            timeline.tick_at_seconds(right.end)
        )


@pytest.mark.parametrize("rate", [1, 2])
def test_browser_serialization_retains_map_and_signed_offset(variable_midi, tmp_path, rate):
    from dataclasses import asdict

    from src.gui.web.muscriptor_result_runtime import MUSCRIPTOR_RESULT_JS

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to verify the production browser MIDI serializer")
    source = apply_midi_export_options(
        variable_midi, tmp_path / "trimmed.mid", MidiExportOptions(trim_leading_bars=True)
    )
    payload = json.dumps(
        dict(
            midi=base64.b64encode(source.read_bytes()).decode(),
            notes=[asdict(n) for n in read_midi_roll_notes(source)],
            rate=rate,
        )
    )
    script = tmp_path / "serialize.cjs"
    script.write_text(
        "global.document={documentElement:{},readyState:'loading',addEventListener(){}};"
        "global.window={};global.MutationObserver=function(){this.observe=function(){}};\n"
        + MUSCRIPTOR_RESULT_JS
        + "\nconst p="
        + payload
        + ";"
        "const bytes=window.musicToMidiMidiEditorRuntime.buildEditedSmf("
        "Buffer.from(p.midi,'base64'),p.notes,120*p.rate,120,false);"
        "process.stdout.write(Buffer.from(bytes).toString('base64'));",
        encoding="utf-8",
    )
    completed = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    output = tmp_path / "browser.mid"
    output.write_bytes(base64.b64decode(completed.stdout))
    assert_same_performance(source, output, rate=rate)
    assert read_muscriptor_bar_offset_seconds(output) == pytest.approx(-2 / rate)


@pytest.mark.parametrize("kind", ["primary", "manual_midi"])
def test_real_engine_postprocessing_and_report_share_the_corrected_map(
    variable_midi, tmp_path, monkeypatch, kind
):
    from src.core import pipeline
    from src.models.data_models import BeatInfo, ProcessingResult
    from src.web_api.engine import InferenceEngine
    from src.web_api.schemas import InferenceOptions, ManualMidiOptions

    class FixturePipeline:
        def __init__(self, config):
            self.config = config

        def process(self, audio, output, progress, **kwargs):
            destination = Path(output) / "model-fixture.mid"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(variable_midi, destination)
            return ProcessingResult(
                midi_path=str(destination),
                total_notes=2,
                beat_info=BeatInfo(
                    bpm=120, time_signature=(4, 4), tempo_map=[(0, 120), (2, 60)]
                ),
            )

    monkeypatch.setattr(pipeline, "MusicToMidiPipeline", FixturePipeline)
    common = dict(
        quantize_notes=True,
        quantize_strength=0,
        quantize_triplet=True,
        quantize_auto=True,
        midi_export={"beat_scale": 2, "beats_per_bar": 3, "smooth_tempo": True},
    )
    options = (
        InferenceOptions(**common)
        if kind == "primary"
        else ManualMidiOptions(route="piano_transkun", **common)
    ).model_dump()
    result = InferenceEngine().run(
        kind=kind,
        source_path=tmp_path / "input.wav",
        output_dir=tmp_path / "output",
        options=options,
        progress_callback=lambda _: None,
        processor_callback=lambda _: None,
    )
    assert_same_performance(variable_midi, result.artifacts[0].path)
    assert result.result["beat"]["bpm"] == 240
    assert result.result["beat"]["time_signature"] == [3, 4]
    assert result.result["beat"]["is_variable_tempo"]
    assert result.result["quantization"]["strength"] == 0
    assert result.result["quantization"]["triplet"]


@pytest.mark.parametrize("options, rate", [({"beat_scale": 4}, 1), ({"bpm": 300}, 2)])
def test_out_of_range_corrections_preserve_api_and_browser_sources(
    variable_midi, tmp_path, options, rate
):
    from dataclasses import asdict

    from src.gui.web.edited_midi_preview import EditedMidiPreviewRegistry
    from src.web_api.engine import InferenceEngine

    original = variable_midi.read_bytes()
    with pytest.raises(ValueError):
        InferenceEngine._apply_requested_quantization(
            variable_midi, {"midi_export": options}, reference_bpm=120, target_bpm=120 * rate
        )
    assert variable_midi.read_bytes() == original
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(8000), 8000)
    registry = EditedMidiPreviewRegistry()
    token = registry.register(
        request_dir=tmp_path,
        source_midi_path=variable_midi,
        original_audio_path=audio,
        reference_bpm=120,
        muscriptor_groups=False,
        duration_seconds=8,
    )
    with pytest.raises(ValueError):
        registry.render(
            json.dumps(
                dict(
                    token=token,
                    operation="timing",
                    options=options,
                    project_rate=rate,
                    notes=[asdict(n) for n in read_midi_roll_notes(variable_midi)],
                )
            )
        )
    assert variable_midi.read_bytes() == original
    assert not (tmp_path / "timing-corrections").exists()


def test_desktop_correction_retains_a_manual_project_rate(variable_midi, tmp_path, qtbot):
    from src.gui.widgets.muscriptor_result import MuscriptorResultWidget
    from src.models.data_models import BeatInfo, ProcessingResult
    from src.utils.muscriptor_soundfont_downloader import validate_muscriptor_soundfont

    try:
        validate_muscriptor_soundfont()
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"Requires the installed real SoundFont: {exc}")
    audio = tmp_path / "original.wav"
    sf.write(audio, np.zeros(8000 * 8, dtype=np.float32), 8000)
    w = MuscriptorResultWidget(str(audio), [])
    qtbot.addWidget(w)
    try:
        w.finalize_result(
            ProcessingResult(
                midi_path=str(variable_midi),
                total_notes=2,
                beat_info=BeatInfo(
                    bpm=120, time_signature=(4, 4), tempo_map=[(0, 120), (2, 60)]
                ),
            )
        )
        qtbot.waitUntil(lambda: w._asset_worker is None, timeout=15000)
        assert w._playback_engine.is_configured, w.status_label.text()
        w.bpm_spin.setValue(180)
        assert w._bpm_user_overridden
        w._apply_timing_options({"bpm": 90, "beats_per_bar": 3})
        qtbot.waitUntil(lambda: w._asset_worker is None, timeout=15000)
        assert w._detected_bpm == 90
        assert w.bpm_spin.value() == 135
        assert w._playback_engine.is_configured, w.status_label.text()
        corrected = w._midi_path
        w._apply_timing_options({"beat_scale": 4})
        assert w._midi_path == corrected
        assert w._detected_bpm == 90 and w.bpm_spin.value() == 135
    finally:
        w.shutdown()
        w.close()
