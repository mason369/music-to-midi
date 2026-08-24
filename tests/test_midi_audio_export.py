import html
import json
import re
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf
from PyQt6.QtWidgets import QApplication

from src.core import muscriptor_result_assets
from src.core.muscriptor_result_assets import (
    DEFAULT_MIDI_AUDIO_EXPORT_PRESET,
    MIDI_AUDIO_EXPORT_PRESETS,
    get_midi_audio_export_preset,
    render_midi_audio_export,
)
from src.gui.web.muscriptor_result_runtime import (
    MUSCRIPTOR_RESULT_JS,
    build_muscriptor_result_html,
)
from src.gui.widgets import muscriptor_result
from src.gui.widgets.muscriptor_result import MuscriptorResultWidget


def _write_test_midi(path: Path, *, with_note: bool = True) -> Path:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    if with_note:
        track.append(mido.Message("note_on", channel=0, note=60, velocity=90, time=0))
        track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(path)
    return path


@pytest.mark.parametrize(
    ("preset_id", "sample_rate", "subtype", "fluidsynth_format"),
    [
        ("pcm24_48000", 48_000, "PCM_24", "s24"),
        ("pcm16_44100", 44_100, "PCM_16", "s16"),
    ],
)
def test_midi_audio_export_rerenders_and_atomically_publishes_selected_format(
    tmp_path: Path,
    monkeypatch,
    preset_id: str,
    sample_rate: int,
    subtype: str,
    fluidsynth_format: str,
):
    source = _write_test_midi(tmp_path / "source.mid")
    executable = tmp_path / "fluidsynth.exe"
    soundfont = tmp_path / "MuseScore_General.sf2"
    executable.write_bytes(b"runtime")
    soundfont.write_bytes(b"soundfont")
    destination = tmp_path / "result.wav"
    destination.write_bytes(b"previous-complete-file")
    calls = []

    monkeypatch.setattr(
        muscriptor_result_assets,
        "get_fluidsynth_executable",
        lambda: executable,
    )
    monkeypatch.setattr(
        muscriptor_result_assets,
        "validate_muscriptor_soundfont",
        lambda: soundfont,
    )

    def fake_synthesize(
        actual_executable,
        actual_soundfont,
        midi_path,
        output_path,
        cancel_check=None,
        *,
        sample_rate,
        audio_file_format,
    ):
        assert actual_executable == executable
        assert actual_soundfont == soundfont
        assert re.fullmatch(r"\.m2m-[0-9a-f]{16}\.mid", Path(midi_path).name)
        assert cancel_check is not None and not cancel_check()
        calls.append((sample_rate, audio_file_format))
        time_axis = np.arange(2_000, dtype=np.float32) / sample_rate
        tone = 0.1 * np.sin(2 * np.pi * 440.0 * time_axis)
        sf.write(
            output_path,
            np.stack([tone, tone], axis=1),
            sample_rate,
            subtype=subtype,
        )

    monkeypatch.setattr(muscriptor_result_assets, "_synthesize", fake_synthesize)

    result = render_midi_audio_export(source, destination, preset_id)

    info = sf.info(destination)
    assert calls == [(sample_rate, fluidsynth_format)]
    assert info.format == "WAV"
    assert info.subtype == subtype
    assert info.samplerate == sample_rate
    assert info.channels == 2
    assert result.path == destination.resolve()
    assert result.preset.id == preset_id
    assert result.frames == 2_000
    assert result.channels == 2
    assert result.peak > 0.09
    assert not list(tmp_path.glob(".m2m-*"))


def test_midi_audio_export_rejects_wrong_renderer_format_without_replacing_destination(
    tmp_path: Path,
    monkeypatch,
):
    source = _write_test_midi(tmp_path / "source.mid")
    destination = tmp_path / "result.wav"
    previous = b"previous-complete-file"
    destination.write_bytes(previous)
    executable = tmp_path / "fluidsynth.exe"
    soundfont = tmp_path / "MuseScore_General.sf2"
    executable.write_bytes(b"runtime")
    soundfont.write_bytes(b"soundfont")
    monkeypatch.setattr(
        muscriptor_result_assets,
        "get_fluidsynth_executable",
        lambda: executable,
    )
    monkeypatch.setattr(
        muscriptor_result_assets,
        "validate_muscriptor_soundfont",
        lambda: soundfont,
    )

    def wrong_format(
        _executable,
        _soundfont,
        _midi_path,
        output_path,
        cancel_check=None,
        **_settings,
    ):
        assert cancel_check is not None and not cancel_check()
        sf.write(output_path, np.ones((512, 2), dtype=np.float32) * 0.1, 44_100)

    monkeypatch.setattr(muscriptor_result_assets, "_synthesize", wrong_format)

    with pytest.raises(RuntimeError, match="export verification failed"):
        render_midi_audio_export(source, destination, "pcm24_48000")

    assert destination.read_bytes() == previous
    assert not list(tmp_path.glob(".m2m-*"))


def test_empty_edited_midi_exports_exact_verified_stereo_silence(
    tmp_path: Path,
    monkeypatch,
):
    source = _write_test_midi(tmp_path / "empty.mid", with_note=False)
    destination = tmp_path / "empty.wav"
    monkeypatch.setattr(
        muscriptor_result_assets,
        "get_fluidsynth_executable",
        lambda: pytest.fail("Empty MIDI must not invoke FluidSynth"),
    )

    result = render_midi_audio_export(
        source,
        destination,
        DEFAULT_MIDI_AUDIO_EXPORT_PRESET,
        silence_duration_seconds=0.25,
    )

    info = sf.info(destination)
    samples, sample_rate = sf.read(destination, dtype="float32", always_2d=True)
    assert info.subtype == "PCM_24"
    assert info.samplerate == 48_000
    assert info.channels == 2
    assert info.frames == 12_000
    assert sample_rate == 48_000
    assert samples.shape == (12_000, 2)
    assert np.count_nonzero(samples) == 0
    assert result.peak == 0.0


def test_cancelled_midi_audio_export_preserves_existing_destination(tmp_path: Path):
    source = _write_test_midi(tmp_path / "source.mid")
    destination = tmp_path / "result.wav"
    previous = b"previous-complete-file"
    destination.write_bytes(previous)

    with pytest.raises(InterruptedError, match="cancelled"):
        render_midi_audio_export(
            source,
            destination,
            cancel_check=lambda: True,
        )

    assert destination.read_bytes() == previous
    assert not list(tmp_path.glob(".m2m-*"))


@pytest.mark.parametrize(
    ("sample_rate", "audio_file_format", "expected_setting"),
    [(48_000, "s24", "audio.file.format=s24"), (44_100, None, None)],
)
def test_fluidsynth_command_preserves_preview_defaults_and_supports_24_48(
    tmp_path: Path,
    monkeypatch,
    sample_rate: int,
    audio_file_format: str | None,
    expected_setting: str | None,
):
    commands = []

    class CompletedProcess:
        returncode = 0

        def poll(self):
            return 0

        def communicate(self):
            return b"", b""

    def fake_popen(command, **_kwargs):
        commands.append(command)
        output = Path(command[command.index("-F") + 1])
        output.write_bytes(b"rendered")
        return CompletedProcess()

    monkeypatch.setattr(muscriptor_result_assets.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        muscriptor_result_assets,
        "get_fluidsynth_subprocess_env",
        lambda _executable: {},
    )
    destination = tmp_path / "output.wav"
    kwargs = {"sample_rate": sample_rate}
    if audio_file_format is not None:
        kwargs["audio_file_format"] = audio_file_format

    muscriptor_result_assets._synthesize(
        tmp_path / "fluidsynth.exe",
        tmp_path / "soundfont.sf2",
        tmp_path / "source.mid",
        destination,
        **kwargs,
    )

    command = commands[0]
    assert command[command.index("-r") + 1] == str(sample_rate)
    if expected_setting is None:
        assert "-o" not in command
    else:
        assert command[command.index("-o") + 1] == expected_setting
    assert destination.read_bytes() == b"rendered"


def test_browser_workbench_exposes_default_recommended_wav_export_contract():
    state = {
        "playback_audio_path": "C:/tmp/original.wav",
        "midi_path": "C:/tmp/result.mid",
        "transcription_wav": "C:/tmp/transcription.wav",
        "stereo_mix_wav": "C:/tmp/stereo.wav",
        "instrument_wavs": {},
        "notes": [],
        "duration": 1.0,
        "reference_bpm": 120.0,
        "target_bpm": 120.0,
        "preview_api": "./api/render_edited_midi_preview",
        "audio_export_api": "./api/render_edited_midi_audio_export",
        "preview_token": "opaque-token",
    }
    markup = build_muscriptor_result_html(state, lambda key: key, "zh_CN")
    encoded = re.search(r'<pre class="msr-manifest" hidden>(.*?)</pre>', markup).group(1)
    manifest = json.loads(html.unescape(encoded))

    assert manifest["audioExportApi"] == "./api/render_edited_midi_audio_export"
    assert manifest["defaultAudioExportPreset"] == "pcm24_48000"
    assert manifest["audioExportPresets"] == [
        {
            "id": preset.id,
            "bitDepth": preset.bit_depth,
            "sampleRate": preset.sample_rate,
            "subtype": preset.soundfile_subtype,
            "label": f"muscriptor_result.audio_export_{preset.id}",
        }
        for preset in MIDI_AUDIO_EXPORT_PRESETS
    ]
    assert "ResultSession.prototype.downloadTranscriptionAudio" in MUSCRIPTOR_RESULT_JS
    assert "MIDI audio export verification metadata is invalid" in MUSCRIPTOR_RESULT_JS


def test_space_colab_and_shared_runtime_expose_the_same_audio_export_endpoint():
    space_source = Path("space/app.py").read_text(encoding="utf-8")
    notebook = json.loads(Path("colab_notebook.ipynb").read_text(encoding="utf-8"))
    colab_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for source in (space_source, colab_source):
        assert "def render_edited_midi_audio_export" in source
        assert 'api_name="render_edited_midi_audio_export"' in source
        assert '"audio_export_api": "./api/render_edited_midi_audio_export"' in source
        assert "duration_seconds=assets.duration" in source


def test_desktop_download_menu_defaults_to_24_48_and_exports_current_snapshot(
    tmp_path: Path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    source_audio = tmp_path / "source.wav"
    samples = np.zeros(44_100, dtype=np.float32)
    sf.write(source_audio, samples, 44_100, subtype="PCM_16")
    source_midi = _write_test_midi(tmp_path / "source.mid")
    rendered = tmp_path / "preview.wav"
    sf.write(rendered, samples, 44_100, subtype="PCM_16")
    note = muscriptor_result_assets.read_midi_roll_notes(source_midi)[0]
    assets = muscriptor_result_assets.MuscriptorPlaybackAssets(
        notes=(note,),
        duration=1.0,
        transcription_wav=rendered,
        live_transcription_wav=rendered,
        original_wav=source_audio,
        stereo_mix_wav=rendered,
        original_left_wav=source_audio,
        transcription_right_wav=rendered,
        instrument_wavs={note.instrument: rendered},
        instrument_right_wavs={note.instrument: rendered},
        midi_gain_db=0.0,
    )
    destination = tmp_path / "selected-24-48.wav"
    render_calls = []

    def fake_render(
        midi_path,
        output_path,
        preset_id,
        *,
        silence_duration_seconds,
        cancel_check,
    ):
        assert Path(midi_path).name == "current-source-tempo.mid"
        assert Path(midi_path).read_bytes() == source_midi.read_bytes()
        assert not cancel_check()
        preset = get_midi_audio_export_preset(preset_id)
        tone = np.full((4_800, 2), 0.1, dtype=np.float32)
        sf.write(output_path, tone, preset.sample_rate, subtype=preset.soundfile_subtype)
        render_calls.append((preset_id, silence_duration_seconds))
        return muscriptor_result_assets.MidiAudioExportResult(
            path=Path(output_path).resolve(),
            preset=preset,
            frames=4_800,
            channels=2,
            peak=0.1,
        )

    monkeypatch.setattr(muscriptor_result, "render_midi_audio_export", fake_render)
    monkeypatch.setattr(
        muscriptor_result.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(destination), "WAV (*.wav)"),
    )
    widget = MuscriptorResultWidget(str(source_audio), ["acoustic_piano"])
    try:
        widget._midi_path = str(source_midi)
        widget.set_detected_bpm(120.0)
        widget._apply_final_assets(assets)
        assert widget.download_transcription_hq_action.isEnabled()
        assert widget.download_transcription_compat_action.isEnabled()
        assert widget.download_transcription_menu.actions()[0] is (
            widget.download_transcription_hq_action
        )
        assert "24-bit / 48 kHz" in widget.download_transcription_hq_action.text()

        widget._start_midi_audio_export(DEFAULT_MIDI_AUDIO_EXPORT_PRESET)
        worker = widget._audio_export_worker
        assert worker is not None
        assert not widget.download_transcription_hq_action.isEnabled()
        assert worker.wait(5_000)
        for _ in range(5):
            app.processEvents()

        assert render_calls == [("pcm24_48000", 1.0)]
        info = sf.info(destination)
        assert info.subtype == "PCM_24"
        assert info.samplerate == 48_000
        assert info.channels == 2
        assert widget._audio_export_worker is None
        assert widget.download_transcription_hq_action.isEnabled()
        assert not any(widget._audio_export_root.iterdir())

        widget._record_editor_commit((note,), ())
        assert not widget.download_transcription_hq_action.isEnabled()
        assert not widget.download_transcription_compat_action.isEnabled()
    finally:
        widget.shutdown()
        widget.close()
        app.processEvents()


def test_unknown_audio_export_preset_is_rejected_explicitly():
    with pytest.raises(ValueError, match="Unsupported MIDI audio export preset"):
        get_midi_audio_export_preset("pcm32_192000")
