"""A failed download must return from the real Qt action without aborting."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "failure",
    ["directory", "missing_source", "missing_bpm", "dialog", "stereo", "asset_not_ready", "cancel"],
)
@pytest.mark.parametrize("language", ["zh_CN", "en_US"])
def test_failed_download_keeps_qt_alive_and_allows_retry(tmp_path, failure, language):
    source = r"""
import logging
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import mido
from PyQt6.QtWidgets import QApplication, QFileDialog
from src.gui.widgets.muscriptor_result import MuscriptorResultWidget
from src.i18n.translator import set_language, t

root, failure = Path(sys.argv[1]), sys.argv[2]
set_language(sys.argv[3])
audio = root / "source.wav"
with wave.open(str(audio), "wb") as output:
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(8000)
    output.writeframes(b"\0\0" * 8000)
midi_path = root / "source.mid"
midi = mido.MidiFile()
track = mido.MidiTrack()
midi.tracks.append(track)
track.append(mido.MetaMessage("set_tempo", tempo=500000))
track.append(mido.Message("note_on", note=60, velocity=90))
track.append(mido.Message("note_off", note=60, time=480))
midi.save(midi_path)
original = midi_path.read_bytes()

log_path = root / "export.log"
logger = logging.getLogger("src")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(log_path, encoding="utf-8")
logger.addHandler(handler)
app = QApplication([])
widget = MuscriptorResultWidget(str(audio), ["acoustic_piano"])
widget._midi_path = str(midi_path)
widget.set_bpm_context(120, 120)
widget.show()
app.processEvents()
destination = root / "existing.mid"
previous = mido.MidiFile(midi_path)
previous.tracks[0][0].tempo = 600000
previous.tracks[0][1].note = 65
previous.tracks[0][2].note = 65
previous.save(destination)
previous_bytes = destination.read_bytes()
kind = "midi"

def choose(*args):
    return str(destination), "MIDI (*.mid)"

QFileDialog.getSaveFileName = choose
expected_error = {
    "directory": "PermissionError" if sys.platform == "win32" else "IsADirectoryError",
    "missing_source": "FileNotFoundError",
    "missing_bpm": "RuntimeError",
    "dialog": "OSError",
    "stereo": "FileNotFoundError",
    "asset_not_ready": "RuntimeError",
    "cancel": None,
}[failure]
if failure == "directory":
    destination = root / "occupied.mid"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
elif failure == "missing_source":
    widget._midi_path = str(root / "missing.mid")
elif failure == "missing_bpm":
    widget._detected_bpm = None
elif failure == "dialog":
    def reject_dialog(*args):
        raise OSError("file chooser unavailable")
    QFileDialog.getSaveFileName = reject_dialog
elif failure == "cancel":
    QFileDialog.getSaveFileName = lambda *args: ("", "")
if failure == "stereo":
    widget._assets = SimpleNamespace(stereo_mix_wav=root / "missing.wav")
    kind = "stereo"
elif failure == "asset_not_ready":
    kind = "stereo"

from PyQt6.QtGui import QAction
action = widget.download_midi_action if kind == "midi" else QAction(widget)
if kind == "stereo":
    action.triggered.connect(lambda: widget._save_asset("stereo"))
action.trigger()
app.processEvents()
assert widget.isVisible()
assert midi_path.read_bytes() == original
assert (root / "existing.mid").read_bytes() == previous_bytes
if destination.is_dir():
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"
handler.flush()
diagnostic = log_path.read_text(encoding="utf-8")
if failure == "cancel":
    assert "download failed" not in diagnostic
else:
    assert t("muscriptor_result.editor_export_failed", error="").strip() in widget.playback_status_label.text()
    assert "Traceback (most recent call last)" in diagnostic
    assert expected_error in diagnostic

widget._midi_path = str(midi_path)
widget.set_bpm_context(120, 120)
destination = root / ("retry.wav" if kind == "stereo" else "retry.mid")
QFileDialog.getSaveFileName = choose
if kind == "stereo":
    widget._assets = SimpleNamespace(stereo_mix_wav=audio)
action.trigger()
app.processEvents()
assert destination.is_file()
assert midi_path.read_bytes() == original
if kind == "midi":
    exported = mido.MidiFile(destination)
    assert sum(msg.type == "note_on" and msg.velocity > 0 for tr in exported.tracks for msg in tr) == 1
    assert next(msg.tempo for tr in exported.tracks for msg in tr if msg.type == "set_tempo") == 500000
else:
    assert destination.read_bytes() == audio.read_bytes()
    widget._assets = None
widget.shutdown()
widget.close()
app.processEvents()
"""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONIOENCODING": "utf-8"}
    if os.name == "nt":
        env.setdefault("QT_QPA_FONTDIR", str(Path(env["WINDIR"]) / "Fonts"))
    result = subprocess.run(
        [sys.executable, "-c", source, str(tmp_path), failure, language],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, f"{result.returncode}: {result.stdout}\n{result.stderr}"
