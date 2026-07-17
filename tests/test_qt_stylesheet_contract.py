import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import qInstallMessageHandler
from PyQt6.QtWidgets import QApplication

from src.gui.widgets.audio_track_mixer import AudioTrackMixerWidget
from tests.test_audio_track_mixer import _BackendHarness


def test_audio_mixer_does_not_emit_unknown_font_variant_warning(tmp_path):
    app = QApplication.instance() or QApplication([])
    tracks = {}
    for name in ("bass", "drums", "guitar", "piano", "vocals", "other"):
        path = tmp_path / f"{name}.wav"
        path.write_bytes(b"test-audio-placeholder")
        tracks[name] = path

    messages = []
    previous_handler = qInstallMessageHandler(
        lambda _message_type, _context, message: messages.append(message)
    )
    mixer = None
    try:
        harness = _BackendHarness()
        mixer = AudioTrackMixerWidget(
            tracks,
            player_factory=harness.player_factory,
            audio_output_factory=harness.output_factory,
            waveform_loader_factory=harness.waveform_factory,
        )
        mixer.show()
        app.processEvents()
    finally:
        if mixer is not None:
            mixer.shutdown()
            mixer.close()
            mixer.deleteLater()
            app.processEvents()
        qInstallMessageHandler(previous_handler)

    matching_warnings = [
        message
        for message in messages
        if "Unknown property font-variant-numeric" in message
    ]
    assert matching_warnings == []


def test_gui_source_does_not_use_web_only_font_variant_numeric_property():
    mixer_source = Path("src/gui/widgets/audio_track_mixer.py").read_text(
        encoding="utf-8"
    )
    assert "font-variant-numeric" not in mixer_source

def test_main_window_statusbar_labels_use_dark_theme_text_color():
    source = Path("src/gui/main_window.py").read_text(encoding="utf-8")

    assert "QStatusBar QLabel" in source
    assert "status_label_style = \"background: transparent; color: #c8d3e6;\"" in source
    assert "label.setStyleSheet(status_label_style)" in source

def test_main_window_dark_theme_has_global_text_color_fallbacks():
    source = Path("src/gui/main_window.py").read_text(encoding="utf-8")

    assert "QMainWindow QLabel" in source
    assert "QMainWindow QCheckBox" in source
    assert "QToolBar QLabel" in source
    assert "QStatusBar QLabel" in source


def test_wrapped_checkbox_label_has_explicit_dark_theme_text_color():
    source = Path("src/gui/widgets/track_panel.py").read_text(encoding="utf-8")

    assert "self.label.setStyleSheet(\"color: #c8d3e6;\")" in source
