from dataclasses import fields
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.gui.widgets.track_panel import TrackPanel
from src.models.data_models import Config

_APP = QApplication.instance() or QApplication([])


def test_config_no_longer_serializes_transcription_quality():
    field_names = {field.name for field in fields(Config)}

    assert "transcription_quality" not in field_names
    assert "transcription_quality" not in Config.from_dict(
        {"transcription_quality": "fast"}
    ).to_dict()


def test_desktop_track_panel_no_longer_exposes_quality_behavior_hint():
    panel = TrackPanel()

    assert not hasattr(panel, "quality_hint_label")
    assert not hasattr(panel, "get_quality_behavior")


def test_space_and_colab_no_longer_expose_quality_switching():
    space_source = Path("space/app.py").read_text(encoding="utf-8")
    colab_source = Path("colab_notebook.ipynb").read_text(encoding="utf-8")
    combined_source = space_source + "\n" + colab_source

    assert "quality_radio" not in combined_source
    assert "config.transcription_quality" not in combined_source
    assert '["fast", "balanced", "best"]' not in combined_source


def test_user_docs_no_longer_mention_fast_balanced_best_selector():
    readme = Path("README.md").read_text(encoding="utf-8")
    space_readme = Path("space/README.md").read_text(encoding="utf-8")

    assert "fast / balanced / best" not in readme
    assert "fast/balanced/best" not in readme
    assert "转写质量" not in space_readme
