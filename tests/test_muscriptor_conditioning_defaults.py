"""Only explicit user selections may become MuScriptor instrument conditions."""

import ast
import json
from pathlib import Path

import pytest

from src.models.data_models import Config
from src.models.muscriptor_instruments import MUSCRIPTOR_INSTRUMENTS


@pytest.mark.parametrize("filename", ["piano.wav", "song_guitar.wav", "drums.wav"])
@pytest.mark.parametrize("selected", [[], ["electric_bass"], list(MUSCRIPTOR_INSTRUMENTS)])
def test_desktop_import_preserves_explicit_instrument_selection(
    qapp, monkeypatch, tmp_path, filename, selected
):
    from src.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_start_gpu_detection", lambda self: None)
    window = MainWindow(Config())
    try:
        window.track_panel.set_muscriptor_instruments(selected)
        window._on_file_selected(str(tmp_path / filename))
        assert window.track_panel.get_muscriptor_instruments() == selected
    finally:
        window.close()


@pytest.mark.parametrize("name", ["piano", "guitar", "bass", "drums", "vocals"])
def test_desktop_new_stem_keeps_official_automatic_detection(qapp, tmp_path, name):
    from src.gui.widgets.audio_track_mixer import _AudioTrackRow

    row = _AudioTrackRow(name, tmp_path / f"{name}.wav", "#4a9eff")
    try:
        route_index = row.midi_model_selector.findData("muscriptor")
        assert route_index >= 0
        row.midi_model_selector.setCurrentIndex(route_index)
        assert row.selected_muscriptor_instruments() == []
        row.muscriptor_instrument_selector.set_selected_instruments(["acoustic_piano"])
        assert row.selected_muscriptor_instruments() == ["acoustic_piano"]
    finally:
        row.close()


def test_space_and_colab_do_not_infer_hard_conditions_from_filenames():
    notebook = json.loads(Path("colab_notebook.ipynb").read_text(encoding="utf-8"))
    sources = {
        "Space": Path("space/app.py").read_text(encoding="utf-8"),
        "Colab": "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell.get("cell_type") == "code"
        ),
    }
    for platform, source in sources.items():
        calls = [
            node.func.id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert "infer_muscriptor_instruments_from_stem_name" not in calls, platform
        assert "suggest_muscriptor_instruments" not in source, platform


def test_all_supported_catalogs_explain_conditioning_without_filename_promises():
    from src.i18n.translator import Translator

    assert set(Translator.AVAILABLE_LANGUAGES) == {"zh_CN", "en_US"}
    for language in Translator.AVAILABLE_LANGUAGES:
        desktop = json.loads(Path(f"src/i18n/{language}.json").read_text(encoding="utf-8"))
        web = json.loads(Path(f"web/locales/{language}.json").read_text(encoding="utf-8"))
        assert (
            desktop["main"]["engine"]["muscriptor_instruments_desc"]
            == web["config.muscriptor_instruments_help"]
        )
        description = web["config.muscriptor_instruments_help"]
        assert "filename" not in description and "文件名" not in description
