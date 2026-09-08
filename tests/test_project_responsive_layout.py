from unittest.mock import patch

import pytest

from src.gui.main_window import MainWindow
from src.i18n.translator import set_language
from src.models.data_models import Config


@pytest.mark.parametrize("language", ["zh_CN", "en_US"])
@pytest.mark.parametrize("mode", [
    "smart", "vocal_split", "six_stem_split", "piano_transkun",
    "piano_transkun_v2_aug", "piano_aria_amt", "piano_bytedance_pedal",
])
def test_project_buttons_reflow_without_clipping_wide_fonts(qtbot, language, mode):
    set_language(language)
    with patch.object(MainWindow, "_start_gpu_detection", return_value=None):
        window = MainWindow(Config(language=language))
    qtbot.addWidget(window)
    try:
        window.track_panel.set_processing_mode(mode)
        window.progress_widget.set_mode(mode)
        for button in window.project_panel.translated_buttons.values():
            font = button.font()
            font.setStretch(150)
            button.setFont(font)
        window.showNormal()
        window.resize(320, 480)
        window.layout().activate()
        qtbot.waitUntil(
            lambda: window.content_scroll.widget().width()
            <= window.content_scroll.viewport().width()
        )
        panel = window.project_panel
        for key, button in panel.translated_buttons.items():
            assert panel.rect().contains(button.mapTo(panel, button.rect().topLeft())), key
            assert panel.rect().contains(button.mapTo(panel, button.rect().bottomRight())), key
            assert button.width() >= button.sizeHint().width(), key
        if panel.stem_group.isVisible():
            assert panel.stem_heading.text() == panel.words["stems"]
            assert panel.stem_heading.height() >= panel.stem_heading.heightForWidth(
                panel.stem_heading.width()
            )
    finally:
        window.close()
        set_language("zh_CN")
