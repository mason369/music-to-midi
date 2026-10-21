"""Exercise the actual modal timing form and every supported locale."""

import json
import re
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QSpinBox,
)

from src.gui.widgets.midi_timing_controls import MidiTimingControls
from src.i18n.translator import Translator, set_language, t


@pytest.mark.parametrize("language", list(Translator.AVAILABLE_LANGUAGES))
def test_modal_timing_form_emits_only_after_explicit_apply(qtbot, language):
    set_language(language)
    control = MidiTimingControls()
    qtbot.addWidget(control)
    control.show()
    emitted = []
    control.apply_requested.connect(emitted.append)
    checks = []

    def fill_and_close():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QDialog)
        buttons = dialog.findChild(QDialogButtonBox)
        checks.append(buttons.button(QDialogButtonBox.StandardButton.Cancel).text())
        assert emitted == []
        dialog.findChild(QDoubleSpinBox, "timing_bpm").setValue(90)
        scale = dialog.findChild(QComboBox, "timing_beatscale")
        scale.setCurrentIndex(scale.findData(2.0))
        dialog.findChild(QSpinBox).setValue(3)
        dialog.findChild(QDoubleSpinBox, "timing_first_downbeat").setValue(0.75)
        dialog.findChild(QDoubleSpinBox, "timing_fill_gaps").setValue(0.125)
        dialog.findChild(QCheckBox, "timing_trim").setChecked(True)
        dialog.findChild(QCheckBox, "timing_smooth").setChecked(True)
        QTest.mouseClick(
            buttons.button(QDialogButtonBox.StandardButton.Apply), Qt.MouseButton.LeftButton
        )

    QTimer.singleShot(0, fill_and_close)
    QTest.mouseClick(control, Qt.MouseButton.LeftButton)
    assert checks == [t("muscriptor_result.timing_cancel")]
    assert emitted == [
        dict(
            bpm=90,
            beats_per_bar=3,
            beat_unit=4,
            first_downbeat=0.75,
            beat_scale=2,
            fill_gaps=0.125,
            trim_leading_bars=True,
            smooth_tempo=True,
        )
    ]
    QTimer.singleShot(0, lambda: QApplication.activeModalWidget().reject())
    QTest.mouseClick(control, Qt.MouseButton.LeftButton)
    assert len(emitted) == 1
    set_language("zh_CN")


def test_full_catalog_keys_placeholders_and_new_timing_strings_have_no_fallback():
    root = Path(__file__).resolve().parents[1]

    def flatten(data, prefix=""):
        result = {}
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else key
            result.update(flatten(value, name) if isinstance(value, dict) else {name: value})
        return result

    languages = set(Translator.AVAILABLE_LANGUAGES)
    frontend = (root / "web/app.js").read_text(encoding="utf-8")
    web_languages = set(re.findall(r'(\w+):\s*"(?:\./)?locales/', frontend.split("};", 1)[0]))
    assert languages == web_languages
    for folder in ("src/i18n", "web/locales"):
        catalogs = {
            language: flatten(
                json.loads((root / folder / f"{language}.json").read_text(encoding="utf-8"))
            )
            for language in languages
        }
        reference = catalogs["en_US"]
        for language, catalog in catalogs.items():
            assert catalog.keys() == reference.keys()
            for key, value in catalog.items():
                assert isinstance(value, str), (language, key)
                assert sorted(re.findall(r"\{[^{}]+\}", value)) == sorted(
                    re.findall(r"\{[^{}]+\}", reference[key])
                ), (language, key)
                if key.startswith("muscriptor_result.timing_") or key.startswith("timing."):
                    assert value and value != key
                    if folder == "src/i18n":
                        translator = Translator(language)
                        assert (
                            translator.translations[language]["muscriptor_result"][
                                key.split(".", 1)[1]
                            ]
                            == value
                        )
