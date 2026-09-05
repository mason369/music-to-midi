"""Restore process-wide UI language after each independent test."""

import os
from pathlib import Path

import pytest

# Windows' offscreen Qt plugin does not locate the system fonts by itself.
# Use the real installed fonts, as the native desktop does, for layout tests.
if os.name == "nt":
    windows_fonts = Path(os.environ["WINDIR"]) / "Fonts"
    if not windows_fonts.is_dir():
        raise RuntimeError(f"Windows font directory is missing: {windows_fonts}")
    os.environ.setdefault("QT_QPA_FONTDIR", str(windows_fonts))


@pytest.fixture(autouse=True)
def restore_ui_language():
    from src.i18n.translator import get_translator, set_language

    language = get_translator().get_language()
    try:
        yield
    finally:
        set_language(language)
