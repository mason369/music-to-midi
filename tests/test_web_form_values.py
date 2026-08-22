import pytest

from src.gui.web.form_values import normalize_optional_project_bpm
from src.models.data_models import Config


@pytest.mark.parametrize("value", [None, "", 0, 0.0, "0"])
def test_zero_and_empty_web_project_bpm_keep_automatic_detection(value):
    normalized = normalize_optional_project_bpm(value)

    assert normalized is None
    assert Config(custom_bpm=normalized).custom_bpm is None


@pytest.mark.parametrize("value", [30, 120.5, "300"])
def test_valid_web_project_bpm_remains_an_explicit_override(value):
    normalized = normalize_optional_project_bpm(value)

    assert normalized == float(value)
    assert Config(custom_bpm=normalized).custom_bpm == float(value)


@pytest.mark.parametrize("value", [0.1, 29.9, 300.1, 400, "not-a-number"])
def test_invalid_nonzero_web_project_bpm_still_fails_explicitly(value):
    with pytest.raises((TypeError, ValueError)):
        Config(custom_bpm=normalize_optional_project_bpm(value))
