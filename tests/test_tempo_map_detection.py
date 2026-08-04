"""BeatInfo display and automatic Beat This tempo-map defaults."""

from src.models.data_models import BeatInfo, Config


def test_constant_tempo_display_is_single_value():
    info = BeatInfo(bpm=120.0)

    assert not info.is_variable_tempo
    assert info.bpm_display == "120.0"


def test_variable_tempo_display_is_range():
    info = BeatInfo(bpm=90.0, tempo_map=[(0.0, 69.8), (30.0, 128.4)])

    assert info.is_variable_tempo
    assert info.bpm_display == "69.8–128.4"


def test_single_point_tempo_map_is_treated_as_constant():
    info = BeatInfo(bpm=100.0, tempo_map=[(0.0, 100.0)])

    assert not info.is_variable_tempo
    assert info.bpm_display == "100.0"


def test_beat_this_variable_tempo_export_is_enabled_by_default():
    assert Config().enable_tempo_map is True


def test_legacy_disabled_tempo_map_config_is_normalized_to_the_only_production_chain():
    assert Config(enable_tempo_map=False).enable_tempo_map is True
    assert Config.from_dict({"enable_tempo_map": False}).enable_tempo_map is True
