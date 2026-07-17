"""Shared explicit per-track MIDI route configuration.

This module deliberately has no GUI dependency so desktop, Space, and Colab
all use the same route contract after a WAV-only separation job.
"""

from __future__ import annotations

from pathlib import Path

from src.models.data_models import (
    Config,
    MultiInstrumentModel,
    ProcessingMode,
    YourMT3Model,
)

MIDI_ROUTE_YOURMT3_PREFIX = f"{MultiInstrumentModel.YOURMT3.value}:"
MIDI_ROUTE_MIROS = MultiInstrumentModel.MIROS.value
MIDI_ROUTE_PIANO_TRANSKUN = ProcessingMode.PIANO_TRANSKUN.value
MIDI_ROUTE_PIANO_TRANSKUN_V2_AUG = ProcessingMode.PIANO_TRANSKUN_V2_AUG.value
MIDI_ROUTE_PIANO_ARIA_AMT = ProcessingMode.PIANO_ARIA_AMT.value
MIDI_ROUTE_PIANO_BYTEDANCE_PEDAL = ProcessingMode.PIANO_BYTEDANCE_PEDAL.value

YOURMT3_MANUAL_MODELS = (
    YourMT3Model.YMT3_PLUS,
    YourMT3Model.YPTF_SINGLE_NOPS,
    YourMT3Model.YPTF_MULTI_PS,
    YourMT3Model.YPTF_MOE_MULTI_NOPS,
    YourMT3Model.YPTF_MOE_MULTI_PS,
)

MANUAL_MIDI_ROUTES = tuple(
    f"{MIDI_ROUTE_YOURMT3_PREFIX}{model.value}"
    for model in YOURMT3_MANUAL_MODELS
) + (
    MIDI_ROUTE_MIROS,
    MIDI_ROUTE_PIANO_TRANSKUN,
    MIDI_ROUTE_PIANO_TRANSKUN_V2_AUG,
    MIDI_ROUTE_PIANO_ARIA_AMT,
    MIDI_ROUTE_PIANO_BYTEDANCE_PEDAL,
)


def _validate_manual_midi_route(route: str) -> str:
    normalized = str(route)
    if normalized not in MANUAL_MIDI_ROUTES:
        raise ValueError(f"Unsupported manual MIDI route: {normalized!r}")
    return normalized


def build_manual_midi_config(base_config: Config, route: str) -> Config:
    """Build an isolated direct-transcription config for one selected WAV."""
    selected_route = _validate_manual_midi_route(route)
    config = Config.from_dict(base_config.to_dict())

    if selected_route.startswith(MIDI_ROUTE_YOURMT3_PREFIX):
        model_name = selected_route.removeprefix(MIDI_ROUTE_YOURMT3_PREFIX)
        config.processing_mode = ProcessingMode.SMART.value
        config.transcription_backend = MultiInstrumentModel.YOURMT3.value
        config.multi_instrument_model = MultiInstrumentModel.YOURMT3.value
        config.yourmt3_model = model_name
    elif selected_route == MIDI_ROUTE_MIROS:
        config.processing_mode = ProcessingMode.SMART.value
        config.transcription_backend = MultiInstrumentModel.MIROS.value
        config.multi_instrument_model = MultiInstrumentModel.MIROS.value
    else:
        config.processing_mode = selected_route

    config.save_separated_tracks = True
    config.vocal_split_merge_midi = False
    config.validate()
    return config


def manual_midi_output_dir(audio_path: str | Path, route: str) -> Path:
    """Return the deterministic per-route directory beside one separated WAV."""
    selected_route = _validate_manual_midi_route(route)
    resolved_audio = Path(audio_path).resolve()
    return resolved_audio.parent / "midi" / selected_route.replace(":", "_")
