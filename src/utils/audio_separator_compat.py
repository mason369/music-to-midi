from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any


logger = logging.getLogger(__name__)

_PATCH_ATTR = "_music_to_midi_audio_separator_metadata_patch"
_FALLBACK_DISTRIBUTION = SimpleNamespace(version="unknown")


def patch_separator_package_metadata(separator_cls: Any) -> Any:
    """Patch audio-separator metadata lookup for frozen builds missing dist-info."""
    if separator_cls is None or getattr(separator_cls, _PATCH_ATTR, False):
        return separator_cls

    original = getattr(separator_cls, "get_package_distribution", None)
    if not callable(original):
        return separator_cls

    def _patched_get_package_distribution(self, package_name):
        distribution = original(self, package_name)
        if distribution is None and package_name == "audio-separator":
            logger.debug(
                "audio-separator package metadata is unavailable; using placeholder version"
            )
            return _FALLBACK_DISTRIBUTION
        return distribution

    separator_cls.get_package_distribution = _patched_get_package_distribution
    setattr(separator_cls, _PATCH_ATTR, True)
    return separator_cls


def get_separator_cls():
    from audio_separator.separator import Separator

    return patch_separator_package_metadata(Separator)
