"""Authoritative Beat This beat/downbeat detection for every MIDI workflow."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from src.core.beat_this_tracker import BeatThisTracker
from src.models.data_models import BeatInfo, Config

logger = logging.getLogger(__name__)


class BeatDetector:
    """Expose one strict Beat This ``final0`` chain to the product pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self._tracker = BeatThisTracker(config)

    def detect(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> BeatInfo:
        """Detect and validate one grid; never use a second detector or fallback."""

        logger.info("使用 Beat This final0 检测节拍: %s", audio_path)
        return self._tracker.detect(
            audio_path,
            progress_callback=progress_callback,
        )

    def estimate_tempo(self, audio_path: str) -> float:
        """Return the same authoritative tempo used by full MIDI conversion."""

        tempo = float(self.detect(audio_path).bpm)
        logger.info("Beat This 估算 BPM: %.3f", tempo)
        return tempo


__all__ = ["BeatDetector"]
