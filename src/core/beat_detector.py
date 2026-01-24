"""
Beat detection module using librosa.
"""
import logging
from typing import Optional, Callable
import numpy as np

from src.models.data_models import Config, BeatInfo

logger = logging.getLogger(__name__)


class BeatDetector:
    """
    Beat and tempo detection using librosa.

    Features:
    - BPM estimation
    - Beat position detection
    - Downbeat detection (optional)
    """

    def __init__(self, config: Config):
        """
        Initialize beat detector.

        Args:
            config: Application configuration
        """
        self.config = config

    def detect(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> BeatInfo:
        """
        Detect beats and tempo from audio.

        Args:
            audio_path: Path to audio file
            progress_callback: Optional progress callback

        Returns:
            BeatInfo with tempo and beat times
        """
        import librosa

        if progress_callback:
            progress_callback(0.0, "Loading audio for beat detection...")

        logger.info(f"Detecting beats: {audio_path}")

        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)

        if progress_callback:
            progress_callback(0.3, "Analyzing tempo...")

        # Estimate tempo
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Handle both scalar and array tempo
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        if progress_callback:
            progress_callback(0.6, "Finding beat positions...")

        # Convert frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        if progress_callback:
            progress_callback(0.8, "Detecting downbeats...")

        # Try to detect downbeats
        downbeats = self._detect_downbeats(y, sr, beat_times)

        if progress_callback:
            progress_callback(1.0, f"BPM: {tempo:.1f}")

        beat_info = BeatInfo(
            bpm=tempo,
            beat_times=beat_times.tolist(),
            downbeats=downbeats,
            time_signature=(4, 4)
        )

        logger.info(f"Detected BPM: {tempo:.1f}, {len(beat_times)} beats")

        return beat_info

    def _detect_downbeats(
        self,
        y: np.ndarray,
        sr: int,
        beat_times: np.ndarray
    ) -> Optional[list]:
        """
        Detect downbeats (first beat of each measure).

        Args:
            y: Audio signal
            sr: Sample rate
            beat_times: Beat times

        Returns:
            List of downbeat times or None
        """
        try:
            import librosa

            # Compute onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # Get beat strengths
            beat_frames = librosa.time_to_frames(beat_times, sr=sr)
            beat_frames = beat_frames[beat_frames < len(onset_env)]

            if len(beat_frames) < 4:
                return None

            beat_strengths = onset_env[beat_frames]

            # Find strong beats (potential downbeats)
            # Assume 4/4 time signature
            downbeats = []
            for i in range(0, len(beat_times), 4):
                if i < len(beat_times):
                    downbeats.append(beat_times[i])

            return downbeats

        except Exception as e:
            logger.warning(f"Could not detect downbeats: {e}")
            return None

    def estimate_tempo(self, audio_path: str) -> float:
        """
        Quick tempo estimation.

        Args:
            audio_path: Path to audio file

        Returns:
            Estimated BPM
        """
        import librosa

        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        if hasattr(tempo, '__len__'):
            return float(tempo[0]) if len(tempo) > 0 else 120.0
        return float(tempo)
