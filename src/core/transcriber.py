"""
Audio to MIDI transcription module using Basic Pitch.
"""
import logging
from pathlib import Path
from typing import List, Optional, Callable, Tuple
import numpy as np

from src.models.data_models import Config, NoteEvent, TrackType
from src.utils.gpu_utils import get_device

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    Audio to MIDI transcription using Basic Pitch (Spotify).

    Features:
    - Polyphonic pitch detection
    - Instrument-agnostic transcription
    - Pitch bend detection
    """

    def __init__(self, config: Config):
        """
        Initialize transcriber.

        Args:
            config: Application configuration
        """
        self.config = config
        self.onset_threshold = config.onset_threshold
        self.frame_threshold = config.frame_threshold
        self.min_note_length = config.min_note_length
        self.device = get_device(config.use_gpu, config.gpu_device)

    def transcribe(
        self,
        audio_path: str,
        track_type: TrackType = TrackType.OTHER,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[NoteEvent]:
        """
        Transcribe audio file to MIDI note events.

        Args:
            audio_path: Path to audio file
            track_type: Type of track (affects processing parameters)
            progress_callback: Optional progress callback

        Returns:
            List of NoteEvent objects
        """
        logger.info(f"Transcribing: {audio_path} (type: {track_type.value})")

        if progress_callback:
            progress_callback(0.0, f"Transcribing {track_type.value}...")

        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH

            # Adjust thresholds based on track type
            onset_thresh, frame_thresh = self._get_thresholds(track_type)

            if progress_callback:
                progress_callback(0.3, "Running pitch detection...")

            # Run prediction
            model_output, midi_data, note_events = predict(
                audio_path,
                onset_threshold=onset_thresh,
                frame_threshold=frame_thresh,
                minimum_note_length=self.min_note_length,
                minimum_frequency=None,
                maximum_frequency=None,
                multiple_pitch_bends=False,
                melodia_trick=True
            )

            if progress_callback:
                progress_callback(0.8, "Processing note events...")

            # Convert to our NoteEvent format
            notes = self._process_note_events(note_events, track_type)

            if progress_callback:
                progress_callback(1.0, f"Found {len(notes)} notes")

            logger.info(f"Transcribed {len(notes)} notes from {track_type.value}")
            return notes

        except ImportError as e:
            logger.error("Basic Pitch not installed. Install with: pip install basic-pitch")
            raise ImportError("Basic Pitch is required for transcription") from e

    def _get_thresholds(self, track_type: TrackType) -> Tuple[float, float]:
        """Get optimized thresholds for track type."""
        # Adjust thresholds based on track characteristics
        if track_type == TrackType.DRUMS:
            # Higher thresholds for drums (transient-heavy)
            return 0.6, 0.4
        elif track_type == TrackType.BASS:
            # Lower thresholds for bass (sustained notes)
            return 0.4, 0.25
        elif track_type == TrackType.VOCALS:
            # Medium thresholds for vocals
            return 0.5, 0.3
        else:
            # Default thresholds
            return self.onset_threshold, self.frame_threshold

    def _process_note_events(
        self,
        note_events: List,
        track_type: TrackType
    ) -> List[NoteEvent]:
        """
        Process raw note events into NoteEvent objects.

        Args:
            note_events: Raw note events from Basic Pitch
            track_type: Track type for velocity adjustment

        Returns:
            List of NoteEvent objects
        """
        notes = []

        for start_time, end_time, pitch, velocity, pitch_bends in note_events:
            # Ensure pitch is in valid MIDI range
            midi_pitch = int(round(pitch))
            if midi_pitch < 0 or midi_pitch > 127:
                continue

            # Normalize velocity to 0-127
            midi_velocity = int(np.clip(velocity * 127, 1, 127))

            # Apply track-specific velocity scaling
            midi_velocity = self._adjust_velocity(midi_velocity, track_type)

            note = NoteEvent(
                pitch=midi_pitch,
                start_time=float(start_time),
                end_time=float(end_time),
                velocity=midi_velocity
            )
            notes.append(note)

        # Sort by start time
        notes.sort(key=lambda n: n.start_time)

        return notes

    def _adjust_velocity(self, velocity: int, track_type: TrackType) -> int:
        """Adjust velocity based on track type."""
        if track_type == TrackType.DRUMS:
            # Drums are typically louder
            velocity = int(velocity * 1.1)
        elif track_type == TrackType.BASS:
            # Bass is typically more consistent
            velocity = int(velocity * 0.95)
        elif track_type == TrackType.VOCALS:
            # Vocals have dynamic range
            pass

        return int(np.clip(velocity, 1, 127))

    def transcribe_to_midi(
        self,
        audio_path: str,
        output_path: str,
        track_type: TrackType = TrackType.OTHER
    ) -> str:
        """
        Transcribe audio directly to MIDI file.

        Args:
            audio_path: Input audio path
            output_path: Output MIDI path
            track_type: Track type

        Returns:
            Path to output MIDI file
        """
        from basic_pitch.inference import predict_and_save

        logger.info(f"Transcribing to MIDI: {audio_path} -> {output_path}")

        onset_thresh, frame_thresh = self._get_thresholds(track_type)

        predict_and_save(
            [audio_path],
            Path(output_path).parent,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            onset_threshold=onset_thresh,
            frame_threshold=frame_thresh,
            minimum_note_length=self.min_note_length
        )

        return output_path
