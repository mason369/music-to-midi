"""
Source separation module using Demucs v4.
Separates audio into drums, bass, vocals, and other tracks.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Callable
import torch

from src.models.data_models import Config, TrackType
from src.utils.gpu_utils import get_device, clear_gpu_memory

logger = logging.getLogger(__name__)


class SourceSeparator:
    """
    Audio source separation using Demucs v4 (htdemucs).

    Separates audio into 4 stems:
    - drums: Drums and percussion
    - bass: Bass instruments
    - vocals: Vocals and voice
    - other: Other instruments (guitars, keyboards, etc.)
    """

    STEMS = ["drums", "bass", "vocals", "other"]
    MODEL_NAME = "htdemucs"

    def __init__(self, config: Config):
        """
        Initialize source separator.

        Args:
            config: Application configuration
        """
        self.config = config
        self.model = None
        self.device = get_device(config.use_gpu, config.gpu_device)
        self.segment_size = config.segment_size

    def load_model(self) -> None:
        """Load Demucs model (lazy loading to save memory)."""
        if self.model is not None:
            return

        logger.info(f"Loading Demucs model: {self.MODEL_NAME}")

        try:
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels

            self.model = get_model(self.MODEL_NAME)

            if isinstance(self.model, BagOfModels):
                logger.info(f"Loaded bag of {len(self.model.models)} models")
            else:
                logger.info("Loaded single model")

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded on device: {self.device}")

        except ImportError as e:
            logger.error("Demucs not installed. Install with: pip install demucs")
            raise ImportError("Demucs is required for source separation") from e

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            clear_gpu_memory()
            logger.info("Model unloaded")

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, str]:
        """
        Separate audio file into stems.

        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save separated stems
            progress_callback: Optional callback for progress updates (progress, message)

        Returns:
            Dictionary mapping stem names to output file paths
        """
        import torchaudio
        from demucs.apply import apply_model

        self.load_model()

        if progress_callback:
            progress_callback(0.0, "Loading audio...")

        # Load audio
        logger.info(f"Loading audio: {audio_path}")
        wav, sr = torchaudio.load(audio_path)

        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        # Add batch dimension
        wav = wav.unsqueeze(0)

        # Resample if needed
        if sr != self.model.samplerate:
            logger.info(f"Resampling from {sr} to {self.model.samplerate}")
            wav = torchaudio.transforms.Resample(sr, self.model.samplerate)(wav)

        wav = wav.to(self.device)

        if progress_callback:
            progress_callback(0.2, "Separating sources...")

        # Apply model
        logger.info("Applying source separation model...")
        with torch.no_grad():
            sources = apply_model(
                self.model,
                wav,
                device=self.device,
                segment=self.segment_size,
                overlap=0.25,
                progress=True
            )

        if progress_callback:
            progress_callback(0.8, "Saving separated tracks...")

        # Save stems
        output_paths = {}
        os.makedirs(output_dir, exist_ok=True)

        input_name = Path(audio_path).stem

        for i, stem in enumerate(self.model.sources):
            if stem in self.STEMS:
                output_path = os.path.join(output_dir, f"{input_name}_{stem}.wav")
                stem_audio = sources[0, i].cpu()

                torchaudio.save(
                    output_path,
                    stem_audio,
                    self.model.samplerate
                )

                output_paths[stem] = output_path
                logger.info(f"Saved {stem}: {output_path}")

        if progress_callback:
            progress_callback(1.0, "Separation complete")

        return output_paths

    def get_stem_for_track_type(self, track_type: TrackType) -> str:
        """Get stem name for track type."""
        return track_type.value
