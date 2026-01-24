"""
Lyrics recognition module using Whisper and WhisperX.
"""
import logging
from typing import List, Optional, Callable
import warnings

from src.models.data_models import Config, LyricEvent
from src.utils.gpu_utils import get_device, clear_gpu_memory

logger = logging.getLogger(__name__)

# Suppress whisper warnings
warnings.filterwarnings("ignore", category=UserWarning)


class LyricsRecognizer:
    """
    Lyrics recognition and alignment using Whisper and WhisperX.

    Features:
    - Automatic speech recognition
    - Word-level timestamp alignment
    - Multi-language support
    """

    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, config: Config):
        """
        Initialize lyrics recognizer.

        Args:
            config: Application configuration
        """
        self.config = config
        self.model_size = config.whisper_model
        self.language = config.lyrics_language
        self.device = get_device(config.use_gpu, config.gpu_device)
        self.model = None
        self.align_model = None

    def load_model(self) -> None:
        """Load Whisper model."""
        if self.model is not None:
            return

        logger.info(f"Loading Whisper model: {self.model_size}")

        try:
            import whisperx

            compute_type = "float16" if "cuda" in self.device else "int8"

            self.model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type=compute_type
            )

            logger.info(f"Whisper model loaded on {self.device}")

        except ImportError as e:
            logger.error("WhisperX not installed. Install with: pip install whisperx")
            raise ImportError("WhisperX is required for lyrics recognition") from e

    def unload_model(self) -> None:
        """Unload models to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.align_model is not None:
            del self.align_model
            self.align_model = None

        clear_gpu_memory()
        logger.info("Whisper models unloaded")

    def recognize(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[LyricEvent]:
        """
        Recognize lyrics from audio and get word-level timestamps.

        Args:
            audio_path: Path to audio file (preferably isolated vocals)
            progress_callback: Optional progress callback

        Returns:
            List of LyricEvent objects with timestamps
        """
        import whisperx

        self.load_model()

        if progress_callback:
            progress_callback(0.0, "Loading audio...")

        # Load audio
        logger.info(f"Recognizing lyrics: {audio_path}")
        audio = whisperx.load_audio(audio_path)

        if progress_callback:
            progress_callback(0.2, "Transcribing...")

        # Transcribe
        result = self.model.transcribe(
            audio,
            batch_size=16,
            language=self.language
        )

        detected_language = result.get("language", "en")
        logger.info(f"Detected language: {detected_language}")

        if progress_callback:
            progress_callback(0.5, "Aligning words...")

        # Load alignment model for detected language
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Could not load alignment model: {e}")
            # Fall back to segment-level timestamps
            return self._extract_segment_lyrics(result)

        # Align
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False
        )

        if progress_callback:
            progress_callback(0.9, "Processing lyrics...")

        # Extract lyrics
        lyrics = self._extract_word_lyrics(aligned)

        if progress_callback:
            progress_callback(1.0, f"Found {len(lyrics)} words")

        logger.info(f"Recognized {len(lyrics)} words/syllables")
        return lyrics

    def _extract_word_lyrics(self, aligned_result: dict) -> List[LyricEvent]:
        """Extract word-level lyrics from aligned result."""
        lyrics = []

        for segment in aligned_result.get("segments", []):
            for word_info in segment.get("words", []):
                word = word_info.get("word", "").strip()
                start = word_info.get("start", 0)
                end = word_info.get("end", start + 0.1)
                score = word_info.get("score", 1.0)

                if word:
                    lyrics.append(LyricEvent(
                        text=word,
                        start_time=float(start),
                        end_time=float(end),
                        confidence=float(score) if score else 1.0
                    ))

        # Sort by start time
        lyrics.sort(key=lambda l: l.start_time)

        return lyrics

    def _extract_segment_lyrics(self, result: dict) -> List[LyricEvent]:
        """Extract segment-level lyrics (fallback)."""
        lyrics = []

        for segment in result.get("segments", []):
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            end = segment.get("end", start + 1)

            if text:
                # Split into words
                words = text.split()
                duration = (end - start) / len(words) if words else end - start

                for i, word in enumerate(words):
                    word_start = start + i * duration
                    word_end = word_start + duration

                    lyrics.append(LyricEvent(
                        text=word,
                        start_time=float(word_start),
                        end_time=float(word_end),
                        confidence=0.8  # Lower confidence for segment-level
                    ))

        return lyrics

    def get_full_text(self, lyrics: List[LyricEvent]) -> str:
        """Get full lyrics text from events."""
        return " ".join(lyric.text for lyric in lyrics)
