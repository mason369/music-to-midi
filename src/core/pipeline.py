"""
Main processing pipeline orchestrating all modules.
"""
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional, Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor

from src.models.data_models import (
    Config, Project, ProcessingResult, ProcessingProgress,
    ProcessingStage, Track, TrackType, NoteEvent, LyricEvent, BeatInfo
)
from src.core.separator import SourceSeparator
from src.core.transcriber import AudioTranscriber
from src.core.lyrics_recognizer import LyricsRecognizer
from src.core.beat_detector import BeatDetector
from src.core.midi_generator import MidiGenerator
from src.utils.gpu_utils import clear_gpu_memory

logger = logging.getLogger(__name__)


class MusicToMidiPipeline:
    """
    Main processing pipeline for music to MIDI conversion.

    Orchestrates:
    1. Source separation (Demucs)
    2. Beat detection (librosa)
    3. Audio transcription (Basic Pitch)
    4. Lyrics recognition (Whisper)
    5. MIDI generation (mido)
    """

    def __init__(self, config: Config):
        """
        Initialize pipeline.

        Args:
            config: Application configuration
        """
        self.config = config
        self.separator = SourceSeparator(config)
        self.transcriber = AudioTranscriber(config)
        self.lyrics_recognizer = LyricsRecognizer(config)
        self.beat_detector = BeatDetector(config)
        self.midi_generator = MidiGenerator(config)

        self._cancelled = False
        self._progress_callback: Optional[Callable[[ProcessingProgress], None]] = None

    def set_progress_callback(
        self,
        callback: Callable[[ProcessingProgress], None]
    ) -> None:
        """Set progress callback function."""
        self._progress_callback = callback

    def cancel(self) -> None:
        """Cancel ongoing processing."""
        self._cancelled = True
        logger.info("Processing cancelled")

    def _report_progress(
        self,
        stage: ProcessingStage,
        stage_progress: float,
        overall_progress: float,
        message: str
    ) -> None:
        """Report progress to callback."""
        if self._progress_callback:
            progress = ProcessingProgress(
                stage=stage,
                stage_progress=stage_progress,
                overall_progress=overall_progress,
                message=message
            )
            self._progress_callback(progress)

    def process(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> ProcessingResult:
        """
        Process audio file through the complete pipeline.

        Args:
            audio_path: Path to input audio file
            output_dir: Output directory
            progress_callback: Optional progress callback

        Returns:
            ProcessingResult with all outputs
        """
        self._cancelled = False
        self._progress_callback = progress_callback
        start_time = time.time()

        input_name = Path(audio_path).stem
        stems_dir = os.path.join(output_dir, "stems")

        logger.info(f"Starting pipeline for: {audio_path}")

        # Stage 1: Preprocessing
        self._report_progress(
            ProcessingStage.PREPROCESSING,
            0.0, 0.05,
            "Preparing audio..."
        )

        # Stage 2: Source separation
        self._report_progress(
            ProcessingStage.SEPARATION,
            0.0, 0.10,
            "Separating audio sources..."
        )

        if self._cancelled:
            raise InterruptedError("Processing cancelled")

        stem_paths = self.separator.separate(
            audio_path,
            stems_dir,
            lambda p, m: self._report_progress(
                ProcessingStage.SEPARATION,
                p, 0.10 + p * 0.25,
                m
            )
        )

        # Free separator memory
        self.separator.unload_model()
        clear_gpu_memory()

        # Stage 3: Beat detection (parallel with transcription start)
        self._report_progress(
            ProcessingStage.TRANSCRIPTION,
            0.0, 0.35,
            "Detecting tempo..."
        )

        if self._cancelled:
            raise InterruptedError("Processing cancelled")

        beat_info = self.beat_detector.detect(
            audio_path,
            lambda p, m: self._report_progress(
                ProcessingStage.TRANSCRIPTION,
                p * 0.2, 0.35 + p * 0.05,
                m
            )
        )

        # Stage 4: Transcribe all stems
        tracks: Dict[TrackType, List[NoteEvent]] = {}

        stem_types = [
            (TrackType.DRUMS, "drums"),
            (TrackType.BASS, "bass"),
            (TrackType.VOCALS, "vocals"),
            (TrackType.OTHER, "other")
        ]

        for i, (track_type, stem_name) in enumerate(stem_types):
            if self._cancelled:
                raise InterruptedError("Processing cancelled")

            if stem_name in stem_paths:
                base_progress = 0.40 + (i * 0.10)

                self._report_progress(
                    ProcessingStage.TRANSCRIPTION,
                    i / 4, base_progress,
                    f"Transcribing {stem_name}..."
                )

                notes = self.transcriber.transcribe(
                    stem_paths[stem_name],
                    track_type,
                    lambda p, m: self._report_progress(
                        ProcessingStage.TRANSCRIPTION,
                        (i + p) / 4,
                        base_progress + p * 0.10,
                        m
                    )
                )

                tracks[track_type] = notes
                logger.info(f"{stem_name}: {len(notes)} notes")

        # Stage 5: Lyrics recognition
        lyrics: List[LyricEvent] = []

        if "vocals" in stem_paths and self.config.embed_lyrics:
            if self._cancelled:
                raise InterruptedError("Processing cancelled")

            self._report_progress(
                ProcessingStage.LYRICS,
                0.0, 0.80,
                "Recognizing lyrics..."
            )

            try:
                lyrics = self.lyrics_recognizer.recognize(
                    stem_paths["vocals"],
                    lambda p, m: self._report_progress(
                        ProcessingStage.LYRICS,
                        p, 0.80 + p * 0.10,
                        m
                    )
                )
            except Exception as e:
                logger.warning(f"Lyrics recognition failed: {e}")

            # Free recognizer memory
            self.lyrics_recognizer.unload_model()
            clear_gpu_memory()

        # Stage 6: MIDI synthesis
        if self._cancelled:
            raise InterruptedError("Processing cancelled")

        self._report_progress(
            ProcessingStage.SYNTHESIS,
            0.0, 0.90,
            "Generating MIDI..."
        )

        midi_path = os.path.join(output_dir, f"{input_name}.mid")

        self.midi_generator.generate(
            tracks=tracks,
            lyrics=lyrics,
            tempo=beat_info.bpm,
            output_path=midi_path,
            embed_lyrics=self.config.embed_lyrics
        )

        # Export LRC if requested
        lrc_path = None
        if lyrics and self.config.export_lrc:
            lrc_path = os.path.join(output_dir, f"{input_name}.lrc")
            self.midi_generator.export_lrc(
                lyrics=lyrics,
                output_path=lrc_path,
                title=input_name
            )

        # Complete
        processing_time = time.time() - start_time

        self._report_progress(
            ProcessingStage.COMPLETE,
            1.0, 1.0,
            f"Complete! ({processing_time:.1f}s)"
        )

        # Build result
        result_tracks = []
        for track_type, notes in tracks.items():
            stem_name = track_type.value
            result_tracks.append(Track(
                type=track_type,
                audio_path=stem_paths.get(stem_name, ""),
                notes=notes
            ))

        result = ProcessingResult(
            midi_path=midi_path,
            lrc_path=lrc_path,
            tracks=result_tracks,
            beat_info=beat_info,
            lyrics=lyrics,
            processing_time=processing_time
        )

        logger.info(f"Pipeline complete: {processing_time:.1f}s")

        return result

    async def process_async(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> ProcessingResult:
        """
        Async version of process method.

        Args:
            audio_path: Path to input audio file
            output_dir: Output directory
            progress_callback: Optional progress callback

        Returns:
            ProcessingResult with all outputs
        """
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                lambda: self.process(audio_path, output_dir, progress_callback)
            )

        return result
