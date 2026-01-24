"""
Processing worker thread for background execution.
"""
import logging
from PyQt6.QtCore import QThread, pyqtSignal

from src.models.data_models import Config, ProcessingProgress, ProcessingResult
from src.core.pipeline import MusicToMidiPipeline

logger = logging.getLogger(__name__)


class ProcessingWorker(QThread):
    """
    Background worker thread for audio processing.

    Runs the processing pipeline in a separate thread to avoid blocking the UI.
    """

    # Signals
    progress_updated = pyqtSignal(ProcessingProgress)
    processing_finished = pyqtSignal(ProcessingResult)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        audio_path: str,
        output_dir: str,
        config: Config,
        parent=None
    ):
        """
        Initialize worker.

        Args:
            audio_path: Path to input audio file
            output_dir: Output directory
            config: Application configuration
            parent: Parent QObject
        """
        super().__init__(parent)

        self.audio_path = audio_path
        self.output_dir = output_dir
        self.config = config
        self.pipeline = MusicToMidiPipeline(config)

    def run(self):
        """Execute processing in background thread."""
        try:
            logger.info(f"Worker starting: {self.audio_path}")

            result = self.pipeline.process(
                self.audio_path,
                self.output_dir,
                self._on_progress
            )

            self.processing_finished.emit(result)
            logger.info("Worker finished successfully")

        except InterruptedError:
            logger.info("Worker cancelled")
            self.error_occurred.emit("Processing cancelled")

        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

    def _on_progress(self, progress: ProcessingProgress):
        """Handle progress update from pipeline."""
        self.progress_updated.emit(progress)

    def cancel(self):
        """Cancel processing."""
        if self.pipeline:
            self.pipeline.cancel()
