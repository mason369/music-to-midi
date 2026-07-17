"""Desktop background worker for WAV separation without automatic MIDI."""

from __future__ import annotations

import logging

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.separation_service import AudioSeparationService, SeparationResult
from src.i18n.translator import Translator
from src.models.data_models import Config, ProcessingProgress
from src.utils.gpu_utils import clear_gpu_memory

logger = logging.getLogger(__name__)

__all__ = ["SeparationResult", "SeparationWorker"]


class SeparationWorker(QThread):
    """Run the shared separation-only service off the desktop UI thread."""

    progress_updated = pyqtSignal(ProcessingProgress)
    separation_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        audio_path: str,
        output_dir: str,
        config: Config,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.audio_path = str(audio_path)
        self.output_dir = str(output_dir)
        self.config = config
        self._translator = Translator(config.language)
        self._cancelled = False
        self._service: AudioSeparationService | None = None

    def run(self) -> None:
        try:
            service = AudioSeparationService(
                self.config,
                progress_callback=self.progress_updated.emit,
                cancel_check=lambda: self._cancelled,
            )
            self._service = service
            result = service.process(self.audio_path, self.output_dir)
            self.separation_finished.emit(result)
        except InterruptedError:
            logger.info("分离工作线程已取消")
            self.error_occurred.emit(self._translator.t("status.cancelled"))
        except Exception as exc:
            logger.error("分离工作线程错误: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))
        finally:
            self._service = None
            try:
                clear_gpu_memory()
            except Exception as exc:
                logger.warning("分离工作线程清理 GPU 内存失败: %s", exc)

    def cancel(self) -> None:
        self._cancelled = True
        if self._service is not None:
            self._service.cancel()
