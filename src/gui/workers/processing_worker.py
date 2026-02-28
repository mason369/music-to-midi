"""
处理工作线程 - 后台执行处理任务
"""
import logging
from PyQt6.QtCore import QThread, pyqtSignal

from src.models.data_models import Config, ProcessingProgress, ProcessingResult
from src.core.pipeline import MusicToMidiPipeline

logger = logging.getLogger(__name__)


class ProcessingWorker(QThread):
    """
    音频处理的后台工作线程

    在独立线程中运行处理流水线，避免阻塞UI
    """

    # 信号
    progress_updated = pyqtSignal(ProcessingProgress)
    processing_finished = pyqtSignal(ProcessingResult)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        audio_path: str,
        output_dir: str,
        config: Config,
        parent=None,
    ):
        """
        初始化工作线程

        参数:
            audio_path: 输入音频文件路径
            output_dir: 输出目录
            config: 应用配置
            parent: 父QObject
        """
        super().__init__(parent)

        self.audio_path = audio_path
        self.output_dir = output_dir
        self.config = config
        self.pipeline = MusicToMidiPipeline(config)

    def run(self):
        """在后台线程中执行处理"""
        try:
            logger.info(f"工作线程启动: {self.audio_path}")

            result = self.pipeline.process(
                self.audio_path,
                self.output_dir,
                self._on_progress,
            )

            self.processing_finished.emit(result)
            logger.info("工作线程成功完成")

        except InterruptedError:
            logger.info("工作线程已取消")
            self.error_occurred.emit("处理已取消")

        except Exception as e:
            logger.error(f"工作线程错误: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

        finally:
            # 确保 GPU 资源被清理
            try:
                if self.pipeline and self.pipeline.yourmt3_transcriber:
                    self.pipeline.yourmt3_transcriber.unload_model()
            except Exception as e:
                logger.warning(f"工作线程清理模型失败: {e}")
            try:
                from src.utils.gpu_utils import clear_gpu_memory
                clear_gpu_memory()
            except Exception as e:
                logger.warning(f"工作线程清理GPU内存失败: {e}")
            self.pipeline = None

    def _on_progress(self, progress: ProcessingProgress):
        """处理来自流水线的进度更新"""
        self.progress_updated.emit(progress)

    def cancel(self):
        """取消处理"""
        if self.pipeline:
            self.pipeline.cancel()
