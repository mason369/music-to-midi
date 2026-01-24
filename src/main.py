"""
音乐转MIDI应用程序入口
"""
import sys
import os
import logging
import warnings
from pathlib import Path

# 在导入其他模块之前抑制第三方库的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TensorFlow 所有日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 警告
os.environ['ABSL_MIN_LOG_LEVEL'] = '2'    # 抑制 absl 日志

# 抑制 Python 警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='tensorflow')
warnings.filterwarnings('ignore', module='keras')
warnings.filterwarnings('ignore', module='basic_pitch')

# 抑制特定库的日志
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('root').setLevel(logging.ERROR)

from src.utils.logger import setup_logger
from src.utils.warnings_filter import setup_chinese_environment


def main():
    """主入口函数"""
    # 设置中文环境（抑制警告 + 修补输出）
    setup_chinese_environment()

    # 设置日志
    log_dir = Path.home() / ".music-to-midi" / "logs"
    logger = setup_logger(log_dir=str(log_dir), level=logging.DEBUG)

    # 设置所有 src.* 模块的日志级别为 DEBUG，这样子模块也会输出详细日志
    logging.getLogger("src").setLevel(logging.DEBUG)
    # 为 src logger 添加相同的处理器（如果没有的话会继承 root）
    for handler in logger.handlers:
        src_logger = logging.getLogger("src")
        if handler not in src_logger.handlers:
            src_logger.addHandler(handler)

    logger.info("正在启动音乐转MIDI应用程序")

    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt

        from src.gui.main_window import MainWindow
        from src.models.data_models import Config

        # 启用高DPI缩放
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        # 创建应用程序
        app = QApplication(sys.argv)
        app.setApplicationName("音乐转MIDI")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("mason369")

        # 应用Fusion样式
        app.setStyle("Fusion")

        # 创建并显示主窗口
        config = Config()
        window = MainWindow(config)
        window.show()

        logger.info("应用程序窗口已显示")

        # 运行事件循环
        sys.exit(app.exec())

    except ImportError as e:
        logger.error(f"导入PyQt6失败: {e}")
        print("错误: 需要PyQt6。请使用以下命令安装: pip install PyQt6")
        sys.exit(1)

    except Exception as e:
        logger.error(f"应用程序错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
