"""
音乐转MIDI应用程序入口
"""
import sys
import os
import logging
import warnings
from pathlib import Path

from src.utils.runtime_paths import bootstrap_runtime_environment, get_logs_dir

# 在导入其他模块之前抑制第三方库的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TensorFlow 所有日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 警告
os.environ['ABSL_MIN_LOG_LEVEL'] = '2'    # 抑制 absl 日志

# 预先注入 bundled ffmpeg/bin 到 PATH，供 librosa/audioread/subprocess 使用
bootstrap_runtime_environment()

# 动态设置 OpenMP/MKL 线程数（必须在 import torch 之前，否则无效）
# 使用全部物理核心，最大化 CPU 利用率
try:
    import psutil as _psutil
    _phys = _psutil.cpu_count(logical=False) or (os.cpu_count() or 4)
except Exception:
    _phys = os.cpu_count() or 4
_omp_threads = str(max(2, _phys))
os.environ.setdefault('OMP_NUM_THREADS', _omp_threads)
os.environ.setdefault('MKL_NUM_THREADS', _omp_threads)
del _omp_threads, _phys
try:
    del _psutil
except NameError:
    pass

# 修复 Windows 特殊路径（中文用户名、空格、括号等）下 PyTorch DLL 加载失败的问题
# 必须在任何 import torch 之前执行
import platform as _plat
if _plat.system() == "Windows":
    try:
        import importlib.util as _ilu
        import re as _re
        _spec = _ilu.find_spec("torch")
        if _spec and _spec.origin:
            _torch_lib = os.path.join(os.path.dirname(_spec.origin), "lib")
            if os.path.isdir(_torch_lib) and _re.search(r'[\s\(\)\[\]{}]|[^\x00-\x7F]', _torch_lib):
                import ctypes as _ct
                import glob as _gl
                # 获取 8.3 短路径名
                _buf = _ct.create_unicode_buffer(512)
                _ret = _ct.windll.kernel32.GetShortPathNameW(_torch_lib, _buf, 512)
                _short = _buf.value if 0 < _ret < 512 else _torch_lib
                # 注入 PATH
                _path = os.environ.get("PATH", "")
                if _short not in _path:
                    os.environ["PATH"] = _short + os.pathsep + _path
                os.add_dll_directory(_short)
                # 预加载 VC++ 运行时
                for _vcrt in ("vcruntime140.dll", "msvcp140.dll", "vcruntime140_1.dll"):
                    try:
                        _ct.CDLL(_vcrt)
                    except OSError:
                        pass
                # 用短路径预加载所有 torch DLL
                _k32 = _ct.WinDLL("kernel32.dll", use_last_error=True)
                _k32.LoadLibraryW.restype = _ct.c_void_p
                for _dll in sorted(_gl.glob(os.path.join(_short, "*.dll"))):
                    try:
                        _k32.LoadLibraryW(_dll)
                    except Exception:
                        pass
    except Exception as e:
        # DLL 预加载失败不阻塞启动，但记录诊断信息
        import logging as _log
        _log.getLogger(__name__).debug("Windows torch DLL 预加载失败: %s", e)
del _plat

# 在 PyQt6 之前预加载 torch，避免 PyQt6 DLL 与 torch DLL 冲突（WinError 1114）
try:
    import torch  # noqa: F401
    # 使用全部物理核心，避免默认 1 线程限制
    try:
        import psutil as _psutil
        _cores = _psutil.cpu_count(logical=False) or (os.cpu_count() or 4)
        del _psutil
    except ImportError:
        _cores = os.cpu_count() or 4
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', str(_cores))))
    del _cores
    # 固定输入尺寸下启用 cuDNN 自动调优（约提升 10-30% 推理速度）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    # torchaudio 2.9+ 默认使用 torchcodec 后端，但该包未安装时会报错
    # 强制使用 soundfile 后端（已在 requirements.txt 中包含）
    import torchaudio
    torchaudio.set_audio_backend("soundfile")
except Exception as e:
    import logging as _log
    _log.getLogger(__name__).debug("torch 预加载失败（将在需要时重试）: %s", e)

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
logging.getLogger().setLevel(logging.ERROR)  # 根 logger

from src.utils.logger import setup_logger
from src.utils.warnings_filter import setup_chinese_environment


def _is_4k_display() -> bool:
    """检测主显示器是否为 4K（>= 3840x2160）。

    在 QApplication 创建之前调用，使用平台原生 API 获取物理分辨率。
    """
    import platform
    try:
        if platform.system() == "Windows":
            import ctypes
            # 设置 DPI 感知以获取真实物理分辨率（而非缩放后的逻辑分辨率）
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass
            w = ctypes.windll.user32.GetSystemMetrics(0)  # SM_CXSCREEN
            h = ctypes.windll.user32.GetSystemMetrics(1)  # SM_CYSCREEN
            return w >= 3840 and h >= 2160
    except Exception:
        pass
    return False


def main():
    """主入口函数"""
    # 设置中文环境（抑制警告 + 修补输出）
    setup_chinese_environment()

    # 设置日志
    logger = setup_logger(log_dir=str(get_logs_dir()), level=logging.DEBUG)

    # 设置所有 src.* 模块的日志级别为 DEBUG，这样子模块也会输出详细日志
    src_logger = logging.getLogger("src")
    src_logger.setLevel(logging.DEBUG)
    # 禁止向 root logger 传播，避免第三方库注入的处理器产生重复输出
    src_logger.propagate = False
    # 为 src logger 添加相同的处理器（如果没有的话）
    for handler in logger.handlers:
        if handler not in src_logger.handlers:
            src_logger.addHandler(handler)

    logger.info("正在启动音乐转MIDI应用程序")

    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont, QFontDatabase

        from src.gui.main_window import MainWindow
        from src.models.data_models import Config

        # 高DPI缩放策略：
        #   4K 屏幕使用 PassThrough 保留精确缩放比例，避免 Floor 将 1.5x 截断为 1x 导致界面过小
        #   非 4K 屏幕使用 Floor 避免界面过大导致文字被遮挡
        if _is_4k_display():
            logger.info("检测到 4K 显示器，使用 PassThrough DPI 策略")
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
        else:
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.Floor
            )

        # 创建应用程序
        app = QApplication(sys.argv)
        app.setApplicationName("音乐转MIDI")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("mason369")

        # 应用Fusion样式
        app.setStyle("Fusion")

        # 设置中文字体和Emoji字体（确保中文及图标正确显示）
        import platform as _platform
        if _platform.system() != "Windows":
            available = QFontDatabase.families()
            # 主字体：优先支持CJK中文的字体
            ui_font_family = "sans-serif"
            for family in ("Noto Sans CJK SC", "WenQuanYi Micro Hei",
                           "WenQuanYi Zen Hei", "Ubuntu"):
                if family in available:
                    ui_font_family = family
                    logger.info(f"已设置应用字体: {family}")
                    break
            ui_font = QFont(ui_font_family, 10)
            app.setFont(ui_font)

            # 设置字体替换：对于无法渲染的字符，回退到Emoji字体
            for emoji_font in ("Noto Color Emoji", "Symbola"):
                if emoji_font in available:
                    QFont.insertSubstitutions(ui_font_family, [emoji_font])
                    QFont.insertSubstitutions("sans-serif", [emoji_font])
                    logger.info(f"已设置Emoji回退字体: {emoji_font}")
                    break

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
