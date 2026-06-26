"""
音乐转MIDI应用程序入口
"""
import sys
import os
import logging
import multiprocessing
import warnings
import argparse
import json
import traceback
import types
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from pathlib import Path

from src import __version__
from src.utils.runtime_paths import bootstrap_runtime_environment, get_logs_dir
from src.utils.warnings_filter import ensure_standard_streams

# 在导入其他模块之前抑制第三方库的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TensorFlow 所有日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用 oneDNN 警告
os.environ['ABSL_MIN_LOG_LEVEL'] = '2'    # 抑制 absl 日志

# PyInstaller windowed/portable 模式下标准流可能为 None，先补成安全可写流
ensure_standard_streams()

# 预先注入 bundled ffmpeg/bin 到 PATH，供 librosa/audioread/subprocess 使用
bootstrap_runtime_environment()

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


def _prepare_torch_runtime_before_pyqt() -> None:
    """Prepare torch DLLs before importing PyQt on GUI startup."""
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
            logging.getLogger(__name__).debug("Windows torch DLL 预加载失败: %s", e)

    # 在 PyQt6 之前预加载 torch，避免 PyQt6 DLL 与 torch DLL 冲突（WinError 1114）
    try:
        import torch  # noqa: F401
        # torchaudio 2.9+ 默认使用 torchcodec 后端，但该包未安装时会报错
        # 老版本可显式切到 soundfile；新版本 dispatcher 模式下该调用会变成 no-op 并给出弃用告警
        import torchaudio

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*set_audio_backend has been deprecated.*",
                category=UserWarning,
            )
            torchaudio.set_audio_backend("soundfile")
    except Exception as e:
        logging.getLogger(__name__).debug("torch 预加载失败（将在需要时重试）: %s", e)


def _is_4k_display() -> bool:
    """检测主显示器是否为 4K（>= 3840x2160）。

    在 QApplication 创建之前调用。Windows 上使用 EnumDisplaySettingsW
    读取当前显示模式，避免提前设置进程 DPI awareness，防止 Qt 初始化时
    再设置 DPI awareness 触发 Access denied 警告。
    """
    import platform
    try:
        if platform.system() == "Windows":
            import ctypes
            from ctypes import wintypes

            CCHDEVICENAME = 32
            CCHFORMNAME = 32
            ENUM_CURRENT_SETTINGS = -1

            class DEVMODEW(ctypes.Structure):
                _fields_ = [
                    ("dmDeviceName", wintypes.WCHAR * CCHDEVICENAME),
                    ("dmSpecVersion", wintypes.WORD),
                    ("dmDriverVersion", wintypes.WORD),
                    ("dmSize", wintypes.WORD),
                    ("dmDriverExtra", wintypes.WORD),
                    ("dmFields", wintypes.DWORD),
                    ("dmOrientation", wintypes.SHORT),
                    ("dmPaperSize", wintypes.SHORT),
                    ("dmPaperLength", wintypes.SHORT),
                    ("dmPaperWidth", wintypes.SHORT),
                    ("dmScale", wintypes.SHORT),
                    ("dmCopies", wintypes.SHORT),
                    ("dmDefaultSource", wintypes.SHORT),
                    ("dmPrintQuality", wintypes.SHORT),
                    ("dmColor", wintypes.SHORT),
                    ("dmDuplex", wintypes.SHORT),
                    ("dmYResolution", wintypes.SHORT),
                    ("dmTTOption", wintypes.SHORT),
                    ("dmCollate", wintypes.SHORT),
                    ("dmFormName", wintypes.WCHAR * CCHFORMNAME),
                    ("dmLogPixels", wintypes.WORD),
                    ("dmBitsPerPel", wintypes.DWORD),
                    ("dmPelsWidth", wintypes.DWORD),
                    ("dmPelsHeight", wintypes.DWORD),
                    ("dmDisplayFlags", wintypes.DWORD),
                    ("dmDisplayFrequency", wintypes.DWORD),
                    ("dmICMMethod", wintypes.DWORD),
                    ("dmICMIntent", wintypes.DWORD),
                    ("dmMediaType", wintypes.DWORD),
                    ("dmDitherType", wintypes.DWORD),
                    ("dmReserved1", wintypes.DWORD),
                    ("dmReserved2", wintypes.DWORD),
                    ("dmPanningWidth", wintypes.DWORD),
                    ("dmPanningHeight", wintypes.DWORD),
                ]

            devmode = DEVMODEW()
            devmode.dmSize = ctypes.sizeof(DEVMODEW)
            if ctypes.windll.user32.EnumDisplaySettingsW(None, ENUM_CURRENT_SETTINGS, ctypes.byref(devmode)):
                return devmode.dmPelsWidth >= 3840 and devmode.dmPelsHeight >= 2160
    except Exception as exc:
        logging.getLogger(__name__).debug("4K 显示器检测失败: %s", exc)
    return False


def _run_self_test(
    transcriber_cls=None,
    success_message: str = "SELF-TEST OK: YourMT3+ available",
    load_model: bool = True,
) -> int:
    """运行无界面自检，供发布烟测和终端诊断使用。"""
    setup_chinese_environment()
    logger = setup_logger(log_dir=str(get_logs_dir()), level=logging.DEBUG)
    from src.i18n.translator import t

    try:
        if transcriber_cls is None:
            from src.core.yourmt3_transcriber import YourMT3Transcriber as transcriber_cls

        logger.info(t("startup.portable_self_test_starting"))
        if not transcriber_cls.is_available():
            reason_getter = getattr(transcriber_cls, "get_unavailable_reason", None)
            reason = reason_getter() if callable(reason_getter) else t("startup.yourmt3_unavailable")
            logger.error(t("startup.portable_self_test_failed", reason=reason))
            print(reason)
            return 1

        transcriber = None
        try:
            from src.models.data_models import Config

            transcriber = transcriber_cls(Config())
            load_model_fn = getattr(transcriber, "_load_model", None)
            if load_model and callable(load_model_fn):
                logger.info(t("startup.self_test_loading_yourmt3_model"))
                load_model_fn()
        finally:
            if transcriber is not None:
                unload_model = getattr(transcriber, "unload_model", None)
                if callable(unload_model):
                    unload_model()

        logger.info(t("startup.portable_self_test_passed"))
        print(success_message)
        return 0
    except Exception as e:
        logger.error(t("startup.portable_self_test_exception", error=e), exc_info=True)
        print(f"SELF-TEST FAILED: {e}")
        return 1


@contextmanager
def _temporary_onnxruntime_stub():
    existing = sys.modules.get("onnxruntime")
    if existing is not None:
        yield
        return

    stub = types.ModuleType("onnxruntime")
    stub.__dict__.update(
        {
            "__version__": "0.0",
            "__path__": [],
            "get_available_providers": lambda: [],
            "SessionOptions": type("SessionOptions", (), {}),
            "InferenceSession": type("InferenceSession", (), {}),
        }
    )
    stub.__spec__ = ModuleSpec("onnxruntime", loader=None, is_package=True)
    sys.modules["onnxruntime"] = stub
    try:
        yield
    finally:
        if sys.modules.get("onnxruntime") is stub:
            sys.modules.pop("onnxruntime", None)


def _run_miros_worker(argv=None) -> int:
    """Run bundled MIROS transcription without opening the GUI."""
    parser = argparse.ArgumentParser(description="Internal MIROS worker")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--status-json")
    args = parser.parse_args(argv)

    def write_status(payload) -> None:
        if not args.status_json:
            return
        status_path = Path(args.status_json)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    repo_path_added = False
    repo_dir = os.getcwd()
    try:
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
            repo_path_added = True

        # torchmetrics imports optional DNSMOS audio metrics during Lightning import.
        # MIROS inference does not use those metrics, so keep onnxruntime isolated
        # while importing the upstream transcribe module.
        with _temporary_onnxruntime_stub():
            from transcribe import transcribe

        input_path = Path(args.input)
        if not input_path.is_file():
            raise FileNotFoundError(f"MIROS input audio does not exist: {input_path}")

        transcribe(str(input_path), args.output)
        output_path = Path(args.output)
        write_status(
            {
                "ok": True,
                "output": str(output_path),
                "output_exists": output_path.exists(),
                "output_size": output_path.stat().st_size if output_path.exists() else None,
            }
        )
        return 0
    except Exception as exc:
        failure_status = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            write_status(failure_status)
        except Exception as status_exc:
            print(
                f"MIROS worker failed: {exc}; also failed to write status: {status_exc}",
                file=sys.stderr,
            )
            logging.getLogger(__name__).error(
                "MIROS worker failed and status write failed: %s",
                status_exc,
                exc_info=True,
            )
        print(f"MIROS worker failed: {exc}", file=sys.stderr)
        logging.getLogger(__name__).error("MIROS worker failed: %s", exc, exc_info=True)
        return 1
    finally:
        if repo_path_added:
            try:
                sys.path.remove(repo_dir)
            except ValueError:
                pass


def main():
    """主入口函数"""
    multiprocessing.freeze_support()
    if "--miros-worker" in sys.argv:
        worker_index = sys.argv.index("--miros-worker")
        exit_code = _run_miros_worker(sys.argv[worker_index + 1:])
        if getattr(sys, "frozen", False):
            os._exit(exit_code)
            return
        sys.exit(exit_code)

    # Set console encoding before any localized CLI output.
    setup_chinese_environment()

    if "-h" in sys.argv or "--help" in sys.argv:
        from src.i18n.translator import t

        print(
            f"{t('cli.usage')}: python -m src.main [--self-test] [--self-test-no-load] [--self-test-miros]\n"
            "\n"
            f"{t('cli.options')}:\n"
            f"  -h, --help          {t('cli.help')}\n"
            f"  --self-test         {t('cli.self_test')}\n"
            f"  --self-test-no-load {t('cli.self_test_no_load')}\n"
            f"  --self-test-miros   {t('cli.self_test_miros')}\n"
            f"  --miros-worker      {t('cli.miros_worker')}"
        )
        sys.exit(0)
    if "--self-test-no-load" in sys.argv:
        sys.exit(
            _run_self_test(
                success_message="SELF-TEST OK: YourMT3+ available without model load",
                load_model=False,
            )
        )
    if "--self-test" in sys.argv:
        sys.exit(_run_self_test())
    if "--self-test-miros" in sys.argv:
        from src.core.miros_transcriber import MirosTranscriber

        sys.exit(
            _run_self_test(
                transcriber_cls=MirosTranscriber,
                success_message="SELF-TEST OK: MIROS available",
                load_model=False,
            )
        )

    _prepare_torch_runtime_before_pyqt()

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

    from src.i18n.translator import t

    logger.info(t("startup.application_starting"))

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
            logger.info(t("startup.detected_4k_passthrough"))
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
        else:
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.Floor
            )

        # 创建应用程序
        app = QApplication(sys.argv)
        app.setApplicationName(t("app.name"))
        app.setApplicationVersion(__version__)
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
                    logger.info(t("startup.app_font_selected", font=family))
                    break
            ui_font = QFont(ui_font_family, 10)
            app.setFont(ui_font)

            # 设置字体替换：对于无法渲染的字符，回退到Emoji字体
            for emoji_font in ("Noto Color Emoji", "Symbola"):
                if emoji_font in available:
                    QFont.insertSubstitutions(ui_font_family, [emoji_font])
                    QFont.insertSubstitutions("sans-serif", [emoji_font])
                    logger.info(t("startup.emoji_fallback_selected", font=emoji_font))
                    break

        # 创建并显示主窗口
        config = Config()
        window = MainWindow(config)
        window.show()

        logger.info(t("startup.window_shown"))

        # 运行事件循环
        sys.exit(app.exec())

    except ImportError as e:
        logger.error(t("startup.import_pyqt_failed", error=e))
        print(t("startup.install_pyqt_hint"))
        sys.exit(1)

    except Exception as e:
        logger.error(t("startup.application_error", error=e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
