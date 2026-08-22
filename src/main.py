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


def _is_web_api_runtime() -> bool:
    return "--web-api" in sys.argv or (
        getattr(sys, "frozen", False)
        and Path(sys.executable).stem.casefold() in {"musictomidibackend", "musictomidibackendxpu"}
    )


_IS_WEB_API_RUNTIME = _is_web_api_runtime()

if __name__ == "__main__":
    # Source runs must be stopped before importing GUI/runtime dependencies when
    # a global or foreign Python environment would leak incompatible packages.
    from src.utils.source_runtime import require_source_runtime_identity

    require_source_runtime_identity()

from src import __version__
from src.utils.midi_output import validate_midi_output
from src.utils.runtime_paths import (
    activate_audio_separator_runtime,
    bootstrap_runtime_environment,
    get_bundle_roots,
    get_logs_dir,
)
from src.utils.warnings_filter import ensure_standard_streams

# The API executable must preserve its configured log level and dependency
# warnings. These desktop-only noise controls used to run before entry-point
# dispatch, which made the backend's info/debug settings ineffective.
if not _IS_WEB_API_RUNTIME:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["ABSL_MIN_LOG_LEVEL"] = "2"

# PyInstaller windowed/portable 模式下标准流可能为 None，先补成安全可写流
ensure_standard_streams()

# 预先注入 bundled ffmpeg/bin 到 PATH，供 librosa/audioread/subprocess 使用
bootstrap_runtime_environment()

if not _IS_WEB_API_RUNTIME:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", module="tensorflow")
    warnings.filterwarnings("ignore", module="keras")
    warnings.filterwarnings("ignore", module="basic_pitch")

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("keras").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

from src.utils.logger import setup_logger
from src.utils.warnings_filter import setup_chinese_environment

_BUNDLED_VC_RUNTIME_HANDLES: list[object] = []


def _preload_bundled_windows_vc_runtime() -> None:
    """Load the bundle's coherent VC runtime before Qt can load its older copies."""
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return
    if _BUNDLED_VC_RUNTIME_HANDLES:
        return

    import ctypes

    bundle_roots = [root for root in get_bundle_roots() if root.is_dir()]
    runtime_root = next(
        (
            root
            for root in bundle_roots
            if all(
                (root / dll_name).is_file()
                for dll_name in (
                    "msvcp140.dll",
                    "vcruntime140.dll",
                    "vcruntime140_1.dll",
                )
            )
        ),
        None,
    )
    if runtime_root is None:
        searched = ", ".join(str(root) for root in bundle_roots) or "<none>"
        raise RuntimeError(
            "Portable bundle is missing its required Visual C++ runtime DLL set "
            f"(msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll); searched: {searched}"
        )

    for dll_name in (
        "msvcp140.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    ):
        dll_path = runtime_root / dll_name
        try:
            handle = ctypes.WinDLL(str(dll_path))
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load required bundled Visual C++ runtime: {dll_path}"
            ) from exc
        _BUNDLED_VC_RUNTIME_HANDLES.append(handle)


def _prepare_torch_runtime_before_pyqt() -> None:
    """Load every required desktop native runtime before importing PyQt.

    Importing ONNX Runtime after QtMultimedia or after a real ByteDance CUDA
    inference can fail on Windows with ``DLL initialization routine failed``.
    Keeping the native load order deterministic here lets the same long-lived
    GUI process run PyTorch piano backends and later run ONNX separation.
    """
    _preload_bundled_windows_vc_runtime()
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
                if os.path.isdir(_torch_lib) and _re.search(
                    r"[\s\(\)\[\]{}]|[^\x00-\x7F]", _torch_lib
                ):
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

        # 只预加载 torchaudio 的原生运行时。2.9 起已移除全局 backend
        # setter；项目转写器显式读取流水线生成的 WAV，不依赖可变全局状态。
        import torchaudio  # noqa: F401
    except Exception as e:
        if getattr(sys, "frozen", False):
            raise RuntimeError(
                "Portable torch/torchaudio runtime failed before PyQt startup"
            ) from e
        logging.getLogger(__name__).debug("torch 预加载失败（将在需要时重试）: %s", e)

    # audio-separator is a required desktop component, not an optional route.
    # Bind its ONNX Runtime before PyQt/QtMultimedia can load a conflicting
    # native dependency and before any CUDA model initializes cuDNN kernels.
    try:
        activate_audio_separator_runtime()
        import onnxruntime  # noqa: F401
        from audio_separator.separator import Separator  # noqa: F401
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Required desktop separation runtime failed before PyQt startup: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


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
            reason = (
                reason_getter() if callable(reason_getter) else t("startup.yourmt3_unavailable")
            )
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


def _run_gui_runtime_self_test() -> int:
    """Validate the frozen GUI/Qt/ONNX Runtime load order used by separation modes."""
    setup_chinese_environment()
    logger = setup_logger(log_dir=str(get_logs_dir()), level=logging.DEBUG)
    try:
        _prepare_torch_runtime_before_pyqt()
        from PyQt6.QtWidgets import QApplication  # noqa: F401

        import onnxruntime as ort
        from audio_separator.separator import Separator  # noqa: F401
        from src.utils.gpu_utils import (
            ensure_accelerator_runtime_compatibility,
            get_accelerator_type,
            get_device,
        )

        requested = os.environ.get("MUSIC_TO_MIDI_ACCELERATOR", "").strip().lower()
        detected = get_accelerator_type()
        accelerator = requested or detected
        if accelerator not in {"cuda", "xpu"}:
            raise RuntimeError(
                "GUI runtime self-test requires CUDA or Intel XPU, "
                f"but detected {detected!r} (requested={requested or '<auto>'})"
            )
        if requested and detected != requested:
            raise RuntimeError(
                "Packaged accelerator does not match MUSIC_TO_MIDI_ACCELERATOR: "
                f"requested={requested}, detected={detected}"
            )
        device = get_device(prefer_gpu=True, gpu_index=0)
        if not device.startswith(f"{accelerator}:"):
            raise RuntimeError(f"Expected {accelerator}:0 runtime device, got {device}")
        ensure_accelerator_runtime_compatibility(device)
        providers = ort.get_available_providers()
        expected_provider = (
            "CUDAExecutionProvider" if accelerator == "cuda" else "OpenVINOExecutionProvider"
        )
        if expected_provider not in providers:
            raise RuntimeError(
                f"Packaged ONNX Runtime did not expose {expected_provider}: " + ", ".join(providers)
            )
        logger.info(
            "GUI runtime self-test passed: accelerator=%s device=%s providers=%s",
            accelerator,
            device,
            providers,
        )
        print("SELF-TEST OK: GUI + Qt + ONNX Runtime " f"{accelerator.upper()} load order")
        return 0
    except Exception as exc:
        logger.error("GUI runtime self-test failed: %s", exc, exc_info=True)
        print(f"SELF-TEST FAILED: GUI runtime load order: {exc}")
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
    parser.add_argument("--events-jsonl")
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
        input_path = Path(args.input)
        if not input_path.is_file():
            raise FileNotFoundError(f"MIROS input audio does not exist: {input_path}")

        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
            repo_path_added = True

        # torchmetrics imports optional DNSMOS audio metrics during Lightning import.
        # MIROS inference does not use those metrics, so keep onnxruntime isolated
        # while importing the upstream transcribe module.
        with _temporary_onnxruntime_stub():
            from src.core.miros_stream_worker import run_miros_stream_worker

            run_miros_stream_worker(
                repo_dir,
                input_path,
                args.output,
                args.events_jsonl,
            )
        output_path = validate_midi_output(args.output, "MIROS worker")
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
    from src.model_profile_runtime_probe import (
        MODEL_PROFILE_RUNTIME_PROBE_SWITCH,
        run_model_profile_runtime_probe,
    )

    if MODEL_PROFILE_RUNTIME_PROBE_SWITCH in sys.argv:
        probe_index = sys.argv.index(MODEL_PROFILE_RUNTIME_PROBE_SWITCH)
        probe_arguments = sys.argv[probe_index + 1 :]
        if len(probe_arguments) != 1:
            print(
                f"{MODEL_PROFILE_RUNTIME_PROBE_SWITCH} requires exactly one profile ID",
                file=sys.stderr,
            )
            exit_code = 2
        else:
            exit_code = run_model_profile_runtime_probe(probe_arguments[0])
        if getattr(sys, "frozen", False):
            os._exit(exit_code)
            return
        sys.exit(exit_code)
    if "--web-inference-worker" in sys.argv:
        worker_index = sys.argv.index("--web-inference-worker")
        from src.web_api.inference_process import run_inference_worker

        exit_code = run_inference_worker(sys.argv[worker_index + 1 :])
        if getattr(sys, "frozen", False):
            os._exit(exit_code)
            return
        sys.exit(exit_code)
    if "--miros-worker" in sys.argv:
        worker_index = sys.argv.index("--miros-worker")
        exit_code = _run_miros_worker(sys.argv[worker_index + 1 :])
        if getattr(sys, "frozen", False):
            os._exit(exit_code)
            return
        sys.exit(exit_code)
    if _is_web_api_runtime():
        backend_arguments = list(sys.argv[1:])
        if "--web-api" in backend_arguments:
            backend_arguments.remove("--web-api")
        from src.web_api.__main__ import main as run_web_api

        sys.exit(run_web_api(backend_arguments))

    # Set console encoding before any localized CLI output.
    setup_chinese_environment()

    if "-h" in sys.argv or "--help" in sys.argv:
        from src.i18n.translator import t

        print(
            f"{t('cli.usage')}: python -m src.main [--self-test] [--self-test-no-load] "
            "[--self-test-miros] [--self-test-gui-runtime] [--web-api ...]\n"
            "\n"
            f"{t('cli.options')}:\n"
            f"  -h, --help          {t('cli.help')}\n"
            f"  --self-test         {t('cli.self_test')}\n"
            f"  --self-test-no-load {t('cli.self_test_no_load')}\n"
            f"  --self-test-miros   {t('cli.self_test_miros')}\n"
            "  --self-test-gui-runtime  Validate Qt + CUDA/OpenVINO accelerator runtime\n"
            "  --web-api           Run the standalone inference HTTP service\n"
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

    if "--self-test-gui-runtime" in sys.argv:
        sys.exit(_run_gui_runtime_self_test())

    # 设置日志
    logger = setup_logger(log_dir=str(get_logs_dir()), level=logging.DEBUG)

    try:
        _prepare_torch_runtime_before_pyqt()
    except Exception:
        logger.critical(
            "Required portable runtime failed before PyQt startup",
            exc_info=True,
        )
        raise

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

        # DPI 缩放与面板分辨率无关。保留 Windows 的 125%/150%/175%
        # 等分数缩放，让 Qt 按每块屏幕的逻辑像素完成布局。
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
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
            for family in (
                "Noto Sans CJK SC",
                "WenQuanYi Micro Hei",
                "WenQuanYi Zen Hei",
                "Ubuntu",
            ):
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
