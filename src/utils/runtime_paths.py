"""
运行时路径与外部工具解析。

目标：
1. 开发环境保持现有目录行为。
2. PyInstaller 打包后优先读取 exe 邻近目录中的 bundled 资源。
3. 需要写入的日志/缓存目录优先使用便携包内 runtime 目录，不可写时回退到用户目录。
"""
from __future__ import annotations

import os
import shutil
import sys
import importlib
from pathlib import Path
from typing import Iterable, List, Optional


APP_DATA_DIRNAME = ".music-to-midi"


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_executable_dir() -> Path:
    if is_frozen_app():
        return Path(sys.executable).resolve().parent
    return get_project_root()


def get_bundle_roots() -> List[Path]:
    roots: List[Path] = [get_executable_dir()]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))

    unique: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def get_legacy_app_data_dir() -> Path:
    return Path.home() / APP_DATA_DIRNAME


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def get_runtime_data_dir() -> Path:
    if is_frozen_app():
        portable_dir = get_executable_dir() / "runtime"
        if _is_writable_dir(portable_dir):
            return portable_dir
    return get_legacy_app_data_dir()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_dir() -> Path:
    return ensure_dir(get_runtime_data_dir() / "logs")


def _find_existing_relative_path(relative_paths: Iterable[str]) -> Optional[Path]:
    for root in get_bundle_roots():
        for rel in relative_paths:
            candidate = root / rel
            if candidate.exists():
                return candidate
    return None


def get_resource_path(relative_path: str) -> Path:
    bundled = _find_existing_relative_path([relative_path])
    if bundled is not None:
        return bundled
    return get_project_root() / relative_path


def get_audio_separator_model_dir() -> Path:
    bundled = _find_existing_relative_path(
        [
            "models/audio-separator",
            "assets/models/audio-separator",
        ]
    )
    if bundled is not None:
        return bundled
    return ensure_dir(get_runtime_data_dir() / "models" / "audio-separator")


def get_yourmt3_source_dir() -> Optional[Path]:
    candidates = [
        "YourMT3/amt/src",
        "external/YourMT3/amt/src",
    ]
    bundled = _find_existing_relative_path(candidates)
    if bundled is not None:
        return bundled

    for base in (
        Path("YourMT3"),
        get_project_root() / "YourMT3",
        Path("external/YourMT3"),
        get_project_root() / "external/YourMT3",
    ):
        candidate = base / "amt" / "src"
        if candidate.exists():
            return candidate
    return None


def get_miros_source_dir() -> Optional[Path]:
    candidates = [
        "ai4m-miros",
        "external/ai4m-miros",
        "MIROS",
        "external/MIROS",
    ]
    bundled = _find_existing_relative_path(candidates)
    if bundled is not None:
        return bundled

    for base in (
        Path("ai4m-miros"),
        get_project_root() / "ai4m-miros",
        Path("external/ai4m-miros"),
        get_project_root() / "external/ai4m-miros",
        Path("MIROS"),
        get_project_root() / "MIROS",
        Path("external/MIROS"),
        get_project_root() / "external/MIROS",
    ):
        if (base / "main.py").exists() and (base / "transcribe.py").exists():
            return base
    return None


def get_yourmt3_search_roots() -> List[Path]:
    roots: List[Path] = []

    for candidate in (
        _find_existing_relative_path(["models/yourmt3_all"]),
        _find_existing_relative_path(["models/yourmt3"]),
    ):
        if candidate is not None:
            roots.append(candidate)

    roots.append(get_runtime_data_dir() / "models" / "yourmt3_all")
    roots.append(Path.home() / ".cache" / "music_ai_models" / "yourmt3_all")

    unique: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def get_yourmt3_download_root() -> Path:
    if is_frozen_app():
        return ensure_dir(get_runtime_data_dir() / "models" / "yourmt3_all")
    return ensure_dir(Path.home() / ".cache" / "music_ai_models" / "yourmt3_all")


def get_ffmpeg_bin_dir() -> Optional[Path]:
    candidate = _find_existing_relative_path(
        [
            "tools/ffmpeg/bin",
            "tools/ffmpeg",
            "ffmpeg/bin",
            "ffmpeg",
        ]
    )
    if candidate is not None:
        return candidate

    for exe_name in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(exe_name)
        if found:
            return Path(found).resolve().parent
    return None


def get_ffmpeg_executable() -> str:
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    bin_dir = get_ffmpeg_bin_dir()
    if bin_dir is not None:
        candidate = bin_dir / exe_name
        if candidate.exists():
            return str(candidate)

    found = shutil.which("ffmpeg")
    return found or "ffmpeg"


def get_ffprobe_executable() -> str:
    exe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    bin_dir = get_ffmpeg_bin_dir()
    if bin_dir is not None:
        candidate = bin_dir / exe_name
        if candidate.exists():
            return str(candidate)

    found = shutil.which("ffprobe")
    return found or "ffprobe"


def _get_existing_bundle_dirs(relative_dirs: Iterable[str]) -> List[Path]:
    candidates: List[Path] = []
    for root in get_bundle_roots():
        for relative_dir in relative_dirs:
            candidate = root / relative_dir
            if candidate.is_dir():
                candidates.append(candidate)

    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key not in seen:
            unique.append(resolved)
            seen.add(key)
    return unique


def get_native_library_dirs() -> List[Path]:
    return _get_existing_bundle_dirs(("torch/lib",))


def get_audio_separator_native_library_dirs() -> List[Path]:
    return _get_existing_bundle_dirs(("onnxruntime/capi",))


def _prepend_path_entries(entries: Iterable[Path]) -> None:
    current_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    merged: List[str] = []
    seen: set[str] = set()

    for entry in [str(path) for path in entries] + current_entries:
        if entry and entry not in seen:
            merged.append(entry)
            seen.add(entry)

    if merged:
        os.environ["PATH"] = os.pathsep.join(merged)


def _register_windows_dll_directories(entries: Iterable[Path]) -> None:
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if not callable(add_dll_directory):
        return

    for entry in entries:
        try:
            add_dll_directory(str(entry))
        except OSError:
            continue


def _preload_bundled_onnxruntime_libraries(entries: Iterable[Path]) -> None:
    capi_dirs = [
        entry for entry in entries
        if entry.name == "capi" and entry.parent.name == "onnxruntime"
    ]
    if not capi_dirs:
        return

    try:
        import ctypes
    except Exception:
        return

    if os.name == "nt":
        for capi_dir in capi_dirs:
            for lib_name in ("onnxruntime.dll", "onnxruntime_providers_shared.dll"):
                candidate = capi_dir / lib_name
                if not candidate.exists():
                    continue
                try:
                    ctypes.WinDLL(str(candidate))
                except OSError:
                    continue
        return

    rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)
    for capi_dir in capi_dirs:
        for pattern in ("libonnxruntime.so*", "libonnxruntime_providers_shared.so*"):
            for candidate in sorted(capi_dir.glob(pattern)):
                if not candidate.is_file():
                    continue
                try:
                    ctypes.CDLL(str(candidate), mode=rtld_global)
                except OSError:
                    continue


def _ensure_torch_loaded_before_onnxruntime() -> None:
    if "torch" in sys.modules:
        return
    try:
        importlib.import_module("torch")
    except Exception:
        return


def _bootstrap_ffmpeg_environment() -> None:
    ffmpeg_bin = get_ffmpeg_bin_dir()
    if ffmpeg_bin is not None:
        current_path = os.environ.get("PATH", "")
        ffmpeg_bin_str = str(ffmpeg_bin)
        if ffmpeg_bin_str not in current_path.split(os.pathsep):
            os.environ["PATH"] = ffmpeg_bin_str + os.pathsep + current_path

        ffmpeg_path = Path(ffmpeg_bin) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        ffprobe_path = Path(ffmpeg_bin) / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if ffmpeg_path.exists():
            os.environ.setdefault("FFMPEG_BINARY", str(ffmpeg_path))
            os.environ.setdefault("IMAGEIO_FFMPEG_EXE", str(ffmpeg_path))
        if ffprobe_path.exists():
            os.environ.setdefault("FFPROBE_BINARY", str(ffprobe_path))


def bootstrap_runtime_environment() -> None:
    native_library_dirs = get_native_library_dirs()
    if native_library_dirs:
        _prepend_path_entries(native_library_dirs)
        _register_windows_dll_directories(native_library_dirs)

    _bootstrap_ffmpeg_environment()


def activate_audio_separator_runtime() -> None:
    # Load PyTorch first so Windows resolves its CUDA/cuDNN stack before
    # onnxruntime/capi enters PATH. This avoids YourMT3 inference corruption
    # when both runtimes coexist in the same process.
    torch_library_dirs = get_native_library_dirs()
    if torch_library_dirs:
        _prepend_path_entries(torch_library_dirs)
        _register_windows_dll_directories(torch_library_dirs)

    _ensure_torch_loaded_before_onnxruntime()

    onnxruntime_library_dirs = get_audio_separator_native_library_dirs()
    if onnxruntime_library_dirs:
        _prepend_path_entries(onnxruntime_library_dirs)
        _register_windows_dll_directories(onnxruntime_library_dirs)
        _preload_bundled_onnxruntime_libraries(onnxruntime_library_dirs)

    _bootstrap_ffmpeg_environment()
