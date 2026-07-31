"""Preload one coherent MSVC runtime in PyInstaller analysis subprocesses."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


_BUILD_VC_RUNTIME_HANDLES: list[object] = []


def _preload_build_vc_runtime() -> None:
    runtime_dir_value = os.environ.get("MUSIC_TO_MIDI_BUILD_VC_RUNTIME_DIR")
    if os.name != "nt" or not runtime_dir_value:
        return

    runtime_dir = Path(runtime_dir_value)
    for dll_name in (
        "msvcp140.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    ):
        dll_path = runtime_dir / dll_name
        if not dll_path.is_file():
            sys.stderr.write(
                f"Required PyInstaller build runtime is missing: {dll_path}\n"
            )
            os._exit(86)
        try:
            handle = ctypes.WinDLL(str(dll_path))
        except OSError as exc:
            sys.stderr.write(
                f"Failed to preload PyInstaller build runtime {dll_path}: {exc}\n"
            )
            os._exit(86)
        _BUILD_VC_RUNTIME_HANDLES.append(handle)


_preload_build_vc_runtime()
