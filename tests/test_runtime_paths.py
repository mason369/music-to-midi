import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.utils import runtime_paths


class RuntimePathBootstrapTests(unittest.TestCase):
    def test_bootstrap_runtime_environment_registers_only_torch_native_dirs_by_default(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            torch_lib = root / "torch" / "lib"
            ort_capi = root / "onnxruntime" / "capi"
            ffmpeg_bin = root / "ffmpeg" / "bin"
            torch_lib.mkdir(parents=True)
            ort_capi.mkdir(parents=True)
            ffmpeg_bin.mkdir(parents=True)
            (ffmpeg_bin / "ffmpeg.exe").write_text("", encoding="utf-8")
            (ffmpeg_bin / "ffprobe.exe").write_text("", encoding="utf-8")

            with patch.object(runtime_paths, "get_bundle_roots", return_value=[root]), patch.object(
                runtime_paths.os, "add_dll_directory", create=True
            ) as add_dll_directory, patch.dict(runtime_paths.os.environ, {"PATH": ""}, clear=True):
                runtime_paths.bootstrap_runtime_environment()
                path_entries = {
                    str(Path(entry).resolve())
                    for entry in runtime_paths.os.environ["PATH"].split(os.pathsep)
                    if entry
                }

            registered = {str(Path(call.args[0]).resolve()) for call in add_dll_directory.call_args_list}
            self.assertIn(str(torch_lib.resolve()), registered)
            self.assertIn(str(torch_lib.resolve()), path_entries)
            self.assertNotIn(str(ort_capi.resolve()), registered)
            self.assertNotIn(str(ort_capi.resolve()), path_entries)

    def test_activate_audio_separator_runtime_registers_bundled_onnxruntime_dirs(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            torch_lib = root / "torch" / "lib"
            ort_capi = root / "onnxruntime" / "capi"
            torch_lib.mkdir(parents=True)
            ort_capi.mkdir(parents=True)
            (ort_capi / "onnxruntime.dll").write_text("", encoding="utf-8")
            (ort_capi / "onnxruntime_providers_shared.dll").write_text("", encoding="utf-8")

            with patch.object(runtime_paths, "get_bundle_roots", return_value=[root]), patch.object(
                runtime_paths.os, "add_dll_directory", create=True
            ) as add_dll_directory, patch.dict(runtime_paths.os.environ, {"PATH": ""}, clear=True), patch.object(
                runtime_paths, "_ensure_torch_loaded_before_onnxruntime"
            ) as ensure_torch_loaded, patch.object(
                runtime_paths, "_preload_bundled_onnxruntime_libraries"
            ) as preload_ort:
                runtime_paths.activate_audio_separator_runtime()
                path_entries = {
                    str(Path(entry).resolve())
                    for entry in runtime_paths.os.environ["PATH"].split(os.pathsep)
                    if entry
                }

            registered = {str(Path(call.args[0]).resolve()) for call in add_dll_directory.call_args_list}
            self.assertIn(str(ort_capi.resolve()), registered)
            self.assertIn(str(ort_capi.resolve()), path_entries)
            ensure_torch_loaded.assert_called_once()
            preload_ort.assert_called_once()


if __name__ == "__main__":
    unittest.main()
