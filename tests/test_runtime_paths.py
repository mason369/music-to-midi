import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.utils import runtime_paths


class RuntimePathBootstrapTests(unittest.TestCase):
    def test_bootstrap_runtime_environment_registers_bundled_native_library_dirs(self):
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
            self.assertIn(str(ort_capi.resolve()), registered)
            self.assertIn(str(torch_lib.resolve()), path_entries)
            self.assertIn(str(ort_capi.resolve()), path_entries)


if __name__ == "__main__":
    unittest.main()
