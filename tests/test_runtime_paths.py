import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils import runtime_paths


class TestRuntimePaths(unittest.TestCase):
    def test_get_ffmpeg_executable_prefers_bundled_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "tools" / "ffmpeg" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            ffmpeg = bin_dir / "ffmpeg.exe"
            ffmpeg.write_bytes(b"exe")

            with mock.patch.object(runtime_paths, "get_bundle_roots", return_value=[root]):
                with mock.patch("src.utils.runtime_paths.shutil.which", return_value=None):
                    resolved = runtime_paths.get_ffmpeg_executable()

            self.assertEqual(Path(resolved), ffmpeg)

    def test_get_resource_path_uses_existing_bundle_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resource = root / "resources" / "icons" / "app.ico"
            resource.parent.mkdir(parents=True, exist_ok=True)
            resource.write_bytes(b"ico")

            with mock.patch.object(runtime_paths, "get_bundle_roots", return_value=[root]):
                resolved = runtime_paths.get_resource_path("resources/icons/app.ico")

            self.assertEqual(resolved, resource)

    def test_frozen_runtime_prefers_portable_runtime_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe_dir = Path(tmp)
            runtime_dir = exe_dir / "runtime"

            with mock.patch.object(runtime_paths, "is_frozen_app", return_value=True):
                with mock.patch.object(runtime_paths, "get_executable_dir", return_value=exe_dir):
                    result = runtime_paths.get_runtime_data_dir()

            self.assertEqual(result, runtime_dir)
            self.assertTrue(runtime_dir.exists())


if __name__ == "__main__":
    unittest.main()
