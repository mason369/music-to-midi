import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core import yourmt3_transcriber
from src.core.yourmt3_transcriber import YourMT3Transcriber


class YourMT3ImportIsolationTests(unittest.TestCase):
    def test_clear_yourmt3_import_state_removes_conflicting_top_level_modules(self):
        fake_modules = {
            "model": types.ModuleType("model"),
            "model.ymt3": types.ModuleType("model.ymt3"),
            "utils": types.ModuleType("utils"),
            "utils.task_manager": types.ModuleType("utils.task_manager"),
            "config": types.ModuleType("config"),
            "config.config": types.ModuleType("config.config"),
        }

        original_modules = {}
        try:
            for name, module in fake_modules.items():
                original_modules[name] = sys.modules.get(name)
                sys.modules[name] = module

            yourmt3_transcriber._clear_yourmt3_import_state()

            for name in fake_modules:
                self.assertNotIn(name, sys.modules)
        finally:
            for name in fake_modules:
                sys.modules.pop(name, None)
                if original_modules.get(name) is not None:
                    sys.modules[name] = original_modules[name]

    def test_is_available_checks_source_tree_without_importing_yourmt3_modules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            amt_src = Path(tmpdir)
            for relative_path in (
                Path("model/ymt3.py"),
                Path("utils/task_manager.py"),
                Path("config/config.py"),
            ):
                target = amt_src / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# stub\n", encoding="utf-8")

            fake_model = amt_src / "model.ckpt"
            fake_model.write_text("checkpoint", encoding="utf-8")

            real_import = __import__
            imported_names = []

            def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                imported_names.append(name)
                if name == "pytorch_lightning":
                    return types.SimpleNamespace(__name__="pytorch_lightning")
                if name == "yourmt3":
                    raise ModuleNotFoundError("No module named 'yourmt3'")
                if name == "model.ymt3":
                    raise AssertionError("is_available() should not import model.ymt3")
                return real_import(name, globals, locals, fromlist, level)

            with patch("src.core.yourmt3_transcriber._import_torch", return_value=object()), patch(
                "src.core.yourmt3_transcriber._get_yourmt3_amt_src_path",
                return_value=str(amt_src),
            ), patch("src.utils.yourmt3_downloader.get_model_path", return_value=fake_model), patch(
                "builtins.__import__", side_effect=fake_import
            ):
                available = YourMT3Transcriber.is_available()

            self.assertTrue(available)
            self.assertNotIn("model.ymt3", imported_names)


if __name__ == "__main__":
    unittest.main()
