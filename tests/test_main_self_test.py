import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import src.main as main_module


class MainSelfTestTests(unittest.TestCase):
    def test_self_test_returns_zero_when_yourmt3_available(self):
        fake_logger = Mock()

        class FakeYourMT3Transcriber:
            def __init__(self, _config):
                pass

            @staticmethod
            def is_available():
                return True

        stdout = io.StringIO()
        with patch.object(main_module, "setup_chinese_environment"), patch.object(
            main_module, "get_logs_dir", return_value="logs"
        ), patch.object(main_module, "setup_logger", return_value=fake_logger), redirect_stdout(stdout):
            exit_code = main_module._run_self_test(transcriber_cls=FakeYourMT3Transcriber)

        self.assertEqual(exit_code, 0)
        self.assertIn("SELF-TEST OK", stdout.getvalue())

    def test_self_test_loads_and_unloads_model_when_backend_is_available(self):
        fake_logger = Mock()
        calls = []

        class FakeYourMT3Transcriber:
            def __init__(self, _config):
                calls.append("init")

            @staticmethod
            def is_available():
                calls.append("is_available")
                return True

            def _load_model(self):
                calls.append("load_model")

            def unload_model(self):
                calls.append("unload_model")

        stdout = io.StringIO()
        with patch.object(main_module, "setup_chinese_environment"), patch.object(
            main_module, "get_logs_dir", return_value="logs"
        ), patch.object(main_module, "setup_logger", return_value=fake_logger), redirect_stdout(stdout):
            exit_code = main_module._run_self_test(transcriber_cls=FakeYourMT3Transcriber)

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, ["is_available", "init", "load_model", "unload_model"])
        self.assertIn("SELF-TEST OK", stdout.getvalue())

    def test_self_test_returns_one_with_reason_when_yourmt3_unavailable(self):
        fake_logger = Mock()

        class FakeYourMT3Transcriber:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def get_unavailable_reason():
                return "缺少打包依赖"

        stdout = io.StringIO()
        with patch.object(main_module, "setup_chinese_environment"), patch.object(
            main_module, "get_logs_dir", return_value="logs"
        ), patch.object(main_module, "setup_logger", return_value=fake_logger), redirect_stdout(stdout):
            exit_code = main_module._run_self_test(transcriber_cls=FakeYourMT3Transcriber)

        self.assertEqual(exit_code, 1)
        self.assertIn("缺少打包依赖", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
