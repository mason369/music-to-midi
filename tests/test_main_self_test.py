import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
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

    def test_self_test_can_check_miros_without_loading_model(self):
        fake_logger = Mock()
        calls = []

        class FakeMirosTranscriber:
            def __init__(self, _config):
                calls.append("init")

            @staticmethod
            def is_available():
                calls.append("is_available")
                return True

            def _load_model(self):
                calls.append("load_model")

        stdout = io.StringIO()
        with patch.object(main_module, "setup_chinese_environment"), patch.object(
            main_module, "get_logs_dir", return_value="logs"
        ), patch.object(main_module, "setup_logger", return_value=fake_logger), redirect_stdout(stdout):
            exit_code = main_module._run_self_test(
                transcriber_cls=FakeMirosTranscriber,
                success_message="SELF-TEST OK: MIROS available",
                load_model=False,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, ["is_available", "init"])
        self.assertIn("SELF-TEST OK: MIROS available", stdout.getvalue())

    def test_self_test_can_check_yourmt3_without_loading_model(self):
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
            exit_code = main_module._run_self_test(
                transcriber_cls=FakeYourMT3Transcriber,
                success_message="SELF-TEST OK: YourMT3+ available without model load",
                load_model=False,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls, ["is_available", "init", "unload_model"])
        self.assertIn("without model load", stdout.getvalue())

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

    def test_main_dispatches_miros_worker_before_gui_startup(self):
        with patch.object(sys, "argv", ["MusicToMidi.exe", "--miros-worker", "-i", "in.wav", "-o", "out.mid"]), patch.object(
            main_module,
            "_run_miros_worker",
            return_value=7,
        ) as worker, patch(
            "PyQt6.QtWidgets.QApplication",
            create=True,
        ) as qapplication:
            with self.assertRaises(SystemExit) as cm:
                main_module.main()

        self.assertEqual(cm.exception.code, 7)
        worker.assert_called_once_with(["-i", "in.wav", "-o", "out.mid"])
        qapplication.assert_not_called()

    def test_main_dispatches_miros_worker_before_torch_preload(self):
        with patch.object(sys, "argv", ["MusicToMidi.exe", "--miros-worker", "-i", "in.wav", "-o", "out.mid"]), patch.object(
            main_module,
            "_run_miros_worker",
            return_value=7,
        ), patch.object(main_module, "_prepare_torch_runtime_before_pyqt") as prepare_torch:
            with self.assertRaises(SystemExit):
                main_module.main()

        prepare_torch.assert_not_called()

    def test_main_uses_hard_exit_for_frozen_miros_worker(self):
        with patch.object(sys, "argv", ["MusicToMidi.exe", "--miros-worker", "-i", "in.wav", "-o", "out.mid"]), patch.object(
            sys,
            "frozen",
            True,
            create=True,
        ), patch.object(
            main_module,
            "_run_miros_worker",
            return_value=7,
        ) as worker, patch.object(main_module.os, "_exit") as hard_exit:
            main_module.main()

        worker.assert_called_once_with(["-i", "in.wav", "-o", "out.mid"])
        hard_exit.assert_called_once_with(7)

    def test_main_help_exits_before_gui_startup_and_torch_preload(self):
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["MusicToMidi.exe", "--help"]), patch.object(
            main_module, "_prepare_torch_runtime_before_pyqt"
        ) as prepare_torch, patch("PyQt6.QtWidgets.QApplication", create=True) as qapplication, redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as cm:
                main_module.main()

        self.assertEqual(cm.exception.code, 0)
        self.assertIn("用法: python -m src.main", stdout.getvalue())
        self.assertIn("检查 YourMT3+ 可用性", stdout.getvalue())
        self.assertIn("--self-test", stdout.getvalue())
        self.assertIn("--self-test-no-load", stdout.getvalue())
        self.assertIn("--self-test-miros", stdout.getvalue())
        prepare_torch.assert_not_called()
        qapplication.assert_not_called()

    def test_main_dispatches_self_test_no_load_before_gui_startup_and_torch_preload(self):
        with patch.object(sys, "argv", ["MusicToMidi.exe", "--self-test-no-load"]), patch.object(
            main_module,
            "_run_self_test",
            return_value=0,
        ) as self_test, patch.object(main_module, "_prepare_torch_runtime_before_pyqt") as prepare_torch, patch(
            "PyQt6.QtWidgets.QApplication",
            create=True,
        ) as qapplication:
            with self.assertRaises(SystemExit) as cm:
                main_module.main()

        self.assertEqual(cm.exception.code, 0)
        self_test.assert_called_once_with(
            success_message="SELF-TEST OK: YourMT3+ available without model load",
            load_model=False,
        )
        prepare_torch.assert_not_called()
        qapplication.assert_not_called()

    def test_miros_worker_writes_failure_status_json(self):
        fake_transcribe = types.ModuleType("transcribe")

        def boom(_input, _output):
            raise RuntimeError("miros exploded")

        fake_transcribe.transcribe = boom

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "in.wav"
            input_path.write_bytes(b"wav")
            status_path = Path(tmp) / "miros-status.json"
            with patch.dict(sys.modules, {"transcribe": fake_transcribe}):
                exit_code = main_module._run_miros_worker(
                    [
                        "-i",
                        str(input_path),
                        "-o",
                        str(Path(tmp) / "out.mid"),
                        "--status-json",
                        str(status_path),
                    ]
                )

            payload = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["ok"])
        self.assertIn("miros exploded", payload["error"])
        self.assertIn("RuntimeError: miros exploded", payload["traceback"])

    def test_miros_worker_fails_fast_for_missing_input_before_import(self):
        fake_transcribe = types.ModuleType("transcribe")
        calls = []

        def should_not_run(_input, _output):
            calls.append("called")
            raise AssertionError("transcribe should not load a model for missing input")

        fake_transcribe.transcribe = should_not_run

        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "miros-status.json"
            missing_input = Path(tmp) / "missing.wav"
            with patch.dict(sys.modules, {"transcribe": fake_transcribe}):
                exit_code = main_module._run_miros_worker(
                    [
                        "-i",
                        str(missing_input),
                        "-o",
                        str(Path(tmp) / "out.mid"),
                        "--status-json",
                        str(status_path),
                    ]
                )

            payload = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(calls, [])
        self.assertFalse(payload["ok"])
        self.assertIn("MIROS input audio does not exist", payload["error"])

    def test_miros_worker_stubs_optional_onnxruntime_during_transcribe_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "transcribe.py").write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        "import onnxruntime",
                        "if getattr(onnxruntime, '__version__', None) != '0.0':",
                        "    raise RuntimeError('onnxruntime was not isolated')",
                        "def transcribe(input_path, output_path):",
                        "    Path(output_path).write_bytes("
                        "b'MThd\\x00\\x00\\x00\\x06\\x00\\x00\\x00\\x01\\x01\\xe0'"
                        "+ b'MTrk\\x00\\x00\\x00\\x04\\x00\\xff\\x2f\\x00')",
                    ]
                ),
                encoding="utf-8",
            )
            input_path = root / "in.wav"
            input_path.write_bytes(b"wav")
            output_path = root / "out.mid"
            old_cwd = os.getcwd()
            old_transcribe = sys.modules.pop("transcribe", None)
            old_onnxruntime = sys.modules.pop("onnxruntime", None)
            os.chdir(root)
            try:
                exit_code = main_module._run_miros_worker(
                    ["-i", str(input_path), "-o", str(output_path)]
                )
            finally:
                os.chdir(old_cwd)
                sys.modules.pop("transcribe", None)
                if old_transcribe is not None:
                    sys.modules["transcribe"] = old_transcribe
                if old_onnxruntime is not None:
                    sys.modules["onnxruntime"] = old_onnxruntime
                else:
                    sys.modules.pop("onnxruntime", None)
            output_exists = output_path.exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_exists)


if __name__ == "__main__":
    unittest.main()
