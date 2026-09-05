import builtins
import io
import sys
import unittest
import warnings

from src.utils import warnings_filter


class WarningsFilterTests(unittest.TestCase):
    def test_setup_chinese_environment_recovers_missing_standard_streams(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_print = builtins.print
        original_showwarning = warnings.showwarning

        try:
            sys.stdout = None
            sys.stderr = None

            warnings_filter.setup_chinese_environment()

            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)
            self.assertTrue(hasattr(sys.stdout, "write"))
            self.assertTrue(hasattr(sys.stderr, "write"))

            builtins.print("portable runtime still prints safely")
            warnings.showwarning(UserWarning("portable warning"), UserWarning, __file__, 1)
            sys.stderr.write("portable stderr still writes safely\n")
            sys.stderr.flush()
        finally:
            builtins.print = original_print
            warnings.showwarning = original_showwarning
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_setup_chinese_environment_recovers_broken_standard_streams(self):
        class BrokenStream(io.StringIO):
            def write(self, _text):
                raise OSError(22, "Invalid argument")

            def flush(self):
                raise OSError(22, "Invalid argument")

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_print = builtins.print
        original_showwarning = warnings.showwarning

        try:
            sys.stdout = BrokenStream()
            sys.stderr = BrokenStream()

            warnings_filter.setup_chinese_environment()

            builtins.print("portable runtime still prints safely")
            sys.stderr.write("portable stderr still writes safely\n")
            sys.stderr.flush()
        finally:
            builtins.print = original_print
            warnings.showwarning = original_showwarning
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_setup_chinese_environment_recovers_frozen_pseudo_standard_streams(self):
        class FrozenPseudoStream(io.StringIO):
            name = "<stdout>"

            def isatty(self):
                return False

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_print = builtins.print
        original_showwarning = warnings.showwarning
        had_frozen = hasattr(sys, "frozen")
        original_frozen = getattr(sys, "frozen", None)

        try:
            sys.frozen = True
            sys.stdout = FrozenPseudoStream()
            sys.stderr = FrozenPseudoStream()

            warnings_filter.setup_chinese_environment()

            self.assertIsNot(sys.stdout, sys.__stdout__)
            builtins.print("portable runtime still prints safely")
            sys.stderr.write("portable stderr still writes safely\n")
            sys.stderr.flush()
        finally:
            if had_frozen:
                sys.frozen = original_frozen
            else:
                delattr(sys, "frozen")
            builtins.print = original_print
            warnings.showwarning = original_showwarning
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_frozen_console_entries_preserve_redirected_standard_streams(self):
        class RedirectedConsoleStream(io.StringIO):
            name = "<stdout>"

            def isatty(self):
                return False

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_executable = sys.executable
        had_frozen = hasattr(sys, "frozen")
        original_frozen = getattr(sys, "frozen", None)

        try:
            sys.frozen = True
            for executable_name in (
                "MusicToMidiCLI.exe",
                "MusicToMidiCLIXpu.exe",
                "MusicToMidiBackend.exe",
                "MusicToMidiBackendXpu.exe",
            ):
                with self.subTest(executable_name=executable_name):
                    redirected_stdout = RedirectedConsoleStream()
                    redirected_stderr = RedirectedConsoleStream()
                    sys.executable = rf"C:\\portable\\{executable_name}"
                    sys.stdout = redirected_stdout
                    sys.stderr = redirected_stderr

                    warnings_filter.ensure_standard_streams()

                    self.assertIs(sys.stdout, redirected_stdout)
                    self.assertIs(sys.stderr, redirected_stderr)
                    print("captured CLI output")
                    self.assertIn("captured CLI output", redirected_stdout.getvalue())
        finally:
            if had_frozen:
                sys.frozen = original_frozen
            else:
                delattr(sys, "frozen")
            sys.executable = original_executable
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    unittest.main()
