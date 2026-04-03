import builtins
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


if __name__ == "__main__":
    unittest.main()
