import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class MainStartupContractTests(unittest.TestCase):
    def test_main_does_not_force_global_torch_thread_tuning(self):
        source = (REPO_ROOT / "src" / "main.py").read_text(encoding="utf-8")

        self.assertNotIn("torch.set_num_threads(", source)
        self.assertNotIn("OMP_NUM_THREADS", source)
        self.assertNotIn("MKL_NUM_THREADS", source)

    def test_main_does_not_enable_global_cudnn_benchmark(self):
        source = (REPO_ROOT / "src" / "main.py").read_text(encoding="utf-8")

        self.assertNotIn("torch.backends.cudnn.benchmark = True", source)

    def test_4k_detection_does_not_preempt_qt_dpi_awareness(self):
        source = (REPO_ROOT / "src" / "main.py").read_text(encoding="utf-8")

        self.assertIn("EnumDisplaySettingsW", source)
        self.assertNotIn("SetProcessDpiAwareness", source)
        self.assertNotIn("SetProcessDPIAware", source)

    def test_main_application_name_uses_i18n(self):
        source = (REPO_ROOT / "src" / "main.py").read_text(encoding="utf-8")

        self.assertIn('app.setApplicationName(t("app.name"))', source)
        self.assertNotIn('app.setApplicationName("音乐转MIDI")', source)


if __name__ == "__main__":
    unittest.main()
