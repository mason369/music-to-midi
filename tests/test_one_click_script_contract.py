import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class OneClickScriptContractTests(unittest.TestCase):
    def test_windows_launcher_checks_aria_amt_package_and_model(self):
        script = (REPO_ROOT / "run.ps1").read_text(encoding="utf-8")

        self.assertIn("find_spec('amt.run')", script)
        self.assertIn("download_aria_amt_model.py", script)
        self.assertIn("Aria-AMT", script)

    def test_windows_launcher_checks_bytedance_pedal_package_and_model(self):
        script = (REPO_ROOT / "run.ps1").read_text(encoding="utf-8")

        self.assertIn("ByteDancePianoTranscriber", script)
        self.assertIn("download_bytedance_piano_model.py", script)
        self.assertIn("ByteDance Piano", script)

    def test_windows_launcher_checks_miros_backend_and_model(self):
        script = (REPO_ROOT / "run.ps1").read_text(encoding="utf-8")

        self.assertIn("MirosTranscriber", script)
        self.assertIn("MirosTranscriber.is_available()", script)
        self.assertIn("download_miros_model.py", script)

    def test_windows_launcher_checks_all_selectable_yourmt3_models(self):
        script = (REPO_ROOT / "run.ps1").read_text(encoding="utf-8")

        self.assertIn("OFFICIAL_YOURMT3_MODEL_KEYS", script)
        self.assertIn("missing YourMT3+ official model modes", script)

    def test_windows_installer_downloads_aria_amt_model(self):
        script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

        self.assertIn("download_aria_amt_model.py", script)
        self.assertIn("Aria-AMT", script)
        self.assertIn("Python 3.11", script)
        self.assertIn("requirements-without-aria-amt.txt", script)
        self.assertIn("audio-separator==0.41.1", script)
        self.assertIn("--no-deps", script)

    def test_windows_installer_downloads_bytedance_pedal_model(self):
        script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

        self.assertIn("download_bytedance_piano_model.py", script)
        self.assertIn("ByteDance Piano", script)

    def test_windows_installer_stops_when_required_model_downloads_fail(self):
        script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

        self.assertIn("Write-Err \"YourMT3+ 官方模式模型下载失败\"", script)
        self.assertIn("Write-Err \"BS-RoFormer 模型下载失败\"", script)
        self.assertNotIn("按 Ctrl+C 可跳过", script)
        self.assertNotIn("稍后手动执行", script)

    def test_windows_installer_prepares_miros_without_masking_failures(self):
        script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

        self.assertIn("download_miros_model.py", script)
        self.assertIn("https://github.com/amt-os/ai4m-miros.git", script)
        self.assertIn('external\\ai4m-miros', script)
        self.assertIn("MirosTranscriber.PRETRAINED_REL_PATH", script)
        self.assertIn("MirosTranscriber.CHECKPOINT_REL_PATH", script)
        self.assertIn("MIROS 权重准备失败", script)
        self.assertIn("Push-Location $mirosDir", script)
        self.assertIn("Pop-Location", script)
        self.assertIn("sys.exit(0 if reason == '' else 1)", script)
        self.assertNotIn("将继续使用当前本地版本", script)

    def test_linux_launcher_checks_aria_amt_package_and_model(self):
        script = (REPO_ROOT / "run.sh").read_text(encoding="utf-8")

        self.assertIn("download_aria_amt_model.py", script)
        self.assertIn("Aria-AMT", script)

    def test_linux_launcher_checks_bytedance_pedal_package_and_model(self):
        script = (REPO_ROOT / "run.sh").read_text(encoding="utf-8")

        self.assertIn("ByteDancePianoTranscriber", script)
        self.assertIn("download_bytedance_piano_model.py", script)
        self.assertIn("ByteDance Piano", script)

    def test_linux_launcher_checks_miros_backend_and_model(self):
        script = (REPO_ROOT / "run.sh").read_text(encoding="utf-8")

        self.assertIn("MirosTranscriber", script)
        self.assertIn("MirosTranscriber.is_available()", script)
        self.assertIn("download_miros_model.py", script)

    def test_linux_launcher_checks_all_selectable_yourmt3_models(self):
        script = (REPO_ROOT / "run.sh").read_text(encoding="utf-8")

        self.assertIn("OFFICIAL_YOURMT3_MODEL_KEYS", script)
        self.assertIn("missing YourMT3+ official model modes", script)

    def test_linux_installer_downloads_aria_amt_model_and_requires_python_311(self):
        script = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertIn("download_aria_amt_model.py", script)
        self.assertIn("python3.11", script)
        self.assertIn("Aria-AMT", script)
        self.assertIn("requirements-without-aria-amt.txt", script)
        self.assertIn("audio-separator==0.41.1", script)
        self.assertIn("--no-deps", script)

    def test_linux_installer_downloads_bytedance_pedal_model(self):
        script = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertIn("download_bytedance_piano_model.py", script)
        self.assertIn("ByteDance Piano", script)

    def test_linux_installer_stops_when_required_model_downloads_fail(self):
        script = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertIn('error "YourMT3+ 官方模式模型下载失败"', script)
        self.assertIn('error "BS-RoFormer model download failed"', script)
        self.assertNotIn("Press Ctrl+C to skip", script)
        self.assertNotIn("可稍后手动运行", script)

    def test_linux_installer_prepares_miros_without_masking_failures(self):
        script = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertIn("download_miros_model.py", script)
        self.assertIn("MirosTranscriber.is_model_available()", script)
        self.assertIn("MIROS", script)
        self.assertNotIn("MIROS download failed", script)

    def test_project_requires_python_matches_aria_amt_requirement(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('requires-python = ">=3.11"', pyproject)

    def test_requirements_avoid_audio_separator_numpy_resolver_conflict(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("numpy>=1.26.0,<2", requirements)
        self.assertNotIn("audio-separator", "\n".join(
            line for line in requirements.splitlines() if not line.lstrip().startswith("#")
        ))

    def test_requirements_include_miros_runtime_basics(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("h5py", requirements)
        self.assertIn("mirdata", requirements)
        self.assertIn("chardet>=5,<6", requirements)
        self.assertIn("onnxruntime==1.23.2", requirements)

    def test_requirements_match_miros_transformers_version(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("transformers==4.48.3", requirements)

    def test_windows_installer_uses_same_runtime_dependency_pins(self):
        script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

        self.assertIn("requests>=2.32.5,<3", script)
        self.assertIn("chardet>=5,<6", script)
        self.assertIn("onnxruntime==1.23.2", script)


if __name__ == "__main__":
    unittest.main()
