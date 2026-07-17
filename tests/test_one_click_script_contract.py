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

    def test_windows_launcher_checks_six_stem_assets(self):
        script = (REPO_ROOT / "run.ps1").read_text(encoding="utf-8")

        for expected in (
            "validate_multistem_assets",
            "download_multistem_model.py",
            "BS-RoFormer SW Fixed checkpoint:",
            "BS-RoFormer SW Fixed 六声部分离模型",
            "is_vocal_model_available",
            "is_accompaniment_model_available",
            "from download_accompaniment_model import",
            "Leap XE + PolarFormer 人声分离模型检查通过",
            "is_transkun_v2_aug_model_available",
            "download_transkun_v2_aug_model.py",
            "TransKun V2 Aug 模型检查通过",
        ):
            self.assertIn(expected, script)

    def test_windows_installer_downloads_aria_amt_model(self):
        script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

        self.assertIn("download_aria_amt_model.py", script)
        self.assertIn("Aria-AMT", script)
        self.assertIn("Python 3.11", script)
        self.assertIn("requirements-without-aria-amt.txt", script)
        self.assertIn("audio-separator==0.44.1", script)
        self.assertIn(
            '"aria-amt @ https://github.com/EleutherAI/aria-amt/archive/'
            'a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip" --no-deps',
            script,
        )
        self.assertIn("AriaAmtTranscriber.get_unavailable_reason()", script)
        self.assertIn("--force-reinstall", script)

    def test_windows_installer_downloads_bytedance_pedal_model(self):
        script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

        self.assertIn("download_bytedance_piano_model.py", script)
        self.assertIn("ByteDance Piano", script)

    def test_installers_use_blackwell_compatible_cuda_torch_for_cuda12(self):
        for script_name in ("install.ps1", "install.sh"):
            script = (REPO_ROOT / script_name).read_text(encoding="utf-8")

            self.assertIn("torch==2.7.0", script)
            self.assertIn("torchaudio==2.7.0", script)
            self.assertIn("torchvision==0.22.0", script)
            self.assertIn("https://download.pytorch.org/whl/cu128", script)
            self.assertIn("CUDA 12.8", script)
            self.assertNotIn("torch==2.4.0", script)
            self.assertNotIn("https://download.pytorch.org/whl/cu121", script)

    def test_requirements_pin_the_exact_supported_torch_trio(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("torch==2.7.0", requirements)
        self.assertIn("torchaudio==2.7.0", requirements)
        self.assertIn("torchvision==0.22.0", requirements)
        self.assertNotIn("torch>=", requirements)
        self.assertNotIn("torchaudio>=", requirements)
        self.assertNotIn("torchvision>=", requirements)

    def test_installers_require_and_probe_the_complete_nvidia_cuda_runtime(self):
        for script_name in ("install.ps1", "install.sh"):
            script = (REPO_ROOT / script_name).read_text(encoding="utf-8")

            self.assertIn("CUDA 12.8", script)
            self.assertIn("torch.cuda.is_available()", script)
            self.assertIn("torch.cuda.synchronize()", script)
            self.assertIn("CUDAExecutionProvider", script)
            self.assertIn("--force-reinstall", script)
            self.assertNotIn("https://download.pytorch.org/whl/cpu", script)
            self.assertNotIn("https://download.pytorch.org/whl/cu118", script)
            self.assertNotIn("intel_extension_for_pytorch", script)

    def test_launchers_recheck_exact_versions_cuda_tensor_and_ort_provider(self):
        for script_name in ("run.ps1", "run.sh"):
            script = (REPO_ROOT / script_name).read_text(encoding="utf-8")

            for expected in (
                '"torch": "2.7.0"' if script_name.endswith(".sh") else "'torch': '2.7.0'",
                "torch.cuda.is_available()",
                "torch.cuda.synchronize()",
                "CUDAExecutionProvider",
                "ROCm runtime is unsupported",
            ):
                self.assertIn(expected, script)

    def test_windows_installer_stops_when_required_model_downloads_fail(self):
        script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

        for expected in (
            "download_sota_models.py",
            'Write-Err "统一模型集合下载或校验失败"',
            "download_multistem_model.py",
            "download_vocal_model.py",
            'Write-Err "Leap XE 90-band 人声模型下载或校验失败"',
            "download_accompaniment_model.py",
            'Write-Err "PolarFormer 伴奏模型下载或校验失败"',
            "download_transkun_v2_aug_model.py",
            'Write-Err "TransKun V2 Aug 模型下载或校验失败"',
        ):
            self.assertIn(expected, script)

        self.assertNotIn("vocal_rvc ensemble", script)
        self.assertNotIn("karaoke ensemble", script)
        self.assertNotIn("download_vocal_harmony_model.py", script)
        self.assertNotIn("按 Ctrl+C 可跳过", script)
        self.assertNotIn("稍后手动执行", script)

    def test_windows_installer_prepares_miros_without_masking_failures(self):
        script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

        self.assertIn("download_miros_model.py", script)
        self.assertIn("https://github.com/amt-os/ai4m-miros.git", script)
        self.assertIn("external\\ai4m-miros", script)
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

    def test_linux_launcher_checks_six_stem_assets(self):
        script = (REPO_ROOT / "run.sh").read_text(encoding="utf-8")

        for expected in (
            "validate_multistem_assets",
            "download_multistem_model.py",
            "BS-RoFormer SW Fixed checkpoint:",
            "BS-RoFormer SW Fixed six-stem",
            "is_vocal_model_available",
            "is_accompaniment_model_available",
            "from download_accompaniment_model import",
            "Leap XE vocals or PolarFormer accompaniment assets missing/invalid",
            "is_transkun_v2_aug_model_available",
            "download_transkun_v2_aug_model.py",
            "TransKun V2 Aug model missing or checksum validation failed",
        ):
            self.assertIn(expected, script)

    def test_linux_installer_downloads_aria_amt_model_and_requires_python_311(self):
        script = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertIn("download_aria_amt_model.py", script)
        self.assertIn("python3.11", script)
        self.assertIn("Aria-AMT", script)
        self.assertIn("requirements-without-aria-amt.txt", script)
        self.assertIn("audio-separator==0.44.1", script)
        self.assertIn(
            '"aria-amt @ https://github.com/EleutherAI/aria-amt/archive/'
            'a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip" --no-deps',
            script,
        )
        self.assertIn("AriaAmtTranscriber.get_unavailable_reason()", script)
        self.assertIn("--force-reinstall", script)

    def test_linux_installer_downloads_bytedance_pedal_model(self):
        script = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertIn("download_bytedance_piano_model.py", script)
        self.assertIn("ByteDance Piano", script)

    def test_linux_installer_reports_dependency_import_failures_under_set_e(self):
        script = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertIn('if dep_output=$("$PYTHON" -c', script)
        self.assertIn('error "  $name import failed (full error shown above)"', script)
        self.assertNotIn("DEPS_OK", script)
        self.assertIn('error "  ffmpeg 未找到；音频转换所需运行时不完整"', script)

    def test_linux_installer_stops_when_required_model_downloads_fail(self):
        script = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

        for expected in (
            'if ! "$PYTHON" "${REPO_DIR}/download_sota_models.py"; then',
            'error "统一模型集合下载或校验失败"',
            "download_multistem_model.py",
            'if ! "$PYTHON" "${REPO_DIR}/download_vocal_model.py"; then',
            'error "Leap XE vocals model download or verification failed"',
            'if ! "$PYTHON" "${REPO_DIR}/download_accompaniment_model.py"; then',
            'error "PolarFormer accompaniment model download or verification failed"',
            'if ! "$PYTHON" "${REPO_DIR}/download_transkun_v2_aug_model.py"; then',
            'error "TransKun V2 Aug model download or verification failed"',
        ):
            self.assertIn(expected, script)

        self.assertNotIn("vocal_rvc ensemble", script)
        self.assertNotIn("karaoke ensemble", script)
        self.assertNotIn("download_vocal_harmony_model.py", script)
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

    def test_project_metadata_does_not_advertise_a_dependencyless_cli_or_os_independence(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        package_init = (REPO_ROOT / "src" / "__init__.py").read_text(encoding="utf-8")

        self.assertIn('version = "1.0.18"', pyproject)
        self.assertIn('__version__ = "1.0.18"', package_init)
        self.assertNotIn("Operating System :: OS Independent", pyproject)
        self.assertIn("Operating System :: Microsoft :: Windows", pyproject)
        self.assertIn("Operating System :: POSIX :: Linux", pyproject)
        self.assertNotIn("[project.scripts]", pyproject)
        self.assertNotIn('music-to-midi = "src.main:main"', pyproject)

    def test_requirements_avoid_audio_separator_numpy_resolver_conflict(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("numpy>=1.26.0,<2", requirements)
        self.assertNotIn(
            "audio-separator",
            "\n".join(
                line for line in requirements.splitlines() if not line.lstrip().startswith("#")
            ),
        )

    def test_requirements_avoid_aria_amt_torchaudio_resolver_conflict(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        active_requirements = "\n".join(
            line for line in requirements.splitlines() if not line.lstrip().startswith("#")
        )

        self.assertNotIn("aria-amt @", active_requirements)
        self.assertIn(
            "ariautils @ https://github.com/EleutherAI/aria-utils/archive/"
            "93da092204e5b1189ed8e0259f6156266fd086a7.zip",
            active_requirements,
        )
        self.assertIn("safetensors>=0.4.0,<1", active_requirements)
        self.assertIn("orjson>=3.9.0,<4", active_requirements)

    def test_requirements_include_miros_runtime_basics(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("einops>=0.7.0,<1", requirements)
        self.assertIn("smart-open>=6.0.0,<8", requirements)
        self.assertIn("h5py", requirements)
        self.assertIn("mirdata", requirements)
        self.assertIn("pretty-midi>=0.2.10,<1", requirements)
        self.assertIn("soxr>=0.3.7,<1", requirements)
        self.assertIn("mido>=1.3.0,<2", requirements)
        self.assertIn("soundfile>=0.12.0,<1", requirements)
        self.assertIn("chardet>=5,<6", requirements)
        self.assertIn('onnxruntime-gpu==1.23.2; platform_system != "Darwin"', requirements)
        self.assertIn('onnxruntime==1.23.2; platform_system == "Darwin"', requirements)

    def test_requirements_include_bytedance_pedal_transitive_runtime_dependency(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("piano-transcription-inference==0.0.6", requirements)
        self.assertIn("transkun==2.0.1", requirements)
        self.assertIn("torchlibrosa>=0.1.0,<0.2", requirements)
        self.assertIn("matplotlib", requirements)

    def test_requirements_match_miros_transformers_version(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("transformers==4.48.3", requirements)

    def test_windows_installer_uses_same_runtime_dependency_pins(self):
        script = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")

        self.assertIn("requests>=2.32.5,<3", script)
        self.assertIn("chardet>=5,<6", script)
        self.assertIn("onnxruntime-gpu==1.23.2", script)
        self.assertIn("CUDAExecutionProvider", script)

    def test_launchers_require_exact_yourmt3_source_and_default_transkun_identities(self):
        for script_name in ("run.ps1", "run.sh"):
            script = (REPO_ROOT / script_name).read_text(encoding="utf-8")

            self.assertIn("validate_patched_yourmt3_source", script)
            self.assertIn("validate_default_transkun_runtime", script)
            self.assertIn("TransKun default runtime:", script)
            self.assertIn("transkun==2.0.1", script)

    def test_installers_repair_and_revalidate_default_transkun_without_masking_failure(self):
        for script_name in ("install.ps1", "install.sh"):
            script = (REPO_ROOT / script_name).read_text(encoding="utf-8")

            self.assertIn("validate_default_transkun_runtime", script)
            self.assertIn("transkun==2.0.1", script)
            self.assertIn("--force-reinstall", script)
            self.assertIn("validate_patched_yourmt3_source", script)

    def test_linux_installer_rejects_unsupported_rocm_full_stack_explicitly(self):
        script = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertIn("检测到 AMD/ROCm", script)
        self.assertIn("不会静默改用 CPU", script)
        self.assertNotIn("download.pytorch.org/whl/rocm5.7", script)
        self.assertNotIn("download.pytorch.org/whl/rocm6.3", script)

    def test_linux_installer_keeps_version_controlled_launcher_as_single_source(self):
        script = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertIn("验证版本控制的启动脚本", script)
        self.assertNotIn("LAUNCH_EOF", script)
        self.assertNotIn('cat > "${REPO_DIR}/run.sh"', script)

    def test_batch_entry_points_expose_powershell_failures_and_forward_arguments(self):
        for script_name, target in (("run.bat", "run.ps1"), ("install.bat", "install.ps1")):
            script = (REPO_ROOT / script_name).read_text(encoding="utf-8")

            self.assertIn("-NoProfile", script)
            self.assertIn(f'-File "%~dp0{target}" %*', script)
            self.assertIn("exit /b", script)
            self.assertNotIn("SilentlyContinue", script)


if __name__ == "__main__":
    unittest.main()
