import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PortableReleaseContractTests(unittest.TestCase):
    def test_torch_openmp_repair_helper_exists(self):
        helper = REPO_ROOT / "tools" / "repair_torch_openmp.py"

        self.assertTrue(helper.exists(), "Expected reusable Torch OpenMP repair helper to exist")
        source = helper.read_text(encoding="utf-8")
        self.assertIn("libomp140.x86_64.dll", source)
        self.assertIn("def main(", source)

    def test_release_workflow_invokes_torch_openmp_repair_helper(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("repair_torch_openmp.py", workflow)

    def test_build_portable_invokes_torch_openmp_repair_helper(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn("repair_torch_openmp.py", script)

    def test_pyinstaller_spec_bundles_lightning_dependencies(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("pytorch_lightning", spec)
        self.assertIn("lightning_fabric", spec)
        self.assertIn("lightning_utilities", spec)
        self.assertIn("torchmetrics", spec)
        self.assertIn("collect_all('wandb')", spec)
        self.assertIn("collect_all('PIL')", spec)
        self.assertIn("collect_all('onnxruntime')", spec)
        self.assertIn("collect_all('mir_eval')", spec)

    def test_pyinstaller_spec_does_not_exclude_pillow(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")
        excludes_section = spec.split("excludes=[", 1)[1].split("],", 1)[0]

        self.assertNotIn("'PIL'", excludes_section)

    def test_pyinstaller_spec_supports_miros_bundle_root(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MIROS_DIR", spec)
        self.assertIn("ai4m-miros", spec)

    def test_pyinstaller_spec_bundles_audio_separator_metadata(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("copy_metadata('audio-separator')", spec)

    def test_release_workflow_uses_timeout_and_retry_for_release_uploads(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("upload_asset_with_retry", workflow)
        self.assertIn("timeout 30m gh release upload", workflow)

    def test_release_workflow_uses_python_311_for_portable_builds(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("python-version: '3.11'", workflow)

    def test_release_workflow_uses_portable_build_script_on_windows(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("pwsh -ExecutionPolicy Bypass -File .\\build_portable.ps1", workflow)

    def test_release_workflow_stages_linux_bundle_assets_before_pyinstaller(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("build/portable_assets", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR", workflow)

    def test_release_workflow_smoke_tests_built_binary(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("--self-test", workflow)
        self.assertIn('Join-Path $env:RUNNER_TEMP "MusicToMidi-smoke"', workflow)
        self.assertIn('mktemp -d "${RUNNER_TEMP:-/tmp}/MusicToMidi-smoke.XXXXXX"', workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MIROS_DIR", workflow)

    def test_release_workflow_isolates_windows_smoke_test_and_package_rename(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("MusicToMidi-smoke", workflow)
        self.assertIn("Start-Sleep -Seconds 3", workflow)
        self.assertIn("Move-Item -LiteralPath 'dist/MusicToMidi'", workflow)

    def test_release_workflow_windows_smoke_test_checks_runtime_log(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("Start-Process -FilePath (Join-Path $smokeDir 'MusicToMidi.exe')", workflow)
        self.assertIn("$logs = Join-Path $smokeDir 'runtime\\logs'", workflow)
        self.assertIn("Get-ChildItem -LiteralPath $logs -File", workflow)
        self.assertIn("便携包自检通过", workflow)

    def test_release_workflow_installs_audio_separator_without_resolver_conflicts(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("requirements-build.txt", workflow)
        self.assertIn("pip install numpy==1.26.4", workflow)
        self.assertIn("Pillow==12.0.0", workflow)
        self.assertIn("pytorch-lightning==2.6.1", workflow)
        self.assertIn("torchmetrics==1.8.2", workflow)
        self.assertIn("onnxruntime==1.23.2", workflow)
        self.assertIn("audio-separator==0.41.1 --no-deps", workflow)
        self.assertIn("six==1.17.0", workflow)

    def test_release_workflow_filters_pinned_runtime_packages_from_requirements_build(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn('"pytorch-lightning"', workflow)
        self.assertIn('"torchmetrics"', workflow)
        self.assertIn('"onnxruntime"', workflow)

    def test_release_notes_describe_split_archives_generically(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("MusicToMidi-Windows-CPU-Portable.*", workflow)
        self.assertIn("MusicToMidi-Linux-CPU-Portable.*", workflow)
        self.assertIn("同名前缀的全部分卷", workflow)

    def test_build_portable_collects_miros_bundle_assets(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MIROS_DIR", script)
        self.assertIn("ai4m-miros", script)

    def test_build_portable_collects_real_ffmpeg_binaries_into_bin_layout(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn('Join-Path $FfmpegBundle "bin"', script)
        self.assertIn("lib\\ffmpeg\\tools\\ffmpeg\\bin", script)

    def test_build_portable_script_uses_ascii_only(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertEqual(script, script.encode("ascii").decode("ascii"))


if __name__ == "__main__":
    unittest.main()
