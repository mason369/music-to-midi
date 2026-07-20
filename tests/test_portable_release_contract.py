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

    def test_release_workflow_runs_official_midi_route_contract(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("tests/test_official_midi_routes.py", workflow)

    def test_release_workflow_smoke_tests_miros_worker_missing_input_before_heavy_import(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("--miros-worker", workflow)
        self.assertIn("miros-worker-missing-input-smoke.json", workflow)
        self.assertNotIn("miros-worker-import-smoke", workflow)
        self.assertIn("_internal\\external\\ai4m-miros", workflow)
        self.assertIn("_internal/external/ai4m-miros", workflow)

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

    def test_pyinstaller_spec_bundles_aria_amt_package_config(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("_collect_aria_amt_config_datas", spec)
        self.assertIn('"config"', spec)
        self.assertIn("aria_amt_config_datas", spec)
        self.assertIn("copy_metadata('aria-amt')", spec)

    def test_pyinstaller_spec_bundles_pinned_piano_backend_metadata(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("copy_metadata('piano-transcription-inference')", spec)
        self.assertIn("copy_metadata('transkun')", spec)

    def test_pyinstaller_spec_bundles_bytedance_pedal_backend(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR", spec)
        self.assertIn("models/bytedance_piano", spec)
        self.assertIn("collect_all('piano_transcription_inference')", spec)
        self.assertIn("collect_all('torchlibrosa')", spec)

    def test_pyinstaller_spec_bundles_bytedance_pedal_matplotlib_dependency(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")
        excludes_section = spec.split("excludes=[", 1)[1].split("],", 1)[0]

        self.assertIn("collect_all('matplotlib')", spec)
        self.assertIn("matplotlib_hiddenimports", spec)
        self.assertNotIn("'matplotlib'", excludes_section)

    def test_pyinstaller_spec_does_not_exclude_pillow(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")
        excludes_section = spec.split("excludes=[", 1)[1].split("],", 1)[0]

        self.assertNotIn("'PIL'", excludes_section)

    def test_pyinstaller_spec_supports_miros_bundle_root(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MIROS_DIR", spec)
        self.assertIn("ai4m-miros", spec)

    def test_pyinstaller_spec_bundles_complete_muscriptor_runtime(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        for expected in (
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_DIR",
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_ASSETS_DIR",
            "MUSIC_TO_MIDI_BUNDLE_FLUIDSYNTH_DIR",
            "models/muscriptor_large",
            "models/muscriptor_assets",
            "resources/fluidsynth",
            "copy_metadata('muscriptor')",
            "collect_all('muscriptor')",
        ):
            self.assertIn(expected, spec)

    def test_pyinstaller_spec_bundles_miros_dynamic_runtime_dependencies(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        for package_name in ("smart_open", "einops", "soundfile", "pretty_midi", "soxr", "mido"):
            self.assertIn(f"collect_all('{package_name}')", spec)
            self.assertIn(f"{package_name}_hiddenimports", spec)

    def test_pyinstaller_spec_bundles_audio_separator_metadata(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("copy_metadata('audio-separator')", spec)
        self.assertIn("collect_all('audio_separator')", spec)

    def test_release_notes_describe_gpu_compatibility_without_overpromising_specific_generations(
        self,
    ):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("与内置 PyTorch/CUDA 兼容的 NVIDIA 显卡", workflow)
        self.assertIn("当前显卡与内置 PyTorch/CUDA 不兼容", workflow)
        self.assertNotIn("GTX 750 Ti 及以上", workflow)

    def test_release_workflow_checks_version_tag_parity_before_mutating_release(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("Version contract verified:", workflow)
        self.assertIn('Path("pyproject.toml")', workflow)
        self.assertIn('Path("src/__init__.py")', workflow)
        self.assertIn('expected_tag = f"v{project_version}"', workflow)
        self.assertLess(
            workflow.index("校验 pyproject、运行时版本与发布标签一致"),
            workflow.index("创建 Release（如不存在）"),
        )

    def test_portable_usage_and_asset_log_cover_all_seven_routes_and_backends(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        for mode in (
            "SMART",
            "VOCAL_SPLIT",
            "SIX_STEM_SPLIT",
            "PIANO_TRANSKUN",
            "PIANO_TRANSKUN_V2_AUG",
            "PIANO_ARIA_AMT",
            "PIANO_BYTEDANCE_PEDAL",
        ):
            self.assertGreaterEqual(workflow.count(mode), 2)
        self.assertIn("YourMT3+、MIROS 或 MuScriptor-large", workflow)
        self.assertIn("都会各自调用所选后端", workflow)
        self.assertIn("YourMT3+ (5 checkpoints)", workflow)
        self.assertIn("MIROS (source + pretrained + fine-tuned)", workflow)
        self.assertIn(
            "MuScriptor-large (checkpoint + config + SoundFont + FluidSynth)",
            workflow,
        )
        self.assertIn("TransKun 2.0.1 default V2", workflow)

    def test_release_workflow_uses_timeout_and_retry_for_release_uploads(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("upload_asset_with_retry", workflow)
        self.assertIn("timeout 30m gh release upload", workflow)

    def test_release_workflow_cleans_build_cache_before_compressing_large_bundles(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("rm -rf build", workflow)
        self.assertIn('"$HOME/.cache/music_ai_models"', workflow)
        self.assertIn('tar -czf - -C dist "${NAME}" | split -b 1900M -', workflow)
        self.assertNotIn('tar -czf "${NAME}-Portable.tar.gz" -C dist "${NAME}"', workflow)

    def test_release_workflow_uses_low_memory_7z_and_tests_archives(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("-mx=5", workflow)
        self.assertIn("-md=64m", workflow)
        self.assertIn("-mmt=on", workflow)
        self.assertIn("7z t", workflow)
        self.assertIn('"${NAME}-Portable.7z.001"', workflow)

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
        self.assertIn("download_aria_amt_model.py", workflow)
        self.assertIn("download_bytedance_piano_model.py", workflow)
        self.assertIn("download_miros_model.py", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MIROS_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_ASSETS_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_FLUIDSYNTH_DIR", workflow)
        self.assertIn("gated MuScriptor-large", workflow)

    def test_release_workflow_downloads_and_packages_gated_muscriptor_assets(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("HF_TOKEN: ${{ secrets.HF_TOKEN }}", workflow)
        self.assertIn("python download_sota_models.py", workflow)
        self.assertIn("python download_fluidsynth_runtime.py", workflow)
        self.assertIn("get_cached_muscriptor_paths(validate_hashes=True)", workflow)
        self.assertIn("download_muscriptor_soundfont", workflow)
        self.assertIn("command -v fluidsynth", workflow)

    def test_build_portable_collects_and_validates_muscriptor_assets(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        for expected in (
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_DIR",
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_ASSETS_DIR",
            "MUSIC_TO_MIDI_BUNDLE_FLUIDSYNTH_DIR",
            "MuScriptor portable assets verified",
            "download_sota_models.py after accepting the Hugging Face terms",
        ):
            self.assertIn(expected, script)

    def test_release_workflow_prepares_miros_from_packaged_release_assets(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("python download_miros_model.py", workflow)
        self.assertIn("MirosTranscriber.is_model_available()", workflow)
        self.assertIn("ls -lh external/ai4m-miros/model/musicfm/data/pretrained_msd.pt", workflow)
        self.assertIn(
            "ls -lh external/ai4m-miros/logs/Multi_longer_seq_length_frozen_enc_silu/le2bzt53/checkpoints/last.ckpt",
            workflow,
        )
        self.assertIn('MIROS_PORTABLE_RELEASE_TAG="v1.0.16"', workflow)
        self.assertIn("MusicToMidi-Linux-GPU-Portable.tar.gz.part", workflow)
        self.assertIn("Streaming packaged MIROS backend", workflow)
        self.assertIn("tar -xz", workflow)
        self.assertIn("_internal/external/ai4m-miros", workflow)
        self.assertNotIn("canonical Google Drive source", workflow)
        self.assertNotIn("miros-last.ckpt.partaa", workflow)
        self.assertNotIn('if [ -d "$GITHUB_WORKSPACE/.tmp/ai4m-miros" ]', workflow)

    def test_release_workflow_smoke_tests_built_binary(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("--self-test-no-load", workflow)
        self.assertIn('Join-Path $env:RUNNER_TEMP "MusicToMidi-smoke"', workflow)
        self.assertIn('SMOKE_DIR="$(pwd)/dist/MusicToMidi"', workflow)
        self.assertIn('SMOKE_EXE="$SMOKE_DIR/MusicToMidi"', workflow)
        self.assertIn("QT_QPA_PLATFORM=offscreen", workflow)
        self.assertNotIn('cp -a ./dist/MusicToMidi/. "$SMOKE_DIR/"', workflow)
        self.assertIn('rm -rf "$SMOKE_DIR/runtime"', workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MIROS_DIR", workflow)

    def test_release_workflow_self_test_has_timeout_and_log_diagnostics(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("$SelfTestTimeoutSeconds = 900", workflow)
        self.assertIn("$proc.WaitForExit($SelfTestTimeoutSeconds * 1000)", workflow)
        self.assertIn("$proc.Kill($true)", workflow)
        self.assertIn("Portable self-test timed out after ${SelfTestTimeoutSeconds}s", workflow)
        self.assertIn("SELF_TEST_TIMEOUT_SECONDS=900", workflow)
        self.assertIn("timeout-minutes: 20", workflow)
        self.assertIn(
            'timeout --signal=TERM --kill-after=30s "${SELF_TEST_TIMEOUT_SECONDS}s"', workflow
        )
        self.assertIn('"$SMOKE_EXE" --self-test-no-load', workflow)
        self.assertIn('2>&1 | tee "$SELF_TEST_OUTPUT"', workflow)
        self.assertIn("SELF_TEST_EXIT=${PIPESTATUS[0]}", workflow)
        self.assertIn('[ "$SELF_TEST_EXIT" -eq 137 ]', workflow)
        self.assertIn("dump_linux_portable_logs", workflow)
        self.assertIn(
            "Portable self-test did not write the success marker to runtime logs", workflow
        )
        self.assertIn("MIROS_WORKER_TIMEOUT_SECONDS=120", workflow)
        self.assertIn("MIROS_WORKER_OUTPUT=", workflow)
        self.assertIn("$MirosWorkerTimeoutSeconds = 120", workflow)
        self.assertIn("$worker.WaitForExit($MirosWorkerTimeoutSeconds * 1000)", workflow)
        self.assertIn(
            "MIROS worker missing-input smoke timed out after ${MirosWorkerTimeoutSeconds}s",
            workflow,
        )
        self.assertIn(
            'timeout --signal=TERM --kill-after=30s "${MIROS_WORKER_TIMEOUT_SECONDS}s"', workflow
        )
        self.assertIn('"$SMOKE_EXE" \\', workflow)
        self.assertIn('2>&1 | tee "$MIROS_WORKER_OUTPUT"', workflow)
        self.assertIn("WORKER_EXIT=${PIPESTATUS[0]}", workflow)
        self.assertIn('[ "$WORKER_EXIT" -eq 137 ]', workflow)
        self.assertIn("dump_miros_worker_logs", workflow)
        self.assertIn("MIROS worker missing-input smoke returned an unexpected status", workflow)
        self.assertIn("MIROS input audio does not exist", workflow)
        self.assertIn(
            "MIROS worker missing-input smoke failed before the expected missing-input check",
            workflow,
        )

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
        self.assertIn("onnxruntime-gpu==1.23.2", workflow)
        self.assertIn("audio-separator==0.44.1 --no-deps", workflow)
        self.assertIn(
            '"aria-amt @ https://github.com/EleutherAI/aria-amt/archive/'
            'a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip" --no-deps',
            workflow,
        )
        self.assertIn("six==1.17.0", workflow)

    def test_release_workflow_matches_supported_torch_runtime(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0", workflow)
        self.assertIn("https://download.pytorch.org/whl/cu128", workflow)
        self.assertNotIn("https://download.pytorch.org/whl/cpu", workflow)
        self.assertNotIn("torch==2.4.0", workflow)
        self.assertNotIn("cu121", workflow)

    def test_release_workflow_builds_only_gpu_portable_variants(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("variant: GPU", workflow)
        self.assertIn("platform: Windows", workflow)
        self.assertIn("platform: Linux", workflow)
        self.assertNotIn("variant: CPU", workflow)
        self.assertNotIn("MusicToMidi-Windows-CPU", workflow)
        self.assertNotIn("MusicToMidi-Linux-CPU", workflow)

    def test_release_workflow_removes_stale_cpu_assets_before_upload(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("删除旧 CPU 发布资产", workflow)
        self.assertIn('select(test("^MusicToMidi-.*-CPU-Portable"))', workflow)
        self.assertIn("gh release delete-asset", workflow)
        self.assertLess(
            workflow.index("删除旧 CPU 发布资产"),
            workflow.index("上传资源到 Release"),
        )

    def test_release_workflow_updates_existing_release_notes(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("更新 Release 说明", workflow)
        self.assertIn("gh release edit", workflow)
        self.assertIn("--notes-file release-notes.md", workflow)
        self.assertLess(
            workflow.index("更新 Release 说明"),
            workflow.index("上传资源到 Release"),
        )

    def test_release_workflow_filters_pinned_runtime_packages_from_requirements_build(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn('"pytorch-lightning"', workflow)
        self.assertIn('"torchmetrics"', workflow)
        self.assertIn('"onnxruntime"', workflow)

    def test_release_notes_describe_split_archives_generically(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("MusicToMidi-Windows-GPU-Portable.*", workflow)
        self.assertIn("MusicToMidi-Linux-GPU-Portable.*", workflow)
        self.assertNotIn("MusicToMidi-Windows-CPU-Portable.*", workflow)
        self.assertNotIn("MusicToMidi-Linux-CPU-Portable.*", workflow)
        self.assertIn("同名前缀的全部分卷", workflow)
        self.assertIn("当前包含 7 种处理模式", workflow)
        self.assertNotIn("旧版 6 种处理模式", workflow)
        self.assertIn("ByteDance Pedal", workflow)

    def test_build_portable_collects_miros_bundle_assets(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MIROS_DIR", script)
        self.assertIn("ai4m-miros", script)
        self.assertIn("external\\ai4m-miros", script)
        self.assertIn("Required asset missing", script)
        self.assertNotIn("[skip] $Label not found", script)

    def test_build_portable_validates_six_stem_assets(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn("Assert-SixStemAssets", script)
        self.assertIn("download_multistem_model.py", script)
        self.assertIn("--check-only", script)
        self.assertIn("audio-separator source", script)
        self.assertIn("audio-separator bundle", script)

    def test_release_workflow_validates_six_stem_assets_after_sota_download(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("BS-RoFormer SW six-stem assets", workflow)
        self.assertIn("python download_multistem_model.py --check-only", workflow)

    def test_build_portable_fails_when_clean_or_pyinstaller_fails(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn("function Remove-PathIfExists", script)
        self.assertIn("Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop", script)
        self.assertIn("$PyInstallerExitCode = $LASTEXITCODE", script)
        self.assertIn("PyInstaller build failed with exit code", script)
        self.assertNotIn(
            'Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $Root "build")',
            script,
        )
        self.assertNotIn(
            'Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $Root "dist")',
            script,
        )

    def test_pyinstaller_spec_collects_miros_from_external_checkout(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn('"external", "ai4m-miros"', spec)

    def test_packaging_fails_when_required_assets_or_ffmpeg_tools_are_missing(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("Required portable bundle directory is missing", spec)
        self.assertIn("_require_ffmpeg_tools(ffmpeg_dir)", spec)
        self.assertIn('FFMPEG_SOURCE="$(command -v ffmpeg)" || {', workflow)
        self.assertIn('FFPROBE_SOURCE="$(command -v ffprobe)" || {', workflow)
        self.assertIn('test -s "$BUILD_ASSET_ROOT/ffmpeg/bin/ffprobe"', workflow)

    def test_release_records_exact_ffmpeg_license_build_and_hash_evidence(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        for expected in (
            "FFMPEG_BUILD_AUDIT.txt",
            "Package source:",
            "Package version:",
            "ffmpeg -version",
            "ffmpeg -buildconf",
            "ffmpeg -L",
            "sha256sum",
            "--enable-nonfree",
            "is not redistributable; refusing portable build",
        ):
            self.assertIn(expected, workflow)
        self.assertGreaterEqual(workflow.count("FFMPEG_BUILD_AUDIT.txt"), 4)

    def test_build_portable_collects_aria_amt_bundle_assets(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn("MUSIC_TO_MIDI_BUNDLE_ARIA_AMT_DIR", script)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_ARIA_DIR", script)
        self.assertIn("aria_amt", script)

    def test_build_portable_collects_bytedance_pedal_bundle_assets(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn("MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR", script)
        self.assertIn("bytedance_piano", script)
        self.assertIn("ByteDance Piano models", script)

    def test_build_portable_collects_real_ffmpeg_binaries_into_bin_layout(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn('Join-Path $FfmpegBundle "bin"', script)
        self.assertIn("lib\\ffmpeg\\tools\\ffmpeg\\bin", script)

    def test_build_portable_requires_cuda_enabled_torch_runtime(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn("torch.version.cuda", script)
        self.assertIn("CPU-only PyTorch runtime", script)
        self.assertIn("(2, 7, 0)", script)
        self.assertIn("(12, 8)", script)
        self.assertIn("https://download.pytorch.org/whl/cu128", script)

    def test_build_portable_script_uses_ascii_only(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertEqual(script, script.encode("ascii").decode("ascii"))


if __name__ == "__main__":
    unittest.main()
