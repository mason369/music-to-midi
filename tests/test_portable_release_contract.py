import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class PortableReleaseContractTests(unittest.TestCase):
    def test_pyinstaller_spec_removes_only_conflicting_pyqt_vc_runtime_copies(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("_remove_conflicting_pyqt_vc_runtime_binaries", spec)
        self.assertIn("_require_root_vc_runtime_binaries", spec)
        self.assertIn('"pyqt6/qt6/bin/msvcp140.dll"', spec)
        self.assertIn('"pyqt6/qt6/bin/vcruntime140.dll"', spec)
        self.assertIn('"pyqt6/qt6/bin/vcruntime140_1.dll"', spec)
        self.assertNotIn('"pyqt6/qt6/bin/msvcp140_1.dll"', spec)
        self.assertNotIn('"pyqt6/qt6/bin/msvcp140_2.dll"', spec)

    def test_portable_build_preloads_coherent_vc_runtime_in_pyinstaller_children(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")
        bootstrap = REPO_ROOT / "tools" / "pyinstaller_bootstrap" / "sitecustomize.py"

        self.assertTrue(bootstrap.is_file())
        bootstrap_source = bootstrap.read_text(encoding="utf-8")
        self.assertIn("MUSIC_TO_MIDI_BUILD_VC_RUNTIME_DIR", bootstrap_source)
        self.assertIn("ctypes.WinDLL", bootstrap_source)
        self.assertIn("os._exit(86)", bootstrap_source)
        self.assertIn("tools\\pyinstaller_bootstrap", script)
        self.assertIn("MUSIC_TO_MIDI_BUILD_VC_RUNTIME_DIR", script)
        self.assertIn("$env:PYTHONPATH", script)

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

    def test_pyinstaller_spec_excludes_test_only_submodule_trees(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("_exclude_submodule_prefixes", spec)
        self.assertIn("'torch.testing'", spec)
        self.assertIn("'matplotlib.tests'", spec)

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

    def test_pyinstaller_spec_bundles_the_only_beat_this_runtime_and_checkpoint(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("MUSIC_TO_MIDI_BUNDLE_BEAT_THIS_DIR", spec)
        self.assertIn("models/beat_this", spec)
        self.assertIn("copy_metadata('beat-this')", spec)
        self.assertIn("collect_all('beat_this')", spec)

    def test_pyinstaller_spec_bundles_bytedance_pedal_matplotlib_dependency(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")
        excludes_section = spec.split("excludes=[", 1)[1].split("],", 1)[0]

        self.assertIn(
            "matplotlib_datas, matplotlib_binaries, matplotlib_hiddenimports = collect_all(",
            spec,
        )
        self.assertIn("_exclude_submodule_prefixes('matplotlib.tests')", spec)
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
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_SMALL_DIR",
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_MEDIUM_DIR",
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_LARGE_DIR",
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_DIR",
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_ASSETS_DIR",
            "MUSIC_TO_MIDI_BUNDLE_FLUIDSYNTH_DIR",
            "models/muscriptor_small",
            "models/muscriptor_medium",
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

    def test_pyinstaller_spec_includes_conditional_source_runtime_gate(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("'src.utils.source_runtime'", spec)

    def test_pyinstaller_spec_bundles_xpu_web_inference_worker(self):
        spec = (REPO_ROOT / "MusicToMidi.spec").read_text(encoding="utf-8")

        self.assertIn("'src.web_api.inference_process'", spec)

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
        self.assertIn(
            "五种 YourMT3+ checkpoint、MIROS 或 MuScriptor Small / Medium / Large",
            workflow,
        )
        self.assertIn("分离阶段不调用 MIDI 后端", workflow)
        self.assertIn("逐轨显式选择 13 条 MIDI 路线之一", workflow)
        self.assertNotIn("都会各自调用所选后端", workflow)
        self.assertNotIn("每个 stem 独立转写并合并 MIDI", workflow)
        self.assertIn("YourMT3+ (5 checkpoints)", workflow)
        self.assertIn("MIROS (source + pretrained + fine-tuned)", workflow)
        self.assertIn(
            "MuScriptor Small/Medium/Large (three checkpoints + configs + SoundFont + FluidSynth)",
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
        self.assertIn('tar -czf - -C dist "${LINUX_ROLE_NAMES[@]}"', workflow)
        self.assertIn(
            '| split -b 1900M - "${LINUX_SHARED_NAME}-Portable.tar.gz.part"',
            workflow,
        )
        self.assertIn(
            'tar -czf "${LINUX_FRONTEND_NAME}-Portable.tar.gz"',
            workflow,
        )
        self.assertNotIn("Linux 当前只发布桌面 App", workflow)

    def test_release_workflow_uses_low_memory_7z_and_tests_archives(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("-mx=5", workflow)
        self.assertIn("-md=64m", workflow)
        self.assertIn("-mmt=on", workflow)
        self.assertIn("-twim", workflow)
        self.assertIn("-snh", workflow)
        self.assertIn("-v1900m", workflow)
        self.assertIn("7z t", workflow)
        self.assertIn('"${WINDOWS_SHARED_NAME}-Portable.wim.001"', workflow)
        self.assertIn("HARDLINK_PROBE_REL", workflow)
        self.assertIn("os.path.samefile", workflow)
        self.assertEqual(
            workflow.count(
                'HARDLINK_PROBE_REL="_internal/models/audio-separator/BS-Rofo-SW-Fixed.yaml"'
            ),
            3,
        )
        self.assertNotIn("download_checks.json", workflow)
        self.assertIn('"${WINDOWS_FRONTEND_NAME}-Portable.zip"', workflow)
        self.assertNotIn('for WINDOWS_NAME in "${WINDOWS_NAMES[@]}"', workflow)

    def test_release_workflow_uses_python_311_for_portable_builds(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("python-version: '3.11'", workflow)
        self.assertNotIn("cache: 'pip'", workflow)

    def test_release_workflow_builds_portable_backend_and_frontend_on_windows(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        web_build = (REPO_ROOT / "build_web_executables.ps1").read_text(encoding="utf-8")

        self.assertIn(
            "$pythonExe = Join-Path $env:pythonLocation 'python.exe'",
            workflow,
        )
        self.assertIn("& .\\build_web_executables.ps1 -PythonExe $pythonExe", workflow)
        self.assertIn('Join-Path $Root "build_portable.ps1"', web_build)
        self.assertIn('Join-Path $ResolvedDistRoot "MusicToMidi-WebFrontend"', web_build)
        self.assertIn("Join-Path $ResolvedDistRoot $AppName", web_build)
        self.assertIn("Join-Path $ResolvedDistRoot $BackendName", web_build)

    def test_release_workflow_stages_linux_bundle_assets_before_pyinstaller(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("build/portable_assets", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR", workflow)
        self.assertIn("download_aria_amt_model.py", workflow)
        self.assertIn("download_bytedance_piano_model.py", workflow)
        self.assertIn("download_miros_model.py", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_BEAT_THIS_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MIROS_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_SMALL_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_MEDIUM_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_LARGE_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_ASSETS_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_FLUIDSYNTH_DIR", workflow)
        self.assertIn("gated MuScriptor Small/Medium/Large", workflow)
        self.assertIn(".music-to-midi/models/beat_this/final0.ckpt", workflow)

    def test_release_workflow_downloads_and_packages_gated_muscriptor_assets(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("HF_TOKEN: ${{ secrets.HF_TOKEN }}", workflow)
        self.assertIn("python download_sota_models.py", workflow)
        self.assertIn("python download_fluidsynth_runtime.py", workflow)
        self.assertIn('for model_size in ("small", "medium", "large")', workflow)
        self.assertIn("validate_hashes=True", workflow)
        self.assertIn("Packaged MuScriptor Small/Medium/Large assets verified", workflow)
        self.assertIn("download_muscriptor_soundfont", workflow)
        self.assertIn("command -v fluidsynth", workflow)
        linux_system_packages = workflow.split("sudo apt-get install -y \\", 1)[1].split("\n\n", 1)[
            0
        ]
        self.assertIn("fluidsynth \\", linux_system_packages)
        self.assertIn("pip install pyinstaller pytest pytest-timeout", workflow)
        self.assertIn("pytest -vv --timeout=60 --timeout-method=thread", workflow)

    def test_build_portable_collects_and_validates_muscriptor_assets(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        for expected in (
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_SMALL_DIR",
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_MEDIUM_DIR",
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_LARGE_DIR",
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_DIR",
            "MUSIC_TO_MIDI_BUNDLE_MUSCRIPTOR_ASSETS_DIR",
            "MUSIC_TO_MIDI_BUNDLE_FLUIDSYNTH_DIR",
            "MuScriptor Small/Medium/Large portable assets verified",
            "Packaged MuScriptor Small/Medium/Large assets verified",
            "download_sota_models.py after accepting all three Hugging Face model terms",
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
        self.assertIn("Resolve-Path -LiteralPath '.\\dist\\MusicToMidi-App'", workflow)
        self.assertIn("Resolve-Path -LiteralPath '.\\dist\\MusicToMidi-WebBackend'", workflow)
        self.assertIn("Resolve-Path -LiteralPath '.\\dist\\MusicToMidi-WebFrontend'", workflow)
        self.assertIn("foreach ($runtimeRoot in $runtimeRoots)", workflow)
        self.assertNotIn(
            "Copy-Item -LiteralPath '.\\dist\\MusicToMidi-App' -Destination",
            workflow,
        )
        self.assertIn('APP_SMOKE_DIR="$(pwd)/dist/MusicToMidi-App"', workflow)
        self.assertIn(
            'BACKEND_SMOKE_DIR="$(pwd)/dist/MusicToMidi-WebBackend"',
            workflow,
        )
        self.assertIn(
            'FRONTEND_SMOKE_DIR="$(pwd)/dist/MusicToMidi-WebFrontend"',
            workflow,
        )
        self.assertIn('SMOKE_EXE="$APP_SMOKE_DIR/MusicToMidi"', workflow)
        self.assertIn('BACKEND_EXE="$BACKEND_SMOKE_DIR/MusicToMidiBackend"', workflow)
        self.assertIn('FRONTEND_EXE="$FRONTEND_SMOKE_DIR/MusicToMidiFrontend"', workflow)
        self.assertIn("QT_QPA_PLATFORM=offscreen", workflow)
        self.assertNotIn('cp -a ./dist/MusicToMidi/. "$SMOKE_DIR/"', workflow)
        self.assertIn('rm -rf "${RUNTIME_ROOTS[@]}" "$WEB_JOB_DIR"', workflow)
        self.assertIn(
            "Packaged Linux App, WebBackend, and WebFrontend smoke passed",
            workflow,
        )
        self.assertIn('"http://127.0.0.1:18765/api/v1/health"', workflow)
        self.assertIn('"http://127.0.0.1:15173/runtime-config.json"', workflow)
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

    def test_release_workflow_smokes_windows_roles_without_copying_large_trees(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("$runtimeRoots = @(", workflow)
        self.assertNotIn("MusicToMidi-smoke", workflow)
        self.assertNotIn("Copy-Item -LiteralPath '.\\dist\\MusicToMidi-App'", workflow)
        self.assertNotIn("Copy-Item -LiteralPath '.\\dist\\MusicToMidi-WebBackend'", workflow)
        self.assertIn("Move-Item -LiteralPath 'dist/MusicToMidi-App'", workflow)
        self.assertIn("Move-Item -LiteralPath 'dist/MusicToMidi-WebBackend'", workflow)
        self.assertIn("Move-Item -LiteralPath 'dist/MusicToMidi-WebFrontend'", workflow)

    def test_release_workflow_windows_smoke_test_checks_runtime_log(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn(
            "Start-Process -FilePath (Join-Path $appSmokeDir 'MusicToMidi.exe')", workflow
        )
        self.assertIn("$logs = Join-Path $appSmokeDir 'runtime\\logs'", workflow)
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
        self.assertIn("## 音乐转MIDI转换器 ${TAG}", workflow)
        self.assertIn("## Music to MIDI Converter ${TAG}", workflow)
        self.assertIn("不能匿名全自动下载", workflow)
        self.assertIn("anonymous fully automatic download is not available", workflow)
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

        for expected_update in (
            "MuScriptor Small / Medium / Large：v0.3.0 节拍对齐与长音频高质量推理",
            "d73147e75e5b9b0c0a79ebe154587db4fd603e0c",
            "卷帘拍线/强拍/小节底色",
            "MuseScore 音符轨 tempo",
            r"\`SMART\` 模式",
            "生成阶段的硬约束",
            "按 5 秒分片边转写边预览",
            "播放范围严格限制在已经生成的 MIDI",
            "所有转 MIDI 路线统一使用钢琴卷帘与音轨控制",
            r"\`Ctrl/Alt + 滚轮\`",
            r"\`Shift + 滚轮\`",
            "Hugging Face Space 和 Colab 同步七种处理模式",
        ):
            self.assertIn(expected_update, workflow)
        self.assertIn("MusicToMidi-Windows-GPU-Portable.wim.*", workflow)
        self.assertIn("MusicToMidi-Windows-WebFrontend-Portable.zip", workflow)
        self.assertNotIn("MusicToMidi-Windows-GPU-App-Portable.*", workflow)
        self.assertNotIn("MusicToMidi-Windows-GPU-WebBackend-Portable.*", workflow)
        self.assertIn("MusicToMidi-Linux-GPU-Portable.tar.gz.part*", workflow)
        self.assertIn("MusicToMidi-Linux-WebFrontend-Portable.tar.gz", workflow)
        self.assertNotIn("MusicToMidi-Linux-GPU-App-Portable.*", workflow)
        self.assertIn("Linux unified archive restored App, WebBackend, and WebFrontend", workflow)
        self.assertNotIn("MusicToMidi-Windows-CPU-Portable.*", workflow)
        self.assertNotIn("MusicToMidi-Linux-CPU-Portable.*", workflow)
        self.assertIn("从 \\`.wim.001\\` 解压到 NTFS", workflow)
        self.assertIn("三个同级独立目录", workflow)
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

    def test_build_portable_replaces_asset_directories_instead_of_merging_stale_files(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn(
            "Refusing to replace $Label because source and destination are identical",
            script,
        )
        self.assertIn(
            "Remove-Item -LiteralPath $destinationPath -Recurse -Force -ErrorAction Stop",
            script,
        )

    def test_build_portable_supports_disjoint_roots_and_same_volume_hardlinks(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")
        xpu_wrapper = (REPO_ROOT / "build_portable_xpu.ps1").read_text(encoding="utf-8")

        for expected in (
            '[string]$BuildRoot = ""',
            '[string]$DistRoot = ""',
            "Resolve-BuildOutputRoot",
            "Assert-DisjointOutputRoots",
            'Join-Path $ResolvedBuildRoot "portable_assets"',
            'Join-Path $ResolvedBuildRoot "pyinstaller-$Accelerator"',
            "$PyInstallerDistPath = $ResolvedDistRoot",
            "New-Item -ItemType HardLink",
            "[IO.Path]::GetPathRoot($sourcePath)",
            "[IO.Path]::GetPathRoot($destinationPath)",
            "-HardlinkSameVolume:$HardlinkAssetStaging",
        ):
            self.assertIn(expected, script)
        copy_tree = script.split("function Copy-Tree", 1)[1].split(
            "function Assert-XpuPortableRuntimeDlls", 1
        )[0]
        self.assertNotIn("catch", copy_tree)
        for expected in (
            '$hardlinkSource.LinkType -eq "SymbolicLink"',
            "$linkTargets = @($hardlinkSource.Target)",
            "[IO.Path]::GetFullPath($linkTarget)",
            "Symbolic-link target for ${Label} is missing",
            "-Target $hardlinkSource.FullName -ErrorAction Stop",
            "$totalBytes += $hardlinkSource.Length",
        ):
            self.assertIn(expected, copy_tree)
        self.assertIn("-HardlinkAssetStaging", xpu_wrapper)

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

    def test_windows_ffmpeg_download_is_immutable_and_hash_verified(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        installer = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")
        release_tag = "autobuild-2026-07-31-14-10"
        asset_name = "ffmpeg-n7.1.5-12-g1fdbca85aa-win64-gpl-7.1.zip"
        sha256 = "c067a1ca58f4fc4449f4bab0890fbcd65cbb3e5f46e066cf9c768e06c0c1d4d9"

        for source in (workflow, installer):
            self.assertIn(release_tag, source)
            self.assertIn(asset_name, source)
            self.assertIn(sha256, source)
            self.assertIn("Get-FileHash", source)
        self.assertNotIn("choco install ffmpeg", workflow)
        self.assertNotIn("releases/download/latest/ffmpeg", installer)

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

    def test_build_portable_collects_and_validates_beat_this_final0(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn("MUSIC_TO_MIDI_BUNDLE_BEAT_THIS_DIR", script)
        self.assertIn("download_beat_this_model.py", script)
        self.assertIn("Beat This final0 model", script)
        self.assertIn("Packaged Beat This final0 asset verified", script)

    def test_build_portable_collects_real_ffmpeg_binaries_into_bin_layout(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn('Join-Path $FfmpegBundle "bin"', script)
        self.assertIn("lib\\ffmpeg\\tools\\ffmpeg\\bin", script)

    def test_build_portable_requires_cuda_enabled_torch_runtime(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertIn("torch.version.cuda", script)
        self.assertIn("CPU-only PyTorch runtime", script)
        self.assertIn('"torch": "2.7.0"', script)
        self.assertIn('"torchaudio": "2.7.0"', script)
        self.assertIn('"torchvision": "0.22.0"', script)
        self.assertIn('cuda_version != "12.8"', script)
        self.assertIn('installed_version.split("+", 1)[0]', script)
        self.assertNotIn("torch_tuple <", script)
        self.assertNotIn("cuda_tuple <", script)
        self.assertIn("https://download.pytorch.org/whl/cu128", script)

    def test_build_portable_script_uses_ascii_only(self):
        script = (REPO_ROOT / "build_portable.ps1").read_text(encoding="utf-8")

        self.assertEqual(script, script.encode("ascii").decode("ascii"))


if __name__ == "__main__":
    unittest.main()
