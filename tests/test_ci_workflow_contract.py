import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


class CiWorkflowContractTests(unittest.TestCase):
    def test_release_workflow_uses_node24_compatible_action_majors(self):
        workflow = (WORKFLOWS_DIR / "release.yml").read_text(encoding="utf-8")

        self.assertIn("actions/checkout@v6", workflow)
        self.assertIn("actions/setup-python@v6", workflow)
        self.assertIn("actions/upload-artifact@v7", workflow)
        self.assertIn("actions/download-artifact@v8", workflow)

    def test_build_workflow_uses_node24_compatible_action_majors(self):
        workflow = (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8")

        self.assertIn("actions/checkout@v6", workflow)
        self.assertIn("actions/setup-python@v6", workflow)
        self.assertNotIn("actions/upload-artifact", workflow)

    def test_build_workflow_installs_audio_separator_for_source_tests(self):
        workflow = (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8")

        self.assertIn("audio-separator==0.44.1 --no-deps", workflow)

    def test_build_workflow_installs_pinned_space_web_dependencies_for_import_tests(self):
        workflow = (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8")

        self.assertIn(
            "pip install gradio==6.17.3 fastapi==0.136.3 starlette==1.6.0 soundfile==0.14.0",
            workflow,
        )

    def test_build_workflow_installs_aria_amt_without_resolver_conflict(self):
        workflow = (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8")

        self.assertIn(
            '"aria-amt @ https://github.com/EleutherAI/aria-amt/archive/'
            'a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip" --no-deps',
            workflow,
        )

    def test_build_workflow_installs_pinned_muscriptor_without_resolver_conflict(self):
        workflow = (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8")

        self.assertIn(
            '"muscriptor @ https://github.com/muscriptor/muscriptor/archive/'
            'd73147e75e5b9b0c0a79ebe154587db4fd603e0c.zip" --no-deps',
            workflow,
        )
        self.assertNotIn("patch_muscriptor_runtime.py", workflow)

    def test_build_workflow_installs_audio_separator_runtime_pins(self):
        workflow = (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8")

        self.assertIn("beartype==0.18.5", workflow)
        self.assertIn("onnx-weekly==1.21.0.dev20260223", workflow)
        self.assertIn("onnxruntime-gpu==1.23.2", workflow)
        self.assertIn("samplerate==0.1.0", workflow)
        self.assertIn("six==1.17.0", workflow)

    def test_build_workflow_uses_the_supported_cuda_torch_for_source_tests(self):
        workflow = (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8")

        self.assertIn("torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0", workflow)
        self.assertIn("https://download.pytorch.org/whl/cu128", workflow)
        self.assertNotIn("https://download.pytorch.org/whl/cpu", workflow)

    def test_push_pr_ci_is_contract_only_and_release_owns_portable_builds(self):
        build_workflow = (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8")
        release_workflow = (WORKFLOWS_DIR / "release.yml").read_text(encoding="utf-8")

        self.assertIn("portable artifacts are release-only", build_workflow)
        self.assertIn("no portable artifact", build_workflow)
        self.assertNotIn("pyinstaller MusicToMidi.spec", build_workflow)
        self.assertNotIn("download_sota_models.py", build_workflow)
        self.assertNotIn("MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR=", build_workflow)
        self.assertNotIn("actions/upload-artifact", build_workflow)

        self.assertIn("python download_sota_models.py", release_workflow)
        self.assertIn("pyinstaller MusicToMidi.spec", release_workflow)
        for bundle_name in (
            "AUDIO_SEPARATOR",
            "YOURMT3",
            "ARIA_AMT",
            "BYTEDANCE_PIANO",
            "BEAT_THIS",
            "TRANSKUN_V2_AUG",
            "MIROS",
            "MUSCRIPTOR",
            "MUSCRIPTOR_ASSETS",
            "FLUIDSYNTH",
            "FFMPEG",
        ):
            self.assertIn(
                f"MUSIC_TO_MIDI_BUNDLE_{bundle_name}_DIR=",
                release_workflow,
            )

    def test_build_workflow_does_not_mask_test_failures(self):
        workflow = (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8")

        self.assertIn("pytest tests/ -v --tb=short", workflow)
        self.assertNotIn("pytest tests/ -v --tb=short || true", workflow)

    def test_build_workflow_flake8_is_compatible_with_black_slice_formatting(self):
        workflow = (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8")

        flake8_command = next(
            line.strip() for line in workflow.splitlines() if line.strip().startswith("flake8 src/")
        )
        self.assertIn("--max-line-length=100", flake8_command)
        ignored_codes = flake8_command.split("--ignore=", 1)[1].split()[0].split(",")
        self.assertIn("E203", ignored_codes)

    def test_linux_ci_installs_egl_for_pyqt_qtgui_imports(self):
        build_workflow = (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8")
        release_workflow = (WORKFLOWS_DIR / "release.yml").read_text(encoding="utf-8")

        self.assertIn("libegl1", build_workflow)
        self.assertIn("libegl1", release_workflow)

    def test_linux_source_ci_provides_a_real_qt_audio_output(self):
        workflows = {
            "source": (WORKFLOWS_DIR / "build.yml").read_text(encoding="utf-8"),
            "release": (WORKFLOWS_DIR / "release.yml").read_text(encoding="utf-8"),
        }

        for label, workflow in workflows.items():
            with self.subTest(workflow=label):
                self.assertIn("pulseaudio", workflow)
                self.assertIn("pulseaudio-utils", workflow)
                self.assertIn("module-null-sink", workflow)
                self.assertIn("sink_name=music_to_midi_ci rate=44100 channels=2", workflow)
                self.assertIn("pactl set-default-sink music_to_midi_ci", workflow)

        release_workflow = workflows["release"]
        self.assertLess(
            release_workflow.index("pactl set-default-sink music_to_midi_ci"),
            release_workflow.index("运行便携发布回归测试"),
        )

    def test_windows_release_ci_provides_a_verified_qt_audio_output(self):
        workflow = (WORKFLOWS_DIR / "release.yml").read_text(encoding="utf-8")

        self.assertIn(
            "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip",
            workflow,
        )
        self.assertIn(
            "66FD0A4D9F4896FF41632B7E3D53892C085C4561F53E8AE8D0F0BC10EEDD1CDD",
            workflow,
        )
        self.assertIn("00859AAC6A54B8C1B3C139DE67846E64E7B82DB2", workflow)
        self.assertIn(
            "https://raw.githubusercontent.com/LABSN/sound-ci-helpers/"
            "20a2b9bbd21fd005d8fe89933901506dba84ea4e/windows/devcon.exe",
            workflow,
        )
        self.assertIn(
            "97CFF42F8C0FE4FBDF991273159516BF78090625A933C3983EBD6F62284E329A",
            workflow,
        )
        self.assertIn("VBAudioVACWDM", workflow)
        self.assertIn("Get-AuthenticodeSignature", workflow)
        self.assertIn("Get-PnpDevice -Class AudioEndpoint -PresentOnly", workflow)
        self.assertIn("*CABLE Input*", workflow)
        self.assertIn("QMediaDevices.audioOutputs()", workflow)
        self.assertIn("QMediaDevices.defaultAudioOutput()", workflow)
        self.assertNotIn("choco install vb-cable", workflow.lower())
        self.assertNotIn("VBCABLE_Setup_x64.exe", workflow)
        self.assertLess(
            workflow.index("安装并验证虚拟音频输出 (Windows)"),
            workflow.index("校验 Qt 默认音频输出 (Windows)"),
        )
        self.assertLess(
            workflow.index("校验 Qt 默认音频输出 (Windows)"),
            workflow.index("运行便携发布回归测试"),
        )

    def test_hf_sync_workflow_uses_node24_compatible_action_majors(self):
        workflow = (WORKFLOWS_DIR / "sync_to_hf.yml").read_text(encoding="utf-8")

        self.assertIn("actions/checkout@v6", workflow)

    def test_hf_sync_workflow_packages_space_runtime_downloaders(self):
        workflow = (WORKFLOWS_DIR / "sync_to_hf.yml").read_text(encoding="utf-8")

        for script_name in (
            "download_sota_models.py",
            "download_beat_this_model.py",
            "download_multistem_model.py",
            "download_vocal_model.py",
            "download_vocal_harmony_model.py",
            "download_aria_amt_model.py",
            "download_bytedance_piano_model.py",
            "download_muscriptor_model.py",
            "download_fluidsynth_runtime.py",
        ):
            with self.subTest(script_name=script_name):
                self.assertIn(f"- '{script_name}'", workflow)
                self.assertIn(f"cp {script_name}", workflow)

        self.assertIn("- 'packages.txt'", workflow)
        self.assertIn('cp packages.txt          "$WORK/"', workflow)

    def test_hf_sync_workflow_ships_the_shared_web_mixer_runtime(self):
        workflow = (WORKFLOWS_DIR / "sync_to_hf.yml").read_text(encoding="utf-8")
        space_app = (REPO_ROOT / "space" / "app.py").read_text(encoding="utf-8")

        # space/app.py imports the Qt-free shared mixer runtime; the Space
        # must receive it or it crashes on boot with ModuleNotFoundError.
        self.assertIn("from src.gui.web.track_mixer_runtime import (", space_app)
        self.assertIn("- 'src/gui/web/**'", workflow)
        self.assertIn('cp src/gui/__init__.py "$WORK/src/gui/"', workflow)
        self.assertIn('cp -r src/gui/web "$WORK/src/gui/"', workflow)
        self.assertNotIn('rm -rf "$WORK/src/gui"', workflow)


if __name__ == "__main__":
    unittest.main()
