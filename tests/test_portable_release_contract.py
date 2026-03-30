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

    def test_release_workflow_uses_timeout_and_retry_for_release_uploads(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("upload_asset_with_retry", workflow)
        self.assertIn("timeout 30m gh release upload", workflow)

    def test_release_workflow_uses_portable_build_script_on_windows(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("powershell -ExecutionPolicy Bypass -File .\\build_portable.ps1", workflow)

    def test_release_workflow_stages_linux_bundle_assets_before_pyinstaller(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("build/portable_assets", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR", workflow)
        self.assertIn("MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR", workflow)


if __name__ == "__main__":
    unittest.main()
