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
        self.assertIn("actions/upload-artifact@v7", workflow)

    def test_hf_sync_workflow_uses_node24_compatible_action_majors(self):
        workflow = (WORKFLOWS_DIR / "sync_to_hf.yml").read_text(encoding="utf-8")

        self.assertIn("actions/checkout@v6", workflow)


if __name__ == "__main__":
    unittest.main()
