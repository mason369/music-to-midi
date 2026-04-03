import json
import re
import unittest
from pathlib import Path


class TestColabNotebookDependencies(unittest.TestCase):
    @staticmethod
    def _load_notebook_source_text():
        notebook_path = Path("colab_notebook.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        sources = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                sources.append("".join(cell.get("source", [])))
        return "\n".join(sources)

    def test_aria_amt_requirement_is_shell_safe(self):
        source_text = self._load_notebook_source_text()

        self.assertIn(
            "aria-amt@git+https://github.com/EleutherAI/aria-amt.git",
            source_text,
        )
        self.assertNotIn(
            "aria-amt @ git+https://github.com/EleutherAI/aria-amt.git",
            source_text,
        )

    def test_notebook_preserves_preinstalled_torch_and_avoids_reinstall(self):
        source_text = self._load_notebook_source_text()
        package_block_match = re.search(
            r"packages = \[\n(?P<block>.*?)\n\]",
            source_text,
            flags=re.S,
        )
        self.assertIsNotNone(package_block_match)
        package_block = package_block_match.group("block")

        self.assertIn(
            "检测 Colab 预装 torch 版本",
            source_text,
        )
        self.assertIn(
            'log(f"torch=={torch.__version__}")',
            source_text,
        )
        self.assertIn(
            'log(f"CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}")',
            source_text,
        )
        self.assertNotIn('"torchaudio"', package_block)

    def test_pip_install_uses_shell_safe_quoting(self):
        source_text = self._load_notebook_source_text()
        self.assertIn("import shlex", source_text)
        self.assertIn(
            "quoted_packages = \" \".join(shlex.quote(pkg) for pkg in packages)",
            source_text,
        )
        self.assertIn(
            "run_cmd(\"python -m pip install \" + quoted_packages)",
            source_text,
        )

    def test_post_install_logs_torch_family_versions(self):
        source_text = self._load_notebook_source_text()
        self.assertIn(
            "关键包版本",
            source_text,
        )
        self.assertIn(
            'for module_name in ["torch", "torchaudio", "torchvision", "gradio", "huggingface_hub", "lightning", "librosa", "amt"]:',
            source_text,
        )
        self.assertIn(
            "importlib.import_module(module_name)",
            source_text,
        )
        self.assertIn(
            'log(f"{module_name}=={version}")',
            source_text,
        )


if __name__ == "__main__":
    unittest.main()
