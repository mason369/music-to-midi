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

    @staticmethod
    def _load_notebook_all_source_text():
        notebook_path = Path("colab_notebook.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        sources = []
        for cell in notebook.get("cells", []):
            sources.append("".join(cell.get("source", [])))
        return "\n".join(sources)

    def test_restored_modes_and_dependencies_are_present(self):
        source_text = self._load_notebook_source_text()

        for restored_text in (
            "aria-amt",
            "Aria-AMT",
            "Transkun",
            "六声部分离 + 分别转写",
            "six_stem_split",
            "download_multistem_model.py",
            "download_aria_amt_model.py",
        ):
            with self.subTest(restored_text=restored_text):
                self.assertIn(restored_text, source_text)

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
            'for module_name in ["torch", "torchaudio", "torchvision", "gradio", "huggingface_hub", "lightning", "librosa"]:',
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

    def test_colab_intro_and_ui_match_restored_mode_surface(self):
        all_source_text = self._load_notebook_all_source_text()
        code_source_text = self._load_notebook_source_text()

        for expected_text in (
            "当前 Colab 版提供五种处理模式",
            "完整混音多乐器转写（SMART）",
            "人声/伴奏分离后分别转写（VOCAL_SPLIT）",
            "六声部分离 + 分别转写",
            "钢琴专用转写 (Transkun)",
            "钢琴专用转写 (Aria-AMT)",
        ):
            with self.subTest(expected_text=expected_text):
                self.assertIn(expected_text, all_source_text)

        for expected_ui_text in (
            "输出说明",
            "_accompaniment.mid",
            "_vocal.mid",
            "_vocal_accompaniment_merged.mid",
        ):
            with self.subTest(expected_ui_text=expected_ui_text):
                self.assertIn(expected_ui_text, code_source_text)


if __name__ == "__main__":
    unittest.main()
