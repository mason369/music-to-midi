import json
import unittest
from pathlib import Path


class TestColabNotebookDependencies(unittest.TestCase):
    def test_aria_amt_requirement_is_shell_safe(self):
        notebook_path = Path("colab_notebook.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        sources = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                sources.append("".join(cell.get("source", [])))
        source_text = "\n".join(sources)

        self.assertIn(
            "aria-amt@git+https://github.com/EleutherAI/aria-amt.git",
            source_text,
        )
        self.assertNotIn(
            "aria-amt @ git+https://github.com/EleutherAI/aria-amt.git",
            source_text,
        )


if __name__ == "__main__":
    unittest.main()
