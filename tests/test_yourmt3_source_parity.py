import json
import subprocess
import sys
from pathlib import Path

from src.utils.yourmt3_source_identity import (
    PATCHED_YOURMT3_MANIFEST_FILE_COUNT,
    PATCHED_YOURMT3_MANIFEST_SHA256,
    validate_patched_yourmt3_source,
)


def test_desktop_space_and_colab_use_the_same_patched_yourmt3_source_tree():
    space_source = Path("space/app.py").read_text(encoding="utf-8")
    sync_workflow = Path(".github/workflows/sync_to_hf.yml").read_text(encoding="utf-8")
    notebook = json.loads(Path("colab_notebook.ipynb").read_text(encoding="utf-8"))
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert '("hf-root", app_root)' in space_source
    assert '("repository-parent", app_root.parent)' in space_source
    assert "validate_patched_yourmt3_source" in space_source
    assert "CONTROLLED_PROJECT_ROOTS" in space_source
    assert "snapshot_download" not in space_source
    assert 'cp -r YourMT3/amt/src "$WORK/YourMT3/amt/"' in sync_workflow
    assert "- 'YourMT3/amt/src/**'" in sync_workflow
    assert "/content/music-to-midi/YourMT3/amt/src" in notebook_source
    assert "validate_patched_yourmt3_source(amt_src)" in notebook_source
    assert "snapshot_download" not in notebook_source
    assert "list_repo_files" not in notebook_source

    manifest_sha256, file_count = validate_patched_yourmt3_source("YourMT3/amt/src")
    assert manifest_sha256 == PATCHED_YOURMT3_MANIFEST_SHA256
    assert file_count == PATCHED_YOURMT3_MANIFEST_FILE_COUNT


def test_space_imports_from_the_repository_parent_layout_when_started_in_space_dir():
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        "import app; "
        "print(app.SPACE_PROJECT_LAYOUT); "
        "print(app.PROJECT_ROOT); "
        "assert app.SPACE_PROJECT_LAYOUT == 'repository-parent'; "
        f"assert app.PROJECT_ROOT == __import__('pathlib').Path({str(repo_root)!r})"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root / "space",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert "repository-parent" in completed.stdout
