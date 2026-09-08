"""Project data must survive paths longer than the Win32 MAX_PATH boundary."""
from pathlib import Path
import os

import pytest

from src.projects import ProjectRunner, ProjectStore
from src.projects.store import project_path
from tests.test_persistent_projects import Engine, audio


def test_long_project_paths_round_trip_and_reuse_without_inference(tmp_path):
    root = tmp_path / ("long-project-" + "a" * 95) / ("nested-" + "b" * 90)
    store = ProjectStore(root)
    store.create()
    source = audio(tmp_path / "input.wav")
    store.add([source])
    assert ProjectRunner(store, Engine).run()["status"] == "succeeded"
    song = store.load().songs[0]
    artifact = song.checkpoints[song.primary_key].artifacts[0]
    assert len(str(store.verify(artifact))) > 260
    assert not artifact.path.startswith(("/", "\\"))
    assert ":" not in artifact.path
    # Both caller spellings must resolve to one project asset.
    assert store.asset(root / artifact.path, id="same", kind="midi").sha256 == artifact.sha256
    archive = store.export_archive(root / "saved-project.mtmproject")
    imported = ProjectStore.import_archive(archive, root.parent / ("restored-" + "c" * 80))

    def forbidden():
        raise AssertionError("An unchanged imported project repeated inference")

    assert ProjectRunner(imported, forbidden).run()["status"] == "succeeded"
    assert imported.verify(imported.load().songs[0].source).is_file()
    with pytest.raises(ValueError, match="覆盖"):
        store.export_archive(root / artifact.path)


def test_project_path_is_idempotent_and_platform_native(tmp_path):
    path = project_path(tmp_path)
    assert project_path(path) == path
    if os.name == "nt":
        assert str(path).startswith("\\\\?\\")
    else:
        assert path == tmp_path.resolve()
