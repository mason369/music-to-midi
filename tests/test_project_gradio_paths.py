"""Exercise real Gradio file serving for portable project path namespaces."""

import gradio as gr
from fastapi.testclient import TestClient

from src.gui.web.track_mixer_runtime import gradio_allowed_paths, track_file_url
from src.gui.web.project_files import cache_project_file
from src.projects.store import project_path


def test_gradio_serves_project_files_in_both_namespaces_without_exposing_siblings(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    audio = root / "音轨 with spaces.wav"
    audio.write_bytes(b"project audio")
    private = tmp_path / "unrelated.txt"
    private.write_bytes(b"outside project")
    with gr.Blocks() as demo:
        gr.Markdown("Project file access")
        published = [cache_project_file(path) for path in (audio, project_path(audio))]
    demo.allowed_paths = gradio_allowed_paths(root)
    with TestClient(gr.routes.App.create_app(demo)) as client:
        for path in published:
            response = client.get(track_file_url(path))
            assert response.status_code == 200, response.text
            assert response.content == audio.read_bytes()
        for path in (private, project_path(private)):
            response = client.get(track_file_url(path))
            assert response.status_code == 403, response.text
    assert set(gradio_allowed_paths(root, project_path(root))) == set(demo.allowed_paths)
