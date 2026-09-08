"""Project lifecycle for browser sessions and long-lived local servers."""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path

from src.projects import ProjectRunner, ProjectStore, Workflow
from src.projects.store import PROJECT_FILE


class ProjectService:
    def __init__(self, root, engine_factory=None):
        self.root = Path(root).resolve()
        self.engine_factory = engine_factory
        self._lock = threading.Lock()
        self._runs = {}
        self._errors = {}
        self._closed = False

    def store(self, project_id):
        if not isinstance(project_id, str) or not re.fullmatch(r"[0-9a-f]{32}", project_id):
            raise ValueError("项目 ID 无效")
        store = ProjectStore(self.root / project_id)
        if not store.path.is_file():
            raise FileNotFoundError("项目不存在")
        return store

    def snapshot(self, project_id):
        result = self.store(project_id).snapshot()
        result["storage_id"] = project_id
        with self._lock:
            run = self._runs.get(project_id)
            result["active"] = bool(run and run[1].is_alive()) or self.store(project_id).is_busy()
            if project_id in self._errors:
                result["service_error"] = self._errors[project_id]
        if not result["active"] and result["status"] == "running":
            result["status"] = "interrupted"
            for song in result["songs"]:
                if song["status"] == "running":
                    song["status"] = "interrupted"
            result["counts"]["interrupted"] += result["counts"]["running"]
            result["counts"]["running"] = 0
        return result

    def list(self):
        results = []
        for path in self.root.glob(f"*/{PROJECT_FILE}"):
            if re.fullmatch(r"[0-9a-f]{32}", path.parent.name):
                results.append(self.snapshot(path.parent.name))
        return sorted(results, key=lambda value: value["updated_at"], reverse=True)

    def create(self, name, workflow=None):
        project_id = uuid.uuid4().hex
        ProjectStore(self.root / project_id).create(name, workflow or Workflow())
        return self.snapshot(project_id)

    def start(self, project_id, song_ids=None, track_ids=None, primary_only=False):
        store = self.store(project_id)
        with self._lock:
            if self._closed:
                raise RuntimeError("项目服务已关闭")
            existing = self._runs.get(project_id)
            if existing and existing[1].is_alive():
                raise RuntimeError("项目正在执行")
            runner = ProjectRunner(store, self.engine_factory)

            def execute():
                try:
                    runner.run(song_ids, track_ids=track_ids, primary_only=primary_only)
                except Exception as exc:
                    with self._lock:
                        self._errors[project_id] = f"{type(exc).__name__}: {exc}"

            thread = threading.Thread(target=execute, name=f"project-{project_id}", daemon=False)
            self._errors.pop(project_id, None)
            self._runs[project_id] = (runner, thread)
            thread.start()
        return self.snapshot(project_id)

    def cancel(self, project_id):
        with self._lock:
            run = self._runs.get(project_id)
        if run and run[1].is_alive():
            run[0].cancel()
        return self.snapshot(project_id)

    def close(self):
        with self._lock:
            self._closed = True
            runs = list(self._runs.values())
        for runner, thread in runs:
            if thread.is_alive():
                runner.cancel()
        for _, thread in runs:
            thread.join()
