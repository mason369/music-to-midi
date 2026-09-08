"""Versioned JSON projects, verified assets and process-safe atomic writes.

A project owns its input copies and outputs. A workflow contains intentions;
checkpoints contain evidence. Editing a workflow never starts inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.utils.inference_lock import _try_lock_file, _unlock_file
from src.web_api.schemas import InferenceOptions, ManualMidiOptions

PROJECT_FILE = "music-to-midi-project.json"
PROJECT_FORMAT = "music-to-midi-project"
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
SPLIT_STEMS = {
    "vocal_split": ("vocals", "accompaniment"),
    "six_stem_split": ("bass", "drums", "guitar", "piano", "vocals", "other"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def step_signature(kind: str, source_sha: str, original_sha: str, options: dict) -> str:
    identity_options = (
        {"processing_mode": options["processing_mode"]}
        if kind == "primary" and options["processing_mode"] in SPLIT_STEMS
        else {
            key: value
            for key, value in options.items()
            if key not in {"language", "gpu_device", "use_gpu"}
        }
    )
    return fingerprint(
        {
            "contract": 1,
            "kind": kind,
            "source": source_sha,
            "original": original_sha,
            "options": identity_options,
        }
    )


def select_checkpoints(song) -> None:
    """Resolve the current plan without deleting previous model results."""
    primary_key = step_signature(
        "primary",
        song.source.sha256,
        song.source.sha256,
        song.workflow.primary.model_dump(mode="json"),
    )
    primary = song.checkpoints.get(primary_key)
    song.primary_key = primary_key if primary else None
    song.track_keys = {}
    if primary and primary.status == "succeeded":
        for stem, options in song.workflow.stems.items():
            source = song.extra_tracks.get(stem) or next(
                (a for a in primary.artifacts if a.kind == "audio_track" and a.track_id == stem),
                None,
            )
            if source:
                key = step_signature(
                    "manual_midi",
                    source.sha256,
                    song.source.sha256,
                    options.model_dump(mode="json"),
                )
                if key in song.checkpoints:
                    song.track_keys[stem] = key


def plan_complete(song) -> bool:
    if not song.primary_key or song.checkpoints[song.primary_key].status != "succeeded":
        return False
    return all(
        stem in song.track_keys and song.checkpoints[song.track_keys[stem]].status == "succeeded"
        for stem in song.workflow.stems
    )


def atomic_json(path: Path, value: object) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive(path: Path):
    """The OS releases ownership on process death; a stale file is not a lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b", buffering=0) as stream:
        if path.stat().st_size == 0:
            stream.write(b"\0")
        if not _try_lock_file(stream):
            raise RuntimeError("项目正在其他窗口或进程中使用，请等待当前操作结束")
        try:
            yield
        finally:
            _unlock_file(stream)


@contextmanager
def state_access(root: Path):
    """Serialize reads with atomic replacement, including on Windows NTFS.

    Run ownership remains a separate, nonblocking lock. This short critical
    section covers only JSON IO; inference never holds the state mutex.
    """
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class Overlapped(ctypes.Structure):
            _fields_ = [
                ("Internal", ctypes.c_size_t),
                ("InternalHigh", ctypes.c_size_t),
                ("Offset", wintypes.DWORD),
                ("OffsetHigh", wintypes.DWORD),
                ("hEvent", wintypes.HANDLE),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LockFileEx.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(Overlapped),
        )
        kernel.LockFileEx.restype = wintypes.BOOL
        kernel.UnlockFileEx.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(Overlapped),
        )
        kernel.UnlockFileEx.restype = wintypes.BOOL
        with (root / ".state.lock").open("a+b", buffering=0) as stream:
            handle = msvcrt.get_osfhandle(stream.fileno())
            overlap = Overlapped()
            if not kernel.LockFileEx(handle, 2, 0, 1, 0, ctypes.byref(overlap)):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                yield
            finally:
                if not kernel.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(overlap)):
                    raise ctypes.WinError(ctypes.get_last_error())
    else:
        import fcntl

        with (root / ".state.lock").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Workflow(StrictModel):
    name: str = "默认工作流"
    primary: InferenceOptions = Field(default_factory=InferenceOptions)
    stems: dict[str, ManualMidiOptions] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_stems(self):
        allowed = SPLIT_STEMS.get(self.primary.processing_mode, ())
        custom = {
            key for key in self.stems if allowed and re.fullmatch(r"custom-[0-9a-f]{32}", key)
        }
        if set(self.stems) - set(allowed) - custom:
            raise ValueError(f"当前模式不支持这些声部：{sorted(set(self.stems) - set(allowed))}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "Workflow":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        atomic_json(Path(path), self.reusable().model_dump(mode="json"))

    def reusable(self) -> "Workflow":
        """External track identities belong to a song, never to a reusable preset."""
        workflow = self.model_copy(deep=True)
        workflow.stems = {
            key: value for key, value in workflow.stems.items() if not key.startswith("custom-")
        }
        return workflow


class Asset(StrictModel):
    path: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    id: str = "source"
    kind: str = "source"
    track_id: str | None = None
    source_artifact_id: str | None = None
    media_type: str = "application/octet-stream"

    @model_validator(mode="after")
    def relative_path(self):
        p = PurePosixPath(self.path)
        if (
            not self.path
            or p.is_absolute()
            or ".." in p.parts
            or "\\" in self.path
            or ":" in self.path
        ):
            raise ValueError("项目资源必须使用项目内的相对路径")
        return self


class Checkpoint(StrictModel):
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: Literal["primary", "manual_midi"]
    options: dict
    status: Literal["running", "succeeded", "failed", "cancelled", "interrupted"]
    output_dir: str
    artifacts: list[Asset] = Field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    started_at: str = Field(default_factory=now)
    finished_at: str | None = None


class TrackView(StrictModel):
    muted: bool = False
    solo: bool = False
    gain_db: float = Field(default=0, ge=-60, le=0)
    offset: float = Field(default=0, ge=-10, le=10)


class Song(StrictModel):
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    name: str
    source: Asset
    workflow: Workflow
    checkpoints: dict[str, Checkpoint] = Field(default_factory=dict)
    primary_key: str | None = None
    track_keys: dict[str, str] = Field(default_factory=dict)
    mixer: dict[str, TrackView] = Field(default_factory=dict)
    extra_tracks: dict[str, Asset] = Field(default_factory=dict)
    status: Literal["pending", "running", "succeeded", "failed", "cancelled", "interrupted"] = (
        "pending"
    )
    error: str | None = None


class ProjectDocument(StrictModel):
    format: Literal["music-to-midi-project"] = PROJECT_FORMAT
    schema_version: Literal[1] = 1
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r"^[0-9a-f]{32}$")
    name: str = "音乐项目"
    revision: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)
    workflow: Workflow = Field(default_factory=Workflow)
    songs: list[Song] = Field(default_factory=list)
    status: Literal["pending", "running", "succeeded", "failed", "cancelled", "interrupted"] = (
        "pending"
    )
    progress: dict = Field(default_factory=dict)
    error: str | None = None
    last_run_status: Literal["succeeded", "failed", "cancelled"] | None = None

    @model_validator(mode="after")
    def unique_ids(self):
        if len({song.id for song in self.songs}) != len(self.songs):
            raise ValueError("项目歌曲 ID 重复")
        for song in self.songs:
            if any(key != step.signature for key, step in song.checkpoints.items()):
                raise ValueError("项目断点签名与索引不一致")
            if any(
                key.startswith("custom-") and key not in song.extra_tracks
                for key in song.workflow.stems
            ):
                raise ValueError("项目工作流引用了不存在的外部音轨")
        return self


def project_path(value: str | Path) -> Path:
    """Use Win32 extended paths without requiring a machine registry change.

    Checkpoints still store portable relative paths. The prefix is only used
    for filesystem access, including filenames created below the project root.
    """
    path = Path(value).expanduser().absolute()
    if os.name == "nt":
        native = str(path)
        if not native.startswith("\\\\?\\"):
            native = (
                "\\\\?\\UNC\\" + native[2:]
                if native.startswith("\\\\") else "\\\\?\\" + native
            )
        path = Path(native)
    return path.resolve()


class ProjectStore:
    def __init__(self, root: str | Path):
        candidate = project_path(root)
        self.root = candidate.parent if candidate.name == PROJECT_FILE else candidate
        self.path = self.root / PROJECT_FILE

    @property
    def run_lock(self) -> Path:
        return self.root / ".run.lock"

    def is_busy(self) -> bool:
        if not self.run_lock.exists():
            return False
        with self.run_lock.open("r+b", buffering=0) as stream:
            if not _try_lock_file(stream):
                return True
            _unlock_file(stream)
            return False

    def load(self) -> ProjectDocument:
        with state_access(self.root):
            return self._read()

    def _read(self):
        return ProjectDocument.model_validate_json(self.path.read_text(encoding="utf-8"))

    def create(self, name: str = "音乐项目", workflow: Workflow | None = None) -> ProjectDocument:
        self.root.mkdir(parents=True, exist_ok=True)
        with exclusive(self.run_lock), state_access(self.root):
            if self.path.exists():
                raise FileExistsError(f"项目已存在：{self.path}")
            document = ProjectDocument(name=name, workflow=workflow or Workflow())
            atomic_json(self.path, document.model_dump(mode="json"))
            return document

    def save(self, document: ProjectDocument) -> None:
        """CAS prevents stale UI snapshots from erasing progress or other edits."""
        with state_access(self.root):
            current = self._read()
            if current.id != document.id or current.revision != document.revision:
                raise RuntimeError("项目已被更新，请重新加载后再保存")
            validated = ProjectDocument.model_validate(document.model_dump())
            validated.revision += 1
            validated.updated_at = now()
            atomic_json(self.path, validated.model_dump(mode="json"))
            document.revision = validated.revision
            document.updated_at = validated.updated_at

    def resolve(self, relative: str) -> Path:
        p = PurePosixPath(relative)
        if p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative:
            raise ValueError("项目资源路径越界")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("项目资源路径越界")
        return path

    def verify(self, asset: Asset) -> Path:
        path = self.resolve(asset.path)
        if (
            not path.is_file()
            or path.stat().st_size != asset.size
            or digest_file(path) != asset.sha256
        ):
            raise RuntimeError(f"项目资源丢失或内容发生变化：{asset.path}")
        return path

    def asset(self, path: Path, **values) -> Asset:
        resolved = project_path(path)
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise ValueError(f"资源不在项目中：{path}")
        return Asset(
            path=resolved.relative_to(self.root).as_posix(),
            size=resolved.stat().st_size,
            sha256=digest_file(resolved),
            **values,
        )

    def add(self, paths: list[str | Path], workflow: Workflow | None = None) -> list[str]:
        """Import serially, committing each verified input before the next one."""
        with exclusive(self.run_lock):
            document = self.load()
            added = []
            for value in paths:
                source = project_path(value)
                if (
                    source.suffix.lower() not in AUDIO_SUFFIXES
                    or not source.is_file()
                    or source.stat().st_size <= 0
                ):
                    raise ValueError(f"输入音频不存在、为空或格式不支持：{source}")
                sha = digest_file(source)
                existing = next(
                    (song for song in document.songs if song.source.sha256 == sha), None
                )
                if existing:
                    self.verify(existing.source)
                    added.append(existing.id)
                    continue
                song_id = uuid.uuid4().hex
                destination = (
                    self.root / "songs" / song_id / "input" / ("source" + source.suffix.lower())
                )
                destination.parent.mkdir(parents=True)
                try:
                    with source.open("rb") as incoming, destination.open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                        outgoing.flush()
                        os.fsync(outgoing.fileno())
                    asset = self.asset(destination)
                    if asset.sha256 != sha:
                        raise RuntimeError(f"导入时原始音频发生变化：{source}")
                    document.songs.append(
                        Song(
                            id=song_id,
                            name=source.name,
                            source=asset,
                            workflow=(workflow or document.workflow).model_copy(deep=True),
                        )
                    )
                    document.status = "pending"
                    self.save(document)
                except BaseException:
                    # Only this uncommitted, newly allocated import directory is disposable.
                    if not any(song.id == song_id for song in self.load().songs):
                        shutil.rmtree(destination.parent.parent)
                    raise
                added.append(song_id)
            return added

    def configure(
        self,
        workflow: Workflow,
        song_ids: list[str] | None = None,
        revision: int | None = None,
        *,
        clear_custom_stems: bool = False,
    ) -> ProjectDocument:
        with exclusive(self.run_lock):
            document = self.load()
            if revision is not None and document.revision != revision:
                raise RuntimeError("项目已被更新，请重新加载后再保存")
            ids = set(song_ids if song_ids is not None else (s.id for s in document.songs))
            if ids - {s.id for s in document.songs}:
                raise ValueError("找不到所选歌曲")
            custom = {
                key: value for key, value in workflow.stems.items() if key.startswith("custom-")
            }
            available = {
                key for song in document.songs if song.id in ids for key in song.extra_tracks
            }
            if set(custom) - available:
                raise ValueError("工作流引用了所选歌曲中不存在的外部音轨")
            document.workflow = workflow.reusable()
            for song in document.songs:
                if song.id not in ids:
                    continue
                configured = workflow.reusable()
                if configured.primary.processing_mode in SPLIT_STEMS:
                    if not clear_custom_stems:
                        configured.stems.update(
                            {
                                key: value.model_copy(deep=True)
                                for key, value in song.workflow.stems.items()
                                if key in song.extra_tracks
                            }
                        )
                    configured.stems.update(
                        {
                            key: value.model_copy(deep=True)
                            for key, value in custom.items()
                            if key in song.extra_tracks
                        }
                    )
                if song.workflow != configured:
                    song.workflow = configured
                    select_checkpoints(song)
                    song.status = "succeeded" if plan_complete(song) else "pending"
                    song.error = None
            document.status = (
                "succeeded"
                if document.songs and all(s.status == "succeeded" for s in document.songs)
                else "pending"
            )
            self.save(document)
            return document

    def snapshot(self) -> dict:
        document = self.load()
        result = document.model_dump(mode="json")
        result["counts"] = {
            status: sum(song.status == status for song in document.songs)
            for status in ("pending", "running", "succeeded", "failed", "cancelled", "interrupted")
        }
        result["counts"]["total"] = len(document.songs)
        return result

    def update_song_state(
        self,
        song_id: str,
        mixer: dict[str, TrackView],
        stems: dict[str, ManualMidiOptions],
        revision: int | None = None,
    ):
        with exclusive(self.run_lock):
            document = self.load()
            if revision is not None and document.revision != revision:
                raise RuntimeError("项目已被更新，请重新加载后再保存")
            song = next((s for s in document.songs if s.id == song_id), None)
            if song is None:
                raise ValueError("找不到所选歌曲")
            workflow = song.workflow.model_dump(mode="json")
            workflow["stems"] = {key: value.model_dump(mode="json") for key, value in stems.items()}
            song.workflow = Workflow.model_validate(workflow)
            allowed = set(SPLIT_STEMS.get(song.workflow.primary.processing_mode, ())) | set(
                song.extra_tracks
            )
            if set(mixer) - allowed:
                raise ValueError("混音设置包含未知音轨")
            song.mixer = mixer
            select_checkpoints(song)
            song.status = "succeeded" if plan_complete(song) else "pending"
            document.status = (
                "succeeded" if all(s.status == "succeeded" for s in document.songs) else "pending"
            )
            self.save(document)
            return document

    def add_track(self, song_id: str, source_path: str | Path) -> str:
        with exclusive(self.run_lock):
            document = self.load()
            song = next((s for s in document.songs if s.id == song_id), None)
            if song is None or song.workflow.primary.processing_mode not in SPLIT_STEMS:
                raise ValueError("只能向分离项目歌曲添加音轨")
            source = project_path(source_path)
            if not source.is_file() or source.suffix.lower() not in AUDIO_SUFFIXES:
                raise ValueError("外部音轨不存在或格式不支持")
            identity = digest_file(source)
            key = "custom-" + identity[:32]
            if key in song.extra_tracks:
                self.verify(song.extra_tracks[key])
                return key
            target = self.root / "songs" / song_id / "tracks" / key / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            asset = self.asset(target, id=key, kind="audio_track", track_id=key)
            if asset.sha256 != identity:
                raise RuntimeError("导入时外部音轨内容发生变化")
            song.extra_tracks[key] = asset
            self.save(document)
            return key

    def export_archive(self, destination: str | Path) -> Path:
        target = project_path(destination)
        with exclusive(self.run_lock):
            document = self.load()
            assets = {song.source.path: song.source for song in document.songs}
            for song in document.songs:
                assets.update({a.path: a for a in song.extra_tracks.values()})
                for checkpoint in song.checkpoints.values():
                    if checkpoint.status == "succeeded":
                        assets.update({asset.path: asset for asset in checkpoint.artifacts})
            for asset in assets.values():
                self.verify(asset)
            if target == self.path or target in (self.resolve(p) for p in assets):
                raise ValueError("项目包不能覆盖项目资源")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                with zipfile.ZipFile(
                    temporary, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True
                ) as archive:
                    archive.writestr(PROJECT_FILE, document.model_dump_json(indent=2))
                    for path in sorted(assets):
                        archive.write(self.resolve(path), path)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def import_archive(
        cls, archive_path: str | Path, destination: str | Path, max_bytes: int = 64 * 1024**3
    ) -> "ProjectStore":
        root = project_path(destination)
        if root.exists():
            raise FileExistsError(f"导入目标已存在：{root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".project-import-", dir=root.parent))
        try:
            with zipfile.ZipFile(project_path(archive_path)) as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
                if len(names) != len(set(names)) or sum(e.file_size for e in entries) > max_bytes:
                    raise ValueError("项目包存在重复资源或超出大小限制")
                if (
                    PROJECT_FILE not in names
                    or archive.getinfo(PROJECT_FILE).file_size > 32 * 1024**2
                ):
                    raise ValueError("缺少有效项目清单或清单过大")
                document = ProjectDocument.model_validate_json(archive.read(PROJECT_FILE))
                assets = {s.source.path: s.source for s in document.songs}
                for song in document.songs:
                    assets.update({a.path: a for a in song.extra_tracks.values()})
                    for step in song.checkpoints.values():
                        if step.status == "succeeded":
                            assets.update({a.path: a for a in step.artifacts})
                if set(names) != {PROJECT_FILE, *assets}:
                    raise ValueError("项目包文件与清单不一致")
                store = cls(temporary)
                for name, asset in assets.items():
                    entry = archive.getinfo(name)
                    if (
                        entry.file_size != asset.size
                        or ((entry.external_attr >> 16) & 0o170000) == 0o120000
                    ):
                        raise ValueError("项目资源大小不一致或包含符号链接")
                    path = store.resolve(name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as source, path.open("xb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
                    store.verify(asset)
                # A project transferred without its process cannot still be running.
                if document.status == "running":
                    document.status = "interrupted"
                for song in document.songs:
                    if song.status == "running":
                        song.status = "interrupted"
                atomic_json(store.path, document.model_dump(mode="json"))
            os.rename(temporary, root)
            return cls(root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
