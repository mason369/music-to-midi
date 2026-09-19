"""Pinned, unmodified ChordMini BTC resources for source and frozen apps."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from src.utils.runtime_paths import get_resource_path

CHORDMINI_COMMIT = "aa6e3a8d7b017f082fd2aaff9329d5c26af49c03"
CHECKPOINT_SHA256 = "e0a12ca6d881f81e01dfb459b4836ec7dac8d8f87b0170a412b5807256b3a1fc"
ARCHIVE_SHA256 = "e164737fd7b0b07158c50fa0da299f5bc0f69188e06b2e3eb607f2735c1cd6f2"
RECIPE_ID = "telknet-chordmini-btc-logit-overlap50-gaussian9-v1"
SOURCE_URL = f"https://codeload.github.com/ptnghia-j/ChordMini/zip/{CHORDMINI_COMMIT}"


def source_manifest() -> dict[str, str]:
    return json.loads(
        get_resource_path("resources/chordmini-source.json").read_text(encoding="utf-8")
    )


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_chordmini(root: Path) -> Path:
    """Hash actual bytes; Windows timestamps can miss same-size rapid rewrites."""
    root = Path(root).resolve()
    for relative, expected in source_manifest().items():
        if file_sha256(root / relative) != expected:
            raise RuntimeError(f"ChordMini resource identity mismatch: {relative}")
    return root


def chordmini_cache_dir() -> Path:
    return Path.home() / ".cache/music_ai_models/chordmini" / CHORDMINI_COMMIT


def get_chordmini_runtime() -> Path:
    bundled = get_resource_path("models/chordmini")
    candidate = bundled if bundled.is_dir() else chordmini_cache_dir()
    try:
        return validate_chordmini(candidate)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ChordMini BTC resources missing; run python download_chordmini_model.py"
        ) from exc


def prepare_chordmini(*, archive: Path | None = None) -> Path:
    """Prepare one atomic snapshot; inference never downloads or substitutes models."""
    target = chordmini_cache_dir()
    if target.exists():
        return validate_chordmini(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=target.parent) as work:
        work = Path(work)
        if archive is None:
            archive = work / "source.zip"
            with (
                urllib.request.urlopen(SOURCE_URL, timeout=60) as response,
                archive.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
        if file_sha256(Path(archive)) != ARCHIVE_SHA256:
            raise RuntimeError("ChordMini source archive identity mismatch")
        stage = work / "ChordMini"
        with zipfile.ZipFile(archive) as source:
            for relative in source_manifest():
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read(f"ChordMini-{CHORDMINI_COMMIT}/{relative}"))
        validate_chordmini(stage)
        try:
            stage.rename(target)
        except FileExistsError:
            validate_chordmini(target)
    return validate_chordmini(target)
