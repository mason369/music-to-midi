"""Prepare the local ai4m-miros backend source tree and weights."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from src.core.miros_transcriber import MirosTranscriber

logger = logging.getLogger(__name__)

MIROS_REPO_URL = "https://github.com/amt-os/ai4m-miros.git"
MIROS_FINETUNED_FILE_ID = "1hp-6D1yYvPxXCXDQyXRQRJArle8R-VfB"
MIROS_PRETRAINED_FILE_ID = "1FqqMfcdqeiRr1v7sdrfkqPpr0Vs7e9nZ"
MIROS_MIN_CHECKPOINT_BYTES = 4_000_000_000
MIROS_MIN_PRETRAINED_BYTES = 1_000_000_000


def _log(printer: Optional[Callable[[str], None]], message: str) -> None:
    if printer:
        printer(message)
        return
    logger.info(message)


def _run(command: list[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def _google_drive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _download_google_drive_file(
    file_id: str,
    destination: Path,
    min_size: int,
    printer: Optional[Callable[[str], None]],
) -> None:
    if destination.exists() and destination.stat().st_size >= min_size:
        _log(printer, f"MIROS weight already exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".download")
    if partial.exists():
        partial.unlink()

    _log(printer, f"Downloading MIROS weight: {destination.name}")
    try:
        _run(["curl", "-L", "--fail", "-o", str(partial), _google_drive_url(file_id)])
    except Exception:
        if partial.exists():
            partial.unlink()
        raise

    size = partial.stat().st_size
    if size < min_size:
        partial.unlink()
        raise RuntimeError(
            f"Downloaded MIROS weight is incomplete: {destination.name} "
            f"{size} bytes < {min_size} bytes"
        )

    partial.replace(destination)
    _log(printer, f"MIROS weight saved: {destination}")


def _ensure_repo(repo_dir: Path, printer: Optional[Callable[[str], None]]) -> None:
    if (repo_dir / "main.py").exists() and (repo_dir / "transcribe.py").exists():
        _log(printer, f"MIROS source exists: {repo_dir}")
        return

    if repo_dir.exists() and any(repo_dir.iterdir()):
        raise RuntimeError(f"MIROS source directory is incomplete and not empty: {repo_dir}")

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("git") is None:
        raise RuntimeError("git is required to clone ai4m-miros")

    _log(printer, f"Cloning MIROS source: {MIROS_REPO_URL}")
    _run(["git", "clone", "--depth=1", MIROS_REPO_URL, str(repo_dir)])

    if not (repo_dir / "main.py").exists() or not (repo_dir / "transcribe.py").exists():
        raise RuntimeError(f"MIROS source clone is incomplete: {repo_dir}")


def prepare_miros_model(
    repo_dir: Path | str = Path("external") / "ai4m-miros",
    printer: Optional[Callable[[str], None]] = None,
) -> Path:
    repo = Path(repo_dir)
    _ensure_repo(repo, printer)

    _download_google_drive_file(
        MIROS_FINETUNED_FILE_ID,
        repo / MirosTranscriber.CHECKPOINT_REL_PATH,
        MIROS_MIN_CHECKPOINT_BYTES,
        printer,
    )
    _download_google_drive_file(
        MIROS_PRETRAINED_FILE_ID,
        repo / MirosTranscriber.PRETRAINED_REL_PATH,
        MIROS_MIN_PRETRAINED_BYTES,
        printer,
    )

    missing = [
        str(path)
        for path in (
            repo / MirosTranscriber.CHECKPOINT_REL_PATH,
            repo / MirosTranscriber.PRETRAINED_REL_PATH,
        )
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"MIROS preparation failed; missing files: {', '.join(missing)}")

    _log(printer, f"MIROS source and weights are ready: {repo}")
    return repo


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare ai4m-miros source and model weights")
    parser.add_argument(
        "--repo-dir",
        default=str(Path("external") / "ai4m-miros"),
        help="Target ai4m-miros checkout directory",
    )
    args = parser.parse_args(argv)

    try:
        prepare_miros_model(Path(args.repo_dir), print)
        return 0
    except Exception as exc:
        print(f"[ERROR] MIROS preparation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
