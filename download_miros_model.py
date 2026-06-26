"""Prepare the local ai4m-miros backend source tree and weights."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlencode, urljoin, urlparse

from src.core.miros_transcriber import MirosTranscriber

logger = logging.getLogger(__name__)

MIROS_REPO_URL = "https://github.com/amt-os/ai4m-miros.git"
MIROS_FINETUNED_FILE_ID = "1hp-6D1yYvPxXCXDQyXRQRJArle8R-VfB"
MIROS_PRETRAINED_URL = "https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt"
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


class _GoogleDriveConfirmParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_download_form = False
        self.action: Optional[str] = None
        self.fields: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attributes = {key: value for key, value in attrs}
        if tag == "form" and attributes.get("id") == "download-form":
            self._in_download_form = True
            self.action = attributes.get("action")
            return
        if self._in_download_form and tag == "input":
            name = attributes.get("name")
            value = attributes.get("value")
            if name and value is not None:
                self.fields[name] = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._in_download_form:
            self._in_download_form = False


class _GoogleDriveErrorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._capture: Optional[str] = None
        self.title: list[str] = []
        self.error_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        classes = set(attributes.get("class", "").split())
        if tag == "title":
            self._capture = "title"
            return
        if tag in {"p", "div"} and classes.intersection({"uc-error-caption", "uc-error-subcaption"}):
            self._capture = "error"

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._capture == "title":
            self._capture = None
        elif tag in {"p", "div"} and self._capture == "error":
            self._capture = None

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._capture == "title":
            self.title.append(text)
        elif self._capture == "error":
            self.error_text.append(text)


def _confirmed_google_drive_url(download_page: Path) -> Optional[str]:
    text = download_page.read_text(encoding="utf-8", errors="ignore")
    if "download-form" not in text:
        return None

    parser = _GoogleDriveConfirmParser()
    parser.feed(text)
    if not parser.action or not parser.fields:
        return None

    action = urljoin("https://drive.google.com/", parser.action)
    parsed = urlparse(action)
    if parsed.netloc not in {"drive.usercontent.google.com", "drive.google.com"}:
        raise RuntimeError(f"Unexpected Google Drive confirmation host: {parsed.netloc}")

    return f"{action}?{urlencode(parser.fields)}"


def _google_drive_error_message(download_page: Path) -> Optional[str]:
    text = download_page.read_text(encoding="utf-8", errors="ignore")
    lower_text = text.lower()
    if "google drive" not in lower_text:
        return None

    parser = _GoogleDriveErrorParser()
    parser.feed(text)
    details = " ".join(parser.error_text or parser.title).strip()
    if "quota exceeded" in lower_text or "too many users have viewed or downloaded this file" in lower_text:
        return (
            "Google Drive quota exceeded for the official MIROS fine-tuned checkpoint"
            + (f": {details}" if details else "")
        )
    if "<html" in lower_text:
        return (
            "Google Drive returned an HTML page instead of checkpoint bytes"
            + (f": {details}" if details else "")
        )
    return None


def _download_http_file(
    url: str,
    destination: Path,
    min_size: int,
    printer: Optional[Callable[[str], None]],
    label: str,
) -> None:
    if destination.exists() and destination.stat().st_size >= min_size:
        _log(printer, f"{label} already exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".download")
    if partial.exists():
        partial.unlink()

    _log(printer, f"Downloading {label}: {destination.name}")
    try:
        _run(["curl", "-L", "--fail", "-o", str(partial), url])
    except Exception:
        if partial.exists():
            partial.unlink()
        raise

    size = partial.stat().st_size
    if size < min_size:
        partial.unlink()
        raise RuntimeError(
            f"Downloaded {label} is incomplete: {destination.name} "
            f"{size} bytes < {min_size} bytes"
        )

    partial.replace(destination)
    _log(printer, f"{label} saved: {destination}")


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
        if partial.stat().st_size < min_size:
            confirmed_url = _confirmed_google_drive_url(partial)
            if confirmed_url:
                _log(printer, f"Confirming Google Drive download: {destination.name}")
                partial.unlink()
                _run(["curl", "-L", "--fail", "-o", str(partial), confirmed_url])
    except Exception:
        if partial.exists():
            partial.unlink()
        raise

    size = partial.stat().st_size
    if size < min_size:
        drive_error = _google_drive_error_message(partial)
        partial.unlink()
        if drive_error:
            raise RuntimeError(f"{destination.name}: {drive_error}")
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
    _download_http_file(
        MIROS_PRETRAINED_URL,
        repo / MirosTranscriber.PRETRAINED_REL_PATH,
        MIROS_MIN_PRETRAINED_BYTES,
        printer,
        "MIROS MusicFM pretrained weight",
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
