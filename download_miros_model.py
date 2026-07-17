"""Prepare the local ai4m-miros backend source tree and weights."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlencode, urljoin, urlparse

import src.core.miros_transcriber as miros_runtime
from src.core.miros_transcriber import MirosTranscriber

logger = logging.getLogger(__name__)

MIROS_REPO_URL = "https://github.com/amt-os/ai4m-miros.git"
MIROS_FINETUNED_FILE_ID = "1hp-6D1yYvPxXCXDQyXRQRJArle8R-VfB"
MIROS_SOURCE_COMMIT = miros_runtime.MIROS_SOURCE_COMMIT
MIROS_UNPATCHED_SOURCE_SHA256 = miros_runtime.MIROS_UNPATCHED_SOURCE_SHA256
MIROS_PRETRAINED_COMMIT = miros_runtime.MIROS_PRETRAINED_COMMIT
MIROS_PRETRAINED_URL = (
    f"https://huggingface.co/minzwon/MusicFM/resolve/"
    f"{MIROS_PRETRAINED_COMMIT}/pretrained_msd.pt"
)
MIROS_PRETRAINED_EXACT_BYTES = miros_runtime.MIROS_PRETRAINED_EXACT_BYTES
MIROS_PRETRAINED_SHA256 = miros_runtime.MIROS_PRETRAINED_SHA256
MIROS_FINETUNED_EXACT_BYTES = miros_runtime.MIROS_FINETUNED_EXACT_BYTES
MIROS_FINETUNED_SHA256 = miros_runtime.MIROS_FINETUNED_SHA256
MIROS_MIRROR_DIR_ENV = "MUSIC_TO_MIDI_MIROS_MIRROR_DIR"
MIROS_DECMOD_REL_PATH = Path("model/decmod.py")
MIROS_ROPE_REL_PATH = Path("model/RoPE/RoPE.py")
MIROS_FLEX_ATTENTION_BAD_IMPORT = (
    "    from ...integrations.flex_attention import make_flex_block_causal_mask"
)
MIROS_ROPE_OLD_IMPORT = "from torch.cuda.amp import autocast"
MIROS_ROPE_NEW_IMPORT = "from torch.amp import autocast"
MIROS_ROPE_OLD_DECORATOR = "@autocast(enabled=False)"
MIROS_ROPE_NEW_DECORATOR = '@autocast("cuda", enabled=False)'
MIROS_TRANSCRIBE_REL_PATH = Path("transcribe.py")
MIROS_AUDIO_SEGMENTS_GPU_BLOCK = (
    "    audio_segments = torch.from_numpy(audio_segments.astype('float32'))"
    ".to(device).unsqueeze(1) # (n_seg, 1, seg_sz)"
)
MIROS_AUDIO_SEGMENTS_CPU_BLOCK = "\n".join(
    (
        "    # Keep the complete song on CPU. YourMT3.inference_file moves only the",
        "    # current bounded batch to the model device.",
        "    audio_segments = torch.from_numpy(audio_segments.astype('float32'))"
        ".unsqueeze(1) # (n_seg, 1, seg_sz)",
    )
)
MIROS_INFERENCE_CONTEXT_OLD = "    with torch.cuda.amp.autocast(dtype=torch.bfloat16):"
MIROS_INFERENCE_CONTEXT_NEW = "\n".join(
    (
        "    with torch.inference_mode(), torch.autocast(",
        "            device_type=device.type,",
        "            dtype=torch.bfloat16,",
        '            enabled=device.type == "cuda"):',
    )
)
MIROS_INFERENCE_BATCH_OLD = (
    "        pred_token_arr, _ = model.inference_file(bsz=8, " "audio_segments=audio_segments)"
)
MIROS_INFERENCE_BATCH_NEW = (
    "        pred_token_arr, _ = model.inference_file(bsz=1, " "audio_segments=audio_segments)"
)


def _log(printer: Optional[Callable[[str], None]], message: str) -> None:
    if printer:
        printer(message)
        return
    logger.info(message)


def _run(command: list[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def _google_drive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _verify_checkpoint_container(path: Path) -> None:
    with path.open("rb") as checkpoint:
        header = checkpoint.read(4)
    if not header.startswith(b"PK"):
        return

    try:
        with zipfile.ZipFile(path):
            pass
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"MIROS checkpoint container is invalid or incomplete: {path}") from exc


def _temporary_file_for(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".download",
        dir=str(destination.parent),
    )
    os.close(handle)
    return Path(name)


def _require_file_identity(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    error = miros_runtime.get_miros_file_identity_error(
        path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        label=label,
    )
    if error:
        raise RuntimeError(error)


def _require_source_identity(repo_dir: Path) -> None:
    error = miros_runtime.get_miros_source_identity_error(repo_dir)
    if error:
        raise RuntimeError(error)


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
        if tag in {"p", "div"} and classes.intersection(
            {"uc-error-caption", "uc-error-subcaption"}
        ):
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
    if (
        "quota exceeded" in lower_text
        or "too many users have viewed or downloaded this file" in lower_text
    ):
        return "Google Drive quota exceeded for the official MIROS fine-tuned checkpoint" + (
            f": {details}" if details else ""
        )
    if "<html" in lower_text:
        return "Google Drive returned an HTML page instead of checkpoint bytes" + (
            f": {details}" if details else ""
        )
    return None


def _download_http_file(
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    printer: Optional[Callable[[str], None]],
    label: str,
) -> None:
    if destination.exists():
        _require_file_identity(
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=label,
        )
        _log(printer, f"{label} already exists: {destination}")
        return

    partial = _temporary_file_for(destination)

    _log(printer, f"Downloading {label}: {destination.name}")
    try:
        _run(["curl", "-L", "--fail", "-o", str(partial), url])
        _require_file_identity(
            partial,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=f"downloaded {label}",
        )
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()
    _log(printer, f"{label} saved: {destination}")


def _copy_verified_file(
    source: Path,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    _require_file_identity(
        source,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        label=f"mirrored {label}",
    )
    partial = _temporary_file_for(destination)
    try:
        shutil.copyfile(source, partial)
        _require_file_identity(
            partial,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=f"copied {label}",
        )
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def _restore_from_mirror(
    mirror_dir: Path,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    printer: Optional[Callable[[str], None]],
    label: str,
) -> bool:
    for direct in (mirror_dir / destination.name, mirror_dir / f"miros-{destination.name}"):
        if direct.is_file():
            _log(printer, f"Restoring MIROS weight from release mirror: {direct}")
            _copy_verified_file(
                direct,
                destination,
                expected_size,
                expected_sha256,
                label,
            )
            return True

    parts = sorted(mirror_dir.glob(f"{destination.name}.part*"))
    if not parts:
        parts = sorted(mirror_dir.glob(f"miros-{destination.name}.part*"))
    if not parts:
        return False

    _log(printer, f"Restoring MIROS weight from release mirror parts: {destination.name}")
    partial = _temporary_file_for(destination)
    try:
        with partial.open("wb") as output:
            for part in parts:
                with part.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output)
        _require_file_identity(
            partial,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=f"assembled mirrored {label}",
        )
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()
    return True


def _download_google_drive_file(
    file_id: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    printer: Optional[Callable[[str], None]],
) -> None:
    label = "MIROS fine-tuned checkpoint"
    if destination.exists():
        _require_file_identity(
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=label,
        )
        _verify_checkpoint_container(destination)
        _log(printer, f"MIROS weight already exists: {destination}")
        return

    mirror_dir = os.environ.get(MIROS_MIRROR_DIR_ENV)
    if mirror_dir:
        mirror = Path(mirror_dir)
        if not mirror.is_dir():
            raise RuntimeError(f"MIROS release mirror directory does not exist: {mirror}")
        if _restore_from_mirror(
            mirror,
            destination,
            expected_size,
            expected_sha256,
            printer,
            label,
        ):
            _verify_checkpoint_container(destination)
            _log(printer, f"MIROS weight restored: {destination}")
            return
        raise RuntimeError(
            f"MIROS release mirror does not contain {destination.name}; "
            f"expected {destination.name}, miros-{destination.name}, or part files in {mirror}"
        )

    partial = _temporary_file_for(destination)
    _log(printer, f"Downloading MIROS weight: {destination.name}")
    try:
        _run(["curl", "-L", "--fail", "-o", str(partial), _google_drive_url(file_id)])
        if partial.stat().st_size != expected_size and partial.stat().st_size <= 10 * 1024 * 1024:
            confirmed_url = _confirmed_google_drive_url(partial)
            if confirmed_url:
                _log(printer, f"Confirming Google Drive download: {destination.name}")
                _run(["curl", "-L", "--fail", "-o", str(partial), confirmed_url])
        identity_error = miros_runtime.get_miros_file_identity_error(
            partial,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=f"downloaded {label}",
        )
        if identity_error:
            drive_error = (
                _google_drive_error_message(partial)
                if partial.stat().st_size <= 10 * 1024 * 1024
                else None
            )
            if drive_error:
                raise RuntimeError(f"{destination.name}: {drive_error}")
            raise RuntimeError(identity_error)
        _verify_checkpoint_container(partial)
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()
    _log(printer, f"MIROS weight saved: {destination}")


def _ensure_repo(repo_dir: Path, printer: Optional[Callable[[str], None]]) -> None:
    if repo_dir.exists():
        if not repo_dir.is_dir():
            raise RuntimeError(f"MIROS source path is not a directory: {repo_dir}")
        if any(repo_dir.iterdir()):
            _patch_miros_source(repo_dir, printer)
            _require_source_identity(repo_dir)
            _log(printer, f"MIROS pinned source exists: {repo_dir}")
            return
        repo_dir.rmdir()

    if shutil.which("git") is None:
        raise RuntimeError("git is required to clone ai4m-miros")

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(
            prefix=f".{repo_dir.name}.",
            suffix=".download",
            dir=str(repo_dir.parent),
        )
    )
    _log(
        printer,
        f"Cloning MIROS source commit {miros_runtime.MIROS_SOURCE_COMMIT}: {MIROS_REPO_URL}",
    )
    try:
        _run(["git", "init"], cwd=partial)
        _run(["git", "remote", "add", "origin", MIROS_REPO_URL], cwd=partial)
        _run(
            [
                "git",
                "fetch",
                "--depth=1",
                "origin",
                miros_runtime.MIROS_SOURCE_COMMIT,
            ],
            cwd=partial,
        )
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=partial)
        _patch_miros_source(partial, printer)
        _require_source_identity(partial)
        partial.replace(repo_dir)
    finally:
        if partial.exists():
            shutil.rmtree(partial)


def _patch_miros_source(repo_dir: Path, printer: Optional[Callable[[str], None]]) -> None:
    preparation_error = miros_runtime.get_miros_source_preparation_error(repo_dir)
    if preparation_error:
        raise RuntimeError(preparation_error)

    decmod_path = repo_dir / MIROS_DECMOD_REL_PATH
    if not decmod_path.is_file():
        raise RuntimeError(f"MIROS source is incomplete; missing {decmod_path}")

    text = decmod_path.read_text(encoding="utf-8")
    bad_import_count = text.count(MIROS_FLEX_ATTENTION_BAD_IMPORT)
    if bad_import_count not in {0, 1}:
        raise RuntimeError(
            "Unexpected MIROS decmod flex-attention import count: "
            f"expected 0 or 1, got {bad_import_count} ({decmod_path})"
        )
    if bad_import_count == 1:
        text = text.replace(MIROS_FLEX_ATTENTION_BAD_IMPORT + "\n", "", 1)
        decmod_path.write_text(text, encoding="utf-8")
        _log(printer, f"Patched MIROS flex attention import: {decmod_path}")

    rope_path = repo_dir / MIROS_ROPE_REL_PATH
    if not rope_path.is_file():
        raise RuntimeError(f"MIROS source is incomplete; missing {rope_path}")
    rope_text = rope_path.read_text(encoding="utf-8")
    old_import_count = rope_text.count(MIROS_ROPE_OLD_IMPORT)
    new_import_count = rope_text.count(MIROS_ROPE_NEW_IMPORT)
    old_decorator_count = rope_text.count(MIROS_ROPE_OLD_DECORATOR)
    new_decorator_count = rope_text.count(MIROS_ROPE_NEW_DECORATOR)
    if (old_import_count, new_import_count) not in {(1, 0), (0, 1)}:
        raise RuntimeError(
            "Unexpected MIROS RoPE autocast import state: "
            f"old={old_import_count}, new={new_import_count} ({rope_path})"
        )
    if (old_decorator_count, new_decorator_count) not in {(2, 0), (0, 2)}:
        raise RuntimeError(
            "Unexpected MIROS RoPE autocast decorator state: "
            f"old={old_decorator_count}, new={new_decorator_count} ({rope_path})"
        )
    if old_import_count == 1:
        rope_text = rope_text.replace(MIROS_ROPE_OLD_IMPORT, MIROS_ROPE_NEW_IMPORT, 1)
    if old_decorator_count == 2:
        rope_text = rope_text.replace(
            MIROS_ROPE_OLD_DECORATOR,
            MIROS_ROPE_NEW_DECORATOR,
            2,
        )
    if old_import_count == 1 or old_decorator_count == 2:
        rope_path.write_text(rope_text, encoding="utf-8")
        _log(printer, f"Patched MIROS RoPE autocast compatibility: {rope_path}")

    transcribe_path = repo_dir / MIROS_TRANSCRIBE_REL_PATH
    if not transcribe_path.is_file():
        raise RuntimeError(f"MIROS source is incomplete; missing {transcribe_path}")
    transcribe_text = transcribe_path.read_text(encoding="utf-8")
    inference_replacements = (
        (
            MIROS_AUDIO_SEGMENTS_GPU_BLOCK,
            MIROS_AUDIO_SEGMENTS_CPU_BLOCK,
            "audio segment residency",
        ),
        (
            MIROS_INFERENCE_CONTEXT_OLD,
            MIROS_INFERENCE_CONTEXT_NEW,
            "inference mode",
        ),
        (
            MIROS_INFERENCE_BATCH_OLD,
            MIROS_INFERENCE_BATCH_NEW,
            "inference batch size",
        ),
    )
    memory_patch_applied = False
    for old_fragment, new_fragment, label in inference_replacements:
        old_count = transcribe_text.count(old_fragment)
        new_count = transcribe_text.count(new_fragment)
        if (old_count, new_count) not in {(1, 0), (0, 1)}:
            raise RuntimeError(
                f"Unexpected MIROS {label} patch state: "
                f"old={old_count}, new={new_count} ({transcribe_path})"
            )
        if old_count == 1:
            transcribe_text = transcribe_text.replace(
                old_fragment,
                new_fragment,
                1,
            )
            memory_patch_applied = True
    if memory_patch_applied:
        transcribe_path.write_text(transcribe_text, encoding="utf-8")
        _log(
            printer,
            f"Patched MIROS bounded inference memory usage: {transcribe_path}",
        )


def prepare_miros_model(
    repo_dir: Path | str = Path("external") / "ai4m-miros",
    printer: Optional[Callable[[str], None]] = None,
) -> Path:
    repo = Path(repo_dir)
    _ensure_repo(repo, printer)

    _download_google_drive_file(
        MIROS_FINETUNED_FILE_ID,
        repo / MirosTranscriber.CHECKPOINT_REL_PATH,
        miros_runtime.MIROS_FINETUNED_EXACT_BYTES,
        miros_runtime.MIROS_FINETUNED_SHA256,
        printer,
    )
    _download_http_file(
        MIROS_PRETRAINED_URL,
        repo / MirosTranscriber.PRETRAINED_REL_PATH,
        miros_runtime.MIROS_PRETRAINED_EXACT_BYTES,
        miros_runtime.MIROS_PRETRAINED_SHA256,
        printer,
        "MIROS MusicFM pretrained weight",
    )

    _require_source_identity(repo)
    weight_error = miros_runtime.get_miros_weight_identity_error(repo)
    if weight_error:
        raise RuntimeError(f"MIROS preparation failed: {weight_error}")

    _log(
        printer,
        "MIROS pinned source and exact-identity weights are ready: "
        f"{repo} (source={miros_runtime.MIROS_SOURCE_COMMIT})",
    )
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
