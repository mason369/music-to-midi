"""Reproducible MuScriptor source identity and best-quality runtime patching.

The project installs official v0.3.0 (beat/bar display support, onset-phase
correction, and MuseScore tempo metadata), overlays the two runtime files from
the fixed overlap/restart PR head, and then deterministically restores the
v0.3.0 BeatGrid API in the one file touched by both changes. Every source file
is verified by SHA-256 before and after replacement. The installed runtime is
therefore an explicit composite of two reviewed upstream revisions, not an
ephemeral GitHub merge ref or a whole-file overwrite that drops either feature.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

MUSCRIPTOR_PACKAGE_VERSION = "0.3.0"
MUSCRIPTOR_SOURCE_COMMIT = "d73147e75e5b9b0c0a79ebe154587db4fd603e0c"
MUSCRIPTOR_QUALITY_PATCH_COMMIT = "edaebd3126336bd7eb4467dcf675d77f4e7772f0"
MUSCRIPTOR_SOURCE_REQUIREMENT = (
    "muscriptor @ https://github.com/muscriptor/muscriptor/archive/"
    f"{MUSCRIPTOR_SOURCE_COMMIT}.zip"
)

_RAW_PATCH_ROOT = (
    "https://raw.githubusercontent.com/muscriptor/muscriptor/" f"{MUSCRIPTOR_QUALITY_PATCH_COMMIT}/"
)

# Base hashes prove that the patch is applied only to the intended beat-grid
# source commit. Overlay hashes prove the downloaded PR #58 files. Patched
# hashes prove the exact composite of beat-grid and overlap/restart behavior.
_PATCH_FILES = {
    "transcription_model.py": {
        "base_sha256": "d301cee911ad0e2fdf90bf4ad659a7e2a55e30d9028fa3604df3bb31b12fc88d",
        "overlay_sha256": "16b9a27e4b35f117a8b9bd439fd98cf3a7228030216d42596dbc393cf9a6d2ea",
        "patched_sha256": "229206d5bb2411e9f29df48e3bf863a74d7aca8ab18f3e7c41661eec3e91369f",
        "url": _RAW_PATCH_ROOT + "muscriptor/transcription_model.py",
    },
    "tokenizer/mt3.py": {
        "base_sha256": "c37c7d9b101f6498bf8cf5964ab90df8b4df59f2e9960a192ed43b7fa9ae4a62",
        "overlay_sha256": "b4a95a6560cc0238d3a5b44e500dd2e30c85488d912161db89528ca60a23b110",
        "patched_sha256": "b4a95a6560cc0238d3a5b44e500dd2e30c85488d912161db89528ca60a23b110",
        "url": _RAW_PATCH_ROOT + "muscriptor/tokenizer/mt3.py",
    },
}

# These files distinguish official v0.3.0 from older wheels and prove that its
# native precision, onset correction, and MuseScore tempo changes all exist.
_MAIN_IDENTITY_FILES = {
    "models/lm.py": "8600f40d7b5242b37d0db5dbf16eddde6b493d2243ac4177bc86750bcea554a2",
    "accelerator.py": "1bc8001093c7fbc888191fc649c874c238d3966b3363d6458833157761ae40e5",
    "tokenizer/notes.py": "686ee5a6d336bf74a67390c72af376c00e42061e0d72354346e2e52b95363f33",
    "utils/beats.py": "58da8484da7c0372aa54690d5d6a801c9ae14c6897ac52aaf81acdee85af6bd9",
    "utils/midi.py": "21d16e15b073ec0522b31c6468a166dac4ae7c76f5ea318077945330ca572348",
}


def _compose_beat_grid_overlay(relative: str, payload: bytes) -> bytes:
    """Restore v0.3.0's BeatGrid API after applying PR #58's older file."""

    if relative != "transcription_model.py":
        return payload
    text = payload.decode("utf-8")
    replacements = (
        (
            "from muscriptor.utils.audio import load_audio, resample\n"
            "from muscriptor.utils.download import download_companion, download_if_necessary\n",
            "from muscriptor.utils.audio import load_audio, resample\n"
            "from muscriptor.utils.beats import (\n"
            "    BeatDetectionError,\n"
            "    BeatGrid,\n"
            "    TempoDetection,\n"
            "    detect_grid,\n"
            ")\n"
            "from muscriptor.utils.download import download_companion, download_if_necessary\n",
        ),
        (
            "        overlap: float = 0.0,\n"
            "        allow_reset: bool = False,\n"
            "    ) -> bytes:\n"
            '        """Same as :meth:`transcribe` but returns a MIDI file as bytes."""\n'
            "        events = self.transcribe(\n",
            "        overlap: float = 0.0,\n"
            "        allow_reset: bool = False,\n"
            '        detect_tempo: TempoDetection = "best-effort",\n'
            "    ) -> bytes:\n"
            '        """Same as :meth:`transcribe` but returns a MIDI file as bytes."""\n'
            "        beat_grid = self.detect_beat_grid_for(audio, detect_tempo)\n"
            "        events = self.transcribe(\n",
        ),
        (
            "        return self.events_to_midi_bytes(events)\n\n"
            "    def events_to_midi_bytes(\n"
            "        self, events: Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent]\n"
            "    ) -> bytes:\n",
            "        return self.events_to_midi_bytes(events, beat_grid=beat_grid)\n\n"
            "    def detect_beat_grid_for(\n"
            "        self,\n"
            "        audio: str | Path | tuple[torch.Tensor, int],\n"
            '        mode: TempoDetection = "best-effort",\n'
            "    ) -> BeatGrid | None:\n"
            '        """Detect the beat grid of `audio`, or None if there isn\'t a usable one.\n\n'
            "        Accepts the same input forms as :meth:`transcribe`.\n"
            "        `mode` decides what a failed detection means: raise (True), skip detection\n"
            "        entirely (False), or warn and fall back to the placeholder tempo\n"
            '        ("best-effort").\n'
            '        """\n'
            "        if mode is False:\n"
            "            return None\n"
            "        tensor, sample_rate = audio if isinstance(audio, tuple) else (audio, None)\n"
            "        try:\n"
            "            return detect_grid(self._load_wav(tensor, sample_rate), _SAMPLE_RATE)\n"
            "        except BeatDetectionError as e:\n"
            "            if mode is True:\n"
            "                raise\n"
            "            print(\n"
            '                f"Warning: {e}; falling back to the placeholder tempo",\n'
            "                file=sys.stderr,\n"
            "            )\n"
            "            return None\n\n"
            "    def events_to_midi_bytes(\n"
            "        self,\n"
            "        events: Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent],\n"
            "        beat_grid: BeatGrid | None = None,\n"
            "    ) -> bytes:\n",
        ),
        (
            "        midi = notes_to_midi(notes, program_names=program_names)\n",
            "        midi = notes_to_midi(notes, program_names=program_names, beat_grid=beat_grid)\n",
        ),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(
                "MuScriptor composite patch anchor mismatch: "
                f"file={relative}, anchor={old.splitlines()[0]!r}, count={text.count(old)}"
            )
        text = text.replace(old, new, 1)
    return text.encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_package_root(package_root: str | Path | None = None) -> Path:
    if package_root is not None:
        root = Path(package_root).resolve()
    else:
        spec = importlib.util.find_spec("muscriptor")
        locations = [] if spec is None else list(spec.submodule_search_locations or [])
        if len(locations) != 1:
            raise RuntimeError(
                "Unable to locate one installed MuScriptor package directory: "
                f"locations={locations!r}"
            )
        root = Path(locations[0]).resolve()
    if not root.is_dir():
        raise RuntimeError(f"MuScriptor package directory does not exist: {root}")
    return root


def validate_muscriptor_runtime_identity(
    package_root: str | Path | None = None,
) -> str:
    """Return an explicit identity error, or an empty string when exact."""

    try:
        installed_version = importlib.metadata.version("muscriptor")
    except importlib.metadata.PackageNotFoundError:
        return "MuScriptor package metadata is not installed"
    if installed_version != MUSCRIPTOR_PACKAGE_VERSION:
        return (
            "MuScriptor package version mismatch: "
            f"expected {MUSCRIPTOR_PACKAGE_VERSION}, got {installed_version}"
        )
    try:
        root = _installed_package_root(package_root)
    except Exception as exc:
        return str(exc)

    expected = {relative: spec["patched_sha256"] for relative, spec in _PATCH_FILES.items()}
    expected.update(_MAIN_IDENTITY_FILES)
    for relative, expected_sha256 in expected.items():
        path = root / relative
        if not path.is_file():
            return f"MuScriptor identity file is missing: {path}"
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            return (
                "MuScriptor source identity mismatch: "
                f"path={path}, expected={expected_sha256}, actual={actual_sha256}"
            )
    return ""


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "music-to-midi-muscriptor-patcher/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def apply_muscriptor_quality_patch(
    *,
    package_root: str | Path | None = None,
    fetcher: Callable[[str], bytes] | None = None,
    printer: Callable[[str], None] = print,
) -> Path:
    """Apply and verify the fixed overlap/restart files transactionally."""

    installed_version = importlib.metadata.version("muscriptor")
    if installed_version != MUSCRIPTOR_PACKAGE_VERSION:
        raise RuntimeError(
            "Refusing to patch an unexpected MuScriptor version: "
            f"expected={MUSCRIPTOR_PACKAGE_VERSION}, actual={installed_version}"
        )
    root = _installed_package_root(package_root)
    fetch = fetcher or _download_bytes

    replacements: dict[str, bytes] = {}
    for relative, spec in _PATCH_FILES.items():
        target = root / relative
        if not target.is_file():
            raise RuntimeError(f"MuScriptor patch target is missing: {target}")
        actual_sha256 = _sha256_file(target)
        if actual_sha256 == spec["patched_sha256"]:
            continue
        if actual_sha256 != spec["base_sha256"]:
            raise RuntimeError(
                "Refusing to patch an unknown MuScriptor source file: "
                f"path={target}, expected_base={spec['base_sha256']}, "
                f"actual={actual_sha256}"
            )
        payload = fetch(str(spec["url"]))
        downloaded_sha256 = _sha256_bytes(payload)
        if downloaded_sha256 != spec["overlay_sha256"]:
            raise RuntimeError(
                "Downloaded MuScriptor quality patch failed SHA-256 verification: "
                f"url={spec['url']}, expected={spec['overlay_sha256']}, "
                f"actual={downloaded_sha256}"
            )
        payload = _compose_beat_grid_overlay(relative, payload)
        composite_sha256 = _sha256_bytes(payload)
        if composite_sha256 != spec["patched_sha256"]:
            raise RuntimeError(
                "Composed MuScriptor beat-grid/quality patch failed SHA-256 verification: "
                f"path={relative}, expected={spec['patched_sha256']}, "
                f"actual={composite_sha256}"
            )
        replacements[relative] = payload

    staging_root = Path(tempfile.mkdtemp(prefix="muscriptor-quality-patch-"))
    try:
        staged_paths: dict[str, Path] = {}
        backup_paths: dict[str, Path] = {}
        for relative, payload in replacements.items():
            staged = staging_root / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(payload)
            staged_paths[relative] = staged
            backup = staging_root / "backups" / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, backup)
            backup_paths[relative] = backup

        applied: list[str] = []
        try:
            for relative, staged in staged_paths.items():
                target = root / relative
                os.replace(staged, target)
                applied.append(relative)
                printer(f"Applied verified MuScriptor quality patch: {relative}")
        except Exception:
            for relative in reversed(applied):
                os.replace(backup_paths[relative], root / relative)
            raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    identity_error = validate_muscriptor_runtime_identity(root)
    if identity_error:
        raise RuntimeError(
            "MuScriptor quality patch completed but runtime identity is invalid: "
            f"{identity_error}"
        )
    printer(
        "MuScriptor runtime verified: "
        f"main={MUSCRIPTOR_SOURCE_COMMIT}, quality_patch={MUSCRIPTOR_QUALITY_PATCH_COMMIT}"
    )
    return root


__all__ = [
    "MUSCRIPTOR_PACKAGE_VERSION",
    "MUSCRIPTOR_QUALITY_PATCH_COMMIT",
    "MUSCRIPTOR_SOURCE_COMMIT",
    "MUSCRIPTOR_SOURCE_REQUIREMENT",
    "apply_muscriptor_quality_patch",
    "validate_muscriptor_runtime_identity",
]
