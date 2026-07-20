"""Pinned FluidSynth runtime discovery and Windows preparation."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

FLUIDSYNTH_VERSION = "2.5.6"
FLUIDSYNTH_WINDOWS_ARCHIVE = f"fluidsynth-v{FLUIDSYNTH_VERSION}-win10-x64-cpp11.zip"
FLUIDSYNTH_WINDOWS_URL = (
    f"https://github.com/FluidSynth/fluidsynth/releases/download/v{FLUIDSYNTH_VERSION}/"
    f"{FLUIDSYNTH_WINDOWS_ARCHIVE}"
)
FLUIDSYNTH_WINDOWS_ARCHIVE_BYTES = 2_656_559
FLUIDSYNTH_WINDOWS_ARCHIVE_SHA256 = (
    "a4b8bd4f133b7b6770537f6c18b2b2b93579338d51e26f777d025e40e15a7e81"
)


def _cache_root() -> Path:
    return Path.home() / ".cache" / "music_ai_models" / "fluidsynth" / FLUIDSYNTH_VERSION


def get_fluidsynth_subprocess_env(executable: Path) -> dict[str, str]:
    """Return the native-library environment for the selected runtime."""

    environment = os.environ.copy()
    bin_dir = executable.resolve().parent
    runtime_root = bin_dir.parent
    library_dir = runtime_root / "lib"
    path_entries = [str(bin_dir)]
    if library_dir.is_dir():
        if os.name == "nt":
            path_entries.append(str(library_dir))
        else:
            current = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = (
                str(library_dir) if not current else str(library_dir) + os.pathsep + current
            )
    current_path = environment.get("PATH", "")
    environment["PATH"] = os.pathsep.join(path_entries + ([current_path] if current_path else []))
    return environment


def _validate_runtime(executable: Path, *, require_pinned_version: bool) -> Path:
    if not executable.is_file() or executable.stat().st_size <= 0:
        raise RuntimeError(f"FluidSynth executable is missing or empty: {executable}")
    completed = subprocess.run(
        [str(executable), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        env=get_fluidsynth_subprocess_env(executable),
    )
    version_text = completed.stdout or ""
    if completed.returncode != 0 or (
        require_pinned_version and FLUIDSYNTH_VERSION not in version_text
    ):
        raise RuntimeError(
            "FluidSynth runtime identity mismatch: "
            f"expected {FLUIDSYNTH_VERSION}, executable={executable}, "
            f"output={version_text.strip()!r}"
        )
    return executable.resolve()


def get_fluidsynth_executable() -> Path:
    """Resolve only a verified runtime; never substitute another synthesizer."""

    configured = os.environ.get("MUSIC_TO_MIDI_FLUIDSYNTH", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(
        Path(__file__).resolve().parents[2]
        / "resources"
        / "fluidsynth"
        / "bin"
        / ("fluidsynth.exe" if os.name == "nt" else "fluidsynth")
    )
    if os.name == "nt":
        candidates.extend(
            [
                _cache_root() / "bin" / "fluidsynth.exe",
            ]
        )
    discovered = shutil.which("fluidsynth")
    if discovered:
        candidates.append(Path(discovered))

    errors: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            return _validate_runtime(
                candidate,
                require_pinned_version=(os.name == "nt"),
            )
        except RuntimeError as exc:
            errors.append(str(exc))

    detail = f" Runtime checks: {'; '.join(errors)}" if errors else ""
    if os.name == "nt":
        raise RuntimeError(
            "FluidSynth 2.5.6 is required for the MuScriptor official playback/export "
            "workbench. Run `python download_fluidsynth_runtime.py`." + detail
        )
    raise RuntimeError(
        "A working FluidSynth runtime is required for the MuScriptor official playback/export "
        "workbench. Install the system `fluidsynth` package." + detail
    )


def download_fluidsynth_windows(*, printer=print) -> Path:
    """Install the exact official Windows x64 release into the project cache."""

    if os.name != "nt":
        return get_fluidsynth_executable()
    target_root = _cache_root()
    target_exe = target_root / "bin" / "fluidsynth.exe"
    if target_exe.is_file():
        return _validate_runtime(target_exe, require_pinned_version=True)

    target_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="fluidsynth-prepare-", dir=target_root.parent
    ) as temp_dir:
        temp_root = Path(temp_dir)
        archive = temp_root / FLUIDSYNTH_WINDOWS_ARCHIVE
        printer(f"Downloading pinned FluidSynth {FLUIDSYNTH_VERSION}: {FLUIDSYNTH_WINDOWS_URL}")
        urllib.request.urlretrieve(FLUIDSYNTH_WINDOWS_URL, archive)
        size = archive.stat().st_size
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if size != FLUIDSYNTH_WINDOWS_ARCHIVE_BYTES or digest != FLUIDSYNTH_WINDOWS_ARCHIVE_SHA256:
            raise RuntimeError(
                "FluidSynth archive identity mismatch: "
                f"size={size}, sha256={digest}, path={archive}"
            )

        extracted = temp_root / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            root = extracted.resolve()
            for member in bundle.infolist():
                destination = (extracted / member.filename).resolve()
                try:
                    destination.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Unsafe FluidSynth archive member: {member.filename!r}"
                    ) from exc
            bundle.extractall(extracted)

        executables = list(extracted.rglob("fluidsynth.exe"))
        if len(executables) != 1:
            raise RuntimeError(
                "FluidSynth archive must contain exactly one executable; "
                f"found {[str(path) for path in executables]!r}"
            )
        distribution_root = executables[0].parent.parent
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.copytree(distribution_root, target_root)

    executable = _validate_runtime(target_exe, require_pinned_version=True)
    printer(f"FluidSynth runtime verified: {executable}")
    return executable
