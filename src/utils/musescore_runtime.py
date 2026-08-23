"""MuseScore Studio 4 runtime discovery and pinned runtime preparation.

MuseScore is an external GPL-licensed application.  This module only resolves,
validates, or explicitly prepares that runtime; importing it never downloads or
silently substitutes another score renderer.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from src.utils.runtime_paths import get_bundle_roots

MUSESCORE_VERSION = "4.7.4"
MUSESCORE_BUILD = "260706075"
MUSESCORE_MINIMUM_MAJOR = 4
MUSESCORE_ENV_VAR = "MUSIC_TO_MIDI_MUSESCORE"
MUSESCORE_COMPAT_ENV_VAR = "MUSCRIPTOR_MUSESCORE"

MUSESCORE_WINDOWS_INSTALLER = f"MuseScore-Studio-{MUSESCORE_VERSION}.{MUSESCORE_BUILD}-x86_64.msi"
MUSESCORE_WINDOWS_URL = (
    "https://github.com/musescore/MuseScore/releases/download/"
    f"v{MUSESCORE_VERSION}/{MUSESCORE_WINDOWS_INSTALLER}"
)
MUSESCORE_WINDOWS_INSTALLER_BYTES = 127_807_488
MUSESCORE_WINDOWS_INSTALLER_SHA256 = (
    "64fe70e5cb9ffe159d047d1e88db567bd101f60d36b0de28feb674716929a378"
)

MUSESCORE_LINUX_APPIMAGE = f"MuseScore-Studio-{MUSESCORE_VERSION}.{MUSESCORE_BUILD}-x86_64.AppImage"
MUSESCORE_LINUX_URL = (
    "https://github.com/musescore/MuseScore/releases/download/"
    f"v{MUSESCORE_VERSION}/{MUSESCORE_LINUX_APPIMAGE}"
)
MUSESCORE_LINUX_APPIMAGE_BYTES = 192_231_928
MUSESCORE_LINUX_APPIMAGE_SHA256 = "9233ed1b87d3e6b45722278f3c286dcd41e83da778bd0f80a1dd04949696ad93"
MUSESCORE_LICENSE_NAME = "LICENSE.txt"
MUSESCORE_LICENSE_URL = (
    f"https://raw.githubusercontent.com/musescore/MuseScore/v{MUSESCORE_VERSION}/"
    + MUSESCORE_LICENSE_NAME
)
MUSESCORE_LICENSE_BYTES = 36_493
MUSESCORE_LICENSE_SHA256 = "73e75f61f0dfce4fd83ad1f3f45d1d6f9ea8cdfd21a92d7a9216d47e5b3bbb88"

_VERSION_PATTERN = re.compile(r"\b(\d+)\.(\d+)(?:\.(\d+))?")
_BINARY_NAMES = (
    "MuseScore4",
    "musescore4",
    "mscore4portable",
    "musescore",
    "mscore",
)


class MuseScoreRuntimeError(RuntimeError):
    """The required MuseScore Studio runtime is absent or has wrong identity."""


def _cache_root() -> Path:
    return Path.home() / ".cache" / "music_ai_models" / "musescore" / MUSESCORE_VERSION


def get_musescore_subprocess_env() -> dict[str, str]:
    """Return a deterministic headless environment for MuseScore CLI calls."""

    environment = os.environ.copy()
    if sys.platform.startswith("linux"):
        environment.setdefault("QT_QPA_PLATFORM", "offscreen")
        environment.setdefault("MU_QT_QPA_PLATFORM", "offscreen")
    return environment


def _creation_flags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def musescore_version(executable: str | Path) -> tuple[int, ...] | None:
    """Return the executable version, or ``None`` for a non-MuseScore binary."""

    path = Path(executable).expanduser()
    if not path.is_file():
        return None
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=get_musescore_subprocess_env(),
            creationflags=_creation_flags(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = _VERSION_PATTERN.search(completed.stdout or "")
    if completed.returncode != 0 or match is None:
        return None
    return tuple(int(value) for value in match.groups() if value is not None)


def validate_musescore_runtime(
    executable: str | Path,
    *,
    require_pinned_version: bool = False,
) -> Path:
    """Validate a real MuseScore 4+ executable and optionally its exact release."""

    path = Path(executable).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise MuseScoreRuntimeError(f"MuseScore executable is missing or empty: {path}")
    version = musescore_version(path)
    if version is None:
        raise MuseScoreRuntimeError(f"Executable did not identify as MuseScore Studio: {path}")
    if version[0] < MUSESCORE_MINIMUM_MAJOR:
        shown = ".".join(str(value) for value in version)
        raise MuseScoreRuntimeError(
            f"MuseScore Studio {MUSESCORE_MINIMUM_MAJOR}+ is required; found {shown}: {path}"
        )
    if require_pinned_version and version[:3] != tuple(
        int(value) for value in MUSESCORE_VERSION.split(".")
    ):
        shown = ".".join(str(value) for value in version)
        raise MuseScoreRuntimeError(
            f"MuseScore runtime identity mismatch: expected {MUSESCORE_VERSION}, "
            f"found {shown}: {path}"
        )
    return path


def _distribution_root(executable: Path) -> Path:
    if executable.name.lower() == "musescore4.exe" and executable.parent.name.lower() == "bin":
        return executable.parent.parent
    if executable.name == "AppRun":
        return executable.parent
    raise MuseScoreRuntimeError(
        f"Unsupported MuseScore distribution layout for pinned validation: {executable}"
    )


def validate_pinned_musescore_distribution(
    executable: str | Path | None = None,
) -> Path:
    """Require the exact release executable and its upstream GPL-3.0 license."""

    resolved = validate_musescore_runtime(
        executable if executable is not None else get_musescore_executable(),
        require_pinned_version=True,
    )
    license_path = _distribution_root(resolved) / MUSESCORE_LICENSE_NAME
    if not license_path.is_file():
        raise MuseScoreRuntimeError(
            f"Pinned MuseScore distribution is missing {MUSESCORE_LICENSE_NAME}: " f"{license_path}"
        )
    if license_path.stat().st_size != MUSESCORE_LICENSE_BYTES:
        raise MuseScoreRuntimeError(
            "MuseScore license size mismatch: "
            f"expected {MUSESCORE_LICENSE_BYTES}, got {license_path.stat().st_size}: "
            f"{license_path}"
        )
    digest = hashlib.sha256(license_path.read_bytes()).hexdigest()
    if digest != MUSESCORE_LICENSE_SHA256:
        raise MuseScoreRuntimeError(
            "MuseScore license SHA-256 mismatch: "
            f"expected {MUSESCORE_LICENSE_SHA256}, got {digest}: {license_path}"
        )
    return resolved


def _bundle_candidates() -> list[Path]:
    relative_paths = (
        "resources/musescore/bin/MuseScore4.exe",
        "resources/musescore/AppRun",
        "musescore/bin/MuseScore4.exe",
        "musescore/AppRun",
    )
    return [root / relative for root in get_bundle_roots() for relative in relative_paths]


def _system_candidates() -> list[Path]:
    candidates: list[Path] = []
    if os.name == "nt":
        for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            root = os.environ.get(variable, "").strip()
            if root:
                candidates.append(Path(root) / "MuseScore 4" / "bin" / "MuseScore4.exe")
    elif sys.platform == "darwin":
        candidates.extend(
            [
                Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore"),
                Path.home() / "Applications" / "MuseScore 4.app" / "Contents" / "MacOS" / "mscore",
            ]
        )
    else:
        candidates.extend(
            [
                Path("/opt/musescore/AppRun"),
                Path.home() / "MuseScore.AppImage",
                Path.home() / "Applications" / "MuseScore.AppImage",
            ]
        )
    for name in _BINARY_NAMES:
        discovered = shutil.which(name)
        if discovered:
            candidates.append(Path(discovered))
    return candidates


def _configured_executable() -> Path | None:
    primary = os.environ.get(MUSESCORE_ENV_VAR, "").strip()
    compatibility = os.environ.get(MUSESCORE_COMPAT_ENV_VAR, "").strip()
    if primary and compatibility:
        if Path(primary).expanduser().resolve() != Path(compatibility).expanduser().resolve():
            raise MuseScoreRuntimeError(
                f"{MUSESCORE_ENV_VAR} and {MUSESCORE_COMPAT_ENV_VAR} point to different files"
            )
    configured = primary or compatibility
    return Path(configured).expanduser() if configured else None


def get_musescore_executable() -> Path:
    """Resolve MuseScore 4+ without downloading or hiding an explicit bad path."""

    configured = _configured_executable()
    if configured is not None:
        try:
            return validate_musescore_runtime(configured)
        except MuseScoreRuntimeError as exc:
            raise MuseScoreRuntimeError(
                f"Configured MuseScore runtime is invalid ({MUSESCORE_ENV_VAR}): {exc}"
            ) from exc

    cache_candidates = (
        _cache_root() / "bin" / "MuseScore4.exe",
        _cache_root() / "AppRun",
    )
    errors: list[str] = []
    seen: set[str] = set()
    for candidate in [*_bundle_candidates(), *cache_candidates, *_system_candidates()]:
        key = os.path.normcase(str(candidate.expanduser().resolve()))
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        try:
            return validate_musescore_runtime(candidate)
        except MuseScoreRuntimeError as exc:
            errors.append(str(exc))
    detail = f" Runtime checks: {'; '.join(errors)}" if errors else ""
    raise MuseScoreRuntimeError(
        "MuseScore Studio 4+ is required to generate MusicXML/PDF sheet music. "
        "Run `python download_musescore_runtime.py`, install MuseScore Studio 4, "
        f"or set {MUSESCORE_ENV_VAR} to MuseScore4.exe/AppRun." + detail
    )


def _download_verified(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    printer,
) -> None:
    printer(f"Downloading pinned MuseScore Studio {MUSESCORE_VERSION}: {url}")
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "music-to-midi"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as target:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            target.write(block)
            digest.update(block)
            size += len(block)
    actual_sha256 = digest.hexdigest()
    if size != expected_size or actual_sha256 != expected_sha256:
        raise MuseScoreRuntimeError(
            "MuseScore download identity mismatch: "
            f"expected={expected_size}/{expected_sha256}, actual={size}/{actual_sha256}, "
            f"path={destination}"
        )


def _prepare_windows_runtime(temp_root: Path, *, printer) -> Path:
    installer = temp_root / MUSESCORE_WINDOWS_INSTALLER
    _download_verified(
        MUSESCORE_WINDOWS_URL,
        installer,
        expected_size=MUSESCORE_WINDOWS_INSTALLER_BYTES,
        expected_sha256=MUSESCORE_WINDOWS_INSTALLER_SHA256,
        printer=printer,
    )
    extracted = temp_root / "extracted"
    extracted.mkdir()
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    msiexec = shutil.which("msiexec.exe") or str(system_root / "System32" / "msiexec.exe")
    completed = subprocess.run(
        [
            str(msiexec),
            "/a",
            str(installer),
            "/qn",
            "/norestart",
            f"TARGETDIR={extracted}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        creationflags=_creation_flags(),
    )
    if completed.returncode != 0:
        raise MuseScoreRuntimeError(
            f"MuseScore MSI administrative extraction failed ({completed.returncode}): "
            f"{(completed.stdout or '').strip()}"
        )
    executables = [
        path for path in extracted.rglob("MuseScore4.exe") if path.parent.name.lower() == "bin"
    ]
    if len(executables) != 1:
        raise MuseScoreRuntimeError(
            "MuseScore MSI must contain exactly one bin/MuseScore4.exe; "
            f"found {[str(path) for path in executables]!r}"
        )
    return executables[0].parent.parent


def _prepare_linux_runtime(temp_root: Path, *, printer) -> Path:
    appimage = temp_root / MUSESCORE_LINUX_APPIMAGE
    _download_verified(
        MUSESCORE_LINUX_URL,
        appimage,
        expected_size=MUSESCORE_LINUX_APPIMAGE_BYTES,
        expected_sha256=MUSESCORE_LINUX_APPIMAGE_SHA256,
        printer=printer,
    )
    appimage.chmod(appimage.stat().st_mode | 0o111)
    completed = subprocess.run(
        [str(appimage), "--appimage-extract"],
        check=False,
        cwd=temp_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=get_musescore_subprocess_env(),
    )
    extracted = temp_root / "squashfs-root"
    if completed.returncode != 0 or not (extracted / "AppRun").is_file():
        raise MuseScoreRuntimeError(
            "MuseScore AppImage extraction failed: "
            f"returncode={completed.returncode}, output={(completed.stdout or '').strip()!r}"
        )
    return extracted


def download_musescore_runtime(*, printer=print) -> Path:
    """Prepare the exact official runtime in the project cache and verify it."""

    target_root = _cache_root()
    target_executable = (
        target_root / "bin" / "MuseScore4.exe" if os.name == "nt" else target_root / "AppRun"
    )
    if target_executable.is_file():
        try:
            return validate_pinned_musescore_distribution(target_executable)
        except MuseScoreRuntimeError as exc:
            printer(f"Replacing invalid cached MuseScore distribution: {exc}")
    if os.name != "nt" and not sys.platform.startswith("linux"):
        raise MuseScoreRuntimeError(
            "Automatic MuseScore runtime preparation currently supports Windows x64 and "
            "Linux x86_64; install MuseScore Studio 4 and set MUSIC_TO_MIDI_MUSESCORE."
        )

    target_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="musescore-prepare-", dir=target_root.parent
    ) as temp_dir:
        temp_root = Path(temp_dir)
        distribution_root = (
            _prepare_windows_runtime(temp_root, printer=printer)
            if os.name == "nt"
            else _prepare_linux_runtime(temp_root, printer=printer)
        )
        staged = temp_root / "staged-runtime"
        shutil.copytree(distribution_root, staged)
        staged_executable = (
            staged / "bin" / "MuseScore4.exe" if os.name == "nt" else staged / "AppRun"
        )
        _download_verified(
            MUSESCORE_LICENSE_URL,
            staged / MUSESCORE_LICENSE_NAME,
            expected_size=MUSESCORE_LICENSE_BYTES,
            expected_sha256=MUSESCORE_LICENSE_SHA256,
            printer=printer,
        )
        validate_pinned_musescore_distribution(staged_executable)
        if target_root.exists():
            shutil.rmtree(target_root)
        staged.replace(target_root)

    executable = validate_pinned_musescore_distribution(target_executable)
    printer(f"MuseScore Studio runtime verified: {executable}")
    return executable


__all__ = [
    "MUSESCORE_BUILD",
    "MUSESCORE_COMPAT_ENV_VAR",
    "MUSESCORE_ENV_VAR",
    "MUSESCORE_LINUX_APPIMAGE",
    "MUSESCORE_LINUX_APPIMAGE_BYTES",
    "MUSESCORE_LINUX_APPIMAGE_SHA256",
    "MUSESCORE_LINUX_URL",
    "MUSESCORE_LICENSE_BYTES",
    "MUSESCORE_LICENSE_NAME",
    "MUSESCORE_LICENSE_SHA256",
    "MUSESCORE_LICENSE_URL",
    "MUSESCORE_MINIMUM_MAJOR",
    "MUSESCORE_VERSION",
    "MUSESCORE_WINDOWS_INSTALLER",
    "MUSESCORE_WINDOWS_INSTALLER_BYTES",
    "MUSESCORE_WINDOWS_INSTALLER_SHA256",
    "MUSESCORE_WINDOWS_URL",
    "MuseScoreRuntimeError",
    "download_musescore_runtime",
    "get_musescore_executable",
    "get_musescore_subprocess_env",
    "musescore_version",
    "validate_musescore_runtime",
    "validate_pinned_musescore_distribution",
]
