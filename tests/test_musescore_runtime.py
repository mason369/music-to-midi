"""Pinned runtime staging must preserve the official AppDir structure."""

import hashlib
import os
from pathlib import Path
import subprocess

import pytest

import src.utils.musescore_runtime as runtime


def test_runtime_staging_preserves_relative_cyclic_and_dangling_links(tmp_path, monkeypatch):
    target = tmp_path / "cache" / runtime.MUSESCORE_VERSION
    license_bytes = b"GPL test fixture"
    prepared = []
    executable_relative = Path("bin/MuseScore4.exe" if os.name == "nt" else "AppRun")

    def prepare(temp_root, *, printer):
        distribution = temp_root / "extracted"
        executable = distribution / executable_relative
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"pinned runtime fixture")
        executable.chmod(0o755)
        (distribution / "share").mkdir()
        (distribution / "share/icon.png").write_bytes(b"original icon")
        (distribution / "usr").symlink_to(".", target_is_directory=True)
        (distribution / "icon.png").symlink_to(Path("share/icon.png"))
        (distribution / "manual.gz").symlink_to(Path("share/missing-manual.gz"))
        prepared.append(distribution)
        return distribution

    def download_license(_url, destination, **_kwargs):
        destination.write_bytes(license_bytes)

    monkeypatch.setattr(runtime, "_cache_root", lambda: target)
    monkeypatch.setattr(runtime, "_prepare_windows_runtime", prepare)
    monkeypatch.setattr(runtime, "_prepare_linux_runtime", prepare)
    monkeypatch.setattr(runtime, "_download_verified", download_license)
    monkeypatch.setattr(runtime, "musescore_version", lambda _path: (4, 7, 4))
    monkeypatch.setattr(runtime, "MUSESCORE_LICENSE_BYTES", len(license_bytes))
    monkeypatch.setattr(runtime, "MUSESCORE_LICENSE_SHA256", hashlib.sha256(license_bytes).hexdigest())

    executable = runtime.download_musescore_runtime(printer=lambda _message: None)

    assert executable == target / executable_relative
    assert executable.read_bytes() == b"pinned runtime fixture"
    assert (target / "usr").is_symlink()
    assert os.readlink(target / "usr") == "."
    assert os.readlink(target / "icon.png") == str(Path("share/icon.png"))
    assert (target / "icon.png").read_bytes() == b"original icon"
    assert os.readlink(target / "manual.gz") == str(Path("share/missing-manual.gz"))
    assert not (target / "manual.gz").exists()
    assert (target / "LICENSE.txt").read_bytes() == license_bytes
    assert not prepared[0].exists()
    assert runtime.download_musescore_runtime(printer=lambda _message: None) == executable
    assert len(prepared) == 1


@pytest.mark.parametrize(
    ("output", "returncode", "expected"),
    [
        ("/lib/x86_64-linux-gnu/libpipewire-0.3.so.0\nMuseScore4 4.7.4\n", 0, (4, 7, 4)),
        ("MuseScore4 4.7.4\r\n", 0, (4, 7, 4)),
        ("MuseScore Studio 4.7.4\n", 0, (4, 7, 4)),
        ("MuseScore version 3.6.2\n", 0, (3, 6, 2)),
        ("Other Editor 4.7.4\n", 0, None),
        ("/lib/x86_64-linux-gnu/libpipewire-0.3.so.0\n", 0, None),
        ("Failed to start MuseScore4 4.7.4\n", 0, None),
        ("", 0, None),
        ("MuseScore4 4.7.4\n", 1, None),
    ],
)
def test_version_requires_successful_musescore_identity_line(tmp_path, monkeypatch, output, returncode, expected):
    executable = tmp_path / "renderer"
    executable.write_bytes(b"runtime fixture")
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, returncode, output),
    )
    assert runtime.musescore_version(executable) == expected
