from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


CONDA_PACKAGE_URL = (
    "https://api.anaconda.org/download/conda-forge/llvm-openmp/19.1.7/"
    "win-64/llvm-openmp-19.1.7-h30eaf37_1.conda"
)
TARGET_DLL_NAME = "libomp140.x86_64.dll"


def _print(message: str) -> None:
    print(message, flush=True)


def resolve_torch_lib_dir(explicit_dir: str | None = None) -> Path:
    if explicit_dir:
        return Path(explicit_dir).expanduser().resolve()

    import importlib.util

    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        raise RuntimeError("Unable to locate installed torch package")

    return Path(spec.origin).resolve().parent / "lib"


def download_conda_package(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(CONDA_PACKAGE_URL, destination)
    return destination


def extract_libomp_from_conda(conda_path: Path, destination: Path) -> Path:
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError(
            "zstandard is required to extract llvm-openmp; install it with "
            "`python -m pip install zstandard`"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="torch-openmp-") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(conda_path) as archive:
            zst_members = [
                name for name in archive.namelist()
                if name.startswith("pkg-") and name.endswith(".tar.zst")
            ]
            if not zst_members:
                raise RuntimeError("llvm-openmp conda package did not contain pkg-*.tar.zst")
            archive.extract(zst_members[0], tmp_path)

        zst_path = tmp_path / zst_members[0]
        dctx = zstandard.ZstdDecompressor()
        with zst_path.open("rb") as handle:
            tar_data = dctx.decompress(handle.read(), max_output_size=50 * 1024 * 1024)

        with tarfile.open(fileobj=io.BytesIO(tar_data)) as tar:
            member = tar.getmember("Library/bin/libomp.dll")
            fileobj = tar.extractfile(member)
            if fileobj is None:
                raise RuntimeError("Failed to extract Library/bin/libomp.dll from llvm-openmp")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(fileobj.read())

    return destination


def ensure_torch_openmp(torch_lib_dir: str | None = None) -> Path:
    if os.name != "nt":
        raise RuntimeError("Torch OpenMP repair helper is only intended for Windows")

    torch_lib = resolve_torch_lib_dir(torch_lib_dir)
    if not torch_lib.exists():
        raise RuntimeError(f"Torch lib directory does not exist: {torch_lib}")

    target = torch_lib / TARGET_DLL_NAME
    if target.exists():
        _print(f"[ok] {TARGET_DLL_NAME} already exists: {target}")
        return target

    with tempfile.TemporaryDirectory(prefix="llvm-openmp-") as tmp:
        conda_path = Path(tmp) / "llvm-openmp.conda"
        _print(f"[info] Downloading llvm-openmp package to {conda_path}")
        download_conda_package(conda_path)
        _print(f"[info] Extracting {TARGET_DLL_NAME} into {torch_lib}")
        extract_libomp_from_conda(conda_path, target)

    _print(f"[ok] Repaired {TARGET_DLL_NAME}: {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ensure libomp140.x86_64.dll exists in the active torch/lib directory."
    )
    parser.add_argument(
        "--torch-lib-dir",
        help="Explicit torch lib directory. Defaults to the current interpreter's torch/lib.",
    )
    args = parser.parse_args(argv)

    try:
        ensure_torch_openmp(args.torch_lib_dir)
    except Exception as exc:
        _print(f"[error] Torch OpenMP repair failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
