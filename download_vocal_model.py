from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Callable, Optional

LEAP_REPO_ID = "pcunwa/BS-Roformer-Leap"
LEAP_REVISION = "4e47d6662ae82eaa8b4ac4329fe66099a843b48e"
LEAP_CHECKPOINT_REPO_PATH = "Xe/bs_leap_xe_voc.ckpt"
LEAP_CONFIG_REPO_PATH = "Xe/leap_xe_config_voc.yaml"
LEAP_CHECKPOINT_NAME = Path(LEAP_CHECKPOINT_REPO_PATH).name
LEAP_CONFIG_NAME = Path(LEAP_CONFIG_REPO_PATH).name
LEAP_CHECKPOINT_SIZE = 267_796_851
LEAP_CHECKPOINT_SHA256 = "b739c1d2d87a81cd3dd3844ed9ad0bd678708c7a0a761a03a1aaff9af79a096d"
LEAP_CONFIG_SIZE = 2_190
LEAP_CONFIG_SHA256 = "d3cb8c84be2e9bcbc64c1086e2256dc097ab5addebc5ce507818e90ff8cbdc25"

# Compatibility names used by the installer and aggregate downloader.
ROFORMER_MODEL = LEAP_CHECKPOINT_NAME
ROFORMER_MODELS = (LEAP_CHECKPOINT_NAME, LEAP_CONFIG_NAME)
DEFAULT_CACHE_DIR = Path.home() / ".music-to-midi" / "models" / "audio-separator"


def _resolve_cache_dir(cache_dir: Optional[Path]) -> Path:
    return Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR


def _find_asset(cache_dir: Path, asset_name: str) -> Optional[Path]:
    direct = cache_dir / asset_name
    if direct.is_file() and direct.stat().st_size > 0:
        return direct
    for path in cache_dir.rglob(asset_name):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def resolve_vocal_model_path(
    cache_dir: Optional[Path] = None,
    model_name: str = LEAP_CHECKPOINT_NAME,
) -> Path:
    root = _resolve_cache_dir(cache_dir)
    return _find_asset(root, model_name) or root / model_name


def resolve_vocal_config_path(cache_dir: Optional[Path] = None) -> Path:
    root = _resolve_cache_dir(cache_dir)
    return _find_asset(root, LEAP_CONFIG_NAME) or root / LEAP_CONFIG_NAME


def resolve_vocal_model_paths(
    cache_dir: Optional[Path] = None,
    model_names: tuple[str, ...] = ROFORMER_MODELS,
) -> tuple[Path, ...]:
    root = _resolve_cache_dir(cache_dir)
    return tuple(_find_asset(root, name) or root / name for name in model_names)


def _checkpoint_has_expected_identity(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == LEAP_CHECKPOINT_SIZE
        and _sha256(path) == LEAP_CHECKPOINT_SHA256
    )


def _config_has_expected_identity(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == LEAP_CONFIG_SIZE
        and _sha256(path) == LEAP_CONFIG_SHA256
    )


def is_vocal_model_available(cache_dir: Optional[Path] = None) -> bool:
    checkpoint = resolve_vocal_model_path(cache_dir)
    config = resolve_vocal_config_path(cache_dir)
    return (
        _checkpoint_has_expected_identity(checkpoint)
        and _config_has_expected_identity(config)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_asset(cache_dir: Path, repo_path: str, downloader) -> Path:
    downloaded = Path(
        downloader(
            repo_id=LEAP_REPO_ID,
            filename=repo_path,
            revision=LEAP_REVISION,
            repo_type="model",
            local_dir=str(cache_dir),
        )
    )
    if not downloaded.is_file() or downloaded.stat().st_size <= 0:
        raise RuntimeError(f"Hugging Face download did not create a valid asset: {downloaded}")
    return downloaded


def download_vocal_model(
    cache_dir: Optional[Path] = None,
    model_name: str = ROFORMER_MODEL,
    separator_cls=None,
    printer: Callable[[str], None] = print,
    downloader=None,
) -> Path:
    """Download this project's pinned Leap XE 90-band vocals checkpoint and config."""
    del separator_cls  # Kept only for source compatibility with older callers.
    if model_name not in (ROFORMER_MODEL, LEAP_CHECKPOINT_NAME):
        raise ValueError(
            f"Unsupported vocal model {model_name!r}; expected {LEAP_CHECKPOINT_NAME!r}"
        )

    cache_dir = _resolve_cache_dir(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_vocal_model_path(cache_dir)
    config = resolve_vocal_config_path(cache_dir)
    if (
        _checkpoint_has_expected_identity(checkpoint)
        and _config_has_expected_identity(config)
    ):
        printer("BS-RoFormer Leap XE 90-band vocals assets already exist; skipping download.")
        return checkpoint

    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to download BS-RoFormer Leap XE assets"
            ) from exc
        downloader = hf_hub_download

    printer(f"Downloading BS-RoFormer Leap XE vocals from {LEAP_REPO_ID}@{LEAP_REVISION}")
    checkpoint = _download_asset(cache_dir, LEAP_CHECKPOINT_REPO_PATH, downloader)
    config = _download_asset(cache_dir, LEAP_CONFIG_REPO_PATH, downloader)

    if checkpoint.stat().st_size != LEAP_CHECKPOINT_SIZE:
        raise RuntimeError(
            f"Leap XE checkpoint size mismatch: expected {LEAP_CHECKPOINT_SIZE}, "
            f"got {checkpoint.stat().st_size} ({checkpoint})"
        )
    actual_sha256 = _sha256(checkpoint)
    if actual_sha256 != LEAP_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Leap XE checkpoint SHA256 mismatch: "
            f"expected {LEAP_CHECKPOINT_SHA256}, got {actual_sha256}"
        )
    if not _config_has_expected_identity(config):
        raise RuntimeError(
            "Leap XE config identity mismatch: "
            f"expected size={LEAP_CONFIG_SIZE}, sha256={LEAP_CONFIG_SHA256}, path={config}"
        )

    printer(f"Leap XE checkpoint ready: {checkpoint}")
    printer(f"Leap XE config ready: {config}")
    return checkpoint


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download BS-RoFormer Leap XE 90-band vocals assets"
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Model cache directory (default: ~/.music-to-midi/models/audio-separator)",
    )
    args = parser.parse_args(argv)

    try:
        download_vocal_model(cache_dir=Path(args.cache_dir))
        return 0
    except Exception as exc:
        print(f"[error] BS-RoFormer Leap XE download failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
