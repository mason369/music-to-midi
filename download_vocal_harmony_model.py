from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Callable, Optional

POLARFORMER_REPO_ID = "bgkb/bs_polarformer"
POLARFORMER_REVISION = "9158719ee2173edd480a735764627526506fe4af"
POLARFORMER_ONNX_NAME = "bs_polarformer.onnx"
POLARFORMER_CONFIG_NAME = "model_bs_polarformer_float16.yaml"
POLARFORMER_ONNX_SIZE = 210_652_828
POLARFORMER_ONNX_SHA256 = "1c6857c34556c72d4094d4515c5725549bf987a63a1a8c37a7e7fc111b525c50"
POLARFORMER_CONFIG_SIZE = 3_599
POLARFORMER_CONFIG_SHA256 = (
    "0348205cb562a58e9724870a4cf43e5d2c49ae87258159b2827c4e42ed51b00d"
)

# Compatibility names retained for installers that still import this module by its
# historical filename. These now refer exclusively to the PolarFormer accompaniment leg.
CHORUS_PRESET = "polarformer_onnx"
CHORUS_MODEL = POLARFORMER_ONNX_NAME
CHORUS_MODELS = (POLARFORMER_ONNX_NAME, POLARFORMER_CONFIG_NAME)
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


def resolve_accompaniment_model_path(cache_dir: Optional[Path] = None) -> Path:
    root = _resolve_cache_dir(cache_dir)
    return _find_asset(root, POLARFORMER_ONNX_NAME) or root / POLARFORMER_ONNX_NAME


def resolve_accompaniment_config_path(cache_dir: Optional[Path] = None) -> Path:
    root = _resolve_cache_dir(cache_dir)
    return _find_asset(root, POLARFORMER_CONFIG_NAME) or root / POLARFORMER_CONFIG_NAME


def resolve_chorus_model_path(
    cache_dir: Optional[Path] = None,
    model_name: str = CHORUS_MODELS[0],
) -> Path:
    root = _resolve_cache_dir(cache_dir)
    return _find_asset(root, model_name) or root / model_name


def resolve_chorus_model_paths(
    cache_dir: Optional[Path] = None,
    model_names: tuple[str, ...] = CHORUS_MODELS,
) -> tuple[Path, ...]:
    root = _resolve_cache_dir(cache_dir)
    return tuple(_find_asset(root, name) or root / name for name in model_names)


def _onnx_has_expected_identity(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == POLARFORMER_ONNX_SIZE
        and _sha256(path) == POLARFORMER_ONNX_SHA256
    )


def _config_has_expected_identity(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == POLARFORMER_CONFIG_SIZE
        and _sha256(path) == POLARFORMER_CONFIG_SHA256
    )


def is_accompaniment_model_available(cache_dir: Optional[Path] = None) -> bool:
    model = resolve_accompaniment_model_path(cache_dir)
    config = resolve_accompaniment_config_path(cache_dir)
    return _onnx_has_expected_identity(model) and _config_has_expected_identity(config)


def is_chorus_model_available(cache_dir: Optional[Path] = None) -> bool:
    return is_accompaniment_model_available(cache_dir)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_asset(cache_dir: Path, filename: str, downloader) -> Path:
    downloaded = Path(
        downloader(
            repo_id=POLARFORMER_REPO_ID,
            filename=filename,
            revision=POLARFORMER_REVISION,
            repo_type="model",
            local_dir=str(cache_dir),
        )
    )
    if not downloaded.is_file() or downloaded.stat().st_size <= 0:
        raise RuntimeError(f"Hugging Face download did not create a valid asset: {downloaded}")
    return downloaded


def download_accompaniment_model(
    cache_dir: Optional[Path] = None,
    model_name: str = CHORUS_MODEL,
    separator_cls=None,
    printer: Callable[[str], None] = print,
    downloader=None,
) -> Path:
    """Download the pinned public FP32 BS PolarFormer ONNX model and config."""
    del separator_cls  # Kept only for source compatibility with older callers.
    if model_name not in (CHORUS_MODEL, POLARFORMER_ONNX_NAME):
        raise ValueError(
            f"Unsupported accompaniment model {model_name!r}; "
            f"expected {POLARFORMER_ONNX_NAME!r}"
        )

    cache_dir = _resolve_cache_dir(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = resolve_accompaniment_model_path(cache_dir)
    config_path = resolve_accompaniment_config_path(cache_dir)
    if (
        _onnx_has_expected_identity(onnx_path)
        and _config_has_expected_identity(config_path)
    ):
        printer("BS PolarFormer public FP32 ONNX assets already exist; skipping download.")
        return onnx_path

    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to download BS PolarFormer assets"
            ) from exc
        downloader = hf_hub_download

    printer(
        f"Downloading BS PolarFormer public ONNX from "
        f"{POLARFORMER_REPO_ID}@{POLARFORMER_REVISION}"
    )
    onnx_path = _download_asset(cache_dir, POLARFORMER_ONNX_NAME, downloader)
    config_path = _download_asset(cache_dir, POLARFORMER_CONFIG_NAME, downloader)

    if onnx_path.stat().st_size != POLARFORMER_ONNX_SIZE:
        raise RuntimeError(
            f"PolarFormer ONNX size mismatch: expected {POLARFORMER_ONNX_SIZE}, "
            f"got {onnx_path.stat().st_size} ({onnx_path})"
        )
    actual_sha256 = _sha256(onnx_path)
    if actual_sha256 != POLARFORMER_ONNX_SHA256:
        raise RuntimeError(
            "PolarFormer ONNX SHA256 mismatch: "
            f"expected {POLARFORMER_ONNX_SHA256}, got {actual_sha256}"
        )
    if not _config_has_expected_identity(config_path):
        raise RuntimeError(
            "PolarFormer config identity mismatch: "
            f"expected size={POLARFORMER_CONFIG_SIZE}, "
            f"sha256={POLARFORMER_CONFIG_SHA256}, path={config_path}"
        )

    printer(f"PolarFormer ONNX ready: {onnx_path}")
    printer(f"PolarFormer config ready: {config_path}")
    return onnx_path


def download_chorus_model(
    cache_dir: Optional[Path] = None,
    model_name: str = CHORUS_MODEL,
    separator_cls=None,
    printer: Callable[[str], None] = print,
    downloader=None,
) -> Path:
    """Compatibility entry point for the PolarFormer accompaniment downloader."""
    return download_accompaniment_model(
        cache_dir=cache_dir,
        model_name=model_name,
        separator_cls=separator_cls,
        printer=printer,
        downloader=downloader,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download BS PolarFormer public FP32 ONNX accompaniment assets"
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Model cache directory (default: ~/.music-to-midi/models/audio-separator)",
    )
    args = parser.parse_args(argv)

    try:
        download_accompaniment_model(cache_dir=Path(args.cache_dir))
        return 0
    except Exception as exc:
        print(f"[error] BS PolarFormer download failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
