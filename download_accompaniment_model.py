"""Prepare the pinned standard Leap Instrumental checkpoint and author config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

from src.utils.artifact_identity import validate_file_identity

LEAP_INSTRUMENTAL_REPO_ID = "pcunwa/BS-Roformer-Leap"
LEAP_INSTRUMENTAL_REVISION = "4e47d6662ae82eaa8b4ac4329fe66099a843b48e"
LEAP_INSTRUMENTAL_CHECKPOINT_NAME = "bs_roformer_leap_inst.ckpt"
LEAP_INSTRUMENTAL_CONFIG_NAME = "bs_leap_inst_conf.yaml"
LEAP_INSTRUMENTAL_CHECKPOINT_SIZE = 238_212_233
LEAP_INSTRUMENTAL_CHECKPOINT_SHA256 = (
    "1459b5eb70d7e1a805d8d080cb0d1abfcd06dc3931daedecad59944a8f7a0a57"
)
LEAP_INSTRUMENTAL_CONFIG_SIZE = 1_960
LEAP_INSTRUMENTAL_CONFIG_SHA256 = "2004e01f7019e9dd7bad44dd213b337dd055806811266f0f0ac925b3da33cb29"
DEFAULT_CACHE_DIR = Path.home() / ".music-to-midi" / "models" / "audio-separator"

# Historical installer entry points select the same current accompaniment model.
CHORUS_PRESET = "leap_instrumental"
CHORUS_MODEL = LEAP_INSTRUMENTAL_CHECKPOINT_NAME
CHORUS_MODELS = (LEAP_INSTRUMENTAL_CHECKPOINT_NAME, LEAP_INSTRUMENTAL_CONFIG_NAME)


def _resolve_asset(cache_dir: Optional[Path], filename: str) -> Path:
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    direct = root / filename
    if direct.is_file():
        return direct
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if len(matches) > 1:
        raise RuntimeError(f"伴奏模型路径不唯一: {filename}: {matches}")
    return matches[0] if matches else direct


def resolve_accompaniment_model_path(cache_dir: Optional[Path] = None) -> Path:
    return _resolve_asset(cache_dir, LEAP_INSTRUMENTAL_CHECKPOINT_NAME)


def resolve_accompaniment_config_path(cache_dir: Optional[Path] = None) -> Path:
    return _resolve_asset(cache_dir, LEAP_INSTRUMENTAL_CONFIG_NAME)


def resolve_chorus_model_path(
    cache_dir: Optional[Path] = None, model_name: str = CHORUS_MODEL
) -> Path:
    if model_name not in CHORUS_MODELS:
        raise ValueError(f"不支持的伴奏模型资源: {model_name!r}")
    return _resolve_asset(cache_dir, model_name)


def resolve_chorus_model_paths(
    cache_dir: Optional[Path] = None, model_names: tuple[str, ...] = CHORUS_MODELS
) -> tuple[Path, ...]:
    return tuple(resolve_chorus_model_path(cache_dir, name) for name in model_names)


def validate_accompaniment_assets(cache_dir: Optional[Path] = None) -> tuple[Path, Path]:
    checkpoint = validate_file_identity(
        resolve_accompaniment_model_path(cache_dir),
        expected_size=LEAP_INSTRUMENTAL_CHECKPOINT_SIZE,
        expected_sha256=LEAP_INSTRUMENTAL_CHECKPOINT_SHA256,
        label="Leap Instrumental checkpoint",
    )
    config = validate_file_identity(
        resolve_accompaniment_config_path(cache_dir),
        expected_size=LEAP_INSTRUMENTAL_CONFIG_SIZE,
        expected_sha256=LEAP_INSTRUMENTAL_CONFIG_SHA256,
        label="Leap Instrumental config",
    )
    return checkpoint, config


def is_accompaniment_model_available(cache_dir: Optional[Path] = None) -> bool:
    try:
        validate_accompaniment_assets(cache_dir)
    except (OSError, RuntimeError):
        return False
    return True


def is_chorus_model_available(cache_dir: Optional[Path] = None) -> bool:
    return is_accompaniment_model_available(cache_dir)


def download_accompaniment_model(
    cache_dir: Optional[Path] = None,
    model_name: str = CHORUS_MODEL,
    separator_cls=None,
    printer: Callable[[str], None] = print,
    downloader=None,
) -> Path:
    """Download missing pinned assets; existing corrupt files fail explicitly."""
    del separator_cls
    if model_name != LEAP_INSTRUMENTAL_CHECKPOINT_NAME:
        raise ValueError(f"不支持的伴奏模型: {model_name!r}；需要 {CHORUS_MODEL!r}")
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    for name, size, digest, label in (
        (
            LEAP_INSTRUMENTAL_CHECKPOINT_NAME,
            LEAP_INSTRUMENTAL_CHECKPOINT_SIZE,
            LEAP_INSTRUMENTAL_CHECKPOINT_SHA256,
            "Leap Instrumental checkpoint",
        ),
        (
            LEAP_INSTRUMENTAL_CONFIG_NAME,
            LEAP_INSTRUMENTAL_CONFIG_SIZE,
            LEAP_INSTRUMENTAL_CONFIG_SHA256,
            "Leap Instrumental config",
        ),
    ):
        path = _resolve_asset(root, name)
        if not path.exists():
            if downloader is None:
                from huggingface_hub import hf_hub_download

                downloader = hf_hub_download
            printer(f"正在下载 Leap Instrumental: {name}")
            path = Path(
                downloader(
                    repo_id=LEAP_INSTRUMENTAL_REPO_ID,
                    revision=LEAP_INSTRUMENTAL_REVISION,
                    filename=name,
                    repo_type="model",
                    local_dir=str(root),
                )
            )
        validate_file_identity(path, expected_size=size, expected_sha256=digest, label=label)
    checkpoint, _ = validate_accompaniment_assets(root)
    printer("Leap Instrumental 权重与原始配置已通过大小及 SHA-256 校验。")
    return checkpoint


def download_chorus_model(
    cache_dir: Optional[Path] = None,
    model_name: str = CHORUS_MODEL,
    separator_cls=None,
    printer: Callable[[str], None] = print,
    downloader=None,
) -> Path:
    return download_accompaniment_model(
        cache_dir=cache_dir,
        model_name=model_name,
        separator_cls=separator_cls,
        printer=printer,
        downloader=downloader,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="下载并校验标准 Leap Instrumental 伴奏模型")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="模型缓存目录")
    args = parser.parse_args(argv)
    try:
        download_accompaniment_model(cache_dir=Path(args.cache_dir))
    except Exception as exc:
        print(f"[错误] Leap Instrumental 下载或校验失败: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
