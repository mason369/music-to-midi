"""Prepare the directories read by the pinned, unmodified Aria audio transform."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ARIA_AUDIO_ASSET_DIRECTORIES = ("impulse", "noise", "applause")


def prepare_aria_audio_assets(package_root: Path | None = None) -> tuple[Path, ...]:
    """Create empty augmentation directories before inference or a read-only install.

    Official inference calls AudioTransform with augmentation disabled, but its
    constructor still scans these directories and tries to create missing ones.
    Existing training assets are preserved; package source is never patched.
    """
    if package_root is None:
        spec = importlib.util.find_spec("amt")
        if spec is None or spec.origin is None:
            raise RuntimeError("Aria-AMT 安装目录不可用")
        package_root = Path(spec.origin).parent
    directories = tuple(package_root / "assets" / name for name in ARIA_AUDIO_ASSET_DIRECTORIES)
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            # Test the same directory read that the upstream constructor performs.
            list(directory.iterdir())
        except OSError as exc:
            raise RuntimeError(
                f"Aria-AMT 音频资源目录无法准备或读取: {directory}。"
                "请在安装或构建阶段运行 python -m src.utils.aria_amt_runtime。"
                f"原始错误: {exc}"
            ) from exc
    return directories


if __name__ == "__main__":
    for asset_directory in prepare_aria_audio_assets():
        print(f"Aria-AMT 音频资源目录已准备: {asset_directory}")
