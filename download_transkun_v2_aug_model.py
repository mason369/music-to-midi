"""Download and verify the official TransKun V2 Aug checkpoint archive."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from src.core.transkun_v2_aug_transcriber import (
    TRANSKUN_V2_AUG_ARCHIVE_NAME,
    TRANSKUN_V2_AUG_ARCHIVE_SHA256,
    TRANSKUN_V2_AUG_ARCHIVE_SIZE,
    TRANSKUN_V2_AUG_CACHE_ROOT,
    TRANSKUN_V2_AUG_CHECKPOINT_NAME,
    TRANSKUN_V2_AUG_CHECKPOINT_SHA256,
    TRANSKUN_V2_AUG_CHECKPOINT_SIZE,
    TRANSKUN_V2_AUG_CONFIG_NAME,
    TRANSKUN_V2_AUG_CONFIG_SHA256,
    TRANSKUN_V2_AUG_CONFIG_SIZE,
    TRANSKUN_V2_AUG_DOWNLOAD_URL,
    TRANSKUN_V2_AUG_MODEL_DIR_NAME,
    validate_transkun_v2_aug_model_files,
)

logger = logging.getLogger(__name__)

TRANSKUN_V2_AUG_DOWNLOAD_ATTEMPTS = 3
TRANSKUN_V2_AUG_RETRY_DELAY_SECONDS = 5


def _log(printer: Optional[Callable[[str], None]], message: str) -> None:
    if printer is not None:
        printer(message)
        return
    logger.info(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_transkun_v2_aug_model_dir(
    cache_root: Path = TRANSKUN_V2_AUG_CACHE_ROOT,
) -> Path:
    return Path(cache_root) / TRANSKUN_V2_AUG_MODEL_DIR_NAME


def is_transkun_v2_aug_model_available(
    cache_root: Path = TRANSKUN_V2_AUG_CACHE_ROOT,
) -> bool:
    model_dir = resolve_transkun_v2_aug_model_dir(cache_root)
    return validate_transkun_v2_aug_model_files(model_dir) == ""


def _download_archive(url: str, output_path: Path) -> None:
    request = urllib.request.Request(
        url=url,
        headers={"User-Agent": "music-to-midi-downloader/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with output_path.open("wb") as output_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)


def _is_retriable_download_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return 500 <= int(error.code) <= 599
    return isinstance(error, (TimeoutError, urllib.error.URLError))


def _validate_archive(archive_path: Path) -> None:
    actual_size = archive_path.stat().st_size
    if actual_size != TRANSKUN_V2_AUG_ARCHIVE_SIZE:
        raise RuntimeError(
            f"TransKun V2 Aug 压缩包大小不匹配: {actual_size} != {TRANSKUN_V2_AUG_ARCHIVE_SIZE}"
        )
    actual_sha256 = _sha256_file(archive_path)
    if actual_sha256 != TRANSKUN_V2_AUG_ARCHIVE_SHA256:
        raise RuntimeError(
            "TransKun V2 Aug 压缩包 SHA256 不匹配: "
            f"{actual_sha256} != {TRANSKUN_V2_AUG_ARCHIVE_SHA256}"
        )


def _extract_verified_model(archive_path: Path, model_dir: Path) -> None:
    expected = {
        f"{TRANSKUN_V2_AUG_MODEL_DIR_NAME}/{TRANSKUN_V2_AUG_CHECKPOINT_NAME}": (
            TRANSKUN_V2_AUG_CHECKPOINT_NAME,
            TRANSKUN_V2_AUG_CHECKPOINT_SIZE,
            TRANSKUN_V2_AUG_CHECKPOINT_SHA256,
        ),
        f"{TRANSKUN_V2_AUG_MODEL_DIR_NAME}/{TRANSKUN_V2_AUG_CONFIG_NAME}": (
            TRANSKUN_V2_AUG_CONFIG_NAME,
            TRANSKUN_V2_AUG_CONFIG_SIZE,
            TRANSKUN_V2_AUG_CONFIG_SHA256,
        ),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    partial_paths: list[tuple[Path, Path]] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            archived_files = {info.filename for info in archive.infolist() if not info.is_dir()}
            if archived_files != set(expected):
                raise RuntimeError(
                    f"TransKun V2 Aug 压缩包文件布局不匹配: {sorted(archived_files)!r}"
                )

            for archive_name, (target_name, expected_size, expected_sha256) in expected.items():
                info = archive.getinfo(archive_name)
                if info.file_size != expected_size:
                    raise RuntimeError(
                        f"压缩包内 {archive_name} 大小不匹配: "
                        f"{info.file_size} != {expected_size}"
                    )

                target_path = model_dir / target_name
                partial_path = target_path.with_suffix(target_path.suffix + ".download")
                partial_paths.append((partial_path, target_path))
                if partial_path.exists():
                    partial_path.unlink()

                digest = hashlib.sha256()
                with archive.open(info, "r") as source, partial_path.open("wb") as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        output.write(chunk)

                actual_sha256 = digest.hexdigest()
                if actual_sha256 != expected_sha256:
                    raise RuntimeError(
                        f"压缩包内 {archive_name} SHA256 不匹配: "
                        f"{actual_sha256} != {expected_sha256}"
                    )

        for partial_path, target_path in partial_paths:
            partial_path.replace(target_path)
    except Exception:
        for partial_path, _target_path in partial_paths:
            if partial_path.exists():
                partial_path.unlink()
        raise

    validation_error = validate_transkun_v2_aug_model_files(model_dir)
    if validation_error:
        raise RuntimeError(validation_error)


def download_transkun_v2_aug_model(
    cache_root: Path = TRANSKUN_V2_AUG_CACHE_ROOT,
    downloader: Optional[Callable[[str, Path], None]] = None,
    printer: Optional[Callable[[str], None]] = print,
) -> Path:
    cache_root = Path(cache_root)
    model_dir = resolve_transkun_v2_aug_model_dir(cache_root)
    validation_error = validate_transkun_v2_aug_model_files(model_dir)
    if not validation_error:
        _log(printer, f"TransKun V2 Aug 模型已存在且校验通过: {model_dir}")
        return model_dir

    if model_dir.exists():
        _log(printer, f"现有 TransKun V2 Aug 模型无效: {validation_error}")

    cache_root.mkdir(parents=True, exist_ok=True)
    archive_path = cache_root / TRANSKUN_V2_AUG_ARCHIVE_NAME
    archive_valid = False
    if archive_path.is_file():
        try:
            _validate_archive(archive_path)
            archive_valid = True
            _log(printer, f"使用已校验的官方压缩包: {archive_path}")
        except RuntimeError as exc:
            _log(printer, f"现有压缩包无效，将重新下载: {exc}")

    if not archive_valid:
        if downloader is None:
            downloader = _download_archive
        partial_archive = archive_path.with_suffix(archive_path.suffix + ".download")
        if partial_archive.exists():
            partial_archive.unlink()

        for attempt in range(1, TRANSKUN_V2_AUG_DOWNLOAD_ATTEMPTS + 1):
            _log(
                printer,
                f"正在下载 {TRANSKUN_V2_AUG_ARCHIVE_NAME} "
                f"({attempt}/{TRANSKUN_V2_AUG_DOWNLOAD_ATTEMPTS})...",
            )
            try:
                downloader(TRANSKUN_V2_AUG_DOWNLOAD_URL, partial_archive)
                _validate_archive(partial_archive)
                partial_archive.replace(archive_path)
                break
            except Exception as exc:
                if partial_archive.exists():
                    partial_archive.unlink()
                if (
                    attempt >= TRANSKUN_V2_AUG_DOWNLOAD_ATTEMPTS
                    or not _is_retriable_download_error(exc)
                ):
                    raise
                _log(
                    printer,
                    f"下载暂时失败: {exc}。{TRANSKUN_V2_AUG_RETRY_DELAY_SECONDS} 秒后重试...",
                )
                time.sleep(TRANSKUN_V2_AUG_RETRY_DELAY_SECONDS)

    _extract_verified_model(archive_path, model_dir)
    _log(printer, f"TransKun V2 Aug 模型已下载并校验: {model_dir}")
    return model_dir


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="下载官方 TransKun V2 Aug checkpoint")
    parser.add_argument(
        "--cache-dir",
        default=str(TRANSKUN_V2_AUG_CACHE_ROOT),
        help="模型缓存根目录",
    )
    args = parser.parse_args(argv)

    try:
        download_transkun_v2_aug_model(cache_root=Path(args.cache_dir))
        return 0
    except Exception as exc:
        print(f"[错误] TransKun V2 Aug 模型下载失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
