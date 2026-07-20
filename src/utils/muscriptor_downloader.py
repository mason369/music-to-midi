"""Pinned, gated MuScriptor-large model preparation and validation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.utils.artifact_identity import validate_file_identity
from src.utils.runtime_paths import get_resource_path

MUSCRIPTOR_REPO_ID = "MuScriptor/muscriptor-large"
MUSCRIPTOR_REPO_REVISION = "8809fdfbed2affa7ade94a7059e746e3880720e7"
MUSCRIPTOR_MODEL_FILENAME = "model.safetensors"
MUSCRIPTOR_MODEL_EXACT_BYTES = 5_465_642_136
MUSCRIPTOR_MODEL_SHA256 = "ac4eb6ea87dfc26b6ca6b954c6b967ab87ad4c7d08e078b25214f13ed051f397"
MUSCRIPTOR_CONFIG_FILENAME = "config.json"
MUSCRIPTOR_CONFIG_EXACT_BYTES = 125
MUSCRIPTOR_CONFIG_SHA256 = "16bedd02b18770e43740419b0d5777f231047e96e8987f498e8a1123c39c9852"


def _auth_help() -> str:
    return (
        "MuScriptor-large 权重受 Hugging Face 门禁和 CC-BY-NC-4.0 许可约束。\n"
        f"1. 在 https://huggingface.co/{MUSCRIPTOR_REPO_ID} 接受许可；\n"
        "2. 运行 `hf auth login`，或设置具有读取权限的 HF_TOKEN；\n"
        "3. 再运行 `python download_muscriptor_model.py`。"
    )


def _cached_file(filename: str) -> Path | None:
    bundled = get_resource_path(f"models/muscriptor_large/{filename}")
    if bundled.is_file():
        return bundled

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    cached = try_to_load_from_cache(
        repo_id=MUSCRIPTOR_REPO_ID,
        filename=filename,
        revision=MUSCRIPTOR_REPO_REVISION,
    )
    if not isinstance(cached, str):
        return None
    path = Path(cached)
    return path if path.is_file() else None


@lru_cache(maxsize=8)
def _validate_exact_muscriptor_files(
    weights_path: str,
    weights_size: int,
    weights_modified_ns: int,
    weights_changed_ns: int,
    config_path: str,
    config_size: int,
    config_modified_ns: int,
    config_changed_ns: int,
) -> None:
    """Hash one unchanged snapshot once per process, including repeated stem jobs."""

    del (
        weights_size,
        weights_modified_ns,
        weights_changed_ns,
        config_size,
        config_modified_ns,
        config_changed_ns,
    )
    validate_file_identity(
        Path(weights_path),
        expected_size=MUSCRIPTOR_MODEL_EXACT_BYTES,
        expected_sha256=MUSCRIPTOR_MODEL_SHA256,
        label="MuScriptor-large model.safetensors",
    )
    validate_file_identity(
        Path(config_path),
        expected_size=MUSCRIPTOR_CONFIG_EXACT_BYTES,
        expected_sha256=MUSCRIPTOR_CONFIG_SHA256,
        label="MuScriptor-large config.json",
    )


def get_cached_muscriptor_paths(*, validate_hashes: bool = True) -> tuple[Path, Path]:
    """Resolve the exact pinned files from the Hub cache without network access."""

    weights = _cached_file(MUSCRIPTOR_MODEL_FILENAME)
    config = _cached_file(MUSCRIPTOR_CONFIG_FILENAME)
    if weights is None or config is None:
        missing = []
        if weights is None:
            missing.append(MUSCRIPTOR_MODEL_FILENAME)
        if config is None:
            missing.append(MUSCRIPTOR_CONFIG_FILENAME)
        raise RuntimeError(
            "MuScriptor-large 固定版本资源未准备："
            f"{', '.join(missing)}（revision {MUSCRIPTOR_REPO_REVISION}）。\n" + _auth_help()
        )

    if validate_hashes:
        weights_stat = weights.stat()
        config_stat = config.stat()
        _validate_exact_muscriptor_files(
            str(weights.resolve()),
            int(weights_stat.st_size),
            int(weights_stat.st_mtime_ns),
            int(weights_stat.st_ctime_ns),
            str(config.resolve()),
            int(config_stat.st_size),
            int(config_stat.st_mtime_ns),
            int(config_stat.st_ctime_ns),
        )
    else:
        if weights.stat().st_size != MUSCRIPTOR_MODEL_EXACT_BYTES:
            raise RuntimeError(
                "MuScriptor-large model.safetensors size mismatch: "
                f"expected {MUSCRIPTOR_MODEL_EXACT_BYTES}, got {weights.stat().st_size} "
                f"({weights})"
            )
        if config.stat().st_size != MUSCRIPTOR_CONFIG_EXACT_BYTES:
            raise RuntimeError(
                "MuScriptor-large config.json size mismatch: "
                f"expected {MUSCRIPTOR_CONFIG_EXACT_BYTES}, got {config.stat().st_size} "
                f"({config})"
            )
    return weights, config


def download_muscriptor_large_model(*, printer=print) -> tuple[Path, Path]:
    """Download the pinned gated snapshot and require exact file identities."""

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            "缺少 huggingface_hub，无法下载 MuScriptor-large；请先安装 requirements.txt。"
        ) from exc

    try:
        printer(
            "Preparing pinned MuScriptor-large snapshot "
            f"{MUSCRIPTOR_REPO_REVISION} ({MUSCRIPTOR_MODEL_EXACT_BYTES} bytes)"
        )
        hf_hub_download(
            repo_id=MUSCRIPTOR_REPO_ID,
            filename=MUSCRIPTOR_CONFIG_FILENAME,
            revision=MUSCRIPTOR_REPO_REVISION,
        )
        hf_hub_download(
            repo_id=MUSCRIPTOR_REPO_ID,
            filename=MUSCRIPTOR_MODEL_FILENAME,
            revision=MUSCRIPTOR_REPO_REVISION,
        )
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise RuntimeError(_auth_help()) from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {401, 403}:
            raise RuntimeError(_auth_help()) from exc
        raise RuntimeError(f"下载 MuScriptor-large 失败：{exc}") from exc

    paths = get_cached_muscriptor_paths(validate_hashes=True)
    printer(f"MuScriptor-large identity verified: {paths[0]}")
    return paths
