"""Pinned, gated MuScriptor checkpoint preparation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.models.data_models import MuscriptorModel
from src.utils.artifact_identity import validate_file_identity
from src.utils.runtime_paths import get_resource_path

MUSCRIPTOR_MODEL_FILENAME = "model.safetensors"
MUSCRIPTOR_CONFIG_FILENAME = "config.json"


@dataclass(frozen=True)
class MuscriptorArtifact:
    model_size: str
    repo_id: str
    revision: str
    model_bytes: int
    model_sha256: str
    config_bytes: int
    config_sha256: str

    @property
    def display_name(self) -> str:
        return f"MuScriptor-{self.model_size}"


MUSCRIPTOR_ARTIFACTS = {
    MuscriptorModel.SMALL.value: MuscriptorArtifact(
        model_size=MuscriptorModel.SMALL.value,
        repo_id="MuScriptor/muscriptor-small",
        revision="8c127f603b807520fa465c838e9bfee8a91ada4e",
        model_bytes=411_888_600,
        model_sha256="bbd482c786b895cf7d8f44185073d951adae2ebb8a66f82ca84cd1f84569549c",
        config_bytes=124,
        config_sha256="3008fc481e4a1cd978e337eb3759260c270892204db5039235ac939e1f42aeb2",
    ),
    MuscriptorModel.MEDIUM.value: MuscriptorArtifact(
        model_size=MuscriptorModel.MEDIUM.value,
        repo_id="MuScriptor/muscriptor-medium",
        revision="f32236969308476e01fd3aae67357de5feb05a2d",
        model_bytes=1_228_144_472,
        model_sha256="ac80adbdf85d87231735fd948af7013441c0afced316c4e9067fd5d8a7fb97ec",
        config_bytes=126,
        config_sha256="43e13a70fc9ae0af36b7447c06f3eac2282daeb69d79c1ff840ede7fdaa26a3b",
    ),
    MuscriptorModel.LARGE.value: MuscriptorArtifact(
        model_size=MuscriptorModel.LARGE.value,
        repo_id="MuScriptor/muscriptor-large",
        revision="8809fdfbed2affa7ade94a7059e746e3880720e7",
        model_bytes=5_465_642_136,
        model_sha256="ac4eb6ea87dfc26b6ca6b954c6b967ab87ad4c7d08e078b25214f13ed051f397",
        config_bytes=125,
        config_sha256="16bedd02b18770e43740419b0d5777f231047e96e8987f498e8a1123c39c9852",
    ),
}

# Backward-compatible aliases for release and installer contracts that still
# refer specifically to the original Large checkpoint.
_LARGE_ARTIFACT = MUSCRIPTOR_ARTIFACTS[MuscriptorModel.LARGE.value]
MUSCRIPTOR_REPO_ID = _LARGE_ARTIFACT.repo_id
MUSCRIPTOR_REPO_REVISION = _LARGE_ARTIFACT.revision
MUSCRIPTOR_MODEL_EXACT_BYTES = _LARGE_ARTIFACT.model_bytes
MUSCRIPTOR_MODEL_SHA256 = _LARGE_ARTIFACT.model_sha256
MUSCRIPTOR_CONFIG_EXACT_BYTES = _LARGE_ARTIFACT.config_bytes
MUSCRIPTOR_CONFIG_SHA256 = _LARGE_ARTIFACT.config_sha256


def normalize_muscriptor_model(model_size: str | MuscriptorModel) -> str:
    value = model_size.value if isinstance(model_size, MuscriptorModel) else str(model_size)
    normalized = value.strip().lower()
    if normalized not in MUSCRIPTOR_ARTIFACTS:
        raise ValueError(f"Unsupported MuScriptor model size: {model_size!r}")
    return normalized


def get_muscriptor_artifact(model_size: str | MuscriptorModel) -> MuscriptorArtifact:
    return MUSCRIPTOR_ARTIFACTS[normalize_muscriptor_model(model_size)]


def _auth_help(artifact: MuscriptorArtifact) -> str:
    return (
        f"{artifact.display_name} 权重受 Hugging Face 门禁和 CC-BY-NC-4.0 许可约束。\n"
        f"1. 在 https://huggingface.co/{artifact.repo_id} 接受许可；\n"
        "2. 运行 `hf auth login`，或设置具有读取权限的 HF_TOKEN；\n"
        f"3. 再运行 `python download_muscriptor_model.py --size {artifact.model_size}`。"
    )


def _cached_file(artifact: MuscriptorArtifact, filename: str) -> Path | None:
    bundled = get_resource_path(f"models/muscriptor_{artifact.model_size}/{filename}")
    if bundled.is_file():
        return bundled

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    cached = try_to_load_from_cache(
        repo_id=artifact.repo_id,
        filename=filename,
        revision=artifact.revision,
    )
    if not isinstance(cached, str):
        return None
    path = Path(cached)
    return path if path.is_file() else None


@lru_cache(maxsize=24)
def _validate_exact_muscriptor_files(
    model_size: str,
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
    artifact = get_muscriptor_artifact(model_size)
    validate_file_identity(
        Path(weights_path),
        expected_size=artifact.model_bytes,
        expected_sha256=artifact.model_sha256,
        label=f"{artifact.display_name} model.safetensors",
    )
    validate_file_identity(
        Path(config_path),
        expected_size=artifact.config_bytes,
        expected_sha256=artifact.config_sha256,
        label=f"{artifact.display_name} config.json",
    )


def get_cached_muscriptor_paths(
    model_size: str | MuscriptorModel = MuscriptorModel.LARGE.value,
    *,
    validate_hashes: bool = True,
) -> tuple[Path, Path]:
    """Resolve one exact pinned checkpoint from bundled assets or the Hub cache."""

    artifact = get_muscriptor_artifact(model_size)
    weights = _cached_file(artifact, MUSCRIPTOR_MODEL_FILENAME)
    config = _cached_file(artifact, MUSCRIPTOR_CONFIG_FILENAME)
    if weights is None or config is None:
        missing = []
        if weights is None:
            missing.append(MUSCRIPTOR_MODEL_FILENAME)
        if config is None:
            missing.append(MUSCRIPTOR_CONFIG_FILENAME)
        raise RuntimeError(
            f"{artifact.display_name} 固定版本资源未准备：{', '.join(missing)}"
            f"（revision {artifact.revision}）。\n{_auth_help(artifact)}"
        )

    if validate_hashes:
        weights_stat = weights.stat()
        config_stat = config.stat()
        _validate_exact_muscriptor_files(
            artifact.model_size,
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
        if weights.stat().st_size != artifact.model_bytes:
            raise RuntimeError(
                f"{artifact.display_name} model.safetensors size mismatch: "
                f"expected {artifact.model_bytes}, got {weights.stat().st_size} ({weights})"
            )
        if config.stat().st_size != artifact.config_bytes:
            raise RuntimeError(
                f"{artifact.display_name} config.json size mismatch: "
                f"expected {artifact.config_bytes}, got {config.stat().st_size} ({config})"
            )
    return weights, config


def preflight_muscriptor_download_access(*, printer=print) -> None:
    """Fail early when a required gated checkpoint cannot be downloaded.

    A fully cached, size-validated checkpoint needs no network request.  Every
    missing checkpoint is checked with a metadata-only request against its
    pinned ``config.json`` so a fresh aggregate install does not download the
    other model families before discovering that the Hugging Face account has
    not accepted one of the three MuScriptor gates.
    """

    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
        )
    except ImportError as exc:
        raise RuntimeError(
            "缺少 huggingface_hub，无法预检 MuScriptor gated 模型访问权限；"
            "请先安装 requirements.txt。"
        ) from exc

    inaccessible: list[MuscriptorArtifact] = []
    for artifact in MUSCRIPTOR_ARTIFACTS.values():
        try:
            cached_weights, _ = get_cached_muscriptor_paths(
                artifact.model_size,
                validate_hashes=False,
            )
        except RuntimeError:
            cached_weights = None

        if cached_weights is not None:
            printer(f"{artifact.display_name} 已有完整本地缓存，无需联网授权预检。")
            continue

        try:
            metadata = get_hf_file_metadata(
                hf_hub_url(
                    repo_id=artifact.repo_id,
                    filename=MUSCRIPTOR_CONFIG_FILENAME,
                    revision=artifact.revision,
                )
            )
        except GatedRepoError:
            inaccessible.append(artifact)
            continue
        except HfHubHTTPError as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status in {401, 403}:
                inaccessible.append(artifact)
                continue
            raise RuntimeError(
                f"MuScriptor gated 访问预检网络请求失败（{artifact.repo_id}）：{exc}"
            ) from exc

        if metadata.commit_hash != artifact.revision:
            raise RuntimeError(
                f"{artifact.display_name} gated 访问预检返回了错误 revision："
                f"expected {artifact.revision}, got {metadata.commit_hash}"
            )
        if metadata.size != artifact.config_bytes:
            raise RuntimeError(
                f"{artifact.display_name} gated 访问预检返回了错误 config.json 大小："
                f"expected {artifact.config_bytes}, got {metadata.size}"
            )
        printer(f"{artifact.display_name} gated 访问权限已验证。")

    if inaccessible:
        missing_repos = "\n".join(
            f"   - https://huggingface.co/{artifact.repo_id}" for artifact in inaccessible
        )
        raise RuntimeError(
            "MuScriptor Small / Medium / Large 是 gated 模型，无法匿名全自动下载。\n"
            "必须使用同一个 Hugging Face 账户在浏览器中逐项接受三个仓库的条款；"
            "命令行登录不能代替网页接受条款。\n"
            f"当前账户无权访问：\n{missing_repos}\n"
            "接受条款后运行项目虚拟环境中的 `hf auth login`，或设置具有读取权限的 "
            "HF_TOKEN，再重新运行安装脚本。"
        )


def download_muscriptor_model(
    model_size: str | MuscriptorModel = MuscriptorModel.LARGE.value,
    *,
    printer=print,
) -> tuple[Path, Path]:
    """Download one pinned gated snapshot and require exact file identities."""

    artifact = get_muscriptor_artifact(model_size)
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            f"缺少 huggingface_hub，无法下载 {artifact.display_name}；"
            "请先安装 requirements.txt。"
        ) from exc

    try:
        printer(
            f"Preparing pinned {artifact.display_name} snapshot "
            f"{artifact.revision} ({artifact.model_bytes} bytes)"
        )
        hf_hub_download(
            repo_id=artifact.repo_id,
            filename=MUSCRIPTOR_CONFIG_FILENAME,
            revision=artifact.revision,
        )
        hf_hub_download(
            repo_id=artifact.repo_id,
            filename=MUSCRIPTOR_MODEL_FILENAME,
            revision=artifact.revision,
        )
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise RuntimeError(_auth_help(artifact)) from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {401, 403}:
            raise RuntimeError(_auth_help(artifact)) from exc
        raise RuntimeError(f"下载 {artifact.display_name} 失败：{exc}") from exc

    paths = get_cached_muscriptor_paths(artifact.model_size, validate_hashes=True)
    printer(f"{artifact.display_name} identity verified: {paths[0]}")
    return paths


def download_muscriptor_large_model(*, printer=print) -> tuple[Path, Path]:
    """Backward-compatible Large-checkpoint downloader."""

    return download_muscriptor_model(MuscriptorModel.LARGE, printer=printer)


def download_muscriptor_medium_model(*, printer=print) -> tuple[Path, Path]:
    return download_muscriptor_model(MuscriptorModel.MEDIUM, printer=printer)


def download_muscriptor_small_model(*, printer=print) -> tuple[Path, Path]:
    return download_muscriptor_model(MuscriptorModel.SMALL, printer=printer)
