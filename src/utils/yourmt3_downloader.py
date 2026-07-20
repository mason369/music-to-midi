"""
YourMT3+ 模型下载工具

处理模型的下载、缓存和验证
支持 Hugging Face 镜像站点和断点续传
"""

import hashlib
import logging
import os
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Callable

from src.utils.runtime_paths import (
    get_runtime_data_dir,
    get_yourmt3_download_root,
    get_yourmt3_search_roots,
)

logger = logging.getLogger(__name__)

_ALLOW_INSECURE_HF_DOWNLOAD = "ALLOW_INSECURE_HF_DOWNLOAD"


def _env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _fix_ssl_if_needed():
    """
    检测并修复 SSL 证书验证问题。
    在企业网络/校园网/代理环境下，HTTPS 流量可能被拦截导致证书验证失败。
    """
    test_url = "https://huggingface.co"
    try:
        req = urllib.request.Request(test_url, method="HEAD")
        urllib.request.urlopen(req, timeout=10)
        return
    except (ssl.SSLCertVerificationError, urllib.error.URLError):
        pass
    except Exception:
        pass

    # 尝试 certifi
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        req = urllib.request.Request(test_url, method="HEAD")
        ctx = ssl.create_default_context(cafile=certifi.where())
        urllib.request.urlopen(req, timeout=10, context=ctx)
        logger.info("已使用 certifi 证书修复 SSL 验证")
        return
    except Exception:
        pass

    if not _env_flag_enabled(_ALLOW_INSECURE_HF_DOWNLOAD):
        raise RuntimeError(
            "SSL 证书验证失败，已停止下载。\n"
            "请配置系统/代理 CA，或设置 SSL_CERT_FILE / REQUESTS_CA_BUNDLE 指向可信 CA。\n"
            f"仅在你明确接受风险时，才可设置 {_ALLOW_INSECURE_HF_DOWNLOAD}=1 跳过验证。"
        )

    logger.warning(
        "SSL certificate verification failed; %s=1 is set, disabling verification for Hugging Face download.",
        _ALLOW_INSECURE_HF_DOWNLOAD,
    )
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["CURL_CA_BUNDLE"] = ""
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass


# Hugging Face 镜像站点列表（用于国内网络环境）
HF_MIRRORS = [
    None,  # 官方站点
    "https://hf-mirror.com",  # 国内镜像
    "https://huggingface.co",  # 备用
]

# 模型配置 - YourMT3+ (2024年7月发布，MLSP2024论文)
YOURMT3_MODELS = {
    # YourMT3+ 模型族。
    "ymt3_plus": {
        "name": "YMT3+",
        "ui_label": "YMT3+",
        "description": "YourMT3+ 基线模型，使用原始 YMT3+ 风格配置。",
        "ui_description": "Baseline YourMT3+ model using MT3 tokens with singing extension.",
        "checkpoint": "YMT3+",
        "size_mb": 2000,
        "features": ["YourMT3+ baseline", "MT3 tokens with singing extension"],
        "features_zh": ["YourMT3+ 基线模型", "带歌声扩展的 MT3 token"],
    },
    "yptf_single_nops": {
        "name": "YPTF+Single (noPS)",
        "ui_label": "YPTF+Single (noPS)",
        "description": "Perceiver-TF 编码器 + 单解码器，不使用音高偏移增强。",
        "ui_description": "Single-decoder Perceiver-TF model without pitch-shift augmentation.",
        "checkpoint": "YPTF+Single (noPS)",
        "size_mb": 2000,
        "features": ["Perceiver-TF", "single decoder", "no pitch-shift augmentation"],
        "features_zh": ["Perceiver-TF", "单解码器", "不使用音高偏移增强"],
    },
    "yptf_multi_ps": {
        "name": "YPTF+Multi (PS)",
        "ui_label": "YPTF+Multi (PS)",
        "description": "Perceiver-TF 编码器 + multi-t5 多通道解码，使用音高偏移增强。",
        "ui_description": "Multi-channel Perceiver-TF checkpoint using multi-t5 / mc13_full_plus_256 style decoding with pitch-shift augmentation.",
        "checkpoint": "YPTF+Multi (PS)",
        "size_mb": 2000,
        "features": [
            "Perceiver-TF",
            "multi-t5",
            "multi-channel decoding",
            "pitch-shift augmentation",
        ],
        "features_zh": ["Perceiver-TF", "multi-t5", "多通道解码", "使用音高偏移增强"],
    },
    "yptf_moe_multi_nops": {
        "name": "YPTF.MoE+Multi (noPS)",
        "ui_label": "YPTF.MoE+Multi (noPS)",
        "description": "本项目默认模型，采用 Perceiver-TF、MoE 与 multi-t5 多通道解码，不使用音高偏移增强。",
        "ui_description": "Project default using Perceiver-TF, MoE, and multi-channel decoding without pitch-shift augmentation.",
        "checkpoint": "YPTF.MoE+Multi (noPS)",
        "size_mb": 2500,
        "recommended": True,
        "features": ["MoE", "Perceiver-TF", "multi-t5", "multi-channel decoding"],
        "features_zh": ["MoE", "Perceiver-TF", "multi-t5", "多通道解码"],
    },
    "yptf_moe_multi_ps": {
        "name": "YPTF.MoE+Multi (PS)",
        "ui_label": "YPTF.MoE+Multi (PS)",
        "description": "Perceiver-TF + MoE + multi-t5 多通道解码，使用音高偏移增强。",
        "ui_description": "Perceiver-TF, MoE, and multi-channel decoding with pitch-shift augmentation.",
        "checkpoint": "YPTF.MoE+Multi (PS)",
        "size_mb": 2500,
        "features": ["MoE", "Perceiver-TF", "multi-t5", "pitch-shift augmentation"],
        "features_zh": ["MoE", "Perceiver-TF", "multi-t5", "使用音高偏移增强"],
    },
    # 传统checkpoint名称格式（兼容旧版）
    "mc13_256_all_cross_v6": {
        "name": "YourMT3+ 全乐器模型 (传统版)",
        "ui_label": "YourMT3+ legacy multi",
        "description": "支持多种乐器的通用转写模型",
        "ui_description": "General multi-instrument YourMT3+ model with cross-dataset augmentation.",
        "checkpoint": "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt",
        "size_mb": 2000,
        "default": True,  # 保持向后兼容
        "features": ["通用多乐器", "跨数据集增强"],
        "features_en": ["general multi-instrument transcription", "cross-dataset augmentation"],
        "features_zh": ["通用多乐器", "跨数据集增强"],
    },
}

OFFICIAL_YOURMT3_MODEL_KEYS = (
    "ymt3_plus",
    "yptf_single_nops",
    "yptf_multi_ps",
    "yptf_moe_multi_nops",
    "yptf_moe_multi_ps",
)

# 统一的 Hugging Face 仓库
YOURMT3_REPO_ID = "mimbres/YourMT3"
YOURMT3_REVISION = "5e66c1ea173a8186e0d20432b841d3180cc015b5"
YOURMT3_MODEL_IDENTITIES = {
    "ymt3_plus": {
        "filename": (
            "amt/logs/2024/notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72/"
            "checkpoints/model.ckpt"
        ),
        "size": 542_707_465,
        "sha256": "76673d4289aae1a66984b3fee59157d484cda8fdf9b56279251b137321d448e6",
    },
    "yptf_single_nops": {
        "filename": (
            "amt/logs/2024/ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100/"
            "checkpoints/model.ckpt"
        ),
        "size": 361_050_039,
        "sha256": "507ff129bf68f65b3b3439368706fc413523fc729d994e4da2c3f50860b05dde",
    },
    "yptf_multi_ps": {
        "filename": (
            "amt/logs/2024/ptf_mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_"
            "full_plus_2psn_nl26_sb_b26r_800k/checkpoints/model.ckpt"
        ),
        "size": 541_553_263,
        "sha256": "f7ed46a7c61244bd35485143a791b3cf46515ef96239793845ae8e4084865130",
    },
    "yptf_moe_multi_nops": {
        "filename": (
            "amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_"
            "rp_b36_nops/checkpoints/last.ckpt"
        ),
        "size": 561_544_628,
        "sha256": "ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25",
    },
    "yptf_moe_multi_ps": {
        "filename": (
            "amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_"
            "rp_b80_ps2/checkpoints/model.ckpt"
        ),
        "size": 758_957_292,
        "sha256": "7427055b51c3c8c86f6a35493cf0741d8db29186052055b59193590a82e6ec01",
    },
}
YOURMT3_MODEL_IDENTITIES["mc13_256_all_cross_v6"] = YOURMT3_MODEL_IDENTITIES["yptf_multi_ps"]

# Checkpoint 名称到实际目录名的映射表
# 用于将 Hugging Face 上的模型名称映射到本地文件系统结构
CHECKPOINT_FILENAME_MAP = {
    "YMT3+": "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72",
    "YPTF+Single (noPS)": "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100",
    "YPTF+Multi (PS)": "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k",
    "YPTF.MoE+Multi (PS)": "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2",
    "YPTF.MoE+Multi (noPS)": "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops",
    "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt": "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k",
}

# 项目默认模型
DEFAULT_MODEL = "yptf_moe_multi_nops"


def get_model_cache_dir() -> Path:
    """获取模型缓存目录"""
    # 该目录用于保存下载元信息与 huggingface 临时缓存，而非最终 checkpoint 主目录
    if getattr(sys, "frozen", False):
        cache_dir = get_runtime_data_dir() / "hf_cache" / "yourmt3"
    else:
        cache_dir = Path.home() / ".cache" / "yourmt3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def resolve_model_checkpoint_path(checkpoint_name: str) -> Optional[Path]:
    """
    将 checkpoint 名称解析为实际文件系统路径。

    不依赖硬编码目录结构，在整个 cache 目录下递归搜索包含
    dir_name 的 model.ckpt / last.ckpt，兼容仓库路径前缀变化。
    """
    normalized_name = checkpoint_name.strip()

    if normalized_name in CHECKPOINT_FILENAME_MAP:
        dir_name = CHECKPOINT_FILENAME_MAP[normalized_name]
    else:
        dir_name = normalized_name.replace("@model.ckpt", "")

    search_roots = get_yourmt3_search_roots()
    for cache_root in search_roots:
        if not cache_root.exists():
            continue
        for filename in ("model.ckpt", "last.ckpt"):
            for path in cache_root.rglob(filename):
                if dir_name in str(path):
                    logger.debug(f"找到模型 checkpoint: {path}")
                    return path

    logger.debug(f"未找到模型文件 '{checkpoint_name}' (搜索根目录: {search_roots})")
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_for_model_reference(model_name: str) -> Optional[dict]:
    identity = YOURMT3_MODEL_IDENTITIES.get(model_name)
    if identity is not None:
        return identity
    for key, info in YOURMT3_MODELS.items():
        checkpoint = str(info.get("checkpoint", ""))
        if model_name == checkpoint or model_name == CHECKPOINT_FILENAME_MAP.get(checkpoint):
            return YOURMT3_MODEL_IDENTITIES.get(key)
    return None


def _path_matches_model_identity(path: Path, identity: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(identity["size"])
        and _sha256_file(path) == identity["sha256"]
    )


def get_model_path(model_name: str = DEFAULT_MODEL) -> Optional[Path]:
    """
    获取模型文件路径（智能解析）

    支持多种输入格式:
    1. 短名称: "yptf_moe_multi_nops" -> 查询 YOURMT3_MODELS 获取 checkpoint
    2. Checkpoint 名称: "YPTF.MoE+Multi (noPS)" -> 映射到实际目录
    3. 完整目录名: "mc13_256_g4_all_v7..." -> 直接查找

    参数:
        model_name: 模型名称（多种格式）

    返回:
        模型文件路径，如果不存在返回 None
    """
    identity = _identity_for_model_reference(model_name)

    # 方式1: 尝试作为短名称查询
    if model_name in YOURMT3_MODELS:
        checkpoint = YOURMT3_MODELS[model_name]["checkpoint"]
        logger.debug(f"查询模型 '{model_name}' -> checkpoint '{checkpoint}'")

        # 使用新的路径解析函数
        model_path = resolve_model_checkpoint_path(checkpoint)
        if model_path:
            if identity is None or _path_matches_model_identity(model_path, identity):
                return model_path
            logger.error("YourMT3 checkpoint identity mismatch: %s", model_path)

        # 如果新路径解析失败，尝试旧的 cache_dir 方式（向后兼容）
        logger.debug(f"新格式路径未找到，尝试旧缓存目录")
        cache_dir = get_model_cache_dir()
        legacy_path = cache_dir / model_name / "model.ckpt"
        if legacy_path.exists():
            if identity is None or _path_matches_model_identity(legacy_path, identity):
                return legacy_path
            logger.error("Legacy YourMT3 checkpoint identity mismatch: %s", legacy_path)

    # 方式2: 尝试作为 checkpoint 名称直接解析
    model_path = resolve_model_checkpoint_path(model_name)
    if model_path:
        if identity is None or _path_matches_model_identity(model_path, identity):
            return model_path
        logger.error("YourMT3 checkpoint identity mismatch: %s", model_path)

    # 方式3: 向后兼容 - 旧的 cache_dir 结构
    cache_dir = get_model_cache_dir()
    model_dir = cache_dir / model_name

    if model_dir.exists():
        # 检查模型文件是否完整
        checkpoint_file = model_dir / "model.ckpt"
        config_file = model_dir / "config.json"

        if checkpoint_file.exists():
            if identity is None or _path_matches_model_identity(checkpoint_file, identity):
                return checkpoint_file
            logger.error("Cached YourMT3 checkpoint identity mismatch: %s", checkpoint_file)
        elif identity is None and (model_dir / "pytorch_model.bin").exists():
            return model_dir

    return None


def is_model_available(model_name: str = DEFAULT_MODEL) -> bool:
    """检查模型是否已下载"""
    return get_model_path(model_name) is not None


def download_model(
    model_name: str = DEFAULT_MODEL,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """
    下载 YourMT3+ 模型 (支持镜像站点和断点续传)

    参数:
        model_name: 模型名称
        progress_callback: 进度回调

    返回:
        已通过官方 revision、文件大小和 SHA256 校验的 checkpoint 路径
    """
    if model_name not in YOURMT3_MODELS:
        raise ValueError(f"未知模型: {model_name}")

    model_info = YOURMT3_MODELS[model_name]
    identity = YOURMT3_MODEL_IDENTITIES.get(model_name)
    if identity is None:
        raise RuntimeError(f"YourMT3 模型缺少官方文件身份配置: {model_name}")

    if progress_callback:
        progress_callback(0.0, f"准备下载 {model_info['name']}...")

    logger.info(f"开始下载 YourMT3+ 模型: {model_name}")

    # 修复 SSL 证书问题（企业网络/校园网/代理环境）
    _fix_ssl_if_needed()

    # 检查是否已下载
    existing_path = get_model_path(model_name)
    if existing_path:
        logger.info(f"模型已存在: {existing_path}")
        if progress_callback:
            progress_callback(1.0, "模型已就绪")
        return existing_path

    cache_dir = get_model_cache_dir()
    model_dir = cache_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    last_error = None

    # 尝试使用不同的镜像站点下载
    for mirror_idx, endpoint in enumerate(HF_MIRRORS):
        mirror_msg = "官方站点" if endpoint is None else endpoint
        try:
            # 尝试使用 huggingface_hub 下载
            try:
                from huggingface_hub import hf_hub_download
                import shutil

                if progress_callback:
                    progress_callback(0.1, f"尝试从 {mirror_msg} 下载...")

                logger.info(f"尝试镜像站点 [{mirror_idx+1}/{len(HF_MIRRORS)}]: {mirror_msg}")

                checkpoint_filename = str(model_info["checkpoint"])
                actual_filename = str(identity["filename"])
                logger.info(
                    "下载固定 YourMT3 checkpoint: %s @ %s",
                    actual_filename,
                    YOURMT3_REVISION,
                )

                # 设置环境变量以使用镜像
                old_endpoint = os.environ.get("HF_ENDPOINT")
                if endpoint:
                    os.environ["HF_ENDPOINT"] = endpoint

                try:
                    # repo_type="space" 是必须的，因为模型存放在 Hugging Face Space 中。
                    checkpoint_path = Path(
                        hf_hub_download(
                            repo_id=YOURMT3_REPO_ID,
                            repo_type="space",
                            revision=YOURMT3_REVISION,
                            filename=actual_filename,
                            cache_dir=str(cache_dir / "hf_cache"),
                            resume_download=True,
                            local_files_only=False,
                        )
                    )
                    if not _path_matches_model_identity(checkpoint_path, identity):
                        raise RuntimeError(
                            "下载的 YourMT3 checkpoint 身份校验失败: "
                            f"{checkpoint_path}; expected size={identity['size']}, "
                            f"sha256={identity['sha256']}"
                        )

                    # 复制到 cache 目录，保留仓库原始相对路径结构
                    target_path = get_yourmt3_download_root() / actual_filename
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(checkpoint_path, target_path)
                    if not _path_matches_model_identity(target_path, identity):
                        raise RuntimeError(f"YourMT3 checkpoint 复制后校验失败: {target_path}")

                    # 创建配置文件
                    config_path = model_dir / "config.json"
                    import json

                    config = {
                        "model_name": model_name,
                        "repo_id": YOURMT3_REPO_ID,
                        "revision": YOURMT3_REVISION,
                        "checkpoint": checkpoint_filename,
                        "filename": actual_filename,
                        "size": identity["size"],
                        "sha256": identity["sha256"],
                        "version": "YourMT3+ (MLSP2024)",
                        "paper": "arXiv:2407.04822",
                    }
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)

                    if progress_callback:
                        progress_callback(1.0, "下载完成")

                    logger.info(f"模型下载完成并通过身份校验: {target_path}")
                    return target_path

                finally:
                    # 恢复原来的环境变量
                    if old_endpoint is not None:
                        os.environ["HF_ENDPOINT"] = old_endpoint
                    elif "HF_ENDPOINT" in os.environ:
                        del os.environ["HF_ENDPOINT"]

            except ImportError:
                logger.warning("huggingface_hub 未安装")
                raise ImportError(
                    "无法下载模型。请安装 huggingface_hub: pip install huggingface_hub"
                )

        except Exception as e:
            last_error = e
            logger.warning(f"从 {mirror_msg if endpoint else '官方站点'} 下载失败: {e}")
            if progress_callback:
                progress_callback(0.0, f"重试中... ({mirror_idx+1}/{len(HF_MIRRORS)})")
            continue

    # 所有镜像都失败了
    logger.error(f"所有镜像站点下载均失败: {last_error}")

    # 提供详细的错误信息
    error_msg = f"""模型下载失败: {last_error}

请尝试以下解决方案:

1. 检查网络连接
2. 手动下载模型:
   - 访问: https://huggingface.co/{YOURMT3_REPO_ID}
   - 下载文件: {YOURMT3_MODELS['mc13_256_all_cross_v6']['checkpoint']}
   - 重命名为: model.ckpt
   - 放置到: {model_dir}/

3. 使用国内镜像:
   - 设置环境变量: export HF_ENDPOINT=https://hf-mirror.com
   - 重新运行下载

4. 确保安装了最新版本:
   pip install --upgrade huggingface_hub
"""
    raise RuntimeError(error_msg)


def ensure_model_available(
    model_name: str = DEFAULT_MODEL,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """
    确保模型可用，如果不存在则下载

    参数:
        model_name: 模型名称
        progress_callback: 进度回调

    返回:
        模型目录路径
    """
    model_path = get_model_path(model_name)
    if model_path:
        if progress_callback:
            progress_callback(1.0, "模型已就绪")
        return model_path

    return download_model(model_name, progress_callback)


def list_available_models() -> dict:
    """列出所有可用模型及其状态"""
    result = {}
    for name, info in YOURMT3_MODELS.items():
        result[name] = {
            **info,
            "downloaded": is_model_available(name),
            "path": str(get_model_path(name)) if is_model_available(name) else None,
        }
    return result
