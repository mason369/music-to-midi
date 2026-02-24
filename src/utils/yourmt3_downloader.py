"""
YourMT3+ 模型下载工具

处理模型的下载、缓存和验证
支持 Hugging Face 镜像站点和断点续传
"""
import logging
import os
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Hugging Face 镜像站点列表（用于国内网络环境）
HF_MIRRORS = [
    None,  # 官方站点
    "https://hf-mirror.com",  # 国内镜像
    "https://huggingface.co",  # 备用
]

# 模型配置 - YourMT3+ (2024年7月发布，MLSP2024论文)
YOURMT3_MODELS = {
    # YPTF.MoE+Multi 系列 - 最强性能 (Mixture of Experts)
    "yptf_moe_multi_ps": {
        "name": "YPTF.MoE+Multi (PS) - 顶级性能",
        "description": "混合专家模型，支持音高偏移增强，2025 AMT Challenge 顶级性能",
        "checkpoint": "YPTF.MoE+Multi (PS)",  # 带音高偏移，鲁棒性更好
        "size_mb": 2500,
        "recommended": True,
        "features": ["MoE架构", "Perceiver编码器", "多通道解码器", "音高偏移增强"],
    },
    "yptf_moe_multi_nops": {
        "name": "YPTF.MoE+Multi (noPS)",
        "description": "混合专家模型，不含音高偏移增强",
        "checkpoint": "YPTF.MoE+Multi (noPS)",
        "size_mb": 2500,
        "features": ["MoE架构", "Perceiver编码器", "多通道解码器"],
    },

    # YPTF+Multi 系列 - 标准高性能
    "yptf_multi_ps": {
        "name": "YPTF+Multi (PS)",
        "description": "标准高性能模型，支持音高偏移增强",
        "checkpoint": "YPTF+Multi (PS)",
        "size_mb": 2000,
        "features": ["Perceiver编码器", "多通道解码器", "音高偏移增强"],
    },
    "yptf_multi_nops": {
        "name": "YPTF+Multi (noPS)",
        "description": "标准高性能模型，不含音高偏移增强",
        "checkpoint": "YPTF+Multi (noPS)",
        "size_mb": 2000,
        "features": ["Perceiver编码器", "多通道解码器"],
    },

    # 传统checkpoint名称格式（兼容旧版）
    "mc13_256_all_cross_v6": {
        "name": "YourMT3+ 全乐器模型 (传统版)",
        "description": "支持多种乐器的通用转写模型",
        "checkpoint": "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt",
        "size_mb": 2000,
        "default": True,  # 保持向后兼容
        "features": ["通用多乐器", "跨数据集增强"],
    },
}

# 统一的 Hugging Face 仓库
YOURMT3_REPO_ID = "mimbres/YourMT3"

# Checkpoint 名称到实际目录名的映射表
# 用于将 Hugging Face 上的模型名称映射到本地文件系统结构
CHECKPOINT_FILENAME_MAP = {
    "YPTF.MoE+Multi (PS)": "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2",
    "YPTF.MoE+Multi (noPS)": "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops",
    "YPTF+Multi (PS)": "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100",
    "YPTF+Multi (noPS)": "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72",
    "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt": "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k",
}

# 默认模型 - 使用性能最强的 MoE 版本
DEFAULT_MODEL = "yptf_moe_multi_ps"


def get_model_cache_dir() -> Path:
    """获取模型缓存目录"""
    # 使用用户目录下的隐藏文件夹
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

    cache_root = Path.home() / ".cache/music_ai_models/yourmt3_all"
    if not cache_root.exists():
        return None

    # 递归搜索：找到路径中包含 dir_name 的 checkpoint 文件
    for filename in ("model.ckpt", "last.ckpt"):
        for path in cache_root.rglob(filename):
            if dir_name in str(path):
                logger.debug(f"找到模型 checkpoint: {path}")
                return path

    logger.debug(f"未找到模型文件 '{checkpoint_name}' (搜索根目录: {cache_root})")
    return None


def get_model_path(model_name: str = DEFAULT_MODEL) -> Optional[Path]:
    """
    获取模型文件路径（智能解析）

    支持多种输入格式:
    1. 短名称: "yptf_moe_multi_ps" -> 查询 YOURMT3_MODELS 获取 checkpoint
    2. Checkpoint 名称: "YPTF.MoE+Multi (PS)" -> 映射到实际目录
    3. 完整目录名: "mc13_256_g4_all_v7..." -> 直接查找

    参数:
        model_name: 模型名称（多种格式）

    返回:
        模型文件路径，如果不存在返回 None
    """
    # 方式1: 尝试作为短名称查询
    if model_name in YOURMT3_MODELS:
        checkpoint = YOURMT3_MODELS[model_name]["checkpoint"]
        logger.debug(f"查询模型 '{model_name}' -> checkpoint '{checkpoint}'")

        # 使用新的路径解析函数
        model_path = resolve_model_checkpoint_path(checkpoint)
        if model_path:
            return model_path

        # 如果新路径解析失败，尝试旧的 cache_dir 方式（向后兼容）
        logger.debug(f"新格式路径未找到，尝试旧缓存目录")
        cache_dir = get_model_cache_dir()
        legacy_path = cache_dir / model_name / "model.ckpt"
        if legacy_path.exists():
            return cache_dir / model_name

    # 方式2: 尝试作为 checkpoint 名称直接解析
    model_path = resolve_model_checkpoint_path(model_name)
    if model_path:
        return model_path

    # 方式3: 向后兼容 - 旧的 cache_dir 结构
    cache_dir = get_model_cache_dir()
    model_dir = cache_dir / model_name

    if model_dir.exists():
        # 检查模型文件是否完整
        checkpoint_file = model_dir / "model.ckpt"
        config_file = model_dir / "config.json"

        if checkpoint_file.exists() or (model_dir / "pytorch_model.bin").exists():
            return model_dir

    return None


def is_model_available(model_name: str = DEFAULT_MODEL) -> bool:
    """检查模型是否已下载"""
    return get_model_path(model_name) is not None


def download_model(
    model_name: str = DEFAULT_MODEL,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Path:
    """
    下载 YourMT3+ 模型 (支持镜像站点和断点续传)

    参数:
        model_name: 模型名称
        progress_callback: 进度回调

    返回:
        模型目录路径
    """
    if model_name not in YOURMT3_MODELS:
        raise ValueError(f"未知模型: {model_name}")

    model_info = YOURMT3_MODELS[model_name]

    if progress_callback:
        progress_callback(0.0, f"准备下载 {model_info['name']}...")

    logger.info(f"开始下载 YourMT3+ 模型: {model_name}")

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
        try:
            # 尝试使用 huggingface_hub 下载
            try:
                from huggingface_hub import hf_hub_download
                import shutil

                mirror_msg = "官方站点" if endpoint is None else endpoint
                if progress_callback:
                    progress_callback(0.1, f"尝试从 {mirror_msg} 下载...")

                logger.info(f"尝试镜像站点 [{mirror_idx+1}/{len(HF_MIRRORS)}]: {mirror_msg}")

                # 获取checkpoint文件名
                checkpoint_filename = model_info.get("checkpoint")
                if not checkpoint_filename:
                    # 如果没有指定checkpoint，使用默认的MoE模型
                    checkpoint_filename = YOURMT3_MODELS["yptf_moe_multi_ps"]["checkpoint"]
                    logger.warning(f"模型 {model_name} 没有专用checkpoint，使用默认MoE模型")

                logger.info(f"下载 checkpoint: {checkpoint_filename}")

                # 将 checkpoint 名称（如 "YPTF.MoE+Multi (PS)"）映射到实际目录名
                # 仓库结构：amt/logs/2024/{dir_name}/checkpoints/model.ckpt
                if checkpoint_filename in CHECKPOINT_FILENAME_MAP:
                    dir_name = CHECKPOINT_FILENAME_MAP[checkpoint_filename]
                elif "@model.ckpt" in checkpoint_filename:
                    dir_name = checkpoint_filename.replace("@model.ckpt", "")
                else:
                    dir_name = checkpoint_filename

                # 动态查询仓库中的实际 checkpoint 路径，避免硬编码导致路径失效
                from huggingface_hub import list_repo_files as _list_files
                repo_files = list(_list_files(YOURMT3_REPO_ID, repo_type="space"))
                actual_filename = next(
                    (f for f in repo_files if dir_name in f and f.endswith("model.ckpt")),
                    None
                )
                alt_filename = next(
                    (f for f in repo_files if dir_name in f and f.endswith("last.ckpt")),
                    None
                )
                if not actual_filename and not alt_filename:
                    raise FileNotFoundError(
                        f"在仓库 {YOURMT3_REPO_ID} 中未找到包含 '{dir_name}' 的 checkpoint"
                    )
                actual_filename = actual_filename or alt_filename
                logger.info(f"动态解析仓库路径: {actual_filename}")

                # 设置环境变量以使用镜像
                old_endpoint = os.environ.get("HF_ENDPOINT")
                if endpoint:
                    os.environ["HF_ENDPOINT"] = endpoint

                try:
                    # 下载模型checkpoint (支持断点续传)
                    # repo_type="space" 是必须的，因为模型存放在 HuggingFace Spaces 中
                    try:
                        checkpoint_path = hf_hub_download(
                            repo_id=YOURMT3_REPO_ID,
                            repo_type="space",
                            filename=actual_filename,
                            cache_dir=str(cache_dir / "hf_cache"),
                            resume_download=True,
                            local_files_only=False,
                        )
                    except Exception as e:
                        if alt_filename:
                            logger.warning(f"model.ckpt 下载失败: {e}，尝试 last.ckpt")
                            try:
                                checkpoint_path = hf_hub_download(
                                    repo_id=YOURMT3_REPO_ID,
                                    repo_type="space",
                                    filename=alt_filename,
                                    cache_dir=str(cache_dir / "hf_cache"),
                                    resume_download=True,
                                    local_files_only=False,
                                )
                                logger.info(f"成功使用 last.ckpt")
                            except Exception:
                                raise e
                        else:
                            raise e

                    # 复制到 cache 目录，保留仓库原始相对路径结构
                    target_dir = (
                        Path.home() / ".cache/music_ai_models/yourmt3_all" /
                        Path(actual_filename).parent
                    )
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / "model.ckpt"
                    shutil.copy2(checkpoint_path, target_path)

                    # 创建配置文件
                    config_path = model_dir / "config.json"
                    import json
                    config = {
                        "model_name": model_name,
                        "repo_id": YOURMT3_REPO_ID,
                        "checkpoint": checkpoint_filename,
                        "version": "YourMT3+ (MLSP2024)",
                        "paper": "arXiv:2407.04822",
                    }
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)

                    if progress_callback:
                        progress_callback(1.0, "下载完成")

                    logger.info(f"模型下载完成: {model_dir}")
                    return model_dir

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

    # 清理不完整的下载
    if model_dir.exists():
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)

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
    progress_callback: Optional[Callable[[float, str], None]] = None
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
