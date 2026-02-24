"""
GPU检测和管理工具 - 支持 CUDA (NVIDIA)、ROCm (AMD)、MPS (Apple)、Intel XPU 和 CPU
"""
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _get_torch():
    """获取 torch 模块，失败返回 None"""
    try:
        import torch
        return torch
    except ImportError:
        return None


def get_accelerator_type() -> str:
    """
    检测可用的加速器类型。

    返回:
        'cuda'  - NVIDIA GPU (CUDA)
        'rocm'  - AMD GPU (ROCm，通过 torch.cuda 接口)
        'mps'   - Apple Silicon GPU (Metal Performance Shaders)
        'xpu'   - Intel GPU (通过 Intel Extension for PyTorch)
        'cpu'   - 无GPU加速，回退到CPU
    """
    torch = _get_torch()
    if torch is None:
        return "cpu"

    # NVIDIA CUDA 或 AMD ROCm（ROCm 使用 torch.cuda 接口）
    if torch.cuda.is_available():
        # 区分 ROCm 与 CUDA
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return "rocm"
        return "cuda"

    # Apple Silicon MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"

    # Intel GPU (需要 intel_extension_for_pytorch)
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu"
    except ImportError:
        pass

    return "cpu"


def is_gpu_available() -> bool:
    """检查是否有任何类型的GPU加速可用（CUDA、ROCm、MPS 或 Intel XPU）"""
    return get_accelerator_type() != "cpu"


def is_cuda_available() -> bool:
    """检查 NVIDIA CUDA 是否可用（保持向后兼容）"""
    return get_accelerator_type() == "cuda"


def get_device(prefer_gpu: bool = True, gpu_index: int = 0) -> str:
    """
    获取最佳可用计算设备。

    参数:
        prefer_gpu: 是否优先使用GPU
        gpu_index:  多GPU环境下使用哪个GPU

    返回:
        设备字符串，如 'cuda:0'、'rocm:0'（实际仍为 cuda:0）、'mps'、'xpu:0' 或 'cpu'
    """
    if not prefer_gpu:
        logger.info("使用 CPU（由配置指定）")
        return "cpu"

    accel = get_accelerator_type()

    if accel in ("cuda", "rocm"):
        torch = _get_torch()
        count = torch.cuda.device_count() if torch else 0
        idx = min(gpu_index, count - 1) if count > 0 else 0
        device = f"cuda:{idx}"
        try:
            name = torch.cuda.get_device_name(idx)
            accel_label = "ROCm/AMD" if accel == "rocm" else "CUDA/NVIDIA"
            logger.info(f"使用 {accel_label} GPU: {name} ({device})")
        except Exception:
            pass
        return device

    if accel == "mps":
        logger.info("使用 Apple MPS GPU")
        return "mps"

    if accel == "xpu":
        torch = _get_torch()
        try:
            count = torch.xpu.device_count() if torch and hasattr(torch, 'xpu') else 0
            idx = min(gpu_index, count - 1) if count > 0 else 0
            device = f"xpu:{idx}"
            name = torch.xpu.get_device_name(idx) if hasattr(torch.xpu, 'get_device_name') else "Intel GPU"
            logger.info(f"使用 Intel XPU: {name} ({device})")
            return device
        except Exception:
            logger.info("使用 Intel XPU: xpu:0")
            return "xpu:0"

    logger.info("未检测到GPU，使用 CPU")
    return "cpu"


def get_gpu_count() -> int:
    """获取可用GPU数量（CUDA/ROCm/XPU）"""
    torch = _get_torch()
    if torch is None:
        return 0
    accel = get_accelerator_type()
    if accel in ("cuda", "rocm"):
        return torch.cuda.device_count()
    if accel == "mps":
        return 1
    if accel == "xpu":
        try:
            return torch.xpu.device_count() if hasattr(torch, 'xpu') else 0
        except Exception:
            return 1
    return 0


def get_gpu_info() -> List[dict]:
    """
    获取所有可用GPU信息。

    返回:
        包含GPU信息字典的列表，每项含 name、total_memory_gb 等字段
    """
    torch = _get_torch()
    gpus = []

    if torch is None:
        return gpus

    accel = get_accelerator_type()

    if accel in ("cuda", "rocm"):
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "type": accel.upper(),
                    "total_memory": props.total_memory,
                    "total_memory_gb": props.total_memory / (1024 ** 3),
                    "major": props.major,
                    "minor": props.minor,
                })
            except Exception as e:
                logger.warning(f"获取 GPU {i} 信息失败: {e}")

    elif accel == "mps":
        gpus.append({
            "index": 0,
            "name": "Apple Silicon GPU (MPS)",
            "type": "MPS",
            "total_memory": 0,
            "total_memory_gb": 0.0,
        })

    elif accel == "xpu":
        try:
            count = torch.xpu.device_count() if hasattr(torch, 'xpu') else 1
            for i in range(count):
                name = "Intel GPU"
                try:
                    if hasattr(torch.xpu, 'get_device_name'):
                        name = torch.xpu.get_device_name(i)
                except Exception:
                    pass
                gpus.append({
                    "index": i,
                    "name": name,
                    "type": "XPU",
                    "total_memory": 0,
                    "total_memory_gb": 0.0,
                })
        except Exception as e:
            logger.warning(f"获取 Intel XPU 信息失败: {e}")

    return gpus


def get_memory_info(device: str = None) -> Optional[Tuple[float, float]]:
    """
    获取GPU内存使用情况。

    参数:
        device: 设备字符串（默认自动检测）

    返回:
        (已用GB, 总共GB) 或 None
    """
    torch = _get_torch()
    if torch is None:
        return None

    if device is None:
        device = get_device()

    try:
        if device.startswith("cuda") and torch.cuda.is_available():
            device_idx = int(device.split(":")[1]) if ":" in device else 0
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
            return (allocated, total)
        if device.startswith("xpu") and hasattr(torch, 'xpu') and torch.xpu.is_available():
            # Intel XPU 暂无标准内存查询 API，返回 None
            return None
    except Exception as e:
        logger.debug(f"获取显存信息失败: {e}")

    return None


def clear_gpu_memory() -> None:
    """清除GPU显存缓存（CUDA/ROCm）"""
    torch = _get_torch()
    if torch is None:
        return

    accel = get_accelerator_type()
    try:
        if accel in ("cuda", "rocm") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU显存缓存已清除")
        elif accel == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            logger.info("MPS显存缓存已清除")
        elif accel == "xpu" and hasattr(torch, 'xpu') and torch.xpu.is_available():
            if hasattr(torch.xpu, 'empty_cache'):
                torch.xpu.empty_cache()
                logger.info("Intel XPU显存缓存已清除")
    except Exception as e:
        logger.warning(f"清除GPU显存失败: {e}")


def normalize_device_for_whisperx(device: str) -> str:
    """
    将设备字符串规范化为 WhisperX 兼容格式。
    WhisperX 不支持 'cuda:0' 形式，需简化为 'cuda'。
    """
    if device.startswith("cuda:"):
        return "cuda"
    return device


def get_accelerator_label() -> str:
    """获取加速器的友好标签字符串，用于UI显示"""
    accel = get_accelerator_type()
    if accel == "cpu":
        return "CPU"
    gpus = get_gpu_info()
    if gpus:
        name = gpus[0]["name"]
        accel_type = gpus[0]["type"]
        return f"{accel_type}: {name}"
    labels = {"cuda": "CUDA GPU", "rocm": "ROCm GPU", "mps": "Apple MPS", "xpu": "Intel XPU"}
    return labels.get(accel, "GPU")


def diagnose_gpu() -> dict:
    """诊断GPU可用性，返回详细信息字典"""
    torch = _get_torch()
    accel = get_accelerator_type()

    result = {
        "torch_version": torch.__version__ if torch else None,
        "accelerator": accel,
        "cuda_available": accel in ("cuda", "rocm"),
        "cuda_version": None,
        "rocm_version": None,
        "cudnn_available": False,
        "cudnn_version": None,
        "mps_available": accel == "mps",
        "xpu_available": accel == "xpu",
        "gpu_count": get_gpu_count(),
        "gpu_names": [g["name"] for g in get_gpu_info()],
        "tensorflow_gpu": False,
        "tf_gpu_count": 0,
        "tf_version": None,
        "tf_gpu_names": [],
        "tf_is_cpu_only": None,
    }

    if torch and accel in ("cuda", "rocm"):
        result["cuda_version"] = getattr(torch.version, 'cuda', None)
        result["rocm_version"] = getattr(torch.version, 'hip', None)
        result["cudnn_available"] = torch.backends.cudnn.is_available()
        if result["cudnn_available"]:
            result["cudnn_version"] = str(torch.backends.cudnn.version())

    # TensorFlow 诊断
    try:
        import tensorflow as tf
        result["tf_version"] = tf.__version__
        gpus = tf.config.list_physical_devices('GPU')
        result["tensorflow_gpu"] = len(gpus) > 0
        result["tf_gpu_count"] = len(gpus)
        result["tf_gpu_names"] = [gpu.name for gpu in gpus]
        result["tf_is_cpu_only"] = "cpu" in tf.__file__.lower()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"TensorFlow 诊断失败: {e}")

    return result


def configure_tensorflow_gpu() -> bool:
    """配置 TensorFlow GPU 内存增长模式"""
    try:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"TensorFlow GPU 已配置: {len(gpus)} 个设备")
            return True
        return False
    except ImportError:
        logger.debug("TensorFlow 未安装")
        return False
    except Exception as e:
        logger.warning(f"TensorFlow GPU 配置异常: {e}")
        return False


def print_gpu_diagnosis() -> None:
    """打印详细的 GPU 诊断信息"""
    info = diagnose_gpu()

    print("\n" + "=" * 50)
    print("GPU 诊断信息")
    print("=" * 50)

    print(f"\n[PyTorch]")
    print(f"  版本: {info['torch_version'] or '未安装'}")
    accel = info.get("accelerator", "cpu")
    print(f"  加速器类型: {accel.upper()}")

    if info['cuda_available']:
        if info.get('rocm_version'):
            print(f"  ROCm 版本: {info['rocm_version']}")
        else:
            print(f"  CUDA 版本: {info['cuda_version']}")
        print(f"  cuDNN 可用: {info['cudnn_available']}")
        if info['cudnn_available']:
            print(f"  cuDNN 版本: {info['cudnn_version']}")
        print(f"  GPU 数量: {info['gpu_count']}")
        for i, name in enumerate(info['gpu_names']):
            print(f"    GPU {i}: {name}")

    if info['mps_available']:
        print(f"  Apple MPS 可用: True")

    if info.get('xpu_available'):
        print(f"  Intel XPU 可用: True")

    if accel == "cpu":
        print(f"  [!] 未检测到GPU，将使用CPU运行（速度较慢）")

    print(f"\n[TensorFlow]")
    print(f"  版本: {info['tf_version'] or '未安装'}")
    print(f"  GPU 可用: {info['tensorflow_gpu']}")
    if info['tf_gpu_names']:
        for name in info['tf_gpu_names']:
            print(f"    {name}")

    print("\n" + "=" * 50)
