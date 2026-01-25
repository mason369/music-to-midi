"""
GPU检测和管理工具
"""
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """检查CUDA是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_count() -> int:
    """获取可用GPU数量"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 0


def get_gpu_info() -> List[dict]:
    """
    获取可用GPU信息

    返回:
        包含GPU信息（名称、内存等）的字典列表
    """
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "major": props.major,
                    "minor": props.minor,
                })
    except ImportError:
        logger.warning("未安装PyTorch，无法检测GPU")
    except Exception as e:
        logger.error(f"检测GPU时出错: {e}")

    return gpus


def get_device(prefer_gpu: bool = True, gpu_index: int = 0) -> str:
    """
    获取最佳可用设备

    参数:
        prefer_gpu: 是否优先使用GPU
        gpu_index: 多GPU时使用哪个GPU

    返回:
        设备字符串（'cuda:0', 'mps', 或 'cpu'）
    """
    try:
        import torch

        if prefer_gpu:
            if torch.cuda.is_available():
                device = f"cuda:{gpu_index}"
                logger.info(f"使用GPU: {torch.cuda.get_device_name(gpu_index)}")
                return device
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("使用Apple MPS")
                return "mps"

        logger.info("使用CPU")
        return "cpu"
    except ImportError:
        return "cpu"


def get_memory_info(device: str = "cuda:0") -> Optional[Tuple[float, float]]:
    """
    获取GPU内存使用情况

    参数:
        device: 设备字符串

    返回:
        (已用GB, 总共GB) 元组，如不可用则返回None
    """
    try:
        import torch
        if device.startswith("cuda") and torch.cuda.is_available():
            device_idx = int(device.split(":")[1]) if ":" in device else 0
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
            return (allocated, total)
    except Exception as e:
        logger.error(f"获取内存信息时出错: {e}")

    return None


def clear_gpu_memory() -> None:
    """清除GPU内存缓存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU内存缓存已清除")
    except Exception as e:
        logger.error(f"清除GPU内存时出错: {e}")


def diagnose_gpu() -> dict:
    """
    诊断GPU可用性问题

    返回包含以下信息的字典:
    - torch_version: PyTorch 版本
    - cuda_available: CUDA 是否可用
    - cuda_version: CUDA 版本
    - cudnn_available: cuDNN 是否可用
    - gpu_count: GPU 数量
    - gpu_names: GPU 名称列表
    - tensorflow_gpu: TensorFlow GPU 是否可用
    - tf_gpu_count: TensorFlow 检测到的 GPU 数量
    """
    result = {
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_available": False,
        "cudnn_version": None,
        "gpu_count": 0,
        "gpu_names": [],
        "tensorflow_gpu": False,
        "tf_gpu_count": 0,
        "tf_version": None,
        "tf_gpu_names": [],
        "tf_is_cpu_only": None,
        "mps_available": False,
    }

    # PyTorch 诊断
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()

        if result["cuda_available"]:
            result["cuda_version"] = torch.version.cuda
            result["cudnn_available"] = torch.backends.cudnn.is_available()
            if result["cudnn_available"]:
                result["cudnn_version"] = str(torch.backends.cudnn.version())
            result["gpu_count"] = torch.cuda.device_count()

            for i in range(result["gpu_count"]):
                result["gpu_names"].append(torch.cuda.get_device_name(i))

        # macOS MPS 支持
        if hasattr(torch.backends, 'mps'):
            result["mps_available"] = torch.backends.mps.is_available()

    except ImportError:
        logger.warning("PyTorch 未安装")
    except Exception as e:
        logger.warning(f"PyTorch 诊断失败: {e}")

    # TensorFlow 诊断（Basic Pitch 使用 TensorFlow）
    try:
        import tensorflow as tf
        result["tf_version"] = tf.__version__
        gpus = tf.config.list_physical_devices('GPU')
        result["tensorflow_gpu"] = len(gpus) > 0
        result["tf_gpu_count"] = len(gpus)
        result["tf_gpu_names"] = []

        if gpus:
            for gpu in gpus:
                result["tf_gpu_names"].append(gpu.name)

        # 检查是否是CPU版本
        result["tf_is_cpu_only"] = "cpu" in tf.__file__.lower() or "intel" in tf.__file__.lower()

    except ImportError:
        logger.debug("TensorFlow 未安装")
        result["tf_version"] = None
        result["tf_is_cpu_only"] = None
    except Exception as e:
        logger.warning(f"TensorFlow 诊断失败: {e}")

    return result


def configure_tensorflow_gpu() -> bool:
    """
    配置 TensorFlow GPU

    启用 GPU 内存增长以避免占用所有 GPU 内存。
    Basic Pitch 使用 TensorFlow 后端，需要显式配置。

    返回:
        True 如果成功配置 GPU，False 如果使用 CPU
    """
    try:
        import os
        # 减少 TensorFlow 日志输出
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # 为每个 GPU 启用内存增长
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"TensorFlow GPU 已配置: {len(gpus)} 个设备")
                return True
            except RuntimeError as e:
                # 如果 GPU 已经被初始化，无法修改配置
                logger.warning(f"TensorFlow GPU 配置失败: {e}")
                return False
        else:
            logger.info("TensorFlow 未检测到 GPU，将使用 CPU")
            return False

    except ImportError:
        logger.debug("TensorFlow 未安装")
        return False
    except Exception as e:
        logger.warning(f"TensorFlow GPU 配置异常: {e}")
        return False


def normalize_device_for_whisperx(device: str) -> str:
    """
    规范化设备字符串用于 WhisperX 对齐模型

    WhisperX 的对齐模型不支持 'cuda:0' 格式的设备字符串，
    需要简化为 'cuda'。

    参数:
        device: 原始设备字符串 (如 'cuda:0', 'cuda:1', 'cpu')

    返回:
        规范化后的设备字符串 (如 'cuda', 'cpu')
    """
    if device.startswith("cuda:"):
        return "cuda"
    return device


def print_gpu_diagnosis() -> None:
    """打印详细的 GPU 诊断信息"""
    info = diagnose_gpu()

    print("\n" + "=" * 50)
    print("GPU 诊断信息")
    print("=" * 50)

    print(f"\n[PyTorch]")
    print(f"  版本: {info['torch_version'] or '未安装'}")
    print(f"  CUDA 可用: {info['cuda_available']}")

    if info['cuda_available']:
        print(f"  CUDA 版本: {info['cuda_version']}")
        print(f"  cuDNN 可用: {info['cudnn_available']}")
        if info['cudnn_available']:
            print(f"  cuDNN 版本: {info['cudnn_version']}")
        print(f"  GPU 数量: {info['gpu_count']}")
        for i, name in enumerate(info['gpu_names']):
            print(f"    GPU {i}: {name}")

    if info['mps_available']:
        print(f"  Apple MPS 可用: True")

    print(f"\n[TensorFlow]")
    print(f"  版本: {info['tf_version'] or '未安装'}")
    if info['tf_is_cpu_only'] is not None:
        print(f"  CPU版本: {info['tf_is_cpu_only']}")
    print(f"  GPU 可用: {info['tensorflow_gpu']}")
    print(f"  GPU 数量: {info['tf_gpu_count']}")
    if info['tf_gpu_names']:
        for name in info['tf_gpu_names']:
            print(f"    {name}")

    if info['tf_is_cpu_only']:
        print(f"\n  警告: TensorFlow 使用CPU版本，建议安装 tensorflow[and-cuda]")

    print("\n" + "=" * 50)
