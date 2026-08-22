"""
GPU检测和管理工具 - 支持 CUDA (NVIDIA)、ROCm (AMD)、MPS (Apple)、Intel XPU 和 CPU
"""
import gc
import logging
import os
import platform
import sys
import threading
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_UNSUPPORTED_CUDA_ARCH_MARKERS = (
    "no kernel image is available for execution on the device",
    "device kernel image is invalid",
)


def _get_short_path(long_path: str) -> str:
    """
    获取 Windows 8.3 短路径名，消除空格、括号、非 ASCII 等特殊字符。
    例如 'D:\\music-to-midi-master (1)\\...' → 'D:\\MUSIC-~1\\...'
    """
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(512)
        ret = ctypes.windll.kernel32.GetShortPathNameW(long_path, buf, 512)
        if ret > 0 and ret < 512:
            return buf.value
    except Exception:
        pass
    return long_path


def _fix_torch_dll_path():
    """
    修复 Windows 特殊路径下 PyTorch DLL 加载失败的问题。

    PyTorch 的 _load_dll_libraries() 用完整长路径调用 LoadLibraryExW，
    路径中的空格、括号、中文等字符会导致 c10.dll 加载失败（WinError 1114）。

    解决方案：在 import torch 之前，用 8.3 短路径预加载所有 torch DLL，
    这样 PyTorch 再次加载时发现 DLL 已在内存中，不会重复加载。
    """
    if getattr(_fix_torch_dll_path, '_done', False):
        return
    _fix_torch_dll_path._done = True

    if platform.system() != "Windows":
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.origin is None:
            return
        torch_lib = os.path.join(os.path.dirname(spec.origin), "lib")
        if not os.path.isdir(torch_lib):
            return

        # 检查路径是否包含特殊字符（空格、括号、非 ASCII）
        import re
        if not re.search(r'[\s\(\)\[\]{}]|[^\x00-\x7F]', torch_lib):
            return  # 路径正常，无需修复

        torch_lib_short = _get_short_path(torch_lib)
        if torch_lib_short == torch_lib:
            # 短路径获取失败（可能 8.3 名称被禁用），回退到 PATH 注入
            current_path = os.environ.get("PATH", "")
            if torch_lib not in current_path:
                os.environ["PATH"] = torch_lib + os.pathsep + current_path
            try:
                os.add_dll_directory(torch_lib)
            except OSError:
                pass
            return

        # 注入短路径到 PATH 和 DLL 搜索目录
        current_path = os.environ.get("PATH", "")
        if torch_lib_short not in current_path:
            os.environ["PATH"] = torch_lib_short + os.pathsep + current_path
        try:
            os.add_dll_directory(torch_lib_short)
        except OSError:
            pass

        # 预加载 VC++ 运行时（c10.dll 的前置依赖）
        import ctypes
        for vcrt in ("vcruntime140.dll", "msvcp140.dll", "vcruntime140_1.dll"):
            try:
                ctypes.CDLL(vcrt)
            except OSError:
                pass

        # 用短路径预加载所有 torch DLL，使其驻留内存
        # PyTorch 再次加载时会发现已在内存中，跳过路径解析
        import glob
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        kernel32.LoadLibraryW.restype = ctypes.c_void_p
        for dll in sorted(glob.glob(os.path.join(torch_lib_short, "*.dll"))):
            try:
                kernel32.LoadLibraryW(dll)
            except Exception:
                pass
    except Exception as e:
        logger.debug("torch DLL 预加载失败（非关键）: %s", e)


_torch_module = None  # 缓存 torch 模块引用
_torch_checked = False  # 是否已尝试加载过 torch
_torch_lock = threading.Lock()


def get_torch_dll_troubleshooting(*, frozen: bool) -> str:
    """Return accelerator-specific guidance for a Windows torch DLL load failure."""
    accelerator = os.environ.get("MUSIC_TO_MIDI_ACCELERATOR", "cuda").strip().lower()
    if accelerator == "xpu":
        runtime_action = (
            "重新解压完整的 MusicToMidi-XPU 便携包"
            if frozen
            else "运行 install_xpu.ps1 重建独立的 venv-xpu"
        )
        return (
            "  1. 未安装 Visual C++ Redistributable 2022: "
            "https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
            "  2. Intel GPU 驱动未安装或不支持当前原生 PyTorch XPU 运行时\n"
            "  3. XPU 的 libiomp5md.dll 缺失，或错误混入 CUDA 使用的 "
            f"libomp140.x86_64.dll；{runtime_action}"
        )

    runtime_action = (
        "重新解压完整的 NVIDIA CUDA 便携包"
        if frozen
        else "重新运行 install.bat 自动修复"
    )
    return (
        "  1. 未安装 Visual C++ Redistributable 2022: "
        "https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
        "  2. 没有 NVIDIA 显卡或未安装兼容 CUDA 12.8 的驱动\n"
        f"  3. libomp140.x86_64.dll 缺失；{runtime_action}"
    )


def _get_torch():
    """获取 torch 模块，失败返回 None（结果会被缓存）"""
    global _torch_module, _torch_checked
    if _torch_checked:
        return _torch_module
    with _torch_lock:
        if _torch_checked:
            return _torch_module
        try:
            _fix_torch_dll_path()
            import torch
            _torch_module = torch
            _torch_checked = True
            return torch
        except ImportError:
            _torch_checked = True
            return None
        except OSError as e:
            _torch_checked = True
            import re
            # 判断是否是路径特殊字符导致的问题
            err_str = str(e)
            # 从错误信息中提取 DLL 路径
            path_match = re.search(r'Error loading "([^"]+)"', err_str)
            dll_path = path_match.group(1) if path_match else ""
            has_special = bool(re.search(r'[\s\(\)\[\]{}]|[^\x00-\x7F]', dll_path))

            if has_special:
                logger.warning(f"torch 加载失败（路径含特殊字符）: {e}")
                logger.warning("建议将项目移动到纯英文且无空格的路径，如 C:\\MusicToMidi")
            else:
                logger.warning(f"torch DLL 加载失败: {e}")
                is_frozen = getattr(sys, 'frozen', False)
                logger.warning(
                    "可能原因：\n%s",
                    get_torch_dll_troubleshooting(frozen=is_frozen),
                )
            return None


def _get_cuda_device_index(device: str) -> int:
    if ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def _get_device_index(device: str) -> int:
    """Return an explicit accelerator index from a canonical torch device string."""

    if ":" not in device:
        return 0
    try:
        index = int(device.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"无效的 PyTorch 设备字符串: {device!r}") from exc
    if index < 0:
        raise ValueError(f"PyTorch 设备索引不能为负数: {device!r}")
    return index


def is_unsupported_cuda_architecture_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _UNSUPPORTED_CUDA_ARCH_MARKERS)


def rewrite_cuda_runtime_error(error: BaseException, device: Optional[str] = None) -> str:
    if not is_unsupported_cuda_architecture_error(error):
        return str(error)

    torch = _get_torch()
    device_name = "未知 NVIDIA GPU"
    compute_capability = "未知"
    supported_arches = "未知"
    torch_version = None
    cuda_version = None

    if torch is not None:
        torch_version = getattr(torch, "__version__", None)
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            if device is None:
                device = "cuda:0"
            props = torch.cuda.get_device_properties(_get_cuda_device_index(device))
            device_name = getattr(props, "name", device_name)
            major = getattr(props, "major", None)
            minor = getattr(props, "minor", None)
            if major is not None and minor is not None:
                compute_capability = f"{major}.{minor} (sm_{major}{minor})"
        except Exception:
            pass
        try:
            arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
            if arch_list:
                supported_arches = ", ".join(arch_list)
        except Exception:
            pass

    details = [
        "检测到当前 NVIDIA 显卡架构与内置 PyTorch/CUDA 运行时不兼容，无法使用 GPU 执行转写。",
        f"当前显卡: {device_name}",
        f"计算能力: {compute_capability}",
    ]
    if torch_version:
        details.append(f"PyTorch 版本: {torch_version}")
    if cuda_version:
        details.append(f"CUDA 版本: {cuda_version}")
    details.extend(
        [
            f"内置支持架构: {supported_arches}",
            "这通常表示显卡架构过旧或过新，不在当前构建支持范围内。",
            "请改用兼容的 NVIDIA 显卡，或使用与该显卡兼容的 PyTorch/CUDA 重新打包。",
        ]
    )
    return "\n".join(details)


def ensure_cuda_runtime_compatibility(device: str) -> None:
    if not device.startswith("cuda"):
        return

    torch = _get_torch()
    if torch is None:
        return

    device_index = _get_cuda_device_index(device)
    try:
        probe = torch.zeros(1, device=device)
        add_in_place = getattr(probe, "add_", None)
        if callable(add_in_place):
            add_in_place(1)
        synchronize = getattr(torch.cuda, "synchronize", None)
        if callable(synchronize):
            synchronize(device_index)
    except RuntimeError as error:
        if is_unsupported_cuda_architecture_error(error):
            raise RuntimeError(rewrite_cuda_runtime_error(error, device)) from error
        raise


def ensure_xpu_runtime_compatibility(device: str) -> None:
    """Prove that native PyTorch XPU can execute and synchronize on ``device``.

    Merely exposing ``torch.xpu`` is not enough: a non-XPU wheel can contain the
    namespace while no Intel runtime or device is usable.  This probe is kept
    deliberately small, but it performs a real device allocation, kernel, host
    readback and synchronization.  Every failure is surfaced to the caller.
    """

    if not str(device).startswith("xpu"):
        return

    torch = _get_torch()
    if torch is None:
        raise RuntimeError("已请求 Intel XPU，但 PyTorch 无法导入")
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        raise RuntimeError(
            "已请求 Intel XPU，但当前 PyTorch 不包含 torch.xpu；"
            "当前项目需要固定的 PyTorch XPU 运行时"
        )
    if not bool(xpu.is_available()):
        raise RuntimeError(
            "已请求 Intel XPU，但 torch.xpu.is_available() 为 False；"
            "请检查 Intel 显卡驱动和 PyTorch XPU 运行时"
        )

    device_index = _get_device_index(str(device))
    device_count = int(xpu.device_count())
    if device_count <= 0:
        raise RuntimeError("torch.xpu 可用但未报告任何 Intel XPU 设备")
    if device_index >= device_count:
        raise RuntimeError(
            "Intel XPU 设备索引不可用: "
            f"requested={device_index}, count={device_count}"
        )

    try:
        probe = torch.ones(1, device=f"xpu:{device_index}")
        result = probe.add(1)
        if result.device.type != "xpu" or result.device.index != device_index:
            raise RuntimeError(
                "Intel XPU 张量探针落在错误设备: "
                f"expected=xpu:{device_index}, actual={result.device}"
            )
        observed = float(result.item())
        if observed != 2.0:
            raise RuntimeError(f"Intel XPU 张量探针结果无效: {observed!r}")
        xpu.synchronize(device_index)
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("Intel XPU"):
            raise
        raise RuntimeError(f"Intel XPU 运行时探针失败 ({device}): {exc}") from exc


def ensure_accelerator_runtime_compatibility(device: str) -> None:
    """Validate the explicitly selected CUDA or XPU accelerator runtime."""

    normalized = str(device)
    if normalized.startswith("cuda"):
        ensure_cuda_runtime_compatibility(normalized)
    elif normalized.startswith("xpu"):
        ensure_xpu_runtime_compatibility(normalized)


def ensure_module_on_device(module, device: str, label: str) -> None:
    """Require all parameters and buffers of a torch module on ``device``."""

    torch = _get_torch()
    if torch is None:
        raise RuntimeError(f"{label} 设备校验失败：PyTorch 无法导入")
    expected = torch.device(device)
    wrong = []
    seen = 0
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        seen += 1
        actual = tensor.device
        same_type = actual.type == expected.type
        same_index = expected.type == "cpu" or actual.index == expected.index
        if not (same_type and same_index):
            wrong.append(f"{name or '<unnamed>'}={actual}")
            if len(wrong) >= 8:
                break
    if seen == 0:
        raise RuntimeError(f"{label} 没有可校验的参数或缓冲区")
    if wrong:
        raise RuntimeError(
            f"{label} 未完整驻留在 {expected}；错误张量: " + ", ".join(wrong)
        )


def get_accelerator_type() -> str:
    """
    检测可用的加速器类型。

    返回:
        'cuda'  - NVIDIA GPU (CUDA)
        'rocm'  - AMD GPU (ROCm，通过 torch.cuda 接口)
        'mps'   - Apple Silicon GPU (Metal Performance Shaders)
        'xpu'   - Intel GPU (原生 PyTorch XPU)
        'directml' - 任意 GPU (通过 DirectML，支持 NVIDIA/AMD/Intel)
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

    # Intel GPU uses native PyTorch XPU.  IPEX is retired and is intentionally
    # neither imported nor required here.
    xpu = getattr(torch, "xpu", None)
    if xpu is not None:
        try:
            if xpu.is_available():
                return "xpu"
        except Exception as exc:
            logger.warning("Intel XPU 可用性检测失败: %s", exc)

    # DirectML（跨厂商 GPU 加速：NVIDIA/AMD/Intel，Windows 专用）
    try:
        import torch_directml
        if torch_directml.is_available():
            return "directml"
    except ImportError:
        pass

    return "cpu"


def is_gpu_available() -> bool:
    """检查是否有任何类型的GPU加速可用（CUDA、ROCm、MPS、Intel XPU 或 DirectML）"""
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
    from src.i18n.translator import t

    if not prefer_gpu:
        logger.info(t("startup.device_config_cpu"))
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
            logger.info(
                t(
                    "startup.device_accelerator_gpu",
                    accelerator=accel_label,
                    name=name,
                    device=device,
                )
            )
        except Exception:
            pass
        return device

    if accel == "mps":
        logger.info(t("startup.device_apple_mps"))
        return "mps"

    if accel == "xpu":
        torch = _get_torch()
        if torch is None or not hasattr(torch, "xpu"):
            raise RuntimeError("检测到 Intel XPU，但当前 PyTorch 缺少 torch.xpu")
        count = int(torch.xpu.device_count())
        if count <= 0:
            raise RuntimeError("检测到 Intel XPU，但 torch.xpu.device_count() 为 0")
        idx = int(gpu_index)
        if idx < 0 or idx >= count:
            raise RuntimeError(
                f"Intel XPU 设备索引不可用: requested={idx}, count={count}"
            )
        device = f"xpu:{idx}"
        ensure_xpu_runtime_compatibility(device)
        name = (
            torch.xpu.get_device_name(idx)
            if hasattr(torch.xpu, "get_device_name")
            else "Intel GPU"
        )
        logger.info(t("startup.device_intel_xpu", name=name, device=device))
        return device

    if accel == "directml":
        try:
            import torch_directml
            idx = min(gpu_index, torch_directml.device_count() - 1) if torch_directml.device_count() > 0 else 0
            device = torch_directml.device(idx)
            name = torch_directml.device_name(idx)
            logger.info(t("startup.device_directml_gpu", name=name, device=device))
            return str(device)
        except Exception:
            logger.info(t("startup.device_directml_default"))
            return "privateuseone:0"

    logger.info(t("startup.device_no_gpu_cpu"))
    return "cpu"


def get_gpu_count() -> int:
    """获取可用GPU数量（CUDA/ROCm/XPU/DirectML）"""
    torch = _get_torch()
    if torch is None:
        return 0
    accel = get_accelerator_type()
    if accel in ("cuda", "rocm"):
        return torch.cuda.device_count()
    if accel == "mps":
        return 1
    if accel == "xpu":
        return int(torch.xpu.device_count()) if hasattr(torch, "xpu") else 0
    if accel == "directml":
        try:
            import torch_directml
            return torch_directml.device_count()
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
            count = torch.xpu.device_count() if hasattr(torch, 'xpu') else 0
            for i in range(count):
                name = "Intel GPU"
                try:
                    if hasattr(torch.xpu, 'get_device_name'):
                        name = torch.xpu.get_device_name(i)
                except Exception:
                    pass
                props = (
                    torch.xpu.get_device_properties(i)
                    if hasattr(torch.xpu, "get_device_properties")
                    else None
                )
                total_memory = int(getattr(props, "total_memory", 0) or 0)
                gpus.append({
                    "index": i,
                    "name": name,
                    "type": "XPU",
                    "total_memory": total_memory,
                    "total_memory_gb": total_memory / (1024 ** 3),
                })
        except Exception as e:
            logger.warning(f"获取 Intel XPU 信息失败: {e}")

    elif accel == "directml":
        try:
            import torch_directml
            count = torch_directml.device_count()
            for i in range(count):
                name = torch_directml.device_name(i).strip('\x00').strip()
                gpus.append({
                    "index": i,
                    "name": name,
                    "type": "DirectML",
                    "total_memory": 0,
                    "total_memory_gb": 0.0,
                })
        except Exception as e:
            logger.warning(f"获取 DirectML GPU 信息失败: {e}")

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
        accel = get_accelerator_type()
        if accel == "cpu":
            return None
        device = get_device()

    try:
        if device.startswith("cuda") and torch.cuda.is_available():
            device_idx = int(device.split(":")[1]) if ":" in device else 0
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
            return (allocated, total)
        if device.startswith("xpu") and hasattr(torch, 'xpu') and torch.xpu.is_available():
            device_idx = _get_device_index(device)
            allocated = torch.xpu.memory_allocated(device_idx) / (1024 ** 3)
            total = torch.xpu.get_device_properties(device_idx).total_memory / (1024 ** 3)
            return (allocated, total)
    except Exception as e:
        logger.debug(f"获取显存信息失败: {e}")

    return None


def clear_gpu_memory() -> None:
    """Release completed accelerator work and unreferenced device allocations."""
    torch = _get_torch()
    if torch is None:
        return

    accel = get_accelerator_type()
    try:
        if accel in ("cuda", "rocm") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info("GPU显存缓存已清除")
        elif accel == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            logger.info("MPS显存缓存已清除")
        elif accel == "xpu" and hasattr(torch, 'xpu') and torch.xpu.is_available():
            # XPU client GPUs use unified memory, so a model retained by a
            # Python reference cycle consumes the same Windows commit budget
            # needed by the next inference process. Collect unreachable model
            # graphs first, wait for their queued work to finish, and only then
            # return unused blocks from the XPU caching allocator.
            gc.collect()
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
            logger.info("Intel XPU队列已同步，未引用模型与显存缓存已清除")
    except Exception as e:
        logger.warning(f"清除GPU显存失败: {e}")


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
    labels = {"cuda": "CUDA GPU", "rocm": "ROCm GPU", "mps": "Apple MPS", "xpu": "Intel XPU", "directml": "DirectML GPU"}
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


def get_available_memory_gb() -> float:
    """
    获取当前系统可用内存（GB）。

    返回:
        可用内存（GB），失败时返回 4.0 作为保守默认值
    """
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        logger.warning("无法获取可用内存，使用默认值 4.0 GB")
        return 4.0


def get_optimal_thread_count() -> int:
    """
    返回推荐的 PyTorch 线程数。

    策略：使用全部物理核心，最大化 CPU 利用率。

    返回:
        推荐线程数
    """
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or (os.cpu_count() or 4)
    except Exception:
        physical_cores = os.cpu_count() or 4

    threads = max(2, physical_cores)
    logger.debug(f"推荐线程数: {threads}（物理核数: {physical_cores}）")
    return threads


def get_system_performance_profile() -> dict:
    """
    检测系统硬件配置，返回性能档位信息。

    返回:
        {
            "tier": "high" | "medium" | "low",
            "ram_gb": float,
            "cpu_cores": int,
            "has_gpu": bool,
            "gpu_vram_gb": float | None,
            "device": str,
        }
    """
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        physical_cores = psutil.cpu_count(logical=False) or (os.cpu_count() or 1)
    except ImportError:
        # psutil 未安装时使用保守估计
        ram_gb = 8.0
        physical_cores = os.cpu_count() or 1

    has_gpu = is_gpu_available()
    gpu_vram_gb = None
    device = get_device()
    accel = get_accelerator_type()

    if has_gpu:
        mem = get_memory_info(device)
        if mem:
            gpu_vram_gb = mem[1]

    # 分档逻辑
    if accel == "directml":
        # DirectML：集显共享系统内存，按 RAM 分档；独显按显存分档
        gpus = get_gpu_info()
        gpu_name = gpus[0]["name"].lower() if gpus else ""
        is_integrated = any(k in gpu_name for k in ("uhd", "iris", "vega", "radeon graphics"))
        if is_integrated:
            # 集显共享系统 RAM，按 CPU 模式分档但标记为有 GPU
            if ram_gb >= 30 and physical_cores >= 8:
                tier = "high"
            elif ram_gb >= 14 and physical_cores >= 4:
                tier = "medium"
            else:
                tier = "low"
        else:
            # 独显（Arc/Radeon RX/GeForce 等）
            tier = "high" if ram_gb >= 14 else "medium"
    elif has_gpu and gpu_vram_gb is not None and gpu_vram_gb >= 6:
        tier = "high"
    elif has_gpu:
        tier = "medium"
    elif ram_gb >= 30 and physical_cores >= 8:
        tier = "high"
    elif ram_gb >= 14 and physical_cores >= 4:
        tier = "medium"
    else:
        tier = "low"

    profile = {
        "tier": tier,
        "ram_gb": round(ram_gb, 1),
        "cpu_cores": physical_cores,
        "has_gpu": has_gpu,
        "gpu_vram_gb": round(gpu_vram_gb, 1) if gpu_vram_gb else None,
        "device": device,
    }
    logger.info(f"系统性能档位: {tier} (RAM={profile['ram_gb']}GB, "
                f"CPU={physical_cores}核, GPU={'有' if has_gpu else '无'}"
                f"{f', VRAM={gpu_vram_gb:.1f}GB' if gpu_vram_gb else ''})")
    return profile


def get_optimal_batch_size(n_segments: int, quality: str, device: str,
                           ultra_quality: bool = False) -> int:
    """
    根据系统性能档位、分段数、质量模式和设备，动态计算最优批处理大小。

    参数:
        n_segments: 音频分段数
        quality: 内部质量标记；当前应用入口固定传入 ``best``
        device: 设备字符串 (如 "cpu", "cuda:0")
        ultra_quality: 是否沿用旧版 ultra-quality 配置

    返回:
        推荐的 batch size
    """
    batch_override = os.getenv("MUSIC_TO_MIDI_YOURMT3_BATCH_SIZE")
    if batch_override:
        try:
            override_value = int(batch_override)
        except ValueError as exc:
            raise ValueError(
                "MUSIC_TO_MIDI_YOURMT3_BATCH_SIZE must be a positive integer"
            ) from exc
        if override_value < 1:
            raise ValueError(
                "MUSIC_TO_MIDI_YOURMT3_BATCH_SIZE must be a positive integer"
            )
        logger.info("使用显式 YourMT3 batch size: %s", override_value)
        return override_value

    profile = get_system_performance_profile()
    tier = profile["tier"]
    is_best = (quality == "best" or ultra_quality)

    # GPU 模式（含 DirectML）
    is_gpu_device = profile["has_gpu"] and not device.startswith("cpu")
    is_directml = "privateuseone" in device

    is_cuda = device.startswith("cuda")
    is_xpu = device.startswith("xpu")

    if is_gpu_device and not is_directml:
        vram = profile["gpu_vram_gb"] or 4.0
        if is_cuda:
            # CUDA YourMT3 generation is autoregressive; high batch sizes can
            # thrash memory and stall without raising OOM on long audio.
            if is_best:
                if vram >= 10:
                    bsz_table = {100: 8, 300: 4, 999999: 2}
                elif vram >= 6:
                    bsz_table = {100: 6, 300: 3, 999999: 2}
                else:
                    bsz_table = {100: 4, 300: 2, 999999: 1}
            else:
                if vram >= 10:
                    bsz_table = {150: 8, 400: 4, 999999: 2}
                elif vram >= 6:
                    bsz_table = {150: 6, 400: 3, 999999: 2}
                else:
                    bsz_table = {150: 4, 400: 2, 999999: 1}
        elif is_xpu:
            # YourMT3 is autoregressive and the XPU route deliberately runs in
            # FP32.  Start with one segment per batch until a specific device is
            # measured; this avoids hiding memory pressure behind OOM retries.
            bsz_table = {999999: 1}
        else:
            # ROCm/MPS：无 fp16 autocast，使用保守 bsz
            if is_best:
                if vram >= 10:
                    bsz_table = {100: 24, 300: 20, 999999: 16}
                elif vram >= 6:
                    bsz_table = {100: 16, 300: 12, 999999: 10}
                else:
                    bsz_table = {100: 12, 300: 10, 999999: 8}
            else:
                if vram >= 10:
                    bsz_table = {150: 20, 400: 16, 999999: 12}
                elif vram >= 6:
                    bsz_table = {150: 12, 400: 10, 999999: 8}
                else:
                    bsz_table = {150: 10, 400: 8, 999999: 6}
    elif is_directml:
        # DirectML：集显共享系统内存，按 tier 分档（与 CPU 类似但有 GPU 加速）
        if tier == "high":
            if is_best:
                bsz_table = {100: 8, 300: 6, 999999: 4}
            else:
                bsz_table = {150: 6, 400: 4, 999999: 2}
        elif tier == "medium":
            if is_best:
                bsz_table = {100: 4, 300: 3, 999999: 2}
            else:
                bsz_table = {150: 4, 400: 2, 999999: 1}
        else:
            if is_best:
                bsz_table = {100: 2, 300: 1, 999999: 1}
            else:
                bsz_table = {150: 2, 400: 1, 999999: 1}
    else:
        # CPU 模式：根据可用内存动态计算批处理大小，尽量吃满资源
        available_mem = get_available_memory_gb()
        # 每段约需 0.5GB 内存，使用可用内存的 90%
        mem_based_bsz = max(1, int(available_mem * 0.9 / 0.5))

        if tier == "high":
            cap = 16 if is_best else 12
            if is_best:
                bsz_table = {100: min(cap, mem_based_bsz), 300: min(cap, mem_based_bsz), 999999: min(cap, mem_based_bsz)}
            else:
                bsz_table = {150: min(cap, mem_based_bsz), 400: min(cap, mem_based_bsz), 999999: min(cap, mem_based_bsz)}
        elif tier == "medium":
            cap = 12 if is_best else 8
            if is_best:
                bsz_table = {100: min(cap, mem_based_bsz), 300: min(cap, mem_based_bsz), 999999: min(cap, mem_based_bsz)}
            else:
                bsz_table = {150: min(cap, mem_based_bsz), 400: min(cap, mem_based_bsz), 999999: min(cap, mem_based_bsz)}
        else:
            cap = 6 if is_best else 4
            if is_best:
                bsz_table = {100: min(cap, mem_based_bsz), 300: min(cap, mem_based_bsz), 999999: max(1, min(cap, mem_based_bsz))}
            else:
                bsz_table = {150: min(cap, mem_based_bsz), 400: min(cap, mem_based_bsz), 999999: max(1, min(cap, mem_based_bsz))}

        logger.info(f"CPU 动态内存: 可用={available_mem:.1f}GB, 内存推算bsz={mem_based_bsz}, cap={cap}")

    bsz = 1
    for threshold, size in sorted(bsz_table.items()):
        if n_segments < threshold:
            bsz = size
            break

    logger.info(f"动态批处理: tier={tier}, segments={n_segments}, "
                f"quality={quality}, bsz={bsz}")
    return bsz
