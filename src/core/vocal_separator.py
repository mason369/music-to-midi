"""
人声分离模块 - 使用 audio-separator + BS-RoFormer 模型分离人声与伴奏
"""
import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional, Callable, Dict

logger = logging.getLogger(__name__)

# BS-RoFormer 模型文件名（SDR 12.9755）
_ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


class VocalSeparator:
    """
    使用 audio-separator + BS-RoFormer 模型将音频分离为人声和伴奏。

    输出两个 WAV 文件：vocals.wav 和 accompaniment.wav

    注意：audio-separator 的 Separator 类自行检测 GPU 设备，
    不接受外部 device 参数。separator.separate() 是阻塞调用，
    分离过程中无法响应取消，仅在调用前后检查取消状态。
    """

    def __init__(self):
        self._cancelled = False
        self._cancel_check: Optional[Callable[[], bool]] = None

    def set_cancel_check(self, fn: Callable[[], bool]) -> None:
        self._cancel_check = fn

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("用户取消了处理")
        if self._cancel_check and self._cancel_check():
            raise InterruptedError("用户取消了处理")

    def cancel(self) -> None:
        self._cancelled = True

    @staticmethod
    def is_available() -> bool:
        """检查 audio-separator 是否可用"""
        try:
            from audio_separator.separator import Separator  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _get_model_cache_dir() -> str:
        """返回模型缓存目录"""
        cache_dir = (
            Path.home() / ".music-to-midi" / "models" / "audio-separator"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, str]:
        """
        分离音频为人声和伴奏。

        参数:
            audio_path: 输入音频路径
            output_dir: 输出目录
            progress_callback: 进度回调 (progress 0-1, message)

        返回:
            {"vocals": vocals_path, "no_vocals": accompaniment_path}
        """
        from audio_separator.separator import Separator
        from src.utils.gpu_utils import clear_gpu_memory

        self._cancelled = False
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(Path(audio_path).name).stem

        if progress_callback:
            progress_callback(0.0, "正在加载 BS-RoFormer 模型...")

        separator = None

        try:
            # 创建 Separator 实例
            load_start = time.time()
            logger.info("正在下载/加载 BS-RoFormer 预训练模型...")

            separator = Separator(
                output_dir=output_dir,
                model_file_dir=self._get_model_cache_dir(),
                output_format="WAV",
            )
            separator.load_model(_ROFORMER_MODEL)

            # 记录实际使用的设备（由 audio-separator 自动检测）
            actual_device = getattr(
                separator, 'torch_device', 'unknown'
            )
            logger.info(
                f"BS-RoFormer 模型加载完成: "
                f"耗时={time.time() - load_start:.1f}s, "
                f"设备={actual_device}"
            )

            self._check_cancelled()

            if progress_callback:
                progress_callback(0.2, "正在分离人声与伴奏...")

            logger.info(f"开始分离: {audio_path}")

            # 启动后台线程定期更新进度
            # （separator.separate 是阻塞调用无回调）
            sep_start = time.time()
            _done = threading.Event()

            # 通过 soundfile 获取音频时长用于估算进度
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                duration_sec = info.duration
            except Exception:
                duration_sec = 180.0  # 默认估算 3 分钟

            is_cpu = str(actual_device) == "cpu"

            def _progress_ticker():
                """基于已用时间估算进度，每 3 秒更新一次"""
                if is_cpu:
                    speed_est = 0.15
                else:
                    speed_est = 3.0
                estimated_total = duration_sec / speed_est
                while not _done.wait(3.0):
                    try:
                        elapsed = time.time() - sep_start
                        pct = min(
                            0.95,
                            elapsed / max(estimated_total, 1)
                        )
                        # 映射到 0.2 ~ 0.8 范围（与其他回调点一致）
                        local_progress = 0.2 + pct * 0.6
                        remaining = max(0, estimated_total - elapsed)
                        if progress_callback:
                            progress_callback(
                                local_progress,
                                f"正在分离人声与伴奏... "
                                f"已用时 {elapsed:.0f}s, "
                                f"预计剩余 {remaining:.0f}s"
                            )
                        logger.debug(
                            f"BS-RoFormer 推理中: {elapsed:.0f}s / "
                            f"~{estimated_total:.0f}s 估计"
                        )
                    except Exception as e:
                        logger.warning(f"进度更新失败: {e}")

            ticker = None
            if progress_callback:
                ticker = threading.Thread(
                    target=_progress_ticker, daemon=True
                )
                ticker.start()

            try:
                output_files = separator.separate(audio_path)
            finally:
                _done.set()
                if ticker is not None:
                    ticker.join(timeout=2)

            sep_elapsed = time.time() - sep_start
            speed_ratio = (
                duration_sec / sep_elapsed if sep_elapsed > 0 else 0
            )
            logger.info(
                f"BS-RoFormer 推理完成: 耗时={sep_elapsed:.1f}s, "
                f"速度={speed_ratio:.2f}x 实时"
            )

            self._check_cancelled()

            # 检查 separator 是否返回了有效输出
            if not output_files:
                raise RuntimeError(
                    "audio-separator 未返回任何输出文件，"
                    "分离可能已静默失败，请检查日志"
                )

            if progress_callback:
                progress_callback(0.8, "正在保存分离结果...")

            # audio-separator 输出文件名含 (Vocals) / (Instrumental) 后缀
            # 需要重命名为统一格式
            vocals_path = os.path.join(
                output_dir, f"{stem}_vocals.wav"
            )
            accompaniment_path = os.path.join(
                output_dir, f"{stem}_accompaniment.wav"
            )

            vocals_found = False
            instrumental_found = False

            for fpath in output_files:
                fpath_str = os.path.normpath(str(fpath))
                # audio-separator 可能返回纯文件名，需拼接 output_dir
                if not os.path.isabs(fpath_str):
                    fpath_str = os.path.join(output_dir, fpath_str)
                fname_lower = os.path.basename(fpath_str).lower()
                norm_vocals = os.path.normpath(vocals_path)
                norm_accomp = os.path.normpath(accompaniment_path)
                if (
                    "vocal" in fname_lower
                    and "instrumental" not in fname_lower
                ):
                    if os.path.exists(fpath_str):
                        vocals_found = True
                        if os.path.abspath(fpath_str).lower() != os.path.abspath(norm_vocals).lower():
                            os.replace(fpath_str, vocals_path)
                elif "instrumental" in fname_lower:
                    if os.path.exists(fpath_str):
                        instrumental_found = True
                        if os.path.abspath(fpath_str).lower() != os.path.abspath(norm_accomp).lower():
                            os.replace(fpath_str, accompaniment_path)

            if not vocals_found or not instrumental_found:
                logger.warning(
                    f"audio-separator 输出文件匹配异常: "
                    f"{output_files}, "
                    f"vocals_found={vocals_found}, "
                    f"instrumental_found={instrumental_found}"
                )
                # 回退：尝试按顺序分配未匹配的文件
                for fpath in output_files:
                    fpath_str = os.path.normpath(str(fpath))
                    if not os.path.isabs(fpath_str):
                        fpath_str = os.path.join(output_dir, fpath_str)
                    if not os.path.exists(fpath_str):
                        continue
                    if os.path.abspath(fpath_str).lower() in (
                        os.path.abspath(vocals_path).lower(),
                        os.path.abspath(accompaniment_path).lower(),
                    ):
                        continue
                    if (
                        not instrumental_found
                        and not os.path.exists(accompaniment_path)
                    ):
                        os.replace(fpath_str, accompaniment_path)
                        instrumental_found = True
                    elif (
                        not vocals_found
                        and not os.path.exists(vocals_path)
                    ):
                        os.replace(fpath_str, vocals_path)
                        vocals_found = True

            if not os.path.exists(vocals_path):
                raise RuntimeError(
                    f"人声文件未找到: {vocals_path}, "
                    f"audio-separator 输出: {output_files}"
                )
            if not os.path.exists(accompaniment_path):
                raise RuntimeError(
                    f"伴奏文件未找到: {accompaniment_path}, "
                    f"audio-separator 输出: {output_files}"
                )

            logger.info(f"人声已保存: {vocals_path}")
            logger.info(f"伴奏已保存: {accompaniment_path}")
        except InterruptedError:
            raise
        except Exception as e:
            logger.error(f"音频分离失败: {e}")
            raise RuntimeError(f"音频分离失败: {e}") from e
        finally:
            # 无论成功或失败都释放资源
            try:
                if separator is not None:
                    del separator
            except Exception as e:
                logger.debug("释放 separator 失败: %s", e)
            clear_gpu_memory()

        if progress_callback:
            progress_callback(1.0, "分离完成")

        return {
            "vocals": vocals_path,
            "no_vocals": accompaniment_path,
        }
