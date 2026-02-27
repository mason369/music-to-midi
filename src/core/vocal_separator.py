"""
人声分离模块 - 使用 Demucs htdemucs 模型分离人声与伴奏
"""
import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional, Callable, Dict

logger = logging.getLogger(__name__)


class VocalSeparator:
    """
    使用 Demucs htdemucs 模型将音频分离为人声和伴奏。

    输出两个 WAV 文件：vocals.wav 和 accompaniment.wav
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            from src.utils.gpu_utils import get_device
            device = get_device()
        self._device = device
        self._model = None
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
        """检查 Demucs 是否可用"""
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            return True
        except ImportError:
            return False

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
        import torch
        import soundfile as sf
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from demucs.audio import AudioFile
        from src.utils.gpu_utils import clear_gpu_memory, get_available_memory_gb, get_optimal_thread_count

        # 确保 CPU 模式下充分利用多核
        if self._device == "cpu":
            optimal_threads = get_optimal_thread_count()
            torch.set_num_threads(optimal_threads)
            logger.info(f"Demucs CPU 线程数: {optimal_threads}")

        self._cancelled = False
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(audio_path).stem

        if progress_callback:
            progress_callback(0.0, "正在加载 Demucs htdemucs 模型...")

        logger.info(f"加载 Demucs htdemucs 模型，设备: {self._device}")

        model = None
        sources = None
        wav = None

        try:
            logger.info("正在下载/加载 Demucs htdemucs 预训练模型...")
            import time as _time
            _load_start = _time.time()
            model = get_model("htdemucs")
            logger.info(f"Demucs 模型加载完成: 耗时={_time.time() - _load_start:.1f}s")
            logger.info(f"正在将模型移至 {self._device}...")
            model.to(self._device)
            logger.info("模型已就绪")
        except Exception as e:
            logger.error(f"Demucs 模型加载失败: {e}")
            raise RuntimeError(f"Demucs 模型加载失败: {e}") from e

        try:
            self._check_cancelled()

            if progress_callback:
                progress_callback(0.2, "正在分离人声与伴奏...")

            logger.info(f"开始分离: {audio_path}")

            # 加载音频并重采样到模型采样率
            load_start = time.time()
            af = AudioFile(audio_path)
            wav = af.read(
                streams=0,
                samplerate=model.samplerate,
                channels=model.audio_channels,
            )
            duration_sec = wav.shape[-1] / model.samplerate
            logger.info(f"音频加载完成: {duration_sec:.1f}秒, 采样率={model.samplerate}Hz, "
                         f"耗时={time.time() - load_start:.1f}s")

            ref = wav.mean(0)
            ref_std = ref.std()
            if ref_std < 1e-7:
                ref_std = torch.tensor(1.0)
            wav = (wav - ref.mean()) / ref_std
            wav = wav.to(self._device)
            is_cpu = self._device == "cpu"
            available_mem = get_available_memory_gb()
            num_workers = 0 if is_cpu else max(1, get_optimal_thread_count() // 2)

            # 计算分段信息（BagOfModels 包装器没有 segment，从内部模型获取）
            if hasattr(model, 'segment'):
                segment_sec = float(model.segment)
            elif hasattr(model, 'models') and model.models:
                segment_sec = float(model.models[0].segment)
            else:
                segment_sec = 7.8  # htdemucs 默认值
            segment_samples = int(model.samplerate * segment_sec)
            total_samples = wav.shape[-1]
            n_chunks = max(1, (total_samples + segment_samples - 1) // segment_samples)

            logger.info(f"Demucs 性能参数: num_workers={num_workers}, segment={segment_sec:.1f}s, "
                         f"可用内存={available_mem:.1f}GB, torch线程={torch.get_num_threads()}")
            logger.info(f"Demucs 分段处理: {n_chunks} 个分段, 每段 {segment_sec:.1f}s, "
                         f"总时长 {duration_sec:.1f}s")

            if progress_callback:
                progress_callback(0.25, f"正在分离人声与伴奏（{n_chunks}个分段, 约{duration_sec:.0f}秒音频）...")

            # 运行模型（带进度估算）
            sep_start = time.time()
            logger.info("Demucs apply_model 开始推理...")

            # overlap 和 shifts 提升分离纯净度（尤其人声）
            # GPU 模式可以更激进，CPU 模式适度
            if is_cpu:
                overlap = 0.25   # 默认值
                shifts = 1       # 1 次随机偏移平均
            else:
                overlap = 0.5    # 50% 重叠，显著减少边界伪影
                shifts = 3       # 3 次随机偏移平均，大幅提升纯净度

            logger.info(f"Demucs 质量参数: overlap={overlap}, shifts={shifts}")

            # 启动后台线程定期更新进度（apply_model 本身无回调）
            _done = threading.Event()

            def _progress_ticker(_shifts=shifts):
                """基于已用时间估算进度，每 3 秒更新一次"""
                # 根据设备和 shifts 估算速度
                # GPU ~6x 实时（shifts=1），shifts=3 约 2x 实时
                # CPU ~0.1x 实时（shifts=1）
                if is_cpu:
                    speed_est = 0.1
                else:
                    speed_est = 6.0 / max(_shifts, 1)
                estimated_total = duration_sec / speed_est
                while not _done.wait(3.0):
                    try:
                        elapsed = time.time() - sep_start
                        # 进度从 0.25 到 0.75（留 0.75-0.8 给保存）
                        pct = min(0.95, elapsed / max(estimated_total, 1))
                        overall = 0.25 + pct * 0.50
                        remaining = max(0, estimated_total - elapsed)
                        if progress_callback:
                            progress_callback(overall, f"正在分离人声与伴奏... 已用时 {elapsed:.0f}s, 预计剩余 {remaining:.0f}s")
                        logger.debug(f"Demucs 推理中: {elapsed:.0f}s / ~{estimated_total:.0f}s 估计")
                    except Exception as e:
                        logger.warning(f"进度更新失败: {e}")
                        break

            if progress_callback:
                ticker = threading.Thread(target=_progress_ticker, daemon=True)
                ticker.start()

            try:
                sources = apply_model(
                    model, wav[None], device=self._device,
                    split=True, num_workers=num_workers,
                    overlap=overlap, shifts=shifts,
                )[0]
            finally:
                _done.set()

            sep_elapsed = time.time() - sep_start
            speed_ratio = duration_sec / sep_elapsed if sep_elapsed > 0 else 0
            logger.info(f"Demucs 推理完成: 耗时={sep_elapsed:.1f}s, "
                         f"速度={speed_ratio:.2f}x 实时")

            self._check_cancelled()

            if progress_callback:
                progress_callback(0.8, "正在保存分离结果...")

            # 还原归一化（确保 ref_std 与 sources 在同一设备上）
            sources = sources * ref_std.to(sources.device) + ref.mean().to(sources.device)

            # htdemucs 输出 stems 顺序与 model.sources 对应
            source_names = model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
            sources_dict = dict(zip(source_names, sources))

            if "vocals" not in sources_dict:
                available = list(sources_dict.keys())
                raise RuntimeError(
                    f"Demucs 输出中未找到 vocals stem，可用 stems: {available}"
                )

            vocals = sources_dict["vocals"]  # tensor (channels, samples)
            non_vocal_keys = [k for k in sources_dict if k != "vocals"]
            if not non_vocal_keys:
                raise RuntimeError("Demucs 输出中只有 vocals，无法提取伴奏")
            accompaniment = sources_dict[non_vocal_keys[0]]
            for k in non_vocal_keys[1:]:
                accompaniment = accompaniment + sources_dict[k]

            sample_rate = model.samplerate

            # 转为 numpy 并保存
            vocals_np = vocals.cpu().numpy()
            accompaniment_np = accompaniment.cpu().numpy()

            # demucs 输出形状: (channels, samples)
            # soundfile 需要 (samples, channels)
            if vocals_np.ndim == 2:
                vocals_np = vocals_np.T
                accompaniment_np = accompaniment_np.T

            vocals_path = os.path.join(output_dir, f"{stem}_vocals.wav")
            accompaniment_path = os.path.join(output_dir, f"{stem}_accompaniment.wav")

            sf.write(vocals_path, vocals_np, sample_rate)
            sf.write(accompaniment_path, accompaniment_np, sample_rate)

            logger.info(f"人声已保存: {vocals_path}")
            logger.info(f"伴奏已保存: {accompaniment_path}")
        except InterruptedError:
            raise
        except Exception as e:
            logger.error(f"音频分离失败: {e}")
            raise RuntimeError(f"音频分离失败: {e}") from e
        finally:
            # 无论成功或失败都释放 GPU 资源
            del model, sources, wav
            clear_gpu_memory()

        if progress_callback:
            progress_callback(1.0, "分离完成")

        return {
            "vocals": vocals_path,
            "no_vocals": accompaniment_path,
        }
