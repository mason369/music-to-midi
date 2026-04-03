"""Transkun piano transcription wrapper."""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing
from contextlib import ExitStack
from importlib import resources
from pathlib import Path
from typing import Callable, Optional

from src.models.data_models import Config
from src.utils.gpu_utils import clear_gpu_memory, get_device

logger = logging.getLogger(__name__)


def _transkun_worker(
    audio_path: str,
    output_path: str,
    weight_path: str,
    conf_path: str,
    device: str,
    result_queue,
) -> None:
    try:
        import moduleconf
        import soxr
        import torch
        from transkun.Data import writeMidi
        from transkun.transcribe import readAudio

        conf_manager = moduleconf.parseFromFile(str(conf_path))
        transkun_cls = conf_manager["Model"].module.TransKun
        conf = conf_manager["Model"].config

        checkpoint = torch.load(str(weight_path), map_location=device)
        model = transkun_cls(conf=conf).to(device)

        state_dict = checkpoint.get("best_state_dict") or checkpoint.get("state_dict")
        if state_dict is None:
            raise RuntimeError("Transkun checkpoint 缺少 state_dict")
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        fs, audio = readAudio(str(audio_path))
        if fs != model.fs:
            audio = soxr.resample(audio, fs, model.fs)

        with torch.no_grad():
            notes_est = model.transcribe(
                torch.from_numpy(audio).to(device),
                discardSecondHalf=False,
            )

        output_midi = writeMidi(notes_est)
        output_midi.write(str(output_path))
        result_queue.put({"ok": str(output_path)})
    except Exception as exc:  # pragma: no cover - exercised through parent wrapper
        result_queue.put({"error": str(exc)})


class TranskunTranscriber:
    """Run Transkun's packaged piano model and export a single-track MIDI."""

    WEIGHT_NAME = "2.0.pt"
    CONF_NAME = "2.0.conf"

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._cancelled = False
        self._cancel_check: Optional[Callable[[], bool]] = None
        self._process = None

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("transkun.transcribe") is not None

    @classmethod
    def _get_packaged_resource(cls, name: str):
        if not cls.is_available():
            return None
        try:
            return resources.files("transkun").joinpath("pretrained", name)
        except (ModuleNotFoundError, FileNotFoundError):
            return None

    def is_model_available(self) -> bool:
        weight = self._get_packaged_resource(self.WEIGHT_NAME)
        conf = self._get_packaged_resource(self.CONF_NAME)
        return bool(weight and conf and weight.is_file() and conf.is_file())

    def set_cancel_check(self, callback: Optional[Callable[[], bool]]) -> None:
        self._cancel_check = callback

    def cancel(self) -> None:
        self._cancelled = True
        process = self._process
        if process is not None and self._process_is_alive(process):
            logger.info("正在终止 Transkun 子进程...")
            process.terminate()

    @staticmethod
    def _process_is_alive(process) -> bool:
        is_alive = getattr(process, "is_alive", None)
        if callable(is_alive):
            return bool(is_alive())
        poll = getattr(process, "poll", None)
        if callable(poll):
            return poll() is None
        return False

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("Transkun 转写处理已取消")
        if self._cancel_check and self._cancel_check():
            raise InterruptedError("Transkun 转写处理已取消")

    def _resolve_runtime_device(self) -> str:
        preferred = get_device(self.config.use_gpu, self.config.gpu_device)
        if preferred.startswith("cuda"):
            return preferred
        if preferred != "cpu":
            logger.warning(
                "Transkun 当前仅对 CPU/CUDA 路径做了集成验证，检测到设备 %s，回退 CPU",
                preferred,
            )
        return "cpu"

    def transcribe(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        input_path = Path(audio_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            raise RuntimeError(
                "Transkun 未安装。请执行: python -m pip install transkun"
            )
        if not self.is_model_available():
            raise RuntimeError(
                "Transkun 预训练资源缺失或安装不完整。请执行: "
                "python -m pip install --force-reinstall transkun"
            )

        self._cancelled = False
        device = self._resolve_runtime_device()
        logger.info("Running Transkun transcription on %s: %s", device, input_path)

        try:
            with ExitStack() as stack:
                weight_path = stack.enter_context(
                    resources.as_file(self._get_packaged_resource(self.WEIGHT_NAME))
                )
                conf_path = stack.enter_context(
                    resources.as_file(self._get_packaged_resource(self.CONF_NAME))
                )

                if progress_callback:
                    progress_callback(0.05, f"正在准备 Transkun ({device})...")

                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=_transkun_worker,
                    args=(
                        str(input_path),
                        str(out_path),
                        str(weight_path),
                        str(conf_path),
                        device,
                        result_queue,
                    ),
                )
                self._process = process
                process.start()

                if progress_callback:
                    progress_callback(0.30, "正在运行 Transkun 钢琴转写...")

                while process.is_alive():
                    self._check_cancelled()
                    process.join(timeout=0.2)

                self._process = None
                self._check_cancelled()

                if result_queue.empty():
                    raise RuntimeError("Transkun 子进程未返回结果")
                result = result_queue.get()
                if "error" in result:
                    raise RuntimeError(result["error"])

        except InterruptedError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Transkun 转写失败: {exc}") from exc
        finally:
            process = self._process
            if process is not None and self._process_is_alive(process):
                process.terminate()
                join = getattr(process, "join", None)
                if callable(join):
                    join(timeout=1.0)
            self._process = None
            try:
                clear_gpu_memory()
            except Exception:
                pass

        if progress_callback:
            progress_callback(1.0, "Transkun 转写完成")

        logger.info("Transkun output: %s", out_path)
        return str(out_path)
