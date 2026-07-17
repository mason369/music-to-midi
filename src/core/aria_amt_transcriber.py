"""Aria-AMT piano transcription wrapper."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Callable, Optional

import torchaudio

from src.i18n.translator import Translator
from src.utils.artifact_identity import validate_file_identity
from src.utils.gpu_utils import ensure_cuda_runtime_compatibility, rewrite_cuda_runtime_error
from src.utils.midi_output import publish_midi_output
from src.utils.runtime_paths import get_aria_amt_dir, is_frozen_app

logger = logging.getLogger(__name__)

ARIA_AMT_CHECKPOINT_NAME = "piano-medium-double-1.0.safetensors"
ARIA_AMT_MODEL_CONFIG_NAME = "medium-double"
ARIA_AMT_PACKAGE_NAME = "aria-amt"
ARIA_AMT_SOURCE_REVISION = "a1ab73fc901d1759ec3bc173c146b3c6a3040261"
ARIA_AMT_SOURCE_ARCHIVE_URL = (
    "https://github.com/EleutherAI/aria-amt/archive/" f"{ARIA_AMT_SOURCE_REVISION}.zip"
)
ARIA_AMT_SOURCE_REQUIREMENT = f"aria-amt @ {ARIA_AMT_SOURCE_ARCHIVE_URL}"
ARIA_AMT_CHECKPOINT_REVISION = "8cc4cf5c83b47f2689ac256a947b2a57c17a4c8b"
ARIA_AMT_CHECKPOINT_URL = (
    "https://huggingface.co/datasets/loubb/aria-midi/resolve/"
    f"{ARIA_AMT_CHECKPOINT_REVISION}/"
    "piano-medium-double-1.0.safetensors?download=true"
)
ARIA_AMT_CHECKPOINT_SIZE = 446_577_344
ARIA_AMT_CHECKPOINT_SHA256 = "089d3129dbe93246aeda55efe668c8a48af08afaf9dd15c64cef0a07c0fb30a4"
ARIA_AMT_CACHE_DIR = Path.home() / ".cache" / "music_ai_models" / "aria_amt"


def get_aria_amt_runtime_unavailable_reason() -> str:
    """Return why the installed Aria-AMT source is not the pinned revision."""

    try:
        if importlib.util.find_spec("amt.run") is None:
            return "Aria-AMT 未安装。请安装固定源码版本: " f"{ARIA_AMT_SOURCE_REQUIREMENT}"
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return f"Aria-AMT 模块不可用: {exc}"

    try:
        distribution = metadata.distribution(ARIA_AMT_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return (
            "Aria-AMT 缺少安装元数据，无法验证源码版本。请重新安装固定源码版本: "
            f"{ARIA_AMT_SOURCE_REQUIREMENT}"
        )

    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        return (
            "Aria-AMT 缺少 direct_url.json，无法验证源码提交。请重新安装固定源码版本: "
            f"{ARIA_AMT_SOURCE_REQUIREMENT}"
        )

    try:
        direct_url = json.loads(direct_url_text)
    except (TypeError, json.JSONDecodeError) as exc:
        return f"Aria-AMT direct_url.json 无效，无法验证源码提交: {exc}"

    vcs_info = direct_url.get("vcs_info") or {}
    if vcs_info:
        if vcs_info.get("commit_id") == ARIA_AMT_SOURCE_REVISION:
            return ""
    elif direct_url.get("url") == ARIA_AMT_SOURCE_ARCHIVE_URL:
        return ""

    observed_commit = vcs_info.get("commit_id") or "unknown"
    observed_url = direct_url.get("url") or "unknown"
    return (
        "Aria-AMT 源码版本不匹配: "
        f"expected commit {ARIA_AMT_SOURCE_REVISION}, "
        f"got commit {observed_commit}, source {observed_url}。"
        f"请重新安装固定源码版本: {ARIA_AMT_SOURCE_REQUIREMENT}"
    )


def validate_aria_amt_checkpoint(path: Path) -> Path:
    """Require the exact pinned Aria-AMT checkpoint artifact."""

    return validate_file_identity(
        path,
        expected_size=ARIA_AMT_CHECKPOINT_SIZE,
        expected_sha256=ARIA_AMT_CHECKPOINT_SHA256,
        label="Aria-AMT checkpoint",
    )


class AriaAmtTranscriber:
    def __init__(
        self, checkpoint_path: Optional[Path] = None, language: str = Translator.DEFAULT_LANGUAGE
    ):
        if checkpoint_path is None:
            checkpoint_path = self.default_checkpoint_path()
        self.checkpoint_path = Path(checkpoint_path)
        self._cancelled = False
        self._process: Optional[subprocess.Popen[str]] = None
        self._translator = Translator(language)

    def _pt(self, key: str, **kwargs) -> str:
        return self._translator.t(key, **kwargs)

    @staticmethod
    def default_checkpoint_path() -> Path:
        return get_aria_amt_dir() / ARIA_AMT_CHECKPOINT_NAME

    @staticmethod
    def is_available() -> bool:
        return get_aria_amt_runtime_unavailable_reason() == ""

    @staticmethod
    def get_unavailable_reason() -> str:
        return get_aria_amt_runtime_unavailable_reason()

    def is_model_available(self) -> bool:
        try:
            validate_aria_amt_checkpoint(self.checkpoint_path)
            return True
        except (OSError, RuntimeError):
            return False

    @staticmethod
    def _guess_output_midi(save_dir: Path, audio_path: Path) -> Optional[Path]:
        direct = save_dir / f"{audio_path.stem}.mid"
        if direct.exists():
            return direct
        midis = sorted(save_dir.glob("*.mid"))
        return midis[0] if midis else None

    @staticmethod
    def _format_missing_output_error(out_path: Path, temp_dir: Path) -> str:
        lines = [
            "Aria-AMT 未生成 MIDI 输出",
            f"期望输出: {out_path.resolve()}",
            f"临时输出目录: {temp_dir.resolve()}",
        ]
        if temp_dir.exists():
            entries = sorted(path.resolve() for path in temp_dir.iterdir())
            if entries:
                lines.append("临时输出目录内容:")
                lines.extend(f"  {entry}" for entry in entries[:20])
                if len(entries) > 20:
                    lines.append(f"  ... 另外 {len(entries) - 20} 个")
            else:
                lines.append("临时输出目录为空")
        else:
            lines.append("临时输出目录不存在")
        return "\n".join(lines)

    @staticmethod
    def _save_token_sequence_as_midi(tokenizer, sequence: list, save_path: Path) -> None:
        last_onset = None
        for token in reversed(sequence):
            if isinstance(token, tuple) and token[0] == "onset":
                last_onset = token[1]
                break
        if last_onset is None:
            raise RuntimeError("Aria-AMT 未生成有效的 onset token，无法保存 MIDI")

        midi_dict = tokenizer.detokenize(tokenized_seq=sequence, len_ms=last_onset)
        midi_dict.remove_redundant_pedals()
        midi = midi_dict.to_midi()
        midi.save(str(save_path))

    def _load_aria_model(self):
        from amt.config import load_model_config
        from amt.inference.model import AmtEncoderDecoder, ModelConfig
        from amt.tokenizer import AmtTokenizer
        from amt.utils import _load_weight

        tokenizer = AmtTokenizer()
        model_config = ModelConfig(**load_model_config(ARIA_AMT_MODEL_CONFIG_NAME))
        model_config.set_vocab_size(tokenizer.vocab_size)
        model = AmtEncoderDecoder(model_config)
        model_state = _load_weight(ckpt_path=str(self.checkpoint_path))

        normalized_state = {}
        for key, value in model_state.items():
            if key.startswith("_orig_mod."):
                normalized_state[key[len("_orig_mod.") :]] = value
            else:
                normalized_state[key] = value
        model.load_state_dict(normalized_state)
        return model, tokenizer

    @staticmethod
    def _iter_windows_wav_segments(
        input_path: Path,
        sample_rate: int,
        chunk_len_seconds: int,
        stride_factor: int,
    ):
        import torch
        import torch.nn.functional as torch_functional

        waveform, original_sample_rate = torchaudio.load(str(input_path))
        waveform = waveform.mean(0)
        if original_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=original_sample_rate,
                new_freq=sample_rate,
            )

        chunk_samples = int(sample_rate * chunk_len_seconds)
        stride_samples = int(chunk_samples // stride_factor)
        if len(waveform) <= chunk_samples:
            yield torch_functional.pad(waveform, (0, chunk_samples - len(waveform)))
            return

        buffer = torch.tensor([], dtype=torch.float32)
        for start in range(0, len(waveform), stride_samples):
            stride_segment = waveform[start : start + stride_samples]
            if stride_segment.shape[0] < stride_samples:
                stride_segment = torch_functional.pad(
                    stride_segment,
                    (0, stride_samples - stride_segment.shape[0]),
                    mode="constant",
                    value=0.0,
                )

            buffer = torch.cat((buffer, stride_segment))
            if len(buffer) < chunk_samples:
                continue
            yield buffer[:chunk_samples]
            buffer = buffer[stride_samples:]

            if start + stride_samples >= len(waveform):
                break

        if len(buffer) > 0:
            yield torch_functional.pad(buffer, (0, chunk_samples - len(buffer)))

    def _run_transcription_windows_single_file(
        self,
        input_path: Path,
        temp_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        try:
            import torch
            from torch.cuda import is_available as cuda_is_available
            from torch.cuda import is_bf16_supported

            from amt.audio import AudioTransform
            from amt.config import load_config
            from amt.inference import transcribe as transcribe_module

            if not cuda_is_available():
                raise RuntimeError("CUDA device not found")
            ensure_cuda_runtime_compatibility("cuda:0")

            model, tokenizer = self._load_aria_model()
            model.decoder.setup_cache(
                batch_size=1,
                max_seq_len=transcribe_module.MAX_BLOCK_LEN,
                dtype=torch.bfloat16 if is_bf16_supported() else torch.float,
            )
            model.cuda()
            model.eval()
            audio_transform = AudioTransform().cuda()
            audio_config = load_config()["audio"]

            sequence = [tokenizer.bos_tok]
            concat_sequence = [tokenizer.bos_tok]
            for index, audio_segment in enumerate(
                self._iter_windows_wav_segments(
                    input_path=input_path,
                    sample_rate=audio_config["sample_rate"],
                    chunk_len_seconds=audio_config["chunk_len"],
                    stride_factor=transcribe_module.STRIDE_FACTOR,
                )
            ):
                if self._cancelled:
                    raise InterruptedError("Aria-AMT 转写处理已取消")

                if progress_callback:
                    progress_callback(
                        0.10 + min(index, 8) * 0.08, self._pt("progress.running_aria_amt")
                    )

                init_index = len(sequence)
                silent_intervals = transcribe_module._get_silent_intervals(audio_segment)
                (sequence,) = transcribe_module.process_segments(
                    tasks=[((audio_segment, sequence), 0)],
                    model=model,
                    audio_transform=audio_transform,
                    tokenizer=tokenizer,
                    logger=logger,
                )
                adjusted_sequence = transcribe_module._process_silent_intervals(
                    sequence,
                    intervals=silent_intervals,
                    tokenizer=tokenizer,
                )
                if len(adjusted_sequence) < len(sequence) - 15:
                    sequence = adjusted_sequence
                next_sequence = transcribe_module._truncate_seq(
                    sequence,
                    transcribe_module.CHUNK_LEN_MS,
                    transcribe_module.LEN_MS - transcribe_module.CHUNK_LEN_MS,
                )

                if sequence[-1] == tokenizer.eos_tok:
                    sequence = sequence[:-1]
                concat_sequence += transcribe_module._shift_onset(
                    sequence[init_index:],
                    index * transcribe_module.CHUNK_LEN_MS,
                )
                sequence = [tokenizer.bos_tok] if len(next_sequence) == 1 else next_sequence

            if len(concat_sequence) < 10:
                raise RuntimeError("Aria-AMT 推理结果为空或过短，未生成可保存的 MIDI")

            self._save_token_sequence_as_midi(
                tokenizer,
                concat_sequence,
                temp_dir / f"{input_path.stem}.mid",
            )
        except InterruptedError:
            raise
        except Exception as exc:
            friendly_message = rewrite_cuda_runtime_error(exc, "cuda:0")
            raise RuntimeError(f"Aria-AMT 转写失败:\n{friendly_message}") from exc

    def _run_transcription_in_process(self, input_path: Path, temp_dir: Path) -> None:
        try:
            run_module = importlib.import_module("amt.run")
            run_module.transcribe(
                model_name=ARIA_AMT_MODEL_CONFIG_NAME,
                checkpoint_path=str(self.checkpoint_path),
                load_path=str(input_path),
                load_dir=None,
                save_dir=str(temp_dir),
                batch_size=1,
            )
        except Exception as exc:
            friendly_message = rewrite_cuda_runtime_error(exc, "cuda:0")
            raise RuntimeError(f"Aria-AMT 转写失败:\n{friendly_message}") from exc

    def _run_transcription_subprocess(self, input_path: Path, temp_dir: Path) -> None:
        command = [
            sys.executable,
            "-m",
            "amt.run",
            "transcribe",
            ARIA_AMT_MODEL_CONFIG_NAME,
            str(self.checkpoint_path),
            "-load_path",
            str(input_path),
            "-save_dir",
            str(temp_dir),
            "-bs",
            "1",
        ]

        logger.info("Running Aria-AMT transcription: %s", " ".join(command))
        process_env = dict(os.environ)
        process_env["PYTHONIOENCODING"] = "utf-8"
        process_env["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=process_env,
        )
        self._process = process
        try:
            if self._cancelled:
                self.cancel()
            stdout, stderr = self._communicate_subprocess(process)
        finally:
            self._process = None

        if self._cancelled:
            raise InterruptedError("Aria-AMT 转写处理已取消")

        if process.returncode != 0:
            error = RuntimeError("Aria-AMT 转写失败:\n" f"{stdout}\n{stderr}")
            friendly_message = rewrite_cuda_runtime_error(error, "cuda:0")
            if friendly_message != str(error):
                raise RuntimeError(f"Aria-AMT 转写失败:\n{friendly_message}") from error
            raise error

    def _communicate_subprocess(self, process) -> tuple[str, str]:
        """Drain/reap the subprocess; enforce the cancellation deadline off the GUI thread."""

        cancel_deadline = None
        while True:
            try:
                return process.communicate(timeout=0.1)
            except subprocess.TimeoutExpired:
                if not self._cancelled:
                    continue

                if cancel_deadline is None:
                    cancel_deadline = time.monotonic() + 5.0
                if time.monotonic() < cancel_deadline:
                    continue

                logger.warning("Aria-AMT 子进程未在 5 秒内退出，强制终止")
                try:
                    process.kill()
                except ProcessLookupError:
                    logger.info("Aria-AMT 子进程已在强制终止前退出")

                try:
                    return process.communicate(timeout=5.0)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("Aria-AMT 子进程在强制终止后仍未退出") from exc

    def transcribe(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        if self._cancelled:
            raise InterruptedError("Aria-AMT 转写处理已取消")

        input_path = Path(audio_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            raise RuntimeError(self.get_unavailable_reason())
        if not self.is_model_available():
            raise RuntimeError(
                "Aria-AMT 模型权重缺失或身份校验失败。"
                f"期望大小 {ARIA_AMT_CHECKPOINT_SIZE} bytes，"
                f"SHA-256 {ARIA_AMT_CHECKPOINT_SHA256}。请执行: "
                "python download_aria_amt_model.py"
            )

        with tempfile.TemporaryDirectory(
            prefix=".aria_amt_",
            dir=out_path.parent,
        ) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            if progress_callback:
                progress_callback(0.05, self._pt("progress.loading_aria_amt"))

            if platform.system() == "Windows":
                self._run_transcription_windows_single_file(input_path, temp_dir, progress_callback)
            elif is_frozen_app():
                self._run_transcription_in_process(input_path, temp_dir)
            else:
                self._run_transcription_subprocess(input_path, temp_dir)

            if self._cancelled:
                raise InterruptedError("Aria-AMT 转写处理已取消")

            midi_path = self._guess_output_midi(temp_dir, input_path)
            if midi_path is None:
                raise RuntimeError(self._format_missing_output_error(out_path, temp_dir))
            publish_midi_output(midi_path, out_path, "Aria-AMT")

        if progress_callback:
            progress_callback(1.0, self._pt("progress.aria_amt_complete"))

        logger.info("Aria-AMT output: %s", out_path)
        return str(out_path)

    def cancel(self) -> None:
        """快速发出取消/terminate；等待与强杀由转写 worker 负责。"""

        self._cancelled = True
        process = self._process
        if process is not None and process.poll() is None:
            logger.info("正在终止 Aria-AMT 子进程...")
            try:
                process.terminate()
            except ProcessLookupError:
                logger.info("Aria-AMT 子进程已在终止请求前退出")
