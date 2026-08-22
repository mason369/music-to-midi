"""Aria-AMT piano transcription wrapper."""

from __future__ import annotations

import importlib.util
import json
import logging
import math
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
from src.models.data_models import Config
from src.core.muscriptor_result_assets import read_midi_roll_notes
from src.core.transcription_stream import snapshot_event
from src.utils.audio_utils import load_audio_tensor
from src.utils.artifact_identity import validate_file_identity
from src.utils.gpu_utils import (
    ensure_accelerator_runtime_compatibility,
    ensure_cuda_runtime_compatibility,
    ensure_module_on_device,
    get_accelerator_type,
    get_device,
    rewrite_cuda_runtime_error,
)
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
ARIA_AMT_TOKEN_VOCAB_SIZE = 3419
ARIA_AMT_ONSET_VELOCITY_TOKEN_START = 392
ARIA_AMT_ONSET_VELOCITY_TOKEN_END = 3418


def _setup_aria_decoder_cache_on_xpu(
    decoder,
    *,
    device: str,
    batch_size: int,
    max_seq_len: int,
    dtype,
    max_audio_len: int = 1500,
) -> None:
    """Mirror the pinned decoder cache layout without its hard-coded CUDA move."""

    import torch
    from amt.inference.model import KVCache

    target = torch.device(device)
    if target.type != "xpu":
        raise ValueError(f"Aria-AMT XPU cache requested for non-XPU device: {target}")
    decoder.causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)).to(
        target
    )
    for block in decoder.blocks:
        block.attn.kv_cache = KVCache(
            max_batch_size=batch_size,
            max_seq_length=max_seq_len,
            n_heads=decoder.n_head,
            head_dim=decoder.n_state // decoder.n_head,
            dtype=dtype,
        ).to(target)
        block.cross_attn.kv_cache = KVCache(
            max_batch_size=batch_size,
            max_seq_length=max_audio_len,
            n_heads=decoder.n_head,
            head_dim=decoder.n_state // decoder.n_head,
            dtype=dtype,
        ).to(target)


def _recalculate_aria_token_ids_on_device(
    logits,
    token_ids,
    *,
    column_indices,
    interval_mask,
):
    """Mirror the pinned Aria-AMT token refinement on the logits device.

    The fixed upstream TorchScript helper constructs its static mask on the
    literal CUDA backend.  Supplying the mask explicitly preserves the exact
    pinned math while keeping every tensor on the already validated device.
    """

    import torch

    if logits.ndim != 2 or logits.shape[1] != ARIA_AMT_TOKEN_VOCAB_SIZE:
        raise RuntimeError(
            "Aria-AMT token logits violate the pinned vocabulary contract: "
            f"expected=(*, {ARIA_AMT_TOKEN_VOCAB_SIZE}), actual={tuple(logits.shape)}"
        )
    if token_ids.ndim != 1 or token_ids.shape[0] != logits.shape[0]:
        raise RuntimeError(
            "Aria-AMT token IDs violate the pinned batch contract: "
            f"logits={tuple(logits.shape)}, token_ids={tuple(token_ids.shape)}"
        )
    expected_mask_shape = (1, ARIA_AMT_TOKEN_VOCAB_SIZE)
    if tuple(column_indices.shape) != expected_mask_shape:
        raise RuntimeError(
            "Aria-AMT static token columns have the wrong shape: "
            f"expected={expected_mask_shape}, actual={tuple(column_indices.shape)}"
        )
    if tuple(interval_mask.shape) != expected_mask_shape or interval_mask.dtype != torch.bool:
        raise RuntimeError(
            "Aria-AMT static interval mask violates the pinned contract: "
            f"shape={tuple(interval_mask.shape)}, dtype={interval_mask.dtype}"
        )

    target = logits.device
    for label, tensor in (
        ("token_ids", token_ids),
        ("column_indices", column_indices),
        ("interval_mask", interval_mask),
    ):
        if tensor.device != target:
            raise RuntimeError(f"Aria-AMT {label} is on {tensor.device}, expected {target}")

    probabilities = torch.softmax(logits, dim=-1)
    expanded_token_ids = token_ids.unsqueeze(1)
    beam_mask = (column_indices <= expanded_token_ids + 2) & (
        column_indices >= expanded_token_ids - 2
    )
    original_token_mask = torch.zeros_like(probabilities, dtype=torch.bool)
    original_token_mask.scatter_(1, expanded_token_ids, 1)
    combined_mask = (interval_mask & beam_mask) | original_token_mask
    probabilities[~combined_mask] = 0
    weighted_indices = probabilities * column_indices.float()
    return (weighted_indices.sum(dim=1) / (probabilities.sum(dim=1) + 1e-9)).round().to(torch.long)


def _process_aria_segments_on_device(
    *,
    tasks: list,
    model,
    audio_transform,
    tokenizer,
    transcribe_module,
    device: str,
):
    """Pinned Aria-AMT segment decoder with explicit non-CUDA placement.

    Aria-AMT's fixed upstream revision hard-codes ``Tensor.cuda()`` inside
    ``process_segments``.  The algorithm below mirrors that function and calls
    the same upstream decoding helpers, while every constructed tensor is
    placed on the one validated XPU device.
    """

    import torch

    target = torch.device(device)
    if target.type != "xpu":
        raise ValueError(f"explicit Aria segment device must be XPU, got {target}")

    with torch.inference_mode():
        token_column_indices = torch.arange(
            ARIA_AMT_TOKEN_VOCAB_SIZE,
            device=target,
        ).unsqueeze(0)
        token_interval_mask = (token_column_indices >= ARIA_AMT_ONSET_VELOCITY_TOKEN_START) & (
            token_column_indices <= ARIA_AMT_ONSET_VELOCITY_TOKEN_END
        )
        log_mels = audio_transform.log_mel(
            torch.stack([audio_segment.to(target) for (audio_segment, _prefix), _ in tasks])
        )
        audio_features = model.encoder(xa=log_mels)

        raw_prefixes = [prefix for (_audio_segment, prefix), _ in tasks]
        prefix_lens = torch.tensor(
            [len(prefix) for prefix in raw_prefixes],
            dtype=torch.int,
            device=target,
        )
        min_prefix_len = int(prefix_lens.min().item())
        prefixes = [
            tokenizer.trunc_seq(prefix, transcribe_module.MAX_BLOCK_LEN) for prefix in raw_prefixes
        ]
        sequence = torch.stack(
            [torch.tensor(tokenizer.encode(prefix), device=target) for prefix in prefixes]
        )
        eos_indices = torch.full(
            (len(prefixes),),
            transcribe_module.MAX_BLOCK_LEN,
            dtype=torch.int,
            device=target,
        )

        final_index = min_prefix_len
        for index in range(min_prefix_len, transcribe_module.MAX_BLOCK_LEN - 1):
            final_index = index
            if index == min_prefix_len:
                logits, next_token_ids = transcribe_module.prefill(
                    model,
                    x=sequence[:, :index],
                    xa=audio_features,
                    x_input_pos=torch.arange(0, index, device=target),
                    xa_input_pos=torch.arange(
                        0,
                        audio_features.shape[1],
                        device=target,
                    ),
                )
            else:
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    logits, next_token_ids = transcribe_module.decode_token(
                        model,
                        x=sequence[:, index - 1 : index],
                        xa=audio_features,
                        x_input_pos=torch.tensor(
                            [index - 1],
                            device=target,
                            dtype=torch.int,
                        ),
                        xa_input_pos=torch.tensor([], device=target, dtype=torch.int),
                    )
            if bool(torch.isnan(logits).any().item()):
                raise RuntimeError("Aria-AMT XPU inference produced NaN logits")

            logits[:, 389] *= 1.05
            next_token_ids = torch.argmax(logits, dim=-1)
            next_token_ids = _recalculate_aria_token_ids_on_device(
                logits,
                next_token_ids,
                column_indices=token_column_indices,
                interval_mask=token_interval_mask,
            )
            transcribe_module.update_seq_end_idxs_(
                next_tok_ids=next_token_ids,
                seq=sequence,
                eos_idxs=eos_indices,
                prefix_lens=prefix_lens,
                idx=index,
            )
            if all(value <= index for value in eos_indices.tolist()):
                break

        if not all(value <= final_index for value in eos_indices.tolist()):
            logger.warning("Context length overflow when transcribing Aria-AMT segment(s)")

        return [
            tokenizer.decode(sequence[row, : int(eos_indices[row].item()) + 1].tolist())
            for row in range(sequence.shape[0])
        ]


def get_aria_amt_runtime_unavailable_reason() -> str:
    """Return why the installed Aria-AMT source is not the pinned revision."""

    try:
        if importlib.util.find_spec("amt.run") is None:
            return "Aria-AMT 未安装。固定源码版本：" f"{ARIA_AMT_SOURCE_REQUIREMENT}"
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return f"Aria-AMT 模块不可用: {exc}"

    try:
        distribution = metadata.distribution(ARIA_AMT_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return (
            "Aria-AMT 安装信息缺失，无法验证源码版本。固定源码版本："
            f"{ARIA_AMT_SOURCE_REQUIREMENT}"
        )

    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        return (
            "Aria-AMT 缺少安装来源信息，无法验证源码提交。固定源码版本："
            f"{ARIA_AMT_SOURCE_REQUIREMENT}"
        )

    try:
        direct_url = json.loads(direct_url_text)
    except (TypeError, json.JSONDecodeError) as exc:
        return f"Aria-AMT 安装来源信息无效，无法验证源码提交：{exc}"

    vcs_info = direct_url.get("vcs_info") or {}
    if vcs_info:
        if vcs_info.get("commit_id") == ARIA_AMT_SOURCE_REVISION:
            return ""
    elif direct_url.get("url") == ARIA_AMT_SOURCE_ARCHIVE_URL:
        return ""

    observed_commit = vcs_info.get("commit_id") or "unknown"
    observed_url = direct_url.get("url") or "unknown"
    return (
        "Aria-AMT 源码版本不匹配："
        f"需要提交 {ARIA_AMT_SOURCE_REVISION}，"
        f"当前提交 {observed_commit}，来源 {observed_url}。"
        f"固定源码版本：{ARIA_AMT_SOURCE_REQUIREMENT}"
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
        self,
        checkpoint_path: Optional[Path] = None,
        language: str = Translator.DEFAULT_LANGUAGE,
        config: Optional[Config] = None,
    ):
        self._config_explicit = config is not None
        self.config = config or Config(language=language)
        if checkpoint_path is None:
            checkpoint_path = self.default_checkpoint_path()
        self.checkpoint_path = Path(checkpoint_path)
        self._cancelled = False
        self._process: Optional[subprocess.Popen[str]] = None
        self._translator = Translator(language)

    def _resolve_runtime_device(self) -> str:
        # Preserve the historical direct-construction contract used by callers
        # that do not provide Config: Aria-AMT was CUDA-only and selected cuda:0.
        # The application pipeline always provides Config and therefore uses
        # the shared CUDA/XPU device resolver.
        device = (
            get_device(self.config.use_gpu, self.config.gpu_device)
            if self._config_explicit
            else "cuda:0"
        )
        if not device.startswith(("cuda", "xpu")):
            raise RuntimeError(
                "Aria-AMT 需要 CUDA 或 Intel XPU 加速器；"
                f"当前显式设备为 {device}，不会静默切换运行时"
            )
        if device.startswith("cuda"):
            ensure_cuda_runtime_compatibility(device)
        else:
            ensure_accelerator_runtime_compatibility(device)
        return device

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

        waveform, original_sample_rate = load_audio_tensor(input_path)
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
        device = "cuda:0"
        try:
            device = self._resolve_runtime_device()
            import torch

            from amt.audio import AudioTransform
            from amt.config import load_config
            from amt.inference import transcribe as transcribe_module

            model, tokenizer = self._load_aria_model()
            cache_dtype = torch.float
            if device.startswith("cuda"):
                checker = getattr(torch.cuda, "is_bf16_supported", None)
                if callable(checker) and checker():
                    cache_dtype = torch.bfloat16
            if device.startswith("cuda"):
                model.decoder.setup_cache(
                    batch_size=1,
                    max_seq_len=transcribe_module.MAX_BLOCK_LEN,
                    dtype=cache_dtype,
                )
                model.cuda()
            else:
                model.to(device)
                _setup_aria_decoder_cache_on_xpu(
                    model.decoder,
                    device=device,
                    batch_size=1,
                    max_seq_len=transcribe_module.MAX_BLOCK_LEN,
                    dtype=cache_dtype,
                )
            model.eval()
            audio_transform = AudioTransform()
            if device.startswith("cuda"):
                audio_transform = audio_transform.cuda()
            else:
                audio_transform = audio_transform.to(device)
                ensure_module_on_device(model, device, "Aria-AMT model")
                ensure_module_on_device(audio_transform, device, "Aria-AMT audio transform")
            audio_config = load_config()["audio"]
            event_callback = getattr(self, "_event_callback", None)
            duration_seconds = 0.0
            total_windows = 0
            if event_callback is not None:
                audio_info = torchaudio.info(str(input_path))
                duration_seconds = audio_info.num_frames / float(audio_info.sample_rate)
                chunk_seconds = float(audio_config["chunk_len"])
                stride_seconds = chunk_seconds / float(transcribe_module.STRIDE_FACTOR)
                if duration_seconds <= chunk_seconds:
                    total_windows = 1
                else:
                    stride_count = math.ceil(duration_seconds / stride_seconds)
                    total_windows = max(
                        1,
                        stride_count - int(transcribe_module.STRIDE_FACTOR) + 2,
                    )

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
                segment_tasks = [((audio_segment, sequence), 0)]
                if device.startswith("xpu"):
                    (sequence,) = _process_aria_segments_on_device(
                        tasks=segment_tasks,
                        model=model,
                        audio_transform=audio_transform,
                        tokenizer=tokenizer,
                        transcribe_module=transcribe_module,
                        device=device,
                    )
                else:
                    (sequence,) = transcribe_module.process_segments(
                        tasks=segment_tasks,
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

                if event_callback is not None:
                    completed = index + 1
                    is_final = completed >= total_windows
                    frontier = (
                        duration_seconds
                        if is_final
                        else min(
                            duration_seconds,
                            completed * transcribe_module.CHUNK_LEN_MS / 1000.0,
                        )
                    )
                    preview_notes = []
                    preview_path = temp_dir / ".aria-stream-preview.mid"
                    try:
                        self._save_token_sequence_as_midi(
                            tokenizer,
                            concat_sequence,
                            preview_path,
                        )
                    except RuntimeError as exc:
                        if "onset token" not in str(exc):
                            raise
                    else:
                        preview_notes = [
                            {
                                "instrument": note.instrument,
                                "program": note.program,
                                "is_drum": note.is_drum,
                                "pitch": note.pitch,
                                "velocity": note.velocity,
                                "start": note.start,
                                "end": note.end,
                            }
                            for note in read_midi_roll_notes(preview_path)
                            if note.end <= frontier + 1e-6
                        ]
                    event_callback(
                        snapshot_event(
                            backend="Aria-AMT",
                            completed=completed,
                            total=total_windows,
                            frontier_seconds=frontier,
                            duration_seconds=duration_seconds,
                            notes=preview_notes,
                        )
                    )

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
            friendly_message = rewrite_cuda_runtime_error(exc, device)
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
                f"SHA-256 {ARIA_AMT_CHECKPOINT_SHA256}。准备命令："
                "python download_aria_amt_model.py"
            )

        with tempfile.TemporaryDirectory(
            prefix=".aria_amt_",
            dir=out_path.parent,
        ) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            if progress_callback:
                progress_callback(0.05, self._pt("progress.loading_aria_amt"))

            if (
                platform.system() == "Windows"
                or get_accelerator_type() == "xpu"
                or getattr(self, "_event_callback", None) is not None
            ):
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

    def set_event_callback(self, callback: Optional[Callable[[dict], None]]) -> None:
        self._event_callback = callback
