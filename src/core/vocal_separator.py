"""Project-selected Leap XE + PolarFormer vocal/accompaniment separation."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

from download_accompaniment_model import (
    is_accompaniment_model_available,
    resolve_accompaniment_config_path,
    resolve_accompaniment_model_path,
)
from download_vocal_model import (
    is_vocal_model_available,
    resolve_vocal_config_path,
    resolve_vocal_model_path,
)
from src.i18n.translator import Translator
from src.utils.gpu_utils import (
    clear_gpu_memory,
    ensure_cuda_runtime_compatibility,
    get_device,
    is_unsupported_cuda_architecture_error,
    rewrite_cuda_runtime_error,
)
from src.utils.runtime_paths import (
    activate_audio_separator_runtime,
    get_audio_separator_model_dir,
)

logger = logging.getLogger(__name__)

LEAP_CHECKPOINT_NAME = "bs_leap_xe_voc.ckpt"
LEAP_CONFIG_NAME = "leap_xe_config_voc.yaml"
POLARFORMER_ONNX_NAME = "bs_polarformer.onnx"
POLARFORMER_CONFIG_NAME = "model_bs_polarformer_float16.yaml"

# Compatibility constants retained for callers that imported the old names.
ROFORMER_MODEL = LEAP_CHECKPOINT_NAME
ROFORMER_REQUIRED_MODELS = (LEAP_CHECKPOINT_NAME, LEAP_CONFIG_NAME)
KARAOKE_MODEL = POLARFORMER_ONNX_NAME
KARAOKE_REQUIRED_MODELS = (POLARFORMER_ONNX_NAME, POLARFORMER_CONFIG_NAME)


def _resolve_verified_model_assets(cache_dir: Path) -> tuple[Path, Path, Path, Path]:
    invalid_groups = []
    if not is_vocal_model_available(cache_dir):
        invalid_groups.append("Leap XE checkpoint/config")
    if not is_accompaniment_model_available(cache_dir):
        invalid_groups.append("PolarFormer ONNX/config")
    if invalid_groups:
        raise RuntimeError(
            "Vocal split model assets are missing or failed exact size/SHA256 "
            f"validation: {', '.join(invalid_groups)}. Run download_vocal_model.py "
            "and download_accompaniment_model.py."
        )
    return (
        resolve_vocal_model_path(cache_dir),
        resolve_vocal_config_path(cache_dir),
        resolve_accompaniment_model_path(cache_dir),
        resolve_accompaniment_config_path(cache_dir),
    )


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid model config: {path}")
    return data


def _load_stereo_audio(audio_path: str, sample_rate: int):
    import librosa
    import numpy as np

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack((audio, audio), axis=0)
    elif audio.ndim == 2 and audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.ndim != 2 or audio.shape[0] != 2:
        raise RuntimeError(f"Expected mono or stereo input, got audio shape {tuple(audio.shape)}")
    if audio.shape[1] <= 0:
        raise RuntimeError(f"Audio contains no samples: {audio_path}")
    return audio


def _resolve_torch_device(requested_device: Optional[str]):
    import torch

    device_name = str(requested_device or get_device(prefer_gpu=True)).strip().lower()
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is unavailable: {device_name}")
        ensure_cuda_runtime_compatibility(device_name)
        return torch.device(device_name)
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported Leap XE device: {device_name!r}")


def _leap_reference_forward(model, raw_audio):
    """ZFTurbo-compatible BS-Roformer inference forward for Leap XE.

    Derived from the MIT-licensed ZFTurbo Music-Source-Separation-Training
    v1.0.20 BS-RoFormer forward implementation. See THIRD_PARTY_NOTICES.md.

    audio-separator 0.44.1 carries a closely related class, but its forward
    squeezes the one-stem axis and omits both the zero-DC mask and the explicit
    iSTFT output length used by the model author's v1.0.20 implementation.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    device = raw_audio.device
    if raw_audio.ndim == 2:
        raw_audio = raw_audio.unsqueeze(1)
    if raw_audio.ndim != 3:
        raise RuntimeError(f"Leap XE expected 3D audio input, got {tuple(raw_audio.shape)}")
    channels = raw_audio.shape[1]
    if channels != model.audio_channels:
        raise RuntimeError(f"Leap XE expected {model.audio_channels} channel(s), got {channels}")

    raw_audio_length = raw_audio.shape[-1]
    batch_size, channels, _ = raw_audio.shape
    packed_audio = raw_audio.reshape(batch_size * channels, raw_audio_length)
    stft_window = model.stft_window_fn(device=device)
    if device.type == "mps":
        stft_repr = torch.stft(
            packed_audio.cpu(),
            **model.stft_kwargs,
            window=stft_window.cpu(),
            return_complex=True,
        ).to(device)
    else:
        stft_repr = torch.stft(
            packed_audio,
            **model.stft_kwargs,
            window=stft_window,
            return_complex=True,
        )
    stft_repr = torch.view_as_real(stft_repr)
    freq_bins, frame_count, complex_components = stft_repr.shape[1:]
    stft_repr = stft_repr.reshape(
        batch_size,
        channels,
        freq_bins,
        frame_count,
        complex_components,
    )
    stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(
        batch_size, freq_bins * channels, frame_count, complex_components
    )

    flattened_freq = stft_repr.shape[1]
    features = stft_repr.permute(0, 2, 1, 3).reshape(
        batch_size, frame_count, flattened_freq * complex_components
    )
    if model.use_torch_checkpoint:
        features = checkpoint(model.band_split, features, use_reentrant=False)
    else:
        features = model.band_split(features)

    skip_store = [None] * len(model.layers)
    for layer_index, transformer_block in enumerate(model.layers):
        if len(transformer_block) == 3:
            linear_transformer, time_transformer, freq_transformer = transformer_block
            feature_shape = features.shape
            features = features.reshape(feature_shape[0], -1, feature_shape[-1])
            if model.use_torch_checkpoint:
                features = checkpoint(
                    linear_transformer,
                    features,
                    use_reentrant=False,
                )
            else:
                features = linear_transformer(features)
            features = features.reshape(feature_shape)
        else:
            time_transformer, freq_transformer = transformer_block

        if model.skip_connection:
            for previous in skip_store[:layer_index]:
                features = features + previous
        features = features.permute(0, 2, 1, 3)
        time_shape = features.shape
        features = features.reshape(-1, time_shape[-2], time_shape[-1])
        if model.use_torch_checkpoint:
            features = checkpoint(
                time_transformer,
                features,
                use_reentrant=False,
            )
        else:
            features = time_transformer(features)
        features = features.reshape(time_shape)

        features = features.permute(0, 2, 1, 3)
        freq_shape = features.shape
        features = features.reshape(-1, freq_shape[-2], freq_shape[-1])
        if model.use_torch_checkpoint:
            features = checkpoint(
                freq_transformer,
                features,
                use_reentrant=False,
            )
        else:
            features = freq_transformer(features)
        features = features.reshape(freq_shape)
        if model.skip_connection:
            skip_store[layer_index] = features

    features = model.final_norm(features)
    mask = torch.stack([head(features) for head in model.mask_estimators], dim=1)
    stem_count, mask_frames, mask_components = mask.shape[1:]
    if mask_components % 2:
        raise RuntimeError(f"Leap XE mask has an invalid complex dimension: {mask_components}")
    mask = mask.reshape(batch_size, stem_count, mask_frames, mask_components // 2, 2).permute(
        0, 1, 3, 2, 4
    )

    stft_repr = stft_repr.unsqueeze(1)
    stft_repr = torch.view_as_complex(stft_repr) * torch.view_as_complex(mask)
    output_freq_bins = stft_repr.shape[2] // model.audio_channels
    stft_repr = (
        stft_repr.reshape(
            batch_size,
            stem_count,
            output_freq_bins,
            model.audio_channels,
            frame_count,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(
            batch_size * stem_count * model.audio_channels,
            output_freq_bins,
            frame_count,
        )
    )
    if bool(getattr(model, "zero_dc", True)):
        stft_repr = stft_repr.index_fill(
            1,
            torch.tensor([0], device=device),
            0.0,
        )

    if device.type == "mps":
        reconstructed = torch.istft(
            stft_repr.cpu(),
            **model.stft_kwargs,
            window=stft_window.cpu(),
            return_complex=False,
            length=raw_audio_length,
        ).to(device)
    else:
        reconstructed = torch.istft(
            stft_repr,
            **model.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=raw_audio_length,
        )
    reconstructed = reconstructed.reshape(
        batch_size,
        model.num_stems,
        model.audio_channels,
        raw_audio_length,
    )
    if reconstructed.shape[-1] != raw_audio_length:
        raise RuntimeError(
            "Leap XE iSTFT length mismatch: "
            f"expected {raw_audio_length}, got {reconstructed.shape[-1]}"
        )
    return reconstructed


def _build_leap_model(model_config: dict, checkpoint_path: Path, device):
    import torch

    activate_audio_separator_runtime()
    from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer

    class LeapReferenceBSRoformer(BSRoformer):
        def forward(self, raw_audio, target=None, return_loss_breakdown=False):
            if target is not None or return_loss_breakdown:
                raise RuntimeError("LeapReferenceBSRoformer is inference-only")
            return _leap_reference_forward(self, raw_audio)

    model = LeapReferenceBSRoformer(
        dim=model_config["dim"],
        depth=model_config["depth"],
        stereo=model_config.get("stereo", False),
        num_stems=model_config.get("num_stems", 1),
        time_transformer_depth=model_config.get("time_transformer_depth", 1),
        freq_transformer_depth=model_config.get("freq_transformer_depth", 1),
        linear_transformer_depth=model_config.get("linear_transformer_depth", 0),
        freqs_per_bands=tuple(model_config["freqs_per_bands"]),
        dim_head=model_config.get("dim_head", 64),
        heads=model_config.get("heads", 8),
        attn_dropout=model_config.get("attn_dropout", 0.0),
        ff_dropout=model_config.get("ff_dropout", 0.0),
        flash_attn=model_config.get("flash_attn", True),
        mlp_expansion_factor=model_config.get("mlp_expansion_factor", 4),
        use_torch_checkpoint=model_config.get("use_torch_checkpoint", False),
        skip_connection=model_config.get("skip_connection", False),
        zero_dc=model_config.get("zero_dc", True),
        dim_freqs_in=model_config.get("dim_freqs_in", 1025),
        stft_n_fft=model_config.get("stft_n_fft", 2048),
        stft_hop_length=model_config.get("stft_hop_length", 512),
        stft_win_length=model_config.get("stft_win_length", 2048),
        stft_normalized=model_config.get("stft_normalized", False),
        mask_estimator_depth=model_config.get("mask_estimator_depth", 2),
        multi_stft_resolution_loss_weight=model_config.get(
            "multi_stft_resolution_loss_weight", 1.0
        ),
        multi_stft_resolutions_window_sizes=tuple(
            model_config.get("multi_stft_resolutions_window_sizes", (4096, 2048, 1024, 512, 256))
        ),
        multi_stft_hop_size=model_config.get("multi_stft_hop_size", 147),
        multi_stft_normalized=model_config.get("multi_stft_normalized", False),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def _leap_chunk_starts(total_samples: int, chunk_size: int, step: int) -> list[int]:
    if total_samples <= 0 or chunk_size <= 0 or step <= 0:
        raise ValueError(
            "Leap XE chunk scheduling requires positive total_samples, chunk_size, and step"
        )
    return list(range(0, total_samples, step))


def _audio_chunk_progress_message(
    translate: Callable[..., str],
    *,
    model: str,
    role_key: str,
    done: int,
    total: int,
    chunk_size: int,
    step: int,
    sample_rate: int,
) -> str:
    """Describe overlapped long-audio inference without implying workflow stages."""
    if sample_rate <= 0:
        raise ValueError(f"Audio chunk progress requires a positive sample rate: {sample_rate}")
    if chunk_size <= 0 or step <= 0 or step > chunk_size:
        raise ValueError(
            "Audio chunk progress requires 0 < step <= chunk_size: "
            f"chunk_size={chunk_size}, step={step}"
        )

    def _seconds(sample_count: int) -> str:
        return f"{sample_count / sample_rate:.1f}".rstrip("0").rstrip(".")

    return translate(
        "progress.audio_chunk_progress",
        model=model,
        role=translate(role_key),
        done=done,
        total=total,
        chunk_seconds=_seconds(chunk_size),
        overlap_seconds=_seconds(chunk_size - step),
    )


def _audio_chunk_running_message(
    translate: Callable[..., str],
    *,
    model: str,
    role_key: str,
    current: str,
    total: int,
    model_file: str,
    engine: str,
    chunk_size: int,
    step: int,
    sample_rate: int,
) -> str:
    """Describe the chunk that is about to enter a potentially long inference call."""
    if sample_rate <= 0:
        raise ValueError(f"Audio chunk activity requires a positive sample rate: {sample_rate}")
    if chunk_size <= 0 or step <= 0 or step > chunk_size:
        raise ValueError(
            "Audio chunk activity requires 0 < step <= chunk_size: "
            f"chunk_size={chunk_size}, step={step}"
        )

    def _seconds(sample_count: int) -> str:
        return f"{sample_count / sample_rate:.1f}".rstrip("0").rstrip(".")

    return translate(
        "progress.audio_chunk_running",
        model=model,
        role=translate(role_key),
        current=current,
        total=total,
        model_file=model_file,
        engine=engine,
        chunk_seconds=_seconds(chunk_size),
        overlap_seconds=_seconds(chunk_size - step),
    )


def _run_leap_vocals_leg(
    *,
    audio_path: str,
    checkpoint_path: Path,
    config_path: Path,
    requested_device: Optional[str],
    progress_callback: Optional[Callable[[float, str], None]],
    translate: Callable[..., str],
    cancel_check: Callable[[], None],
):
    import numpy as np
    import torch

    config = _load_yaml(config_path)
    model_config = dict(config["model"])
    audio_config = config["audio"]
    inference_config = config["inference"]
    training_config = config["training"]
    sample_rate = int(audio_config.get("sample_rate", 44_100))
    mix = _load_stereo_audio(audio_path, sample_rate)
    original_length = mix.shape[1]

    configured_chunk_size = inference_config.get("chunk_size", audio_config.get("chunk_size"))
    if configured_chunk_size is None:
        raise RuntimeError("Leap XE config has no inference.chunk_size or audio.chunk_size")
    chunk_size = int(configured_chunk_size)
    num_overlap = int(inference_config.get("num_overlap", 2))
    batch_size = int(inference_config.get("batch_size", 1))
    num_stems = int(model_config.get("num_stems", 1))
    if chunk_size < 10 or num_overlap <= 0 or batch_size <= 0 or num_stems != 1:
        raise RuntimeError(
            "Invalid Leap XE inference config: "
            f"chunk_size={chunk_size}, num_overlap={num_overlap}, "
            f"batch_size={batch_size}, num_stems={num_stems}"
        )
    step = chunk_size // num_overlap
    if step <= 0:
        raise RuntimeError(
            f"Leap XE overlap produces a zero step: chunk={chunk_size}, overlap={num_overlap}"
        )
    fade_size = chunk_size // 10
    border = chunk_size - step

    mix_tensor = torch.from_numpy(mix).float()
    reflected_padding = original_length > 2 * border and border > 0
    if reflected_padding:
        mix_tensor = torch.nn.functional.pad(
            mix_tensor,
            (border, border),
            mode="reflect",
        )
    starts = _leap_chunk_starts(mix_tensor.shape[-1], chunk_size, step)
    result = torch.zeros((num_stems,) + tuple(mix_tensor.shape), dtype=torch.float32)
    counter = torch.zeros_like(result)
    window = torch.ones(chunk_size, dtype=torch.float32)
    window[:fade_size] = torch.linspace(0.0, 1.0, fade_size)
    window[-fade_size:] = torch.linspace(1.0, 0.0, fade_size)
    device = _resolve_torch_device(requested_device)
    use_amp = bool(training_config.get("use_amp", True)) and device.type == "cuda"
    engine = f"PyTorch · {device} · {'AMP/FP16' if use_amp else 'FP32'} · batch={batch_size}"
    model = None

    try:
        if progress_callback is not None:
            progress_callback(
                0.01,
                translate(
                    "progress.separation_model_loading",
                    model="Leap XE",
                    role=translate("progress.audio_chunk_role_vocals"),
                    model_file=checkpoint_path.name,
                    engine=engine,
                ),
            )
        model = _build_leap_model(model_config, checkpoint_path, device)
        logger.info(
            "Leap XE vocals leg loaded on %s: chunk=%s step=%s chunks=%s batch=%s amp=%s",
            device,
            chunk_size,
            step,
            len(starts),
            batch_size,
            use_amp,
        )
        batch_data = []
        batch_locations = []
        processed_chunks = 0
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
            for index, start in enumerate(starts):
                cancel_check()
                part = mix_tensor[:, start : start + chunk_size].to(device)
                segment_length = part.shape[-1]
                pad_mode = "reflect" if segment_length > chunk_size // 2 else "constant"
                part = torch.nn.functional.pad(
                    part,
                    (0, chunk_size - segment_length),
                    mode=pad_mode,
                    value=0.0,
                )
                batch_data.append(part)
                batch_locations.append((start, segment_length))

                if len(batch_data) < batch_size and index + 1 < len(starts):
                    continue

                if progress_callback is not None:
                    first_chunk = processed_chunks + 1
                    last_chunk = processed_chunks + len(batch_locations)
                    current = (
                        str(first_chunk)
                        if first_chunk == last_chunk
                        else f"{first_chunk}-{last_chunk}"
                    )
                    progress_callback(
                        0.03 + 0.47 * (processed_chunks / len(starts)),
                        _audio_chunk_running_message(
                            translate,
                            model="Leap XE",
                            role_key="progress.audio_chunk_role_vocals",
                            current=current,
                            total=len(starts),
                            model_file=checkpoint_path.name,
                            engine=engine,
                            chunk_size=chunk_size,
                            step=step,
                            sample_rate=sample_rate,
                        ),
                    )
                prediction = model(torch.stack(batch_data, dim=0)).detach().cpu()
                if prediction.ndim == 3 and tuple(prediction.shape[:2]) == (
                    len(batch_locations),
                    mix_tensor.shape[0],
                ):
                    prediction = prediction.unsqueeze(1)
                expected_prefix = (len(batch_locations), num_stems, mix_tensor.shape[0])
                if prediction.ndim != 4 or tuple(prediction.shape[:3]) != expected_prefix:
                    raise RuntimeError(
                        "Leap XE returned an invalid batch shape: "
                        f"expected ({expected_prefix[0]}, {expected_prefix[1]}, "
                        f"{expected_prefix[2]}, samples), got {tuple(prediction.shape)}"
                    )
                batch_window = window.clone()
                if start == 0:
                    batch_window[:fade_size] = 1.0
                if start + step >= mix_tensor.shape[-1]:
                    batch_window[-fade_size:] = 1.0

                for batch_index, (segment_start, segment_length) in enumerate(batch_locations):
                    result[..., segment_start : segment_start + segment_length] += (
                        prediction[batch_index, ..., :segment_length]
                        * batch_window[:segment_length]
                    )
                    counter[..., segment_start : segment_start + segment_length] += batch_window[
                        :segment_length
                    ]
                processed_chunks += len(batch_locations)
                batch_data.clear()
                batch_locations.clear()

        estimated_sources = torch.nan_to_num(result / counter, nan=0.0)
        if reflected_padding:
            estimated_sources = estimated_sources[..., border:-border]
        if estimated_sources.shape[-1] != original_length:
            raise RuntimeError(
                "Leap XE reconstruction length mismatch: "
                f"expected {original_length}, got {estimated_sources.shape[-1]}"
            )
        vocals = estimated_sources[0].numpy()
        np.nan_to_num(vocals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return vocals, sample_rate
    finally:
        if model is not None:
            del model
        clear_gpu_memory()


def _prepare_polar_stft(audio, stft_kwargs: dict, win_length: int):
    import torch

    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    raw_audio = audio_tensor.reshape(-1, audio_tensor.shape[-1])
    stft_window = torch.hann_window(win_length)
    stft = torch.stft(
        raw_audio,
        **stft_kwargs,
        window=stft_window,
        return_complex=True,
    )
    stft_real = torch.view_as_real(stft)
    channels, frequencies, frames, complex_parts = stft_real.shape
    stft_repr = (
        stft_real.reshape(1, channels, frequencies, frames, complex_parts)
        .permute(0, 2, 1, 3, 4)
        .reshape(1, frequencies * channels, frames, complex_parts)
    )
    features = stft_repr.permute(0, 2, 1, 3).reshape(
        1, frames, frequencies * channels * complex_parts
    )
    return features, stft_repr, stft_window, raw_audio.shape[-1]


def _reconstruct_polar_audio(
    stft_repr,
    mask,
    stft_kwargs: dict,
    stft_window,
    audio_channels: int,
    raw_audio_length: int,
):
    import torch

    mask_tensor = torch.from_numpy(mask)
    stft_complex = torch.view_as_complex(stft_repr.unsqueeze(1).contiguous())
    mask_complex = torch.view_as_complex(mask_tensor.contiguous())
    masked = stft_complex * mask_complex
    batch, stems, combined_frequencies, frames = masked.shape
    if combined_frequencies % audio_channels != 0:
        raise RuntimeError(
            "PolarFormer mask has an invalid frequency/channel dimension: "
            f"{combined_frequencies}"
        )
    frequencies = combined_frequencies // audio_channels
    masked = (
        masked.reshape(batch, stems, frequencies, audio_channels, frames)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch * stems * audio_channels, frequencies, frames)
    )
    masked[:, 0, :] = 0.0
    reconstructed = torch.istft(
        masked,
        **stft_kwargs,
        window=stft_window,
        return_complex=False,
        length=raw_audio_length,
    )
    return reconstructed.reshape(batch, stems, audio_channels, raw_audio_length)


def _resolve_onnx_providers(requested_device: Optional[str], ort_module) -> list:
    available = set(ort_module.get_available_providers())
    device_name = str(requested_device or "").strip().lower()
    if not device_name:
        device_name = "cuda:0" if "CUDAExecutionProvider" in available else "cpu"

    if device_name == "cpu":
        if "CPUExecutionProvider" not in available:
            raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
        return ["CPUExecutionProvider"]
    if device_name.startswith("cuda"):
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDA was requested for PolarFormer but ONNX Runtime has no "
                "CUDAExecutionProvider"
            )
        try:
            device_id = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {device_name!r}") from exc
        return [("CUDAExecutionProvider", {"device_id": device_id})]
    raise ValueError(f"Unsupported PolarFormer ONNX device: {device_name!r}")


def _run_polarformer_accompaniment_leg(
    *,
    audio_path: str,
    onnx_path: Path,
    config_path: Path,
    requested_device: Optional[str],
    progress_callback: Optional[Callable[[float, str], None]],
    translate: Callable[..., str],
    cancel_check: Callable[[], None],
    active_run_options_callback: Optional[Callable[[Optional[object]], None]] = None,
):
    import numpy as np

    activate_audio_separator_runtime()
    import onnxruntime as ort

    config = _load_yaml(config_path)
    sample_rate = int(config["audio"]["sample_rate"])
    mix = _load_stereo_audio(audio_path, sample_rate)
    total_samples = mix.shape[1]
    model_config = config["model"]
    inference_config = config["inference"]
    stft_kwargs = {
        "n_fft": int(model_config["stft_n_fft"]),
        "hop_length": int(model_config["stft_hop_length"]),
        "win_length": int(model_config["stft_win_length"]),
        "normalized": bool(model_config.get("stft_normalized", False)),
    }
    audio_channels = 2 if model_config.get("stereo", False) else 1
    if audio_channels != 2:
        raise RuntimeError(
            f"PolarFormer config expected {audio_channels} audio channel(s), not stereo"
        )
    chunk_size = int(inference_config.get("chunk_size", 882_000))
    num_overlap = int(inference_config.get("num_overlap", 2))
    if chunk_size <= 0 or num_overlap <= 0:
        raise RuntimeError(
            f"Invalid PolarFormer chunk settings: chunk_size={chunk_size}, overlap={num_overlap}"
        )
    step = max(1, chunk_size // num_overlap)
    positions = list(range(0, total_samples, step))
    providers = _resolve_onnx_providers(requested_device, ort)
    requested_provider_names = " + ".join(
        str(provider[0] if isinstance(provider, tuple) else provider) for provider in providers
    )
    requested_engine = f"ONNX Runtime · {requested_provider_names}"
    if progress_callback is not None:
        progress_callback(
            0.51,
            translate(
                "progress.separation_model_loading",
                model="PolarFormer",
                role=translate("progress.audio_chunk_role_accompaniment"),
                model_file=onnx_path.name,
                engine=requested_engine,
            ),
        )
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "stft_features":
        raise RuntimeError(
            "Unexpected PolarFormer ONNX input contract: " f"{[item.name for item in inputs]}"
        )
    actual_engine = f"ONNX Runtime · {' + '.join(session.get_providers())}"
    logger.info(
        "PolarFormer accompaniment leg loaded with providers=%s chunks=%s",
        session.get_providers(),
        len(positions),
    )

    vocals = np.zeros((2, total_samples), dtype=np.float32)
    counter = np.zeros(total_samples, dtype=np.float32)
    for index, start in enumerate(positions):
        cancel_check()
        if progress_callback is not None:
            progress_callback(
                0.52 + 0.43 * (index / len(positions)),
                _audio_chunk_running_message(
                    translate,
                    model="PolarFormer",
                    role_key="progress.audio_chunk_role_accompaniment",
                    current=str(index + 1),
                    total=len(positions),
                    model_file=onnx_path.name,
                    engine=actual_engine,
                    chunk_size=chunk_size,
                    step=step,
                    sample_rate=sample_rate,
                ),
            )
        end = min(start + chunk_size, total_samples)
        chunk = mix[:, start:end]
        if chunk.shape[1] < chunk_size:
            chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[1])))
        features, stft_repr, stft_window, raw_audio_length = _prepare_polar_stft(
            chunk,
            stft_kwargs,
            stft_kwargs["win_length"],
        )
        run_options = ort.RunOptions()
        if active_run_options_callback is not None:
            active_run_options_callback(run_options)
        try:
            cancel_check()
            try:
                mask = session.run(
                    None,
                    {"stft_features": features.numpy()},
                    run_options=run_options,
                )[0]
            except Exception:
                # RunOptions.terminate makes ONNX Runtime raise its own non-RuntimeError
                # exception. Convert it to cancellation only when the application flag
                # confirms a real user request; otherwise preserve the inference error.
                cancel_check()
                raise
            cancel_check()
        finally:
            if active_run_options_callback is not None:
                active_run_options_callback(None)
        prediction = _reconstruct_polar_audio(
            stft_repr,
            mask,
            stft_kwargs,
            stft_window,
            audio_channels,
            raw_audio_length,
        )[0, 0].numpy()
        actual_length = end - start
        vocals[:, start:end] += prediction[:, :actual_length]
        counter[start:end] += 1.0

    if np.any(counter <= 0):
        raise RuntimeError("PolarFormer did not cover every output sample")
    vocals /= counter[np.newaxis, :]
    accompaniment = mix - vocals
    np.nan_to_num(accompaniment, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return accompaniment, sample_rate


def _write_and_validate_wav(path: Path, audio, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.T, sample_rate, subtype="PCM_16")
    info = sf.info(str(path))
    if info.frames <= 0 or info.channels <= 0 or info.samplerate != sample_rate:
        raise RuntimeError(f"Separator did not create a valid WAV file: {path}")


class VocalSeparator:
    """Run independent Leap XE vocals and PolarFormer accompaniment legs."""

    def __init__(
        self,
        language: str = Translator.DEFAULT_LANGUAGE,
        *,
        primary_device: Optional[str] = None,
        accompaniment_device: Optional[str] = None,
        karaoke_device: Optional[str] = None,
    ):
        if accompaniment_device and karaoke_device and accompaniment_device != karaoke_device:
            raise ValueError(
                "accompaniment_device and legacy karaoke_device cannot select different devices"
            )
        self._cancelled = False
        self._cancel_check: Optional[Callable[[], bool]] = None
        self._cancel_lock = threading.Lock()
        self._active_onnx_run_options: Optional[object] = None
        self._translator = Translator(language)
        self.primary_device = primary_device
        self.accompaniment_device = accompaniment_device or karaoke_device

    def _pt(self, key: str, **kwargs) -> str:
        return self._translator.t(key, **kwargs)

    def set_cancel_check(self, fn: Callable[[], bool]) -> None:
        self._cancel_check = fn

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("用户取消了处理")
        if self._cancel_check and self._cancel_check():
            raise InterruptedError("用户取消了处理")

    def _set_active_onnx_run_options(self, run_options: Optional[object]) -> None:
        """Publish the one ONNX run that cancel() may safely terminate."""
        with self._cancel_lock:
            self._active_onnx_run_options = run_options
            if run_options is not None and self._cancelled:
                run_options.terminate = True

    def cancel(self) -> None:
        with self._cancel_lock:
            self._cancelled = True
            run_options = self._active_onnx_run_options
            if run_options is not None:
                run_options.terminate = True

    @staticmethod
    def is_available() -> bool:
        try:
            activate_audio_separator_runtime()
            import librosa  # noqa: F401
            import onnxruntime  # noqa: F401
            import soundfile  # noqa: F401
            import torch  # noqa: F401
            import yaml  # noqa: F401
            from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import (  # noqa: F401,E501
                BSRoformer,
            )

            return True
        except (ImportError, OSError):
            return False

    @staticmethod
    def is_model_available() -> bool:
        cache_dir = get_audio_separator_model_dir()
        return is_vocal_model_available(cache_dir) and is_accompaniment_model_available(cache_dir)

    @staticmethod
    def _get_model_cache_dir() -> str:
        return str(get_audio_separator_model_dir())

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, str]:
        """Create one real Leap XE vocal WAV and one real PolarFormer accompaniment WAV."""
        self._check_cancelled()
        input_path = Path(audio_path)
        if not input_path.is_file() or input_path.stat().st_size <= 0:
            raise FileNotFoundError(f"Input audio does not exist or is empty: {input_path}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        vocals_path = output_path / f"{stem}_vocals.wav"
        accompaniment_path = output_path / f"{stem}_accompaniment.wav"
        cache_dir = get_audio_separator_model_dir()
        leap_checkpoint, leap_config, polar_onnx, polar_config = _resolve_verified_model_assets(
            cache_dir
        )

        if progress_callback is not None:
            progress_callback(0.0, self._pt("progress.loading_vocal_separator"))

        try:
            logger.info(
                "Starting project-selected vocal split: Leap XE=%s (%s), " "PolarFormer=%s (%s)",
                leap_checkpoint,
                self.primary_device or "auto",
                polar_onnx,
                self.accompaniment_device or "auto",
            )
            vocals, vocals_sample_rate = _run_leap_vocals_leg(
                audio_path=str(input_path),
                checkpoint_path=leap_checkpoint,
                config_path=leap_config,
                requested_device=self.primary_device,
                progress_callback=progress_callback,
                translate=self._pt,
                cancel_check=self._check_cancelled,
            )
            self._check_cancelled()
            if progress_callback is not None:
                progress_callback(
                    0.505,
                    self._pt(
                        "progress.separation_model_switching",
                        from_model="Leap XE",
                        from_role=self._pt("progress.audio_chunk_role_vocals"),
                        to_model="PolarFormer",
                        to_role=self._pt("progress.audio_chunk_role_accompaniment"),
                        model_file=polar_onnx.name,
                    ),
                )
            _write_and_validate_wav(vocals_path, vocals, vocals_sample_rate)

            accompaniment, accompaniment_sample_rate = _run_polarformer_accompaniment_leg(
                audio_path=str(input_path),
                onnx_path=polar_onnx,
                config_path=polar_config,
                requested_device=self.accompaniment_device,
                progress_callback=progress_callback,
                translate=self._pt,
                cancel_check=self._check_cancelled,
                active_run_options_callback=self._set_active_onnx_run_options,
            )
            self._check_cancelled()
            if progress_callback is not None:
                progress_callback(0.97, self._pt("progress.saving_separation_results"))
            _write_and_validate_wav(
                accompaniment_path,
                accompaniment,
                accompaniment_sample_rate,
            )
        except InterruptedError:
            raise
        except RuntimeError as exc:
            if is_unsupported_cuda_architecture_error(exc):
                friendly = rewrite_cuda_runtime_error(exc)
                logger.error("Vocal separation CUDA architecture is incompatible\n%s", friendly)
                raise RuntimeError(f"人声分离失败（GPU 不兼容）:\n{friendly}") from exc
            logger.error("Vocal separation failed: %s", exc)
            raise RuntimeError(f"人声分离失败: {exc}") from exc
        except Exception as exc:
            logger.exception("Vocal separation failed")
            raise RuntimeError(f"人声分离失败: {exc}") from exc
        finally:
            clear_gpu_memory()

        if progress_callback is not None:
            progress_callback(1.0, self._pt("progress.separation_complete"))

        return {
            "vocals": str(vocals_path),
            "accompaniment": str(accompaniment_path),
            "no_vocals": str(accompaniment_path),
        }
