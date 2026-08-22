"""Memory-bounded loading for the pinned MuScriptor safetensors checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path

from src.models.data_models import MuscriptorProcessingChain

logger = logging.getLogger(__name__)


def _safetensors_open_kwargs(target_device) -> dict[str, str]:
    """Select the non-mmap reader required by native XPU on Windows.

    Repeated reads of the 5.1 GiB MuScriptor Large checkpoint exposed an
    intermittent native access violation while safetensors' mmap backend was
    materializing a CPU tensor. Safetensors 0.8.0 provides the official
    ``pread`` backend for platforms where mapping the complete file is unsafe.
    It keeps the existing one-tensor-at-a-time memory bound without retaining
    a mapped PyTorch storage.
    """

    if getattr(target_device, "type", None) == "xpu":
        print(
            f"Loading MuScriptor checkpoint with safetensors backend=pread on {target_device}",
            flush=True,
        )
        return {"backend": "pread"}
    return {}


def _build_telknet_boundary_recovery_model_class(upstream, torch):
    """Create the project-owned Issue #74 recovery subclass.

    The pinned ``muscriptor`` package remains byte-for-byte upstream v0.3.0.
    Only the instantiated model class changes, so source identity checks keep
    proving the official dependency while this downstream product fix remains
    explicit and independently testable.
    """

    import warnings

    from muscriptor.events import ChunkBoundary, OpenNoteTracker, ProgressEvent

    segment_duration = float(upstream._SEGMENT_DURATION)

    class TelkNetBoundaryRecoveryTranscriptionModel(upstream.TranscriptionModel):
        def transcribe(
            self,
            audio,
            use_sampling: bool = False,
            temperature: float = 1.0,
            cfg_coef: float = 1.0,
            instruments: list[str] | None = None,
            batch_size: int | None = None,
            no_eos_is_ok: bool = True,
            beam_size: int = 1,
            prelude_forcing: bool = True,
        ):
            if getattr(self, "_telknet_transcription_active", False):
                raise RuntimeError(
                    "Concurrent MuScriptor transcription on one model instance is unsupported"
                )
            self._telknet_transcription_active = True
            self._telknet_declared_multi_instrument_source = bool(
                instruments and len(set(instruments)) > 1
            )
            try:
                yield from super().transcribe(
                    audio,
                    use_sampling=use_sampling,
                    temperature=temperature,
                    cfg_coef=cfg_coef,
                    instruments=instruments,
                    batch_size=batch_size,
                    no_eos_is_ok=no_eos_is_ok,
                    beam_size=beam_size,
                    prelude_forcing=prelude_forcing,
                )
            finally:
                self._telknet_declared_multi_instrument_source = False
                self._telknet_transcription_active = False

        def _generate_token_stream(
            self,
            all_conditions,
            seek_times: list[float],
            batch_size: int,
            max_gen_len: int,
            use_sampling: bool,
            temperature: float,
            cfg_coef: float,
            no_eos_is_ok: bool,
            prelude_forcing: bool = True,
            beam_size: int = 1,
            forbidden_tokens=None,
        ):
            """Recover only the proven recurrent single-program boundary state."""

            eos_id = self._tokenizer.eos_id
            num_chunks = len(seek_times)
            tracker = None
            if prelude_forcing and batch_size == 1:
                tracker = OpenNoteTracker(
                    self._tokenizer._vocab,
                    self._tokenizer.frame_rate,
                )
            previous_single_open_program: int | None = None
            observed_instrument_classes: set[tuple[str, int]] = set()
            drum_evidence_grace_seconds = 4 * segment_duration
            declared_multi_instrument_source = bool(
                getattr(self, "_telknet_declared_multi_instrument_source", False)
            )

            def observe_instrument_event(event) -> None:
                if event.type == "program":
                    observed_instrument_classes.add(("program", event.value))
                elif event.type == "drum":
                    observed_instrument_classes.add(("drums", 0))

            def boundary(chunk_index: int):
                next_seek_time = (
                    seek_times[chunk_index + 1] if chunk_index + 1 < num_chunks else None
                )
                return ChunkBoundary(seek_times[chunk_index], next_seek_time)

            for batch_start in range(0, num_chunks, batch_size):
                batch_conditions = all_conditions[batch_start : batch_start + batch_size]
                count = len(batch_conditions)
                buffers: list[list[int]] = [[] for _ in range(count)]
                done = [False] * count
                active = 0

                chunk_boundary = boundary(batch_start)
                prompt = None
                recovery_prelude: list[int] | None = None
                recovery_body_started = False
                recovery_generated_prefix: list[int] = []
                if tracker is not None:
                    tracker.feed(chunk_boundary)
                    if batch_start > 0:
                        open_keys = tracker.open_keys()
                        if open_keys:
                            open_programs = {program for program, _pitch in open_keys}
                            single_program = (
                                next(iter(open_programs)) if len(open_programs) == 1 else None
                            )
                            observed_pitched_programs = {
                                value
                                for kind, value in observed_instrument_classes
                                if kind == "program"
                            }
                            has_decoded_pitched_diversity = len(observed_pitched_programs) > 1
                            has_mature_drum_diversity = (
                                bool(observed_pitched_programs)
                                and ("drums", 0) in observed_instrument_classes
                                and seek_times[batch_start] >= drum_evidence_grace_seconds
                            )
                            has_prior_instrument_diversity = (
                                declared_multi_instrument_source
                                or has_decoded_pitched_diversity
                                or has_mature_drum_diversity
                            )
                            repeated_single_program = (
                                has_prior_instrument_diversity
                                and single_program is not None
                                and previous_single_open_program == single_program
                            )
                            immediate_bass_recovery = (
                                has_prior_instrument_diversity
                                and single_program is not None
                                and 32 <= single_program <= 39
                            )
                            if repeated_single_program or immediate_bass_recovery:
                                recovery_prelude = self._tokenizer.tie_section_token_ids(open_keys)
                            else:
                                prompt = torch.tensor(
                                    [self._tokenizer.tie_section_token_ids(open_keys)],
                                    device=self._device,
                                    dtype=torch.long,
                                )
                        else:
                            single_program = None
                        previous_single_open_program = single_program
                yield chunk_boundary
                if recovery_prelude is not None:
                    assert tracker is not None
                    for token in recovery_prelude:
                        tracker.feed(token)
                        yield token

                for step in self._model.generate(
                    prompt=prompt,
                    conditions=batch_conditions,
                    max_gen_len=max_gen_len,
                    use_sampling=use_sampling,
                    temp=temperature,
                    top_k=0,
                    top_p=0.0,
                    cfg_coef=cfg_coef,
                    early_stop_on_token=eos_id,
                    beam_size=beam_size,
                    forbidden_tokens=forbidden_tokens,
                ):
                    row = step.tolist()
                    for batch_index in range(count):
                        if done[batch_index]:
                            continue
                        token = row[batch_index]
                        if token == eos_id:
                            done[batch_index] = True
                            continue

                        if recovery_prelude is not None and not recovery_body_started:
                            event_type = self._tokenizer._vocab[token].type
                            if event_type in ("program", "pitch"):
                                recovery_generated_prefix.append(token)
                                continue
                            recovery_body_started = True
                            if event_type == "tie":
                                recovery_generated_prefix.clear()
                                continue
                            assert tracker is not None
                            for prefix_token in recovery_generated_prefix:
                                tracker.feed(prefix_token)
                                observe_instrument_event(self._tokenizer._vocab[prefix_token])
                                if batch_index == active:
                                    yield prefix_token
                                else:
                                    buffers[batch_index].append(prefix_token)
                            recovery_generated_prefix.clear()

                        if tracker is not None:
                            tracker.feed(token)
                        observe_instrument_event(self._tokenizer._vocab[token])
                        if batch_index == active:
                            yield token
                        else:
                            buffers[batch_index].append(token)

                    while active < count and done[active]:
                        active += 1
                        if active < count:
                            yield boundary(batch_start + active)
                            yield from buffers[active]
                            buffers[active] = []

                for batch_index in range(active, count):
                    if not done[batch_index]:
                        chunk_index = batch_start + batch_index
                        message = (
                            f"chunk {chunk_index} "
                            f"(seek={seek_times[chunk_index]:.1f}s) did not emit "
                            f"EOS within {max_gen_len} tokens"
                        )
                        if no_eos_is_ok:
                            warnings.warn(message, RuntimeWarning, stacklevel=2)
                        else:
                            raise RuntimeError(
                                message + " (this is only raised under --strict-eos)"
                            )
                    if batch_index != active:
                        yield boundary(batch_start + batch_index)
                        yield from buffers[batch_index]

                yield ProgressEvent(completed=batch_start + count, total=num_chunks)

    TelkNetBoundaryRecoveryTranscriptionModel.__name__ = "TelkNetBoundaryRecoveryTranscriptionModel"
    return TelkNetBoundaryRecoveryTranscriptionModel


def _remap_checkpoint_key(key: str) -> str:
    """Apply the pinned upstream single-codebook compatibility mapping."""

    if key.startswith(("emb.1.", "linears.1.")):
        raise ValueError(
            "Checkpoint has more than one codebook (n_q > 1); "
            "only single-stream models are supported."
        )
    if key.startswith("emb.0."):
        return "emb." + key[len("emb.0.") :]
    if key.startswith("linears.0."):
        return "linear." + key[len("linears.0.") :]
    return key


def _open_safetensors_source(weights_path: str | Path, target_device):
    """Open the lazy checkpoint reader before unified-memory model allocation."""

    from safetensors import safe_open

    checkpoint = Path(weights_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MuScriptor checkpoint does not exist: {checkpoint}")
    reader_options = _safetensors_open_kwargs(target_device)
    reader_backend = reader_options.get("backend", "mmap")
    logger.info(
        "Loading MuScriptor checkpoint with safetensors backend=%s on %s",
        reader_backend,
        target_device,
    )
    return (
        safe_open(
            checkpoint,
            framework="pt",
            device="cpu",
            **reader_options,
        ),
        reader_backend,
    )


def _copy_safetensors_source_in_place(model, source, target_device, reader_backend: str) -> int:
    """Validate and copy one tensor at a time from an already-open source."""

    import torch

    destination = model.state_dict()
    source_keys = list(source.offset_keys() if reader_backend == "pread" else source.keys())
    key_map = {key: _remap_checkpoint_key(key) for key in source_keys}
    if len(set(key_map.values())) != len(key_map):
        raise RuntimeError("MuScriptor checkpoint key remapping produced duplicates")

    missing = sorted(set(destination) - set(key_map.values()))
    unexpected = sorted(set(key_map.values()) - set(destination))
    if missing or unexpected:
        raise RuntimeError(
            "MuScriptor checkpoint/model key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    # Validate the complete checkpoint before mutating any parameter.
    for source_key, target_key in key_map.items():
        target = destination[target_key]
        tensor_slice = source.get_slice(source_key)
        source_shape = tuple(int(value) for value in tensor_slice.get_shape())
        if source_shape != tuple(target.shape):
            raise RuntimeError(
                "MuScriptor checkpoint tensor shape mismatch: "
                f"key={source_key}, checkpoint={source_shape}, model={tuple(target.shape)}"
            )
        source_dtype = tensor_slice.get_dtype()
        expected_source_dtype = {
            torch.bool: "BOOL",
            torch.uint8: "U8",
            torch.int8: "I8",
            torch.int16: "I16",
            torch.int32: "I32",
            torch.int64: "I64",
            torch.float16: "F16",
            torch.bfloat16: "BF16",
            torch.float32: "F32",
            torch.float64: "F64",
        }.get(target.dtype)
        if expected_source_dtype is None:
            raise RuntimeError(
                "MuScriptor model uses an unsupported destination dtype: "
                f"key={target_key}, model={target.dtype}"
            )
        if source_dtype != expected_source_dtype:
            raise RuntimeError(
                "MuScriptor checkpoint/model dtype mismatch: "
                f"key={source_key}, checkpoint={source_dtype}, model={target.dtype}"
            )
        # Upstream _build_model(device) constructs the transformer on the
        # requested accelerator but leaves a small conditioner projection
        # on CPU until its final model.to(device) call after state loading.
        if target.device.type != "cpu" and target.device != target_device:
            raise RuntimeError(
                "MuScriptor model parameter is on the wrong device before loading: "
                f"key={target_key}, expected={target_device} or cpu, "
                f"actual={target.device}"
            )

    with torch.no_grad():
        for source_key, target_key in key_map.items():
            # Keep only one small owned CPU tensor alive and synchronize
            # before releasing its storage; the largest pinned tensor is
            # ~38 MiB. XPU uses safetensors' pread backend above, so this
            # tensor never aliases the complete checkpoint mapping.
            mapped_tensor = source.get_tensor(source_key)
            tensor = mapped_tensor.clone()
            del mapped_tensor
            target = destination[target_key]
            target.copy_(tensor, non_blocking=False)
            if target.is_cuda:
                torch.cuda.synchronize(target.device)
            elif target.device.type == "xpu":
                torch.xpu.synchronize(target.device.index)
            del tensor

    return len(source_keys)


def _load_safetensors_state_in_place(model, weights_path: str | Path, device) -> int:
    """Open, validate, and copy a checkpoint into an existing model."""

    import torch

    target_device = torch.device(device)
    source_context, reader_backend = _open_safetensors_source(weights_path, target_device)
    with source_context as source:
        return _copy_safetensors_source_in_place(
            model,
            source,
            target_device,
            reader_backend,
        )


def _normalize_processing_chain(processing_chain: str) -> str:
    normalized = str(processing_chain).strip().lower()
    valid = {chain.value for chain in MuscriptorProcessingChain}
    if normalized not in valid:
        raise ValueError(f"Unsupported MuScriptor processing chain: {processing_chain!r}")
    return normalized


def load_muscriptor_model_memory_bounded(
    weights_path: str | Path,
    device: str,
    processing_chain: str = MuscriptorProcessingChain.OFFICIAL.value,
):
    """Load official weights into the explicitly selected processing class."""

    import torch
    import muscriptor.transcription_model as upstream

    selected_chain = _normalize_processing_chain(processing_chain)
    checkpoint = Path(weights_path).resolve()
    target_device = torch.device(device)
    config = upstream._resolve_config(checkpoint, checkpoint)
    if target_device.type == "xpu":
        source_context, reader_backend = _open_safetensors_source(checkpoint, target_device)
        with source_context as source:
            model = upstream._build_model(target_device, config)
            model.eval()
            tensor_count = _copy_safetensors_source_in_place(
                model,
                source,
                target_device,
                reader_backend,
            )
    else:
        model = upstream._build_model(target_device, config)
        model.eval()
        tensor_count = _load_safetensors_state_in_place(model, checkpoint, target_device)
    model.to(target_device)
    wrong_devices = sorted(
        name for name, tensor in model.state_dict().items() if tensor.device != target_device
    )
    if wrong_devices:
        raise RuntimeError(
            "MuScriptor model did not finish moving to the requested device: "
            f"device={target_device}, keys={wrong_devices}"
        )

    tokenizer = upstream.MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS",
        max_shift_steps=1001,
    )
    logger.info(
        "Loaded %s MuScriptor tensors in place on %s using the upstream dtype policy "
        "and %s processing chain",
        tensor_count,
        target_device,
        selected_chain,
    )
    model_class = (
        upstream.TranscriptionModel
        if selected_chain == MuscriptorProcessingChain.OFFICIAL.value
        else _build_telknet_boundary_recovery_model_class(upstream, torch)
    )
    return model_class(
        model=model,
        tokenizer=tokenizer,
        device=target_device,
    )


__all__ = ["load_muscriptor_model_memory_bounded"]
