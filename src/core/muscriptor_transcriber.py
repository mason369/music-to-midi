"""Strict wrapper for the public MuScriptor transcription checkpoints."""

from __future__ import annotations

import gc
import inspect
import logging
import warnings
from pathlib import Path
from typing import Callable, Optional

from src.core.muscriptor_midi import (
    require_allowed_muscriptor_event_instrument,
    validate_muscriptor_midi_constraint,
)
from src.i18n.translator import Translator
from src.models.data_models import Config
from src.models.muscriptor_instruments import (
    MUSCRIPTOR_REPRESENTATIVE_PROGRAMS,
    validate_muscriptor_instruments,
)
from src.utils.midi_output import (
    publish_midi_output,
    remove_temporary_midi,
    unique_midi_temp_path,
)
from src.utils.muscriptor_downloader import (
    get_cached_muscriptor_paths,
    get_muscriptor_artifact,
    normalize_muscriptor_model,
)
from src.utils.muscriptor_source_patch import (
    MUSCRIPTOR_PACKAGE_VERSION,
    MUSCRIPTOR_QUALITY_PATCH_COMMIT,
    MUSCRIPTOR_SOURCE_COMMIT,
    MUSCRIPTOR_SOURCE_REQUIREMENT,
    validate_muscriptor_runtime_identity,
)

logger = logging.getLogger(__name__)


class MuscriptorTranscriber:
    """Run an explicit MuScriptor size with model-native hard instrument constraints."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._model = None
        self._runtime_details: dict[str, object] | None = None
        self._cancelled = False
        self._cancel_check: Optional[Callable[[], bool]] = None
        self._event_callback: Optional[Callable[[dict[str, object]], None]] = None
        self._translator = Translator(getattr(self.config, "language", Translator.DEFAULT_LANGUAGE))
        self.last_detected_instruments: list[str] = []

    @staticmethod
    def _runtime_unavailable_reason() -> str:
        identity_error = validate_muscriptor_runtime_identity()
        if identity_error:
            return (
                "MuScriptor runtime source identity is invalid. "
                f"{identity_error}. Install {MUSCRIPTOR_SOURCE_REQUIREMENT}, then run "
                "python patch_muscriptor_runtime.py."
            )

        try:
            from muscriptor import TranscriptionModel
            from muscriptor.tokenizer.mt3 import MT3Tokenizer
        except Exception as exc:
            return f"MuScriptor 运行时导入失败：{exc}"

        transcribe_parameters = inspect.signature(TranscriptionModel.transcribe).parameters
        if (
            "instruments" not in transcribe_parameters
            or "prelude_forcing" not in transcribe_parameters
            or "overlap" not in transcribe_parameters
            or "allow_reset" not in transcribe_parameters
        ):
            return (
                "MuScriptor runtime is missing the required hard-mask or "
                "overlap/restart API. Required identity: "
                f"main={MUSCRIPTOR_SOURCE_COMMIT}, "
                f"quality_patch={MUSCRIPTOR_QUALITY_PATCH_COMMIT}."
            )
        if not callable(getattr(MT3Tokenizer, "forbidden_token_ids", None)):
            return (
                "MuScriptor tokenizer is missing forbidden_token_ids; "
                "decode-time instrument constraints cannot be guaranteed."
            )
        if not callable(getattr(MT3Tokenizer, "overlap_prompt_token_ids", None)):
            return (
                "MuScriptor tokenizer is missing overlap_prompt_token_ids; "
                "best-quality overlapping-window transcription cannot run."
            )
        return ""

    def _selected_model_size(self) -> str:
        return normalize_muscriptor_model(getattr(self.config, "muscriptor_model", "large"))

    def get_unavailable_reason(self) -> str:
        runtime_error = self._runtime_unavailable_reason()
        if runtime_error:
            return runtime_error
        try:
            get_cached_muscriptor_paths(
                self._selected_model_size(),
                validate_hashes=False,
            )
        except Exception as exc:
            return str(exc)
        return ""

    @classmethod
    def is_available(cls) -> bool:
        return cls(Config()).get_unavailable_reason() == ""

    def is_selected_model_available(self) -> bool:
        return self.get_unavailable_reason() == ""

    def set_cancel_check(self, callback: Callable[[], bool]) -> None:
        self._cancel_check = callback

    def set_event_callback(self, callback: Optional[Callable[[dict[str, object]], None]]) -> None:
        self._event_callback = callback

    def cancel(self) -> None:
        self._cancelled = True

    def _check_cancelled(self) -> None:
        if self._cancelled or (self._cancel_check is not None and self._cancel_check()):
            raise InterruptedError("MuScriptor transcription cancelled")

    def load_model(self):
        if self._model is not None:
            return self._model

        runtime_error = self._runtime_unavailable_reason()
        if runtime_error:
            raise RuntimeError(runtime_error)
        model_size = self._selected_model_size()
        artifact = get_muscriptor_artifact(model_size)
        weights, _config = get_cached_muscriptor_paths(model_size, validate_hashes=True)

        import torch
        from muscriptor import TranscriptionModel

        if self.config.use_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"{artifact.display_name} 已选择 GPU 推理，但当前 PyTorch 看不到 CUDA；"
                    "不会静默切换到 CPU。"
                )
            device = f"cuda:{int(self.config.gpu_device)}"
        else:
            device = "cpu"

        self._check_cancelled()
        logger.info("Loading pinned %s on %s from %s", artifact.display_name, device, weights)
        self._model = TranscriptionModel.load_model(weights_path=weights, device=device)
        logger.info(
            "%s loaded with its pinned upstream precision and cache configuration",
            artifact.display_name,
        )
        self._runtime_details = {
            "type": "runtime",
            "model": artifact.display_name,
            "device": device,
            "gpu": device,
            "compute_dtype": "upstream",
            "kv_cache_dtype": "upstream",
            "kv_cache_reused_layers": 0,
            "batch_size": 1,
            "prelude_forcing": True,
            "quality_mode": "overlap_restart",
            "overlap_seconds": 2.5,
            "allow_reset": True,
            "strict_eos": True,
            "package_version": MUSCRIPTOR_PACKAGE_VERSION,
            "source_commit": MUSCRIPTOR_SOURCE_COMMIT,
            "quality_patch_commit": MUSCRIPTOR_QUALITY_PATCH_COMMIT,
        }
        self._check_cancelled()
        return self._model

    def _emit_event(self, payload: dict[str, object]) -> None:
        if self._event_callback is not None:
            self._event_callback(payload)

    def transcribe_to_midi(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """Stream official events, verify the constraint, and publish official MIDI."""

        source = Path(audio_path)
        if not source.is_file():
            raise FileNotFoundError(f"MuScriptor input audio does not exist: {source}")
        selected = validate_muscriptor_instruments(
            getattr(self.config, "muscriptor_instruments", [])
        )
        model = self.load_model()

        from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent

        self._cancelled = False
        self.last_detected_instruments = []
        detected: set[str] = set()
        official_events: list[object] = []
        pending_note_ends: list[dict[str, object]] = []
        self._check_cancelled()
        if self._runtime_details is not None:
            self._emit_event(dict(self._runtime_details))

        def flush_note_ends() -> None:
            if not pending_note_ends:
                return
            self._emit_event({"type": "note_batch", "notes": list(pending_note_ends)})
            pending_note_ends.clear()

        import torch

        # MuScriptor already disables gradients in its generator, but the outer
        # inference context also covers condition construction, event streaming,
        # and the per-chunk state tensors. This removes autograd view/version
        # bookkeeping without changing model precision or chunk ordering.
        with warnings.catch_warnings(), torch.inference_mode():
            # A quality-mode prompt overflow is a real loss of the requested
            # overlap context. Upstream currently warns and continues; the
            # product contract exposes that downgrade as a hard failure.
            warnings.filterwarnings(
                "error",
                category=RuntimeWarning,
                message=r"chunk .*: overlap prompt .*exceeds the generation budget.*",
            )
            warnings.filterwarnings(
                "error",
                category=RuntimeWarning,
                message=r"chunk .*: tie prologue .*exceeds the generation budget.*",
            )
            events = model.transcribe(
                source,
                instruments=selected or None,
                use_sampling=False,
                batch_size=1,
                beam_size=1,
                prelude_forcing=True,
                no_eos_is_ok=False,
                overlap=2.5,
                allow_reset=True,
            )
            for event in events:
                self._check_cancelled()
                if isinstance(event, ProgressEvent):
                    # A dense polyphonic chunk can contain hundreds of events. One
                    # queued Qt signal per note can starve the GUI thread, so publish
                    # the completed notes as one chunk-owned batch.
                    flush_note_ends()
                    completed = int(event.completed)
                    total = max(1, int(event.total))
                    progress = max(0.0, min(1.0, completed / total))
                    if progress_callback is not None:
                        progress_callback(
                            progress,
                            self._translator.t(
                                "progress.muscriptor_chunks",
                                completed=completed,
                                total=total,
                            ),
                        )
                    self._emit_event({"type": "progress", "completed": completed, "total": total})
                elif isinstance(event, NoteStartEvent):
                    instrument = str(event.instrument)
                    require_allowed_muscriptor_event_instrument(instrument, selected)
                    if instrument not in detected:
                        detected.add(instrument)
                        self.last_detected_instruments.append(instrument)
                elif isinstance(event, NoteEndEvent):
                    instrument = str(event.start_event.instrument)
                    require_allowed_muscriptor_event_instrument(instrument, selected)
                    pending_note_ends.append(
                        {
                            "index": int(event.start_event_index),
                            "instrument": instrument,
                            "pitch": int(event.start_event.pitch),
                            "start_time": float(event.start_event.start_time),
                            "end_time": float(event.end_time),
                            "program": MUSCRIPTOR_REPRESENTATIVE_PROGRAMS.get(instrument),
                            "is_drum": instrument == "drums",
                        }
                    )
                else:
                    raise RuntimeError(
                        f"MuScriptor returned an unsupported event type: {type(event).__name__}"
                    )
                official_events.append(event)

        flush_note_ends()

        self._check_cancelled()
        midi_bytes = model.events_to_midi_bytes(iter(official_events))
        temporary = unique_midi_temp_path(output_path, "muscriptor-official")
        try:
            temporary.write_bytes(midi_bytes)
            validate_muscriptor_midi_constraint(temporary, selected)
            published = publish_midi_output(
                temporary,
                output_path,
                get_muscriptor_artifact(self._selected_model_size()).display_name,
            )
        finally:
            remove_temporary_midi(temporary)

        if progress_callback is not None:
            progress_callback(1.0, self._translator.t("progress.muscriptor_complete"))
        return published

    def unload_model(self) -> None:
        model = self._model
        self._model = None
        self._runtime_details = None
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("MuScriptor CUDA cache cleanup failed: %s", exc)
