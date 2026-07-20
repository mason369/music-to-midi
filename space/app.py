"""
Music to MIDI - Gradio Web 界面
视觉风格对齐 PyQt6 桌面版暗色主题。
"""

import atexit
import importlib
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from copy import deepcopy
from pathlib import Path

sys.setrecursionlimit(3000)

APP_TEMP_DIR = tempfile.gettempdir()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["NUMBA_CACHE_DIR"] = os.path.join(APP_TEMP_DIR, "numba_cache")
os.environ["MPLCONFIGDIR"] = os.path.join(APP_TEMP_DIR, "matplotlib")

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_space_project_layout(app_dir: str) -> tuple[str, Path, tuple[Path, ...]]:
    """Resolve only the HF-root or repository-parent source layouts."""

    app_root = Path(app_dir).resolve()
    candidates = (
        ("hf-root", app_root),
        ("repository-parent", app_root.parent),
    )
    required_paths = (
        Path("src") / "__init__.py",
        Path("src") / "utils" / "yourmt3_source_identity.py",
        Path("YourMT3") / "amt" / "src" / "model" / "ymt3.py",
    )
    complete_roots = []
    incomplete_layouts = []
    for layout_name, root in candidates:
        present = [relative for relative in required_paths if (root / relative).is_file()]
        if len(present) == len(required_paths):
            complete_roots.append((layout_name, root))
        elif present:
            missing = [str(relative) for relative in required_paths if relative not in present]
            incomplete_layouts.append(f"{layout_name}={root}: missing {missing}")

    if incomplete_layouts:
        raise RuntimeError(
            "Incomplete controlled Space project layout detected: " + "; ".join(incomplete_layouts)
        )
    if not complete_roots:
        checked = ", ".join(f"{name}={root}" for name, root in candidates)
        raise RuntimeError(
            "Space project sources were not found in either controlled layout: " + checked
        )

    selected_name, selected_root = complete_roots[0]
    return selected_name, selected_root, tuple(root for _name, root in complete_roots)


SPACE_PROJECT_LAYOUT, PROJECT_ROOT, CONTROLLED_PROJECT_ROOTS = _resolve_space_project_layout(
    APP_DIR
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _validate_controlled_yourmt3_layouts() -> tuple[Path, str, int]:
    from src.utils.yourmt3_source_identity import validate_patched_yourmt3_source

    identities = {}
    for project_root in CONTROLLED_PROJECT_ROOTS:
        amt_src = project_root / "YourMT3" / "amt" / "src"
        identities[project_root] = validate_patched_yourmt3_source(amt_src)

    if len(set(identities.values())) != 1:
        details = ", ".join(f"{root}={identity}" for root, identity in identities.items())
        raise RuntimeError(f"Controlled Space YourMT3 source layouts disagree: {details}")

    selected_source = PROJECT_ROOT / "YourMT3" / "amt" / "src"
    manifest_sha256, file_count = identities[PROJECT_ROOT]
    return selected_source, manifest_sha256, file_count


_validate_controlled_yourmt3_layouts()

LOG_FILE = os.path.join(APP_TEMP_DIR, "midi_process.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("music-to-midi-web")

SPACE_OUTPUT_RETENTION_SECONDS = int(
    os.environ.get("MUSIC_TO_MIDI_SPACE_OUTPUT_RETENTION_SECONDS", "86400")
)
if SPACE_OUTPUT_RETENTION_SECONDS <= 0:
    raise RuntimeError("MUSIC_TO_MIDI_SPACE_OUTPUT_RETENTION_SECONDS must be positive")

SPACE_OUTPUT_PARENT = Path(APP_TEMP_DIR) / "music-to-midi-space-results"
SPACE_OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
SPACE_OUTPUT_INSTANCE = Path(
    tempfile.mkdtemp(prefix=f"instance-{os.getpid()}-", dir=SPACE_OUTPUT_PARENT)
).resolve()
SPACE_REQUEST_PREFIX = "request-"


def _remove_space_output_dir(output_dir: str | Path) -> None:
    """Remove one app-owned request directory and reject unsafe paths."""

    candidate = Path(output_dir).resolve()
    if candidate.parent != SPACE_OUTPUT_INSTANCE or not candidate.name.startswith(
        SPACE_REQUEST_PREFIX
    ):
        raise RuntimeError(f"Refusing to remove non-request Space output path: {candidate}")
    if candidate.exists():
        shutil.rmtree(candidate)


def _remove_stale_space_instance(instance_dir: str | Path) -> None:
    """Remove one expired sibling instance without touching the active instance."""

    candidate = Path(instance_dir).resolve()
    if (
        candidate.parent != SPACE_OUTPUT_PARENT.resolve()
        or candidate == SPACE_OUTPUT_INSTANCE
        or not candidate.name.startswith("instance-")
    ):
        raise RuntimeError(f"Refusing to remove non-stale Space instance path: {candidate}")
    if candidate.exists():
        shutil.rmtree(candidate)


def _cleanup_stale_space_outputs(*, now: float | None = None) -> None:
    """Delete expired results and crashed-process instances after the retention window."""

    cutoff = (time.time() if now is None else now) - SPACE_OUTPUT_RETENTION_SECONDS
    for candidate in SPACE_OUTPUT_INSTANCE.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith(SPACE_REQUEST_PREFIX):
            continue
        try:
            modified_at = candidate.stat().st_mtime
        except OSError as exc:
            logger.error("Unable to inspect stale Space output %s: %s", candidate, exc)
            continue
        if modified_at >= cutoff:
            continue
        try:
            _remove_space_output_dir(candidate)
            logger.info("Removed expired Space output directory: %s", candidate)
        except OSError as exc:
            # An old download can still hold a file open on some platforms.  The
            # cleanup failure is recorded and retried on the next request.
            logger.error("Unable to remove expired Space output %s: %s", candidate, exc)

    for instance_dir in SPACE_OUTPUT_PARENT.iterdir():
        if (
            not instance_dir.is_dir()
            or not instance_dir.name.startswith("instance-")
            or instance_dir.resolve() == SPACE_OUTPUT_INSTANCE
        ):
            continue
        try:
            latest_mtime = max(
                [instance_dir.stat().st_mtime]
                + [child.stat().st_mtime for child in instance_dir.iterdir()]
            )
        except OSError as exc:
            logger.error("Unable to inspect stale Space instance %s: %s", instance_dir, exc)
            continue
        if latest_mtime >= cutoff:
            continue
        try:
            _remove_stale_space_instance(instance_dir)
            logger.info("Removed expired Space instance directory: %s", instance_dir)
        except OSError as exc:
            logger.error("Unable to remove expired Space instance %s: %s", instance_dir, exc)


def _create_space_output_dir() -> str:
    _cleanup_stale_space_outputs()
    return tempfile.mkdtemp(prefix=SPACE_REQUEST_PREFIX, dir=SPACE_OUTPUT_INSTANCE)


def _cleanup_space_instance_at_exit() -> None:
    try:
        if SPACE_OUTPUT_INSTANCE.exists():
            shutil.rmtree(SPACE_OUTPUT_INSTANCE)
    except OSError as exc:
        logger.error("Unable to remove Space output instance at shutdown: %s", exc)


atexit.register(_cleanup_space_instance_at_exit)


class _RobustFileHandler(logging.Handler):
    def __init__(self, filename, encoding="utf-8"):
        super().__init__()
        self.filename = filename
        self.encoding = encoding

    def emit(self, record):
        try:
            msg = self.format(record)
            with open(self.filename, "a", encoding=self.encoding) as f:
                f.write(msg + "\n")
        except Exception:
            pass


_file_handler = _RobustFileHandler(LOG_FILE)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
)
_file_handler.setLevel(logging.INFO)
for _name in ("music-to-midi-web", "src.core", "src.utils"):
    logging.getLogger(_name).addHandler(_file_handler)


def clear_logs():
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass


def read_logs():
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().replace("\x00", "")
        lines = content.strip().split("\n")
        return "\n".join(lines[-50:]) if lines and lines[0] else ""
    except Exception as exc:
        return f"[read_logs error] {exc}"


def ensure_yourmt3_code():
    """加载与桌面版、Colab 相同的项目内置 YourMT3 补丁源码。"""
    amt_src, source_sha256, source_file_count = _validate_controlled_yourmt3_layouts()
    logger.info(
        "Bundled patched YourMT3 source ready: layout=%s files=%s manifest_sha256=%s",
        SPACE_PROJECT_LAYOUT,
        source_file_count,
        source_sha256,
    )
    amt_src_text = str(amt_src)
    if amt_src_text not in sys.path:
        sys.path.insert(0, amt_src_text)
        logger.info("Added bundled patched YourMT3 source path: %s", amt_src)


try:
    import gradio_client.utils as _gcu

    _original_json_schema = _gcu._json_schema_to_python_type

    def _patched_json_schema(schema, defs=None):
        if isinstance(schema, bool):
            return "bool"
        return _original_json_schema(schema, defs)

    _gcu._json_schema_to_python_type = _patched_json_schema
except Exception:
    pass


try:
    import spaces
except ImportError as exc:
    # Match spaces.config.boolean exactly so a requested ZeroGPU deployment
    # never degrades into a CPU-only Space merely because the package is gone.
    if os.environ.get("SPACES_ZERO_GPU", "").strip().lower() in {"1", "t", "true"}:
        raise RuntimeError(
            "SPACES_ZERO_GPU is enabled but the required 'spaces' package is unavailable"
        ) from exc
    ZERO_GPU = False
    logger.info("Running without the optional spaces package")
else:
    from spaces.config import Config as SpacesConfig

    # The public Gradio handler intentionally performs admission and downloads
    # before calling our explicitly decorated GPU-only function. HF's optional
    # auto-wrapper would otherwise wrap the public handler in a default 60s GPU
    # window and bypass that boundary.
    spaces.disable_gradio_auto_wrap()
    ZERO_GPU = bool(SpacesConfig.zero_gpu)
    logger.info("ZeroGPU runtime enabled: %s", ZERO_GPU)


import gradio as gr
from gradio.components.base import Component

from src.core.manual_midi import (
    MANUAL_MIDI_ROUTES,
    MIDI_ROUTE_MIROS,
    MIDI_ROUTE_MUSCRIPTOR,
    MIDI_ROUTE_PIANO_ARIA_AMT,
    MIDI_ROUTE_PIANO_BYTEDANCE_PEDAL,
    MIDI_ROUTE_PIANO_TRANSKUN,
    MIDI_ROUTE_PIANO_TRANSKUN_V2_AUG,
    MIDI_ROUTE_YOURMT3_PREFIX,
    build_manual_midi_config,
    manual_midi_output_dir,
)
from src.core.multi_stem_separator import STEM_KEYS
from src.core.separation_service import AudioSeparationService, SeparationResult
from src.gui.web.muscriptor_result_runtime import (
    build_muscriptor_result_html,
    muscriptor_result_head,
)
from src.gui.web.track_mixer_runtime import TRACK_COLORS as _TRACK_COLORS
from src.gui.web.track_mixer_runtime import (
    build_track_mixer_html,
    mixer_head,
)
from src.i18n.translator import Translator
from src.models.data_models import (
    Config,
    MultiInstrumentModel,
    ProcessingMode,
    ProcessingStage,
    YourMT3Model,
)
from src.models.muscriptor_instruments import (
    MUSCRIPTOR_INSTRUMENTS,
    muscriptor_instrument_label,
    validate_muscriptor_instruments,
)
from src.utils.yourmt3_downloader import YOURMT3_MODELS

SPACE_LANGUAGE = os.environ.get("MUSIC_TO_MIDI_LANGUAGE", "zh_CN")
if SPACE_LANGUAGE not in Translator.AVAILABLE_LANGUAGES:
    raise RuntimeError(f"Unsupported MUSIC_TO_MIDI_LANGUAGE: {SPACE_LANGUAGE}")
SPACE_TRANSLATOR = Translator(SPACE_LANGUAGE)


def st(key: str, **kwargs) -> str:
    return SPACE_TRANSLATOR.t(key, **kwargs)


def _normalize_json_schema_bool_nodes(schema):
    if isinstance(schema, dict):
        for key, value in list(schema.items()):
            if key == "additionalProperties" and isinstance(value, bool):
                schema[key] = {}
            else:
                _normalize_json_schema_bool_nodes(value)
    elif isinstance(schema, list):
        for item in schema:
            _normalize_json_schema_bool_nodes(item)
    return schema


_original_component_api_info = Component.api_info


def _patched_component_api_info(self):
    return _normalize_json_schema_bool_nodes(deepcopy(_original_component_api_info(self)))


Component.api_info = _patched_component_api_info


MODE_IDS = (
    ProcessingMode.SMART.value,
    ProcessingMode.VOCAL_SPLIT.value,
    ProcessingMode.SIX_STEM_SPLIT.value,
    ProcessingMode.PIANO_TRANSKUN.value,
    ProcessingMode.PIANO_TRANSKUN_V2_AUG.value,
    ProcessingMode.PIANO_ARIA_AMT.value,
    ProcessingMode.PIANO_BYTEDANCE_PEDAL.value,
)
MODE_LABELS = {mode_id: st(f"main.mode.{mode_id}") for mode_id in MODE_IDS}
MODE_CHOICES = [(MODE_LABELS[mode_id], mode_id) for mode_id in MODE_IDS]
SPLIT_MODE_IDS = frozenset(
    {
        ProcessingMode.VOCAL_SPLIT.value,
        ProcessingMode.SIX_STEM_SPLIT.value,
    }
)
# Only SMART chooses one global multi-instrument backend. Split modes stop
# after WAV separation; each rendered track chooses its own route later.
MULTI_INSTRUMENT_MODE_IDS = {
    ProcessingMode.SMART.value,
}
BACKEND_CHOICES = [
    (st("main.engine.yourmt3"), MultiInstrumentModel.YOURMT3.value),
    (st("main.engine.miros"), MultiInstrumentModel.MIROS.value),
    (st("main.engine.muscriptor"), MultiInstrumentModel.MUSCRIPTOR.value),
]
MUSCRIPTOR_INSTRUMENT_CHOICES = [
    (muscriptor_instrument_label(name, SPACE_LANGUAGE), name) for name in MUSCRIPTOR_INSTRUMENTS
]
YOURMT3_MODEL_CHOICES = [
    (YOURMT3_MODELS[model.value]["ui_label"], model.value)
    for model in (
        YourMT3Model.YMT3_PLUS,
        YourMT3Model.YPTF_SINGLE_NOPS,
        YourMT3Model.YPTF_MULTI_PS,
        YourMT3Model.YPTF_MOE_MULTI_NOPS,
        YourMT3Model.YPTF_MOE_MULTI_PS,
    )
]
STAGE_LABEL_KEYS = {
    ProcessingStage.PREPROCESSING: "preprocessing",
    ProcessingStage.SEPARATION: "separation",
    ProcessingStage.TRANSCRIPTION: "transcription",
    ProcessingStage.VOCAL_TRANSCRIPTION: "vocal_transcription",
    ProcessingStage.SYNTHESIS: "synthesis",
    ProcessingStage.COMPLETE: "complete",
}

_TRACK_ORDER = (
    "bass",
    "drums",
    "guitar",
    "piano",
    "vocals",
    "accompaniment",
    "other",
    "source",
)
# Shared with the desktop mixer and the Colab runtime via
# src.gui.web.track_mixer_runtime.TRACK_COLORS (imported above).
_SUPPORTED_AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a"})


def _manual_midi_route_label(route: str) -> str:
    if route.startswith(MIDI_ROUTE_YOURMT3_PREFIX):
        model_name = route.removeprefix(MIDI_ROUTE_YOURMT3_PREFIX)
        info = YOURMT3_MODELS.get(model_name)
        if info is None:
            raise ValueError(f"Unsupported YourMT3 model route: {route!r}")
        model_label = info.get("ui_label") or info.get("name") or model_name
        return (
            f"{st('dialogs.complete.audio_tracks.manual_midi.multi_instrument')}"
            f" · YourMT3+ · {model_label}"
        )

    route_labels = {
        MIDI_ROUTE_MIROS: st("dialogs.complete.audio_tracks.manual_midi.models.miros"),
        MIDI_ROUTE_MUSCRIPTOR: st("dialogs.complete.audio_tracks.manual_midi.models.muscriptor"),
        MIDI_ROUTE_PIANO_TRANSKUN: st(
            "dialogs.complete.audio_tracks.manual_midi.models.piano_transkun"
        ),
        MIDI_ROUTE_PIANO_TRANSKUN_V2_AUG: st(
            "dialogs.complete.audio_tracks.manual_midi.models.piano_transkun_v2_aug"
        ),
        MIDI_ROUTE_PIANO_ARIA_AMT: st(
            "dialogs.complete.audio_tracks.manual_midi.models.piano_aria_amt"
        ),
        MIDI_ROUTE_PIANO_BYTEDANCE_PEDAL: st(
            "dialogs.complete.audio_tracks.manual_midi.models.piano_bytedance_pedal"
        ),
    }
    try:
        route_label = route_labels[route]
    except KeyError as exc:
        raise ValueError(f"Unsupported manual MIDI route: {route!r}") from exc
    family_key = (
        "dialogs.complete.audio_tracks.manual_midi.multi_instrument"
        if route in {MIDI_ROUTE_MIROS, MIDI_ROUTE_MUSCRIPTOR}
        else "dialogs.complete.audio_tracks.manual_midi.piano"
    )
    return f"{st(family_key)} · {route_label}"


MANUAL_MIDI_ROUTE_CHOICES = [
    (_manual_midi_route_label(route), route) for route in MANUAL_MIDI_ROUTES
]
if len(MANUAL_MIDI_ROUTE_CHOICES) != 11:
    raise RuntimeError(
        "Space requires exactly eleven explicit per-track MIDI routes; "
        f"received {len(MANUAL_MIDI_ROUTE_CHOICES)}"
    )


def _display_track_name(track_name: str) -> str:
    if track_name in _TRACK_ORDER:
        return st(f"dialogs.complete.audio_tracks.track_names.{track_name}")
    return track_name


def _require_active_request_dir(request_dir: str | Path) -> Path:
    candidate = Path(request_dir).resolve()
    if (
        candidate.parent != SPACE_OUTPUT_INSTANCE
        or not candidate.name.startswith(SPACE_REQUEST_PREFIX)
        or not candidate.is_dir()
    ):
        raise RuntimeError(f"Invalid or expired Space request directory: {candidate}")
    return candidate


def _require_owned_request_file(
    request_dir: str | Path,
    file_path: str | Path,
    label: str,
) -> Path:
    request_root = _require_active_request_dir(request_dir)
    candidate = Path(file_path).resolve()
    try:
        candidate.relative_to(request_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{label} escapes the active Space request directory: {candidate}"
        ) from exc
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise RuntimeError(f"{label} is missing or empty: {candidate}")
    return candidate


def _require_owned_request_output_dir(
    request_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    request_root = _require_active_request_dir(request_dir)
    candidate = Path(output_dir).resolve()
    try:
        candidate.relative_to(request_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Manual MIDI output escapes the active Space request directory: {candidate}"
        ) from exc
    if candidate == request_root:
        raise RuntimeError("Manual MIDI output must use a route-specific subdirectory")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _normalize_midi_result_state(
    raw_state,
    request_root: Path,
    *,
    expected_audio_path: str | Path,
) -> dict:
    """Validate one linked MIDI workbench before exposing its files to Gradio."""
    if not isinstance(raw_state, dict):
        raise RuntimeError("Linked MIDI result state must be a dictionary")
    if raw_state.get("kind") not in {"midi_result", "muscriptor_result"}:
        raise RuntimeError("Linked MIDI result state has an unsupported kind")
    audio_path = _require_owned_request_file(
        request_root,
        raw_state.get("audio_path", ""),
        "Linked MIDI source audio",
    )
    if audio_path != Path(expected_audio_path).resolve():
        raise RuntimeError("Linked MIDI result does not belong to its source WAV track")

    def owned_file(key: str, label: str) -> str:
        return str(
            _require_owned_request_file(
                request_root,
                raw_state.get(key, ""),
                label,
            )
        )

    raw_notes = raw_state.get("notes")
    if not isinstance(raw_notes, list) or not raw_notes:
        raise RuntimeError("Linked MIDI result does not contain playable notes")
    notes = []
    for raw_note in raw_notes:
        if not isinstance(raw_note, dict):
            raise RuntimeError("Linked MIDI note entries must be dictionaries")
        note = {
            "instrument": str(raw_note.get("instrument", "")).strip(),
            "pitch": int(raw_note.get("pitch", -1)),
            "velocity": int(raw_note.get("velocity", -1)),
            "start": float(raw_note.get("start", -1.0)),
            "end": float(raw_note.get("end", -1.0)),
        }
        if (
            not note["instrument"]
            or not 0 <= note["pitch"] <= 127
            or not 0 <= note["velocity"] <= 127
            or note["start"] < 0
            or note["end"] <= note["start"]
        ):
            raise RuntimeError(f"Linked MIDI result contains an invalid note: {note!r}")
        notes.append(note)

    instrument_wavs = {}
    raw_instrument_wavs = raw_state.get("instrument_wavs")
    if not isinstance(raw_instrument_wavs, dict) or not raw_instrument_wavs:
        raise RuntimeError("Linked MIDI result does not contain instrument audio buses")
    for instrument, path in raw_instrument_wavs.items():
        instrument_name = str(instrument).strip()
        if not instrument_name:
            raise RuntimeError("Linked MIDI result contains an empty instrument id")
        instrument_wavs[instrument_name] = str(
            _require_owned_request_file(
                request_root,
                path,
                f"Linked MIDI instrument audio {instrument_name}",
            )
        )

    duration = float(raw_state.get("duration", 0.0))
    if duration <= 0:
        raise RuntimeError("Linked MIDI result has no playable duration")
    return {
        "kind": "midi_result",
        "audio_path": str(audio_path),
        "midi_path": owned_file("midi_path", "Linked MIDI file"),
        "selected_instruments": [str(item) for item in raw_state.get("selected_instruments", [])],
        "detected_instruments": [str(item) for item in raw_state.get("detected_instruments", [])],
        "notes": notes,
        "duration": duration,
        "transcription_wav": owned_file("transcription_wav", "Linked MIDI transcription audio"),
        "stereo_mix_wav": owned_file("stereo_mix_wav", "Linked MIDI stereo audio"),
        "instrument_wavs": instrument_wavs,
        "backend_label": str(raw_state.get("backend_label", "")).strip(),
        "source_track_name": str(raw_state.get("source_track_name", "")).strip(),
    }


def _normalize_track_state(track_state) -> dict:
    if not track_state:
        return {}
    if not isinstance(track_state, dict):
        raise RuntimeError("Space track state must be a dictionary")
    mode = str(track_state.get("mode", ""))
    if mode not in SPLIT_MODE_IDS:
        raise RuntimeError(f"Track state has unsupported split mode: {mode!r}")
    request_root = _require_active_request_dir(track_state.get("request_dir", ""))
    raw_tracks = track_state.get("tracks")
    if not isinstance(raw_tracks, list):
        raise RuntimeError("Space track state does not contain audio tracks")

    normalized_tracks = []
    seen_ids = set()
    for raw_track in raw_tracks:
        if not isinstance(raw_track, dict):
            raise RuntimeError("Space track entries must be dictionaries")
        track_id = str(raw_track.get("id", "")).strip()
        track_name = str(raw_track.get("name", "")).strip()
        if not track_id or not track_name or track_id in seen_ids:
            raise RuntimeError(f"Invalid or duplicate Space track identity: {track_id!r}")
        seen_ids.add(track_id)
        audio_path = _require_owned_request_file(
            request_root,
            raw_track.get("audio_path", ""),
            f"Audio track {track_name}",
        )
        route = str(raw_track.get("route", ""))
        if route and route not in MANUAL_MIDI_ROUTES:
            raise RuntimeError(f"Unsupported manual MIDI route in track state: {route!r}")
        midi_path_value = raw_track.get("midi_path")
        midi_path = ""
        if midi_path_value:
            midi_path = str(
                _require_owned_request_file(
                    request_root,
                    midi_path_value,
                    f"Generated MIDI for {track_name}",
                )
            )
        normalized_tracks.append(
            {
                "id": track_id,
                "name": track_name,
                "audio_path": str(audio_path),
                "color": str(raw_track.get("color", "#5eb1ff")),
                "midi_enabled": bool(raw_track.get("midi_enabled", False)),
                "route": route,
                "muscriptor_instruments": validate_muscriptor_instruments(
                    raw_track.get("muscriptor_instruments", [])
                ),
                "status": str(
                    raw_track.get("status")
                    or st("dialogs.complete.audio_tracks.manual_midi.not_selected")
                ),
                "midi_path": midi_path,
            }
        )
    active_track_id = str(track_state.get("active_midi_track_id", "")).strip()
    active_result = None
    if active_track_id:
        active_track = next(
            (track for track in normalized_tracks if track["id"] == active_track_id),
            None,
        )
        if active_track is None:
            raise RuntimeError("Linked MIDI result references an unknown WAV track")
        active_result = _normalize_midi_result_state(
            track_state.get("active_midi_result"),
            request_root,
            expected_audio_path=active_track["audio_path"],
        )
        active_result["source_track_name"] = active_track["name"]
    elif track_state.get("active_midi_result"):
        raise RuntimeError("Linked MIDI result is missing its source WAV track id")

    return {
        "version": 2,
        "mode": mode,
        "request_dir": str(request_root),
        "processing_time": float(track_state.get("processing_time", 0.0)),
        "tracks": normalized_tracks,
        "active_midi_track_id": active_track_id,
        "active_midi_result": active_result,
    }


def _build_track_state(result: SeparationResult, request_dir: str | Path) -> dict:
    request_root = _require_active_request_dir(request_dir)
    result_output_dir = Path(result.output_dir).resolve()
    if result_output_dir != request_root:
        raise RuntimeError(
            "Separation result does not belong to the active Space request: " f"{result_output_dir}"
        )

    if result.mode not in SPLIT_MODE_IDS:
        raise RuntimeError(f"Unsupported separation result mode: {result.mode!r}")
    expected_names = (
        ("vocals", "accompaniment")
        if result.mode == ProcessingMode.VOCAL_SPLIT.value
        else tuple(STEM_KEYS)
    )
    if tuple(result.separated_audio) != expected_names:
        raise RuntimeError(
            "Separated WAV output order is invalid: "
            f"expected={expected_names}, actual={tuple(result.separated_audio)}"
        )

    tracks = []
    for index, track_name in enumerate(expected_names):
        audio_path = _require_owned_request_file(
            request_root,
            result.separated_audio[track_name],
            f"Separated WAV {track_name}",
        )
        tracks.append(
            {
                "id": track_name,
                "name": track_name,
                "audio_path": str(audio_path),
                "color": _TRACK_COLORS[index % len(_TRACK_COLORS)],
                "midi_enabled": False,
                "route": "",
                "muscriptor_instruments": [],
                "status": st("dialogs.complete.audio_tracks.manual_midi.not_selected"),
                "midi_path": "",
            }
        )

    return _normalize_track_state(
        {
            "version": 2,
            "mode": result.mode,
            "request_dir": str(request_root),
            "processing_time": result.processing_time,
            "tracks": tracks,
            "active_midi_track_id": "",
            "active_midi_result": None,
        }
    )


def _build_midi_result_state(
    result,
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    backend_label: str,
    source_track_name: str = "",
    muscriptor_groups: bool = False,
    progress_callback=None,
) -> dict:
    """Build the shared playable MIDI workbench for any transcription backend."""
    from src.core.muscriptor_result_assets import prepare_midi_playback_assets

    assets = prepare_midi_playback_assets(
        result.midi_path,
        audio_path,
        output_dir,
        progress_callback=progress_callback,
        muscriptor_groups=muscriptor_groups,
    )
    detected = list(dict.fromkeys(note.instrument for note in assets.notes))
    selected = list(result.selected_instruments) if muscriptor_groups else detected
    return {
        "kind": "midi_result",
        "audio_path": str(Path(audio_path).resolve()),
        "midi_path": str(Path(result.midi_path).resolve()),
        "selected_instruments": selected,
        "detected_instruments": detected,
        "notes": [
            {
                "instrument": note.instrument,
                "pitch": note.pitch,
                "velocity": note.velocity,
                "start": note.start,
                "end": note.end,
            }
            for note in assets.notes
        ],
        "duration": assets.duration,
        "transcription_wav": str(assets.transcription_wav),
        "stereo_mix_wav": str(assets.stereo_mix_wav),
        "instrument_wavs": {name: str(path) for name, path in assets.instrument_wavs.items()},
        "backend_label": str(backend_label),
        "source_track_name": str(source_track_name),
    }


def _result_backend_label(config: Config) -> str:
    if config.processing_mode == ProcessingMode.SMART.value:
        return {value: label for label, value in BACKEND_CHOICES}[config.transcription_backend]
    return MODE_LABELS[config.processing_mode]


def ensure_model_weights(model_key: str):
    """确保用户选中的 YourMT3+ 官方 checkpoint 已下载。"""
    ensure_yourmt3_code()

    valid_models = {choice.value for choice in YourMT3Model}
    if model_key not in valid_models:
        raise ValueError(f"Unsupported YourMT3 checkpoint: {model_key!r}")

    from src.utils.yourmt3_downloader import download_model, get_model_path

    model_path = get_model_path(model_key)
    if model_path and model_path.exists():
        logger.info("YourMT3+ model found: %s", model_path)
        return

    logger.info("YourMT3+ selected checkpoint missing, downloading: %s", model_key)
    download_model(model_key, progress_callback=lambda _p, msg: logger.info(msg))
    model_path = get_model_path(model_key)
    if not model_path or not model_path.exists():
        raise RuntimeError(f"YourMT3+ checkpoint download produced no model: {model_key}")
    logger.info("YourMT3+ selected checkpoint downloaded: %s", model_path)


def ensure_multistem_weights():
    """确保 BS-RoFormer SW 六轨资源已下载并通过校验。"""
    from download_multistem_model import download_multistem_model

    model_path, config_path = download_multistem_model(printer=logger.info)
    logger.info("BS-RoFormer SW checkpoint ready: %s", model_path)
    logger.info("BS-RoFormer SW config ready: %s", config_path)


def ensure_vocal_split_weights():
    """确保 Leap XE vocals + PolarFormer accompaniment 权重已下载。"""
    from download_accompaniment_model import download_accompaniment_model
    from download_vocal_model import download_vocal_model

    vocal_model = download_vocal_model(printer=logger.info)
    accompaniment_model = download_accompaniment_model(printer=logger.info)
    logger.info("Leap XE vocals checkpoint ready: %s", vocal_model)
    logger.info("PolarFormer accompaniment model ready: %s", accompaniment_model)


def _validate_vocal_split_gpu_runtime() -> None:
    """Validate CUDA runtimes only after ZeroGPU has allocated the worker GPU."""

    import onnxruntime as ort
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Space Vocal Split requires a CUDA GPU; PyTorch reports CUDA unavailable"
        )
    preload_dlls = getattr(ort, "preload_dlls", None)
    if callable(preload_dlls):
        preload_dlls()
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "PolarFormer GPU inference is unavailable: ONNX Runtime providers="
            f"{providers}. Install the pinned onnxruntime-gpu runtime."
        )


def _validate_gpu_runtime_for_request(mode: str) -> None:
    if mode == ProcessingMode.VOCAL_SPLIT.value:
        _validate_vocal_split_gpu_runtime()


def ensure_miros_weights():
    """按需准备 MIROS 源码及两个官方权重。"""
    from download_miros_model import prepare_miros_model

    repo_dir = prepare_miros_model(printer=logger.info)
    logger.info("MIROS source and weights ready: %s", repo_dir)


def ensure_muscriptor_runtime():
    """Install and verify the exact public commit that implements hard masking."""
    from src.core.muscriptor_transcriber import (
        MUSCRIPTOR_SOURCE_COMMIT,
        MUSCRIPTOR_SOURCE_REQUIREMENT,
        MuscriptorTranscriber,
    )

    unavailable = MuscriptorTranscriber._runtime_unavailable_reason()
    if not unavailable:
        return
    logger.info("Installing pinned MuScriptor runtime: %s", unavailable)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-deps",
            "--force-reinstall",
            MUSCRIPTOR_SOURCE_REQUIREMENT,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.stdout:
        logger.info("MuScriptor installer output:\n%s", completed.stdout.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(
            "Pinned MuScriptor runtime installation failed "
            f"for commit {MUSCRIPTOR_SOURCE_COMMIT} (exit={completed.returncode})"
        )
    importlib.invalidate_caches()
    unavailable = MuscriptorTranscriber._runtime_unavailable_reason()
    if unavailable:
        raise RuntimeError(
            "Pinned MuScriptor installation completed but identity/API validation failed: "
            f"{unavailable}"
        )


def ensure_muscriptor_weights():
    """Download the gated, pinned large checkpoint and verify its exact hashes."""
    from src.utils.muscriptor_downloader import download_muscriptor_large_model

    ensure_muscriptor_runtime()
    weights, config = download_muscriptor_large_model(printer=logger.info)
    logger.info("MuScriptor-large checkpoint ready: %s", weights)
    logger.info("MuScriptor-large config ready: %s", config)


def ensure_transkun_v2_aug_weights():
    """按需准备并校验 TransKun V2 Aug 官方 checkpoint。"""
    from download_transkun_v2_aug_model import download_transkun_v2_aug_model

    model_dir = download_transkun_v2_aug_model(printer=logger.info)
    logger.info("TransKun V2 Aug checkpoint ready: %s", model_dir)


ARIA_AMT_RUNTIME_REQUIREMENT = (
    "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/"
    "a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip"
)


def _aria_amt_runtime_available():
    from src.core.aria_amt_transcriber import get_aria_amt_runtime_unavailable_reason

    return get_aria_amt_runtime_unavailable_reason() == ""


def ensure_aria_amt_runtime():
    """Install the pinned Aria-AMT code without letting it replace Space PyTorch."""
    from src.core.aria_amt_transcriber import get_aria_amt_runtime_unavailable_reason

    if _aria_amt_runtime_available():
        return

    unavailable_reason = get_aria_amt_runtime_unavailable_reason()
    logger.info(
        "Installing pinned Aria-AMT runtime without dependencies because identity "
        "validation failed: %s",
        unavailable_reason,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-deps",
            "--force-reinstall",
            ARIA_AMT_RUNTIME_REQUIREMENT,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.stdout:
        logger.info("Aria-AMT installer output:\n%s", completed.stdout.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(
            f"Pinned Aria-AMT runtime installation failed (exit={completed.returncode})"
        )
    importlib.invalidate_caches()
    unavailable_reason = get_aria_amt_runtime_unavailable_reason()
    if unavailable_reason:
        raise RuntimeError(
            "Pinned Aria-AMT installation completed but source identity validation failed: "
            f"{unavailable_reason}"
        )


def ensure_aria_amt_weights():
    """确保 Aria-AMT 钢琴 checkpoint 已下载。"""
    from download_aria_amt_model import download_aria_model, is_aria_model_available

    ensure_aria_amt_runtime()
    if is_aria_model_available():
        logger.info("Aria-AMT checkpoint found")
        return

    logger.info("Aria-AMT checkpoint not found, downloading...")
    download_aria_model()
    logger.info("Aria-AMT checkpoint downloaded")


def ensure_bytedance_piano_weights():
    """确保 ByteDance Piano 带踏板 checkpoint 已下载。"""
    from download_bytedance_piano_model import (
        download_bytedance_piano_model,
        is_bytedance_piano_model_available,
    )

    if is_bytedance_piano_model_available():
        logger.info("ByteDance Piano checkpoint found")
        return

    logger.info("ByteDance Piano checkpoint not found, downloading...")
    download_bytedance_piano_model()
    logger.info("ByteDance Piano checkpoint downloaded")


# Install the pinned code before Gradio begins accepting concurrent requests.
# Failure is fatal and visible; it never downgrades the ZeroGPU PyTorch runtime.
ensure_aria_amt_runtime()

clear_logs()


def get_device_label():
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"
    except Exception:
        return "CPU"


def _require_nonempty_output(path_value, label):
    if not path_value:
        raise RuntimeError(f"Missing required output path: {label}")
    path = Path(path_value)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Required output is missing or empty: {label}={path.resolve()}")
    return str(path)


def _validate_processing_outputs(result, config, request_dir):
    """Validate one direct-conversion MIDI before reporting success."""
    if config.processing_mode in SPLIT_MODE_IDS:
        raise RuntimeError(
            "Split modes are separation-only and cannot use direct MIDI output validation"
        )
    midi_path = _require_owned_request_file(
        request_dir,
        result.midi_path,
        "Direct-conversion MIDI",
    )
    return [str(midi_path)]


def _build_space_request_config(
    mode,
    transcription_backend,
    yourmt3_model,
    muscriptor_instruments=None,
    *,
    vocal_split_merge_midi=False,
    save_separated_tracks=True,
):
    if mode not in MODE_IDS:
        raise RuntimeError(f"Unsupported processing mode: {mode}")

    config = Config()
    config.processing_mode = mode
    config.language = SPACE_LANGUAGE
    config.transcription_backend = transcription_backend
    config.multi_instrument_model = transcription_backend
    config.yourmt3_model = yourmt3_model
    config.muscriptor_instruments = validate_muscriptor_instruments(muscriptor_instruments or [])
    config.vocal_split_merge_midi = bool(
        config.processing_mode == ProcessingMode.VOCAL_SPLIT.value and vocal_split_merge_midi
    )
    config.save_separated_tracks = bool(save_separated_tracks)
    config.validate()
    return config


def _prepare_request_models(
    mode,
    transcription_backend,
    yourmt3_model,
    muscriptor_instruments=None,
) -> None:
    """Strictly prepare only the assets selected for the current job."""

    config = _build_space_request_config(
        mode,
        transcription_backend,
        yourmt3_model,
        muscriptor_instruments,
    )
    if config.processing_mode == ProcessingMode.PIANO_ARIA_AMT.value:
        ensure_aria_amt_weights()
    elif config.processing_mode == ProcessingMode.PIANO_BYTEDANCE_PEDAL.value:
        ensure_bytedance_piano_weights()
    elif config.processing_mode == ProcessingMode.PIANO_TRANSKUN_V2_AUG.value:
        ensure_transkun_v2_aug_weights()
    elif config.processing_mode == ProcessingMode.SMART.value:
        if config.transcription_backend == MultiInstrumentModel.YOURMT3.value:
            ensure_model_weights(config.yourmt3_model)
        elif config.transcription_backend == MultiInstrumentModel.MIROS.value:
            ensure_miros_weights()
        elif config.transcription_backend == MultiInstrumentModel.MUSCRIPTOR.value:
            ensure_muscriptor_weights()
        else:
            raise RuntimeError(
                "Unsupported multi-instrument backend: " f"{config.transcription_backend!r}"
            )
    elif config.processing_mode == ProcessingMode.VOCAL_SPLIT.value:
        ensure_vocal_split_weights()
    elif config.processing_mode == ProcessingMode.SIX_STEM_SPLIT.value:
        ensure_multistem_weights()


_ACTIVE_JOB_LOCK = threading.Lock()
_ACTIVE_JOB = None


def _register_active_job(job) -> None:
    global _ACTIVE_JOB
    with _ACTIVE_JOB_LOCK:
        _ACTIVE_JOB = job


def _unregister_active_job(job) -> None:
    global _ACTIVE_JOB
    if job is None:
        return
    with _ACTIVE_JOB_LOCK:
        if _ACTIVE_JOB is job:
            _ACTIVE_JOB = None


def request_stop_current_job():
    """Ask the running job to stop at the next cooperative checkpoint.

    Mirrors the desktop stop button: MusicToMidiPipeline and
    AudioSeparationService both poll their cancel flag between stages.
    """
    with _ACTIVE_JOB_LOCK:
        job = _ACTIVE_JOB
    if job is None:
        return gr.update()
    job.cancel()
    return st("status.cancelling")


def _convert_impl(
    audio_path,
    mode,
    transcription_backend,
    yourmt3_model,
    muscriptor_instruments,
    progress=gr.Progress(),
):
    """Run one direct audio-to-MIDI mode without creating a track workbench."""
    from src.core.pipeline import MusicToMidiPipeline
    from src.utils.gpu_utils import clear_gpu_memory

    if audio_path is None:
        raise gr.Error(st("space.error.upload_required"))
    if mode in SPLIT_MODE_IDS:
        raise RuntimeError(
            "Split modes must use AudioSeparationService and stop before MIDI conversion"
        )

    clear_logs()
    logger.info("%s: %s", st("space.log.audio_file"), Path(audio_path).name)
    logger.info("%s: %s", st("space.log.processing_mode"), MODE_LABELS.get(mode, mode))

    config = _build_space_request_config(
        mode,
        transcription_backend,
        yourmt3_model,
        muscriptor_instruments,
        vocal_split_merge_midi=False,
        save_separated_tracks=True,
    )
    output_dir = _create_space_output_dir()
    pipeline = MusicToMidiPipeline(config)
    _register_active_job(pipeline)

    def on_progress(p):
        stage_key = STAGE_LABEL_KEYS.get(p.stage)
        stage_name = st(f"main.progress.stages.{stage_key}") if stage_key else str(p.stage)
        progress(p.overall_progress, desc=f"[{stage_name}] {p.message}")

    try:
        result = pipeline.process(
            audio_path=audio_path,
            output_dir=output_dir,
            progress_callback=on_progress,
        )
        output_files = _validate_processing_outputs(result, config, output_dir)
        is_muscriptor = (
            config.transcription_backend == MultiInstrumentModel.MUSCRIPTOR.value
            and config.processing_mode == ProcessingMode.SMART.value
        )
        result_state = _build_midi_result_state(
            result,
            audio_path,
            Path(output_dir) / "midi-playback",
            backend_label=_result_backend_label(config),
            muscriptor_groups=is_muscriptor,
            progress_callback=lambda value, message: progress(
                0.95 + value * 0.05,
                desc=message,
            ),
        )
    except InterruptedError:
        logger.info("Direct conversion cancelled by user")
        try:
            _remove_space_output_dir(output_dir)
        except OSError as cleanup_exc:
            logger.error(
                "Unable to remove cancelled Space output directory %s: %s",
                output_dir,
                cleanup_exc,
            )
        return [], st("status.cancelled"), {}
    except Exception as exc:
        logger.error("Direct conversion failed: %s", exc)
        try:
            _remove_space_output_dir(output_dir)
        except OSError as cleanup_exc:
            logger.error(
                "Unable to remove failed Space output directory %s: %s",
                output_dir,
                cleanup_exc,
            )
        raise gr.Error(st("space.error.conversion_failed", error=exc)) from exc
    finally:
        _unregister_active_job(pipeline)
        try:
            clear_gpu_memory()
        except Exception as cleanup_exc:
            logger.error("Unable to clear GPU memory after direct conversion: %s", cleanup_exc)

    device_label = get_device_label()
    bpm_str = f"{result.beat_info.bpm:.1f}" if result.beat_info else "N/A"
    status_lines = [
        st("space.status.complete_header"),
        f"{st('space.status.elapsed')}: {result.processing_time:.1f} {st('space.status.seconds')}",
        f"{st('space.status.total_notes')}: {result.total_notes}",
        f"{st('dialogs.complete.track_count')}: {len(result.tracks)}",
        f"BPM: {bpm_str}",
        f"{st('space.status.device')}: {device_label}",
        f"{st('space.status.midi_file')}: {Path(result.midi_path).name}",
        st(
            "space.status.output_retention",
            hours=SPACE_OUTPUT_RETENTION_SECONDS / 3600,
        ),
    ]
    logger.info(st("space.log.complete"))
    return output_files, "\n".join(status_lines), result_state


def _separate_impl(
    audio_path,
    mode,
    transcription_backend,
    yourmt3_model,
    progress=gr.Progress(),
):
    """Separate two or six WAV tracks and return a request-owned workbench state."""
    from src.utils.gpu_utils import clear_gpu_memory

    if audio_path is None:
        raise gr.Error(st("space.error.upload_required"))
    if mode not in SPLIT_MODE_IDS:
        raise RuntimeError(f"Unsupported separation-only mode: {mode!r}")

    clear_logs()
    logger.info("%s: %s", st("space.log.audio_file"), Path(audio_path).name)
    logger.info("%s: %s", st("space.log.processing_mode"), MODE_LABELS[mode])

    config = _build_space_request_config(
        mode,
        transcription_backend,
        yourmt3_model,
        vocal_split_merge_midi=False,
        save_separated_tracks=True,
    )
    output_dir = _create_space_output_dir()
    separation = None

    def on_progress(p):
        stage_key = STAGE_LABEL_KEYS.get(p.stage)
        stage_name = st(f"main.progress.stages.{stage_key}") if stage_key else str(p.stage)
        progress(p.overall_progress, desc=f"[{stage_name}] {p.message}")

    try:
        separation = AudioSeparationService(
            config,
            progress_callback=on_progress,
        )
        _register_active_job(separation)
        result = separation.process(audio_path=audio_path, output_dir=output_dir)
        track_state = _build_track_state(result, output_dir)
    except InterruptedError:
        logger.info("WAV separation cancelled by user")
        try:
            _remove_space_output_dir(output_dir)
        except OSError as cleanup_exc:
            logger.error(
                "Unable to remove cancelled Space output directory %s: %s",
                output_dir,
                cleanup_exc,
            )
        return [], st("status.cancelled"), {}
    except Exception as exc:
        logger.error("WAV separation failed: %s", exc)
        try:
            _remove_space_output_dir(output_dir)
        except OSError as cleanup_exc:
            logger.error(
                "Unable to remove failed Space output directory %s: %s",
                output_dir,
                cleanup_exc,
            )
        raise gr.Error(st("space.error.conversion_failed", error=exc)) from exc
    finally:
        _unregister_active_job(separation)
        try:
            clear_gpu_memory()
        except Exception as cleanup_exc:
            logger.error("Unable to clear GPU memory after separation: %s", cleanup_exc)

    output_files = [track["audio_path"] for track in track_state["tracks"]]
    wav_lines = [
        f"  • {track['name']}: {Path(track['audio_path']).name}" for track in track_state["tracks"]
    ]
    status_lines = [
        st("dialogs.complete.audio_tracks.separation_result_title"),
        f"{st('dialogs.complete.audio_tracks.separation_mode')}: {MODE_LABELS[mode]}",
        f"{st('space.status.elapsed')}: {result.processing_time:.1f} {st('space.status.seconds')}",
        f"{st('space.status.separated_audio')}: {len(output_files)}",
        st("dialogs.complete.separated_audio") + ":",
        *wav_lines,
        st("dialogs.complete.audio_tracks.separation_manual_hint"),
        st(
            "space.status.output_retention",
            hours=SPACE_OUTPUT_RETENTION_SECONDS / 3600,
        ),
    ]
    logger.info(st("progress.separation_only_complete", seconds=f"{result.processing_time:.1f}"))
    return output_files, "\n".join(status_lines), track_state


ZERO_GPU_FREE_ACCOUNT_BUDGET_SECONDS = 300
# spaces==0.51.0 applies a 1.5 duration factor to the default Blackwell
# ``large`` allocation.  Treat that as the pinned-package upper bound so the
# dynamic request cannot exceed one logged-in free-account window; H200's 1.0
# factor is therefore admitted conservatively.
ZERO_GPU_LARGE_DURATION_FACTOR = 1.5
ZERO_GPU_BASE_RUNTIME_SECONDS = 180.0
ZERO_GPU_MODE_RUNTIME_MULTIPLIERS = {
    ProcessingMode.SMART.value: 8.0,
    ProcessingMode.VOCAL_SPLIT.value: 30.0,
    ProcessingMode.SIX_STEM_SPLIT.value: 72.0,
    ProcessingMode.PIANO_TRANSKUN.value: 8.0,
    ProcessingMode.PIANO_TRANSKUN_V2_AUG.value: 8.0,
    ProcessingMode.PIANO_ARIA_AMT.value: 8.0,
    ProcessingMode.PIANO_BYTEDANCE_PEDAL.value: 8.0,
}
ZERO_GPU_MULTI_BACKEND_RUNTIME_FACTORS = {
    MultiInstrumentModel.YOURMT3.value: 1.0,
    MultiInstrumentModel.MIROS.value: 2.5,
    MultiInstrumentModel.MUSCRIPTOR.value: 3.0,
}
ZERO_GPU_YOURMT3_MODEL_RUNTIME_FACTORS = {
    YourMT3Model.YMT3_PLUS.value: 1.0,
    YourMT3Model.YPTF_SINGLE_NOPS.value: 1.0,
    YourMT3Model.YPTF_MULTI_PS.value: 1.1,
    YourMT3Model.YPTF_MOE_MULTI_NOPS.value: 1.25,
    YourMT3Model.YPTF_MOE_MULTI_PS.value: 1.3,
}


def _estimate_zerogpu_duration(
    audio_path,
    mode,
    transcription_backend,
    yourmt3_model,
    muscriptor_instruments=None,
    vocal_split_merge_midi=False,
    save_separated_tracks=True,
    progress=None,
):
    """Admit only short clips that fit one logged-in free-account GPU window.

    This is a conservative admission contract, not a promise that remaining
    daily quota or queue capacity is available.  Long songs must use Colab,
    the desktop build, or dedicated GPU hardware.
    """
    del muscriptor_instruments, vocal_split_merge_midi, save_separated_tracks, progress
    if audio_path is None:
        return 60
    if mode not in MODE_IDS:
        raise RuntimeError(f"Unsupported processing mode: {mode}")

    import librosa

    duration_seconds = float(librosa.get_duration(path=audio_path))
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise RuntimeError(f"Unable to estimate a valid audio duration for ZeroGPU: {audio_path}")
    runtime_factor = 1.0
    if mode in MULTI_INSTRUMENT_MODE_IDS:
        if transcription_backend not in ZERO_GPU_MULTI_BACKEND_RUNTIME_FACTORS:
            raise RuntimeError(f"Unsupported multi-instrument backend: {transcription_backend!r}")
        runtime_factor = ZERO_GPU_MULTI_BACKEND_RUNTIME_FACTORS[transcription_backend]
        if transcription_backend == MultiInstrumentModel.YOURMT3.value:
            if yourmt3_model not in ZERO_GPU_YOURMT3_MODEL_RUNTIME_FACTORS:
                raise RuntimeError(f"Unsupported YourMT3 checkpoint: {yourmt3_model!r}")
            runtime_factor *= ZERO_GPU_YOURMT3_MODEL_RUNTIME_FACTORS[yourmt3_model]

    raw_estimated_seconds = max(
        ZERO_GPU_BASE_RUNTIME_SECONDS,
        duration_seconds * ZERO_GPU_MODE_RUNTIME_MULTIPLIERS[mode] * runtime_factor
        + ZERO_GPU_BASE_RUNTIME_SECONDS,
    )
    scheduled_seconds = int(math.ceil(raw_estimated_seconds))
    effective_seconds = scheduled_seconds * ZERO_GPU_LARGE_DURATION_FACTOR
    if effective_seconds > ZERO_GPU_FREE_ACCOUNT_BUDGET_SECONDS:
        raise gr.Error(
            st(
                "space.error.zerogpu_clip_too_long",
                duration=duration_seconds,
                estimate=effective_seconds,
                limit=ZERO_GPU_FREE_ACCOUNT_BUDGET_SECONDS,
            )
        )
    return scheduled_seconds


GPU_CONCURRENCY_ID = "music-to-midi-gpu"


if ZERO_GPU:

    @spaces.GPU(duration=_estimate_zerogpu_duration, size="large")
    def _convert_audio_to_midi_on_gpu(
        audio_path,
        mode,
        transcription_backend,
        yourmt3_model,
        muscriptor_instruments,
        progress=gr.Progress(),
    ):
        _validate_gpu_runtime_for_request(mode)
        if mode in SPLIT_MODE_IDS:
            return _separate_impl(
                audio_path,
                mode,
                transcription_backend,
                yourmt3_model,
                progress=progress,
            )
        return _convert_impl(
            audio_path,
            mode,
            transcription_backend,
            yourmt3_model,
            muscriptor_instruments,
            progress=progress,
        )

else:

    def _convert_audio_to_midi_on_gpu(
        audio_path,
        mode,
        transcription_backend,
        yourmt3_model,
        muscriptor_instruments,
        progress=gr.Progress(),
    ):
        _validate_gpu_runtime_for_request(mode)
        if mode in SPLIT_MODE_IDS:
            return _separate_impl(
                audio_path,
                mode,
                transcription_backend,
                yourmt3_model,
                progress=progress,
            )
        return _convert_impl(
            audio_path,
            mode,
            transcription_backend,
            yourmt3_model,
            muscriptor_instruments,
            progress=progress,
        )


def convert_audio_to_midi(
    audio_path,
    mode,
    transcription_backend,
    yourmt3_model,
    muscriptor_instruments=None,
    progress=gr.Progress(),
):
    """Prepare exactly one selected primary job before requesting a GPU slot."""
    if audio_path is None:
        raise gr.Error(st("space.error.upload_required"))
    if mode not in MODE_IDS:
        raise gr.Error(f"Unsupported processing mode: {mode!r}")
    if ZERO_GPU:
        _estimate_zerogpu_duration(
            audio_path,
            mode,
            transcription_backend,
            yourmt3_model,
            muscriptor_instruments,
            progress=progress,
        )
    _prepare_request_models(
        mode,
        transcription_backend,
        yourmt3_model,
        muscriptor_instruments,
    )
    return _convert_audio_to_midi_on_gpu(
        audio_path,
        mode,
        transcription_backend,
        yourmt3_model,
        muscriptor_instruments,
        progress=progress,
    )


if ZERO_GPU:
    setattr(convert_audio_to_midi, "zerogpu", None)


def _manual_route_config(route: str, muscriptor_instruments=None) -> Config:
    if route not in MANUAL_MIDI_ROUTES:
        raise RuntimeError(f"Unsupported manual MIDI route: {route!r}")
    base_config = Config()
    base_config.language = SPACE_LANGUAGE
    return build_manual_midi_config(
        base_config,
        route,
        muscriptor_instruments=validate_muscriptor_instruments(muscriptor_instruments or []),
    )


def _estimate_manual_zerogpu_duration(
    audio_path,
    request_dir,
    track_id,
    route,
    muscriptor_instruments=None,
    progress=None,
):
    del progress
    if not str(track_id).strip():
        raise RuntimeError("Manual MIDI track identity is missing")
    audio_file = _require_owned_request_file(
        request_dir,
        audio_path,
        f"Manual MIDI input {track_id}",
    )
    config = _manual_route_config(str(route), muscriptor_instruments)
    return _estimate_zerogpu_duration(
        str(audio_file),
        config.processing_mode,
        config.transcription_backend,
        config.yourmt3_model,
        config.muscriptor_instruments,
    )


def _convert_manual_midi_impl(
    audio_path,
    request_dir,
    track_id,
    route,
    muscriptor_instruments,
    progress=gr.Progress(),
):
    from src.core.pipeline import MusicToMidiPipeline
    from src.utils.gpu_utils import clear_gpu_memory

    request_root = _require_active_request_dir(request_dir)
    audio_file = _require_owned_request_file(
        request_root,
        audio_path,
        f"Manual MIDI input {track_id}",
    )
    config = _manual_route_config(str(route), muscriptor_instruments)
    output_dir = _require_owned_request_output_dir(
        request_root,
        manual_midi_output_dir(audio_file, str(route)),
    )

    def on_progress(p):
        stage_key = STAGE_LABEL_KEYS.get(p.stage)
        stage_name = st(f"main.progress.stages.{stage_key}") if stage_key else str(p.stage)
        progress(p.overall_progress, desc=f"[{stage_name}] {p.message}")

    pipeline = MusicToMidiPipeline(config)
    _register_active_job(pipeline)
    try:
        result = pipeline.process(
            audio_path=str(audio_file),
            output_dir=str(output_dir),
            progress_callback=on_progress,
        )
        output_files = _validate_processing_outputs(result, config, request_root)
        if len(output_files) != 1:
            raise RuntimeError(
                f"Manual MIDI conversion returned {len(output_files)} files instead of one"
            )
        is_muscriptor = config.transcription_backend == MultiInstrumentModel.MUSCRIPTOR.value
        result_state = _build_midi_result_state(
            result,
            audio_file,
            output_dir / "midi-playback",
            backend_label=_manual_midi_route_label(str(route)),
            muscriptor_groups=is_muscriptor,
            progress_callback=lambda value, message: progress(
                0.95 + value * 0.05,
                desc=message,
            ),
        )
        return output_files[0], result_state
    finally:
        _unregister_active_job(pipeline)
        try:
            clear_gpu_memory()
        except Exception as cleanup_exc:
            logger.error("Unable to clear GPU memory after manual MIDI: %s", cleanup_exc)


if ZERO_GPU:

    @spaces.GPU(duration=_estimate_manual_zerogpu_duration, size="large")
    def _convert_manual_midi_on_gpu(
        audio_path,
        request_dir,
        track_id,
        route,
        muscriptor_instruments,
        progress=gr.Progress(),
    ):
        config = _manual_route_config(str(route), muscriptor_instruments)
        _validate_gpu_runtime_for_request(config.processing_mode)
        return _convert_manual_midi_impl(
            audio_path,
            request_dir,
            track_id,
            route,
            muscriptor_instruments,
            progress=progress,
        )

else:

    def _convert_manual_midi_on_gpu(
        audio_path,
        request_dir,
        track_id,
        route,
        muscriptor_instruments,
        progress=gr.Progress(),
    ):
        config = _manual_route_config(str(route), muscriptor_instruments)
        _validate_gpu_runtime_for_request(config.processing_mode)
        return _convert_manual_midi_impl(
            audio_path,
            request_dir,
            track_id,
            route,
            muscriptor_instruments,
            progress=progress,
        )


def _convert_one_track(
    track_state,
    track_id,
    midi_enabled,
    route,
    muscriptor_instruments,
    progress=gr.Progress(),
):
    state = _normalize_track_state(track_state)
    if not state:
        raise gr.Error("The separation result has expired or is unavailable")
    selected_track = next(
        (track for track in state["tracks"] if track["id"] == str(track_id)),
        None,
    )
    if selected_track is None:
        raise gr.Error(f"Unknown audio track: {track_id!r}")
    if not midi_enabled:
        raise gr.Error(st("dialogs.complete.audio_tracks.manual_midi.not_selected"))
    if route not in MANUAL_MIDI_ROUTES:
        raise gr.Error(st("dialogs.complete.audio_tracks.manual_midi.model_required"))

    selected_instruments = validate_muscriptor_instruments(muscriptor_instruments or [])
    config = _manual_route_config(str(route), selected_instruments)
    try:
        if ZERO_GPU:
            _estimate_manual_zerogpu_duration(
                selected_track["audio_path"],
                state["request_dir"],
                selected_track["id"],
                route,
                selected_instruments,
                progress=progress,
            )
        _prepare_request_models(
            config.processing_mode,
            config.transcription_backend,
            config.yourmt3_model,
            config.muscriptor_instruments,
        )
        midi_path, midi_result = _convert_manual_midi_on_gpu(
            selected_track["audio_path"],
            state["request_dir"],
            selected_track["id"],
            route,
            selected_instruments,
            progress=progress,
        )
    except InterruptedError:
        logger.info("Per-track MIDI conversion cancelled for %s", track_id)
        cancelled_tracks = []
        for track in state["tracks"]:
            updated = dict(track)
            if track["id"] == selected_track["id"]:
                updated["status"] = st("dialogs.complete.audio_tracks.manual_midi.cancelled")
            cancelled_tracks.append(updated)
        return _normalize_track_state({**state, "tracks": cancelled_tracks})
    except Exception as exc:
        logger.error("Per-track MIDI conversion failed for %s: %s", track_id, exc)
        raise gr.Error(st("dialogs.complete.audio_tracks.manual_midi.failed", error=exc)) from exc

    updated_tracks = []
    for track in state["tracks"]:
        updated = dict(track)
        if track["id"] == selected_track["id"]:
            updated.update(
                {
                    "midi_enabled": True,
                    "route": str(route),
                    "muscriptor_instruments": (
                        selected_instruments if str(route) == MIDI_ROUTE_MUSCRIPTOR else []
                    ),
                    "midi_path": str(midi_path),
                    "status": st(
                        "dialogs.complete.audio_tracks.manual_midi.complete",
                        file=Path(midi_path).name,
                    ),
                }
            )
        updated_tracks.append(updated)
    return _normalize_track_state(
        {
            **state,
            "tracks": updated_tracks,
            "active_midi_track_id": selected_track["id"],
            "active_midi_result": {
                **midi_result,
                "source_track_name": selected_track["name"],
            },
        }
    )


if ZERO_GPU:
    setattr(_convert_one_track, "zerogpu", None)


def _track_control_updates(enabled, route):
    is_enabled = bool(enabled)
    normalized_route = str(route or "")
    if not is_enabled:
        status = st("dialogs.complete.audio_tracks.manual_midi.not_selected")
    elif normalized_route not in MANUAL_MIDI_ROUTES:
        status = st("dialogs.complete.audio_tracks.manual_midi.model_required")
    else:
        status = st(
            "dialogs.complete.audio_tracks.manual_midi.selected",
            model=_manual_midi_route_label(normalized_route),
        )
    return (
        gr.update(interactive=is_enabled),
        gr.update(interactive=is_enabled and normalized_route in MANUAL_MIDI_ROUTES),
        status,
        gr.update(
            visible=normalized_route == MIDI_ROUTE_MUSCRIPTOR,
            interactive=is_enabled and normalized_route == MIDI_ROUTE_MUSCRIPTOR,
        ),
    )


def _add_audio_tracks(uploaded_files, track_state):
    state = _normalize_track_state(track_state)
    if not state:
        raise gr.Error("A separation result is required before adding audio tracks")
    raw_files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    if not raw_files or raw_files == [None]:
        raise gr.Error(st("space.error.upload_required"))

    validated_sources = []
    for raw_file in raw_files:
        raw_path = (
            raw_file if isinstance(raw_file, (str, Path)) else getattr(raw_file, "name", None)
        )
        if not raw_path:
            raise gr.Error("An added audio upload does not expose a file path")
        source = Path(raw_path).resolve()
        if source.suffix.lower() not in _SUPPORTED_AUDIO_SUFFIXES:
            raise gr.Error(
                st(
                    "dialogs.complete.audio_tracks.unsupported_format",
                    path=source,
                    formats=", ".join(sorted(_SUPPORTED_AUDIO_SUFFIXES)),
                )
            )
        if not source.is_file() or source.stat().st_size <= 0:
            raise gr.Error(st("dialogs.complete.audio_tracks.missing_file", path=source))
        validated_sources.append(source)

    request_root = _require_active_request_dir(state["request_dir"])
    destination_dir = request_root / "added_tracks"
    destination_dir.mkdir(parents=True, exist_ok=True)
    tracks = [dict(track) for track in state["tracks"]]
    existing_ids = {track["id"] for track in tracks}

    for source in validated_sources:
        safe_stem = (
            "".join(
                char for char in source.stem if char.isalnum() or char in {"-", "_", " "}
            ).strip()
            or "audio"
        )
        destination = destination_dir / f"{safe_stem}{source.suffix.lower()}"
        collision_index = 2
        while destination.exists():
            destination = destination_dir / (
                f"{safe_stem}_{collision_index}{source.suffix.lower()}"
            )
            collision_index += 1
        shutil.copy2(source, destination)
        copied = _require_owned_request_file(
            request_root,
            destination,
            f"Added audio track {source.name}",
        )

        track_index = len(tracks) + 1
        track_id = f"local_{track_index}"
        while track_id in existing_ids:
            track_index += 1
            track_id = f"local_{track_index}"
        existing_ids.add(track_id)
        tracks.append(
            {
                "id": track_id,
                "name": safe_stem,
                "audio_path": str(copied),
                "color": _TRACK_COLORS[(len(tracks)) % len(_TRACK_COLORS)],
                "midi_enabled": False,
                "route": "",
                "muscriptor_instruments": [],
                "status": st("dialogs.complete.audio_tracks.manual_midi.not_selected"),
                "midi_path": "",
            }
        )

    return _normalize_track_state({**state, "tracks": tracks}), None


def _remove_track(track_state, track_id):
    """Remove one track from the timeline, mirroring the desktop mixer.

    Removing the last track is allowed; the workbench then renders the same
    empty-timeline state as the desktop widget.
    """
    state = _normalize_track_state(track_state)
    if not state:
        raise gr.Error("The separation result has expired or is unavailable")
    target_id = str(track_id)
    remaining = [track for track in state["tracks"] if track["id"] != target_id]
    if len(remaining) == len(state["tracks"]):
        raise gr.Error(f"Unknown audio track: {target_id!r}")
    updates = {**state, "tracks": remaining}
    if state.get("active_midi_track_id") == target_id:
        updates["active_midi_track_id"] = ""
        updates["active_midi_result"] = None
    return _normalize_track_state(updates)


def _close_active_midi_detail(track_state):
    state = _normalize_track_state(track_state)
    if not state:
        return {}
    return _normalize_track_state(
        {
            **state,
            "active_midi_track_id": "",
            "active_midi_result": None,
        }
    )


def _clear_result_state():
    return {}


def update_mode_info(mode):
    if mode not in MODE_IDS:
        raise RuntimeError(f"Unsupported processing mode: {mode}")
    info = (
        f"**{MODE_LABELS[mode]}**\n\n"
        f"{st(f'main.mode.{mode}_desc')}\n\n"
        f"{st(f'main.mode.{mode}_hint')}"
    )
    if mode in SPLIT_MODE_IDS:
        info += "\n\n" + st("dialogs.complete.audio_tracks.separation_manual_hint")
    return info


def _main_action_label(mode):
    if mode in SPLIT_MODE_IDS:
        return "▶  " + st("toolbar.start_separation")
    return "▶  " + st("toolbar.start_convert")


def update_mode_controls(mode, transcription_backend):
    if mode not in MODE_IDS:
        raise RuntimeError(f"Unsupported processing mode: {mode}")
    if transcription_backend not in {model.value for model in MultiInstrumentModel}:
        raise RuntimeError(f"Unsupported multi-instrument backend: {transcription_backend!r}")
    uses_global_backend = mode == ProcessingMode.SMART.value
    shows_yourmt3_model = (
        uses_global_backend and transcription_backend == MultiInstrumentModel.YOURMT3.value
    )
    shows_muscriptor_instruments = (
        uses_global_backend and transcription_backend == MultiInstrumentModel.MUSCRIPTOR.value
    )
    return (
        update_mode_info(mode),
        gr.update(visible=uses_global_backend),
        gr.update(visible=shows_yourmt3_model),
        gr.update(visible=shows_muscriptor_instruments),
        gr.update(value=_main_action_label(mode)),
    )


def update_backend_controls(mode, transcription_backend):
    if mode not in MODE_IDS:
        raise RuntimeError(f"Unsupported processing mode: {mode}")
    if transcription_backend not in {model.value for model in MultiInstrumentModel}:
        raise RuntimeError(f"Unsupported multi-instrument backend: {transcription_backend!r}")
    uses_smart = mode == ProcessingMode.SMART.value
    return (
        gr.update(
            visible=(uses_smart and transcription_backend == MultiInstrumentModel.YOURMT3.value)
        ),
        gr.update(
            visible=(uses_smart and transcription_backend == MultiInstrumentModel.MUSCRIPTOR.value)
        ),
    )


CUSTOM_CSS = """
.gradio-container {
    background: #1a1a2e !important;
    max-width: 1100px !important;
}
.app-header {
    background: #16213e;
    border-bottom: 2px solid #2a2a4a;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 12px;
}
.app-header h1 {
    color: #e0e0e0 !important;
    font-size: 22px !important;
    margin: 0 !important;
}
.app-header p {
    color: #8892a0 !important;
    font-size: 13px !important;
    margin: 4px 0 0 0 !important;
}
.upload-zone {
    background: #1f2940 !important;
    border: 2px dashed #3a4a6a !important;
    border-radius: 16px !important;
    min-height: 120px !important;
}
.convert-btn {
    background: #4a9eff !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 15px !important;
    padding: 12px 32px !important;
    border-radius: 10px !important;
    border: none !important;
    min-height: 48px !important;
}
.result-box textarea {
    background: #16213e !important;
    color: #e0e0e0 !important;
    border: 1px solid #3a4a6a !important;
    border-radius: 8px !important;
    font-family: 'Consolas', 'Ubuntu Mono', monospace !important;
    font-size: 13px !important;
}
.log-box textarea {
    background: #0d1117 !important;
    color: #8dc891 !important;
    border: 1px solid #2a3a4a !important;
    border-radius: 8px !important;
    font-family: 'Consolas', 'Ubuntu Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}
.section-title {
    color: #e0e0e0 !important;
    font-weight: bold !important;
    font-size: 13px !important;
    border-bottom: 1px solid #3a4a6a;
    padding-bottom: 6px;
    margin-bottom: 10px;
}
.mode-info {
    background: #16213e;
    border: 1px solid #3a4a6a;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 8px;
}
.device-badge {
    background: #16213e;
    border: 1px solid #3a4a6a;
    border-radius: 6px;
    padding: 6px 12px;
    text-align: center;
}
.track-workbench {
    background: #101c35 !important;
    border: 1px solid #365f8d !important;
    border-radius: 12px !important;
    margin-top: 16px !important;
    padding: 12px !important;
}
.track-card {
    background: #172b4a !important;
    border: 1px solid #365f8d !important;
    border-radius: 10px !important;
    margin: 10px 0 !important;
    padding: 10px !important;
}
.track-card audio {
    width: 100% !important;
}
.track-midi-status {
    color: #9fbde2 !important;
    font-size: 12px !important;
}
.footer-info {
    text-align: center;
    color: #6a7a8a !important;
    font-size: 12px;
    border-top: 1px solid #2a2a4a;
    padding-top: 12px;
    margin-top: 16px;
}
"""

LOG_POLL_HEAD = """<script>
(function() {
    var pollCount = 0;
    var _pollTimer = setInterval(function() {
        pollCount++;
        var ta = document.querySelector('.log-box textarea');
        if (!ta) return;
        var setter = Object.getOwnPropertyDescriptor(
            HTMLTextAreaElement.prototype, 'value'
        ).set;
        fetch('./api/read_logs', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({data: []})
        })
        .then(function(r) { return r.json(); })
        .then(function(json) {
            var logText = (json.data && json.data[0]) ? json.data[0] : '';
            setter.call(ta, logText || '[poll #' + pollCount + '] waiting for logs...');
            ta.dispatchEvent(new Event('input', {bubbles: true}));
            ta.scrollTop = ta.scrollHeight;
        })
        .catch(function(err) {
            setter.call(ta, '[poll error] ' + err.message);
            ta.dispatchEvent(new Event('input', {bubbles: true}));
        });
    }, 2000);
})();
</script>"""

DEVICE_LABEL = st("space.ui.zerogpu_device") if ZERO_GPU else get_device_label()
ZERO_GPU_NOTE = st("space.ui.zerogpu_note") if ZERO_GPU else ""

with gr.Blocks(
    title=st("space.app.title"),
    css=CUSTOM_CSS,
    head=LOG_POLL_HEAD + mixer_head() + muscriptor_result_head(),
    delete_cache=(3600, SPACE_OUTPUT_RETENTION_SECONDS),
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=["system-ui", "Noto Sans SC", "sans-serif"],
    ).set(
        body_background_fill="#1a1a2e",
        block_background_fill="#1f2940",
        block_border_color="#3a4a6a",
        block_label_text_color="#b0b8c8",
        block_title_text_color="#e0e0e0",
        input_background_fill="#16213e",
        input_border_color="#3a4a6a",
        button_primary_background_fill="#4a9eff",
        button_primary_text_color="white",
        button_secondary_background_fill="#2a3f5f",
        button_secondary_text_color="#e0e0e0",
    ),
) as demo:
    track_state = gr.State({})

    with gr.Group(elem_classes="app-header"):
        gr.Markdown(f"# 🎵 {st('space.app.title')}\n{st('space.app.subtitle')}")

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            gr.Markdown(
                f"**{st('space.ui.audio_section')}**",
                elem_classes="section-title",
            )
            audio_input = gr.Audio(
                label=st("space.ui.audio_input"),
                type="filepath",
                sources=["upload"],
                editable=False,
                elem_classes="upload-zone",
            )
            gr.Markdown(f"<small style='color:#6a7a8a'>{st('space.ui.audio_hint')}</small>")

            gr.Markdown(
                f"**{st('space.ui.track_section')}**",
                elem_classes="section-title",
            )
            mode_radio = gr.Radio(
                choices=MODE_CHOICES,
                value=ProcessingMode.SMART.value,
                label=st("space.ui.mode_label"),
            )
            mode_info = gr.Markdown(
                update_mode_info(ProcessingMode.SMART.value),
                elem_classes="mode-info",
            )
            transcription_backend = gr.Radio(
                choices=BACKEND_CHOICES,
                value=MultiInstrumentModel.YOURMT3.value,
                label=st("main.engine.active_label"),
                visible=True,
            )
            yourmt3_model = gr.Dropdown(
                choices=YOURMT3_MODEL_CHOICES,
                value=YourMT3Model.YPTF_MOE_MULTI_NOPS.value,
                label=st("main.engine.yourmt3_model_label"),
                visible=True,
            )
            muscriptor_instruments = gr.Dropdown(
                choices=MUSCRIPTOR_INSTRUMENT_CHOICES,
                value=[],
                multiselect=True,
                filterable=True,
                label=st("main.engine.muscriptor_instruments_title"),
                info=st("main.engine.muscriptor_instruments_desc"),
                visible=False,
                elem_classes=["muscriptor-instrument-selector"],
            )

            with gr.Row():
                convert_btn = gr.Button(
                    _main_action_label(ProcessingMode.SMART.value),
                    variant="primary",
                    elem_classes="convert-btn",
                    size="lg",
                    scale=4,
                )
                stop_btn = gr.Button(
                    "■  " + st("toolbar.stop"),
                    variant="stop",
                    size="lg",
                    scale=1,
                )

            mode_radio.change(
                fn=update_mode_controls,
                inputs=[mode_radio, transcription_backend],
                outputs=[
                    mode_info,
                    transcription_backend,
                    yourmt3_model,
                    muscriptor_instruments,
                    convert_btn,
                ],
                api_name=False,
                queue=False,
            )
            transcription_backend.change(
                fn=update_backend_controls,
                inputs=[mode_radio, transcription_backend],
                outputs=[yourmt3_model, muscriptor_instruments],
                api_name=False,
                queue=False,
            )

            gr.Markdown(
                f"{st('space.ui.device')}: **{DEVICE_LABEL}**{ZERO_GPU_NOTE}",
                elem_classes="device-badge",
            )

        with gr.Column(scale=5):
            gr.Markdown(
                f"**{st('space.ui.result_section')}**",
                elem_classes="section-title",
            )
            status_output = gr.Textbox(
                label=st("space.ui.status_label"),
                interactive=False,
                lines=7,
                placeholder=st("space.ui.status_placeholder"),
                elem_classes="result-box",
            )

            gr.Markdown(
                f"**{st('space.ui.download_section')}**",
                elem_classes="section-title",
            )
            file_output = gr.File(
                label=st("space.ui.download_label"),
                file_count="multiple",
            )

            gr.Markdown(
                f"**{st('space.ui.logs_section')}**",
                elem_classes="section-title",
            )
            log_output = gr.Textbox(
                label=st("space.ui.logs_label"),
                interactive=False,
                lines=12,
                max_lines=20,
                placeholder=st("space.ui.logs_placeholder"),
                elem_classes="log-box",
            )

    @gr.render(inputs=[track_state, mode_radio])
    def render_track_workbench(current_state, selected_mode):
        if not current_state:
            return
        if current_state.get("kind") in {"midi_result", "muscriptor_result"}:
            if selected_mode in SPLIT_MODE_IDS:
                return
            with gr.Group(elem_classes="track-workbench"):
                gr.Markdown(f"### {current_state.get('backend_label', MODE_LABELS[selected_mode])}")
                gr.HTML(
                    build_muscriptor_result_html(
                        current_state,
                        st,
                        SPACE_LANGUAGE,
                    ),
                    key="muscriptor-result-workbench",
                )
                another = gr.Button(
                    st("muscriptor_result.another"),
                    key="muscriptor-transcribe-another",
                )
                another.click(
                    fn=_clear_result_state,
                    inputs=None,
                    outputs=[track_state],
                    api_name=False,
                    queue=False,
                )
            return
        if selected_mode not in SPLIT_MODE_IDS:
            return
        state = _normalize_track_state(current_state)
        if state["mode"] != selected_mode:
            return

        with gr.Group(elem_classes="track-workbench"):
            gr.Markdown(
                f"## {st('dialogs.complete.audio_tracks.title')}\n\n"
                f"{st('dialogs.complete.audio_tracks.subtitle')}"
            )
            add_audio = gr.File(
                label=st("dialogs.complete.audio_tracks.add_track"),
                file_count="multiple",
                file_types=sorted(_SUPPORTED_AUDIO_SUFFIXES),
                type="filepath",
                key="add-audio-tracks",
            )
            add_audio.change(
                fn=_add_audio_tracks,
                inputs=[add_audio, track_state],
                outputs=[track_state, add_audio],
                api_name=False,
                queue=False,
            )

            # Shared browser mixer: same transport, playhead, mute/solo,
            # volume, offset, zoom/fit/align controls as the desktop widget.
            gr.HTML(
                build_track_mixer_html(state["tracks"], st),
                key="track-mixer",
            )

            for track in state["tracks"]:
                track_id_state = gr.State(track["id"])
                route_selected = track["route"] in MANUAL_MIDI_ROUTES
                with gr.Group(
                    elem_classes="track-card",
                ):
                    with gr.Row(equal_height=True):
                        gr.Markdown(
                            f"### ♪ <span style='color:{track['color']}'>"
                            f"{_display_track_name(track['name'])}</span>\n"
                            f"<small>{Path(track['audio_path']).name}</small>"
                        )
                        remove_track = gr.Button(
                            st("dialogs.complete.audio_tracks.remove"),
                            variant="stop",
                            size="sm",
                            scale=0,
                            key=f"remove-{track['id']}",
                        )
                    with gr.Row(equal_height=True):
                        midi_enabled = gr.Checkbox(
                            value=track["midi_enabled"],
                            label=st("dialogs.complete.audio_tracks.manual_midi.enable"),
                            key=f"midi-enabled-{track['id']}",
                        )
                        midi_route = gr.Dropdown(
                            choices=MANUAL_MIDI_ROUTE_CHOICES,
                            value=track["route"] or None,
                            label=st("dialogs.complete.audio_tracks.manual_midi.select_model"),
                            interactive=track["midi_enabled"],
                            scale=5,
                            key=f"midi-route-{track['id']}",
                        )
                        start_midi = gr.Button(
                            st("dialogs.complete.audio_tracks.manual_midi.start"),
                            variant="primary",
                            interactive=bool(track["midi_enabled"] and route_selected),
                            key=f"midi-start-{track['id']}",
                        )
                    midi_instruments = gr.Dropdown(
                        choices=MUSCRIPTOR_INSTRUMENT_CHOICES,
                        value=track.get("muscriptor_instruments", []),
                        multiselect=True,
                        filterable=True,
                        label=st("main.engine.muscriptor_instruments_title"),
                        info=st("main.engine.muscriptor_instruments_desc"),
                        visible=track["route"] == MIDI_ROUTE_MUSCRIPTOR,
                        interactive=bool(
                            track["midi_enabled"] and track["route"] == MIDI_ROUTE_MUSCRIPTOR
                        ),
                        elem_classes=["muscriptor-instrument-selector"],
                        key=f"midi-instruments-{track['id']}",
                    )
                    midi_status = gr.Markdown(
                        track["status"],
                        elem_classes="track-midi-status",
                        key=f"midi-status-{track['id']}",
                    )
                    if track["midi_path"]:
                        gr.File(
                            value=track["midi_path"],
                            label=st("space.status.midi_file"),
                            key=f"midi-file-{track['id']}",
                        )

                    remove_track.click(
                        fn=_remove_track,
                        inputs=[track_state, track_id_state],
                        outputs=[track_state],
                        api_name=False,
                        queue=False,
                    )
                    midi_enabled.change(
                        fn=_track_control_updates,
                        inputs=[midi_enabled, midi_route],
                        outputs=[
                            midi_route,
                            start_midi,
                            midi_status,
                            midi_instruments,
                        ],
                        api_name=False,
                        queue=False,
                    )
                    midi_route.change(
                        fn=_track_control_updates,
                        inputs=[midi_enabled, midi_route],
                        outputs=[
                            midi_route,
                            start_midi,
                            midi_status,
                            midi_instruments,
                        ],
                        api_name=False,
                        queue=False,
                    )
                    start_midi.click(
                        fn=_convert_one_track,
                        inputs=[
                            track_state,
                            track_id_state,
                            midi_enabled,
                            midi_route,
                            midi_instruments,
                        ],
                        outputs=[track_state],
                        api_name=False,
                        concurrency_limit=1,
                        concurrency_id=GPU_CONCURRENCY_ID,
                    )

            active_midi_result = state.get("active_midi_result")
            if active_midi_result:
                with gr.Group(elem_classes=["track-card", "linked-midi-detail"]):
                    gr.HTML(
                        build_muscriptor_result_html(
                            active_midi_result,
                            st,
                            SPACE_LANGUAGE,
                        ),
                        key="linked-midi-result-workbench",
                    )
                    close_detail = gr.Button(
                        st("muscriptor_result.close_detail"),
                        size="sm",
                        key="close-linked-midi-detail",
                    )
                    close_detail.click(
                        fn=_close_active_midi_detail,
                        inputs=[track_state],
                        outputs=[track_state],
                        api_name=False,
                        queue=False,
                    )

    convert_btn.click(
        fn=convert_audio_to_midi,
        inputs=[
            audio_input,
            mode_radio,
            transcription_backend,
            yourmt3_model,
            muscriptor_instruments,
        ],
        outputs=[file_output, status_output, track_state],
        api_name="convert",
        concurrency_limit=1,
        concurrency_id=GPU_CONCURRENCY_ID,
    )

    stop_btn.click(
        fn=request_stop_current_job,
        inputs=None,
        outputs=[status_output],
        api_name=False,
        queue=False,
    )

    _log_poll_btn = gr.Button(visible=False)
    _log_poll_btn.click(
        fn=read_logs,
        inputs=[],
        outputs=[log_output],
        api_name="read_logs",
        queue=False,
    )

    gr.Markdown(
        '<div class="footer-info">'
        f"{st('space.ui.footer_powered_by')} "
        "<a href='https://github.com/mimbres/YourMT3'>YourMT3+</a> | "
        "<a href='https://github.com/mason369/music-to-midi'>GitHub</a> | "
        "MIT License"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", allowed_paths=[str(SPACE_OUTPUT_INSTANCE)])
