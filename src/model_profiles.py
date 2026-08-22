"""Explicit model-profile preparation and readiness for server deployments.

The container image intentionally contains code and runtime dependencies only.
Model assets are prepared into a persistent volume by an explicit operator
command; inference requests never download missing assets on behalf of users.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from src.model_profile_runtime_probe import MODEL_PROFILE_RUNTIME_PROBE_SWITCH

ENABLED_PROFILES_ENV = "MUSIC_TO_MIDI_ENABLED_PROFILES"
REQUIRE_ENABLED_PROFILES_ENV = "MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES"
MIROS_DIR_ENV = "MUSIC_TO_MIDI_MIROS_DIR"

YOURMT3_PROFILE_IDS = (
    "yourmt3:ymt3_plus",
    "yourmt3:yptf_single_nops",
    "yourmt3:yptf_multi_ps",
    "yourmt3:yptf_moe_multi_nops",
    "yourmt3:yptf_moe_multi_ps",
)
MUSCRIPTOR_PROFILE_IDS = ("muscriptor", "muscriptor:medium", "muscriptor:small")
PIANO_PROFILE_IDS = (
    "piano_transkun",
    "piano_transkun_v2_aug",
    "piano_aria_amt",
    "piano_bytedance_pedal",
)
SEPARATION_PROFILE_IDS = ("vocal_split", "six_stem_split")
DIRECT_MIDI_PROFILE_IDS = (
    *YOURMT3_PROFILE_IDS,
    "miros",
    *MUSCRIPTOR_PROFILE_IDS,
    *PIANO_PROFILE_IDS,
)
ALL_PROFILE_IDS = (*DIRECT_MIDI_PROFILE_IDS, *SEPARATION_PROFILE_IDS)
_PROFILE_ALIASES = {"muscriptor:large": "muscriptor"}


@dataclass(frozen=True)
class ModelProfileStatus:
    id: str
    enabled: bool
    available: bool
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class ModelProfileSnapshot:
    selection_mode: str
    explicit_selection_required: bool
    enabled_profiles: tuple[str, ...]
    profiles: tuple[ModelProfileStatus, ...]
    ready: bool
    readiness_error: str | None = None

    def status(self, profile_id: str) -> ModelProfileStatus:
        canonical = normalize_profile_id(profile_id)
        return next(item for item in self.profiles if item.id == canonical)

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_mode": self.selection_mode,
            "explicit_selection_required": self.explicit_selection_required,
            "enabled_profiles": list(self.enabled_profiles),
            "profiles": [asdict(item) for item in self.profiles],
            "ready": self.ready,
            "readiness_error": self.readiness_error,
        }


def normalize_profile_id(profile_id: str) -> str:
    normalized = str(profile_id).strip().lower()
    normalized = _PROFILE_ALIASES.get(normalized, normalized)
    if normalized not in ALL_PROFILE_IDS:
        raise ValueError(
            f"unsupported model profile {profile_id!r}; expected one of {list(ALL_PROFILE_IDS)}"
        )
    return normalized


def parse_profile_ids(raw: str) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("model profile selection must contain at least one profile")
    return tuple(dict.fromkeys(normalize_profile_id(item) for item in values))


def _parse_required_flag(raw: str | None) -> bool:
    if raw is None or not raw.strip():
        return False
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{REQUIRE_ENABLED_PROFILES_ENV} must be one of 1/0, true/false, yes/no or on/off"
    )


def _selection_from_environment() -> tuple[str, bool, tuple[str, ...], str | None]:
    try:
        required = _parse_required_flag(os.environ.get(REQUIRE_ENABLED_PROFILES_ENV))
    except ValueError as exc:
        return "invalid", False, (), str(exc)

    if ENABLED_PROFILES_ENV in os.environ:
        try:
            selected = parse_profile_ids(os.environ[ENABLED_PROFILES_ENV])
        except ValueError as exc:
            return "invalid", required, (), f"{ENABLED_PROFILES_ENV}: {exc}"
        return "explicit", required, selected, None

    if required:
        return (
            "invalid",
            required,
            (),
            f"{ENABLED_PROFILES_ENV} is required and must list at least one model profile",
        )
    return "auto_discovery", required, ALL_PROFILE_IDS, None


def _beat_this_unavailable_reason() -> str | None:
    from src.core.beat_this_tracker import validate_beat_this_checkpoint

    try:
        validate_beat_this_checkpoint()
    except (OSError, RuntimeError):
        return "Beat This final0 checkpoint is missing or failed its identity check"
    else:
        return None


def _midi_playback_unavailable_reason() -> str | None:
    from src.utils.fluidsynth_runtime import get_fluidsynth_executable
    from src.utils.muscriptor_soundfont_downloader import validate_muscriptor_soundfont

    errors: list[str] = []
    try:
        get_fluidsynth_executable()
    except (OSError, RuntimeError) as exc:
        errors.append(f"FluidSynth runtime failed validation: {exc}")
    try:
        validate_muscriptor_soundfont()
    except (OSError, RuntimeError) as exc:
        errors.append(f"MuseScore General SoundFont failed validation: {exc}")
    return "; ".join(errors) or None


@lru_cache(maxsize=2)
def _audio_separator_runtime_unavailable_reason(profile_id: str) -> str | None:
    if profile_id not in SEPARATION_PROFILE_IDS:
        raise ValueError(f"unsupported audio-separator profile: {profile_id}")

    if getattr(sys, "frozen", False):
        command = [sys.executable, MODEL_PROFILE_RUNTIME_PROBE_SWITCH, profile_id]
    else:
        probe = (
            "import sys\n"
            "from src.model_profile_runtime_probe import run_model_profile_runtime_probe\n"
            "raise SystemExit(run_model_profile_runtime_probe(sys.argv[1]))\n"
        )
        command = [sys.executable, "-c", probe, profile_id]

    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"{profile_id} native runtime probe exceeded 45 seconds"
    if completed.returncode != 0:
        output = (completed.stdout or "").strip()
        if len(output) > 4000:
            output = output[-4000:]
        return (
            f"{profile_id} native runtime probe failed with exit code "
            f"{completed.returncode}: {output or 'no diagnostic output'}"
        )
    return None


def _profile_unavailable_reason(
    profile_id: str,
    beat_reason: str | None,
    playback_reason: str | None = None,
) -> str | None:
    if profile_id in DIRECT_MIDI_PROFILE_IDS and beat_reason:
        return beat_reason
    if profile_id in DIRECT_MIDI_PROFILE_IDS and playback_reason:
        return playback_reason

    if profile_id.startswith("yourmt3:"):
        from src.core.yourmt3_transcriber import YourMT3Transcriber

        model_id = profile_id.split(":", 1)[1]
        if YourMT3Transcriber.is_available(model_id):
            return None
        return YourMT3Transcriber.get_unavailable_reason()

    if profile_id == "miros":
        from src.core.miros_transcriber import MirosTranscriber

        return MirosTranscriber.get_unavailable_reason() or None

    if profile_id in MUSCRIPTOR_PROFILE_IDS:
        from src.core.muscriptor_transcriber import MuscriptorTranscriber
        from src.models.data_models import Config

        model_size = "large" if profile_id == "muscriptor" else profile_id.split(":", 1)[1]
        config = Config()
        config.muscriptor_model = model_size
        return MuscriptorTranscriber(config).get_unavailable_reason() or None

    if profile_id == "piano_transkun":
        from src.core.transkun_transcriber import TranskunTranscriber

        runtime_reason = TranskunTranscriber.get_unavailable_reason()
        if runtime_reason:
            return runtime_reason
        if not TranskunTranscriber().is_model_available():
            return "TransKun 2.0.1 packaged V2 resources failed their identity check"
        return None

    if profile_id == "piano_transkun_v2_aug":
        from src.core.transkun_transcriber import TranskunTranscriber
        from src.core.transkun_v2_aug_transcriber import TranskunV2AugTranscriber

        runtime_reason = TranskunTranscriber.get_unavailable_reason()
        if runtime_reason:
            return runtime_reason
        transcriber = TranskunV2AugTranscriber()
        return transcriber.get_model_validation_error() or None

    if profile_id == "piano_aria_amt":
        from src.core.aria_amt_transcriber import AriaAmtTranscriber

        runtime_reason = AriaAmtTranscriber.get_unavailable_reason()
        if runtime_reason:
            return runtime_reason
        if not AriaAmtTranscriber().is_model_available():
            return "Aria-AMT checkpoint is missing or failed its identity check"
        return None

    if profile_id == "piano_bytedance_pedal":
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        runtime_reason = ByteDancePianoTranscriber.get_unavailable_reason()
        if runtime_reason:
            return runtime_reason
        if not ByteDancePianoTranscriber().is_model_available():
            return "ByteDance Piano checkpoint is missing or failed its identity check"
        return None

    if profile_id == "vocal_split":
        from src.core.vocal_separator import VocalSeparator

        if not VocalSeparator.is_model_available():
            return "Leap XE or PolarFormer assets are missing or failed identity checks"
        return _audio_separator_runtime_unavailable_reason(profile_id)

    if profile_id == "six_stem_split":
        from src.core.multi_stem_separator import SixStemSeparator

        separator = SixStemSeparator()
        if not separator.is_model_available():
            return "BS-RoFormer SW assets are missing or failed identity checks"
        return _audio_separator_runtime_unavailable_reason(profile_id)

    raise AssertionError(f"unhandled model profile: {profile_id}")


@lru_cache(maxsize=16)
def _inspect_model_profiles_cached(
    enabled_raw: str | None,
    enabled_present: bool,
    required_raw: str | None,
    home: str,
    hf_home: str,
    miros_dir: str,
) -> ModelProfileSnapshot:
    del enabled_raw, enabled_present, required_raw, home, hf_home, miros_dir
    selection_mode, required, selected, configuration_error = _selection_from_environment()
    if configuration_error:
        statuses = tuple(
            ModelProfileStatus(
                id=profile_id,
                enabled=False,
                available=False,
                unavailable_reason=configuration_error,
            )
            for profile_id in ALL_PROFILE_IDS
        )
        return ModelProfileSnapshot(
            selection_mode=selection_mode,
            explicit_selection_required=required,
            enabled_profiles=(),
            profiles=statuses,
            ready=False,
            readiness_error=configuration_error,
        )

    beat_reason = (
        _beat_this_unavailable_reason()
        if any(profile_id in DIRECT_MIDI_PROFILE_IDS for profile_id in selected)
        else None
    )
    playback_reason = (
        _midi_playback_unavailable_reason()
        if any(profile_id in DIRECT_MIDI_PROFILE_IDS for profile_id in selected)
        else None
    )
    statuses: list[ModelProfileStatus] = []
    for profile_id in ALL_PROFILE_IDS:
        if profile_id not in selected:
            statuses.append(
                ModelProfileStatus(
                    id=profile_id,
                    enabled=False,
                    available=False,
                    unavailable_reason="profile is not enabled by the server operator",
                )
            )
            continue
        try:
            reason = _profile_unavailable_reason(profile_id, beat_reason, playback_reason)
        except Exception as exc:
            reason = f"model-profile readiness check failed: {exc}"
        statuses.append(
            ModelProfileStatus(
                id=profile_id,
                enabled=True,
                available=reason is None,
                unavailable_reason=reason,
            )
        )

    enabled_statuses = [item for item in statuses if item.enabled]
    if selection_mode == "explicit":
        ready = bool(enabled_statuses) and all(item.available for item in enabled_statuses)
    else:
        ready = any(item.available for item in enabled_statuses)
    unavailable = [item.id for item in enabled_statuses if not item.available]
    if ready:
        readiness_error = None
    elif selection_mode == "explicit" and unavailable:
        readiness_error = "configured model profiles are unavailable: " + ", ".join(unavailable)
    else:
        readiness_error = "no usable model profile is installed"
    return ModelProfileSnapshot(
        selection_mode=selection_mode,
        explicit_selection_required=required,
        enabled_profiles=selected,
        profiles=tuple(statuses),
        ready=ready,
        readiness_error=readiness_error,
    )


def inspect_model_profiles(*, refresh: bool = False) -> ModelProfileSnapshot:
    if refresh:
        _inspect_model_profiles_cached.cache_clear()
        _audio_separator_runtime_unavailable_reason.cache_clear()
    return _inspect_model_profiles_cached(
        os.environ.get(ENABLED_PROFILES_ENV),
        ENABLED_PROFILES_ENV in os.environ,
        os.environ.get(REQUIRE_ENABLED_PROFILES_ENV),
        os.environ.get("HOME", ""),
        os.environ.get("HF_HOME", ""),
        os.environ.get(MIROS_DIR_ENV, ""),
    )


def primary_profile_id(
    *,
    processing_mode: str,
    transcription_backend: str,
    yourmt3_model: str,
    muscriptor_model: str,
) -> str:
    mode = str(processing_mode).strip().lower()
    if mode == "smart":
        backend = str(transcription_backend).strip().lower()
        if backend == "yourmt3":
            return normalize_profile_id(f"yourmt3:{yourmt3_model}")
        if backend == "miros":
            return "miros"
        if backend == "muscriptor":
            size = str(muscriptor_model).strip().lower()
            return normalize_profile_id("muscriptor" if size == "large" else f"muscriptor:{size}")
        raise ValueError(f"unsupported transcription backend: {transcription_backend!r}")
    return normalize_profile_id(mode)


def require_profile_available(
    profile_id: str, snapshot: ModelProfileSnapshot | None = None
) -> ModelProfileStatus:
    status = (snapshot or inspect_model_profiles()).status(profile_id)
    if not status.available:
        raise RuntimeError(
            f"model profile {status.id!r} is unavailable: "
            f"{status.unavailable_reason or 'unknown readiness failure'}"
        )
    return status


def _prepare_one(profile_id: str, printer: Callable[[str], None]) -> None:
    if profile_id.startswith("yourmt3:"):
        from src.utils.yourmt3_downloader import download_model, get_model_path

        model_id = profile_id.split(":", 1)[1]
        download_model(model_id)
        path = get_model_path(model_id)
        if path is None:
            raise RuntimeError(f"YourMT3+ preparation produced no verified checkpoint: {model_id}")
        printer(f"ready {profile_id}: {path}")
        return
    if profile_id == "miros":
        from download_miros_model import prepare_miros_model

        configured = os.environ.get(MIROS_DIR_ENV, "").strip()
        target = (
            Path(configured).expanduser()
            if configured
            else Path(__file__).resolve().parents[1] / "external" / "ai4m-miros"
        )
        printer(f"preparing {profile_id} in {target.resolve()}")
        prepare_miros_model(target, printer=printer)
        return
    if profile_id in MUSCRIPTOR_PROFILE_IDS:
        from src.utils.muscriptor_downloader import download_muscriptor_model

        size = "large" if profile_id == "muscriptor" else profile_id.split(":", 1)[1]
        weights, config = download_muscriptor_model(size, printer=printer)
        printer(f"ready {profile_id}: weights={weights}, config={config}")
        return
    if profile_id == "piano_transkun":
        from download_sota_models import validate_default_transkun_runtime

        printer(f"ready {profile_id}: {validate_default_transkun_runtime()}")
        return
    if profile_id == "piano_transkun_v2_aug":
        from download_transkun_v2_aug_model import download_transkun_v2_aug_model

        printer(f"ready {profile_id}: {download_transkun_v2_aug_model(printer=printer)}")
        return
    if profile_id == "piano_aria_amt":
        from download_aria_amt_model import download_aria_model

        printer(f"ready {profile_id}: {download_aria_model(printer=printer)}")
        return
    if profile_id == "piano_bytedance_pedal":
        from download_bytedance_piano_model import download_bytedance_piano_model

        printer(f"ready {profile_id}: {download_bytedance_piano_model(printer=printer)}")
        return
    if profile_id == "vocal_split":
        from download_accompaniment_model import download_accompaniment_model
        from download_vocal_model import download_vocal_model

        vocal = download_vocal_model(printer=printer)
        accompaniment = download_accompaniment_model(printer=printer)
        printer(f"ready {profile_id}: vocal={vocal}, accompaniment={accompaniment}")
        return
    if profile_id == "six_stem_split":
        from download_multistem_model import download_multistem_model

        weights, config = download_multistem_model(printer=printer)
        printer(f"ready {profile_id}: weights={weights}, config={config}")
        return
    raise AssertionError(f"unhandled model profile: {profile_id}")


def prepare_profiles(
    profile_ids: Iterable[str], *, printer: Callable[[str], None] = print
) -> ModelProfileSnapshot:
    selected = tuple(dict.fromkeys(normalize_profile_id(item) for item in profile_ids))
    if not selected:
        raise ValueError("at least one model profile must be selected")

    if any(profile_id in DIRECT_MIDI_PROFILE_IDS for profile_id in selected):
        from download_beat_this_model import download_beat_this_model
        from src.utils.fluidsynth_runtime import get_fluidsynth_executable
        from src.utils.muscriptor_soundfont_downloader import download_muscriptor_soundfont

        printer(f"ready beat_this: {download_beat_this_model(printer=printer)}")
        printer("ready musescore_soundfont: " f"{download_muscriptor_soundfont(printer=printer)}")
        printer(f"ready fluidsynth: {get_fluidsynth_executable()}")
    for profile_id in ALL_PROFILE_IDS:
        if profile_id in selected:
            _prepare_one(profile_id, printer)

    os.environ[ENABLED_PROFILES_ENV] = ",".join(selected)
    snapshot = inspect_model_profiles(refresh=True)
    unavailable = [item for item in snapshot.profiles if item.enabled and not item.available]
    if unavailable:
        details = "; ".join(
            f"{item.id}: {item.unavailable_reason or 'unknown failure'}" for item in unavailable
        )
        raise RuntimeError(f"model preparation finished but readiness validation failed: {details}")
    return snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or verify explicit Music to MIDI model profiles"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="download and strictly verify selected profiles"
    )
    prepare.add_argument(
        "--profiles",
        required=True,
        help=f"comma-separated profile IDs; choices: {','.join(ALL_PROFILE_IDS)}",
    )
    verify = subparsers.add_parser("verify", help="inspect the configured model volume")
    verify.add_argument("--profiles", default=None, help="override configured profile IDs")
    verify.add_argument("--require-ready", action="store_true")
    subparsers.add_parser("list", help="print supported profile IDs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        print("\n".join(ALL_PROFILE_IDS))
        return 0
    if args.command == "prepare":
        selected = parse_profile_ids(args.profiles)
        snapshot = prepare_profiles(selected)
    else:
        if args.profiles is not None:
            selected = parse_profile_ids(args.profiles)
            os.environ[ENABLED_PROFILES_ENV] = ",".join(selected)
        snapshot = inspect_model_profiles(refresh=True)
    print(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2))
    if getattr(args, "require_ready", False) and not snapshot.ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL_PROFILE_IDS",
    "DIRECT_MIDI_PROFILE_IDS",
    "ENABLED_PROFILES_ENV",
    "MIROS_DIR_ENV",
    "ModelProfileSnapshot",
    "ModelProfileStatus",
    "REQUIRE_ENABLED_PROFILES_ENV",
    "inspect_model_profiles",
    "normalize_profile_id",
    "parse_profile_ids",
    "prepare_profiles",
    "primary_profile_id",
    "require_profile_available",
]
