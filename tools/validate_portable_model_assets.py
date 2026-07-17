"""Strict identity preflight for model assets embedded in portable builds.

This command validates only files that are already present.  It never downloads,
repairs, replaces, or falls back to another model.  ``build_portable.ps1`` runs it
once on the selected source directories and again on the staged bundle.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from importlib import metadata
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import download_vocal_harmony_model as polarformer  # noqa: E402
import download_vocal_model as leap  # noqa: E402
import src.core.aria_amt_transcriber as aria_amt  # noqa: E402
import src.core.bytedance_piano_transcriber as bytedance_piano  # noqa: E402
import src.core.miros_transcriber as miros  # noqa: E402
import src.core.transkun_transcriber as transkun  # noqa: E402
import src.utils.yourmt3_downloader as yourmt3  # noqa: E402
import src.utils.yourmt3_source_identity as yourmt3_source  # noqa: E402
from src.utils.artifact_identity import validate_file_identity  # noqa: E402

AUDIO_SEPARATOR_PACKAGE_VERSION = "0.44.1"
ONNXRUNTIME_GPU_PACKAGE_VERSION = "1.23.2"


def _require_directory(path: Path | str, *, label: str) -> Path:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise RuntimeError(f"{label} directory is missing: {directory}")
    return directory


def _validate_leap_assets(model_dir: Path) -> tuple[Path, Path]:
    checkpoint = leap.resolve_vocal_model_path(model_dir)
    config = leap.resolve_vocal_config_path(model_dir)
    validate_file_identity(
        checkpoint,
        expected_size=leap.LEAP_CHECKPOINT_SIZE,
        expected_sha256=leap.LEAP_CHECKPOINT_SHA256,
        label="Leap XE checkpoint",
    )
    validate_file_identity(
        config,
        expected_size=leap.LEAP_CONFIG_SIZE,
        expected_sha256=leap.LEAP_CONFIG_SHA256,
        label="Leap XE config",
    )
    return checkpoint, config


def _validate_polarformer_assets(model_dir: Path) -> tuple[Path, Path]:
    checkpoint = polarformer.resolve_accompaniment_model_path(model_dir)
    config = polarformer.resolve_accompaniment_config_path(model_dir)
    validate_file_identity(
        checkpoint,
        expected_size=polarformer.POLARFORMER_ONNX_SIZE,
        expected_sha256=polarformer.POLARFORMER_ONNX_SHA256,
        label="PolarFormer ONNX checkpoint",
    )
    validate_file_identity(
        config,
        expected_size=polarformer.POLARFORMER_CONFIG_SIZE,
        expected_sha256=polarformer.POLARFORMER_CONFIG_SHA256,
        label="PolarFormer config",
    )
    return checkpoint, config


def _validate_yourmt3_assets(model_dir: Path) -> tuple[Path, ...]:
    model_keys = tuple(yourmt3.OFFICIAL_YOURMT3_MODEL_KEYS)
    if not model_keys or len(model_keys) != len(set(model_keys)):
        raise RuntimeError(
            "OFFICIAL_YOURMT3_MODEL_KEYS must contain a non-empty set of unique model keys"
        )

    validated: list[Path] = []
    for model_key in model_keys:
        identity = yourmt3.YOURMT3_MODEL_IDENTITIES.get(model_key)
        if not isinstance(identity, dict):
            raise RuntimeError(f"YourMT3 model {model_key!r} has no pinned artifact identity")

        try:
            relative_path = Path(str(identity["filename"]))
            expected_size = int(identity["size"])
            expected_sha256 = str(identity["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"YourMT3 model {model_key!r} has an invalid pinned artifact identity"
            ) from exc

        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(
                f"YourMT3 model {model_key!r} has an unsafe artifact path: {relative_path}"
            )

        if relative_path.name not in {"model.ckpt", "last.ckpt"}:
            raise RuntimeError(
                f"YourMT3 model {model_key!r} has an unsupported checkpoint name: "
                f"{relative_path.name}"
            )

        checkpoint_dir = model_dir / relative_path.parent
        candidate_names = tuple(dict.fromkeys((relative_path.name, "model.ckpt", "last.ckpt")))
        candidates = tuple(checkpoint_dir / name for name in candidate_names)
        present = tuple(path for path in candidates if path.is_file())
        if not present:
            expected_paths = ", ".join(str(path) for path in candidates)
            raise RuntimeError(
                f"YourMT3 {model_key} checkpoint is missing; expected one of: " f"{expected_paths}"
            )

        # YourMT3 runtime intentionally recognizes both model.ckpt and last.ckpt.
        # Validate every present alias so an invalid duplicate can never be hidden by
        # a valid sibling selected first by filesystem traversal.
        for checkpoint in present:
            validate_file_identity(
                checkpoint,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
                label=f"YourMT3 {model_key} checkpoint",
            )
        validated.append(present[0])

    return tuple(validated)


def _validate_yourmt3_source(source_dir: Path) -> tuple[Path]:
    yourmt3_source.validate_patched_yourmt3_source(source_dir)
    return (source_dir,)


def _validate_aria_amt_assets(model_dir: Path) -> tuple[Path]:
    checkpoint = model_dir / aria_amt.ARIA_AMT_CHECKPOINT_NAME
    aria_amt.validate_aria_amt_checkpoint(checkpoint)
    return (checkpoint,)


def _validate_bytedance_piano_assets(model_dir: Path) -> tuple[Path]:
    checkpoint = model_dir / bytedance_piano.BYTEDANCE_PIANO_CHECKPOINT_NAME
    bytedance_piano.validate_bytedance_piano_checkpoint(checkpoint)
    return (checkpoint,)


def _validate_miros_assets(repo_dir: Path) -> tuple[Path, ...]:
    source_error = miros.get_miros_source_identity_error(repo_dir)
    if source_error:
        raise RuntimeError(source_error)

    weight_error = miros.get_miros_weight_identity_error(repo_dir)
    if weight_error:
        raise RuntimeError(weight_error)

    return (
        repo_dir / "main.py",
        repo_dir / "transcribe.py",
        repo_dir / miros.MirosTranscriber.PRETRAINED_REL_PATH,
        repo_dir / miros.MirosTranscriber.CHECKPOINT_REL_PATH,
    )


def _require_distribution_version(package_name: str, expected_version: str) -> str:
    try:
        installed_version = metadata.version(package_name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Required portable runtime distribution is missing: {package_name}"
        ) from exc
    if installed_version != expected_version:
        raise RuntimeError(
            f"Portable runtime version mismatch for {package_name}: "
            f"expected {expected_version}, got {installed_version}"
        )
    return installed_version


def _require_runtime_available(label: str, unavailable_reason: str) -> None:
    if unavailable_reason:
        raise RuntimeError(f"{label} runtime identity validation failed: {unavailable_reason}")


def validate_portable_runtime_identities() -> dict[str, str]:
    """Require exact package/source identities for every bundled runtime backend."""

    audio_separator_version = _require_distribution_version(
        "audio-separator", AUDIO_SEPARATOR_PACKAGE_VERSION
    )
    onnxruntime_version = _require_distribution_version(
        "onnxruntime-gpu", ONNXRUNTIME_GPU_PACKAGE_VERSION
    )

    for module_name in ("audio_separator.separator", "onnxruntime"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(f"Portable runtime import failed for {module_name}: {exc}") from exc

    _require_runtime_available("Aria-AMT", aria_amt.get_aria_amt_runtime_unavailable_reason())
    _require_runtime_available(
        "ByteDance Piano",
        bytedance_piano.ByteDancePianoTranscriber.get_unavailable_reason(),
    )
    _require_runtime_available("TransKun", transkun.TranskunTranscriber.get_unavailable_reason())
    if not transkun.TranskunTranscriber().is_model_available():
        raise RuntimeError(
            "TransKun packaged V2 resources failed exact size/SHA-256 identity validation"
        )

    return {
        "audio-separator": audio_separator_version,
        "onnxruntime-gpu": onnxruntime_version,
        "aria-amt-source": aria_amt.ARIA_AMT_SOURCE_REVISION,
        "piano-transcription-inference": (bytedance_piano.BYTEDANCE_PIANO_PACKAGE_VERSION),
        "transkun": transkun.TRANSKUN_PACKAGE_VERSION,
    }


def validate_portable_model_assets(
    *,
    audio_separator_dir: Path | str,
    yourmt3_dir: Path | str,
    yourmt3_source_dir: Path | str,
    aria_amt_dir: Path | str,
    bytedance_piano_dir: Path | str,
    miros_dir: Path | str,
) -> dict[str, tuple[Path, ...]]:
    """Require every pinned portable model identity represented by these roots."""

    audio_separator_root = _require_directory(audio_separator_dir, label="audio-separator")
    yourmt3_root = _require_directory(yourmt3_dir, label="YourMT3")
    yourmt3_source_root = _require_directory(yourmt3_source_dir, label="patched YourMT3 source")
    aria_amt_root = _require_directory(aria_amt_dir, label="Aria-AMT")
    bytedance_root = _require_directory(bytedance_piano_dir, label="ByteDance Piano")
    miros_root = _require_directory(miros_dir, label="MIROS")

    return {
        "leap_xe": _validate_leap_assets(audio_separator_root),
        "polarformer": _validate_polarformer_assets(audio_separator_root),
        "yourmt3": _validate_yourmt3_assets(yourmt3_root),
        "yourmt3_source": _validate_yourmt3_source(yourmt3_source_root),
        "aria_amt": _validate_aria_amt_assets(aria_amt_root),
        "bytedance_piano": _validate_bytedance_piano_assets(bytedance_root),
        "miros": _validate_miros_assets(miros_root),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate pinned model identities for a portable build"
    )
    parser.add_argument("--audio-separator-dir", type=Path, required=True)
    parser.add_argument("--yourmt3-dir", type=Path, required=True)
    parser.add_argument("--yourmt3-source-dir", type=Path, required=True)
    parser.add_argument("--aria-amt-dir", type=Path, required=True)
    parser.add_argument("--bytedance-piano-dir", type=Path, required=True)
    parser.add_argument("--miros-dir", type=Path, required=True)
    parser.add_argument("--label", default="portable model assets")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        validated = validate_portable_model_assets(
            audio_separator_dir=args.audio_separator_dir,
            yourmt3_dir=args.yourmt3_dir,
            yourmt3_source_dir=args.yourmt3_source_dir,
            aria_amt_dir=args.aria_amt_dir,
            bytedance_piano_dir=args.bytedance_piano_dir,
            miros_dir=args.miros_dir,
        )
        runtime_identities = validate_portable_runtime_identities()
    except Exception as exc:
        print(f"[error] {args.label} identity validation failed: {exc}", file=sys.stderr)
        return 1

    for group, paths in validated.items():
        print(f"[ok] {group}: {len(paths)} pinned artifact(s)")
    for package_name, identity in runtime_identities.items():
        print(f"[ok] runtime {package_name}: {identity}")
    print(f"[ok] {args.label} passed strict pinned identity validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
