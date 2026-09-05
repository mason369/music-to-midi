"""First-party local CLI and sequential batch runner.

The CLI deliberately reuses :class:`src.web_api.engine.InferenceEngine` so the
desktop, HTTP service, packaged executables, and command line execute the same
seven product modes.  Batch items always receive separate output directories;
this prevents same-stem input files from overwriting normalized WAV or MIDI
artifacts produced by the core pipeline.
"""

from __future__ import annotations

import argparse
import errno
from functools import partial
import hashlib
import json
import os
import re
import signal
import stat
import sys
import threading
import traceback
import uuid
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence, TextIO

from pydantic import ValidationError

from src import __version__
from src.core.manual_midi import MANUAL_MIDI_ROUTES, is_muscriptor_midi_route
from src.models.data_models import (
    MidiTrackMode,
    MultiInstrumentModel,
    MuscriptorModel,
    MuscriptorProcessingChain,
    ProcessingMode,
    TempoMode,
    YourMT3Model,
)
from src.models.muscriptor_instruments import validate_muscriptor_instruments
from src.web_api.schemas import InferenceOptions, ManualMidiOptions

SUPPORTED_AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"})
QUANTIZE_GRIDS = ("1/4", "1/8", "1/16", "1/32", "1/64")
COMMANDS = frozenset({"convert", "batch", "track-to-midi", "routes"})
MANIFEST_NAME = "music-to-midi-job.json"
MANIFEST_SCHEMA = 1
RUN_SUMMARY_SCHEMA = 1

_MESSAGES = {
    "zh_CN": {
        "description": "Music to MIDI 命令行：转换单个音频、多个文件或目录。",
        "convert_help": "转换一个或多个音频文件；目录默认只扫描当前层。",
        "batch_help": "批量转换文件或目录，逐个处理音频。",
        "manual_help": "选择模型，将单条 WAV 或音频转换为 MIDI。",
        "routes_help": "列出 track-to-midi 支持的 13 条路线。",
        "inputs": "输入音频文件或目录，可传多个",
        "output": "输出根目录（默认：便携包/项目下 MidiOutput/CLI）",
        "mode": "处理模式",
        "backend": "SMART 模式的多乐器后端",
        "yourmt3_model": "YourMT3+ checkpoint",
        "muscriptor_model": "MuScriptor checkpoint 大小",
        "muscriptor_chain": "MuScriptor 分段衔接",
        "instrument": "MuScriptor 乐器约束；可重复或用逗号分隔",
        "track_mode": "MIDI 轨道布局",
        "tempo_mode": "速度模式",
        "bpm": "手动固定 BPM；使用后速度模式必须为 fixed_manual",
        "quantize": "量化音符，可省略网格（默认 1/32）",
        "gpu_device": "GPU 序号（默认 0）",
        "recursive": "递归扫描输入目录",
        "rerun": "忽略成功清单并创建新的编号输出目录",
        "fail_fast": "任一项目失败后立即停止；默认继续其余独立项目并最终返回失败",
        "dry_run": "只校验参数并列出计划，不加载模型、不创建输出",
        "json": "向 stdout 输出 JSON Lines；诊断信息保留在 stderr",
        "quiet": "隐藏普通进度，只显示错误和最终摘要",
        "verbose": "输出更详细的失败诊断",
        "language": "CLI 和推理语言",
        "route": "逐轨 MIDI 转写模型",
        "tempo_source": "原曲节拍参考；默认校验分离清单中的原曲，独立音频使用自身",
        "start": "[{index}/{total}] 开始：{source}",
        "success": "[{index}/{total}] 完成：{job_dir}",
        "skipped": "[{index}/{total}] 已验证成功清单，跳过：{job_dir}",
        "failed": "[{index}/{total}] 失败：{source}\n  根因：{error}",
        "cancelled": "[{index}/{total}] 已取消：{source}",
        "progress": "[{index}/{total}] {percent:3d}% {message}",
        "warning": "警告：{message}",
        "dry_item": "[{index}/{total}] 计划：{source}",
        "dry_summary": "计划汇总：共 {total} 个输入；未加载模型，也未创建输出。",
        "summary": (
            "汇总：总数 {total}，成功 {succeeded}，跳过 {skipped}，"
            "失败 {failed}，取消 {cancelled}，未执行 {not_run}"
        ),
        "summary_path": "运行清单：{path}",
        "no_auto_midi": "分离模式生成 WAV；对需要转 MIDI 的音轨运行 track-to-midi。",
        "invalid_resume": (
            "候选断点清单无法通过校验，已保留并忽略；" "仅在找不到其他有效清单时重新执行：{path}"
        ),
        "empty_midi": "模型生成的 MIDI 不含音符；产物已保留，请检查输入内容或所选路线。",
        "note_count_corrected_from_file": "模型报告的音符数与最终 MIDI 不一致，已按回读文件中的真实音符数记录。",
        "cancel_requested": "收到取消请求，正在通知当前模型停止……",
        "cancel_delivery_failed": "当前处理器拒绝取消请求：{error}",
        "fatal": "CLI 无法启动：{error}",
        "pipe_closed": "输出管道已关闭，CLI 已停止。已写入的转换结果和清单保留。",
    },
    "en_US": {
        "description": "Music to MIDI CLI for individual files, multiple files, and directories.",
        "convert_help": "Convert one or more audio files; directories are non-recursive by default.",
        "batch_help": "Batch-convert files or directories, processing one audio file at a time.",
        "manual_help": "Choose a model and convert a WAV or audio track to MIDI.",
        "routes_help": "List all 13 routes supported by track-to-midi.",
        "inputs": "One or more input audio files or directories",
        "output": "Output root (default: MidiOutput/CLI beside the bundle/project)",
        "mode": "Processing mode",
        "backend": "Multi-instrument backend for SMART mode",
        "yourmt3_model": "YourMT3+ checkpoint",
        "muscriptor_model": "MuScriptor checkpoint size",
        "muscriptor_chain": "MuScriptor segment handling",
        "instrument": "MuScriptor instrument constraint; repeat or comma-separate",
        "track_mode": "MIDI track layout",
        "tempo_mode": "Tempo mode",
        "bpm": "Manual fixed BPM; tempo mode must be fixed_manual",
        "quantize": "Explicitly quantize; omit GRID to use 1/32",
        "gpu_device": "GPU index (default: 0)",
        "recursive": "Recursively scan input directories",
        "rerun": "Ignore successful manifests and create a new numbered output directory",
        "fail_fast": "Stop after the first failed item; otherwise finish independent items and fail",
        "dry_run": "Validate and print the plan without loading models or creating output",
        "json": "Write JSON Lines to stdout; diagnostics remain on stderr",
        "quiet": "Hide normal progress; keep errors and the final summary",
        "verbose": "Include detailed failure diagnostics",
        "language": "CLI and inference language",
        "route": "Per-track MIDI transcription model",
        "tempo_source": "Beat reference; defaults to the verified separation original, or standalone input",
        "start": "[{index}/{total}] Start: {source}",
        "success": "[{index}/{total}] Complete: {job_dir}",
        "skipped": "[{index}/{total}] Verified successful manifest; skipped: {job_dir}",
        "failed": "[{index}/{total}] Failed: {source}\n  Root cause: {error}",
        "cancelled": "[{index}/{total}] Cancelled: {source}",
        "progress": "[{index}/{total}] {percent:3d}% {message}",
        "warning": "Warning: {message}",
        "dry_item": "[{index}/{total}] Plan: {source}",
        "dry_summary": "Plan summary: {total} input(s); no model loaded and no output created.",
        "summary": (
            "Summary: total {total}, succeeded {succeeded}, skipped {skipped}, "
            "failed {failed}, cancelled {cancelled}, not run {not_run}"
        ),
        "summary_path": "Run manifest: {path}",
        "no_auto_midi": (
            "Separation modes produce WAV files; run track-to-midi for the tracks you want to transcribe."
        ),
        "invalid_resume": (
            "A resume manifest could not be verified and was preserved but ignored; "
            "the item reruns only if no other valid manifest exists: {path}"
        ),
        "empty_midi": (
            "The model produced a MIDI file with no notes; the artifact was preserved so the "
            "input and selected route can be inspected."
        ),
        "note_count_corrected_from_file": (
            "The model-reported note count differed from the final MIDI; the manifest uses the "
            "verified file count."
        ),
        "cancel_requested": "Cancellation requested; asking the active model to stop...",
        "cancel_delivery_failed": "The active processor rejected cancellation: {error}",
        "fatal": "CLI could not start: {error}",
        "pipe_closed": "The output pipe closed; CLI stopped. Saved results and manifests are preserved.",
    },
}


def _normalize_language(value: str | None) -> str:
    normalized = str(value or "").strip().replace("-", "_").lower()
    return "en_US" if normalized.startswith("en") else "zh_CN"


def _message(language: str, key: str, **values: object) -> str:
    return _MESSAGES[_normalize_language(language)][key].format(**values)


def _quality_warning_message(language: str, warning: str) -> str:
    messages = _MESSAGES[_normalize_language(language)]
    return messages.get(warning, warning)


def _argument_language(arguments: Sequence[str]) -> str:
    for index, value in enumerate(arguments):
        if value == "--language" and index + 1 < len(arguments):
            return _normalize_language(arguments[index + 1])
        if value.startswith("--language="):
            return _normalize_language(value.split("=", 1)[1])
    return _normalize_language(os.environ.get("MUSIC_TO_MIDI_LANGUAGE", "zh_CN"))


def _enum_values(enum_type: type[Enum]) -> tuple[str, ...]:
    return tuple(item.value for item in enum_type)


def _cli_choices(values: Iterable[str]) -> tuple[str, ...]:
    choices: list[str] = []
    for value in values:
        choices.append(value)
        dashed = value.replace("_", "-")
        if dashed != value:
            choices.append(dashed)
    return tuple(choices)


def _canonical(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _default_output_root() -> Path:
    from src.utils.runtime_paths import get_executable_dir

    return Path(get_executable_dir()) / "MidiOutput" / "CLI"


def _add_output_options(parser: argparse.ArgumentParser, language: str) -> None:
    parser.add_argument("inputs", nargs="+", help=_message(language, "inputs"))
    parser.add_argument("-o", "--output", type=Path, help=_message(language, "output"))
    parser.add_argument("--recursive", action="store_true", help=_message(language, "recursive"))
    parser.add_argument("--rerun", action="store_true", help=_message(language, "rerun"))
    parser.add_argument("--fail-fast", action="store_true", help=_message(language, "fail_fast"))
    parser.add_argument("--dry-run", action="store_true", help=_message(language, "dry_run"))
    parser.add_argument("--json", action="store_true", help=_message(language, "json"))
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--quiet", action="store_true", help=_message(language, "quiet"))
    display.add_argument("--verbose", action="store_true", help=_message(language, "verbose"))
    parser.add_argument(
        "--language",
        choices=("zh_CN", "en_US", "zh-CN", "en-US"),
        default=language,
        help=_message(language, "language"),
    )


def _add_tempo_options(parser: argparse.ArgumentParser, language: str) -> None:
    parser.add_argument(
        "--tempo-mode",
        choices=_cli_choices(_enum_values(TempoMode)),
        help=_message(language, "tempo_mode"),
    )
    parser.add_argument("--bpm", type=float, help=_message(language, "bpm"))
    parser.add_argument(
        "--quantize",
        nargs="?",
        const="1/32",
        choices=QUANTIZE_GRIDS,
        metavar="GRID",
        help=_message(language, "quantize"),
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        metavar="INDEX",
        help=_message(language, "gpu_device"),
    )


def _add_muscriptor_options(parser: argparse.ArgumentParser, language: str) -> None:
    parser.add_argument(
        "--muscriptor-chain",
        choices=_enum_values(MuscriptorProcessingChain),
        default=None,
        help=_message(language, "muscriptor_chain"),
    )
    parser.add_argument(
        "--instrument",
        action="append",
        default=[],
        metavar="NAME[,NAME]",
        help=_message(language, "instrument"),
    )


class _CliHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, language, **kwargs):
        self.language = language
        super().__init__(*args, **kwargs)

    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = "用法：" if self.language == "zh_CN" else "usage: "
        super().add_usage(usage, actions, groups, prefix)


class _CliArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, language, **kwargs):
        kwargs["add_help"] = False
        kwargs["formatter_class"] = partial(_CliHelpFormatter, language=language)
        super().__init__(*args, **kwargs)
        self._positionals.title = "位置参数" if language == "zh_CN" else "positional arguments"
        self._optionals.title = "选项" if language == "zh_CN" else "options"
        self.add_argument(
            "-h", "--help", action="help",
            help="显示帮助并退出" if language == "zh_CN" else "Show help and exit",
        )


def build_parser(arguments: Sequence[str] | None = None) -> argparse.ArgumentParser:
    arguments = list(arguments or [])
    language = _argument_language(arguments)
    parser = _CliArgumentParser(
        language=language,
        prog=Path(sys.executable).name if getattr(sys, "frozen", False) else "python -m src.cli",
        description=_message(language, "description"),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
        help="显示版本并退出" if language == "zh_CN" else "Show version and exit",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True,
        parser_class=partial(_CliArgumentParser, language=language),
    )

    for command, help_key in (("convert", "convert_help"), ("batch", "batch_help")):
        convert = subparsers.add_parser(command, help=_message(language, help_key))
        _add_output_options(convert, language)
        convert.add_argument(
            "--mode",
            choices=_cli_choices(
                mode.value for mode in ProcessingMode if mode is not ProcessingMode.PIANO
            ),
            default=ProcessingMode.SMART.value,
            help=_message(language, "mode"),
        )
        convert.add_argument(
            "--backend",
            choices=_enum_values(MultiInstrumentModel),
            default=None,
            help=_message(language, "backend"),
        )
        convert.add_argument(
            "--yourmt3-model",
            choices=_cli_choices(
                model.value for model in YourMT3Model if model is not YourMT3Model.LEGACY_MC13
            ),
            default=None,
            help=_message(language, "yourmt3_model"),
        )
        convert.add_argument(
            "--muscriptor-model",
            choices=_enum_values(MuscriptorModel),
            default=None,
            help=_message(language, "muscriptor_model"),
        )
        _add_muscriptor_options(convert, language)
        convert.add_argument(
            "--track-mode",
            choices=_cli_choices(_enum_values(MidiTrackMode)),
            default=None,
            help=_message(language, "track_mode"),
        )
        _add_tempo_options(convert, language)
        convert.set_defaults(job_kind="primary")

    manual = subparsers.add_parser("track-to-midi", help=_message(language, "manual_help"))
    _add_output_options(manual, language)
    manual.add_argument(
        "--route",
        required=True,
        choices=MANUAL_MIDI_ROUTES,
        help=_message(language, "route"),
    )
    manual.add_argument("--tempo-source", type=Path, help=_message(language, "tempo_source"))
    _add_muscriptor_options(manual, language)
    _add_tempo_options(manual, language)
    manual.set_defaults(job_kind="manual_midi")

    routes = subparsers.add_parser("routes", help=_message(language, "routes_help"))
    routes.add_argument("--json", action="store_true", help=_message(language, "json"))
    routes.add_argument(
        "--language",
        choices=("zh_CN", "en_US", "zh-CN", "en-US"),
        default=language,
        help=_message(language, "language"),
    )
    return parser


def _normalize_argv(arguments: Sequence[str]) -> list[str]:
    normalized = list(arguments)
    if normalized and not normalized[0].startswith("-") and normalized[0] not in COMMANDS:
        normalized.insert(0, "convert")
    return normalized


def _parse_instruments(values: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        expanded.extend(part.strip() for part in str(value).split(",") if part.strip())
    return list(validate_muscriptor_instruments(expanded))


def _tempo_values(namespace: argparse.Namespace) -> tuple[str, float | None]:
    tempo_mode = _canonical(namespace.tempo_mode) if namespace.tempo_mode else None
    if namespace.bpm is not None:
        if tempo_mode not in {None, TempoMode.FIXED_MANUAL.value}:
            raise ValueError("--bpm can only be combined with --tempo-mode fixed_manual")
        return TempoMode.FIXED_MANUAL.value, float(namespace.bpm)
    return tempo_mode or TempoMode.FIXED_AUTO.value, None


def _validated_options(namespace: argparse.Namespace) -> dict:
    tempo_mode, custom_bpm = _tempo_values(namespace)
    instruments = _parse_instruments(namespace.instrument)
    muscriptor_chain = namespace.muscriptor_chain or MuscriptorProcessingChain.OFFICIAL.value
    common = {
        "muscriptor_instruments": instruments,
        "muscriptor_processing_chain": muscriptor_chain,
        "tempo_mode": tempo_mode,
        "custom_bpm": custom_bpm,
        "quantize_notes": namespace.quantize is not None,
        "quantize_grid": namespace.quantize or "1/32",
        "use_gpu": True,
        "gpu_device": namespace.gpu_device,
        "language": _normalize_language(namespace.language),
    }
    if namespace.job_kind == "manual_midi":
        if not is_muscriptor_midi_route(namespace.route) and (
            instruments or namespace.muscriptor_chain is not None
        ):
            raise ValueError(
                "--instrument and --muscriptor-chain are only valid with a MuScriptor route"
            )
        return dict(ManualMidiOptions(route=namespace.route, **common).model_dump(mode="json"))
    processing_mode = _canonical(namespace.mode)
    backend = namespace.backend or MultiInstrumentModel.YOURMT3.value
    yourmt3_model = _canonical(namespace.yourmt3_model or YourMT3Model.YPTF_MOE_MULTI_NOPS.value)
    muscriptor_model = namespace.muscriptor_model or MuscriptorModel.LARGE.value
    midi_track_mode = _canonical(namespace.track_mode or MidiTrackMode.MULTI_TRACK.value)

    backend_specific_flags: list[str] = []
    if namespace.backend is not None:
        backend_specific_flags.append("--backend")
    if namespace.yourmt3_model is not None:
        backend_specific_flags.append("--yourmt3-model")
    if namespace.muscriptor_model is not None:
        backend_specific_flags.append("--muscriptor-model")
    if namespace.muscriptor_chain is not None:
        backend_specific_flags.append("--muscriptor-chain")
    if instruments:
        backend_specific_flags.append("--instrument")

    split_modes = {
        ProcessingMode.VOCAL_SPLIT.value,
        ProcessingMode.SIX_STEM_SPLIT.value,
    }
    if processing_mode in split_modes:
        ignored_flags = list(backend_specific_flags)
        if namespace.track_mode is not None:
            ignored_flags.append("--track-mode")
        if namespace.tempo_mode is not None:
            ignored_flags.append("--tempo-mode")
        if namespace.bpm is not None:
            ignored_flags.append("--bpm")
        if namespace.quantize is not None:
            ignored_flags.append("--quantize")
        if ignored_flags:
            raise ValueError(
                f"{processing_mode} produces WAV tracks only; these options do not apply: "
                + ", ".join(ignored_flags)
            )
    elif processing_mode != ProcessingMode.SMART.value:
        ignored_flags = list(backend_specific_flags)
        if namespace.track_mode is not None:
            ignored_flags.append("--track-mode")
        if ignored_flags:
            raise ValueError(
                f"dedicated piano mode {processing_mode} does not use: " + ", ".join(ignored_flags)
            )
    else:
        if backend != MultiInstrumentModel.YOURMT3.value and namespace.yourmt3_model is not None:
            raise ValueError("--yourmt3-model is only valid with --backend yourmt3")
        if backend != MultiInstrumentModel.MUSCRIPTOR.value and (
            namespace.muscriptor_model is not None
            or namespace.muscriptor_chain is not None
            or instruments
        ):
            raise ValueError(
                "--muscriptor-model, --muscriptor-chain, and --instrument require "
                "--backend muscriptor"
            )
    return dict(
        InferenceOptions(
            processing_mode=processing_mode,
            transcription_backend=backend,
            yourmt3_model=yourmt3_model,
            muscriptor_model=muscriptor_model,
            midi_track_mode=midi_track_mode,
            **common,
        ).model_dump(mode="json")
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _files_in_directory(root: Path, recursive: bool, output_root: Path) -> list[Path]:
    files: list[Path] = []
    if not recursive:
        for candidate in root.iterdir():
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
                files.append(candidate.resolve())
        return sorted(files, key=lambda path: str(path).casefold())

    def scan_failed(error: OSError) -> None:
        # os.walk otherwise silently omits inaccessible subdirectories.
        raise error

    for current_root, directory_names, file_names in os.walk(
        root, followlinks=False, onerror=scan_failed
    ):
        current = Path(current_root).resolve()
        retained: list[str] = []
        for name in directory_names:
            candidate = (current / name).resolve()
            if candidate == output_root:
                continue
            retained.append(name)
        directory_names[:] = retained
        for name in file_names:
            candidate = current / name
            if candidate.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
                files.append(candidate.resolve())
    return sorted(files, key=lambda path: str(path).casefold())


def discover_inputs(values: Iterable[str], recursive: bool, output_root: Path) -> list[Path]:
    resolved_output = output_root.expanduser().resolve()
    discovered: list[Path] = []
    errors: list[str] = []
    for raw_value in values:
        candidate = Path(raw_value).expanduser().resolve()
        if not candidate.exists():
            errors.append(f"input does not exist: {candidate}")
            continue
        if candidate.is_file():
            if candidate.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
                errors.append(
                    f"unsupported audio file: {candidate}; expected one of "
                    f"{sorted(SUPPORTED_AUDIO_SUFFIXES)}"
                )
            else:
                discovered.append(candidate)
            continue
        if not candidate.is_dir():
            errors.append(f"input is neither a regular file nor directory: {candidate}")
            continue
        if candidate == resolved_output:
            errors.append(f"input directory cannot equal output root: {candidate}")
            continue
        directory_files = _files_in_directory(candidate, recursive, resolved_output)
        if not directory_files:
            errors.append(f"input directory contains no supported audio files: {candidate}")
        discovered.extend(directory_files)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in discovered:
        key = os.path.normcase(str(path))
        if key not in seen:
            unique.append(path)
            seen.add(key)
    for path in unique:
        try:
            if path.stat().st_size <= 0:
                errors.append(f"input audio file is empty: {path}")
        except OSError as exc:
            errors.append(f"input audio file cannot be read: {path}: {exc}")
    if errors:
        raise ValueError("\n".join(errors))
    if not unique:
        raise ValueError("no supported audio inputs were selected")
    return unique


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_identity(path: Path) -> dict:
    stat = path.stat()
    if stat.st_size <= 0:
        raise ValueError(f"input audio file is empty: {path}")
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _assert_source_unchanged(identity: dict) -> None:
    path = Path(identity["path"])
    stat = path.stat()
    if (
        int(stat.st_size) != identity["size"]
        or int(stat.st_mtime_ns) != identity["mtime_ns"]
        or _sha256_file(path) != identity["sha256"]
    ):
        raise RuntimeError(f"input audio changed while conversion was running: {path}")


def _fingerprint_payload(
    *,
    kind: str,
    source: dict,
    options: dict,
    tempo_source: dict | None,
) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "app_version": __version__,
        "kind": kind,
        "source": {
            "path": source["path"],
            "size": source["size"],
            "sha256": source["sha256"],
        },
        "tempo_source": (
            {
                "path": tempo_source["path"],
                "size": tempo_source["size"],
                "sha256": tempo_source["sha256"],
            }
            if tempo_source is not None
            else None
        ),
        "options": options,
    }


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _safe_job_stem(path: Path) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", path.stem).strip(" .")
    if not value:
        value = "audio"
    if value.upper() in _WINDOWS_RESERVED_NAMES:
        value = f"_{value}"
    return value[:100].rstrip(" .") or "audio"


def _job_candidates(output_root: Path, stem: str) -> list[Path]:
    expression = re.compile(rf"^{re.escape(stem)}(?:-(\d+))?$")
    candidates: list[tuple[int, Path]] = []
    if output_root.is_dir():
        for path in output_root.iterdir():
            if not path.is_dir():
                continue
            match = expression.fullmatch(path.name)
            if match:
                candidates.append((int(match.group(1) or 1), path))
    return [path for _, path in sorted(candidates, key=lambda item: item[0])]


def _validate_manifest_artifacts(manifest: dict, job_dir: Path) -> bool:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    resolved_job = job_dir.resolve()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            return False
        relative_path = artifact.get("path")
        expected_hash = artifact.get("sha256")
        expected_size = artifact.get("size")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            return False
        path = (resolved_job / relative_path).resolve()
        if not _is_relative_to(path, resolved_job) or not path.is_file():
            return False
        if path.stat().st_size != expected_size or _sha256_file(path) != expected_hash:
            return False
    return True


def _find_resumable_job(
    output_root: Path,
    stem: str,
    fingerprint: str,
    warning_callback: Callable[[Path], None],
) -> tuple[Path, dict] | None:
    for job_dir in _job_candidates(output_root, stem):
        manifest_path = job_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            warning_callback(manifest_path)
            continue
        if (
            not isinstance(manifest, dict)
            or type(manifest.get("schema")) is not int
            or manifest["schema"] != MANIFEST_SCHEMA
        ):
            warning_callback(manifest_path)
            continue
        if manifest.get("fingerprint") != fingerprint or manifest.get("status") != "succeeded":
            continue
        saved_payload = manifest.get("fingerprint_payload")
        if not isinstance(saved_payload, dict) or _fingerprint(saved_payload) != fingerprint:
            warning_callback(manifest_path)
            continue
        try:
            valid_artifacts = _validate_manifest_artifacts(manifest, job_dir)
        except (OSError, ValueError, RuntimeError):
            # An unreadable/invalid candidate cannot establish a successful
            # checkpoint. Preserve it and report the failed verification.
            valid_artifacts = False
        if valid_artifacts:
            return job_dir, manifest
        warning_callback(manifest_path)
    return None


def _allocate_job_dir(output_root: Path, stem: str) -> Path:
    index = 1
    while True:
        name = stem if index == 1 else f"{stem}-{index}"
        candidate = output_root / name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate.resolve()
        except FileExistsError:
            index += 1


def _artifact_records(execution_result: object, job_dir: Path) -> list[dict]:
    records: list[dict] = []
    resolved_job = job_dir.resolve()
    artifacts = getattr(execution_result, "artifacts", None)
    if not isinstance(artifacts, tuple) or not artifacts:
        raise RuntimeError("inference returned no artifacts")
    for artifact in artifacts:
        path = Path(getattr(artifact, "path", "")).resolve()
        if not _is_relative_to(path, resolved_job):
            raise RuntimeError(f"inference artifact escaped its job directory: {path}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"inference artifact is missing or empty: {path}")
        records.append(
            {
                "id": str(getattr(artifact, "id", "")),
                "kind": str(getattr(artifact, "kind", "")),
                "path": path.relative_to(resolved_job).as_posix(),
                "media_type": str(getattr(artifact, "media_type", "application/octet-stream")),
                "track_id": getattr(artifact, "track_id", None),
                "size": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return records


class Emitter:
    def __init__(
        self,
        *,
        language: str,
        json_mode: bool,
        quiet: bool,
        stdout: TextIO,
        stderr: TextIO,
    ) -> None:
        self.language = _normalize_language(language)
        self.json_mode = json_mode
        self.quiet = quiet
        self.stdout = stdout
        self.stderr = stderr
        self._last_progress: dict[int, tuple[int, str]] = {}

    def event(
        self,
        name: str,
        *,
        human_key: str | None = None,
        to_stderr: bool = False,
        **data: object,
    ) -> None:
        if self.json_mode:
            payload = {"event": name, "timestamp": _utc_now(), **data}
            print(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                file=self.stdout,
                flush=True,
            )
            return
        if human_key is None or (self.quiet and not to_stderr and name not in {"summary"}):
            return
        print(
            _message(self.language, human_key, **data),
            file=self.stderr if to_stderr else self.stdout,
            flush=True,
        )

    def progress(self, index: int, total: int, progress: object) -> None:
        overall = float(getattr(progress, "overall_progress", 0.0) or 0.0)
        percent = max(0, min(100, int(round(overall * 100))))
        message = str(getattr(progress, "message", "") or "")
        state = (percent, message)
        if self._last_progress.get(index) == state:
            return
        self._last_progress[index] = state
        stage = getattr(getattr(progress, "stage", None), "value", None)
        self.event(
            "progress",
            human_key="progress",
            index=index,
            total=total,
            percent=percent,
            stage=stage,
            message=message,
        )


def _emit_quality_warnings(
    emitter: Emitter,
    *,
    language: str,
    source_path: Path,
    job_dir: Path,
    result_payload: object,
) -> None:
    if not isinstance(result_payload, dict):
        return
    quality_warnings = result_payload.get("quality_warnings", [])
    if not isinstance(quality_warnings, list):
        return
    for warning in quality_warnings:
        warning_name = str(warning)
        emitter.event(
            "quality_warning",
            human_key="warning",
            to_stderr=True,
            source=str(source_path),
            job_dir=str(job_dir),
            warning=warning_name,
            message=_quality_warning_message(language, warning_name),
        )


class CancellationController:
    def __init__(self, emitter: Emitter) -> None:
        self.emitter = emitter
        self.requested = False
        self.active_processor: object | None = None
        self._previous_handlers: dict[int, Any] = {}

    def set_processor(self, processor: object | None) -> None:
        self.active_processor = processor
        if self.requested and processor is not None:
            self._cancel_processor(processor)

    def _cancel_processor(self, processor: object) -> None:
        cancel = getattr(processor, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception as exc:
                self.emitter.event(
                    "cancellation_delivery_failed",
                    human_key="cancel_delivery_failed",
                    to_stderr=True,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

    def request(self, signum: int | None = None, frame: object | None = None) -> None:
        del signum, frame
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        self.emitter.event(
            "cancellation_requested",
            human_key="cancel_requested",
            to_stderr=True,
        )
        if self.active_processor is not None:
            self._cancel_processor(self.active_processor)

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signal_number in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if signal_number is None:
                continue
            self._previous_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, self.request)

    def restore(self) -> None:
        for signal_number, handler in self._previous_handlers.items():
            signal.signal(signal_number, handler)
        self._previous_handlers.clear()


@dataclass
class RunCounts:
    total: int
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    cancelled: int = 0
    not_run: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "skipped": self.skipped,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "not_run": self.not_run,
        }


class InferenceRunner(Protocol):
    def run(self, **kwargs: Any) -> Any: ...  # noqa: E704


def _new_run_summary(command: str, output_root: Path, inputs: list[Path], options: dict) -> dict:
    now = _utc_now()
    return {
        "schema": RUN_SUMMARY_SCHEMA,
        "app_version": __version__,
        "id": uuid.uuid4().hex,
        "command": command,
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "finished_at": None,
        "output_root": str(output_root),
        "inputs": [str(path) for path in inputs],
        "options": options,
        "counts": RunCounts(total=len(inputs)).to_dict(),
        "items": [],
    }


def _run_summary_path(output_root: Path, summary: dict) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"batch-run-{timestamp}-{summary['id'][:8]}.json"


def _write_run_summary(path: Path, summary: dict, counts: RunCounts) -> None:
    summary["updated_at"] = _utc_now()
    summary["counts"] = counts.to_dict()
    _atomic_write_json(path, summary)


def _separation_tempo_source(source_path: Path) -> Path | None:
    """Recover a stem's original mixture from its verified CLI job manifest.

    An explicit reference is handled before this function. Recognized but
    unverifiable separation provenance must fail, never use sparse stem beats.
    """
    split_modes = {ProcessingMode.VOCAL_SPLIT.value, ProcessingMode.SIX_STEM_SPLIT.value}
    for directory in source_path.parents:
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not isinstance(manifest, dict)
                or type(manifest.get("schema")) is not int
                or manifest["schema"] != MANIFEST_SCHEMA
            ):
                raise ValueError("invalid separation manifest schema")
            options = manifest.get("options")
            if not isinstance(options, dict):
                raise ValueError("invalid separation options")
            if manifest.get("kind") != "primary" or options.get("processing_mode") not in split_modes:
                return None
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, list):
                raise ValueError("invalid separated track records")
            matches = [
                artifact for artifact in artifacts
                if isinstance(artifact, dict)
                and isinstance(artifact.get("path"), str)
                and (directory / artifact["path"]).resolve() == source_path
            ]
            if not matches:
                return None
            if manifest.get("status") != "succeeded" or len(matches) != 1:
                raise ValueError("separation is incomplete or has ambiguous track records")
            if not _validate_manifest_artifacts({"artifacts": matches}, directory):
                raise ValueError("separated WAV size or SHA-256 no longer matches its manifest")
            original = manifest.get("source")
            if not isinstance(original, dict) or not isinstance(original.get("path"), str):
                raise ValueError("missing original mixture identity")
            original_path = Path(original["path"]).expanduser().resolve()
            current = _source_identity(original_path)
            if current["size"] != original.get("size") or current["sha256"] != original.get("sha256"):
                raise ValueError("original mixture size or SHA-256 no longer matches its manifest")
            return original_path
        except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"cannot verify original mixture for {source_path}: {exc}; "
                f"manifest={manifest_path}. Use --tempo-source to select a verified reference explicitly."
            ) from exc
    return None


def _validate_tempo_source(
    namespace: argparse.Namespace, source_path: Path | None = None
) -> Path | None:
    value = getattr(namespace, "tempo_source", None)
    if value is None and namespace.job_kind == "manual_midi" and source_path is not None:
        value = _separation_tempo_source(source_path)
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"tempo source does not exist or is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(f"tempo source has an unsupported audio format: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"tempo source audio file is empty: {path}")
    return path


def _run_routes(namespace: argparse.Namespace, stdout: TextIO) -> int:
    if namespace.json:
        print(
            json.dumps(
                {"count": len(MANUAL_MIDI_ROUTES), "routes": list(MANUAL_MIDI_ROUTES)},
                ensure_ascii=False,
                indent=2,
            ),
            file=stdout,
        )
    else:
        for route in MANUAL_MIDI_ROUTES:
            print(route, file=stdout)
    return 0


class _JsonEventStream:
    """Normalize Windows UCRT's closed-pipe EINVAL only at the output boundary."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.broken_pipe = False

    def _call(self, operation: Callable, *args):
        try:
            return operation(*args)
        except OSError as exc:
            is_broken = isinstance(exc, BrokenPipeError)
            if os.name == "nt" and exc.errno == errno.EINVAL:
                is_broken = stat.S_ISFIFO(os.fstat(self.stream.fileno()).st_mode)
            if is_broken:
                self.broken_pipe = True
                raise BrokenPipeError(errno.EPIPE, "CLI output pipe is closed") from exc
            raise

    def write(self, value: str):
        return self._call(self.stream.write, value)

    def flush(self):
        return self._call(self.stream.flush)

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def _json_output_context(stdout: TextIO, stderr: TextIO, *, json_mode: bool = True):
    """Protect CLI output and reserve JSON stdout even for native fd 1 writes.

    The duplicate keeps emitter output on the original destination while
    Python diagnostics, C writes and inherited child stdout go to stderr.
    In-memory streams have no native descriptor and only need Python routing.
    This context belongs to the single-threaded CLI entry, never a web request.
    """
    try:
        output_fd, error_fd = stdout.fileno(), stderr.fileno()
    except (AttributeError, OSError):
        with redirect_stdout(stderr) if json_mode else nullcontext():
            yield stdout
        return
    if output_fd != 1:
        with redirect_stdout(stderr) if json_mode else nullcontext():
            yield stdout
        return

    stdout.flush()
    stderr.flush()
    with os.fdopen(
        os.dup(output_fd), "w", encoding=stdout.encoding or "utf-8",
        errors=stdout.errors or "strict", buffering=1,
    ) as machine_output:
        event_output = _JsonEventStream(machine_output)
        try:
            if json_mode:
                os.dup2(error_fd, output_fd)
            with redirect_stdout(stderr if json_mode else event_output):
                yield event_output
        finally:
            stderr.flush()
            if event_output.broken_pipe:
                # The failed buffered write may be retried by TextIOWrapper's
                # close(). Keep the original error and the explicit failure
                # exit instead of generating another EINVAL during shutdown.
                with open(os.devnull, "w") as sink:
                    os.dup2(sink.fileno(), machine_output.fileno())
            os.dup2(machine_output.fileno(), output_fd)


@contextmanager
def _engine_output_context(namespace: argparse.Namespace, stderr: TextIO):
    if namespace.json:
        with redirect_stdout(stderr):
            yield None
        return
    if namespace.quiet:
        diagnostics = StringIO()
        with redirect_stdout(diagnostics), redirect_stderr(diagnostics):
            yield diagnostics
        return
    with nullcontext():
        yield None


def _replay_failed_engine_diagnostics(diagnostics: StringIO | None, stderr: TextIO) -> None:
    if diagnostics is None:
        return
    value = diagnostics.getvalue()
    if value:
        print(value, file=stderr, end="" if value.endswith("\n") else "\n", flush=True)


def _process_jobs(
    namespace: argparse.Namespace,
    *,
    engine_factory: Callable[[], InferenceRunner],
    stdout: TextIO,
    stderr: TextIO,
    install_signal_handlers: bool,
) -> int:
    language = _normalize_language(namespace.language)
    emitter = Emitter(
        language=language,
        json_mode=bool(namespace.json),
        quiet=bool(namespace.quiet),
        stdout=stdout,
        stderr=stderr,
    )
    options = _validated_options(namespace)
    output_root = (namespace.output or _default_output_root()).expanduser().resolve()
    inputs = discover_inputs(namespace.inputs, bool(namespace.recursive), output_root)
    tempo_source_paths = {
        source: _validate_tempo_source(namespace, source) for source in inputs
    }

    if namespace.dry_run:
        for index, planned_source in enumerate(inputs, 1):
            emitter.event(
                "planned",
                human_key="dry_item",
                index=index,
                total=len(inputs),
                source=str(planned_source),
                kind=namespace.job_kind,
                options=options,
                tempo_source=(
                    str(tempo_source_paths[planned_source] or planned_source)
                    if namespace.job_kind == "manual_midi" else None
                ),
            )
        emitter.event("plan_summary", human_key="dry_summary", total=len(inputs))
        return 0

    tempo_source_identities = {
        path: _source_identity(path)
        for path in set(tempo_source_paths.values()) if path is not None
    }
    output_root.mkdir(parents=True, exist_ok=True)
    if not output_root.is_dir():
        raise RuntimeError(f"output root is not a directory: {output_root}")
    summary = _new_run_summary(namespace.command, output_root, inputs, options)
    summary_path = _run_summary_path(output_root, summary)
    counts = RunCounts(total=len(inputs))
    _write_run_summary(summary_path, summary, counts)
    controller = CancellationController(emitter)
    if install_signal_handlers:
        controller.install()

    try:
        for index, source_path in enumerate(inputs, 1):
            if controller.requested:
                break
            job_dir: Path | None = None
            manifest_path: Path | None = None
            manifest: dict | None = None
            resumable: tuple[Path, dict] | None = None
            try:
                source_identity = _source_identity(source_path)
                tempo_source_path = tempo_source_paths[source_path]
                tempo_source = (
                    tempo_source_identities[tempo_source_path]
                    if tempo_source_path is not None and tempo_source_path != source_path
                    else None
                )
                if tempo_source is not None:
                    _assert_source_unchanged(tempo_source)
                if controller.requested:
                    raise KeyboardInterrupt
                fingerprint_payload = _fingerprint_payload(
                    kind=namespace.job_kind,
                    source=source_identity,
                    options=options,
                    tempo_source=tempo_source,
                )
                fingerprint = _fingerprint(fingerprint_payload)
                stem = _safe_job_stem(source_path)

                def warn_invalid(path: Path) -> None:
                    emitter.event(
                        "resume_artifact_invalid",
                        human_key="invalid_resume",
                        to_stderr=True,
                        path=str(path),
                    )

                if not namespace.rerun:
                    resumable = _find_resumable_job(
                        output_root,
                        stem,
                        fingerprint,
                        warn_invalid,
                    )
                if resumable is None:
                    job_dir = _allocate_job_dir(output_root, stem)
                    manifest_path = job_dir / MANIFEST_NAME
                    started_at = _utc_now()
                    manifest = {
                        "schema": MANIFEST_SCHEMA,
                        "app_version": __version__,
                        "status": "running",
                        "created_at": started_at,
                        "updated_at": started_at,
                        "finished_at": None,
                        "kind": namespace.job_kind,
                        "source": source_identity,
                        "tempo_source": tempo_source,
                        "options": options,
                        "fingerprint_payload": fingerprint_payload,
                        "fingerprint": fingerprint,
                        "result": None,
                        "artifacts": [],
                        "error": None,
                    }
                    _atomic_write_json(manifest_path, manifest)
                if controller.requested:
                    raise KeyboardInterrupt
            except (InterruptedError, KeyboardInterrupt) as exc:
                controller.requested = True
                error_payload = {"type": type(exc).__name__, "message": str(exc)}
                if manifest is not None and manifest_path is not None and manifest_path.is_file():
                    manifest.update(
                        {
                            "status": "cancelled",
                            "updated_at": _utc_now(),
                            "finished_at": _utc_now(),
                            "error": error_payload,
                        }
                    )
                    _atomic_write_json(manifest_path, manifest)
                counts.cancelled += 1
                cancelled_item: dict[str, Any] = {
                    "source": str(source_path),
                    "status": "cancelled",
                    "error": error_payload,
                }
                if job_dir is not None:
                    cancelled_item["job_dir"] = str(job_dir)
                if manifest_path is not None and manifest_path.is_file():
                    cancelled_item["manifest"] = str(manifest_path)
                summary["items"].append(cancelled_item)
                _write_run_summary(summary_path, summary, counts)
                emitter.event(
                    "cancelled",
                    human_key="cancelled",
                    to_stderr=True,
                    index=index,
                    total=len(inputs),
                    source=str(source_path),
                    job_dir=str(job_dir) if job_dir is not None else None,
                )
                break
            except Exception as exc:
                error_payload = {"type": type(exc).__name__, "message": str(exc)}
                if namespace.verbose:
                    error_payload["traceback"] = traceback.format_exc()
                if manifest is not None and manifest_path is not None and manifest_path.is_file():
                    manifest.update(
                        {
                            "status": "failed",
                            "updated_at": _utc_now(),
                            "finished_at": _utc_now(),
                            "error": error_payload,
                        }
                    )
                    _atomic_write_json(manifest_path, manifest)
                counts.failed += 1
                failed_item: dict[str, Any] = {
                    "source": str(source_path),
                    "status": "failed",
                    "error": error_payload,
                }
                if job_dir is not None:
                    failed_item["job_dir"] = str(job_dir)
                if manifest_path is not None and manifest_path.is_file():
                    failed_item["manifest"] = str(manifest_path)
                summary["items"].append(failed_item)
                _write_run_summary(summary_path, summary, counts)
                emitter.event(
                    "failed",
                    human_key="failed",
                    to_stderr=True,
                    index=index,
                    total=len(inputs),
                    source=str(source_path),
                    job_dir=str(job_dir) if job_dir is not None else None,
                    error=(
                        f"{exc}\n{error_payload['traceback']}" if namespace.verbose else str(exc)
                    ),
                    error_type=type(exc).__name__,
                    traceback=error_payload.get("traceback"),
                )
                if namespace.fail_fast:
                    break
                continue

            if resumable is not None:
                job_dir, manifest = resumable
                result_payload = manifest.get("result")
                counts.skipped += 1
                skipped_item: dict[str, Any] = {
                    "source": str(source_path),
                    "status": "skipped",
                    "job_dir": str(job_dir),
                    "manifest": str(job_dir / MANIFEST_NAME),
                    "artifacts": manifest["artifacts"],
                    "result": result_payload,
                }
                summary["items"].append(skipped_item)
                _write_run_summary(summary_path, summary, counts)
                emitter.event(
                    "skipped",
                    human_key="skipped",
                    index=index,
                    total=len(inputs),
                    source=str(source_path),
                    job_dir=str(job_dir),
                    artifacts=manifest["artifacts"],
                    result=result_payload,
                )
                _emit_quality_warnings(
                    emitter,
                    language=language,
                    source_path=source_path,
                    job_dir=job_dir,
                    result_payload=result_payload,
                )
                continue

            if job_dir is None or manifest_path is None or manifest is None:
                raise RuntimeError("CLI job preparation did not produce a runnable manifest")
            emitter.event(
                "started",
                human_key="start",
                index=index,
                total=len(inputs),
                source=str(source_path),
                job_dir=str(job_dir),
            )
            engine_diagnostics: StringIO | None = None
            try:
                with _engine_output_context(namespace, stderr) as engine_diagnostics:
                    engine = engine_factory()
                    execution_result = engine.run(
                        kind=namespace.job_kind,
                        source_path=source_path,
                        output_dir=job_dir,
                        options=options,
                        progress_callback=lambda progress, item_index=index: emitter.progress(
                            item_index, len(inputs), progress
                        ),
                        processor_callback=controller.set_processor,
                        track_id=(
                            source_path.stem if namespace.job_kind == "manual_midi" else None
                        ),
                        tempo_source_path=tempo_source_path,
                    )
                if controller.requested:
                    raise InterruptedError("conversion cancelled by user")
                _assert_source_unchanged(source_identity)
                if tempo_source is not None:
                    _assert_source_unchanged(tempo_source)
                artifacts = _artifact_records(execution_result, job_dir)
                result_payload = getattr(execution_result, "result", None)
                if not isinstance(result_payload, dict):
                    raise RuntimeError("inference returned an invalid result payload")
                manifest.update(
                    {
                        "status": "succeeded",
                        "updated_at": _utc_now(),
                        "finished_at": _utc_now(),
                        "result": result_payload,
                        "artifacts": artifacts,
                    }
                )
                _atomic_write_json(manifest_path, manifest)
            except (InterruptedError, KeyboardInterrupt) as exc:
                controller.requested = True
                counts.cancelled += 1
                manifest.update(
                    {
                        "status": "cancelled",
                        "updated_at": _utc_now(),
                        "finished_at": _utc_now(),
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
                _atomic_write_json(manifest_path, manifest)
                summary["items"].append(
                    {
                        "source": str(source_path),
                        "status": "cancelled",
                        "job_dir": str(job_dir),
                        "manifest": str(manifest_path),
                        "error": manifest["error"],
                    }
                )
                _write_run_summary(summary_path, summary, counts)
                emitter.event(
                    "cancelled",
                    human_key="cancelled",
                    to_stderr=True,
                    index=index,
                    total=len(inputs),
                    source=str(source_path),
                    job_dir=str(job_dir),
                )
                break
            except Exception as exc:
                _replay_failed_engine_diagnostics(engine_diagnostics, stderr)
                error_payload = {"type": type(exc).__name__, "message": str(exc)}
                if namespace.verbose:
                    error_payload["traceback"] = traceback.format_exc()
                manifest.update(
                    {
                        "status": "failed",
                        "updated_at": _utc_now(),
                        "finished_at": _utc_now(),
                        "error": error_payload,
                    }
                )
                _atomic_write_json(manifest_path, manifest)
                counts.failed += 1
                summary["items"].append(
                    {
                        "source": str(source_path),
                        "status": "failed",
                        "job_dir": str(job_dir),
                        "manifest": str(manifest_path),
                        "error": error_payload,
                    }
                )
                _write_run_summary(summary_path, summary, counts)
                emitter.event(
                    "failed",
                    human_key="failed",
                    to_stderr=True,
                    index=index,
                    total=len(inputs),
                    source=str(source_path),
                    job_dir=str(job_dir),
                    error=(
                        f"{exc}\n{error_payload['traceback']}" if namespace.verbose else str(exc)
                    ),
                    error_type=type(exc).__name__,
                    traceback=error_payload.get("traceback"),
                )
                if namespace.fail_fast:
                    break
            else:
                # The job manifest is committed. A later summary/stream failure
                # must stop the run without rewriting this success as a model
                # failure or counting the same input twice.
                counts.succeeded += 1
                summary["items"].append(
                    {
                        "source": str(source_path),
                        "status": "succeeded",
                        "job_dir": str(job_dir),
                        "manifest": str(manifest_path),
                        "result": result_payload,
                        "artifacts": artifacts,
                    }
                )
                _write_run_summary(summary_path, summary, counts)
                emitter.event(
                    "succeeded",
                    human_key="success",
                    index=index,
                    total=len(inputs),
                    source=str(source_path),
                    job_dir=str(job_dir),
                    result=result_payload,
                    artifacts=artifacts,
                )
                _emit_quality_warnings(
                    emitter,
                    language=language,
                    source_path=source_path,
                    job_dir=job_dir,
                    result_payload=result_payload,
                )
                if result_payload.get("manual_midi_required"):
                    emitter.event(
                        "manual_midi_required",
                        human_key="no_auto_midi",
                        source=str(source_path),
                        job_dir=str(job_dir),
                    )
            finally:
                controller.set_processor(None)
    except KeyboardInterrupt:
        controller.requested = True
    finally:
        controller.restore()

    counts.not_run = max(
        0,
        counts.total - counts.succeeded - counts.skipped - counts.failed - counts.cancelled,
    )
    if controller.requested:
        summary["status"] = "cancelled"
    elif counts.failed:
        summary["status"] = "failed"
    else:
        summary["status"] = "succeeded"
    summary["finished_at"] = _utc_now()
    _write_run_summary(summary_path, summary, counts)
    emitter.event("summary", human_key="summary", **counts.to_dict())
    emitter.event(
        "summary_path",
        human_key="summary_path",
        path=str(summary_path),
        status=summary["status"],
    )
    if controller.requested:
        return 130
    return 1 if counts.failed else 0


def main(
    argv: Sequence[str] | None = None,
    *,
    engine_factory: Callable[[], InferenceRunner] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    install_signal_handlers: bool = True,
) -> int:
    arguments = _normalize_argv(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser(arguments)
    namespace = parser.parse_args(arguments)
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    try:
        output_context = _json_output_context(out, err, json_mode=namespace.json)
        with output_context as machine_output:
            if namespace.command == "routes":
                return _run_routes(namespace, machine_output)
            if engine_factory is None:
                from src.web_api.engine import InferenceEngine

                engine_factory = InferenceEngine
            return _process_jobs(
                namespace,
                engine_factory=engine_factory,
                stdout=machine_output,
                stderr=err,
                install_signal_handlers=install_signal_handlers,
            )
    except BrokenPipeError:
        # Follow Python's SIGPIPE guidance: stop with failure and prevent a
        # second shutdown flush to an already closed standard output pipe.
        # Never attempt to report this failure on the broken JSON stream.
        if out is sys.stdout:
            with open(os.devnull, "w") as sink:
                os.dup2(sink.fileno(), out.fileno())
        print(_message(namespace.language, "pipe_closed"), file=err, flush=True)
        return 1
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        language = _normalize_language(getattr(namespace, "language", "zh_CN"))
        if bool(getattr(namespace, "json", False)):
            print(
                json.dumps(
                    {
                        "event": "fatal",
                        "timestamp": _utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=out,
                flush=True,
            )
        else:
            print(_message(language, "fatal", error=exc), file=err, flush=True)
        return 2


__all__ = [
    "MANIFEST_NAME",
    "SUPPORTED_AUDIO_SUFFIXES",
    "build_parser",
    "discover_inputs",
    "main",
]
