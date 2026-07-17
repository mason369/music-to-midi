"""MIROS multi-instrument transcription wrapper."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.i18n.translator import Translator
from src.models.data_models import Config, NoteEvent
from src.utils.gpu_utils import clear_gpu_memory
from src.utils.midi_output import (
    publish_midi_output,
    remove_temporary_midi,
    unique_midi_temp_path,
)
from src.utils.runtime_paths import get_miros_source_dir, is_frozen_app

logger = logging.getLogger(__name__)


MIROS_SOURCE_COMMIT = "668a0aa6357bb3f09e767c9ece378956c2ffd182"
MIROS_UNPATCHED_SOURCE_SHA256 = "38a067527acf6b17f458053077ff143fcedc253edc36f28b69e6ce441c0ca35d"
MIROS_PREVIOUS_PATCHED_SOURCE_SHA256 = (
    "b9613befc9cc353a2bd75eb85ceb5fed76b4d3d1b066bad1aee3a6f8d82fd2a2"
)
MIROS_PATCHED_SOURCE_SHA256 = "69bd2872fe62c345323b3c296e53ecc0ee0133d96e00ddaa7dd25632a39f808c"
MIROS_PRETRAINED_COMMIT = "546287d5e3e9ea5b42a4135d1dbca96ac12a0a9c"
MIROS_PRETRAINED_EXACT_BYTES = 1_316_802_088
MIROS_PRETRAINED_SHA256 = "218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded"
MIROS_FINETUNED_EXACT_BYTES = 4_347_922_234
MIROS_FINETUNED_SHA256 = "b1b8c167b3d2e3eaeb19202cd3fd366bb43492cd7720ff1516e1553c72e356e5"

_MIROS_SOURCE_EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache"}


def _find_module_spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None


def _canonical_source_bytes(data: bytes) -> bytes:
    """Normalize checkout line endings without rewriting binary source assets."""
    if b"\0" in data:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def compute_miros_source_tree_sha256(repo_dir: Path | str) -> str:
    """Hash the complete patched source tree, excluding VCS/cache data and model weights."""
    repo = Path(repo_dir)
    excluded_files = {
        MirosTranscriber.PRETRAINED_REL_PATH.as_posix(),
        MirosTranscriber.CHECKPOINT_REL_PATH.as_posix(),
    }
    entries: list[tuple[str, str]] = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repo)
        if any(part in _MIROS_SOURCE_EXCLUDED_DIRS for part in relative.parts):
            continue
        relative_name = relative.as_posix()
        if relative_name in excluded_files:
            continue
        content_sha256 = hashlib.sha256(_canonical_source_bytes(path.read_bytes())).hexdigest()
        entries.append((relative_name, content_sha256))

    digest = hashlib.sha256()
    for relative_name, content_sha256 in sorted(entries):
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_head_commit(repo_dir: Path) -> tuple[Optional[str], Optional[str]]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"cannot read MIROS Git HEAD: {exc}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return None, f"cannot read MIROS Git HEAD (exit={completed.returncode}): {detail}"
    return completed.stdout.strip().lower(), None


def get_miros_source_identity_error(repo_dir: Path | str) -> str:
    """Return an explicit reason when source is not the pinned commit plus approved patches."""
    repo = Path(repo_dir)
    if not repo.is_dir():
        return f"MIROS source directory does not exist: {repo}"

    for required in (repo / "main.py", repo / "transcribe.py"):
        if not required.is_file():
            return f"MIROS source is incomplete; missing required file: {required}"

    if (repo / ".git").exists():
        head, error = _git_head_commit(repo)
        if error:
            return error
        if head != MIROS_SOURCE_COMMIT:
            return (
                "MIROS source commit mismatch: "
                f"expected {MIROS_SOURCE_COMMIT}, got {head or '<empty>'}"
            )

    try:
        actual_sha256 = compute_miros_source_tree_sha256(repo)
    except OSError as exc:
        return f"cannot hash MIROS source tree {repo}: {exc}"
    if actual_sha256 != MIROS_PATCHED_SOURCE_SHA256:
        return (
            "MIROS patched source tree SHA256 mismatch: "
            f"expected {MIROS_PATCHED_SOURCE_SHA256}, got {actual_sha256}. "
            "Only the deterministic decmod, RoPE, and bounded-inference patches are allowed."
        )
    return ""


def get_miros_source_preparation_error(repo_dir: Path | str) -> str:
    """Accept only pristine/current source or the prior approved upgrade input."""
    repo = Path(repo_dir)
    if not repo.is_dir():
        return f"MIROS source directory does not exist: {repo}"
    for required in (repo / "main.py", repo / "transcribe.py"):
        if not required.is_file():
            return f"MIROS source is incomplete; missing required file: {required}"

    if (repo / ".git").exists():
        head, error = _git_head_commit(repo)
        if error:
            return error
        if head != MIROS_SOURCE_COMMIT:
            return (
                "MIROS source commit mismatch: "
                f"expected {MIROS_SOURCE_COMMIT}, got {head or '<empty>'}"
            )

    try:
        actual_sha256 = compute_miros_source_tree_sha256(repo)
    except OSError as exc:
        return f"cannot hash MIROS source tree {repo}: {exc}"
    allowed = {
        MIROS_UNPATCHED_SOURCE_SHA256,
        MIROS_PREVIOUS_PATCHED_SOURCE_SHA256,
        MIROS_PATCHED_SOURCE_SHA256,
    }
    if actual_sha256 not in allowed:
        return (
            "MIROS source tree is neither the pristine pinned commit nor its approved patched form: "
            f"got {actual_sha256}"
        )
    return ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_miros_file_identity_error(
    path: Path | str,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        return f"{label} is missing: {artifact}"
    try:
        actual_size = artifact.stat().st_size
    except OSError as exc:
        return f"cannot stat {label} {artifact}: {exc}"
    if actual_size != expected_size:
        return (
            f"{label} size mismatch: expected {expected_size} bytes, "
            f"got {actual_size} bytes ({artifact})"
        )
    try:
        actual_sha256 = _sha256_file(artifact)
    except OSError as exc:
        return f"cannot hash {label} {artifact}: {exc}"
    if actual_sha256.lower() != expected_sha256.lower():
        return (
            f"{label} SHA256 mismatch: expected {expected_sha256.lower()}, "
            f"got {actual_sha256.lower()} ({artifact})"
        )
    return ""


def get_miros_weight_identity_error(repo_dir: Path | str) -> str:
    repo = Path(repo_dir)
    checks = (
        (
            repo / MirosTranscriber.PRETRAINED_REL_PATH,
            MIROS_PRETRAINED_EXACT_BYTES,
            MIROS_PRETRAINED_SHA256,
            "MIROS MusicFM pretrained weight",
        ),
        (
            repo / MirosTranscriber.CHECKPOINT_REL_PATH,
            MIROS_FINETUNED_EXACT_BYTES,
            MIROS_FINETUNED_SHA256,
            "MIROS fine-tuned checkpoint",
        ),
    )
    for path, expected_size, expected_sha256, label in checks:
        error = get_miros_file_identity_error(
            path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=label,
        )
        if error:
            return error
    return ""


class MirosTranscriber:
    """Run a local ai4m-miros checkout as an optional backend."""

    PRETRAINED_REL_PATH = Path("model/musicfm/data/pretrained_msd.pt")
    CHECKPOINT_REL_PATH = Path(
        "logs/Multi_longer_seq_length_frozen_enc_silu/le2bzt53/checkpoints/last.ckpt"
    )
    REQUIRED_MODULES = (
        "torch",
        "torchaudio",
        "pytorch_lightning",
        "transformers",
        "einops",
        "torchmetrics",
        "librosa",
        "soundfile",
        "pretty_midi",
        "mir_eval",
        "h5py",
        "soxr",
        "wandb",
        "mido",
        "smart_open",
    )

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._cancelled = False
        self._cancel_check: Optional[Callable[[], bool]] = None
        self._process: Optional[subprocess.Popen[str]] = None
        self._process_lock = threading.Lock()
        self._translator = Translator(getattr(self.config, "language", Translator.DEFAULT_LANGUAGE))

    def _pt(self, key: str, **kwargs) -> str:
        return self._translator.t(key, **kwargs)

    @classmethod
    def _repo_dir(cls) -> Optional[Path]:
        return get_miros_source_dir()

    @classmethod
    def _entrypoint_path(cls) -> Optional[Path]:
        repo_dir = cls._repo_dir()
        if repo_dir is None:
            return None
        entrypoint = repo_dir / "main.py"
        if entrypoint.exists():
            return entrypoint
        return None

    @classmethod
    def _missing_modules(cls) -> List[str]:
        missing: List[str] = []
        for module_name in cls.REQUIRED_MODULES:
            if _find_module_spec(module_name) is None:
                missing.append(module_name)
        return missing

    @classmethod
    def get_unavailable_reason(cls) -> str:
        repo_dir = cls._repo_dir()
        if repo_dir is None:
            return (
                "MIROS 不可用：未找到 ai4m-miros 代码目录。\n\n"
                "请将仓库放到以下任一位置：\n"
                "  ai4m-miros/\n"
                "  external/ai4m-miros/\n\n"
                "上游仓库：\n"
                "  https://github.com/amt-os/ai4m-miros"
            )

        source_error = get_miros_source_identity_error(repo_dir)
        if source_error:
            return (
                "MIROS 不可用：源码身份校验失败。\n\n"
                f"{source_error}\n"
                "请重新运行 download_miros_model.py 获取固定版本源码。"
            )

        missing_modules = cls._missing_modules()
        if missing_modules:
            return (
                "MIROS 不可用：缺少运行依赖。\n\n"
                f"缺少模块：{', '.join(missing_modules)}\n"
                "请先安装 requirements.txt 中的依赖，并补充 ai4m-miros 要求的环境。"
            )

        weight_error = get_miros_weight_identity_error(repo_dir)
        if weight_error:
            return (
                "MIROS 不可用：模型权重身份校验失败。\n\n"
                f"{weight_error}\n"
                "请重新运行 download_miros_model.py 准备固定版本权重。"
            )

        return ""

    @classmethod
    def is_available(cls) -> bool:
        return cls.get_unavailable_reason() == ""

    @classmethod
    def is_model_available(cls) -> bool:
        repo_dir = cls._repo_dir()
        if repo_dir is None:
            return False
        return not (
            get_miros_source_identity_error(repo_dir) or get_miros_weight_identity_error(repo_dir)
        )

    def set_cancel_check(self, callback: Optional[Callable[[], bool]]) -> None:
        self._cancel_check = callback

    def cancel(self) -> None:
        self._cancelled = True
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            logger.info("正在终止 MIROS 子进程...")
            process.terminate()

    def unload_model(self) -> None:
        clear_gpu_memory()

    def reset_cancel(self) -> None:
        self._cancelled = False

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise InterruptedError("MIROS 转写处理已取消")
        if self._cancel_check and self._cancel_check():
            raise InterruptedError("MIROS 转写处理已取消")

    @staticmethod
    def _tail_output(stdout: str, stderr: str, limit: int = 40) -> str:
        lines = [line for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
        if not lines:
            return ""
        return "\n".join(lines[-limit:])

    @staticmethod
    def _format_worker_status(status_path: Optional[Path]) -> str:
        if status_path is None:
            return ""
        if not status_path.exists():
            return f"MIROS worker 状态文件未生成: {status_path}"
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return f"MIROS worker 状态文件无法读取: {status_path}\n{exc}"

        lines = [
            "MIROS worker 状态:",
            f"状态文件: {status_path}",
        ]
        if isinstance(payload, dict):
            if "ok" in payload:
                lines.append(f"ok: {payload['ok']}")
            for key, label in (
                ("error", "错误"),
                ("traceback", "Traceback"),
                ("output", "输出"),
                ("output_exists", "输出是否存在"),
                ("output_size", "输出大小"),
            ):
                value = payload.get(key)
                if value not in (None, ""):
                    lines.append(f"{label}: {value}")
        else:
            lines.append(str(payload))
        return "\n".join(lines)

    @staticmethod
    def _format_process_failure(
        returncode: int,
        stdout: str,
        stderr: str,
        status_path: Optional[Path],
    ) -> str:
        lines = [f"MIROS 子进程退出码: {returncode}"]

        worker_status = MirosTranscriber._format_worker_status(status_path)
        if worker_status:
            lines.append(worker_status)

        tail = MirosTranscriber._tail_output(stdout, stderr)
        if tail:
            lines.append("MIROS 输出尾部:")
            lines.append(tail)
        else:
            lines.append("MIROS 没有输出 stdout/stderr")
        return "\n".join(lines)

    @staticmethod
    def _format_missing_output_error(
        out_path: Path,
        stdout: str,
        stderr: str,
        status_path: Optional[Path] = None,
    ) -> str:
        lines = [
            "MIROS 未生成 MIDI 输出",
            f"期望输出: {out_path}",
            f"输出目录: {out_path.parent}",
        ]

        if out_path.parent.exists():
            midi_candidates = sorted(
                {
                    candidate.resolve()
                    for pattern in ("*.mid", "*.midi")
                    for candidate in out_path.parent.glob(pattern)
                    if candidate.is_file()
                }
            )
            if midi_candidates:
                lines.append("输出目录中的 MIDI 文件:")
                lines.extend(f"  {candidate}" for candidate in midi_candidates[:12])
                if len(midi_candidates) > 12:
                    lines.append(f"  ... 另外 {len(midi_candidates) - 12} 个")
            else:
                lines.append("输出目录中未发现 .mid/.midi 文件")
        else:
            lines.append("输出目录不存在")

        worker_status = MirosTranscriber._format_worker_status(status_path)
        if worker_status:
            lines.append(worker_status)

        tail = MirosTranscriber._tail_output(stdout, stderr)
        if tail:
            lines.append("MIROS 输出尾部:")
            lines.append(tail)
        else:
            lines.append("MIROS 没有输出 stdout/stderr")
        return "\n".join(lines)

    @staticmethod
    def _build_command(
        entrypoint: Path,
        input_path: Path,
        out_path: Path,
        status_path: Optional[Path] = None,
    ) -> List[str]:
        if is_frozen_app():
            command = [
                sys.executable,
                "--miros-worker",
                "-i",
                str(input_path),
                "-o",
                str(out_path),
            ]
            if status_path is not None:
                command.extend(["--status-json", str(status_path)])
            return command
        return [
            sys.executable,
            str(entrypoint),
            "-i",
            str(input_path),
            "-o",
            str(out_path),
        ]

    def transcribe_to_midi(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        repo_dir = self._repo_dir()
        entrypoint = self._entrypoint_path()
        unavailable_reason = self.get_unavailable_reason()
        if unavailable_reason:
            raise RuntimeError(unavailable_reason)
        if repo_dir is None or entrypoint is None:
            raise RuntimeError("MIROS 不可用：未找到可执行入口")

        self._check_cancelled()

        input_path = Path(audio_path).resolve()
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_path = unique_midi_temp_path(out_path, "miros")

        status_path = attempt_path.with_suffix(".status.json") if is_frozen_app() else None
        if status_path is not None and status_path.exists():
            status_path.unlink()

        def _cleanup_attempt_files() -> None:
            remove_temporary_midi(attempt_path)
            if status_path is not None:
                try:
                    status_path.unlink()
                except FileNotFoundError:
                    pass

        def _terminate_and_reap_process() -> None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            else:
                process.wait(timeout=0)

        command = self._build_command(entrypoint, input_path, attempt_path, status_path)

        if progress_callback:
            progress_callback(0.05, self._pt("progress.preparing_miros"))

        logger.info("Running MIROS transcription: %s", " ".join(command))
        process_env = dict(os.environ)
        process_env["PYTHONIOENCODING"] = "utf-8"
        process_env["PYTHONUTF8"] = "1"
        process_env.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )
        process = subprocess.Popen(
            command,
            cwd=str(repo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=process_env,
        )
        with self._process_lock:
            self._process = process

        stdout = ""
        stderr = ""
        started_at = time.monotonic()
        try:
            if progress_callback:
                progress_callback(0.50, self._pt("progress.running_miros"))
            try:
                communicate_params = inspect.signature(process.communicate).parameters
                supports_timeout = "timeout" in communicate_params
            except (TypeError, ValueError):
                supports_timeout = True

            if not supports_timeout:
                stdout, stderr = process.communicate()
            else:
                while True:
                    self._check_cancelled()
                    try:
                        stdout, stderr = process.communicate(timeout=5.0)
                        break
                    except subprocess.TimeoutExpired:
                        elapsed_seconds = int(time.monotonic() - started_at)
                        logger.info(
                            "MIROS inference still running: elapsed=%ss output=%s",
                            elapsed_seconds,
                            attempt_path,
                        )
                        if progress_callback:
                            progress_callback(
                                0.50,
                                self._pt(
                                    "progress.running_miros_elapsed",
                                    seconds=elapsed_seconds,
                                    output=attempt_path.name,
                                ),
                            )
        except InterruptedError:
            _terminate_and_reap_process()
            _cleanup_attempt_files()
            raise
        except Exception:
            _terminate_and_reap_process()
            _cleanup_attempt_files()
            raise
        finally:
            with self._process_lock:
                self._process = None

        try:
            self._check_cancelled()
        except InterruptedError:
            _cleanup_attempt_files()
            raise

        if process.returncode != 0:
            detail = self._format_process_failure(
                process.returncode,
                stdout or "",
                stderr or "",
                status_path,
            )
            _cleanup_attempt_files()
            raise RuntimeError(f"MIROS 转写失败:\n{detail}")
        if not attempt_path.exists():
            detail = self._format_missing_output_error(
                out_path,
                stdout or "",
                stderr or "",
                status_path,
            )
            detail += f"\n本次隔离输出: {attempt_path}"
            _cleanup_attempt_files()
            raise RuntimeError(detail)

        try:
            published_path = publish_midi_output(attempt_path, out_path, "MIROS")
        except Exception as exc:
            detail = self._format_missing_output_error(
                out_path,
                stdout or "",
                stderr or "",
                status_path,
            )
            detail += f"\n本次隔离输出: {attempt_path}"
            _cleanup_attempt_files()
            raise RuntimeError(f"MIROS MIDI 输出校验失败: {exc}\n{detail}") from exc
        _cleanup_attempt_files()

        if progress_callback:
            progress_callback(1.0, self._pt("progress.miros_complete"))
        return published_path

    def transcribe_precise(
        self,
        audio_path: str,
        quality: str = "best",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[Dict[int, List[NoteEvent]], Dict[int, List[NoteEvent]]]:
        import pretty_midi

        with tempfile.TemporaryDirectory(prefix="miros_") as tmp_dir:
            midi_path = Path(tmp_dir) / f"{Path(audio_path).stem}.mid"

            def _midi_progress(progress: float, message: str) -> None:
                if progress_callback:
                    progress_callback(min(0.90, progress * 0.90), message)

            self.transcribe_to_midi(
                audio_path=audio_path,
                output_path=str(midi_path),
                progress_callback=_midi_progress,
            )

            if progress_callback:
                progress_callback(0.92, self._pt("progress.parsing_miros_midi"))
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            instrument_notes: Dict[int, List[NoteEvent]] = defaultdict(list)
            drum_notes: Dict[int, List[NoteEvent]] = defaultdict(list)

            for instrument in midi.instruments:
                program = int(getattr(instrument, "program", 0) or 0)
                for raw_note in getattr(instrument, "notes", []):
                    note = NoteEvent(
                        pitch=int(raw_note.pitch),
                        start_time=float(raw_note.start),
                        end_time=float(raw_note.end),
                        velocity=int(raw_note.velocity),
                        program=program,
                    )
                    if getattr(instrument, "is_drum", False):
                        drum_notes[note.pitch].append(note)
                    else:
                        instrument_notes[program].append(note)

            for notes in instrument_notes.values():
                notes.sort(key=lambda item: item.start_time)
            for notes in drum_notes.values():
                notes.sort(key=lambda item: item.start_time)

        if progress_callback:
            total_notes = sum(len(notes) for notes in instrument_notes.values()) + sum(
                len(notes) for notes in drum_notes.values()
            )
            progress_callback(
                1.0,
                self._pt("progress.miros_precise_complete", note_count=total_notes),
            )
        return dict(instrument_notes), dict(drum_notes)
