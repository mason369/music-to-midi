"""MIROS multi-instrument transcription wrapper."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.i18n.translator import Translator
from src.models.data_models import Config, NoteEvent
from src.utils.gpu_utils import clear_gpu_memory
from src.utils.runtime_paths import get_miros_source_dir, is_frozen_app

logger = logging.getLogger(__name__)


def _find_module_spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None


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

        entrypoint = repo_dir / "main.py"
        transcribe_script = repo_dir / "transcribe.py"
        if not entrypoint.exists() or not transcribe_script.exists():
            return (
                "MIROS 不可用：ai4m-miros 目录不完整。\n\n"
                f"缺少文件：{entrypoint if not entrypoint.exists() else transcribe_script}"
            )

        missing_modules = cls._missing_modules()
        if missing_modules:
            return (
                "MIROS 不可用：缺少运行依赖。\n\n"
                f"缺少模块：{', '.join(missing_modules)}\n"
                "请先安装 requirements.txt 中的依赖，并补充 ai4m-miros 要求的环境。"
            )

        pretrained_path = repo_dir / cls.PRETRAINED_REL_PATH
        checkpoint_path = repo_dir / cls.CHECKPOINT_REL_PATH
        if not pretrained_path.is_file() or not checkpoint_path.is_file():
            missing = [
                str(path)
                for path in (pretrained_path, checkpoint_path)
                if not path.is_file()
            ]
            return (
                "MIROS 不可用：模型权重缺失。\n\n"
                f"缺少文件：{', '.join(missing)}\n"
                "请重新运行 install.ps1 准备 ai4m-miros 权重。"
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
        return (
            (repo_dir / cls.PRETRAINED_REL_PATH).is_file()
            and (repo_dir / cls.CHECKPOINT_REL_PATH).is_file()
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

        self.reset_cancel()
        self._check_cancelled()

        input_path = Path(audio_path).resolve()
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        status_path = (
            out_path.parent / f".{out_path.stem}.miros_status.json"
            if is_frozen_app()
            else None
        )
        if status_path is not None and status_path.exists():
            status_path.unlink()

        command = self._build_command(entrypoint, input_path, out_path, status_path)

        if progress_callback:
            progress_callback(0.05, self._pt("progress.preparing_miros"))

        logger.info("Running MIROS transcription: %s", " ".join(command))
        process = subprocess.Popen(
            command,
            cwd=str(repo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(os.environ),
        )
        with self._process_lock:
            self._process = process

        try:
            if progress_callback:
                progress_callback(0.50, self._pt("progress.running_miros"))
            stdout, stderr = process.communicate()
        finally:
            with self._process_lock:
                self._process = None

        self._check_cancelled()

        if process.returncode != 0:
            detail = self._format_process_failure(
                process.returncode,
                stdout or "",
                stderr or "",
                status_path,
            )
            raise RuntimeError(f"MIROS 转写失败:\n{detail}")
        if not out_path.exists():
            raise RuntimeError(
                self._format_missing_output_error(
                    out_path,
                    stdout or "",
                    stderr or "",
                    status_path,
                )
            )

        if progress_callback:
            progress_callback(1.0, self._pt("progress.miros_complete"))
        return str(out_path)

    def transcribe_precise(
        self,
        audio_path: str,
        quality: str = "best",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[Dict[int, List[NoteEvent]], Dict[int, List[NoteEvent]]]:
        import pretty_midi

        with tempfile.TemporaryDirectory(prefix="miros_") as tmp_dir:
            midi_path = Path(tmp_dir) / f"{Path(audio_path).stem}.mid"
            self.transcribe_to_midi(
                audio_path=audio_path,
                output_path=str(midi_path),
                progress_callback=progress_callback,
            )

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

        return dict(instrument_notes), dict(drum_notes)
