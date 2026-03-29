"""Aria-AMT piano transcription wrapper."""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from src.utils.runtime_paths import get_aria_amt_dir

logger = logging.getLogger(__name__)

ARIA_AMT_CHECKPOINT_NAME = "piano-medium-double-1.0.safetensors"
ARIA_AMT_CHECKPOINT_URL = (
    "https://huggingface.co/datasets/loubb/aria-midi/resolve/main/"
    "piano-medium-double-1.0.safetensors?download=true"
)
ARIA_AMT_CACHE_DIR = Path.home() / ".cache" / "music_ai_models" / "aria_amt"


class AriaAmtTranscriber:
    def __init__(self, checkpoint_path: Optional[Path] = None):
        if checkpoint_path is None:
            checkpoint_path = self.default_checkpoint_path()
        self.checkpoint_path = Path(checkpoint_path)

    @staticmethod
    def default_checkpoint_path() -> Path:
        return get_aria_amt_dir() / ARIA_AMT_CHECKPOINT_NAME

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("amt.run") is not None

    def is_model_available(self) -> bool:
        return self.checkpoint_path.exists() and self.checkpoint_path.stat().st_size > 0

    @staticmethod
    def _guess_output_midi(save_dir: Path, audio_path: Path) -> Optional[Path]:
        direct = save_dir / f"{audio_path.stem}.mid"
        if direct.exists():
            return direct
        midis = sorted(save_dir.glob("*.mid"))
        return midis[0] if midis else None

    def transcribe(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        input_path = Path(audio_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            raise RuntimeError(
                "Aria-AMT 未安装。请执行: "
                "python -m pip install git+https://github.com/EleutherAI/aria-amt.git"
            )
        if not self.is_model_available():
            raise RuntimeError(
                "Aria-AMT 模型权重缺失。请执行: "
                "python download_aria_amt_model.py"
            )

        temp_dir = out_path.parent / ".aria_amt_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            "amt.run",
            "transcribe",
            "--load_path",
            str(input_path),
            "--save_dir",
            str(temp_dir),
            str(self.checkpoint_path),
        ]

        if progress_callback:
            progress_callback(0.05, "正在加载 Aria-AMT...")

        logger.info("Running Aria-AMT transcription: %s", " ".join(command))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Aria-AMT 转写失败:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )

        midi_path = self._guess_output_midi(temp_dir, input_path)
        if midi_path is None or not midi_path.exists():
            raise RuntimeError("Aria-AMT 未生成 MIDI 输出")

        shutil.move(str(midi_path), str(out_path))
        shutil.rmtree(temp_dir, ignore_errors=True)

        if progress_callback:
            progress_callback(1.0, "Aria-AMT 转写完成")

        logger.info("Aria-AMT output: %s", out_path)
        return str(out_path)
