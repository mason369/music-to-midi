"""Split vocal stem into lead/harmony proxy stems using a public BS-RoFormer model."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

CHORUS_MODEL = "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt"


class VocalHarmonySeparator:
    """Approximate lead/harmony split by running male/female vocal separation."""

    @staticmethod
    def _get_model_cache_dir() -> str:
        cache_dir = Path.home() / ".music-to-midi" / "models" / "audio-separator"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)

    @staticmethod
    def is_available() -> bool:
        try:
            from audio_separator.separator import Separator  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def is_model_available() -> bool:
        """检查 chorus 模型文件是否已下载到缓存目录。"""
        cache_dir = Path.home() / ".music-to-midi" / "models" / "audio-separator"
        model_path = cache_dir / CHORUS_MODEL
        if model_path.exists() and model_path.stat().st_size > 0:
            return True
        # 递归搜索
        for path in cache_dir.rglob(CHORUS_MODEL):
            if path.is_file() and path.stat().st_size > 0:
                return True
        return False

    @staticmethod
    def _score_rms(path: Path) -> float:
        try:
            import soundfile as sf

            audio, _sr = sf.read(str(path), always_2d=False)
            arr = np.asarray(audio, dtype=np.float32)
            if arr.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(arr * arr)))
        except Exception:
            return 0.0

    @staticmethod
    def _resolve_outputs(output_dir: Path, output_files) -> list[Path]:
        paths: list[Path] = []
        for item in output_files:
            path = Path(str(item))
            if not path.is_absolute():
                path = output_dir / path
            if path.exists() and path.is_file() and path.suffix.lower() == ".wav":
                paths.append(path)
        if paths:
            return paths
        return [p for p in output_dir.glob("*.wav") if p.is_file()]

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, str]:
        from audio_separator.separator import Separator

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem_name = Path(audio_path).stem

        if progress_callback:
            progress_callback(0.0, "正在加载主唱/和声分离模型...")

        separator = Separator(
            output_dir=str(out_dir),
            model_file_dir=self._get_model_cache_dir(),
            output_format="WAV",
        )
        separator.load_model(CHORUS_MODEL)

        if progress_callback:
            progress_callback(0.25, "正在分离主唱/和声...")

        output_files = separator.separate(audio_path)
        paths = self._resolve_outputs(out_dir, output_files)
        if len(paths) < 2:
            raise RuntimeError("主唱/和声分离输出不足，至少需要 2 个 wav 文件")

        male_path: Optional[Path] = None
        female_path: Optional[Path] = None
        for path in paths:
            lowered = path.name.lower()
            if "male" in lowered and male_path is None:
                male_path = path
            if "female" in lowered and female_path is None:
                female_path = path

        if male_path is None or female_path is None:
            # Fallback: use first two outputs deterministically.
            chosen = sorted(paths)[:2]
            male_path = chosen[0]
            female_path = chosen[1]

        # Proxy mapping: louder stem -> lead, quieter stem -> harmony.
        male_rms = self._score_rms(male_path)
        female_rms = self._score_rms(female_path)
        if female_rms >= male_rms:
            lead_src, harmony_src = female_path, male_path
        else:
            lead_src, harmony_src = male_path, female_path

        lead_path = out_dir / f"{stem_name}_lead_vocals.wav"
        harmony_path = out_dir / f"{stem_name}_harmony_vocals.wav"

        if lead_src.resolve() != lead_path.resolve():
            os.replace(lead_src, lead_path)
        if harmony_src.resolve() != harmony_path.resolve():
            os.replace(harmony_src, harmony_path)

        if progress_callback:
            progress_callback(1.0, "主唱/和声分离完成")

        logger.info(
            "Vocal harmony split complete: lead=%s harmony=%s",
            lead_path,
            harmony_path,
        )
        return {
            "lead_vocals": str(lead_path),
            "harmony_vocals": str(harmony_path),
        }
