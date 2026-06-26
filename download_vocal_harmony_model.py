from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Callable, Optional


CHORUS_PRESET = "karaoke"
CHORUS_MODEL = f"ensemble:{CHORUS_PRESET}"
CHORUS_MODELS = (
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "mel_band_roformer_karaoke_becruily.ckpt",
)
DEFAULT_CACHE_DIR = Path.home() / ".music-to-midi" / "models" / "audio-separator"


def _resolve_cache_dir(cache_dir: Optional[Path]) -> Path:
    return Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR


def resolve_chorus_model_path(
    cache_dir: Optional[Path] = None,
    model_name: str = CHORUS_MODELS[0],
) -> Path:
    return _resolve_cache_dir(cache_dir) / model_name


def resolve_chorus_model_paths(
    cache_dir: Optional[Path] = None,
    model_names: tuple[str, ...] = CHORUS_MODELS,
) -> tuple[Path, ...]:
    root = _resolve_cache_dir(cache_dir)
    return tuple(root / model_name for model_name in model_names)


def _find_downloaded_checkpoint(cache_dir: Path, model_name: str) -> Optional[Path]:
    direct = cache_dir / model_name
    if direct.is_file():
        return direct

    for path in cache_dir.rglob(model_name):
        if path.is_file():
            return path
    return None


def _find_all_required(cache_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for model_name in CHORUS_MODELS:
        path = _find_downloaded_checkpoint(cache_dir, model_name)
        if path and path.stat().st_size > 0:
            found[model_name] = path
    return found


def is_chorus_model_available(cache_dir: Optional[Path] = None) -> bool:
    cache_dir = _resolve_cache_dir(cache_dir)
    return len(_find_all_required(cache_dir)) == len(CHORUS_MODELS)


def _separator_supports_ensemble(separator_cls) -> bool:
    try:
        signature = inspect.signature(separator_cls.__init__)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or parameter.name == "ensemble_preset"
        for parameter in signature.parameters.values()
    )


def _is_ensemble(model_name: str) -> bool:
    return str(model_name).lower().startswith("ensemble:")


def download_chorus_model(
    cache_dir: Optional[Path] = None,
    model_name: str = CHORUS_MODEL,
    separator_cls=None,
    printer: Callable[[str], None] = print,
) -> Path:
    cache_dir = _resolve_cache_dir(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    found = _find_all_required(cache_dir)
    if len(found) == len(CHORUS_MODELS):
        printer("RoFormer karaoke ensemble 模型已存在，跳过下载。")
        return found[CHORUS_MODELS[0]]

    if separator_cls is None:
        try:
            from audio_separator.separator import Separator
        except Exception as exc:
            raise RuntimeError(
                "未找到 audio-separator，无法下载 RoFormer karaoke ensemble 模型。"
            ) from exc
        separator_cls = Separator

    if _is_ensemble(model_name) and not _separator_supports_ensemble(separator_cls):
        raise RuntimeError(
            "当前 audio-separator 不支持 ensemble_preset；"
            "请安装 audio-separator==0.44.1 或更高的 0.44.x 版本。"
        )

    printer(f"正在下载 RoFormer karaoke ensemble: {model_name}")
    printer("模型组：")
    for required_model in CHORUS_MODELS:
        printer(f"  - {required_model}")
    printer(f"目标缓存目录：{cache_dir}")

    separator_kwargs = {
        "output_dir": str(cache_dir),
        "model_file_dir": str(cache_dir),
        "output_format": "WAV",
    }
    if _is_ensemble(model_name):
        separator_kwargs["ensemble_preset"] = model_name.split(":", 1)[1]

    separator = separator_cls(**separator_kwargs)
    if _is_ensemble(model_name):
        separator.load_model()
    else:
        separator.load_model(model_name)

    found = _find_all_required(cache_dir)
    missing = [model for model in CHORUS_MODELS if model not in found]
    if missing:
        raise RuntimeError(
            f"已调用 audio-separator，但缺少 RoFormer karaoke 必需模型: {missing}"
        )

    for model, path in found.items():
        printer(f"模型就绪：{model} ({path.stat().st_size / (1024 * 1024):.1f} MB)")
    return found[CHORUS_MODELS[0]]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="下载 RoFormer karaoke ensemble 主唱/和声分离模型")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="模型缓存目录（默认: ~/.music-to-midi/models/audio-separator）",
    )
    parser.add_argument("--model-name", default=CHORUS_MODEL, help="模型名或 ensemble preset")
    args = parser.parse_args(argv)

    try:
        download_chorus_model(
            cache_dir=Path(args.cache_dir),
            model_name=args.model_name,
        )
        return 0
    except Exception as exc:
        print(f"[错误] RoFormer karaoke ensemble 模型下载失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
