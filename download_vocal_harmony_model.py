import argparse
import sys
from pathlib import Path
from typing import Callable, Optional


CHORUS_MODEL = "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt"
DEFAULT_CACHE_DIR = Path.home() / ".music-to-midi" / "models" / "audio-separator"


def resolve_chorus_model_path(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_name: str = CHORUS_MODEL,
) -> Path:
    return Path(cache_dir) / model_name


def _find_downloaded_checkpoint(cache_dir: Path, model_name: str) -> Optional[Path]:
    direct = resolve_chorus_model_path(cache_dir, model_name)
    if direct.exists():
        return direct

    for path in cache_dir.rglob(model_name):
        if path.is_file():
            return path
    return None


def is_chorus_model_available(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_name: str = CHORUS_MODEL,
) -> bool:
    cache_dir = Path(cache_dir)
    found = _find_downloaded_checkpoint(cache_dir, model_name)
    return bool(found and found.stat().st_size > 0)


def download_chorus_model(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_name: str = CHORUS_MODEL,
    separator_cls=None,
    printer: Callable[[str], None] = print,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = _find_downloaded_checkpoint(cache_dir, model_name)
    if existing:
        size_mb = existing.stat().st_size / (1024 * 1024)
        printer(f"Chorus 模型已存在，跳过下载：{existing} ({size_mb:.1f} MB)")
        return existing

    if separator_cls is None:
        try:
            from audio_separator.separator import Separator
        except Exception as exc:
            raise RuntimeError(
                "未找到 audio-separator，无法下载 Chorus 模型。"
            ) from exc
        separator_cls = Separator

    printer("正在下载 Chorus BS-RoFormer 主唱/和声分离模型...")
    printer("架构：Band-Split RoFormer (lead vocals/harmony vocals)")
    printer(f"目标缓存目录：{cache_dir}")
    printer("提示：首次下载文件较大，请耐心等待。")

    separator = separator_cls(
        output_dir=str(cache_dir),
        model_file_dir=str(cache_dir),
        output_format="WAV",
    )
    separator.load_model(model_name)

    downloaded = _find_downloaded_checkpoint(cache_dir, model_name)
    if downloaded is None:
        raise RuntimeError(
            f"已调用 audio-separator，但未在 {cache_dir} 找到 {model_name}"
        )

    size_mb = downloaded.stat().st_size / (1024 * 1024)
    printer("下载成功！")
    printer(f"模型文件：{downloaded} ({size_mb:.1f} MB)")
    return downloaded


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="下载 Chorus BS-RoFormer 主唱/和声分离模型")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="模型缓存目录（默认: ~/.music-to-midi/models/audio-separator）",
    )
    parser.add_argument(
        "--model-name",
        default=CHORUS_MODEL,
        help="模型文件名",
    )
    args = parser.parse_args(argv)

    try:
        download_chorus_model(
            cache_dir=Path(args.cache_dir),
            model_name=args.model_name,
        )
        return 0
    except Exception as exc:
        print(f"[错误] Chorus 模型下载失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
