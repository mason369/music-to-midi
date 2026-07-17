import argparse
import sys
from pathlib import Path
from typing import Callable, Optional
from urllib.request import Request, urlopen

from src.core.aria_amt_transcriber import (
    ARIA_AMT_CACHE_DIR,
    ARIA_AMT_CHECKPOINT_NAME,
    ARIA_AMT_CHECKPOINT_SHA256,
    ARIA_AMT_CHECKPOINT_SIZE,
    ARIA_AMT_CHECKPOINT_URL,
    validate_aria_amt_checkpoint,
)


def resolve_aria_model_path(
    cache_dir: Path = ARIA_AMT_CACHE_DIR,
    model_name: str = ARIA_AMT_CHECKPOINT_NAME,
) -> Path:
    return Path(cache_dir) / model_name


def is_aria_model_available(
    cache_dir: Path = ARIA_AMT_CACHE_DIR,
    model_name: str = ARIA_AMT_CHECKPOINT_NAME,
) -> bool:
    path = resolve_aria_model_path(cache_dir, model_name)
    try:
        validate_aria_amt_checkpoint(path)
        return True
    except (OSError, RuntimeError):
        return False


def _download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url=url, headers={"User-Agent": "music-to-midi-downloader/1.0"})
    with urlopen(request, timeout=60) as response, output_path.open("wb") as file_handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file_handle.write(chunk)


def download_aria_model(
    cache_dir: Path = ARIA_AMT_CACHE_DIR,
    model_name: str = ARIA_AMT_CHECKPOINT_NAME,
    model_url: str = ARIA_AMT_CHECKPOINT_URL,
    downloader: Optional[Callable[[str, Path], None]] = None,
    printer: Callable[[str], None] = print,
) -> Path:
    model_path = resolve_aria_model_path(cache_dir, model_name)
    if is_aria_model_available(cache_dir, model_name):
        size_mb = model_path.stat().st_size / (1024 * 1024)
        printer(f"Aria-AMT 模型已存在，跳过下载：{model_path} ({size_mb:.1f} MB)")
        return model_path

    if model_path.exists():
        printer(
            "现有 Aria-AMT checkpoint 身份校验失败，将下载固定版本并在校验后替换：" f"{model_path}"
        )

    if downloader is None:
        downloader = _download_file

    printer("正在下载 Aria-AMT 钢琴转写模型...")
    printer("模型: piano-medium-double-1.0.safetensors")
    printer(f"URL: {model_url}")
    printer(f"保存到: {model_path}")
    partial_path = model_path.with_suffix(model_path.suffix + ".part")
    if partial_path.exists():
        partial_path.unlink()

    try:
        downloader(model_url, partial_path)
        validate_aria_amt_checkpoint(partial_path)
        partial_path.replace(model_path)
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise

    size_mb = model_path.stat().st_size / (1024 * 1024)
    printer("下载成功！")
    printer(f"模型文件：{model_path} ({size_mb:.1f} MB)")
    printer("身份校验：" f"{ARIA_AMT_CHECKPOINT_SIZE} bytes, SHA-256 {ARIA_AMT_CHECKPOINT_SHA256}")
    return model_path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="下载 Aria-AMT 钢琴转写模型")
    parser.add_argument(
        "--cache-dir",
        default=str(ARIA_AMT_CACHE_DIR),
        help="模型缓存目录（默认: ~/.cache/music_ai_models/aria_amt）",
    )
    parser.add_argument(
        "--model-name",
        default=ARIA_AMT_CHECKPOINT_NAME,
        help="模型文件名",
    )
    args = parser.parse_args(argv)

    try:
        download_aria_model(
            cache_dir=Path(args.cache_dir),
            model_name=args.model_name,
        )
        return 0
    except Exception as exc:
        print(f"[错误] Aria-AMT 模型下载失败: {exc}")
        print("可稍后手动执行：venv\\Scripts\\python.exe download_aria_amt_model.py")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
