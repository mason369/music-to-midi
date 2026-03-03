import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple
from urllib.request import Request, urlopen


ROFORMER_SW_MODEL = "BS-Rofo-SW-Fixed.ckpt"
ROFORMER_SW_CONFIG = "config_bs_roformer_sw.yaml"
ROFORMER_SW_CHECKPOINT_URL = (
    "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt"
)
ROFORMER_SW_CONFIG_URL = (
    "https://raw.githubusercontent.com/openmirlab/bs-roformer-infer/main/src/bs_roformer/configs/config_bs_roformer_sw.yaml"
)

DEFAULT_CACHE_DIR = Path.home() / ".music-to-midi" / "models" / "audio-separator"


def resolve_multistem_model_paths(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_name: str = ROFORMER_SW_MODEL,
    config_name: str = ROFORMER_SW_CONFIG,
) -> Tuple[Path, Path]:
    cache_path = Path(cache_dir)
    return cache_path / model_name, cache_path / config_name


def is_multistem_model_available(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_name: str = ROFORMER_SW_MODEL,
    config_name: str = ROFORMER_SW_CONFIG,
) -> bool:
    model_path, config_path = resolve_multistem_model_paths(cache_dir, model_name, config_name)
    return (
        model_path.exists()
        and model_path.stat().st_size > 0
        and config_path.exists()
        and config_path.stat().st_size > 0
    )


def _download_file(url: str, output_path: Path, _description: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url=url, headers={"User-Agent": "music-to-midi-downloader/1.0"})
    with urlopen(request, timeout=60) as response, output_path.open("wb") as file_handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file_handle.write(chunk)


def _format_size(path: Path) -> str:
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{size_mb:.1f} MB"


def download_multistem_model(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_name: str = ROFORMER_SW_MODEL,
    config_name: str = ROFORMER_SW_CONFIG,
    checkpoint_url: str = ROFORMER_SW_CHECKPOINT_URL,
    config_url: str = ROFORMER_SW_CONFIG_URL,
    downloader: Optional[Callable[[str, Path, str], None]] = None,
    printer: Callable[[str], None] = print,
) -> Tuple[Path, Path]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    model_path, config_path = resolve_multistem_model_paths(cache_path, model_name, config_name)

    if is_multistem_model_available(cache_path, model_name, config_name):
        printer(f"BS-RoFormer SW 资源已存在，跳过下载：{model_path} ({_format_size(model_path)})")
        printer(f"BS-RoFormer SW 配置已存在：{config_path} ({_format_size(config_path)})")
        return model_path, config_path

    if downloader is None:
        downloader = _download_file

    if not model_path.exists() or model_path.stat().st_size <= 0:
        printer("正在下载 BS-RoFormer SW checkpoint...")
        printer(f"  URL: {checkpoint_url}")
        printer(f"  保存到: {model_path}")
        downloader(checkpoint_url, model_path, "BS Roformer SW checkpoint")
        printer(f"  checkpoint 下载完成: {_format_size(model_path)}")
    else:
        printer(f"检测到已有 checkpoint，跳过: {model_path} ({_format_size(model_path)})")

    if not config_path.exists() or config_path.stat().st_size <= 0:
        printer("正在下载 BS-RoFormer SW config...")
        printer(f"  URL: {config_url}")
        printer(f"  保存到: {config_path}")
        downloader(config_url, config_path, "BS Roformer SW config")
        printer(f"  config 下载完成: {_format_size(config_path)}")
    else:
        printer(f"检测到已有 config，跳过: {config_path} ({_format_size(config_path)})")

    if not is_multistem_model_available(cache_path, model_name, config_name):
        raise RuntimeError(
            f"六声部分离资源不完整: model_exists={model_path.exists()} config_exists={config_path.exists()}"
        )

    printer(f"BS-RoFormer SW 模型就绪: {model_path} ({_format_size(model_path)})")
    printer(f"BS-RoFormer SW 配置就绪: {config_path} ({_format_size(config_path)})")
    return model_path, config_path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="下载 BS-RoFormer SW 六声部分离模型")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="模型缓存目录（默认: ~/.music-to-midi/models/audio-separator）",
    )
    parser.add_argument(
        "--model-name",
        default=ROFORMER_SW_MODEL,
        help="模型文件名",
    )
    parser.add_argument(
        "--config-name",
        default=ROFORMER_SW_CONFIG,
        help="配置文件名",
    )
    args = parser.parse_args(argv)

    try:
        download_multistem_model(
            cache_dir=Path(args.cache_dir),
            model_name=args.model_name,
            config_name=args.config_name,
        )
        return 0
    except Exception as exc:
        print(f"[错误] BS-RoFormer SW 资源下载失败: {exc}")
        print("可稍后手动执行：venv\\Scripts\\python.exe download_multistem_model.py")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
