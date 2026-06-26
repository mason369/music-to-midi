import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from urllib.request import Request, urlopen


ROFORMER_SW_DISPLAY_NAME = "BS-RoFormer SW Fixed"
ROFORMER_SW_REGISTRY_NAME = "Roformer Model: BS-Roformer-SW-Fixed"
ROFORMER_SW_MODEL = "BS-Rofo-SW-Fixed.ckpt"
ROFORMER_SW_CONFIG = "BS-Rofo-SW-Fixed.yaml"
ROFORMER_SW_CHECKPOINT_SIZE = 699_412_152
ROFORMER_SW_CHECKPOINT_SHA256 = (
    "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
)

_REQUIRED_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)
ROFORMER_SW_CHECKPOINT_URL = (
    "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_6stem_fixed.ckpt"
)
ROFORMER_SW_CONFIG_URL = (
    "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_6stem_fixed_config.yaml"
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
    try:
        validate_multistem_assets(cache_dir, model_name, config_name)
    except (OSError, RuntimeError):
        return False
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_file_checksum(
    path: Path,
    expected_sha256: str,
    expected_size: int,
    label: str,
) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} 不存在: {path}")
    if not path.is_file():
        raise RuntimeError(f"{label} 不是文件: {path}")

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{label} 大小不匹配: {path}; "
            f"expected={expected_size} actual={actual_size}"
        )

    actual_sha256 = _file_sha256(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"{label} SHA256 不匹配: {path}; "
            f"expected={expected_sha256.lower()} actual={actual_sha256.lower()}"
        )


def _validate_config_file(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"{ROFORMER_SW_DISPLAY_NAME} config 不存在: {path}")
    if not path.is_file():
        raise RuntimeError(f"{ROFORMER_SW_DISPLAY_NAME} config 不是文件: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{ROFORMER_SW_DISPLAY_NAME} config 是空文件: {path}")


def validate_multistem_assets(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_name: str = ROFORMER_SW_MODEL,
    config_name: str = ROFORMER_SW_CONFIG,
) -> Tuple[Path, Path]:
    model_path, config_path = resolve_multistem_model_paths(cache_dir, model_name, config_name)
    validate_file_checksum(
        model_path,
        ROFORMER_SW_CHECKPOINT_SHA256,
        ROFORMER_SW_CHECKPOINT_SIZE,
        "BS-RoFormer SW checkpoint",
    )
    _validate_config_file(config_path)
    return model_path, config_path


def _download_file(url: str, output_path: Path, _description: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.part")
    if temp_path.exists():
        temp_path.unlink()

    request = Request(url=url, headers={"User-Agent": "music-to-midi-downloader/1.0"})
    try:
        with urlopen(request, timeout=60) as response, temp_path.open("wb") as file_handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file_handle.write(chunk)
        temp_path.replace(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _format_size(path: Path) -> str:
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{size_mb:.1f} MB"


def _plain_yaml_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain_yaml_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_yaml_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain_yaml_value(item) for key, item in value.items()}
    return value


def _load_sw_config(config_path: Path) -> dict[str, Any] | None:
    try:
        import yaml
    except ImportError:
        return None

    try:
        text = config_path.read_text(encoding="utf-8")
    except Exception:
        return None

    class _TupleSafeLoader(yaml.SafeLoader):
        pass

    def _construct_python_tuple(loader, node):
        return list(loader.construct_sequence(node))

    _TupleSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)

    try:
        data = yaml.load(text, Loader=_TupleSafeLoader)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    return _plain_yaml_value(data)


def ensure_multistem_config_compatible(config_path: Path) -> bool:
    try:
        import yaml
    except ImportError:
        return False

    data = _load_sw_config(config_path)
    if data is None:
        return False

    modified = False
    if data.get("is_roformer") is not True:
        data["is_roformer"] = True
        modified = True

    model_section = data.get("model", data)
    required_freqs = list(_REQUIRED_FREQS_PER_BANDS)
    required_num_subbands = len(required_freqs)
    if isinstance(model_section, dict):
        if model_section.get("freqs_per_bands") != required_freqs:
            model_section["freqs_per_bands"] = required_freqs
            modified = True
        if model_section.get("num_subbands") != required_num_subbands:
            model_section["num_subbands"] = required_num_subbands
            modified = True
    else:
        data["model"] = {
            "freqs_per_bands": required_freqs,
            "num_subbands": required_num_subbands,
        }
        modified = True

    if "audio" not in data:
        data["audio"] = {
            "chunk_size": 588800,
            "dim_f": 1024,
            "dim_t": 1151,
            "hop_length": 512,
            "n_fft": 2048,
            "num_channels": 2,
            "sample_rate": 44100,
            "min_mean_abs": 0.001,
        }
        modified = True

    inference = data.get("inference")
    if isinstance(inference, dict) and "dim_t" not in inference:
        inference["dim_t"] = 1151
        inference.setdefault("batch_size", 1)
        modified = True

    plain_data = _plain_yaml_value(data)
    if plain_data != data:
        data = plain_data
        modified = True

    try:
        if "!!python/" in config_path.read_text(encoding="utf-8", errors="ignore"):
            modified = True
    except Exception:
        pass

    if modified:
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return modified


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

    if model_path.exists():
        validate_file_checksum(
            model_path,
            ROFORMER_SW_CHECKPOINT_SHA256,
            ROFORMER_SW_CHECKPOINT_SIZE,
            "BS-RoFormer SW checkpoint",
        )
    if config_path.exists():
        _validate_config_file(config_path)

    if model_path.exists() and config_path.exists():
        printer(f"{ROFORMER_SW_DISPLAY_NAME} checkpoint 已存在，跳过下载：{model_path} ({_format_size(model_path)})")
        printer(f"{ROFORMER_SW_DISPLAY_NAME} config 已存在：{config_path} ({_format_size(config_path)})")
        if ensure_multistem_config_compatible(config_path):
            printer("已补全 BS-RoFormer SW Fixed config 兼容字段")
        return model_path, config_path

    if downloader is None:
        downloader = _download_file

    if not model_path.exists() or model_path.stat().st_size <= 0:
        printer(f"正在下载 {ROFORMER_SW_DISPLAY_NAME} checkpoint...")
        downloader(checkpoint_url, model_path, f"{ROFORMER_SW_DISPLAY_NAME} checkpoint")
    if not config_path.exists() or config_path.stat().st_size <= 0:
        printer(f"正在下载 {ROFORMER_SW_DISPLAY_NAME} config...")
        downloader(config_url, config_path, f"{ROFORMER_SW_DISPLAY_NAME} config")

    validate_multistem_assets(cache_path, model_name, config_name)

    ensure_multistem_config_compatible(config_path)
    return model_path, config_path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=f"下载 {ROFORMER_SW_DISPLAY_NAME} 六声部分离模型")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="模型缓存目录（默认: ~/.music-to-midi/models/audio-separator）",
    )
    parser.add_argument("--model-name", default=ROFORMER_SW_MODEL, help="模型文件名")
    parser.add_argument("--config-name", default=ROFORMER_SW_CONFIG, help="配置文件名")
    parser.add_argument("--check-only", action="store_true", help="只校验本地资源，不下载")
    args = parser.parse_args(argv)

    try:
        if args.check_only:
            model_path, config_path = validate_multistem_assets(
                cache_dir=Path(args.cache_dir),
                model_name=args.model_name,
                config_name=args.config_name,
            )
            print(f"BS-RoFormer SW checkpoint 校验通过: {model_path} ({_format_size(model_path)})")
            print(f"BS-RoFormer SW config 校验通过: {config_path} ({_format_size(config_path)})")
        else:
            download_multistem_model(
                cache_dir=Path(args.cache_dir),
                model_name=args.model_name,
                config_name=args.config_name,
            )
        return 0
    except Exception as exc:
        print(f"[错误] BS-RoFormer SW 资源准备失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
