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
ROFORMER_SW_OFFICIAL_CONFIG = "BS-Rofo-SW-Fixed.official.yaml"
ROFORMER_SW_SOURCE_REVISION = "370198fbb6997e3f5774778254698794e7b1267d"
ROFORMER_SW_CHECKPOINT_SIZE = 699_412_152
ROFORMER_SW_CHECKPOINT_SHA256 = (
    "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
)
ROFORMER_SW_OFFICIAL_CONFIG_SIZE = 3_530
ROFORMER_SW_OFFICIAL_CONFIG_SHA256 = (
    "4678db9430a87ee33e7fad199166928c9adcd322e2df1a812b4bf03726e2a48b"
)
ROFORMER_SW_COMPATIBLE_CONFIG_SIZE = 3_522
ROFORMER_SW_COMPATIBLE_CONFIG_SHA256 = (
    "e7dc288d2456a9a186c451ca551025db408a1cbf3fff2b98c8eb0077129324c3"
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
    "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/"
    f"{ROFORMER_SW_SOURCE_REVISION}/bs_roformer/bs_6stem_fixed.ckpt"
)
ROFORMER_SW_CONFIG_URL = (
    "https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/"
    f"{ROFORMER_SW_SOURCE_REVISION}/bs_roformer/bs_6stem_fixed_config.yaml"
)

DEFAULT_CACHE_DIR = Path.home() / ".music-to-midi" / "models" / "audio-separator"


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def resolve_multistem_model_paths(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_name: str = ROFORMER_SW_MODEL,
    config_name: str = ROFORMER_SW_CONFIG,
) -> Tuple[Path, Path]:
    cache_path = Path(cache_dir)
    return cache_path / model_name, cache_path / config_name


def resolve_multistem_source_config_path(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    config_name: str = ROFORMER_SW_CONFIG,
) -> Path:
    cache_path = Path(cache_dir)
    if config_name == ROFORMER_SW_CONFIG:
        source_name = ROFORMER_SW_OFFICIAL_CONFIG
    else:
        config_path = Path(config_name)
        source_name = f"{config_path.stem}.official{config_path.suffix}"
    return cache_path / source_name


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
    validate_file_checksum(
        path,
        ROFORMER_SW_COMPATIBLE_CONFIG_SHA256,
        ROFORMER_SW_COMPATIBLE_CONFIG_SIZE,
        f"{ROFORMER_SW_DISPLAY_NAME} compatible config",
    )
    data = _load_sw_config(path)
    if data is None:
        raise RuntimeError(f"{ROFORMER_SW_DISPLAY_NAME} compatible config 无法解析: {path}")

    model = data.get("model")
    inference = data.get("inference")
    if data.get("is_roformer") is not True:
        raise RuntimeError(f"{ROFORMER_SW_DISPLAY_NAME} config 缺少 is_roformer=true: {path}")
    if not isinstance(model, dict):
        raise RuntimeError(f"{ROFORMER_SW_DISPLAY_NAME} config 缺少 model 节: {path}")
    if model.get("freqs_per_bands") != list(_REQUIRED_FREQS_PER_BANDS):
        raise RuntimeError(f"{ROFORMER_SW_DISPLAY_NAME} config 频带定义不匹配: {path}")
    if model.get("num_subbands") != len(_REQUIRED_FREQS_PER_BANDS):
        raise RuntimeError(f"{ROFORMER_SW_DISPLAY_NAME} config num_subbands 不匹配: {path}")
    if not isinstance(inference, dict) or inference.get("dim_t") != 1101:
        raise RuntimeError(f"{ROFORMER_SW_DISPLAY_NAME} config inference.dim_t 必须为 1101: {path}")


def _validate_official_config_file(path: Path) -> None:
    validate_file_checksum(
        path,
        ROFORMER_SW_OFFICIAL_CONFIG_SHA256,
        ROFORMER_SW_OFFICIAL_CONFIG_SIZE,
        f"{ROFORMER_SW_DISPLAY_NAME} official config",
    )


def validate_multistem_assets(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_name: str = ROFORMER_SW_MODEL,
    config_name: str = ROFORMER_SW_CONFIG,
) -> Tuple[Path, Path]:
    model_path, config_path = resolve_multistem_model_paths(cache_dir, model_name, config_name)
    source_config_path = resolve_multistem_source_config_path(cache_dir, config_name)
    validate_file_checksum(
        model_path,
        ROFORMER_SW_CHECKPOINT_SHA256,
        ROFORMER_SW_CHECKPOINT_SIZE,
        "BS-RoFormer SW checkpoint",
    )
    _validate_official_config_file(source_config_path)
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


def _download_verified_file(
    url: str,
    output_path: Path,
    description: str,
    expected_sha256: str,
    expected_size: int,
    downloader: Callable[[str, Path, str], None],
) -> None:
    download_path = output_path.with_name(f"{output_path.name}.download")
    if download_path.exists():
        download_path.unlink()
    try:
        downloader(url, download_path, description)
        validate_file_checksum(
            download_path,
            expected_sha256,
            expected_size,
            description,
        )
        download_path.replace(output_path)
    except Exception:
        if download_path.exists():
            download_path.unlink()
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


def _build_compatible_config_bytes(source_config_path: Path) -> bytes:
    """Create the one canonical runtime config from the pinned official YAML."""
    _validate_official_config_file(source_config_path)
    try:
        text = source_config_path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"{ROFORMER_SW_DISPLAY_NAME} official config 不是有效 UTF-8: {source_config_path}"
        ) from exc

    replacements = (
        ("freqs_per_bands: !!python/tuple", "freqs_per_bands:"),
        (
            "multi_stft_resolutions_window_sizes: !!python/tuple",
            "multi_stft_resolutions_window_sizes:",
        ),
        ("mixup_probs: !!python/tuple", "mixup_probs:"),
        (
            "    - 129\n  dim_head: 64\n",
            "    - 129\n"
            f"  num_subbands: {len(_REQUIRED_FREQS_PER_BANDS)}\n"
            "  dim_head: 64\n",
        ),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(
                f"{ROFORMER_SW_DISPLAY_NAME} official config 结构与固定转换规则不匹配: {old!r}"
            )
        text = text.replace(old, new, 1)

    if not text.endswith("\n"):
        raise RuntimeError(f"{ROFORMER_SW_DISPLAY_NAME} official config 必须以 LF 结尾")
    text += "is_roformer: true\n"
    compatible_bytes = text.encode("utf-8")

    actual_size = len(compatible_bytes)
    actual_sha256 = hashlib.sha256(compatible_bytes).hexdigest()
    if (
        actual_size != ROFORMER_SW_COMPATIBLE_CONFIG_SIZE
        or actual_sha256 != ROFORMER_SW_COMPATIBLE_CONFIG_SHA256
    ):
        raise RuntimeError(
            f"{ROFORMER_SW_DISPLAY_NAME} compatible config 生成结果身份不匹配; "
            f"expected_size={ROFORMER_SW_COMPATIBLE_CONFIG_SIZE} actual_size={actual_size} "
            f"expected_sha256={ROFORMER_SW_COMPATIBLE_CONFIG_SHA256} "
            f"actual_sha256={actual_sha256}"
        )
    return compatible_bytes


def ensure_multistem_config_compatible(
    config_path: Path,
    source_config_path: Optional[Path] = None,
) -> bool:
    """Ensure the derived runtime YAML exactly matches the pinned canonical bytes."""
    config_path = Path(config_path)
    if source_config_path is None:
        source_config_path = resolve_multistem_source_config_path(
            config_path.parent,
            config_path.name,
        )
    else:
        source_config_path = Path(source_config_path)

    try:
        _validate_config_file(config_path)
        return False
    except RuntimeError:
        pass

    compatible_bytes = _build_compatible_config_bytes(source_config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config_path.with_name(f"{config_path.name}.part")
    if temp_path.exists():
        temp_path.unlink()
    try:
        temp_path.write_bytes(compatible_bytes)
        _validate_config_file(temp_path)
        temp_path.replace(config_path)
        _validate_config_file(config_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return True


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
    source_config_path = resolve_multistem_source_config_path(cache_path, config_name)

    if model_path.exists():
        validate_file_checksum(
            model_path,
            ROFORMER_SW_CHECKPOINT_SHA256,
            ROFORMER_SW_CHECKPOINT_SIZE,
            "BS-RoFormer SW checkpoint",
        )
    if source_config_path.exists():
        _validate_official_config_file(source_config_path)

    if downloader is None:
        downloader = _download_file

    if not model_path.exists():
        printer(f"正在下载 {ROFORMER_SW_DISPLAY_NAME} checkpoint...")
        _download_verified_file(
            checkpoint_url,
            model_path,
            f"{ROFORMER_SW_DISPLAY_NAME} checkpoint",
            ROFORMER_SW_CHECKPOINT_SHA256,
            ROFORMER_SW_CHECKPOINT_SIZE,
            downloader,
        )
    else:
        printer(
            f"{ROFORMER_SW_DISPLAY_NAME} checkpoint 已严格校验："
            f"{model_path} ({_format_size(model_path)})"
        )

    if not source_config_path.exists():
        printer(f"正在下载 {ROFORMER_SW_DISPLAY_NAME} official config...")
        _download_verified_file(
            config_url,
            source_config_path,
            f"{ROFORMER_SW_DISPLAY_NAME} official config",
            ROFORMER_SW_OFFICIAL_CONFIG_SHA256,
            ROFORMER_SW_OFFICIAL_CONFIG_SIZE,
            downloader,
        )
    else:
        printer(
            f"{ROFORMER_SW_DISPLAY_NAME} official config 已严格校验："
            f"{source_config_path} ({_format_size(source_config_path)})"
        )

    try:
        _validate_config_file(config_path)
        regenerated = False
    except RuntimeError as exc:
        printer(f"正在重建固定 LF compatible config；原文件不可用：{exc}")
        regenerated = ensure_multistem_config_compatible(config_path, source_config_path)
    if regenerated:
        printer(
            f"已生成固定身份的 {ROFORMER_SW_DISPLAY_NAME} compatible config："
            f"{config_path}"
        )

    validate_multistem_assets(cache_path, model_name, config_name)
    return model_path, config_path


def main(argv: list[str]) -> int:
    configure_console_encoding()

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
