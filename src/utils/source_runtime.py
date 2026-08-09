"""Strict runtime gate for running the desktop application from source."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from src.utils.muscriptor_source_identity import (
    MUSCRIPTOR_PACKAGE_VERSION,
    MUSCRIPTOR_SOURCE_REQUIREMENT,
    validate_muscriptor_runtime_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_SOURCE_PYTHON_MIN = (3, 11)
SUPPORTED_SOURCE_PYTHON_MAX = (3, 12)


def _normalized(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((_normalized(path), _normalized(parent))) == _normalized(parent)
    except ValueError:
        return False


def _expected_venv_paths(project_root: Path, platform_name: str) -> tuple[Path, Path]:
    venv_root = project_root / "venv"
    if platform_name == "nt":
        return venv_root, venv_root / "Scripts" / "python.exe"
    return venv_root, venv_root / "bin" / "python"


def _read_system_site_packages_flag(pyvenv_cfg: Path) -> str | None:
    for line in pyvenv_cfg.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "include-system-site-packages":
            return value.strip().lower()
    return None


def validate_source_runtime_environment(
    project_root: str | Path | None = None,
    *,
    executable: str | Path | None = None,
    prefix: str | Path | None = None,
    base_prefix: str | Path | None = None,
    platform_name: str | None = None,
    frozen: bool | None = None,
    version_info: tuple[int, int] | None = None,
    pointer_bits: int | None = None,
) -> str:
    """Return a source-environment error, or an empty string when exact."""

    if frozen if frozen is not None else bool(getattr(sys, "frozen", False)):
        return ""

    runtime_version = version_info or (sys.version_info.major, sys.version_info.minor)
    if not SUPPORTED_SOURCE_PYTHON_MIN <= runtime_version <= SUPPORTED_SOURCE_PYTHON_MAX:
        return (
            "源码运行仅支持 Python 3.11-3.12: "
            f"actual={runtime_version[0]}.{runtime_version[1]}"
        )
    runtime_pointer_bits = pointer_bits or (64 if sys.maxsize > 2**32 else 32)
    if runtime_pointer_bits != 64:
        return f"源码运行仅支持 64 位 Python: actual={runtime_pointer_bits}-bit"

    root = Path(project_root or PROJECT_ROOT).absolute()
    platform_value = platform_name or os.name
    expected_venv, expected_python = _expected_venv_paths(root, platform_value)
    pyvenv_cfg = expected_venv / "pyvenv.cfg"
    if not pyvenv_cfg.is_file():
        return f"项目虚拟环境配置不存在: {pyvenv_cfg}"

    actual_prefix = Path(prefix or sys.prefix).absolute()
    actual_base_prefix = Path(base_prefix or sys.base_prefix).absolute()
    if _normalized(actual_prefix) == _normalized(actual_base_prefix):
        return (
            "当前解释器是全局 Python，而不是隔离的项目虚拟环境: "
            f"prefix={actual_prefix}, base_prefix={actual_base_prefix}"
        )
    if _normalized(actual_prefix) != _normalized(expected_venv):
        return "当前解释器来自其他虚拟环境: " f"expected={expected_venv}, actual={actual_prefix}"

    try:
        system_site_packages = _read_system_site_packages_flag(pyvenv_cfg)
    except (OSError, UnicodeError) as exc:
        return f"无法读取项目虚拟环境配置 {pyvenv_cfg}: {exc}"
    if system_site_packages != "false":
        return (
            "项目虚拟环境必须隔离全局 site-packages: "
            f"{pyvenv_cfg} 中 include-system-site-packages={system_site_packages!r}"
        )

    actual_executable = Path(executable or sys.executable).absolute()
    if _normalized(actual_executable) != _normalized(expected_python):
        return (
            "源码启动没有使用项目固定解释器: "
            f"expected={expected_python}, actual={actual_executable}"
        )
    return ""


def _installed_muscriptor_root() -> Path:
    spec = importlib.util.find_spec("muscriptor")
    locations = [] if spec is None else list(spec.submodule_search_locations or [])
    if len(locations) != 1:
        raise RuntimeError("无法定位唯一的 MuScriptor 安装目录: " f"locations={locations!r}")
    root = Path(locations[0]).resolve()
    if not root.is_dir():
        raise RuntimeError(f"MuScriptor 安装目录不存在: {root}")
    return root


def validate_source_runtime_identity(
    project_root: str | Path | None = None,
    *,
    package_root: str | Path | None = None,
    executable: str | Path | None = None,
    prefix: str | Path | None = None,
    base_prefix: str | Path | None = None,
    platform_name: str | None = None,
    frozen: bool | None = None,
    version_info: tuple[int, int] | None = None,
    pointer_bits: int | None = None,
) -> str:
    """Validate the project venv and its exact official MuScriptor source."""

    frozen_value = frozen if frozen is not None else bool(getattr(sys, "frozen", False))
    environment_error = validate_source_runtime_environment(
        project_root,
        executable=executable,
        prefix=prefix,
        base_prefix=base_prefix,
        platform_name=platform_name,
        frozen=frozen_value,
        version_info=version_info,
        pointer_bits=pointer_bits,
    )
    if environment_error or frozen_value:
        return environment_error

    root = Path(project_root or PROJECT_ROOT).absolute()
    expected_venv, _ = _expected_venv_paths(root, platform_name or os.name)
    expected_venv = expected_venv.resolve()
    try:
        installed_root = (
            Path(package_root).resolve()
            if package_root is not None
            else _installed_muscriptor_root()
        )
    except Exception as exc:
        return str(exc)

    if not _is_within(installed_root, expected_venv) or "site-packages" not in {
        part.lower() for part in installed_root.parts
    }:
        return (
            "MuScriptor 不是从项目虚拟环境的 site-packages 加载: "
            f"expected_under={expected_venv}, actual={installed_root}"
        )

    identity_error = validate_muscriptor_runtime_identity(installed_root)
    if identity_error:
        return f"{identity_error}. Required package: {MUSCRIPTOR_SOURCE_REQUIREMENT}"
    return ""


def source_runtime_failure_message(
    error: str,
    project_root: str | Path | None = None,
    *,
    platform_name: str | None = None,
) -> str:
    root = Path(project_root or PROJECT_ROOT).absolute()
    platform_value = platform_name or os.name
    _, expected_python = _expected_venv_paths(root, platform_value)
    if platform_value == "nt":
        repair_command = ".\\install.bat"
        launch_command = ".\\run.bat"
    else:
        repair_command = "./install.sh"
        launch_command = "./run.sh"
    return "\n".join(
        (
            f"源码运行时校验失败: {error}",
            f"要求的解释器: {expected_python}",
            f"修复环境: {repair_command}",
            f"启动应用: {launch_command}",
            "不支持使用全局 python -m src.main 启动源码版。",
        )
    )


def require_source_runtime_identity() -> None:
    """Stop before GUI imports when the source runtime is not exact."""

    error = validate_source_runtime_identity()
    if error:
        print(source_runtime_failure_message(error), file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    error = validate_source_runtime_identity()
    if error:
        print(source_runtime_failure_message(error), file=sys.stderr)
        return 2
    print(f"源码解释器校验通过: {sys.executable}")
    print(f"MuScriptor {MUSCRIPTOR_PACKAGE_VERSION} 官方源码身份校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROJECT_ROOT",
    "SUPPORTED_SOURCE_PYTHON_MAX",
    "SUPPORTED_SOURCE_PYTHON_MIN",
    "require_source_runtime_identity",
    "source_runtime_failure_message",
    "validate_source_runtime_environment",
    "validate_source_runtime_identity",
]
