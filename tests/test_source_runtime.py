import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils import source_runtime


@pytest.fixture(autouse=True)
def _isolate_accelerator_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep default-venv cases deterministic under CUDA and XPU test runners."""

    monkeypatch.delenv("MUSIC_TO_MIDI_ACCELERATOR", raising=False)


def _runtime_layout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project_root = tmp_path / "project"
    venv_root = project_root / "venv"
    python = venv_root / "Scripts" / "python.exe"
    package_root = venv_root / "Lib" / "site-packages" / "muscriptor"
    python.parent.mkdir(parents=True)
    python.touch()
    package_root.mkdir(parents=True)
    (venv_root / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n", encoding="utf-8"
    )
    return project_root, venv_root, python, package_root


def _validate(
    project_root: Path,
    venv_root: Path,
    python: Path,
    default_package_root: Path,
    **overrides: object,
) -> str:
    arguments: dict[str, object] = {
        "package_root": default_package_root,
        "executable": python,
        "prefix": venv_root,
        "base_prefix": project_root / "base-python",
        "platform_name": "nt",
        "frozen": False,
    }
    arguments.update(overrides)
    return source_runtime.validate_source_runtime_identity(project_root, **arguments)


def test_accepts_isolated_project_venv_and_exact_muscriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, venv_root, python, package_root = _runtime_layout(tmp_path)
    monkeypatch.setattr(source_runtime, "validate_muscriptor_runtime_identity", lambda _root: "")

    assert _validate(project_root, venv_root, python, package_root) == ""


def test_accepts_isolated_xpu_venv_and_reports_its_exact_repair_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    venv_root = project_root / "venv-xpu"
    python = venv_root / "Scripts" / "python.exe"
    package_root = venv_root / "Lib" / "site-packages" / "muscriptor"
    python.parent.mkdir(parents=True)
    python.touch()
    package_root.mkdir(parents=True)
    (venv_root / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n", encoding="utf-8"
    )
    monkeypatch.setenv("MUSIC_TO_MIDI_ACCELERATOR", "xpu")
    monkeypatch.setattr(source_runtime, "validate_muscriptor_runtime_identity", lambda _root: "")

    assert _validate(project_root, venv_root, python, package_root) == ""
    failure = source_runtime.source_runtime_failure_message(
        "broken runtime",
        project_root,
        platform_name="nt",
    )
    assert str(python) in failure


@pytest.mark.parametrize("version_info", ((3, 10), (3, 13), (4, 0)))
def test_rejects_unsupported_source_python_versions(
    tmp_path: Path,
    version_info: tuple[int, int],
) -> None:
    project_root, venv_root, python, package_root = _runtime_layout(tmp_path)

    error = _validate(
        project_root,
        venv_root,
        python,
        package_root,
        version_info=version_info,
    )

    assert "仅支持 Python 3.11-3.12" in error
    assert f"{version_info[0]}.{version_info[1]}" in error


def test_rejects_32_bit_source_python(tmp_path: Path) -> None:
    project_root, venv_root, python, package_root = _runtime_layout(tmp_path)

    error = _validate(
        project_root,
        venv_root,
        python,
        package_root,
        pointer_bits=32,
    )

    assert "仅支持 64 位 Python" in error


def test_rejects_global_python_before_muscriptor_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, venv_root, python, package_root = _runtime_layout(tmp_path)
    called = False

    def unexpected_identity_check(_root: Path) -> str:
        nonlocal called
        called = True
        return ""

    monkeypatch.setattr(
        source_runtime,
        "validate_muscriptor_runtime_identity",
        unexpected_identity_check,
    )
    global_python = project_root / "global" / "python.exe"

    error = _validate(
        project_root,
        venv_root,
        python,
        package_root,
        executable=global_python,
        prefix=global_python.parent,
        base_prefix=global_python.parent,
    )

    assert "全局 Python" in error
    assert called is False


def test_rejects_a_different_virtual_environment(tmp_path: Path) -> None:
    project_root, venv_root, python, package_root = _runtime_layout(tmp_path)
    foreign_venv = project_root / "other-venv"

    error = _validate(
        project_root,
        venv_root,
        python,
        package_root,
        executable=foreign_venv / "Scripts" / "python.exe",
        prefix=foreign_venv,
    )

    assert "其他虚拟环境" in error
    assert str(venv_root) in error


def test_rejects_project_venv_that_inherits_global_site_packages(
    tmp_path: Path,
) -> None:
    project_root, venv_root, python, package_root = _runtime_layout(tmp_path)
    (venv_root / "pyvenv.cfg").write_text("include-system-site-packages = true\n", encoding="utf-8")

    error = _validate(project_root, venv_root, python, package_root)

    assert "必须隔离全局 site-packages" in error
    assert "'true'" in error


def test_rejects_muscriptor_loaded_outside_project_site_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, venv_root, python, package_root = _runtime_layout(tmp_path)
    monkeypatch.setattr(source_runtime, "validate_muscriptor_runtime_identity", lambda _root: "")
    global_package = project_root / "global" / "site-packages" / "muscriptor"
    global_package.mkdir(parents=True)

    error = _validate(
        project_root,
        venv_root,
        python,
        package_root,
        package_root=global_package,
    )

    assert "不是从项目虚拟环境" in error
    assert os.path.normcase(str(global_package)) in os.path.normcase(error)


def test_preserves_exact_muscriptor_identity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root, venv_root, python, package_root = _runtime_layout(tmp_path)
    monkeypatch.setattr(
        source_runtime,
        "validate_muscriptor_runtime_identity",
        lambda _root: "MuScriptor source identity mismatch: actual=bad",
    )

    error = _validate(project_root, venv_root, python, package_root)

    assert "source identity mismatch" in error
    assert source_runtime.MUSCRIPTOR_SOURCE_REQUIREMENT in error


def test_frozen_application_is_outside_source_runtime_gate(tmp_path: Path) -> None:
    assert (
        source_runtime.validate_source_runtime_identity(tmp_path / "missing-project", frozen=True)
        == ""
    )


def test_requirement_failure_exits_before_application_start(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        source_runtime,
        "validate_source_runtime_identity",
        lambda: "current interpreter is global",
    )

    with pytest.raises(SystemExit) as raised:
        source_runtime.require_source_runtime_identity()

    assert raised.value.code == 2
    error_output = capsys.readouterr().err
    assert "源码运行时校验失败" in error_output
    if os.name == "nt":
        assert "install.bat" in error_output
        assert "run.bat" in error_output
    else:
        assert "./install.sh" in error_output
        assert "./run.sh" in error_output


def test_web_api_launcher_gates_source_runtime_before_argument_parsing() -> None:
    from src.web_api import __main__ as web_api_main

    with (
        patch.object(
            web_api_main,
            "require_source_runtime_identity",
            side_effect=SystemExit(2),
        ) as runtime_gate,
        patch.object(web_api_main, "create_app") as create_app,
        patch.object(sys, "argv", ["python", "--invalid-argument"]),
        pytest.raises(SystemExit) as raised,
    ):
        web_api_main.main()

    assert raised.value.code == 2
    runtime_gate.assert_called_once_with()
    create_app.assert_not_called()


@pytest.mark.parametrize("relative_path", ("README.md", "docs/README.md", "docs/README_zh.md"))
def test_source_run_documentation_uses_exact_project_interpreters(
    relative_path: str,
) -> None:
    documentation = (source_runtime.PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

    assert ".\\venv\\Scripts\\python.exe -m src.main" in documentation
    assert "./venv/bin/python -m src.main" in documentation
    assert "include-system-site-packages=false" in documentation
