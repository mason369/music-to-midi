import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_CONTRACT_FILES = (
    "_internal/config/web-frontend.json",
    "_internal/LICENSE",
    "_internal/THIRD_PARTY_NOTICES.md",
    "_internal/web/app.js",
    "_internal/web/assets/app_icon.png",
    "_internal/web/index.html",
    "_internal/web/locales/en_US.json",
    "_internal/web/locales/zh_CN.json",
    "_internal/web/README.md",
    "_internal/web/runtime-config.json",
    "_internal/web/styles.css",
)


def _write_role(root: Path, executable: str, executable_payload: bytes, files: dict[str, bytes]):
    internal = root / "_internal"
    internal.mkdir(parents=True)
    (root / executable).write_bytes(executable_payload)
    for relative, payload in files.items():
        path = internal / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _link_backend_internal(app_root: Path, backend_root: Path, executable: str, payload: bytes):
    (backend_root / "_internal").mkdir(parents=True)
    (backend_root / executable).write_bytes(payload)
    for source in (app_root / "_internal").rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(app_root / "_internal")
        target = backend_root / "_internal" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)


def _write_frontend(root: Path, executable_payload: bytes, runtime_payload: bytes):
    root.mkdir(parents=True)
    (root / "MusicToMidiFrontend.exe").write_bytes(executable_payload)
    for relative in FRONTEND_CONTRACT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"shared frontend contract: {relative}".encode("utf-8"))
    runtime = root / "_internal" / "python311.dll"
    runtime.write_bytes(runtime_payload)


def test_universal_launcher_has_deterministic_strict_accelerator_selection():
    source = (REPO_ROOT / "tools" / "universal_launcher" / "UniversalLauncher.cs").read_text(
        encoding="utf-8"
    )

    assert 'explicitAccelerator != "cuda" && explicitAccelerator != "xpu"' in source
    assert source.index("if (hasNvidia)") < source.index("if (hasSupportedIntelCandidate)")
    assert 'name.IndexOf("Intel"' in source
    assert 'name.IndexOf("Arc"' in source
    assert '"runtimes",' in source
    assert "accelerator" in source
    assert '"MusicToMidiXpu.exe"' in source
    assert '"MusicToMidiBackendXpu.exe"' in source
    assert source.count("Process.Start(startInfo)") == 1
    assert "selected child failure" not in source.lower()


def test_universal_assembler_keeps_roles_and_native_runtimes_separate():
    script = (REPO_ROOT / "build_universal_windows.ps1").read_text(encoding="utf-8")
    usage_files = list((REPO_ROOT / "resources" / "universal").glob("README-*.txt"))
    assert len(usage_files) == 1
    usage = usage_files[0].read_text(encoding="utf-8")

    for role in (
        "MusicToMidi-App",
        "MusicToMidi-WebBackend",
        "MusicToMidi-WebFrontend",
    ):
        assert role in script
    assert 'Join-Path $appRoot "runtimes\\cuda"' in script
    assert 'Join-Path $appRoot "runtimes\\xpu"' in script
    assert 'Join-Path $backendRoot "runtimes\\cuda"' in script
    assert 'Join-Path $backendRoot "runtimes\\xpu"' in script
    assert "Merge-IdenticalRuntimeFiles" in script
    assert "Assert-FrontendContract" in script
    assert "frontend_canonical_source" in script
    assert "frontend_contract_files" in script
    assert "Get-Sha256" in script
    assert "System.Security.Cryptography.SHA256" in script
    assert "New-Item -ItemType HardLink" in script
    assert 'failure_fallback = $false' in script
    assert '".incomplete"' in script
    assert "\u786c\u94fe\u63a5\u590d\u7528" in usage
    assert "\u53ea\u5360\u4e00\u4efd\u7269\u7406\u7a7a\u95f4" in usage
    assert "\u53ef\u72ec\u7acb\u590d\u5236\u5230\u5c40\u57df\u7f51\u5176\u4ed6\u7535\u8111" in usage


@pytest.mark.skipif(os.name != "nt", reason="Windows Universal launcher integration")
def test_universal_launchers_compile_route_and_propagate_exit_codes(tmp_path):
    build_dir = tmp_path / "launcher-build"
    build_script = REPO_ROOT / "scripts" / "build_universal_windows_launchers.ps1"
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(build_script),
            "-OutputDirectory",
            str(build_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    app_root = tmp_path / "MusicToMidi-App"
    backend_root = tmp_path / "MusicToMidi-WebBackend"
    app_root.mkdir()
    backend_root.mkdir()
    shutil.copy2(build_dir / "MusicToMidi.exe", app_root / "MusicToMidi.exe")
    shutil.copy2(
        build_dir / "MusicToMidiBackend.exe",
        backend_root / "MusicToMidiBackend.exe",
    )
    command_interpreter = Path(os.environ["WINDIR"]) / "System32" / "cmd.exe"
    child_paths = (
        app_root / "runtimes" / "cuda" / "MusicToMidi.exe",
        app_root / "runtimes" / "xpu" / "MusicToMidiXpu.exe",
        backend_root / "runtimes" / "cuda" / "MusicToMidiBackend.exe",
        backend_root / "runtimes" / "xpu" / "MusicToMidiBackendXpu.exe",
    )
    for child in child_paths:
        child.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(command_interpreter, child)

    trace = tmp_path / "launcher-trace.tsv"
    environment = os.environ.copy()
    environment["MUSIC_TO_MIDI_UNIVERSAL_NO_DIALOG"] = "1"
    environment["MUSIC_TO_MIDI_UNIVERSAL_TRACE_FILE"] = str(trace)

    environment["MUSIC_TO_MIDI_ACCELERATOR"] = "cuda"
    cuda = subprocess.run(
        [app_root / "MusicToMidi.exe", "/d", "/c", "exit", "21"],
        env=environment,
        timeout=15,
        check=False,
    )
    environment["MUSIC_TO_MIDI_ACCELERATOR"] = "xpu"
    xpu = subprocess.run(
        [backend_root / "MusicToMidiBackend.exe", "/d", "/c", "exit", "22"],
        env=environment,
        timeout=15,
        check=False,
    )
    environment["MUSIC_TO_MIDI_ACCELERATOR"] = "invalid"
    invalid = subprocess.run(
        [backend_root / "MusicToMidiBackend.exe"],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
    )

    assert cuda.returncode == 21
    assert xpu.returncode == 22
    assert invalid.returncode == 70
    assert "只能是 cuda 或 xpu" in invalid.stderr
    selections = trace.read_text(encoding="utf-8").splitlines()
    assert len(selections) == 2
    assert "\tapp\tcuda\t" in selections[0]
    assert "\tbackend\txpu\t" in selections[1]


@pytest.mark.skipif(os.name != "nt", reason="Windows NTFS Universal assembly integration")
def test_universal_assembler_deduplicates_models_and_preserves_role_independence(tmp_path):
    cuda = tmp_path / "cuda"
    xpu = tmp_path / "xpu"
    cuda_app = cuda / "MusicToMidi-App"
    cuda_backend = cuda / "MusicToMidi-WebBackend"
    xpu_app = xpu / "MusicToMidi-XPU-App"
    xpu_backend = xpu / "MusicToMidi-XPU-WebBackend"

    shared_payload = b"one physical model payload"
    _write_role(
        cuda_app,
        "MusicToMidi.exe",
        b"cuda-app",
        {"models/shared.bin": shared_payload, "native/provider.bin": b"cuda"},
    )
    _link_backend_internal(cuda_app, cuda_backend, "MusicToMidiBackend.exe", b"cuda-web")
    _write_role(
        xpu_app,
        "MusicToMidiXpu.exe",
        b"xpu-app",
        {"models/shared.bin": shared_payload, "native/provider.bin": b"xpu"},
    )
    _link_backend_internal(
        xpu_app,
        xpu_backend,
        "MusicToMidiBackendXpu.exe",
        b"xpu-web",
    )
    _write_frontend(
        cuda / "MusicToMidi-WebFrontend",
        executable_payload=b"frontend-python-3.11.5",
        runtime_payload=b"python-3.11.5-runtime",
    )
    _write_frontend(
        xpu / "MusicToMidi-WebFrontend",
        executable_payload=b"frontend-python-3.11.9",
        runtime_payload=b"python-3.11.9-runtime",
    )

    output = tmp_path / "result" / "MusicToMidi-Universal"
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(REPO_ROOT / "build_universal_windows.ps1"),
            "-CudaDistRoot",
            str(cuda),
            "-XpuDistRoot",
            str(xpu),
            "-OutputRoot",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    app = output / "MusicToMidi-App"
    backend = output / "MusicToMidi-WebBackend"
    frontend = output / "MusicToMidi-WebFrontend"
    assert (app / "MusicToMidi.exe").is_file()
    assert (backend / "MusicToMidiBackend.exe").is_file()
    assert (frontend / "MusicToMidiFrontend.exe").is_file()
    assert (frontend / "MusicToMidiFrontend.exe").read_bytes() == b"frontend-python-3.11.5"
    cuda_model = app / "runtimes" / "cuda" / "_internal" / "models" / "shared.bin"
    xpu_model = app / "runtimes" / "xpu" / "_internal" / "models" / "shared.bin"
    cuda_backend_model = (
        backend / "runtimes" / "cuda" / "_internal" / "models" / "shared.bin"
    )
    xpu_backend_model = (
        backend / "runtimes" / "xpu" / "_internal" / "models" / "shared.bin"
    )
    assert os.path.samefile(cuda_model, xpu_model)
    assert os.path.samefile(cuda_model, cuda_backend_model)
    assert os.path.samefile(xpu_model, xpu_backend_model)
    assert not os.path.samefile(
        app / "runtimes" / "cuda" / "_internal" / "native" / "provider.bin",
        app / "runtimes" / "xpu" / "_internal" / "native" / "provider.bin",
    )
    assert not (output / ".incomplete").exists()
    build_info = json.loads((output / "UNIVERSAL_BUILD_INFO.json").read_text(encoding="utf-8"))
    assert build_info["failure_fallback"] is False
    assert build_info["accelerator_priority"] == ["cuda", "xpu"]
    assert build_info["frontend_canonical_source"] == "cuda"
    assert build_info["frontend_contract_files"] == [
        relative.replace("/", "\\") for relative in FRONTEND_CONTRACT_FILES
    ]
    assert build_info["cross_accelerator_shared_files"] >= 1
    assert build_info["cross_accelerator_shared_bytes"] >= len(shared_payload)
