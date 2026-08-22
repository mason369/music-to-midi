import ast
import json
import math
import os
import shutil
import sys
import tempfile
import time
import types
import urllib.request
from pathlib import Path
from unittest import mock

import gradio as gradio
import pytest

from src.models.data_models import MuscriptorProcessingChain, TempoMode


def _function_source(path: str, name: str) -> str:
    source = Path(path).read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    return ast.get_source_segment(source, function)


def _colab_code() -> str:
    notebook = json.loads(Path("colab_notebook.ipynb").read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def test_pinned_gradio_fastapi_starlette_stack_serves_the_homepage():
    from src.gui.web.server_runtime import configure_uvicorn_websocket_protocol

    assert configure_uvicorn_websocket_protocol() == "websockets-sansio"
    with gradio.Blocks() as demo:
        gradio.Markdown("runtime-health")

    try:
        _, local_url, _ = demo.launch(
            server_name="127.0.0.1",
            prevent_thread_lock=True,
            quiet=True,
            theme=gradio.themes.Base(),
        )
        with urllib.request.urlopen(local_url, timeout=10) as response:
            body = response.read().decode("utf-8")
            status = response.status
    finally:
        demo.close()

    assert status == 200
    assert "gradio" in body.lower()


def test_space_request_outputs_have_failure_and_success_cleanup_contracts():
    source = Path("space/app.py").read_text(encoding="utf-8")
    clear_logs_source = _function_source("space/app.py", "clear_logs")
    convert_source = _function_source("space/app.py", "_convert_impl")
    prepare_source = _function_source("space/app.py", "_prepare_request_models")
    ensure_vocal_source = _function_source("space/app.py", "ensure_vocal_split_weights")
    vocal_gpu_source = _function_source("space/app.py", "_validate_vocal_split_gpu_runtime")
    gpu_entry_source = _function_source("space/app.py", "_convert_audio_to_midi_on_gpu")
    public_source = _function_source("space/app.py", "convert_audio_to_midi")

    assert "output_dir = _create_space_output_dir()" in convert_source
    assert "_remove_space_output_dir(output_dir)" in convert_source
    assert "ensure_model_weights" not in convert_source
    assert "ensure_miros_weights" not in convert_source
    assert "ensure_model_weights" in prepare_source
    assert "ensure_miros_weights" in prepare_source
    assert "torch.cuda.is_available" not in ensure_vocal_source
    assert "ort.get_available_providers" not in ensure_vocal_source
    assert "torch.cuda.is_available" in vocal_gpu_source
    assert "ort.get_available_providers" in vocal_gpu_source
    assert "_validate_gpu_runtime_for_request(mode)" in gpu_entry_source
    assert "_prepare_request_models" in public_source
    assert "_convert_audio_to_midi_on_gpu" in public_source
    assert "if audio_path is None" in public_source
    assert "_estimate_zerogpu_duration" in public_source
    assert public_source.index("_estimate_zerogpu_duration") < public_source.index(
        "_prepare_request_models"
    )
    assert public_source.index("_prepare_request_models") < public_source.index(
        "_convert_audio_to_midi_on_gpu"
    )
    assert '@spaces.GPU(duration=_estimate_zerogpu_duration, size="large")' in source
    assert "def _convert_audio_to_midi_on_gpu(" in source
    assert 'api_name="convert",\n        concurrency_limit=1,' in source
    assert (
        'DEVICE_LABEL = st("space.ui.zerogpu_device") if ZERO_GPU else get_device_label()' in source
    )
    assert "SPACE_OUTPUT_RETENTION_SECONDS" in source
    assert "atexit.register(_cleanup_space_instance_at_exit)" in source
    assert "delete_cache=(3600, SPACE_OUTPUT_RETENTION_SECONDS)" in source
    assert "shutil.rmtree(candidate)" in source
    assert "ignore_errors=True" not in source
    assert "except Exception" not in clear_logs_source


def test_space_disables_gradio_auto_wrap_and_detects_the_actual_zerogpu_runtime():
    source = Path("space/app.py").read_text(encoding="utf-8")
    requirements = Path("space/requirements.txt").read_text(encoding="utf-8")

    assert 'os.environ.get("SPACES_ZERO_GPU", "")' in source
    assert "spaces.disable_gradio_auto_wrap()" in source
    assert source.index("spaces.disable_gradio_auto_wrap()") < source.index("import gradio as gr")
    assert "from spaces.config import Config as SpacesConfig" in source
    assert "ZERO_GPU = bool(SpacesConfig.zero_gpu)" in source
    assert '{"1", "t", "true"}' in source
    assert (
        '"yes"'
        not in source[source.index("try:\n    import spaces") : source.index("import gradio as gr")]
    )
    assert "spaces==0.51.1" in requirements
    assert "pydantic==2.10.6" in requirements
    assert "pydantic-core==2.27.2" in requirements
    assert "_patched_json_schema" not in source


def test_space_registered_handler_requests_logged_in_zerogpu_headers_without_wrapping_cpu_work():
    source = Path("space/app.py").read_text(encoding="utf-8")
    assert 'setattr(convert_audio_to_midi, "zerogpu", None)' not in source
    assert 'setattr(_convert_one_track, "zerogpu", None)' not in source
    assert '@spaces.GPU(duration=_estimate_zerogpu_duration, size="large")' in source
    assert '@spaces.GPU(duration=_estimate_manual_zerogpu_duration, size="large")' in source
    assert "api_name=False" not in source
    assert "show_api=False" not in source
    assert 'api_visibility="private"' in source
    assert 'api_visibility="undocumented"' in source
    blocks_arguments = source.split("with gr.Blocks(", 1)[1].split(") as demo:", 1)[0]
    assert "theme=" not in blocks_arguments
    assert "css=" not in blocks_arguments
    assert "head=" not in blocks_arguments
    assert "theme=SPACE_THEME" in source
    assert "css=CUSTOM_CSS" in source
    assert "LOG_POLL_HEAD" not in source
    assert "fetch('./api/read_logs'" not in source
    assert "log_refresh_timer = gr.Timer(2.0)" in source
    assert "log_refresh_timer.tick(" in source
    assert "head=mixer_head() + muscriptor_result_head()" in source
    web_api_launcher = Path("src/web_api/__main__.py").read_text(encoding="utf-8")
    assert "websocket_protocol = configure_uvicorn_websocket_protocol()" in web_api_launcher
    assert "ws=websocket_protocol" in web_api_launcher

    def cpu_prepare_then_gpu(value):
        return value

    with gradio.Blocks() as demo:
        input_box = gradio.Textbox()
        output_box = gradio.Textbox()
        gradio.Button().click(cpu_prepare_then_gpu, input_box, output_box)

    dependency = demo.get_config_file()["dependencies"][0]
    assert "zerogpu" not in dependency
    assert not hasattr(cpu_prepare_then_gpu, "zerogpu")
    assert demo.fns[0].fn is cpu_prepare_then_gpu

    # spaces 0.51.1 no longer needs a Gradio dependency-level ``zerogpu``
    # flag. Its GPU wrapper reads the active Gradio request context at the
    # nested call boundary, preserving the logged-in request while CPU-side
    # admission and model preparation remain outside the GPU allocation.
    import spaces
    from spaces.zero import client as zero_client

    wrapper_source = (Path(spaces.__file__).parent / "zero" / "wrappers.py").read_text(
        encoding="utf-8"
    )
    assert "request_var = gradio_request_var()" in wrapper_source
    assert "request = request_var.get(None)" in wrapper_source
    assert "client.schedule(task_id=task_id, request=request" in wrapper_source

    scheduled = {}
    request = object()

    def capture_headers(actual_request):
        scheduled["request"] = actual_request
        return {"x-ip-token": "test-token"}

    class FakeApiClient:
        def schedule(self, **kwargs):
            scheduled.update(kwargs)
            return (
                zero_client.ScheduleResponse(
                    idle=True,
                    nvidiaIndex=0,
                    nvidiaUUID="test-gpu",
                    allowToken="test-token",
                ),
                types.SimpleNamespace(auth="regular"),
            )

    with (
        mock.patch.object(
            zero_client,
            "_get_headers",
            side_effect=capture_headers,
        ),
        mock.patch.object(
            zero_client,
            "_get_token_and_payload",
            return_value=("test-token", {}),
        ),
        mock.patch.object(zero_client, "get_duration_seconds", return_value=1),
        mock.patch.object(
            zero_client.utils,
            "self_cgroup_device_path",
            return_value="/test/cgroup",
        ),
        mock.patch.object(zero_client, "api_client", return_value=FakeApiClient()),
    ):
        result = zero_client.schedule(task_id=7, request=request, gpu_size="large")

    assert result.allowToken == "test-token"
    assert scheduled["request"] is request
    assert scheduled["token"] == "test-token"
    assert scheduled["token_version"] == 2
    assert scheduled["gpu_size"] == "large"


def test_space_output_helpers_delete_failed_and_expired_request_directories():
    helper_names = (
        "_remove_space_output_dir",
        "_remove_stale_space_instance",
        "_cleanup_stale_space_outputs",
        "_create_space_output_dir",
    )
    helper_source = "\n\n".join(_function_source("space/app.py", name) for name in helper_names)

    with tempfile.TemporaryDirectory() as temp_root:
        parent = Path(temp_root).resolve()
        instance = parent / "instance-current"
        instance.mkdir()
        namespace = {
            "Path": Path,
            "SPACE_OUTPUT_PARENT": parent,
            "SPACE_OUTPUT_INSTANCE": instance,
            "SPACE_OUTPUT_RETENTION_SECONDS": 60,
            "SPACE_REQUEST_PREFIX": "request-",
            "logger": types.SimpleNamespace(error=lambda *args: None, info=lambda *args: None),
            "shutil": shutil,
            "tempfile": tempfile,
            "time": time,
        }
        exec(helper_source, namespace)

        failed = Path(namespace["_create_space_output_dir"]())
        (failed / "partial.mid").write_bytes(b"partial")
        namespace["_remove_space_output_dir"](failed)
        assert not failed.exists()

        expired = Path(namespace["_create_space_output_dir"]())
        (expired / "result.mid").write_bytes(b"midi")
        old_timestamp = time.time() - 120
        os.utime(expired, (old_timestamp, old_timestamp))
        namespace["_cleanup_stale_space_outputs"](now=time.time())
        assert not expired.exists()

        stale_instance = parent / "instance-crashed"
        stale_request = stale_instance / "request-old"
        stale_request.mkdir(parents=True)
        (stale_request / "result.mid").write_bytes(b"midi")
        os.utime(stale_request, (old_timestamp, old_timestamp))
        os.utime(stale_instance, (old_timestamp, old_timestamp))

        fresh_instance = parent / "instance-live"
        fresh_request = fresh_instance / "request-new"
        fresh_request.mkdir(parents=True)
        namespace["_cleanup_stale_space_outputs"](now=time.time())

        assert not stale_instance.exists()
        assert fresh_instance.exists()
        assert instance.exists()
        with pytest.raises(RuntimeError, match="Refusing to remove"):
            namespace["_remove_stale_space_instance"](instance)

        with tempfile.TemporaryDirectory(prefix="not-owned-", dir=instance.parent) as outside:
            with pytest.raises(RuntimeError, match="Refusing to remove"):
                namespace["_remove_space_output_dir"](outside)


def test_zerogpu_duration_contract_rejects_a_request_above_free_window():
    function_source = _function_source("space/app.py", "_estimate_zerogpu_duration")
    fake_multi_model = types.SimpleNamespace(
        YOURMT3=types.SimpleNamespace(value="yourmt3"),
        MUSCRIPTOR=types.SimpleNamespace(value="muscriptor"),
    )
    namespace = {
        "MODE_IDS": ("smart",),
        "MULTI_INSTRUMENT_MODE_IDS": {"smart"},
        "ZERO_GPU_BASE_RUNTIME_SECONDS": 180.0,
        "ZERO_GPU_FREE_ACCOUNT_BUDGET_SECONDS": 300,
        "ZERO_GPU_LARGE_DURATION_FACTOR": 1.5,
        "ZERO_GPU_MODE_RUNTIME_MULTIPLIERS": {"smart": 8.0},
        "ZERO_GPU_MULTI_BACKEND_RUNTIME_FACTORS": {
            "yourmt3": 1.0,
            "miros": 2.5,
            "muscriptor": 3.0,
        },
        "ZERO_GPU_YOURMT3_MODEL_RUNTIME_FACTORS": {
            "yptf_moe_multi_nops": 1.25,
        },
        "ZERO_GPU_MUSCRIPTOR_MODEL_RUNTIME_FACTORS": {
            "large": 3.0,
            "medium": 1.2,
            "small": 0.7,
        },
        "MultiInstrumentModel": fake_multi_model,
        "MuscriptorProcessingChain": MuscriptorProcessingChain,
        "math": math,
        "st": lambda key, **kwargs: f"{key}: {kwargs}",
        "gr": types.SimpleNamespace(Error=RuntimeError),
    }
    exec(function_source, namespace)

    fake_librosa = types.SimpleNamespace(get_duration=lambda *, path: 20.0)
    with mock.patch.dict(sys.modules, {"librosa": fake_librosa}):
        with pytest.raises(RuntimeError, match="zerogpu_clip_too_long"):
            namespace["_estimate_zerogpu_duration"](
                "song.mp3", "smart", "yourmt3", "yptf_moe_multi_nops"
            )

        fake_librosa.get_duration = lambda *, path: 1.0
        yourmt3_estimate = namespace["_estimate_zerogpu_duration"](
            "clip.mp3", "smart", "yourmt3", "yptf_moe_multi_nops"
        )
        miros_estimate = namespace["_estimate_zerogpu_duration"](
            "clip.mp3", "smart", "miros", "unused"
        )
        assert yourmt3_estimate == 190
        assert miros_estimate == 200
        assert miros_estimate > yourmt3_estimate

        for model, accepted_duration, rejected_duration in (
            ("large", 0.833, 0.834),
            ("medium", 2.083, 2.084),
            ("small", 3.571, 3.572),
        ):
            fake_librosa.get_duration = lambda *, path, value=accepted_duration: value
            assert (
                namespace["_estimate_zerogpu_duration"](
                    "clip.mp3", "smart", "muscriptor", "unused", model
                )
                == 200
            )
            fake_librosa.get_duration = lambda *, path, value=rejected_duration: value
            with pytest.raises(RuntimeError, match="zerogpu_clip_too_long"):
                namespace["_estimate_zerogpu_duration"](
                    "clip.mp3", "smart", "muscriptor", "unused", model
                )

        with pytest.raises(RuntimeError, match="Unsupported YourMT3 checkpoint"):
            namespace["_estimate_zerogpu_duration"]("clip.mp3", "smart", "yourmt3", "unknown")


def test_zerogpu_admission_and_model_preparation_happen_before_gpu_entry():
    public_source = _function_source("space/app.py", "convert_audio_to_midi")
    events = []

    def _fake_stream(gpu_fn, args, panel_mode, progress):
        del panel_mode
        yield gpu_fn(*args, progress=progress)

    namespace = {
        "ZERO_GPU": True,
        "MODE_IDS": ("smart",),
        "MuscriptorProcessingChain": MuscriptorProcessingChain,
        "TempoMode": TempoMode,
        "gr": types.SimpleNamespace(Progress=lambda: None, Error=RuntimeError),
        "st": lambda key: key,
        "_estimate_zerogpu_duration": lambda *args, **kwargs: events.append("admit"),
        "_prepare_request_models": lambda *args, **kwargs: events.append("prepare"),
        "_convert_audio_to_midi_on_gpu": lambda *args, **kwargs: (events.append("gpu") or "result"),
        "_stream_gpu_job": _fake_stream,
    }
    exec(public_source, namespace)

    # The public handler is a streaming generator; driving it to completion is
    # what must run admission -> preparation -> GPU entry in that order.
    results = list(
        namespace["convert_audio_to_midi"](
            "clip.wav",
            "smart",
            "yourmt3",
            "yptf_moe_multi_nops",
            progress=None,
        )
    )

    assert results == ["result"]
    assert events == ["admit", "prepare", "gpu"]

    def reject(*args, **kwargs):
        events.append("reject")
        raise RuntimeError("over budget")

    namespace["_estimate_zerogpu_duration"] = reject
    events.clear()
    with pytest.raises(RuntimeError, match="over budget"):
        list(
            namespace["convert_audio_to_midi"](
                "too-long.wav",
                "smart",
                "yourmt3",
                "yptf_moe_multi_nops",
                progress=None,
            )
        )
    assert events == ["reject"]


def test_colab_request_outputs_are_scoped_to_runtime_and_failures_are_deleted():
    source = _colab_code()

    assert (
        'COLAB_OUTPUT_ROOT = Path(tempfile.mkdtemp(prefix="music_to_midi_colab_session_"))'
        in source
    )
    assert 'tempfile.mkdtemp(prefix="request-", dir=COLAB_OUTPUT_ROOT)' in source
    assert "shutil.rmtree(output_dir)" in source
    assert "atexit.register(_cleanup_colab_output_root)" in source
    assert "delete_cache=(3600, 86400)" in source


def test_telknet_claims_state_asset_baseline_reuse_and_project_extensions():
    documentation_paths = (
        "README.md",
        "docs/README.md",
        "docs/README_zh.md",
        "space/README.md",
        "colab_notebook.ipynb",
    )
    combined = "\n".join(Path(path).read_text(encoding="utf-8") for path in documentation_paths)

    for overclaim in (
        "TelkNet" + "-aligned",
        "TelkNet/MVSep" + "-aligned",
        "TelkNet " + "对齐路线",
        "TelkNet/MVSep " + "对齐",
        "与 TelkNet 当前详细路线" + "对齐",
    ):
        assert overclaim not in combined

    assert "YourMT3/MIROS“官方 writer 后只补 tempo" in combined
    assert "52be6fec179be492f5229ba149545ac2833b284a" in combined
    assert "没有证据证明该 `dev` 已部署线上" in combined
    assert "Both separation workflows likewise deliver WAV first" in combined
    assert "MIDI is explicitly triggered" in combined


def test_zfturbo_reference_has_complete_mit_notice():
    notice = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "ZFTurbo/Music-Source-Separation-Training" in notice
    assert "v1.0.20" in notice
    assert "e6279a79bcf861ea355ef7f8f76808a2731b6636" in notice
    assert "Copyright (c) 2024 Roman Solovyev (ZFTurbo)" in notice
    assert "Permission is hereby granted, free of charge" in notice
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in notice
