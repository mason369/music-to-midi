"""A successful source push must not hide a failed production Space."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tools.verify_hf_space import wait_for_space

REVISION = "a" * 40
RUNNING = {
    "sha": REVISION,
    "runtime": {"stage": "RUNNING"},
    "subdomain": "mason369-music-to-midi",
    "cardData": {"sdk_version": "6.17.3"},
}
CONFIG = {"version": "6.17.3", "components": [{"id": 1}], "dependencies": [{"id": 0}]}


@pytest.mark.parametrize(
    "stage", ["BUILDING", "RUNNING_BUILDING", "APP_STARTING", "RUNNING_APP_STARTING"]
)
def test_waits_for_build_then_checks_live_application_without_forwarding_token(stage):
    building = deepcopy(RUNNING)
    building["runtime"]["stage"] = stage
    responses = iter([building, RUNNING, CONFIG])
    calls, sleeps = [], []

    def fetch(url, token=""):
        calls.append((url, token))
        return next(responses)

    wait_for_space(
        "mason369/music-to-midi", REVISION, "test-token", fetch=fetch, sleep=sleeps.append
    )
    assert sleeps == [15]
    assert calls[-1] == ("https://mason369-music-to-midi.hf.space/config", "")
    assert all(token == "test-token" for _url, token in calls[:-1])


@pytest.mark.parametrize(
    "stage",
    [
        "BUILD_ERROR",
        "RUNTIME_ERROR",
        "CONFIG_ERROR",
        "NO_APP_FILE",
        "STOPPED",
        "PAUSED",
        "DELETING",
        "FUTURE_STAGE",
        None,
    ],
)
def test_terminal_failures_stop_without_retry(stage):
    info = deepcopy(RUNNING)
    info["runtime"] = {"stage": stage, "errorMessage": "original dependency failure"}
    calls = []

    def fetch(*args):
        calls.append(args)
        return info

    with pytest.raises(RuntimeError, match="original dependency failure"):
        wait_for_space("mason369/music-to-midi", REVISION, "test", fetch=fetch)
    assert len(calls) == 1


def test_starting_replacement_that_fails_does_not_accept_the_previous_live_app():
    starting = deepcopy(RUNNING)
    starting["runtime"]["stage"] = "RUNNING_APP_STARTING"
    failed = deepcopy(RUNNING)
    failed["runtime"] = {"stage": "RUNTIME_ERROR", "errorMessage": "new app failed"}
    responses = iter([starting, failed])
    calls, sleeps = [], []

    def fetch(url, token=""):
        calls.append(url)
        return next(responses)

    with pytest.raises(RuntimeError, match="new app failed"):
        wait_for_space("mason369/music-to-midi", REVISION, "test", fetch=fetch, sleep=sleeps.append)
    assert sleeps == [15]
    assert len(calls) == 2
    assert all(url.startswith("https://huggingface.co/api/spaces/") for url in calls)


def test_wrong_revision_cannot_pass_using_an_old_healthy_app():
    with pytest.raises(RuntimeError, match="revision differs"):
        wait_for_space("mason369/music-to-midi", "b" * 40, "test", fetch=lambda *_: RUNNING)


@pytest.mark.parametrize("field", ["version", "components", "dependencies"])
def test_running_flag_does_not_replace_real_gradio_surface_check(field):
    bad_config = dict(CONFIG)
    bad_config.pop(field)
    responses = iter([RUNNING, bad_config])
    with pytest.raises(RuntimeError, match="declared Gradio"):
        wait_for_space("mason369/music-to-midi", REVISION, "test", fetch=lambda *_: next(responses))


def test_network_failure_is_not_retried():
    calls = []

    def fetch(*args):
        calls.append(args)
        raise OSError("HTTP 403")

    with pytest.raises(OSError, match="HTTP 403"):
        wait_for_space("mason369/music-to-midi", REVISION, "test", fetch=fetch)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "stage", ["BUILDING", "RUNNING_BUILDING", "APP_STARTING", "RUNNING_APP_STARTING"]
)
def test_build_timeout_is_bounded(stage):
    info = deepcopy(RUNNING)
    info["runtime"]["stage"] = stage
    timestamps = iter([0, 0, 10])
    sleeps = []
    with pytest.raises(TimeoutError):
        wait_for_space(
            "mason369/music-to-midi",
            REVISION,
            "test",
            fetch=lambda *_: info,
            timeout=10,
            sleep=sleeps.append,
            now=lambda: next(timestamps),
        )
    assert sleeps == [10]


def test_sync_workflow_waits_for_exact_pushed_commit_before_success():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/sync_to_hf.yml").read_text("utf-8"))
    steps = workflow["jobs"]["sync"]["steps"]
    push = next(step for step in steps if step.get("id") == "push")
    assert "space_commit=$(git rev-parse HEAD)" in push["run"]
    gate = steps[-1]
    assert "tools/verify_hf_space.py" in gate["run"]
    assert "steps.push.outputs.space_commit" in gate["run"]
    assert "continue-on-error" not in gate
    assert gate["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"
