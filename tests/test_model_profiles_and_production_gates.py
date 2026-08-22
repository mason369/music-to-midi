from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

import src.model_profiles as profile_module
import src.web_api.app as app_module
from src.model_profiles import (
    ALL_PROFILE_IDS,
    ModelProfileSnapshot,
    ModelProfileStatus,
    inspect_model_profiles,
    parse_profile_ids,
)
from src.model_profile_runtime_probe import MODEL_PROFILE_RUNTIME_PROBE_SWITCH
from src.web_api.app import create_app
from src.web_api.jobs import JobManager, QueueCapacityError
from src.web_api.schemas import JobStatus
from src.web_api.server_config import BackendServerConfig, load_backend_config, write_backend_config


class NeverRunEngine:
    def run(self, **_kwargs):  # pragma: no cover - the worker is disabled in these tests
        raise AssertionError("inference must not run")


def all_available_snapshot() -> ModelProfileSnapshot:
    return ModelProfileSnapshot(
        selection_mode="test",
        explicit_selection_required=True,
        enabled_profiles=ALL_PROFILE_IDS,
        profiles=tuple(
            ModelProfileStatus(id=profile_id, enabled=True, available=True)
            for profile_id in ALL_PROFILE_IDS
        ),
        ready=True,
    )


def inference_options(mode: str = "smart") -> dict[str, object]:
    return {
        "processing_mode": mode,
        "transcription_backend": "yourmt3",
        "yourmt3_model": "yptf_moe_multi_nops",
        "muscriptor_model": "large",
        "midi_track_mode": "multi_track",
        "tempo_mode": "fixed_auto",
        "custom_bpm": None,
        "use_gpu": True,
        "gpu_device": 0,
        "language": "zh_CN",
    }


def post_audio(client: TestClient, mode: str = "smart", **overrides):
    options = inference_options(mode)
    options.update(overrides)
    return client.post(
        "/api/v1/jobs",
        files={"audio": ("sample.wav", b"RIFF-production-gate", "audio/wav")},
        data={"options": json.dumps(options)},
    )


def test_profile_parser_is_explicit_and_canonical():
    assert parse_profile_ids("muscriptor:large, yourmt3:yptf_moe_multi_nops") == (
        "muscriptor",
        "yourmt3:yptf_moe_multi_nops",
    )
    with pytest.raises(ValueError, match="at least one"):
        parse_profile_ids(" , ")
    with pytest.raises(ValueError, match="unsupported model profile"):
        parse_profile_ids("unknown")


def test_required_profile_selection_fails_closed_when_missing(monkeypatch):
    monkeypatch.delenv(profile_module.ENABLED_PROFILES_ENV, raising=False)
    monkeypatch.setenv(profile_module.REQUIRE_ENABLED_PROFILES_ENV, "1")
    snapshot = inspect_model_profiles(refresh=True)
    assert snapshot.ready is False
    assert snapshot.selection_mode == "invalid"
    assert profile_module.ENABLED_PROFILES_ENV in str(snapshot.readiness_error)
    assert not any(item.enabled for item in snapshot.profiles)


def test_explicit_selection_does_not_probe_or_advertise_unselected_profiles(monkeypatch):
    monkeypatch.setenv(profile_module.ENABLED_PROFILES_ENV, "miros")
    monkeypatch.setenv(profile_module.REQUIRE_ENABLED_PROFILES_ENV, "1")
    probed: list[str] = []

    monkeypatch.setattr(profile_module, "_beat_this_unavailable_reason", lambda: None)
    monkeypatch.setattr(profile_module, "_midi_playback_unavailable_reason", lambda: None)

    def inspect_one(
        profile_id: str,
        _beat_reason: str | None,
        _playback_reason: str | None,
    ):
        probed.append(profile_id)
        return None

    monkeypatch.setattr(profile_module, "_profile_unavailable_reason", inspect_one)
    snapshot = inspect_model_profiles(refresh=True)
    assert snapshot.ready is True
    assert snapshot.enabled_profiles == ("miros",)
    assert probed == ["miros"]
    assert snapshot.status("miros").available is True
    assert snapshot.status("piano_aria_amt").available is False
    assert "not enabled" in str(snapshot.status("piano_aria_amt").unavailable_reason)


def test_direct_midi_profile_requires_offline_playback_assets(monkeypatch):
    monkeypatch.setenv(profile_module.ENABLED_PROFILES_ENV, "piano_transkun")
    monkeypatch.setenv(profile_module.REQUIRE_ENABLED_PROFILES_ENV, "1")
    monkeypatch.setattr(profile_module, "_beat_this_unavailable_reason", lambda: None)
    monkeypatch.setattr(
        profile_module,
        "_midi_playback_unavailable_reason",
        lambda: "MuseScore General SoundFont failed validation: deliberately absent",
    )

    snapshot = inspect_model_profiles(refresh=True)

    assert snapshot.ready is False
    assert "SoundFont" in str(snapshot.status("piano_transkun").unavailable_reason)


def test_separation_only_profile_does_not_probe_playback_assets(monkeypatch):
    monkeypatch.setenv(profile_module.ENABLED_PROFILES_ENV, "vocal_split")
    monkeypatch.setenv(profile_module.REQUIRE_ENABLED_PROFILES_ENV, "1")
    monkeypatch.setattr(
        profile_module,
        "_midi_playback_unavailable_reason",
        lambda: pytest.fail("WAV-only separation must not require MIDI playback assets"),
    )
    monkeypatch.setattr(profile_module, "_profile_unavailable_reason", lambda *_args: None)

    snapshot = inspect_model_profiles(refresh=True)

    assert snapshot.ready is True


def test_direct_midi_preparation_includes_every_shared_playback_asset(tmp_path, monkeypatch):
    prepared: list[str] = []
    beat = tmp_path / "final0.ckpt"
    soundfont = tmp_path / "MuseScore_General.sf2"
    fluidsynth = tmp_path / "fluidsynth"
    monkeypatch.setattr(
        "download_beat_this_model.download_beat_this_model",
        lambda *, printer: beat,
    )
    monkeypatch.setattr(
        "src.utils.muscriptor_soundfont_downloader.download_muscriptor_soundfont",
        lambda *, printer: soundfont,
    )
    monkeypatch.setattr(
        "src.utils.fluidsynth_runtime.get_fluidsynth_executable",
        lambda: fluidsynth,
    )
    monkeypatch.setattr(
        profile_module,
        "_prepare_one",
        lambda profile_id, _printer: prepared.append(profile_id),
    )
    monkeypatch.setattr(
        profile_module, "inspect_model_profiles", lambda **_kwargs: all_available_snapshot()
    )

    messages: list[str] = []
    snapshot = profile_module.prepare_profiles(["piano_transkun"], printer=messages.append)

    assert snapshot.ready is True
    assert prepared == ["piano_transkun"]
    assert any(str(beat) in message for message in messages)
    assert any(str(soundfont) in message for message in messages)
    assert any(str(fluidsynth) in message for message in messages)


def test_beat_this_readiness_validates_the_runtime_resolved_checkpoint(monkeypatch):
    validated: list[object] = []

    def validate(path=None):
        validated.append(path)

    monkeypatch.setattr("src.core.beat_this_tracker.validate_beat_this_checkpoint", validate)
    assert profile_module._beat_this_unavailable_reason() is None
    assert validated == [None]

    def fail(_path=None):
        raise RuntimeError("identity mismatch")

    monkeypatch.setattr("src.core.beat_this_tracker.validate_beat_this_checkpoint", fail)
    assert "missing or failed" in str(profile_module._beat_this_unavailable_reason())


@pytest.mark.parametrize("frozen", [False, True])
def test_audio_separator_readiness_uses_a_real_isolated_runtime_entrypoint(monkeypatch, frozen):
    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="runtime ready")

    monkeypatch.setattr(profile_module.subprocess, "run", run)
    if frozen:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
    else:
        monkeypatch.delattr(sys, "frozen", raising=False)
    profile_module._audio_separator_runtime_unavailable_reason.cache_clear()

    assert profile_module._audio_separator_runtime_unavailable_reason("vocal_split") is None
    assert len(calls) == 1
    if frozen:
        assert calls[0] == [
            sys.executable,
            MODEL_PROFILE_RUNTIME_PROBE_SWITCH,
            "vocal_split",
        ]
    else:
        assert calls[0][0:2] == [sys.executable, "-c"]
        assert "run_model_profile_runtime_probe" in calls[0][2]
        assert calls[0][3] == "vocal_split"


def test_api_rejects_an_unavailable_profile_before_creating_job_files(tmp_path, monkeypatch):
    statuses = []
    for profile_id in ALL_PROFILE_IDS:
        selected = profile_id == "yourmt3:yptf_moe_multi_nops"
        statuses.append(
            ModelProfileStatus(
                id=profile_id,
                enabled=selected,
                available=False,
                unavailable_reason="checkpoint deliberately absent",
            )
        )
    snapshot = ModelProfileSnapshot(
        selection_mode="explicit",
        explicit_selection_required=True,
        enabled_profiles=("yourmt3:yptf_moe_multi_nops",),
        profiles=tuple(statuses),
        ready=False,
        readiness_error="configured model profiles are unavailable",
    )
    monkeypatch.setattr(app_module, "inspect_model_profiles", lambda: snapshot)
    manager = JobManager(tmp_path, engine=NeverRunEngine(), start_worker=False)
    try:
        with TestClient(create_app(manager=manager)) as client:
            response = post_audio(client)
        assert response.status_code == 503
        assert "checkpoint deliberately absent" in response.json()["detail"]
        assert list(tmp_path.iterdir()) == []
    finally:
        manager.close()


def test_http_queue_limit_returns_429_and_preserves_the_first_job(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "inspect_model_profiles", all_available_snapshot)
    manager = JobManager(
        tmp_path,
        engine=NeverRunEngine(),
        start_worker=False,
        max_queued_jobs=1,
    )
    try:
        with TestClient(create_app(manager=manager)) as client:
            first = post_audio(client)
            second = post_audio(client)
        assert first.status_code == 202
        assert second.status_code == 429
        assert "queue is full" in second.json()["detail"]
        assert manager.health()["queued_jobs"] == 1
    finally:
        manager.close()


def test_storage_floor_returns_507_before_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "inspect_model_profiles", all_available_snapshot)
    manager = JobManager(
        tmp_path,
        engine=NeverRunEngine(),
        start_worker=False,
        min_free_bytes=2**62,
    )
    try:
        with TestClient(create_app(manager=manager)) as client:
            response = post_audio(client)
        assert response.status_code == 507
        assert "free space" in response.json()["detail"]
        assert list(tmp_path.iterdir()) == []
    finally:
        manager.close()


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"use_gpu": False}, "use_gpu must be true"),
        ({"gpu_device": 1}, "gpu_device=0"),
    ],
)
def test_public_deployment_rejects_cpu_or_unexposed_gpu_before_upload(
    tmp_path, monkeypatch, override, expected
):
    monkeypatch.setenv("MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT", "1")
    monkeypatch.setattr(app_module, "inspect_model_profiles", all_available_snapshot)
    manager = JobManager(tmp_path, engine=NeverRunEngine(), start_worker=False)
    try:
        with TestClient(create_app(manager=manager)) as client:
            response = post_audio(client, **override)
        assert response.status_code == 422
        assert expected in response.json()["detail"]
        assert list(tmp_path.iterdir()) == []
    finally:
        manager.close()


def test_public_deployment_rejects_jobs_when_edge_security_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT", "1")
    monkeypatch.delenv("MUSIC_TO_MIDI_EDGE_AUTH", raising=False)
    monkeypatch.delenv("MUSIC_TO_MIDI_TLS_TERMINATED_AT_EDGE", raising=False)
    monkeypatch.setattr(app_module, "inspect_model_profiles", all_available_snapshot)
    manager = JobManager(tmp_path, engine=NeverRunEngine(), start_worker=False)
    try:
        with TestClient(create_app(manager=manager)) as client:
            response = post_audio(client)
        assert response.status_code == 503
        assert "public inference service is not ready" in response.json()["detail"]
        assert "inference worker is not alive" in response.json()["detail"]
        assert "MUSIC_TO_MIDI_EDGE_AUTH=basic" in response.json()["detail"]
        assert list(tmp_path.iterdir()) == []
    finally:
        manager.close()


def test_retention_byte_limit_removes_oldest_terminal_family(tmp_path):
    manager = JobManager(
        tmp_path,
        engine=NeverRunEngine(),
        start_worker=False,
        retention_days=0,
        retention_max_jobs=0,
        retention_max_bytes=0,
    )
    try:
        ids: list[str] = []
        for value in (b"12345678", b"abcdefgh"):
            job_id, job_dir = manager.reserve_job_dir()
            source = job_dir / "input" / "source.wav"
            source.write_bytes(value)
            manager.submit_reserved(
                job_id=job_id,
                kind="primary",
                source_path=source,
                original_filename="source.wav",
                options=inference_options(),
            )
            with manager._condition:
                record = manager._jobs[job_id]
                manager._pending.remove(job_id)
                record.status = JobStatus.SUCCEEDED
                record.finished_at = record.updated_at
                manager._persist_locked(record)
            ids.append(job_id)
            time.sleep(0.01)

        family_sizes = [manager._job_dir_size(job_id) for job_id in ids]
        manager.retention_max_bytes = max(family_sizes) + 1
        result = manager.prune_retained_jobs()
        assert result["deleted_jobs"] == 1
        assert not (tmp_path / ids[0]).exists()
        assert (tmp_path / ids[1]).is_dir()
        assert manager.health()["retention"]["max_bytes"] == max(family_sizes) + 1
    finally:
        manager.close()


def test_retention_count_limit_does_not_keep_one_oversized_family(tmp_path):
    manager = JobManager(
        tmp_path,
        engine=NeverRunEngine(),
        start_worker=False,
        retention_days=0,
        retention_max_jobs=0,
        retention_max_bytes=0,
    )
    try:
        parent_id, parent_dir = manager.reserve_job_dir()
        parent_source = parent_dir / "input" / "source.wav"
        parent_source.write_bytes(b"parent")
        manager.submit_reserved(
            job_id=parent_id,
            kind="primary",
            source_path=parent_source,
            original_filename="parent.wav",
            options=inference_options(),
        )
        child_id, child_dir = manager.reserve_job_dir()
        child_source = child_dir / "input" / "source.wav"
        child_source.write_bytes(b"child")
        manager.submit_reserved(
            job_id=child_id,
            kind="manual_midi",
            source_path=child_source,
            original_filename="child.wav",
            options={"route": "piano_transkun"},
            parent_job_id=parent_id,
        )
        with manager._condition:
            for job_id in (parent_id, child_id):
                record = manager._jobs[job_id]
                manager._pending.remove(job_id)
                record.status = JobStatus.SUCCEEDED
                record.finished_at = record.updated_at
                manager._persist_locked(record)

        manager.retention_max_jobs = 1
        result = manager.prune_retained_jobs()
        assert result["deleted_jobs"] == 2
        assert result["deleted_families"] == 1
        assert not (tmp_path / parent_id).exists()
        assert not (tmp_path / child_id).exists()
    finally:
        manager.close()


def test_backend_config_round_trips_production_capacity_limits(tmp_path):
    path = tmp_path / "backend.json"
    config = BackendServerConfig(
        max_upload_bytes=123456,
        max_queued_jobs=4,
        min_free_bytes=7890,
        retention_days=7,
        retention_max_jobs=50,
        retention_max_bytes=987654,
    )
    write_backend_config(path, config)
    loaded = load_backend_config(path)
    assert loaded.max_upload_bytes == 123456
    assert loaded.max_queued_jobs == 4
    assert loaded.min_free_bytes == 7890
    assert loaded.retention_max_bytes == 987654


def test_job_manager_capacity_exception_is_explicit(tmp_path):
    manager = JobManager(
        tmp_path,
        engine=NeverRunEngine(),
        start_worker=False,
        max_queued_jobs=1,
    )
    try:
        job_id, job_dir = manager.reserve_job_dir()
        source = job_dir / "input" / "source.wav"
        source.write_bytes(b"RIFF")
        manager.submit_reserved(
            job_id=job_id,
            kind="primary",
            source_path=source,
            original_filename="source.wav",
            options=inference_options(),
        )
        with pytest.raises(QueueCapacityError, match="queued=1, limit=1"):
            manager.require_submission_capacity()
    finally:
        manager.close()
