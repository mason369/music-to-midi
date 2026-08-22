"""Run every standalone Web route against real audio and validate downloaded artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf
from mido import MidiFile

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def _require_response(response: httpx.Response, expected: set[int]) -> Any:
    if response.status_code not in expected:
        raise RuntimeError(
            f"HTTP {response.status_code} {response.request.method} {response.request.url}: "
            f"{response.text[:2000]}"
        )
    if response.status_code == 204:
        return None
    return response.json()


def _wait_for_terminal(
    client: httpx.Client,
    api_base: str,
    job: dict[str, Any],
    *,
    timeout: float = 600.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_signature: tuple[str, str] | None = None
    while job["status"] not in TERMINAL_STATUSES:
        signature = (job["status"], str(job.get("progress", {}).get("stage", "")))
        if signature != last_signature:
            print(
                f"  {job['id'][:8]} status={signature[0]} stage={signature[1]}",
                flush=True,
            )
            last_signature = signature
        if time.monotonic() >= deadline:
            raise TimeoutError(f"job did not reach a terminal state: {job['id']}")
        time.sleep(0.5)
        job = _require_response(
            client.get(f"{api_base}/api/v1/jobs/{job['id']}"),
            {200},
        )
    if job["status"] != "succeeded":
        raise RuntimeError(
            f"job {job['id']} ended as {job['status']}: {job.get('error') or job['progress']}"
        )
    return job


def _validate_artifacts(
    client: httpx.Client,
    api_base: str,
    output_root: Path,
    case_name: str,
    job: dict[str, Any],
) -> list[dict[str, Any]]:
    validated = []
    midi_stats = None
    for artifact in job["artifacts"]:
        response = client.get(f"{api_base}{artifact['download_url']}")
        if response.status_code != 200:
            raise RuntimeError(
                f"artifact download failed: {artifact['download_url']} HTTP {response.status_code}"
            )
        data = response.content
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != int(artifact["size"]):
            raise RuntimeError(f"artifact size mismatch: {artifact['name']}")
        if digest != artifact["sha256"]:
            raise RuntimeError(f"artifact SHA-256 mismatch: {artifact['name']}")
        target = (
            output_root
            / "artifacts"
            / (
                f"{_safe_name(case_name)}--{_safe_name(artifact['id'])}--{_safe_name(artifact['name'])}"
            )
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        item = {
            "id": artifact["id"],
            "kind": artifact["kind"],
            "path": str(target),
            "bytes": len(data),
            "sha256": digest,
        }
        if artifact["kind"] == "midi":
            midi = MidiFile(file=BytesIO(data))
            notes = sum(
                1
                for track in midi.tracks
                for message in track
                if message.type == "note_on" and int(message.velocity) > 0
            )
            cc64 = sum(
                1
                for track in midi.tracks
                for message in track
                if message.type == "control_change" and int(message.control) == 64
            )
            item.update({"tracks": len(midi.tracks), "note_on": notes, "cc64": cc64})
            midi_stats = item
        elif artifact["kind"] == "audio_track":
            info = sf.info(str(target))
            if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
                raise RuntimeError(f"invalid WAV artifact: {target}")
            item.update(
                {
                    "frames": int(info.frames),
                    "samplerate": int(info.samplerate),
                    "channels": int(info.channels),
                    "duration": float(info.duration),
                }
            )
        validated.append(item)

    result = job.get("result") or {}
    if midi_stats is not None:
        if int(result.get("total_notes", -1)) != midi_stats["note_on"]:
            raise RuntimeError(f"API note count does not match final MIDI: {case_name}")
        if int(result.get("track_count", -1)) != midi_stats["tracks"]:
            raise RuntimeError(f"API track count does not match final MIDI: {case_name}")
        warnings = set(result.get("quality_warnings") or [])
        if midi_stats["note_on"] == 0 and "empty_midi" not in warnings:
            raise RuntimeError(f"empty MIDI did not expose empty_midi warning: {case_name}")
        if midi_stats["note_on"] > 0 and "empty_midi" in warnings:
            raise RuntimeError(f"non-empty MIDI exposed an empty_midi warning: {case_name}")
    return validated


def _primary_options(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "processing_mode": case["mode"],
        "transcription_backend": case.get("backend", "yourmt3"),
        "yourmt3_model": case.get("yourmt3_model", "yptf_moe_multi_nops"),
        "muscriptor_model": case.get("muscriptor_model", "large"),
        "muscriptor_instruments": [],
        "midi_track_mode": "multi_track",
        "tempo_mode": case.get("tempo_mode", "fixed_auto"),
        "custom_bpm": case.get("custom_bpm"),
        "use_gpu": True,
        "gpu_device": 0,
        "language": "zh_CN",
    }


def _submit_primary(
    client: httpx.Client,
    api_base: str,
    audio_path: Path,
    options: dict[str, Any],
) -> dict[str, Any]:
    with audio_path.open("rb") as stream:
        return _require_response(
            client.post(
                f"{api_base}/api/v1/jobs",
                files={"audio": (audio_path.name, stream, "audio/wav")},
                data={"options": json.dumps(options, ensure_ascii=False)},
            ),
            {202},
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    api_base = args.api_base.rstrip("/")
    mix_audio = Path(args.mix_audio).resolve()
    piano_audio = Path(args.piano_audio).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for path in (mix_audio, piano_audio):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"real-song input is missing or empty: {path}")

    primary_cases = [
        {"name": "smart-yourmt3-ymt3-plus", "mode": "smart", "yourmt3_model": "ymt3_plus"},
        {
            "name": "smart-yourmt3-yptf-single-nops",
            "mode": "smart",
            "yourmt3_model": "yptf_single_nops",
        },
        {
            "name": "smart-yourmt3-yptf-multi-ps-adaptive",
            "mode": "smart",
            "yourmt3_model": "yptf_multi_ps",
            "tempo_mode": "adaptive",
        },
        {
            "name": "smart-yourmt3-yptf-moe-multi-nops",
            "mode": "smart",
            "yourmt3_model": "yptf_moe_multi_nops",
        },
        {
            "name": "smart-yourmt3-yptf-moe-multi-ps-manual-bpm",
            "mode": "smart",
            "yourmt3_model": "yptf_moe_multi_ps",
            "tempo_mode": "fixed_manual",
            "custom_bpm": 123.4,
        },
        {"name": "smart-miros", "mode": "smart", "backend": "miros"},
        {
            "name": "smart-muscriptor-large",
            "mode": "smart",
            "backend": "muscriptor",
            "muscriptor_model": "large",
        },
        {
            "name": "smart-muscriptor-medium",
            "mode": "smart",
            "backend": "muscriptor",
            "muscriptor_model": "medium",
        },
        {
            "name": "smart-muscriptor-small",
            "mode": "smart",
            "backend": "muscriptor",
            "muscriptor_model": "small",
        },
        {"name": "piano-transkun", "mode": "piano_transkun", "audio": "piano"},
        {
            "name": "piano-transkun-v2-aug",
            "mode": "piano_transkun_v2_aug",
            "audio": "piano",
        },
        {"name": "piano-aria-amt", "mode": "piano_aria_amt", "audio": "piano"},
        {
            "name": "piano-bytedance-pedal",
            "mode": "piano_bytedance_pedal",
            "audio": "piano",
        },
        {"name": "vocal-split", "mode": "vocal_split"},
        {"name": "six-stem-split", "mode": "six_stem_split"},
    ]
    results: list[dict[str, Any]] = []
    six_stem_job = None

    with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        health = _require_response(client.get(f"{api_base}/api/v1/health"), {200})
        capabilities = _require_response(client.get(f"{api_base}/api/v1/capabilities"), {200})
        if health.get("api_version") != "2.0" or capabilities.get("api_version") != "2.0":
            raise RuntimeError("acceptance runner requires Web API 2.0")

        for index, case in enumerate(primary_cases, start=1):
            print(f"PRIMARY {index}/{len(primary_cases)} {case['name']}", flush=True)
            audio = piano_audio if case.get("audio") == "piano" else mix_audio
            submitted = _submit_primary(client, api_base, audio, _primary_options(case))
            completed = _wait_for_terminal(client, api_base, submitted)
            artifacts = _validate_artifacts(client, api_base, output_root, case["name"], completed)
            record = {
                "case": case["name"],
                "job_id": completed["id"],
                "status": completed["status"],
                "result": completed["result"],
                "artifacts": artifacts,
            }
            results.append(record)
            if case["name"] == "six-stem-split":
                six_stem_job = completed
            print(
                f"  PASS {case['name']} time={completed['result']['processing_time']:.1f}s",
                flush=True,
            )

        if six_stem_job is None:
            raise RuntimeError("six-stem parent job was not produced")
        manual_routes = [item["id"] for item in capabilities["manual_midi_routes"]]
        if len(manual_routes) != 13:
            raise RuntimeError(f"expected 13 manual routes, got {len(manual_routes)}")
        for index, route in enumerate(manual_routes, start=1):
            print(f"MANUAL {index}/{len(manual_routes)} {route}", flush=True)
            options = {
                "route": route,
                "muscriptor_instruments": [],
                "tempo_mode": "fixed_auto",
                "custom_bpm": None,
                "use_gpu": True,
                "gpu_device": 0,
                "language": "zh_CN",
            }
            submitted = _require_response(
                client.post(
                    f"{api_base}/api/v1/jobs/{six_stem_job['id']}/tracks/piano/midi",
                    json=options,
                ),
                {202},
            )
            completed = _wait_for_terminal(client, api_base, submitted)
            case_name = f"manual-piano-{route}"
            artifacts = _validate_artifacts(client, api_base, output_root, case_name, completed)
            results.append(
                {
                    "case": case_name,
                    "job_id": completed["id"],
                    "parent_job_id": completed["parent_job_id"],
                    "status": completed["status"],
                    "result": completed["result"],
                    "artifacts": artifacts,
                }
            )
            midi = next(item for item in artifacts if item["kind"] == "midi")
            print(
                f"  PASS {route} notes={midi['note_on']} tracks={midi['tracks']} warnings={completed['result']['quality_warnings']}",
                flush=True,
            )

        final_health = _require_response(client.get(f"{api_base}/api/v1/health"), {200})

    report = {
        "api_base": api_base,
        "mix_audio": str(mix_audio),
        "piano_audio": str(piano_audio),
        "primary_cases": len(primary_cases),
        "manual_routes": 13,
        "jobs_validated": len(results),
        "health": final_health,
        "results": results,
    }
    report_path = output_root / "web-real-song-matrix.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"COMPLETE {len(results)} cases report={report_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--mix-audio", required=True)
    parser.add_argument("--piano-audio", required=True)
    parser.add_argument("--output-root", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
