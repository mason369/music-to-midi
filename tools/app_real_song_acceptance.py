"""Exercise every desktop route through Qt controls with real audio.

The runner deliberately uses Qt's offscreen platform: it emits the same file
selection signal and clicks the same buttons as the desktop application, while
leaving the interactive Windows desktop untouched.  Model inference, Beat This
analysis, official MIDI writers, separation, and artifact validation are real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.main import _prepare_torch_runtime_before_pyqt

_prepare_torch_runtime_before_pyqt()

import soundfile as sf
from mido import MidiFile, merge_tracks, tick2second, tempo2bpm
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication, QDialog

from src.core.manual_midi import MANUAL_MIDI_ROUTES
from src.gui.main_window import MainWindow
from src.models.data_models import Config, ProcessingResult, TempoMode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_note_seconds(midi: MidiFile) -> float | None:
    seconds = 0.0
    tempo = 500_000
    for message in merge_tracks(midi.tracks):
        seconds += tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = int(message.tempo)
        elif message.type == "note_on" and int(message.velocity) > 0:
            return seconds
    return None


def _validate_midi(result: ProcessingResult, *, require_cc64: bool = False) -> dict[str, Any]:
    path = Path(result.midi_path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"generated MIDI is missing or empty: {path}")
    midi = MidiFile(path)
    if midi.type == 2:
        raise RuntimeError(
            f"desktop acceptance does not allow independent type-2 timelines: {path}"
        )
    note_on = sum(
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
    tempo_events: list[tuple[int, int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            if message.type == "set_tempo":
                tempo_events.append((track_index, absolute_tick, int(message.tempo)))
    tempo_by_tick: dict[int, int] = {}
    for _track_index, absolute_tick, tempo_us in tempo_events:
        previous = tempo_by_tick.get(absolute_tick)
        if previous is not None and previous != tempo_us:
            raise RuntimeError(
                "final MIDI has conflicting global tempos at one tick: "
                f"tick={absolute_tick}, tempos={[previous, tempo_us]}"
            )
        tempo_by_tick[absolute_tick] = tempo_us
    canonical_tempos = sorted(tempo_by_tick.items())
    meters = [
        (int(message.numerator), int(message.denominator))
        for track in midi.tracks
        for message in track
        if message.type == "time_signature"
    ]
    if note_on != int(result.total_notes):
        raise RuntimeError(
            f"result note count does not match final MIDI: {result.total_notes} != {note_on}"
        )
    if not tempo_events:
        raise RuntimeError(f"final MIDI has no set_tempo event: {path}")
    if require_cc64 and cc64 <= 0:
        raise RuntimeError(f"ByteDance result lost all CC64 events: {path}")

    beat_info = result.beat_info
    expected_meter = beat_info.time_signature if beat_info is not None else None
    if expected_meter is None and meters:
        raise RuntimeError(f"final MIDI fabricated a meter absent from Beat This: {meters}")
    if expected_meter is not None and (not meters or meters[0] != tuple(expected_meter)):
        raise RuntimeError(f"MIDI meter {meters[:1]} does not match Beat This {expected_meter}")
    expected_tempo_events = (
        len(beat_info.tempo_map) if beat_info is not None and len(beat_info.tempo_map) >= 2 else 1
    )
    # MuScriptor intentionally repeats the same canonical map on each note
    # track because MuseScore may ignore a conductor-only tempo.  Validate
    # unique global map points and reject conflicts instead of misclassifying
    # that byte-identical compatibility copy as an extra tempo change.
    if len(canonical_tempos) != expected_tempo_events:
        raise RuntimeError(
            "canonical tempo-map point count mismatch: "
            f"MIDI={len(canonical_tempos)} expected={expected_tempo_events}"
        )

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "tracks": len(midi.tracks),
        "note_on": note_on,
        "cc64": cc64,
        "tempo_bpm": [float(tempo2bpm(tempo_us)) for _tick, tempo_us in canonical_tempos],
        "tempo_map_points": [
            {"tick": tick, "bpm": float(tempo2bpm(tempo_us))} for tick, tempo_us in canonical_tempos
        ],
        "tempo_events_raw": [
            {
                "track": track_index,
                "tick": tick,
                "bpm": float(tempo2bpm(tempo_us)),
            }
            for track_index, tick, tempo_us in tempo_events
        ],
        "time_signatures": [list(item) for item in meters],
        "first_note_seconds": _first_note_seconds(midi),
        "beat_info": asdict(beat_info) if beat_info is not None else None,
    }


def _validate_separation(result: Any) -> list[dict[str, Any]]:
    validated = []
    for name, raw_path in sorted(result.separated_audio.items()):
        path = Path(raw_path).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"separated WAV is missing or empty: {name}={path}")
        info = sf.info(str(path))
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise RuntimeError(f"invalid separated WAV: {name}={path}")
        validated.append(
            {
                "name": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "frames": int(info.frames),
                "samplerate": int(info.samplerate),
                "channels": int(info.channels),
                "duration": float(info.duration),
            }
        )
    return validated


def _wait_for_worker(
    app: QApplication,
    worker: Any,
    *,
    result_signal_name: str,
    timeout_seconds: float,
) -> Any:
    state: dict[str, Any] = {}
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    getattr(worker, result_signal_name).connect(lambda result: state.setdefault("result", result))
    worker.error_occurred.connect(lambda message: state.setdefault("error", str(message)))
    worker.finished.connect(loop.quit)
    timer.start(max(1, int(timeout_seconds * 1000)))
    loop.exec()
    timed_out = timer.isActive() is False and "result" not in state and "error" not in state
    timer.stop()
    app.processEvents()
    if timed_out:
        # On success MainWindow owns the QThread and schedules deleteLater from
        # its finished handler.  Only touch the worker on a real timeout, while
        # its C++ object is still expected to be alive.
        try:
            is_running = worker.isRunning()
        except RuntimeError:
            is_running = False
        if is_running:
            worker.cancel()
            if not worker.wait(60_000):
                raise TimeoutError("desktop worker did not stop after cancellation")
        raise TimeoutError(f"desktop worker exceeded {timeout_seconds:.0f} seconds")
    if "error" in state:
        raise RuntimeError(state["error"])
    if "result" not in state:
        raise RuntimeError("desktop worker finished without a result or explicit error")
    return state["result"]


def _configure_window(window: MainWindow, case: dict[str, Any], output_dir: Path) -> None:
    panel = window.track_panel
    panel.set_processing_mode(case["mode"])
    panel.set_transcription_backend(case.get("backend", "yourmt3"))
    if "yourmt3_model" in case:
        panel.set_yourmt3_model(case["yourmt3_model"])
    if "muscriptor_model" in case:
        panel.set_muscriptor_model(case["muscriptor_model"])
    tempo_mode = case.get("tempo_mode", TempoMode.FIXED_AUTO.value)
    if tempo_mode == TempoMode.FIXED_MANUAL.value:
        panel.set_custom_bpm(float(case["custom_bpm"]))
    else:
        panel.set_tempo_mode(tempo_mode)
    window.output_dir_edit.setText(str(output_dir))


def _new_window(case: dict[str, Any], output_dir: Path) -> MainWindow:
    config = Config(
        language="zh_CN",
        output_dir=str(output_dir),
        use_gpu=True,
        gpu_device=0,
    )
    window = MainWindow(config)
    _configure_window(window, case, output_dir)
    return window


def _click_primary(
    app: QApplication,
    window: MainWindow,
    audio_path: Path,
    *,
    timeout_seconds: float,
) -> Any:
    window.dropzone.file_selected.emit(str(audio_path))
    app.processEvents()
    if not window.start_btn.isEnabled():
        raise RuntimeError("desktop Start button did not enable after file selection")
    window.start_btn.click()
    app.processEvents()
    worker = window.worker
    if worker is None or not worker.isRunning():
        raise RuntimeError("desktop Start button did not launch a worker")
    signal_name = (
        "separation_finished" if hasattr(worker, "separation_finished") else "processing_finished"
    )
    return _wait_for_worker(
        app,
        worker,
        result_signal_name=signal_name,
        timeout_seconds=timeout_seconds,
    )


def _click_manual_route(
    app: QApplication,
    window: MainWindow,
    route: str,
    *,
    timeout_seconds: float,
) -> ProcessingResult:
    mixer = window.audio_mixer
    if mixer is None:
        raise RuntimeError("six-stem result did not create the desktop audio timeline")
    backend = mixer._backends.get("piano")  # Acceptance-only inspection of the rendered row.
    if backend is None:
        raise RuntimeError("six-stem desktop timeline has no piano row")
    row = backend.row
    row.midi_enabled_checkbox.setChecked(True)
    route_index = row.midi_model_selector.findData(route)
    if route_index < 0:
        raise RuntimeError(f"desktop row does not expose manual MIDI route {route!r}")
    row.midi_model_selector.setCurrentIndex(route_index)
    app.processEvents()
    if not row.convert_midi_button.isEnabled():
        raise RuntimeError(f"desktop per-track Convert button is disabled for {route!r}")
    row.convert_midi_button.click()
    app.processEvents()
    worker = window.worker
    if worker is None or not worker.isRunning():
        raise RuntimeError(f"desktop per-track button did not launch {route!r}")
    return _wait_for_worker(
        app,
        worker,
        result_signal_name="processing_finished",
        timeout_seconds=timeout_seconds,
    )


def _close_window(app: QApplication, window: MainWindow) -> None:
    window.close()
    window.deleteLater()
    app.processEvents()


def run(args: argparse.Namespace) -> dict[str, Any]:
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
    if args.primary_cases:
        requested_names = [name.strip() for name in args.primary_cases.split(",") if name.strip()]
        available_names = {case["name"] for case in primary_cases}
        unknown_names = [name for name in requested_names if name not in available_names]
        if unknown_names:
            raise ValueError(f"unknown desktop primary cases: {unknown_names}")
        requested_set = set(requested_names)
        primary_cases = [case for case in primary_cases if case["name"] in requested_set]

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    results: list[dict[str, Any]] = []
    six_stem_window: MainWindow | None = None
    ui_contract: dict[str, Any] | None = None
    started_at = time.monotonic()

    with (
        mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None),
        mock.patch.object(QDialog, "exec", return_value=0),
    ):
        for index, case in enumerate(primary_cases, start=1):
            print(f"PRIMARY {index}/{len(primary_cases)} {case['name']}", flush=True)
            case_root = output_root / "primary" / case["name"]
            case_root.mkdir(parents=True, exist_ok=True)
            window = _new_window(case, case_root)
            audio = piano_audio if case.get("audio") == "piano" else mix_audio
            case_started = time.monotonic()
            result = _click_primary(
                app,
                window,
                audio,
                timeout_seconds=args.timeout,
            )
            if case["mode"] in {"vocal_split", "six_stem_split"}:
                artifacts = _validate_separation(result)
                record = {
                    "case": case["name"],
                    "kind": "separation",
                    "processing_time": float(result.processing_time),
                    "wall_time": time.monotonic() - case_started,
                    "artifacts": artifacts,
                }
            else:
                midi = _validate_midi(
                    result,
                    require_cc64=case["mode"] == "piano_bytedance_pedal",
                )
                if case["name"].startswith("smart-muscriptor"):
                    first_note = midi["first_note_seconds"]
                    if first_note is None or first_note >= 1.0:
                        raise RuntimeError(
                            f"MuScriptor onset unexpectedly shifted: {first_note} seconds"
                        )
                record = {
                    "case": case["name"],
                    "kind": "midi",
                    "processing_time": float(result.processing_time),
                    "wall_time": time.monotonic() - case_started,
                    "artifact": midi,
                }
            results.append(record)
            print(
                f"  PASS {case['name']} wall={record['wall_time']:.1f}s",
                flush=True,
            )

            if ui_contract is None:
                panel = window.track_panel
                ui_contract = {
                    "tempo_label": panel._tempo_label.text(),
                    "tempo_mode_tooltip": panel.tempo_mode_combo.toolTip(),
                    "offscreen_platform": os.environ.get("QT_QPA_PLATFORM"),
                }
                if "Beat This final0" not in " ".join(ui_contract.values()):
                    raise RuntimeError(
                        "desktop tempo controls do not identify official Beat This final0"
                    )
            if case["name"] == "six-stem-split":
                six_stem_window = window
            else:
                _close_window(app, window)

        if args.skip_manual:
            if six_stem_window is not None:
                _close_window(app, six_stem_window)
            report = {
                "runner": "Qt offscreen real-button acceptance",
                "mix_audio": str(mix_audio),
                "piano_audio": str(piano_audio),
                "primary_cases": len(primary_cases),
                "manual_routes": 0,
                "cases_validated": len(results),
                "wall_time": time.monotonic() - started_at,
                "ui_contract": ui_contract,
                "results": results,
            }
            report_path = output_root / "app-real-song-matrix.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"COMPLETE {len(results)} cases report={report_path}", flush=True)
            return report

        if six_stem_window is None:
            raise RuntimeError("six-stem desktop result was not retained")
        mixer = six_stem_window.audio_mixer
        if mixer is None:
            raise RuntimeError("six-stem desktop result has no timeline")
        piano_row = mixer._backends["piano"].row
        exposed_routes = [
            str(piano_row.midi_model_selector.itemData(index))
            for index in range(1, piano_row.midi_model_selector.count())
        ]
        if tuple(exposed_routes) != tuple(MANUAL_MIDI_ROUTES):
            raise RuntimeError(
                f"desktop manual route order differs from shared contract: {exposed_routes}"
            )
        assert ui_contract is not None
        ui_contract["manual_routes"] = exposed_routes

        for index, route in enumerate(MANUAL_MIDI_ROUTES, start=1):
            print(f"MANUAL {index}/{len(MANUAL_MIDI_ROUTES)} {route}", flush=True)
            route_started = time.monotonic()
            result = _click_manual_route(
                app,
                six_stem_window,
                route,
                timeout_seconds=args.timeout,
            )
            midi = _validate_midi(
                result,
                require_cc64=route == "piano_bytedance_pedal",
            )
            if route.startswith("muscriptor"):
                first_note = midi["first_note_seconds"]
                if first_note is None or first_note >= 1.0:
                    raise RuntimeError(
                        f"manual MuScriptor onset unexpectedly shifted: {first_note} seconds"
                    )
            results.append(
                {
                    "case": f"manual-piano-{route}",
                    "kind": "midi",
                    "processing_time": float(result.processing_time),
                    "wall_time": time.monotonic() - route_started,
                    "artifact": midi,
                }
            )
            print(
                f"  PASS {route} notes={midi['note_on']} tracks={midi['tracks']}",
                flush=True,
            )

        _close_window(app, six_stem_window)

    report = {
        "runner": "Qt offscreen real-button acceptance",
        "mix_audio": str(mix_audio),
        "piano_audio": str(piano_audio),
        "primary_cases": len(primary_cases),
        "manual_routes": len(MANUAL_MIDI_ROUTES),
        "cases_validated": len(results),
        "wall_time": time.monotonic() - started_at,
        "ui_contract": ui_contract,
        "results": results,
    }
    report_path = output_root / "app-real-song-matrix.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"COMPLETE {len(results)} cases report={report_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix-audio", required=True)
    parser.add_argument("--piano-audio", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--primary-cases")
    parser.add_argument("--skip-manual", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
