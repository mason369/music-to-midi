"""Adapter from HTTP jobs to the existing, GUI-independent inference core."""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mido

from src.core.manual_midi import build_manual_midi_config
from src.core.midi_quantization import quantize_midi_notes
from src.models.data_models import (
    Config,
    MuscriptorProcessingChain,
    ProcessingMode,
    ProcessingProgress,
    ProcessingResult,
    TempoMode,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactSpec:
    id: str
    kind: str
    path: Path
    track_id: str | None = None

    @property
    def media_type(self) -> str:
        explicit = {
            ".mid": "audio/midi",
            ".midi": "audio/midi",
            ".wav": "audio/wav",
            ".json": "application/json",
            ".csv": "text/csv; charset=utf-8",
            ".zip": "application/zip",
        }
        return explicit.get(self.path.suffix.lower()) or (
            mimetypes.guess_type(self.path.name)[0] or "application/octet-stream"
        )


@dataclass(frozen=True)
class ExecutionResult:
    result: dict
    artifacts: tuple[ArtifactSpec, ...]


ProgressCallback = Callable[[ProcessingProgress], None]
ProcessorCallback = Callable[[object | None], None]


class InferenceEngine:
    """Execute one job without importing any frontend package."""

    @staticmethod
    def _primary_config(options: dict) -> Config:
        config = Config()
        config.language = str(options["language"])
        config.processing_mode = str(options["processing_mode"])
        config.transcription_backend = str(options["transcription_backend"])
        config.multi_instrument_model = str(options["transcription_backend"])
        config.yourmt3_model = str(options["yourmt3_model"])
        config.muscriptor_model = str(options["muscriptor_model"])
        config.muscriptor_processing_chain = str(
            options.get("muscriptor_processing_chain") or MuscriptorProcessingChain.OFFICIAL.value
        )
        config.muscriptor_instruments = list(options.get("muscriptor_instruments") or [])
        config.midi_track_mode = str(options["midi_track_mode"])
        config.tempo_mode = str(options.get("tempo_mode") or TempoMode.FIXED_AUTO.value)
        config.custom_bpm = options.get("custom_bpm")
        # HTTP quantization is an explicit post-write stage below. Keep the
        # legacy NoteEvent generator disabled so no route can quantize twice.
        config.quantize_notes = False
        config.use_gpu = bool(options["use_gpu"])
        config.gpu_device = int(options["gpu_device"])
        config.vocal_split_merge_midi = False
        config.save_separated_tracks = True
        config.validate()
        return config

    @staticmethod
    def _manual_config(options: dict) -> Config:
        base = Config(
            language=str(options["language"]),
            use_gpu=bool(options["use_gpu"]),
            gpu_device=int(options["gpu_device"]),
            tempo_mode=str(options.get("tempo_mode") or TempoMode.FIXED_AUTO.value),
            custom_bpm=options.get("custom_bpm"),
            quantize_notes=False,
            muscriptor_processing_chain=str(
                options.get("muscriptor_processing_chain")
                or MuscriptorProcessingChain.OFFICIAL.value
            ),
        )
        return build_manual_midi_config(
            base,
            str(options["route"]),
            muscriptor_instruments=list(options.get("muscriptor_instruments") or []),
        )

    @staticmethod
    def _beat_payload(result: ProcessingResult) -> dict | None:
        beat = result.beat_info
        if beat is None:
            return None
        return {
            "bpm": float(beat.bpm),
            "bpm_display": beat.bpm_display,
            "source_bpm": beat.source_bpm,
            "fixed_tempo_reliable": beat.fixed_tempo_reliable,
            "tempo_warning": beat.tempo_warning,
            "time_signature": list(beat.time_signature) if beat.time_signature else None,
            "is_variable_tempo": bool(beat.is_variable_tempo),
            "tempo_map": [[float(sec), float(bpm)] for sec, bpm in beat.tempo_map],
            "beat_times": [float(value) for value in beat.beat_times],
            "downbeats": [float(value) for value in (beat.downbeats or [])],
        }

    @classmethod
    def _processing_payload(
        cls,
        result: ProcessingResult,
        config: Config,
        midi_path: Path,
        timing_result: dict | None = None,
    ) -> dict:
        backend = result.transcription_backend
        if not backend:
            backend = (
                config.transcription_backend
                if config.processing_mode == ProcessingMode.SMART.value
                else config.processing_mode
            )
        try:
            midi = mido.MidiFile(str(midi_path))
        except Exception as exc:
            raise RuntimeError(f"produced MIDI could not be parsed: {midi_path}: {exc}") from exc
        total_notes = sum(
            1
            for track in midi.tracks
            for message in track
            if message.type == "note_on" and int(message.velocity) > 0
        )
        quality_warnings = []
        if total_notes == 0:
            quality_warnings.append("empty_midi")
        if int(result.total_notes) != total_notes:
            quality_warnings.append("note_count_corrected_from_file")
        beat = cls._beat_payload(result)
        if beat is not None and timing_result and timing_result.get("midi_export"):
            from src.core.midi_timing import MidiTimeline

            timeline = MidiTimeline.from_midi(midi)
            signature = next(
                (
                    (m.numerator, m.denominator)
                    for tr in midi.tracks
                    for m in tr
                    if m.type == "time_signature"
                ),
                None,
            )
            reference = timing_result["reference_bpm"]
            target = timing_result["target_bpm"]
            from src.core.midi_tempo import read_muscriptor_bar_offset_seconds
            from src.core.midi_timing import reference_timelines

            reference_timeline = reference_timelines(midi, reference)[0]
            offset = (
                read_muscriptor_bar_offset_seconds(midi)
                * reference_timeline.sections[0][2]
                / timeline.sections[0][2]
            )
            last_tick = max(sum(m.time for m in track) for track in midi.tracks)
            beats = [
                reference_timeline.seconds_at_tick(tick) - offset
                for tick in range(0, last_tick + 1, midi.ticks_per_beat)
            ]
            bar_ticks = (
                round(signature[0] * 4 / signature[1] * midi.ticks_per_beat) if signature else 0
            )
            downbeats = (
                [
                    reference_timeline.seconds_at_tick(tick) - offset
                    for tick in range(0, last_tick + 1, max(1, bar_ticks))
                ]
                if bar_ticks
                else []
            )
            beat.update(
                detected_bpm=beat.get("source_bpm") or beat["bpm"],
                bpm=target,
                source_bpm=reference if target != reference else None,
                bpm_display=f"{target:.1f}",
                time_signature=list(signature) if signature else None,
                is_variable_tempo=len(timeline.sections) > 1,
                beat_times=[s for s in beats if s >= 0],
                downbeats=[s for s in downbeats if s >= 0],
                tempo_map=(
                    [[s, 60_000_000 / tempo] for _tick, s, tempo in timeline.sections]
                    if len(timeline.sections) > 1
                    else []
                ),
            )
        return {
            "mode": config.processing_mode,
            "processing_time": float(result.processing_time),
            "total_notes": total_notes,
            "track_count": len(midi.tracks),
            "transcription_backend": backend,
            "selected_instruments": list(result.selected_instruments),
            "detected_instruments": list(result.detected_instruments),
            "beat": beat,
            "quality_warnings": quality_warnings,
        }

    @staticmethod
    def _require_file(path_value: str | Path, label: str) -> Path:
        path = Path(path_value).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"{label} was not produced or is empty: {path}")
        return path

    def run(
        self,
        *,
        kind: str,
        source_path: Path,
        output_dir: Path,
        options: dict,
        progress_callback: ProgressCallback,
        processor_callback: ProcessorCallback,
        track_id: str | None = None,
        tempo_source_path: Path | None = None,
    ) -> ExecutionResult:
        from src.web_api.inference_process import (
            XpuInferenceProcess,
            should_isolate_xpu_inference,
        )

        if should_isolate_xpu_inference(options):
            isolated = XpuInferenceProcess()
            processor_callback(isolated)
            try:
                return isolated.run(
                    kind=kind,
                    source_path=source_path,
                    output_dir=output_dir,
                    options=options,
                    progress_callback=progress_callback,
                    track_id=track_id,
                    tempo_source_path=tempo_source_path,
                )
            finally:
                processor_callback(None)

        return self._run_direct(
            kind=kind,
            source_path=source_path,
            output_dir=output_dir,
            options=options,
            progress_callback=progress_callback,
            processor_callback=processor_callback,
            track_id=track_id,
            tempo_source_path=tempo_source_path,
        )

    @staticmethod
    def _apply_requested_quantization(
        midi_path: Path,
        options: dict,
        *,
        reference_bpm=None,
        target_bpm=None,
    ) -> dict | None:
        from src.core.midi_export_options import MidiExportOptions, apply_midi_export_options

        export = options.get("midi_export")
        if export:
            from src.core.midi_tempo import validated_midi_bpm
            from src.core.midi_timing import MidiTimeline, write_project_tempo_map

            reference_bpm = (
                reference_bpm
                or 60_000_000 / MidiTimeline.from_midi(mido.MidiFile(midi_path)).sections[0][2]
            )
            target_bpm = target_bpm or reference_bpm
            rate = target_bpm / reference_bpm
            new_reference = validated_midi_bpm(
                export.get("bpm") or reference_bpm * float(export.get("beat_scale", 1)),
                "corrected reference",
            )
            validated_midi_bpm(new_reference * rate, "corrected project")
            if rate != 1:
                write_project_tempo_map(midi_path, midi_path, reference_bpm, reference_bpm)
            correction = dict(export)
            correction.update(fill_gaps=0.0, trim_leading_bars=False)
            apply_midi_export_options(midi_path, midi_path, MidiExportOptions(**correction))
        result = None
        if bool(options.get("quantize_notes", False)):
            report = quantize_midi_notes(
                midi_path,
                str(options.get("quantize_grid", "1/32")),
                label="HTTP MIDI note quantization",
                strength=float(options.get("quantize_strength", 1.0)),
                triplet=bool(options.get("quantize_triplet", False)),
                automatic=bool(options.get("quantize_auto", False)),
            )
            result = {
                "enabled": True,
                "grid": report.grid,
                "grid_ticks": report.grid_ticks,
                "paired_note_count": report.paired_note_count,
            }
            if report.strength != 1 or report.triplet or report.automatic:
                result.update(
                    strength=report.strength, triplet=report.triplet, automatic=report.automatic
                )
            if report.grid_step_ticks != report.grid_ticks:
                result["grid_step_ticks"] = report.grid_step_ticks
        if export:
            apply_midi_export_options(
                midi_path,
                midi_path,
                MidiExportOptions(
                    fill_gaps=float(export.get("fill_gaps", 0)),
                    trim_leading_bars=bool(export.get("trim_leading_bars", False)),
                ),
            )
            if rate != 1:
                write_project_tempo_map(
                    midi_path, midi_path, new_reference, new_reference * rate
                )
            result = dict(
                result or {"enabled": False},
                midi_export=dict(export),
                reference_bpm=new_reference,
                target_bpm=new_reference * rate,
            )
        return result

    def _run_direct(
        self,
        *,
        kind: str,
        source_path: Path,
        output_dir: Path,
        options: dict,
        progress_callback: ProgressCallback,
        processor_callback: ProcessorCallback,
        track_id: str | None = None,
        tempo_source_path: Path | None = None,
    ) -> ExecutionResult:
        try:
            if kind == "primary":
                return self._run_primary(
                    source_path,
                    output_dir,
                    options,
                    progress_callback,
                    processor_callback,
                )
            if kind == "manual_midi":
                return self._run_manual(
                    source_path,
                    output_dir,
                    options,
                    progress_callback,
                    processor_callback,
                    track_id,
                    tempo_source_path,
                )
            raise RuntimeError(f"unsupported inference job kind: {kind!r}")
        finally:
            processor_callback(None)
            try:
                from src.utils.gpu_utils import clear_gpu_memory

                clear_gpu_memory()
            except Exception as exc:
                logger.error("GPU cleanup failed after HTTP inference job: %s", exc)

    def _run_primary(
        self,
        source_path: Path,
        output_dir: Path,
        options: dict,
        progress_callback: ProgressCallback,
        processor_callback: ProcessorCallback,
    ) -> ExecutionResult:
        config = self._primary_config(options)
        if config.processing_mode in {
            ProcessingMode.VOCAL_SPLIT.value,
            ProcessingMode.SIX_STEM_SPLIT.value,
        }:
            from src.core.separation_service import AudioSeparationService

            service = AudioSeparationService(config, progress_callback=progress_callback)
            processor_callback(service)
            result = service.process(source_path, output_dir)
            artifacts = []
            tracks = []
            for index, (name, path_value) in enumerate(result.separated_audio.items()):
                path = self._require_file(path_value, f"separated track {name}")
                artifact_id = f"track-{index + 1}-{name}"
                artifacts.append(
                    ArtifactSpec(
                        id=artifact_id,
                        kind="audio_track",
                        path=path,
                        track_id=name,
                    )
                )
                tracks.append(
                    {
                        "id": name,
                        "name": name,
                        "artifact_id": artifact_id,
                        "size": path.stat().st_size,
                    }
                )
            return ExecutionResult(
                result={
                    "mode": result.mode,
                    "processing_time": float(result.processing_time),
                    "track_count": len(tracks),
                    "tracks": tracks,
                    "manual_midi_required": True,
                },
                artifacts=tuple(artifacts),
            )

        from src.core.pipeline import MusicToMidiPipeline

        pipeline = MusicToMidiPipeline(config)
        processor_callback(pipeline)
        result = pipeline.process(str(source_path), str(output_dir), progress_callback)
        if not isinstance(result, ProcessingResult):
            raise TypeError(f"pipeline returned unsupported result type: {type(result)!r}")
        midi_path = self._require_file(result.midi_path, "MIDI output")
        quantization = self._apply_requested_quantization(
            midi_path,
            options,
            reference_bpm=(
                (result.beat_info.source_bpm or result.beat_info.bpm)
                if result.beat_info
                else None
            ),
            target_bpm=result.beat_info.bpm if result.beat_info else None,
        )
        payload = self._processing_payload(result, config, midi_path, quantization)
        if quantization is not None:
            payload["quantization"] = quantization
        return ExecutionResult(
            payload,
            (ArtifactSpec("midi", "midi", midi_path),),
        )

    def _run_manual(
        self,
        source_path: Path,
        output_dir: Path,
        options: dict,
        progress_callback: ProgressCallback,
        processor_callback: ProcessorCallback,
        track_id: str | None,
        tempo_source_path: Path | None,
    ) -> ExecutionResult:
        from src.core.pipeline import MusicToMidiPipeline

        config = self._manual_config(options)
        pipeline = MusicToMidiPipeline(config)
        processor_callback(pipeline)
        resolved_tempo_source = (tempo_source_path or source_path).resolve()
        result = pipeline.process(
            str(source_path),
            str(output_dir),
            progress_callback,
            tempo_audio_path=str(resolved_tempo_source),
        )
        if not isinstance(result, ProcessingResult):
            raise TypeError(f"manual MIDI route returned unsupported type: {type(result)!r}")
        midi_path = self._require_file(result.midi_path, "manual MIDI output")
        quantization = self._apply_requested_quantization(
            midi_path,
            options,
            reference_bpm=(
                (result.beat_info.source_bpm or result.beat_info.bpm)
                if result.beat_info
                else None
            ),
            target_bpm=result.beat_info.bpm if result.beat_info else None,
        )
        payload = self._processing_payload(result, config, midi_path, quantization)
        payload.update({"route": options["route"], "source_track_id": track_id})
        if quantization is not None:
            payload["quantization"] = quantization
        return ExecutionResult(
            payload,
            (ArtifactSpec("midi", "midi", midi_path, track_id=track_id),),
        )
