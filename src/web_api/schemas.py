"""Strict public request/response schemas for the inference API."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.models.data_models import (
    MAX_TEMPO_BPM,
    MIN_TEMPO_BPM,
    MidiTrackMode,
    MultiInstrumentModel,
    MuscriptorModel,
    MuscriptorProcessingChain,
    ProcessingMode,
    TempoMode,
    YourMT3Model,
)

MidiQuantizeGrid = Literal["1/4", "1/8", "1/16", "1/32", "1/64"]


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


class InferenceOptions(BaseModel):
    """All product-level controls required to reproduce one primary job."""

    model_config = ConfigDict(extra="forbid")

    processing_mode: str = ProcessingMode.SMART.value
    transcription_backend: str = MultiInstrumentModel.YOURMT3.value
    yourmt3_model: str = YourMT3Model.YPTF_MOE_MULTI_NOPS.value
    muscriptor_model: str = MuscriptorModel.LARGE.value
    muscriptor_processing_chain: str = MuscriptorProcessingChain.OFFICIAL.value
    muscriptor_instruments: list[str] = Field(default_factory=list)
    midi_track_mode: str = MidiTrackMode.MULTI_TRACK.value
    tempo_mode: str = TempoMode.FIXED_AUTO.value
    custom_bpm: float | None = None
    quantize_notes: bool = False
    quantize_grid: MidiQuantizeGrid = "1/32"
    use_gpu: bool = True
    gpu_device: int = Field(default=0, ge=0)
    language: Literal["zh_CN", "en_US"] = "zh_CN"

    @field_validator("processing_mode")
    @classmethod
    def validate_processing_mode(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        valid = {mode.value for mode in ProcessingMode if mode is not ProcessingMode.PIANO}
        if normalized not in valid:
            raise ValueError(f"unsupported processing mode: {value!r}")
        return normalized

    @field_validator("transcription_backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        valid = {model.value for model in MultiInstrumentModel}
        if normalized not in valid:
            raise ValueError(f"unsupported transcription backend: {value!r}")
        return normalized

    @field_validator("yourmt3_model")
    @classmethod
    def validate_yourmt3_model(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        valid = {model.value for model in YourMT3Model if model is not YourMT3Model.LEGACY_MC13}
        if normalized not in valid:
            raise ValueError(f"unsupported YourMT3 model: {value!r}")
        return normalized

    @field_validator("muscriptor_model")
    @classmethod
    def validate_muscriptor_model(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        valid = {model.value for model in MuscriptorModel}
        if normalized not in valid:
            raise ValueError(f"unsupported MuScriptor model: {value!r}")
        return normalized

    @field_validator("midi_track_mode")
    @classmethod
    def validate_track_mode(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        valid = {mode.value for mode in MidiTrackMode}
        if normalized not in valid:
            raise ValueError(f"unsupported MIDI track mode: {value!r}")
        return normalized

    @field_validator("muscriptor_processing_chain")
    @classmethod
    def validate_muscriptor_processing_chain(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        valid = {chain.value for chain in MuscriptorProcessingChain}
        if normalized not in valid:
            raise ValueError(f"unsupported MuScriptor processing chain: {value!r}")
        return normalized

    @field_validator("tempo_mode")
    @classmethod
    def validate_tempo_mode(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        valid = {mode.value for mode in TempoMode}
        if normalized not in valid:
            raise ValueError(f"unsupported tempo mode: {value!r}")
        return normalized

    @field_validator("custom_bpm")
    @classmethod
    def validate_custom_bpm(cls, value: float | None) -> float | None:
        if value is None:
            return None
        normalized = float(value)
        if not MIN_TEMPO_BPM <= normalized <= MAX_TEMPO_BPM:
            raise ValueError(f"custom BPM must be between {MIN_TEMPO_BPM:g} and {MAX_TEMPO_BPM:g}")
        return normalized

    @model_validator(mode="after")
    def validate_tempo_contract(self):
        if "tempo_mode" not in self.model_fields_set and self.custom_bpm is not None:
            self.tempo_mode = TempoMode.FIXED_MANUAL.value
        if self.tempo_mode == TempoMode.FIXED_MANUAL.value and self.custom_bpm is None:
            raise ValueError("manual tempo mode requires custom_bpm")
        if self.tempo_mode != TempoMode.FIXED_MANUAL.value and self.custom_bpm is not None:
            raise ValueError("custom_bpm is only valid in manual tempo mode")
        return self


class ManualMidiOptions(BaseModel):
    """Explicit route selection for one separated or locally added track."""

    model_config = ConfigDict(extra="forbid")

    route: str
    muscriptor_instruments: list[str] = Field(default_factory=list)
    muscriptor_processing_chain: str = MuscriptorProcessingChain.OFFICIAL.value
    tempo_mode: str = TempoMode.FIXED_AUTO.value
    custom_bpm: float | None = None
    quantize_notes: bool = False
    quantize_grid: MidiQuantizeGrid = "1/32"
    use_gpu: bool = True
    gpu_device: int = Field(default=0, ge=0)
    language: Literal["zh_CN", "en_US"] = "zh_CN"

    @field_validator("route")
    @classmethod
    def validate_route(cls, value: str) -> str:
        from src.core.manual_midi import MANUAL_MIDI_ROUTES

        normalized = str(value).strip()
        if normalized not in MANUAL_MIDI_ROUTES:
            raise ValueError(f"unsupported manual MIDI route: {value!r}")
        return normalized

    @field_validator("tempo_mode")
    @classmethod
    def validate_tempo_mode(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        valid = {mode.value for mode in TempoMode}
        if normalized not in valid:
            raise ValueError(f"unsupported tempo mode: {value!r}")
        return normalized

    @field_validator("muscriptor_processing_chain")
    @classmethod
    def validate_muscriptor_processing_chain(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        valid = {chain.value for chain in MuscriptorProcessingChain}
        if normalized not in valid:
            raise ValueError(f"unsupported MuScriptor processing chain: {value!r}")
        return normalized

    @field_validator("custom_bpm")
    @classmethod
    def validate_custom_bpm(cls, value: float | None) -> float | None:
        if value is None:
            return None
        normalized = float(value)
        if not MIN_TEMPO_BPM <= normalized <= MAX_TEMPO_BPM:
            raise ValueError(f"custom BPM must be between {MIN_TEMPO_BPM:g} and {MAX_TEMPO_BPM:g}")
        return normalized

    @model_validator(mode="after")
    def validate_tempo_contract(self):
        if "tempo_mode" not in self.model_fields_set and self.custom_bpm is not None:
            self.tempo_mode = TempoMode.FIXED_MANUAL.value
        if self.tempo_mode == TempoMode.FIXED_MANUAL.value and self.custom_bpm is None:
            raise ValueError("manual tempo mode requires custom_bpm")
        if self.tempo_mode != TempoMode.FIXED_MANUAL.value and self.custom_bpm is not None:
            raise ValueError("custom_bpm is only valid in manual tempo mode")
        return self


class ProgressSnapshot(BaseModel):
    stage: str = "queued"
    stage_progress: float = 0.0
    overall_progress: float = 0.0
    message: str = ""
    bpm_display: str | None = None
    source_bpm: float | None = None
    target_bpm: float | None = None


class ArtifactSnapshot(BaseModel):
    id: str
    kind: str
    name: str
    media_type: str
    size: int
    track_id: str | None = None
    sha256: str
    download_url: str


class JobSnapshot(BaseModel):
    id: str
    kind: str
    status: JobStatus
    revision: int
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    original_filename: str
    parent_job_id: str | None = None
    retry_of_job_id: str | None = None
    track_id: str | None = None
    queue_position: int | None = None
    request: dict = Field(default_factory=dict)
    progress: ProgressSnapshot
    error: str | None = None
    result: dict | None = None
    artifacts: list[ArtifactSnapshot] = Field(default_factory=list)
