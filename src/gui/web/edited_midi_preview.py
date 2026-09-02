"""Server-side publication and SoundFont rendering for browser MIDI edits."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import secrets
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from src.core.midi_editor import export_edited_midi
from src.core.muscriptor_result_assets import (
    MuscriptorRollNote,
    get_midi_audio_export_preset,
    prepare_midi_playback_assets,
    render_midi_audio_export,
    render_midi_stem_audio_export,
)
from src.gui.web.track_mixer_runtime import track_file_url

_MAX_EDITED_NOTES = 200_000


@dataclass(frozen=True)
class _PreviewContext:
    request_dir: Path
    source_midi_path: Path
    original_audio_path: Path
    reference_bpm: float
    muscriptor_groups: bool
    duration_seconds: float


class EditedMidiPreviewRegistry:
    """Keep opaque, request-scoped render contexts out of browser-controlled paths."""

    def __init__(self) -> None:
        self._contexts: dict[str, _PreviewContext] = {}
        self._cache: dict[tuple[str, str], dict[str, object]] = {}
        self._audio_export_cache: dict[tuple[str, str, str], dict[str, object]] = {}
        self._stem_audio_export_cache: dict[tuple[str, str, str], dict[str, object]] = {}
        self._lock = threading.RLock()
        self._render_lock = threading.Lock()

    def register(
        self,
        *,
        request_dir: str | Path,
        source_midi_path: str | Path,
        original_audio_path: str | Path,
        reference_bpm: float,
        muscriptor_groups: bool,
        duration_seconds: float,
    ) -> str:
        request_root = Path(request_dir).resolve()
        source_midi = Path(source_midi_path).resolve()
        original_audio = Path(original_audio_path).resolve()
        if not request_root.is_dir():
            raise FileNotFoundError(f"Browser preview request directory is missing: {request_root}")
        if not source_midi.is_file() or source_midi.stat().st_size <= 0:
            raise FileNotFoundError(f"Browser preview source MIDI is missing: {source_midi}")
        if not original_audio.is_file() or original_audio.stat().st_size <= 0:
            raise FileNotFoundError(f"Browser preview source audio is missing: {original_audio}")
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"Browser preview duration must be finite and positive: {duration_seconds!r}"
            )
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._contexts[token] = _PreviewContext(
                request_dir=request_root,
                source_midi_path=source_midi,
                original_audio_path=original_audio,
                reference_bpm=float(reference_bpm),
                muscriptor_groups=bool(muscriptor_groups),
                duration_seconds=duration,
            )
        return token

    def require_matching(
        self,
        token: object,
        *,
        request_dir: str | Path,
        source_midi_path: str | Path,
        original_audio_path: str | Path,
    ) -> str:
        value = str(token or "")
        with self._lock:
            context = self._contexts.get(value)
        if context is None:
            raise RuntimeError("Browser MIDI preview token is invalid or expired")
        expected = (
            Path(request_dir).resolve(),
            Path(source_midi_path).resolve(),
            Path(original_audio_path).resolve(),
        )
        actual = (context.request_dir, context.source_midi_path, context.original_audio_path)
        if actual != expected:
            raise RuntimeError("Browser MIDI preview token does not belong to this result")
        return value

    @staticmethod
    def _notes_from_payload(raw_notes: object) -> tuple[MuscriptorRollNote, ...]:
        if not isinstance(raw_notes, list) or len(raw_notes) > _MAX_EDITED_NOTES:
            raise ValueError(
                f"Edited MIDI preview requires a note list of at most {_MAX_EDITED_NOTES} entries"
            )
        notes = []
        for index, raw_note in enumerate(raw_notes):
            if not isinstance(raw_note, dict):
                raise ValueError(f"Edited MIDI note {index} is not an object")
            try:
                notes.append(
                    MuscriptorRollNote(
                        instrument=str(raw_note["instrument"]),
                        pitch=int(raw_note["pitch"]),
                        velocity=int(raw_note["velocity"]),
                        start=float(raw_note["start"]),
                        end=float(raw_note["end"]),
                        program=int(raw_note["program"]),
                        is_drum=bool(raw_note["is_drum"]),
                        track_index=int(raw_note["track_index"]),
                        channel=int(raw_note["channel"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Edited MIDI note {index} is malformed") from exc
        return tuple(notes)

    def render(self, payload_json: str) -> dict[str, object]:
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Edited MIDI preview payload is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Edited MIDI preview payload must be an object")
        token = str(payload.get("token") or "")
        with self._lock:
            context = self._contexts.get(token)
        if context is None:
            raise RuntimeError("Browser MIDI preview token is invalid or expired")
        if not context.request_dir.is_dir():
            raise RuntimeError("Browser MIDI preview request has expired")
        notes = self._notes_from_payload(payload.get("notes"))
        canonical = json.dumps(
            [note.__dict__ for note in notes],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        cache_key = (token, digest)
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        with self._render_lock:
            with self._lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                return dict(cached)
            output_dir = (context.request_dir / "edited-midi-previews" / digest).resolve()
            try:
                output_dir.relative_to(context.request_dir)
            except ValueError as exc:
                raise RuntimeError(
                    f"Edited MIDI preview output escaped its request: {output_dir}"
                ) from exc
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
            edited_midi = export_edited_midi(
                context.source_midi_path,
                output_dir / "edited-source-tempo.mid",
                notes,
                reference_bpm=context.reference_bpm,
                target_bpm=context.reference_bpm,
            )
            assets = prepare_midi_playback_assets(
                edited_midi,
                context.original_audio_path,
                output_dir,
                muscriptor_groups=context.muscriptor_groups,
                allow_empty_notes=True,
            )
            result: dict[str, object] = {
                "digest": digest,
                "duration": float(assets.duration),
                "noteCount": len(assets.notes),
                "instrumentUrls": {
                    name: track_file_url(str(path)) for name, path in assets.instrument_wavs.items()
                },
                "transcriptionUrl": track_file_url(str(assets.transcription_wav)),
                "stereoUrl": track_file_url(str(assets.stereo_mix_wav)),
            }
            with self._lock:
                self._cache[cache_key] = dict(result)
            return result

    def export_audio(self, payload_json: str) -> dict[str, object]:
        """Publish a verified WAV for the exact browser note snapshot and preset."""

        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("MIDI audio export payload is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("MIDI audio export payload must be an object")
        token = str(payload.get("token") or "")
        with self._lock:
            context = self._contexts.get(token)
        if context is None:
            raise RuntimeError("Browser MIDI preview token is invalid or expired")
        if not context.request_dir.is_dir():
            raise RuntimeError("Browser MIDI preview request has expired")
        notes = self._notes_from_payload(payload.get("notes"))
        preset = get_midi_audio_export_preset(str(payload.get("preset") or ""))
        canonical = json.dumps(
            [note.__dict__ for note in notes],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest_bytes = hashlib.sha256(canonical).digest()
        digest = digest_bytes.hex()
        path_digest = base64.urlsafe_b64encode(digest_bytes).decode("ascii").rstrip("=")
        cache_key = (token, digest, preset.id)
        with self._lock:
            cached = self._audio_export_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        with self._render_lock:
            with self._lock:
                cached = self._audio_export_cache.get(cache_key)
            if cached is not None:
                return dict(cached)
            output_dir = (context.request_dir / "midi-wav" / path_digest / preset.id).resolve()
            try:
                output_dir.relative_to(context.request_dir)
            except ValueError as exc:
                raise RuntimeError(f"MIDI audio export escaped its request: {output_dir}") from exc
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
            edited_midi = export_edited_midi(
                context.source_midi_path,
                output_dir / "current-source-tempo.mid",
                notes,
                reference_bpm=context.reference_bpm,
                target_bpm=context.reference_bpm,
            )
            destination = output_dir / "render.wav"
            rendered = render_midi_audio_export(
                edited_midi,
                destination,
                preset.id,
                silence_duration_seconds=context.duration_seconds,
            )
            result: dict[str, object] = {
                "digest": digest,
                "url": track_file_url(str(rendered.path)),
                "filename": f"transcription-{preset.filename_suffix}.wav",
                "presetId": preset.id,
                "bitDepth": preset.bit_depth,
                "sampleRate": preset.sample_rate,
                "subtype": preset.soundfile_subtype,
                "channels": rendered.channels,
                "frames": rendered.frames,
                "peak": rendered.peak,
            }
            with self._lock:
                self._audio_export_cache[cache_key] = dict(result)
            return result

    def export_stems(self, payload_json: str) -> dict[str, object]:
        """Publish aligned instrument WAVs for the exact browser note snapshot."""

        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("MIDI stem audio export payload is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("MIDI stem audio export payload must be an object")
        token = str(payload.get("token") or "")
        with self._lock:
            context = self._contexts.get(token)
        if context is None:
            raise RuntimeError("Browser MIDI preview token is invalid or expired")
        if not context.request_dir.is_dir():
            raise RuntimeError("Browser MIDI preview request has expired")
        notes = self._notes_from_payload(payload.get("notes"))
        preset = get_midi_audio_export_preset(str(payload.get("preset") or ""))
        canonical = json.dumps(
            [note.__dict__ for note in notes],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest_bytes = hashlib.sha256(canonical).digest()
        digest = digest_bytes.hex()
        path_digest = base64.urlsafe_b64encode(digest_bytes).decode("ascii").rstrip("=")
        cache_key = (token, digest, preset.id)
        with self._lock:
            cached = self._stem_audio_export_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        with self._render_lock:
            with self._lock:
                cached = self._stem_audio_export_cache.get(cache_key)
            if cached is not None:
                return dict(cached)
            output_dir = (context.request_dir / "midi-stems" / path_digest / preset.id).resolve()
            try:
                output_dir.relative_to(context.request_dir)
            except ValueError as exc:
                raise RuntimeError(f"MIDI stem export escaped its request: {output_dir}") from exc
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
            edited_midi = export_edited_midi(
                context.source_midi_path,
                output_dir / "current-source-tempo.mid",
                notes,
                reference_bpm=context.reference_bpm,
                target_bpm=context.reference_bpm,
            )
            destination = output_dir / "instrument-stems.zip"
            rendered = render_midi_stem_audio_export(
                edited_midi,
                destination,
                preset.id,
                muscriptor_groups=context.muscriptor_groups,
                minimum_duration_seconds=context.duration_seconds,
            )
            result: dict[str, object] = {
                "digest": digest,
                "url": track_file_url(str(rendered.path)),
                "filename": f"instrument-stems-{preset.filename_suffix}.zip",
                "presetId": preset.id,
                "bitDepth": preset.bit_depth,
                "sampleRate": preset.sample_rate,
                "subtype": preset.soundfile_subtype,
                "frames": rendered.frames,
                "memberCount": len(rendered.members),
                "members": [
                    {
                        "instrument": member.instrument,
                        "filename": member.archive_name,
                        "frames": member.frames,
                        "channels": member.channels,
                        "peak": member.peak,
                    }
                    for member in rendered.members
                ],
            }
            with self._lock:
                self._stem_audio_export_cache[cache_key] = dict(result)
            return result


__all__ = ["EditedMidiPreviewRegistry"]
