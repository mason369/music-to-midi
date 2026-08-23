"""Request-scoped browser publication for edited MIDI sheet-music ZIP files."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path

from src.core.midi_quantization import MIDI_QUANTIZE_GRIDS
from src.core.sheet_music import export_sheet_music_zip
from src.gui.web.track_mixer_runtime import track_file_url
from src.utils.midi_output import validate_midi_output

_MAX_MIDI_BYTES = 32 * 1024 * 1024
_MAX_MIDI_BASE64_CHARS = ((_MAX_MIDI_BYTES + 2) // 3) * 4


@dataclass(frozen=True)
class _SheetExportContext:
    request_dir: Path
    source_path: Path
    source_name: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SheetMusicExportRegistry:
    """Bind opaque browser tokens to request-owned export directories."""

    def __init__(self) -> None:
        self._contexts: dict[str, _SheetExportContext] = {}
        self._cache: dict[tuple[str, str, str], dict[str, object]] = {}
        self._lock = threading.RLock()
        self._render_lock = threading.Lock()

    def register(self, *, request_dir: str | Path, source_midi_path: str | Path) -> str:
        request_root = Path(request_dir).resolve()
        source_midi = validate_midi_output(
            source_midi_path, "browser sheet-music source MIDI"
        ).resolve()
        if not request_root.is_dir():
            raise FileNotFoundError(
                f"Browser sheet-music request directory is missing: {request_root}"
            )
        try:
            source_midi.relative_to(request_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Browser sheet-music source is outside its request: {source_midi}"
            ) from exc
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._contexts[token] = _SheetExportContext(
                request_dir=request_root,
                source_path=source_midi,
                source_name=source_midi.stem,
            )
        return token

    def require_matching(
        self,
        token: object,
        *,
        request_dir: str | Path,
        source_midi_path: str | Path,
    ) -> str:
        token_text = str(token or "")
        request_root = Path(request_dir).resolve()
        source_midi = validate_midi_output(
            source_midi_path, "linked browser sheet-music source MIDI"
        ).resolve()
        with self._lock:
            context = self._contexts.get(token_text)
        if context is None:
            raise RuntimeError("Sheet-music export token is invalid or expired")
        if not request_root.is_dir():
            raise RuntimeError("Sheet-music export request has expired")
        if context.request_dir != request_root or context.source_path != source_midi:
            raise RuntimeError("Sheet-music export token does not match this MIDI result")
        try:
            source_midi.relative_to(request_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Linked sheet-music source is outside its request: {source_midi}"
            ) from exc
        return token_text

    @staticmethod
    def _decode_midi(raw_value: object) -> bytes:
        if not isinstance(raw_value, str) or not raw_value:
            raise ValueError("Sheet-music export is missing edited MIDI data")
        if len(raw_value) > _MAX_MIDI_BASE64_CHARS:
            raise ValueError(f"Sheet-music MIDI exceeds the {_MAX_MIDI_BYTES}-byte request limit")
        try:
            midi_bytes = base64.b64decode(raw_value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Sheet-music MIDI is not valid base64") from exc
        if not 1 <= len(midi_bytes) <= _MAX_MIDI_BYTES:
            raise ValueError(f"Sheet-music MIDI size is outside 1-{_MAX_MIDI_BYTES} bytes")
        return midi_bytes

    @staticmethod
    def _public_result(cached: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in cached.items() if not key.startswith("_")}

    @classmethod
    def _validated_cached_result(
        cls,
        cached: dict[str, object],
        request_dir: Path,
    ) -> dict[str, object]:
        archive_path = Path(str(cached.get("_path") or "")).resolve()
        try:
            archive_path.relative_to(request_dir)
        except ValueError as exc:
            raise RuntimeError(
                f"Cached sheet-music archive escaped its request: {archive_path}"
            ) from exc
        if not archive_path.is_file():
            raise RuntimeError(f"Cached sheet-music archive is missing: {archive_path}")
        expected_size = cached.get("_size")
        expected_sha256 = str(cached.get("_sha256") or "")
        actual_size = archive_path.stat().st_size
        if actual_size != expected_size or _file_sha256(archive_path) != expected_sha256:
            raise RuntimeError(f"Cached sheet-music archive changed after export: {archive_path}")
        return cls._public_result(cached)

    def render(self, payload_json: str) -> dict[str, object]:
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Sheet-music export payload is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Sheet-music export payload must be an object")
        token = str(payload.get("token") or "")
        with self._lock:
            context = self._contexts.get(token)
        if context is None:
            raise RuntimeError("Sheet-music export token is invalid or expired")
        if not context.request_dir.is_dir():
            raise RuntimeError("Sheet-music export request has expired")
        quantize_grid = str(payload.get("quantize_grid") or "")
        if quantize_grid not in MIDI_QUANTIZE_GRIDS:
            raise ValueError(f"Unsupported sheet-music quantization grid: {quantize_grid!r}")
        midi_bytes = self._decode_midi(payload.get("midi_base64"))
        digest = hashlib.sha256(midi_bytes).hexdigest()
        cache_key = (token, digest, quantize_grid)
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return self._validated_cached_result(cached, context.request_dir)

        with self._render_lock:
            with self._lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                return self._validated_cached_result(cached, context.request_dir)
            output_dir = (
                context.request_dir / "sheet-music-exports" / f"{digest}-{quantize_grid[2:]}"
            ).resolve()
            try:
                output_dir.relative_to(context.request_dir)
            except ValueError as exc:
                raise RuntimeError(f"Sheet-music export escaped its request: {output_dir}") from exc
            output_dir.mkdir(parents=True, exist_ok=True)
            midi_path = output_dir / "edited-project-tempo.mid"
            midi_path.write_bytes(midi_bytes)
            validate_midi_output(midi_path, "browser edited sheet-music MIDI")
            archive_path = output_dir / "sheet_music.zip"
            result = export_sheet_music_zip(
                midi_path,
                archive_path,
                quantize_grid=quantize_grid,
            )
            filename_stem = "".join(
                character if character.isalnum() or character in {"-", "_"} else "_"
                for character in context.source_name
            ).strip("_")
            filename = f"{filename_stem or 'music-to-midi'}_sheet_music.zip"
            rendered: dict[str, object] = {
                "digest": digest,
                "url": track_file_url(result.path),
                "filename": filename,
                "memberCount": len(result.members),
                "quantizeGrid": result.quantize_grid,
                "musescoreVersion": result.musescore_version,
                "_path": str(result.path.resolve()),
                "_size": result.path.stat().st_size,
                "_sha256": _file_sha256(result.path),
            }
            with self._lock:
                self._cache[cache_key] = dict(rendered)
            return self._public_result(rendered)


__all__ = ["SheetMusicExportRegistry"]
