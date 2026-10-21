"""Cancellable, cached source-audio analysis with no MIDI/Omnizart fallback."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.core.midi_chords import read_chord_record
from src.i18n.translator import t
from src.utils.chordmini_runtime import RECIPE_ID, get_chordmini_runtime
from src.utils.inference_lock import acquire_accelerator_lock
from src.utils.runtime_paths import get_ffmpeg_executable, get_project_root, get_runtime_data_dir
from src.utils.subprocess_utils import hidden_subprocess_kwargs


def run_chord_process(command, *, env, log_path, cancel_check, timeout=1800):
    with Path(log_path).open("wb") as log:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=get_project_root(),
            stdout=log,
            stderr=subprocess.STDOUT,
            **hidden_subprocess_kwargs(),
        )
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if cancel_check():
                    raise InterruptedError("Chord analysis cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Chord analysis timed out")
                time.sleep(0.05)
            if process.returncode:
                raise RuntimeError(
                    Path(log_path).read_text(encoding="utf-8", errors="replace")[-4000:]
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def chord_worker_command(audio, output):
    prefix = (
        [sys.executable, "--chordmini-worker"]
        if getattr(sys, "frozen", False)
        else [sys.executable, "-m", "src.core.chordmini_worker"]
    )
    return [*prefix, "--audio", str(audio), "--output", str(output)]


def _flac_streaminfo(source):
    with source.open("rb") as stream:
        prefix = stream.read(10)
        offset = 0
        if (
            prefix[:3] == b"ID3"
            and len(prefix) == 10
            and prefix[3] in (2, 3, 4)
            and not any(value & 0x80 for value in prefix[6:10])
        ):
            offset = 10 + sum(value << shift for value, shift in zip(prefix[6:10], (21, 14, 7, 0)))
            if prefix[3] == 4 and prefix[5] & 0x10:
                offset += 10
        stream.seek(offset)
        header = stream.read(42)
    if header[:4] != b"fLaC":
        return None
    if len(header) != 42 or header[4] & 0x7F or header[5:8] != b"\x00\x00\x22":
        raise ValueError(t("muscriptor_result.chord_flac_header_invalid"))
    packed = int.from_bytes(header[18:26], "big")
    info = dict(
        offset=offset,
        samplerate=packed >> 44,
        channels=((packed >> 41) & 7) + 1,
        bits=((packed >> 36) & 31) + 1,
        frames=packed & ((1 << 36) - 1),
        md5=header[26:42],
    )
    if not info["samplerate"] or info["bits"] < 4:
        raise ValueError(t("muscriptor_result.chord_flac_header_invalid"))
    return info


def _decode_flac_pcm32(source, decoded, info, cancel_check):
    import numpy as np
    import soundfile as sf

    maximum_frames = info["samplerate"] * 14400
    if info["frames"] > maximum_frames:
        raise ValueError(t("muscriptor_result.chord_audio_invalid"))
    checksum = hashlib.md5(usedforsecurity=False)
    frames = 0
    with source.open("rb", buffering=0) as stream:
        stream.seek(info["offset"])
        audio = sf.SoundFile(stream.fileno(), closefd=False)
        with audio:
            if (audio.format, audio.samplerate, audio.channels) != (
                "FLAC",
                info["samplerate"],
                info["channels"],
            ):
                raise ValueError(t("muscriptor_result.chord_flac_header_invalid"))
            with sf.SoundFile(
                decoded,
                "w",
                samplerate=audio.samplerate,
                channels=audio.channels,
                format="WAV",
                subtype="PCM_32",
            ) as output:
                while True:
                    if cancel_check():
                        raise InterruptedError("Chord analysis cancelled")
                    block = audio.read(65536, dtype="int32", always_2d=True)
                    if not len(block):
                        break
                    frames += len(block)
                    if frames > maximum_frames:
                        raise ValueError(t("muscriptor_result.chord_audio_invalid"))
                    # RFC 9639: interleaved, signed, little-endian, byte-aligned samples.
                    packed = (block >> (32 - info["bits"])).astype("<i4")
                    raw = packed.view(np.uint8).reshape(-1, 4)[:, : (info["bits"] + 7) // 8]
                    checksum.update(raw.tobytes())
                    output.write(block)
    if not frames:
        raise ValueError(t("muscriptor_result.chord_audio_invalid"))
    if info["frames"] and frames != info["frames"]:
        raise ValueError(
            t("muscriptor_result.chord_flac_incomplete", expected=info["frames"], actual=frames)
        )
    if any(info["md5"]) and checksum.digest() != info["md5"]:
        raise ValueError(t("muscriptor_result.chord_flac_checksum_mismatch"))


def _decode_chord_audio(source, decoded, *, env, log_path, cancel_check):
    if cancel_check():
        raise InterruptedError("Chord analysis cancelled")
    info = _flac_streaminfo(source)
    if info is not None and info["frames"]:
        _decode_flac_pcm32(source, decoded, info, cancel_check)
    else:
        run_chord_process(
            [
                get_ffmpeg_executable(),
                "-v",
                "error",
                "-xerror",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "pcm_s32le",
                str(decoded),
            ],
            env=env,
            log_path=log_path,
            cancel_check=cancel_check,
        )
    if cancel_check():
        raise InterruptedError("Chord analysis cancelled")


def analyze_audio_chords(audio_path, *, gpu_device=0, cancel_check=lambda: False):
    """Decode native-rate PCM32 and use the official model's own frontend."""
    get_chordmini_runtime()
    source = Path(audio_path).resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if cancel_check():
                raise InterruptedError("Chord analysis cancelled")
            digest.update(chunk)
    source_digest = digest.hexdigest()
    cache_root = get_runtime_data_dir() / "chords" / RECIPE_ID
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / f"{source_digest}.json"
    with acquire_accelerator_lock(gpu_device, cancel_check=cancel_check):
        if destination.is_file():
            record = json.loads(destination.read_text(encoding="utf-8"))
            if record.get("input_sha256") != source_digest:
                raise ValueError("Chord cache input identity mismatch")
            read_chord_record(record)
            return record
        with tempfile.TemporaryDirectory(prefix="chord-analysis-") as folder:
            folder = Path(folder)
            decoded, output = folder / "input.wav", folder / "output.json"
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(int(gpu_device)),
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                NUMBA_NUM_THREADS="4",
                PYTHONDONTWRITEBYTECODE="1",
            )
            _decode_chord_audio(
                source,
                decoded,
                env=env,
                log_path=folder / "decode.log",
                cancel_check=cancel_check,
            )
            import soundfile as sf

            info = sf.info(decoded)
            if info.subtype != "PCM_32" or not 0 < info.duration <= 14400:
                raise ValueError(t("muscriptor_result.chord_audio_invalid"))
            run_chord_process(
                chord_worker_command(decoded, output),
                env=env,
                log_path=folder / "model.log",
                cancel_check=cancel_check,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            record["input_sha256"] = source_digest
            record["input_audio"] = dict(
                sample_rate=info.samplerate,
                channels=info.channels,
                frames=info.frames,
                encoding="pcm_s32le",
            )
            read_chord_record(record)
            if cancel_check():
                raise InterruptedError("Chord analysis cancelled")
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json", dir=cache_root, delete=False
            ) as stream:
                pending = Path(stream.name)
                json.dump(record, stream, ensure_ascii=False)
            try:
                pending.replace(destination)
            finally:
                pending.unlink(missing_ok=True)
            return record
