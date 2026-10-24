"""Decode complete FLAC samples despite trailing tags; reject real audio damage."""

import contextlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.core import chord_analysis
from src.i18n.translator import Translator, get_translator, set_language, t
from src.utils.chordmini_runtime import CHECKPOINT_SHA256, CHORDMINI_COMMIT, RECIPE_ID
from src.utils.runtime_paths import get_ffmpeg_executable


def flac_audio(path, *, bits=24, rate=48000, channels=2, frames=130013):
    values = np.random.default_rng(22).integers(
        -(1 << (bits - 1)), 1 << (bits - 1), size=(frames, channels), dtype="int32"
    )
    expected = values << (32 - bits)
    sf.write(path, expected, rate, subtype=f"PCM_{'S8' if bits == 8 else bits}")
    return expected, rate


def decode(source, output, cancel_check=lambda: False):
    chord_analysis._decode_chord_audio(
        source,
        output,
        env=os.environ.copy(),
        log_path=output.with_suffix(".log"),
        cancel_check=cancel_check,
    )


def strict_ffmpeg(source, output):
    return subprocess.run(
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
            str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


@pytest.mark.parametrize(
    "bits,rate,channels",
    [
        (8, 32000, 1),
        (16, 44100, 2),
        (24, 48000, 2),
        (16, 96000, 1),
        (24, 96000, 6),
        (24, 22050, 1),
    ],
)
def test_native_flac_retains_every_sample_and_matches_strict_ffmpeg(tmp_path, bits, rate, channels):
    source, output = tmp_path / "原始音频.FLAC", tmp_path / "decoded.wav"
    expected, _ = flac_audio(source, bits=bits, rate=rate, channels=channels)
    snapshot = source.read_bytes()
    decode(source, output)
    actual, actual_rate = sf.read(output, dtype="int32", always_2d=True)
    assert actual_rate == rate
    assert sf.info(output).subtype == "PCM_32"
    np.testing.assert_array_equal(actual, expected)
    reference = tmp_path / "ffmpeg.wav"
    result = strict_ffmpeg(source, reference)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(actual, sf.read(reference, dtype="int32", always_2d=True)[0])
    assert source.read_bytes() == snapshot


@pytest.mark.parametrize(
    "tail",
    [
        b"TAG" + b"test song".ljust(125, b"\0"),
        b"APETAGEX" + b"\0" * 24,
        b"not audio" * 100,
    ],
)
def test_screenshot_flac_error_is_reproduced_but_complete_samples_decode(tmp_path, tail):
    source, output = tmp_path / "带尾部标签.flac", tmp_path / "decoded.wav"
    expected, rate = flac_audio(source)
    source.write_bytes(source.read_bytes() + tail)
    snapshot = source.read_bytes()
    reference = tmp_path / "strict.wav"
    result = strict_ffmpeg(source, reference)
    # FFmpeg builds differ in whether they parse trailing tags as FLAC frames.
    # Both paths must validate actual audio, rather than require an upstream bug.
    if result.returncode:
        assert "invalid sync code" in result.stderr
        assert "invalid frame header" in result.stderr
    else:
        reference_audio, reference_rate = sf.read(reference, dtype="int32", always_2d=True)
        np.testing.assert_array_equal(reference_audio, expected)
        assert reference_rate == rate
    decode(source, output)
    actual, actual_rate = sf.read(output, dtype="int32", always_2d=True)
    np.testing.assert_array_equal(actual, expected)
    assert actual_rate == rate
    assert source.read_bytes() == snapshot


@pytest.mark.parametrize(
    "change,no_checksum",
    [
        ("truncated", False),
        ("frame", False),
        ("checksum", False),
        ("count", False),
        ("header", False),
        ("truncated", True),
        ("frame", True),
        ("count", True),
    ],
)
def test_real_corruption_is_rejected_without_a_permissive_decoder(
    tmp_path, monkeypatch, change, no_checksum
):
    source = tmp_path / "damaged.flac"
    flac_audio(source)
    content = bytearray(source.read_bytes())
    if no_checksum:
        content[26:42] = b"\0" * 16
    if change == "truncated":
        del content[len(content) // 2 :]
    elif change == "frame":
        content[len(content) // 2] ^= 0x40
    elif change == "checksum":
        content[26] ^= 0x40
    elif change == "count":
        packed = int.from_bytes(content[18:26], "big")
        content[18:26] = (packed + 113).to_bytes(8, "big")
    else:
        content[7] = 33
    source.write_bytes(content)

    def unexpected_decoder(*args, **kwargs):
        pytest.fail("Corrupt FLAC must not fall back to a permissive decoder")

    monkeypatch.setattr(chord_analysis, "run_chord_process", unexpected_decoder)
    with pytest.raises((ValueError, sf.LibsndfileError)):
        decode(source, tmp_path / "decoded.wav")
    assert source.read_bytes() == content


def test_flac_without_optional_streaminfo_checksum_still_checks_frames(tmp_path):
    source = tmp_path / "no-md5.flac"
    expected, rate = flac_audio(source)
    content = bytearray(source.read_bytes())
    content[26:42] = b"\0" * 16
    source.write_bytes(content)
    output = tmp_path / "decoded.wav"
    decode(source, output)
    actual, actual_rate = sf.read(output, dtype="int32", always_2d=True)
    np.testing.assert_array_equal(actual, expected)
    assert actual_rate == rate


@pytest.mark.parametrize("footer", [False, True])
def test_flac_with_leading_id3v2_and_trailing_id3v1_retains_all_audio(tmp_path, footer):
    source = tmp_path / "both-tags.flac"
    expected, rate = flac_audio(source)
    payload = b"TIT2\x00\x00\x00\x05\x00\x00\x03song"
    header = (
        b"ID3\x04\x00" + bytes([0x10 if footer else 0]) + b"\x00\x00\x00" + bytes([len(payload)])
    )
    ending = b"3DI" + header[3:] if footer else b""
    source.write_bytes(header + payload + ending + source.read_bytes() + b"TAG" + b"\0" * 125)
    snapshot = source.read_bytes()
    output = tmp_path / "decoded.wav"
    decode(source, output)
    actual, actual_rate = sf.read(output, dtype="int32", always_2d=True)
    np.testing.assert_array_equal(actual, expected)
    assert actual_rate == rate
    assert source.read_bytes() == snapshot


def test_flac_with_unknown_declared_length_decodes_to_real_eof(tmp_path):
    source = tmp_path / "stream.flac"
    expected, rate = flac_audio(source)
    content = bytearray(source.read_bytes())
    packed = int.from_bytes(content[18:26], "big") & ~((1 << 36) - 1)
    content[18:26] = packed.to_bytes(8, "big")
    source.write_bytes(content)
    output = tmp_path / "decoded.wav"
    decode(source, output)
    actual, actual_rate = sf.read(output, dtype="int32", always_2d=True)
    np.testing.assert_array_equal(actual, expected)
    assert actual_rate == rate


def test_native_flac_decode_can_be_cancelled_between_blocks(tmp_path):
    source = tmp_path / "cancel.flac"
    flac_audio(source)
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(InterruptedError):
        decode(source, tmp_path / "decoded.wav", cancelled)
    assert calls == 3


@pytest.mark.parametrize("format", ["WAV", "MP3"])
def test_non_flac_keeps_strict_ffmpeg_decoding(tmp_path, format):
    source = tmp_path / f"source.{format.lower()}"
    values = np.sin(np.arange(20000) * 0.07) * 0.2
    sf.write(source, values, 32000, format=format)
    output = tmp_path / "decoded.wav"
    decode(source, output)
    reference = tmp_path / "reference.wav"
    result = strict_ffmpeg(source, reference)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(
        sf.read(output, dtype="int32")[0], sf.read(reference, dtype="int32")[0]
    )
    assert sf.info(output).samplerate == 32000


def model_record(duration):
    return dict(
        id="chordmini_btc",
        source_revision=CHORDMINI_COMMIT,
        checkpoint_sha256=CHECKPOINT_SHA256,
        recipe=RECIPE_ID,
        duration=duration,
        frame_seconds=0.05,
        segments=[dict(start=0.0, end=duration, label="C:maj")],
    )


def test_tagged_flac_reaches_worker_only_with_verified_pcm_and_uses_existing_cache(
    tmp_path, monkeypatch
):
    source = tmp_path / "tagged.flac"
    rate_frames = 144000
    expected, rate = flac_audio(source, frames=rate_frames)
    source.write_bytes(source.read_bytes() + b"TAG" + b"\0" * 125)
    record = model_record(rate_frames / rate)
    calls = []
    monkeypatch.setattr(chord_analysis, "get_chordmini_runtime", lambda: tmp_path)
    monkeypatch.setattr(chord_analysis, "get_runtime_data_dir", lambda: tmp_path / "runtime")
    monkeypatch.setattr(
        chord_analysis, "acquire_accelerator_lock", lambda *a, **k: contextlib.nullcontext()
    )

    def worker(command, **kwargs):
        calls.append(command)
        decoded_path = Path(command[command.index("--audio") + 1])
        np.testing.assert_array_equal(
            sf.read(decoded_path, dtype="int32", always_2d=True)[0], expected
        )
        assert sf.info(decoded_path).samplerate == rate
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps(record), encoding="utf-8"
        )

    monkeypatch.setattr(chord_analysis, "run_chord_process", worker)
    actual = chord_analysis.analyze_audio_chords(source)
    assert actual["segments"] == record["segments"]
    assert actual["input_audio"] == dict(
        sample_rate=rate, channels=2, frames=rate_frames, encoding="pcm_s32le"
    )
    assert len(calls) == 1
    assert chord_analysis.analyze_audio_chords(source) == actual
    assert len(calls) == 1


def test_integrity_failure_never_starts_model_or_writes_a_success_cache(tmp_path, monkeypatch):
    source = tmp_path / "bad.flac"
    flac_audio(source)
    content = bytearray(source.read_bytes())
    content[26] ^= 0x40
    source.write_bytes(content)
    monkeypatch.setattr(chord_analysis, "get_chordmini_runtime", lambda: tmp_path)
    monkeypatch.setattr(chord_analysis, "get_runtime_data_dir", lambda: tmp_path / "runtime")
    monkeypatch.setattr(
        chord_analysis, "acquire_accelerator_lock", lambda *a, **k: contextlib.nullcontext()
    )

    def unexpected_worker(*args, **kwargs):
        pytest.fail("The model must not run on unverified audio")

    monkeypatch.setattr(chord_analysis, "run_chord_process", unexpected_worker)
    with pytest.raises(ValueError):
        chord_analysis.analyze_audio_chords(source)
    assert not list((tmp_path / "runtime/chords" / RECIPE_ID).iterdir())


@pytest.mark.parametrize("locale", tuple(Translator.AVAILABLE_LANGUAGES))
def test_flac_integrity_errors_are_translated_in_every_locale(tmp_path, locale):
    source = tmp_path / "bad.flac"
    flac_audio(source)
    content = bytearray(source.read_bytes())
    content[26] ^= 0x40
    source.write_bytes(content)
    previous = get_translator().get_language()
    try:
        set_language(locale)
        with pytest.raises(ValueError) as error:
            decode(source, tmp_path / "decoded.wav")
        assert str(error.value) == t("muscriptor_result.chord_flac_checksum_mismatch")
        assert "muscriptor_result." not in str(error.value)
    finally:
        set_language(previous)
