"""Real MIDI piano-roll data and SoundFont playback assets.

The public MuScriptor names remain as compatibility aliases, but the asset
pipeline accepts every General MIDI program used by the project's other
transcription backends as well.
"""

from __future__ import annotations

import math
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mido

from src.models.muscriptor_instruments import MUSCRIPTOR_REPRESENTATIVE_PROGRAMS
from src.utils.fluidsynth_runtime import (
    get_fluidsynth_executable,
    get_fluidsynth_subprocess_env,
)
from src.utils.muscriptor_soundfont_downloader import download_muscriptor_soundfont

_PROGRAM_TO_INSTRUMENT = {
    program: instrument for instrument, program in MUSCRIPTOR_REPRESENTATIVE_PROGRAMS.items()
}
_SAMPLE_RATE = 44_100
_LIVE_BUS_FADE_SECONDS = 0.03


def _covering_frame_count(duration: float) -> int:
    """Return the smallest 44.1 kHz frame count that fully covers ``duration``."""

    value = float(duration)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Playback duration must be finite and positive: {value}")
    scaled = value * _SAMPLE_RATE
    # ``nextafter`` keeps an exactly frame-aligned duration on that frame while
    # still rounding every genuinely fractional boundary upward.  Using round()
    # here can make rendered audio shorter than the final completed note.
    return max(1, int(math.ceil(math.nextafter(scaled, -math.inf))))


def _safe_asset_name(instrument: str) -> str:
    """Return a portable cache filename without changing the instrument key."""

    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in instrument
    )


@dataclass(frozen=True)
class MuscriptorRollNote:
    instrument: str
    pitch: int
    velocity: int
    start: float
    end: float
    program: int = 0
    is_drum: bool = False


@dataclass(frozen=True)
class MuscriptorPlaybackAssets:
    notes: tuple[MuscriptorRollNote, ...]
    duration: float
    transcription_wav: Path
    live_transcription_wav: Path
    stereo_mix_wav: Path
    original_left_wav: Path
    transcription_right_wav: Path
    instrument_wavs: dict[str, Path]
    instrument_right_wavs: dict[str, Path]
    midi_gain_db: float


@dataclass(frozen=True)
class MuscriptorPreviewAssets:
    """Real SoundFont renders for one completed streaming frontier."""

    notes: tuple[MuscriptorRollNote, ...]
    duration: float
    transcription_wav: Path
    instrument_wavs: dict[str, Path]
    midi_gain_db: float


@dataclass(frozen=True)
class _LivePlaybackBuses:
    transcription_wav: Path
    instrument_wavs: dict[str, Path]
    midi_gain_db: float
    original_left_wav: Path | None = None
    transcription_right_wav: Path | None = None
    instrument_right_wavs: dict[str, Path] | None = None


def _tempo_events(midi: mido.MidiFile) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if message.type == "set_tempo":
                events.append((tick, int(message.tempo)))
    events.sort(key=lambda item: item[0])
    return events


def _tick_to_seconds(tick: int, ticks_per_beat: int, tempos: list[tuple[int, int]]) -> float:
    elapsed = 0.0
    previous_tick = 0
    tempo = 500_000
    for tempo_tick, next_tempo in tempos:
        if tempo_tick > tick:
            break
        elapsed += mido.tick2second(tempo_tick - previous_tick, ticks_per_beat, tempo)
        previous_tick = tempo_tick
        tempo = next_tempo
    elapsed += mido.tick2second(tick - previous_tick, ticks_per_beat, tempo)
    return float(elapsed)


def _generic_instrument_key(program: int, is_drum: bool) -> str:
    return "drums" if is_drum else f"gm:{int(program):03d}"


def read_midi_roll_notes(
    midi_path: str | Path,
    *,
    muscriptor_groups: bool = False,
) -> tuple[MuscriptorRollNote, ...]:
    """Parse a backend writer artifact without reconstructing model events."""

    midi = mido.MidiFile(str(midi_path))
    tempos = _tempo_events(midi)
    notes: list[MuscriptorRollNote] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        programs = {channel: 0 for channel in range(16)}
        active: dict[
            tuple[int, int],
            list[tuple[float, int, str, int, bool]],
        ] = defaultdict(list)
        track_name = ""
        for message in track:
            tick += int(message.time)
            if message.is_meta:
                if message.type == "track_name":
                    track_name = str(message.name).strip().replace(" ", "_")
                continue
            if message.type == "program_change":
                programs[int(message.channel)] = int(message.program)
                continue
            if message.type not in {"note_on", "note_off"}:
                continue
            channel = int(message.channel)
            pitch = int(message.note)
            is_start = message.type == "note_on" and int(message.velocity) > 0
            key = (channel, pitch)
            event_time = _tick_to_seconds(tick, midi.ticks_per_beat, tempos)
            if is_start:
                is_drum = channel == 9
                program = programs[channel]
                if is_drum:
                    instrument = "drums"
                elif muscriptor_groups:
                    instrument = _PROGRAM_TO_INSTRUMENT.get(program, track_name)
                else:
                    instrument = _generic_instrument_key(program, False)
                if (
                    muscriptor_groups
                    and instrument not in MUSCRIPTOR_REPRESENTATIVE_PROGRAMS
                    and instrument != "drums"
                ):
                    raise RuntimeError(
                        "Unknown instrument in MuScriptor final MIDI: "
                        f"track={track_index}, name={track_name!r}, "
                        f"program={programs[channel]}"
                    )
                # Program/channel state belongs to this exact note-on. A later
                # event on another channel may change both values before this
                # note ends, so they cannot be recovered from loop locals at
                # note-off time.
                active[key].append(
                    (
                        event_time,
                        int(message.velocity),
                        instrument,
                        program,
                        is_drum,
                    )
                )
                continue
            if not active[key]:
                continue
            start, velocity, instrument, program, is_drum = active[key].pop(0)
            notes.append(
                MuscriptorRollNote(
                    instrument=instrument,
                    pitch=pitch,
                    velocity=velocity,
                    start=start,
                    end=max(start + 0.01, event_time),
                    program=program,
                    is_drum=is_drum,
                )
            )
    notes.sort(key=lambda note: (note.start, note.pitch, note.instrument, note.end))
    return tuple(notes)


def read_muscriptor_roll_notes(midi_path: str | Path) -> tuple[MuscriptorRollNote, ...]:
    """Parse MuScriptor's grouped-program writer artifact."""

    return read_midi_roll_notes(midi_path, muscriptor_groups=True)


def _write_instrument_midi(
    notes: list[MuscriptorRollNote],
    destination: Path,
) -> None:
    if not notes:
        raise ValueError("Cannot write an empty instrument MIDI")
    first = notes[0]
    instrument = first.instrument
    if any(
        (note.instrument, note.program, note.is_drum) != (instrument, first.program, first.is_drum)
        for note in notes
    ):
        raise ValueError("Instrument MIDI received mixed instruments")

    midi = mido.MidiFile(type=1, ticks_per_beat=1000)
    tempo_track = mido.MidiTrack()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
    midi.tracks.append(tempo_track)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=instrument.replace("_", " "), time=0))
    channel = 9 if first.is_drum or instrument == "drums" else 0
    if channel != 9:
        program = MUSCRIPTOR_REPRESENTATIVE_PROGRAMS.get(instrument, int(first.program))
        track.append(
            mido.Message(
                "program_change",
                program=program,
                channel=channel,
                time=0,
            )
        )
    events: list[tuple[int, int, mido.Message]] = []
    for note in notes:
        start_tick = max(0, int(round(note.start * 1000)))
        end_tick = max(start_tick + 1, int(round(note.end * 1000)))
        events.append(
            (
                start_tick,
                1,
                mido.Message(
                    "note_on",
                    note=note.pitch,
                    velocity=max(1, note.velocity),
                    channel=channel,
                    time=0,
                ),
            )
        )
        events.append(
            (
                end_tick,
                0,
                mido.Message(
                    "note_off",
                    note=note.pitch,
                    velocity=0,
                    channel=channel,
                    time=0,
                ),
            )
        )
    previous = 0
    for absolute_tick, _order, message in sorted(events, key=lambda item: (item[0], item[1])):
        message.time = absolute_tick - previous
        previous = absolute_tick
        track.append(message)
    midi.tracks.append(track)
    midi.save(str(destination))


def _synthesize(
    executable: Path,
    soundfont: Path,
    midi_path: Path,
    output_path: Path,
    cancel_check=None,
) -> None:
    temporary = output_path.with_name(output_path.name + ".part.wav")
    if temporary.exists():
        temporary.unlink()
    process = subprocess.Popen(
        [
            str(executable),
            "-ni",
            "-F",
            str(temporary),
            "-r",
            str(_SAMPLE_RATE),
            str(soundfont),
            str(midi_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=get_fluidsynth_subprocess_env(executable),
    )
    started = time.monotonic()
    while process.poll() is None:
        if cancel_check is not None and cancel_check():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if temporary.exists():
                temporary.unlink()
            raise InterruptedError("MuScriptor playback rendering cancelled")
        if time.monotonic() - started > 600:
            process.kill()
            process.wait(timeout=5)
            if temporary.exists():
                temporary.unlink()
            raise RuntimeError(f"FluidSynth timed out after 600 seconds for {midi_path.name}")
        time.sleep(0.1)
    stdout, stderr = process.communicate()
    if process.returncode != 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
        raise RuntimeError(
            f"FluidSynth failed for {midi_path.name} (exit={process.returncode}): "
            f"{stderr.decode(errors='replace').strip()}"
        )
    temporary.replace(output_path)


def prepare_midi_preview_assets(
    notes: Iterable[MuscriptorRollNote],
    playable_duration: float,
    output_dir: str | Path,
    *,
    reference_audio_path: str | Path,
    cancel_check=None,
) -> MuscriptorPreviewAssets:
    """Render completed backend notes into per-instrument playable WAV files."""

    normalized = tuple(notes)
    if not normalized:
        raise ValueError("Cannot render a MIDI preview without completed notes")

    last_note_end = 0.0
    grouped: dict[str, list[MuscriptorRollNote]] = defaultdict(list)
    for note in normalized:
        if not 0 <= int(note.program) <= 127:
            raise ValueError(f"Invalid MIDI preview program: {note.program}")
        if not 0 <= note.pitch <= 127:
            raise ValueError(f"Invalid MuScriptor preview pitch: {note.pitch}")
        if not 1 <= note.velocity <= 127:
            raise ValueError(f"Invalid MuScriptor preview velocity: {note.velocity}")
        if note.start < 0 or note.end <= note.start:
            raise ValueError(
                "Invalid MuScriptor preview note interval: " f"start={note.start}, end={note.end}"
            )
        grouped[note.instrument].append(note)
        last_note_end = max(last_note_end, note.end)

    duration = float(playable_duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"MIDI preview duration must be finite and positive: {duration}")
    if duration + 1e-6 < last_note_end:
        raise ValueError(
            "MIDI preview frontier precedes a completed note: "
            f"frontier={duration}, note_end={last_note_end}"
        )
    render_boundary = max(duration, last_note_end)

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    check_cancelled = cancel_check or (lambda: False)

    def checkpoint() -> None:
        if check_cancelled():
            raise InterruptedError("MuScriptor preview rendering cancelled")

    checkpoint()
    executable = get_fluidsynth_executable()
    soundfont = download_muscriptor_soundfont(printer=lambda _message: None)
    raw_instrument_wavs: dict[str, Path] = {}
    for instrument, instrument_notes in grouped.items():
        checkpoint()
        asset_name = _safe_asset_name(instrument)
        instrument_midi = destination / f"{asset_name}.mid"
        instrument_wav = destination / f"{asset_name}.wav"
        _write_instrument_midi(instrument_notes, instrument_midi)
        _synthesize(
            executable,
            soundfont,
            instrument_midi,
            instrument_wav,
            cancel_check=check_cancelled,
        )
        raw_instrument_wavs[instrument] = instrument_wav

    checkpoint()
    live_buses = _write_live_playback_buses(
        Path(reference_audio_path).resolve(),
        raw_instrument_wavs,
        destination,
        target_duration=render_boundary,
        cancel_check=check_cancelled,
    )
    import soundfile as sf

    live_info = sf.info(str(live_buses.transcription_wav))
    actual_duration = live_info.frames / live_info.samplerate
    frame_seconds = 1.0 / live_info.samplerate
    if (
        actual_duration + 1e-9 < render_boundary
        or actual_duration - render_boundary > frame_seconds + 1e-9
    ):
        raise RuntimeError(
            "Rendered MIDI preview does not cover its stable frontier: "
            f"frontier={render_boundary}, rendered={actual_duration}"
        )
    if any(note.end > actual_duration + 1e-9 for note in normalized):
        raise RuntimeError(
            "Rendered MIDI preview audio ends before a completed note: "
            f"rendered={actual_duration}, note_end={last_note_end}"
        )
    return MuscriptorPreviewAssets(
        notes=normalized,
        duration=actual_duration,
        transcription_wav=live_buses.transcription_wav,
        instrument_wavs=live_buses.instrument_wavs,
        midi_gain_db=live_buses.midi_gain_db,
    )


def prepare_muscriptor_preview_assets(
    notes: Iterable[MuscriptorRollNote],
    playable_duration: float,
    output_dir: str | Path,
    *,
    reference_audio_path: str | Path,
    cancel_check=None,
) -> MuscriptorPreviewAssets:
    """Compatibility wrapper for the generic preview renderer."""

    return prepare_midi_preview_assets(
        notes,
        playable_duration,
        output_dir,
        reference_audio_path=reference_audio_path,
        cancel_check=cancel_check,
    )


def _load_mono_44k(path: Path):
    import librosa
    import numpy as np

    audio, _sample_rate = librosa.load(str(path), sr=_SAMPLE_RATE, mono=True)
    return np.asarray(audio, dtype="float32")


@lru_cache(maxsize=2)
def _load_reference_mono_44k_cached(path: str, size: int, modified_ns: int):
    del size, modified_ns
    return _load_mono_44k(Path(path))


def _load_reference_mono_44k(path: Path):
    resolved = path.resolve()
    stat = resolved.stat()
    return _load_reference_mono_44k_cached(
        str(resolved),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _audio_rms(audio) -> float:
    import numpy as np

    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype="float64"))))


def _write_scaled_mono_44k(
    source: Path,
    destination: Path,
    *,
    gain: float,
    target_frames: int,
    cancel_check=None,
) -> None:
    """Scale a FluidSynth WAV in bounded blocks and write an exact duration."""

    import numpy as np
    import soundfile as sf

    if target_frames <= 0:
        raise ValueError(f"Scaled playback target must contain frames: {target_frames}")
    check_cancelled = cancel_check or (lambda: False)
    block_frames = 65_536
    fade_frames = min(
        target_frames,
        max(2, int(round(_LIVE_BUS_FADE_SECONDS * _SAMPLE_RATE))),
    )
    fade_start = target_frames - fade_frames
    with sf.SoundFile(source) as reader:
        if reader.samplerate != _SAMPLE_RATE:
            raise RuntimeError(
                "FluidSynth playback asset has an unexpected sample rate: "
                f"{source} ({reader.samplerate} Hz)"
            )
        with sf.SoundFile(
            destination,
            mode="w",
            samplerate=_SAMPLE_RATE,
            channels=1,
            subtype="PCM_16",
        ) as writer:
            written = 0
            source_exhausted = False
            while written < target_frames:
                if check_cancelled():
                    raise InterruptedError("MIDI live-bus preparation cancelled")
                count = min(block_frames, target_frames - written)
                if source_exhausted:
                    mono = np.zeros(count, dtype="float32")
                else:
                    block = reader.read(count, dtype="float32", always_2d=True)
                    source_exhausted = len(block) < count
                    if len(block):
                        mono = block.mean(axis=1, dtype="float32")
                    else:
                        mono = np.zeros(0, dtype="float32")
                    if len(mono) < count:
                        mono = np.pad(mono, (0, count - len(mono)))
                if fade_frames == 1 and written + count == target_frames:
                    mono[-1] = 0.0
                elif fade_frames >= 2 and written + count > fade_start:
                    local_start = max(0, fade_start - written)
                    absolute_frames = np.arange(
                        written + local_start,
                        written + count,
                        dtype="float32",
                    )
                    fade = (target_frames - 1 - absolute_frames) / (fade_frames - 1)
                    mono[local_start:] *= np.clip(fade, 0.0, 1.0)
                writer.write(mono * gain)
                written += count


def _fade_live_bus_tail(audio):
    """End a monitoring bus at the transport boundary without a held sample."""

    import numpy as np

    faded = np.asarray(audio, dtype="float32").copy()
    fade_frames = min(
        len(faded),
        max(2, int(round(_LIVE_BUS_FADE_SECONDS * _SAMPLE_RATE))),
    )
    if fade_frames >= 2:
        faded[-fade_frames:] *= np.linspace(
            1.0,
            0.0,
            fade_frames,
            dtype="float32",
        )
    elif fade_frames == 1:
        faded[-1] = 0.0
    return faded


def _write_live_playback_buses(
    original_audio: Path,
    raw_instrument_wavs: dict[str, Path],
    output_dir: Path,
    *,
    combined_source: Path | None = None,
    target_duration: float | None = None,
    include_stereo: bool = False,
    cancel_check=None,
) -> _LivePlaybackBuses:
    """Create peak-safe, loudness-matched buses for interactive playback."""

    import numpy as np
    import soundfile as sf

    if not original_audio.is_file():
        raise FileNotFoundError(f"Playback reference audio is missing: {original_audio}")
    if not raw_instrument_wavs:
        raise ValueError("Cannot prepare live MIDI playback without instrument audio")
    check_cancelled = cancel_check or (lambda: False)

    def checkpoint() -> None:
        if check_cancelled():
            raise InterruptedError("MIDI live-bus preparation cancelled")

    checkpoint()
    target_frames = None
    if target_duration is not None:
        target_frames = _covering_frame_count(target_duration)

    def fit_duration(audio):
        if target_frames is None:
            return audio
        if len(audio) >= target_frames:
            return audio[:target_frames]
        return np.pad(audio, (0, target_frames - len(audio)))

    original = _load_reference_mono_44k(original_audio)
    if len(original) <= 0:
        raise RuntimeError(f"Playback reference audio is empty: {original_audio}")
    if target_frames is None:
        # The original recording is the transport master. FluidSynth can append
        # several seconds of release/reverb after the last MIDI event; allowing
        # that tail to define the bus length leaves MIDI playing after the song.
        target_frames = len(original)
    max_component_peak = 0.0
    if combined_source is not None:
        if not combined_source.is_file():
            raise FileNotFoundError(f"Combined MIDI render is missing: {combined_source}")
        combined = fit_duration(_load_mono_44k(combined_source))
        for source in raw_instrument_wavs.values():
            checkpoint()
            component = fit_duration(_load_mono_44k(source))
            if len(component):
                max_component_peak = max(
                    max_component_peak,
                    float(np.max(np.abs(component))),
                )
    else:
        combined = (
            np.zeros(target_frames, dtype="float32")
            if target_frames is not None
            else np.zeros(0, dtype="float32")
        )
        for source in raw_instrument_wavs.values():
            checkpoint()
            component = fit_duration(_load_mono_44k(source))
            if len(component) > len(combined):
                combined = np.pad(combined, (0, len(component) - len(combined)))
            combined[: len(component)] += component
            if len(component):
                max_component_peak = max(
                    max_component_peak,
                    float(np.max(np.abs(component))),
                )

    synthesis_rms = _audio_rms(combined)
    if synthesis_rms <= 1e-8:
        raise RuntimeError("FluidSynth produced silent MIDI playback audio")
    combined_peak = float(np.max(np.abs(combined))) if len(combined) else 0.0
    peak_basis = max(combined_peak, max_component_peak)
    if peak_basis <= 1e-8:
        raise RuntimeError("FluidSynth produced zero-peak MIDI playback audio")

    reference = original[: len(combined)]
    if len(reference) < len(combined):
        reference = np.pad(reference, (0, len(combined) - len(reference)))
    reference_rms = _audio_rms(reference)
    # A silent source is valid input. In that case use an explicit -18 dBFS
    # monitoring target so the MIDI-only side remains audible.
    target_rms = reference_rms if reference_rms > 1e-8 else 10.0 ** (-18.0 / 20.0)
    gain = min(target_rms / synthesis_rms, 0.95 / peak_basis)
    if not math.isfinite(gain) or gain <= 0:
        raise RuntimeError(
            "Invalid MIDI monitoring gain: "
            f"reference_rms={reference_rms}, synthesis_rms={synthesis_rms}, peak={peak_basis}"
        )

    live_transcription = output_dir / "midi-live.wav"
    sf.write(
        live_transcription,
        _fade_live_bus_tail(combined * gain),
        _SAMPLE_RATE,
    )
    live_instruments: dict[str, Path] = {}
    for instrument, source in raw_instrument_wavs.items():
        checkpoint()
        destination = output_dir / f"{_safe_asset_name(instrument)}-live.wav"
        _write_scaled_mono_44k(
            source,
            destination,
            gain=gain,
            target_frames=len(combined),
            cancel_check=check_cancelled,
        )
        live_instruments[instrument] = destination

    original_left = None
    transcription_right = None
    instrument_rights: dict[str, Path] | None = None
    if include_stereo:
        original_left, transcription_right, instrument_rights = _write_channelized_playback(
            original_audio,
            live_transcription,
            live_instruments,
            output_dir,
        )
    return _LivePlaybackBuses(
        transcription_wav=live_transcription,
        instrument_wavs=live_instruments,
        midi_gain_db=20.0 * math.log10(gain),
        original_left_wav=original_left,
        transcription_right_wav=transcription_right,
        instrument_right_wavs=instrument_rights,
    )


def _write_channelized_playback(
    original_audio: Path,
    transcription_wav: Path,
    instrument_wavs: dict[str, Path],
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Path]]:
    import numpy as np
    import soundfile as sf

    transcription = _load_mono_44k(transcription_wav)
    if len(transcription) <= 0:
        raise RuntimeError(f"Live MIDI transcription is empty: {transcription_wav}")
    original = _load_reference_mono_44k(original_audio)
    if len(original) >= len(transcription):
        original = original[: len(transcription)]
    else:
        original = np.pad(original, (0, len(transcription) - len(original)))
    original_left = output_dir / "original-left.wav"
    sf.write(original_left, np.stack([original, np.zeros_like(original)], axis=1), _SAMPLE_RATE)

    transcription_right = output_dir / "midi-live-right.wav"
    sf.write(
        transcription_right,
        np.stack([np.zeros_like(transcription), transcription], axis=1),
        _SAMPLE_RATE,
    )

    right_paths: dict[str, Path] = {}
    for instrument, source in instrument_wavs.items():
        mono = _load_mono_44k(source)
        right = output_dir / f"{_safe_asset_name(instrument)}-right.wav"
        sf.write(right, np.stack([np.zeros_like(mono), mono], axis=1), _SAMPLE_RATE)
        right_paths[instrument] = right
    return original_left, transcription_right, right_paths


def _write_official_stereo_mix(
    original_audio: Path,
    live_transcription_wav: Path,
    output_path: Path,
) -> None:
    import numpy as np
    import soundfile as sf

    original = _load_reference_mono_44k(original_audio)
    synthesis = _load_mono_44k(live_transcription_wav)
    length = max(len(original), len(synthesis))
    original = np.pad(original, (0, length - len(original)))
    synthesis = np.pad(synthesis, (0, length - len(synthesis)))
    sf.write(output_path, np.stack([original, synthesis], axis=1), _SAMPLE_RATE)


def prepare_midi_playback_assets(
    midi_path: str | Path,
    original_audio_path: str | Path,
    output_dir: str | Path,
    *,
    progress_callback=None,
    cancel_check=None,
    muscriptor_groups: bool = False,
) -> MuscriptorPlaybackAssets:
    """Create every real audio artifact behind the shared result controls."""

    midi_path = Path(midi_path).resolve()
    original_audio = Path(original_audio_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not midi_path.is_file() or not original_audio.is_file():
        raise FileNotFoundError(
            f"MIDI playback input missing: midi={midi_path}, audio={original_audio}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = progress_callback or (lambda _progress, _message: None)
    check_cancelled = cancel_check or (lambda: False)

    def checkpoint() -> None:
        if check_cancelled():
            raise InterruptedError("MuScriptor playback rendering cancelled")

    checkpoint()
    report(0.02, "Validating FluidSynth")
    executable = get_fluidsynth_executable()
    report(0.05, "Preparing the official MuseScore General SoundFont")
    soundfont = download_muscriptor_soundfont(printer=lambda message: report(0.07, message))
    notes = read_midi_roll_notes(midi_path, muscriptor_groups=muscriptor_groups)
    if not notes:
        raise RuntimeError("MIDI result contains no completed notes to play")
    last_note_end = max(note.end for note in notes)
    source_frames = len(_load_reference_mono_44k(original_audio))
    if source_frames <= 0:
        raise RuntimeError(f"Playback reference audio is empty: {original_audio}")
    transport_boundary = max(source_frames / _SAMPLE_RATE, last_note_end)

    grouped: dict[str, list[MuscriptorRollNote]] = defaultdict(list)
    for note in notes:
        grouped[note.instrument].append(note)

    transcription_wav = output_dir / "transcription.wav"
    report(0.12, "Rendering the full transcription")
    _synthesize(
        executable,
        soundfont,
        midi_path,
        transcription_wav,
        cancel_check=check_cancelled,
    )
    instrument_wavs: dict[str, Path] = {}
    instruments = list(grouped)
    if len(instruments) == 1:
        # Use the backend's complete MIDI directly so controller data such as
        # ByteDance CC64 pedal events remains audible in the single piano bus.
        instrument_wavs[instruments[0]] = transcription_wav
        report(0.80, f"Rendered {instruments[0].replace('_', ' ')}")
    else:
        for index, instrument in enumerate(instruments):
            checkpoint()
            asset_name = _safe_asset_name(instrument)
            instrument_midi = output_dir / f"{asset_name}.mid"
            instrument_wav = output_dir / f"{asset_name}.wav"
            _write_instrument_midi(grouped[instrument], instrument_midi)
            _synthesize(
                executable,
                soundfont,
                instrument_midi,
                instrument_wav,
                cancel_check=check_cancelled,
            )
            instrument_wavs[instrument] = instrument_wav
            report(
                0.18 + (index + 1) / len(instruments) * 0.62,
                f"Rendering {instrument.replace('_', ' ')}",
            )

    checkpoint()
    report(0.84, "Preparing loudness-matched live MIDI buses")
    live_buses = _write_live_playback_buses(
        original_audio,
        instrument_wavs,
        output_dir,
        combined_source=transcription_wav,
        target_duration=transport_boundary,
        include_stereo=True,
        cancel_check=check_cancelled,
    )
    if (
        live_buses.original_left_wav is None
        or live_buses.transcription_right_wav is None
        or live_buses.instrument_right_wavs is None
    ):
        raise RuntimeError("Stereo MIDI playback buses were not created")
    import soundfile as sf

    live_info = sf.info(str(live_buses.transcription_wav))
    transport_duration = live_info.frames / live_info.samplerate
    if last_note_end > transport_duration + 1e-9:
        raise RuntimeError(
            "Rendered MIDI playback audio ends before the final note: "
            f"rendered={transport_duration}, note_end={last_note_end}"
        )
    for playback_path in (
        live_buses.original_left_wav,
        live_buses.transcription_right_wav,
        *live_buses.instrument_wavs.values(),
        *live_buses.instrument_right_wavs.values(),
    ):
        info = sf.info(str(playback_path))
        if info.frames != live_info.frames or info.samplerate != live_info.samplerate:
            raise RuntimeError(
                "MIDI playback buses do not share one transport duration: "
                f"master={live_buses.transcription_wav} ({live_info.frames}), "
                f"mismatch={playback_path} ({info.frames})"
            )
    stereo_mix = output_dir / "original-and-midi-stereo.wav"
    checkpoint()
    report(0.92, "Rendering the downloadable stereo mix")
    _write_official_stereo_mix(original_audio, live_buses.transcription_wav, stereo_mix)
    report(1.0, "MuScriptor playback assets ready")
    return MuscriptorPlaybackAssets(
        notes=notes,
        duration=transport_duration,
        transcription_wav=transcription_wav,
        live_transcription_wav=live_buses.transcription_wav,
        stereo_mix_wav=stereo_mix,
        original_left_wav=live_buses.original_left_wav,
        transcription_right_wav=live_buses.transcription_right_wav,
        instrument_wavs=live_buses.instrument_wavs,
        instrument_right_wavs=live_buses.instrument_right_wavs,
        midi_gain_db=live_buses.midi_gain_db,
    )


def prepare_muscriptor_playback_assets(
    midi_path: str | Path,
    original_audio_path: str | Path,
    output_dir: str | Path,
    *,
    progress_callback=None,
    cancel_check=None,
) -> MuscriptorPlaybackAssets:
    """Compatibility wrapper for MuScriptor's grouped program vocabulary."""

    return prepare_midi_playback_assets(
        midi_path,
        original_audio_path,
        output_dir,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        muscriptor_groups=True,
    )
