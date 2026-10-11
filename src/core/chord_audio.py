"""SoundFont monitoring audio from native chord intervals, independent of MIDI exports."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from src.core.midi_chords import MidiChord
from src.core.muscriptor_result_assets import render_midi_audio_export

CHORD_SAMPLE_RATE = 44_100
CHORD_MAX_VOLUME = 200
CHORD_DEFAULT_VOLUME = 100
_MONITOR_RMS = 10 ** (-18 / 20)
_MONITOR_PEAK = 0.475
_EDGE_FRAMES = 220


def chord_midi(chords: tuple[MidiChord, ...], frames: int, offset: float = 0.0):
    """Use one MIDI tick per PCM frame; display-only bar splits never enter here."""
    if frames <= 0 or not math.isfinite(offset):
        raise ValueError("Chord monitoring requires a positive duration and finite offset")
    events = []
    intervals = []
    previous_end = 0.0
    for chord in chords:
        if (
            not math.isfinite(chord.start)
            or not math.isfinite(chord.end)
            or chord.start < previous_end - 1e-9
            or chord.end <= chord.start
        ):
            raise ValueError("Chord intervals must be finite, ordered and non-overlapping")
        previous_end = chord.end
        start = max(0, round((chord.start + offset) * CHORD_SAMPLE_RATE))
        end = min(frames, round((chord.end + offset) * CHORD_SAMPLE_RATE))
        if not chord.pitches or start >= end:
            continue
        intervals.append((start, end))
        for pitch in chord.pitches:
            events.append((start, 1, mido.Message("note_on", note=pitch, velocity=82)))
            events.append((end, 0, mido.Message("note_off", note=pitch, velocity=0)))
    midi = mido.MidiFile(ticks_per_beat=22_050)
    track = mido.MidiTrack(
        [mido.MetaMessage("set_tempo", tempo=500_000), mido.Message("program_change", program=0)]
    )
    midi.tracks.append(track)
    previous = 0
    for frame, _, message in sorted(events, key=lambda event: (event[0], event[1])):
        track.append(message.copy(time=frame - previous))
        previous = frame
    track.append(mido.MetaMessage("end_of_track", time=frames - previous))
    return midi, intervals


def render_chord_pcm(
    chords: tuple[MidiChord, ...],
    frames: int,
    cache: Path,
    *,
    offset: float = 0.0,
    selection=None,
    cancel_check=None,
) -> np.ndarray:
    """Render stereo PCM at 100%; leave headroom for the full 200% control range."""
    cancelled = cancel_check or (lambda: False)

    def checkpoint():
        if cancelled():
            raise InterruptedError("Chord audio rendering cancelled")

    checkpoint()
    midi, intervals = chord_midi(chords, frames, offset)
    samples = np.zeros((frames, 2), dtype="float32")
    if not intervals:
        samples.setflags(write=False)
        return samples
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chord-render-", dir=cache) as directory:
        source = Path(directory) / "chords.mid"
        wave = Path(directory) / "chords.wav"
        midi.save(source)
        render_midi_audio_export(
            source, wave, "pcm16_44100", cancel_check=cancelled, soundfont_selection=selection
        )
        checkpoint()
        with sf.SoundFile(wave) as audio:
            if audio.samplerate != CHORD_SAMPLE_RATE or audio.channels not in (1, 2):
                raise RuntimeError("Chord renderer returned an unsupported PCM format")
            for start, end in intervals:
                checkpoint()
                if end > audio.frames:
                    raise RuntimeError("Chord renderer truncated a native chord interval")
                audio.seek(start)
                block = audio.read(end - start, dtype="float32", always_2d=True)
                edge = min(_EDGE_FRAMES, len(block) // 2)
                if edge:
                    fade = np.linspace(0, 1, edge, dtype="float32")[:, None]
                    block[:edge] *= fade
                    block[-edge:] *= fade[::-1]
                samples[start:end] = block
    if not np.isfinite(samples).all():
        raise RuntimeError("Chord renderer returned non-finite audio")
    peak = float(np.max(np.abs(samples)))
    active_frames = sum(end - start for start, end in intervals)
    energy = sum(
        float(np.einsum("ij,ij->", block, block, dtype="float64"))
        for block in np.array_split(samples, max(1, frames // CHORD_SAMPLE_RATE))
    )
    rms = math.sqrt(energy / (active_frames * 2))
    if rms <= 1e-8 or peak <= 1e-8:
        raise RuntimeError("SoundFont produced silent chord monitoring audio")
    samples *= min(_MONITOR_RMS / rms, _MONITOR_PEAK / peak)
    checkpoint()
    samples.setflags(write=False)
    return samples
