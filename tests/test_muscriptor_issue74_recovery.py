"""Regression coverage for the downstream MuScriptor Issue #74 repair."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import muscriptor.transcription_model as upstream
from muscriptor.events import (
    ChunkBoundary,
    NoteEndEvent,
    NoteStartEvent,
    ProgressEvent,
    decode_model_tokens,
)
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import build_event_vocab

from src.core.muscriptor_boundary_continuity import continuous_chunk_events
from src.core.muscriptor_model_loader import (
    _build_telknet_boundary_recovery_model_class,
)

_FRAME_RATE = 100
_VOCAB = build_event_vocab(1001)
_INDEX = {(event.type, event.value): index for index, event in enumerate(_VOCAB)}


def _tok(event_type: str, value: int = 0) -> int:
    return _INDEX[(event_type, value)]


_EOS = _tok("EOS")


@pytest.fixture(scope="module")
def tokenizer() -> MT3Tokenizer:
    return MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)


def _encode(
    notes: list[tuple[float, int, int, int, bool]],
    *,
    start_time: float = 0.0,
) -> list[int]:
    """Encode the small note scripts needed by the boundary-state tests."""

    tokens = [_tok("tie")]
    start_tick = round(start_time * _FRAME_RATE)
    current_tick = start_tick
    current_program = None
    current_velocity = None
    for time, program, pitch, velocity, is_drum in sorted(
        notes,
        key=lambda item: (
            round(item[0] * _FRAME_RATE),
            item[4],
            item[1],
            item[3],
            item[2],
        ),
    ):
        tick = round(time * _FRAME_RATE)
        if tick < current_tick:
            raise ValueError("Test note script is not monotonic")
        if tick > current_tick:
            tokens.append(_tok("shift", tick - start_tick))
            current_tick = tick
        if is_drum:
            if velocity != 1:
                raise ValueError("Drum test events must be note-ons")
            if current_velocity != 1:
                tokens.append(_tok("velocity", 1))
                current_velocity = 1
            tokens.append(_tok("drum", pitch))
            continue
        if current_program != program:
            tokens.append(_tok("program", program))
            current_program = program
        if current_velocity != velocity:
            tokens.append(_tok("velocity", velocity))
            current_velocity = velocity
        tokens.append(_tok("pitch", pitch))
    return tokens


def _on(time: float, program: int, pitch: int) -> tuple[float, int, int, int, bool]:
    return (time, program, pitch, 1, False)


def _off(time: float, program: int, pitch: int) -> tuple[float, int, int, int, bool]:
    return (time, program, pitch, 0, False)


def _drum(time: float, pitch: int) -> tuple[float, int, int, int, bool]:
    return (time, 128, pitch, 1, True)


def _rows(tokens: list[int]) -> list[list[int]]:
    return [[token] for token in tokens]


def _run_stream(
    scripts,
    tokenizer: MT3Tokenizer,
    *,
    seek_times: list[float],
    declared_multi_instrument_source: bool = False,
):
    prompts: list[list[int] | None] = []

    def generate(prompt=None, **_kwargs):
        prompts.append(None if prompt is None else prompt[0].tolist())
        if prompt is not None:
            for token in prompt[0].tolist():
                yield torch.tensor([token])
        for row in scripts[len(prompts) - 1]:
            yield torch.tensor(row)

    model_class = _build_telknet_boundary_recovery_model_class(upstream, torch)
    fake = SimpleNamespace(
        _model=SimpleNamespace(generate=generate),
        _tokenizer=tokenizer,
        _device=torch.device("cpu"),
        _telknet_declared_multi_instrument_source=declared_multi_instrument_source,
    )
    stream = list(
        model_class._generate_token_stream(
            fake,
            [object()] * len(seek_times),
            seek_times,
            batch_size=1,
            max_gen_len=64,
            use_sampling=False,
            temperature=1.0,
            cfg_coef=1.0,
            no_eos_is_ok=True,
            prelude_forcing=True,
        )
    )
    return stream, prompts


def _decoded_events(stream, tokenizer: MT3Tokenizer):
    return [
        event
        for event in decode_model_tokens(iter(stream), tokenizer._vocab, lambda program: program)
        if not isinstance(event, ProgressEvent)
    ]


def test_empty_open_note_set_never_forces_a_tie_only_prompt(tokenizer):
    chunk0 = _encode([_on(0.1, 0, 60), _off(0.8, 0, 60)]) + [_EOS]
    chunk1 = _encode([], start_time=5.0) + [_EOS]

    _stream, prompts = _run_stream([_rows(chunk0), _rows(chunk1)], tokenizer, seek_times=[0.0, 5.0])

    assert prompts == [None, None]


def test_recurrent_single_program_feedback_injects_one_exact_decoder_tie(tokenizer):
    chunk0 = _encode([_on(0.1, 0, 60), _on(0.2, 52, 64), _off(0.8, 52, 64)]) + [_EOS]
    chunk1 = _encode([], start_time=5.0) + [_EOS]
    generated_duplicate_tie = tokenizer.tie_section_token_ids([(0, 60)])
    chunk2 = generated_duplicate_tie + [
        _tok("shift", 50),
        _tok("program", 0),
        _tok("velocity", 0),
        _tok("pitch", 60),
        _EOS,
    ]

    stream, prompts = _run_stream(
        [_rows(chunk0), _rows(chunk1), _rows(chunk2)],
        tokenizer,
        seek_times=[0.0, 5.0, 10.0],
    )

    assert prompts[1] == tokenizer.tie_section_token_ids([(0, 60)])
    assert prompts[2] is None
    boundary_index = stream.index(ChunkBoundary(10.0, None))
    injected_tie = tokenizer.tie_section_token_ids([(0, 60)])
    assert stream[boundary_index + 1 : boundary_index + 1 + len(injected_tie)] == injected_tie
    assert stream[boundary_index + 1 + len(injected_tie)] == _tok("shift", 50)

    events = _decoded_events(stream, tokenizer)
    piano_end = next(
        event
        for event in events
        if isinstance(event, NoteEndEvent) and event.start_event.instrument == 0
    )
    assert piano_end.end_time == pytest.approx(10.5)


def test_true_solo_source_keeps_official_teacher_forcing(tokenizer):
    chunk0 = _encode([_on(0.1, 0, 60)]) + [_EOS]
    chunk1 = _encode([], start_time=5.0) + [_EOS]
    chunk2 = _encode([_off(10.5, 0, 60)], start_time=10.0) + [_EOS]

    _stream, prompts = _run_stream(
        [_rows(chunk0), _rows(chunk1), _rows(chunk2)],
        tokenizer,
        seek_times=[0.0, 5.0, 10.0],
    )

    expected = tokenizer.tie_section_token_ids([(0, 60)])
    assert prompts == [None, expected, expected]


def test_mixed_source_bass_recovers_at_first_reproduced_boundary(tokenizer):
    chunk0 = _encode([_on(0.1, 33, 40), _on(0.2, 52, 64), _off(0.8, 52, 64)]) + [_EOS]
    chunk1 = [
        _tok("shift", 50),
        _tok("program", 33),
        _tok("velocity", 0),
        _tok("pitch", 40),
        _EOS,
    ]

    stream, prompts = _run_stream([_rows(chunk0), _rows(chunk1)], tokenizer, seek_times=[0.0, 5.0])

    assert prompts == [None, None]
    bass_end = next(
        event
        for event in _decoded_events(stream, tokenizer)
        if isinstance(event, NoteEndEvent) and event.start_event.instrument == 33
    )
    assert bass_end.end_time == pytest.approx(5.5)


def test_drum_evidence_only_unlocks_recovery_after_cold_start_grace(tokenizer):
    early0 = _encode([_on(0.1, 0, 60), _drum(0.2, 38)]) + [_EOS]
    early1 = _encode([_drum(5.2, 42)], start_time=5.0) + [_EOS]
    early2 = _encode([_off(10.5, 0, 60)], start_time=10.0) + [_EOS]
    _stream, early_prompts = _run_stream(
        [_rows(early0), _rows(early1), _rows(early2)],
        tokenizer,
        seek_times=[0.0, 5.0, 10.0],
    )

    mature0 = _encode([_on(20.1, 0, 60), _drum(20.2, 38)], start_time=20.0) + [_EOS]
    mature1 = _encode([_drum(25.2, 42)], start_time=25.0) + [_EOS]
    mature2 = _encode([_off(30.5, 0, 60)], start_time=30.0) + [_EOS]
    _stream, mature_prompts = _run_stream(
        [_rows(mature0), _rows(mature1), _rows(mature2)],
        tokenizer,
        seek_times=[20.0, 25.0, 30.0],
    )

    expected = tokenizer.tie_section_token_ids([(0, 60)])
    assert early_prompts == [None, expected, expected]
    assert mature_prompts == [None, expected, None]


def test_declared_multi_instrument_source_is_early_diversity_evidence(tokenizer):
    chunk0 = _encode([_on(0.1, 0, 60)]) + [_EOS]
    chunk1 = _encode([], start_time=5.0) + [_EOS]
    chunk2 = _encode([_off(10.5, 0, 60)], start_time=10.0) + [_EOS]

    _stream, prompts = _run_stream(
        [_rows(chunk0), _rows(chunk1), _rows(chunk2)],
        tokenizer,
        seek_times=[0.0, 5.0, 10.0],
        declared_multi_instrument_source=True,
    )

    expected = tokenizer.tie_section_token_ids([(0, 60)])
    assert prompts == [None, expected, None]


def test_exact_five_second_split_note_is_merged_without_moving_end_time():
    first = NoteStartEvent(pitch=60, start_time=1.0, index=1, instrument="piano")
    restart = NoteStartEvent(pitch=60, start_time=5.02, index=2, instrument="piano")

    rewritten, merged = continuous_chunk_events(
        [first, NoteEndEvent(5.0, first), restart, NoteEndEvent(8.0, restart)],
        NoteStartEvent,
        NoteEndEvent,
    )

    assert merged == 1
    assert rewritten == [first, NoteEndEvent(8.0, first)]


@pytest.mark.parametrize(
    ("instrument", "boundary_end", "restart_time"),
    [
        ("drums", 5.0, 5.01),
        ("piano", 4.99, 5.01),
        ("piano", 5.0, 5.031),
    ],
)
def test_boundary_continuity_never_rewrites_unproven_repetitions(
    instrument,
    boundary_end,
    restart_time,
):
    first = NoteStartEvent(pitch=60, start_time=1.0, index=1, instrument=instrument)
    restart = NoteStartEvent(
        pitch=60,
        start_time=restart_time,
        index=2,
        instrument=instrument,
    )
    original = [
        first,
        NoteEndEvent(boundary_end, first),
        restart,
        NoteEndEvent(8.0, restart),
    ]

    rewritten, merged = continuous_chunk_events(original, NoteStartEvent, NoteEndEvent)

    assert merged == 0
    assert rewritten == original
