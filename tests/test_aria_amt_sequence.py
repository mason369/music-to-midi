import copy
import random
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("amt")
from amt.inference import transcribe as upstream
from amt.tokenizer import AmtTokenizer

from src.core.aria_amt_sequence import AriaSequenceStitcher
from src.core.aria_amt_transcriber import AriaAmtTranscriber
from src.i18n.translator import Translator


def _generated_window(index, prefix, failures, tokenizer):
    if index in failures:
        return prefix + [("off", 55), ("off", 55), ("off", 55)]
    onsets = [tok[1] for tok in prefix if isinstance(tok, tuple) and tok[0] == "onset"]
    start = 10000 if onsets and max(onsets) > 2 else 0
    generated = list(prefix)
    for onset, pitch in [(start + 500, 60), (18500, 67), (19000, 72)]:
        generated.extend(
            [
                ("on", pitch),
                ("onset", onset),
                ("vel", 80),
                ("off", pitch),
                ("onset", onset + 300),
            ]
        )
    return generated + [tokenizer.eos_tok]


@contextmanager
def _window_runtime(tmp_path, failures=(), duration=60.0, window_count=5):
    tokenizer = AmtTokenizer()
    transcriber = AriaAmtTranscriber()
    calls, events = [], []
    transcriber.set_event_callback(events.append)

    def process_segments(tasks, **kwargs):
        (index, prefix), _ = tasks[0]
        original_prefix = copy.deepcopy(prefix)
        generated = _generated_window(index, original_prefix, failures, tokenizer)
        tokenizer.trunc_seq(prefix, upstream.MAX_BLOCK_LEN)
        calls.append((index, original_prefix))
        return [generated]

    with (
        patch.object(transcriber, "_resolve_runtime_device", return_value="cuda:0"),
        patch.object(transcriber, "_load_aria_model", return_value=(Mock(), tokenizer)),
        patch.object(transcriber, "_iter_windows_wav_segments", return_value=range(window_count)),
        patch("src.core.aria_amt_transcriber.prepare_aria_audio_assets"),
        patch("amt.audio.AudioTransform"),
        patch(
            "src.core.aria_amt_transcriber.torchaudio.info",
            return_value=SimpleNamespace(
                num_frames=int(duration * 16000),
                sample_rate=16000,
            ),
        ),
        patch.object(upstream, "_get_silent_intervals", return_value=[]),
        patch.object(upstream, "process_segments", side_effect=process_segments),
    ):
        yield transcriber, tokenizer, calls, events


@pytest.mark.parametrize("failures", [(), (1,), (1, 3)])
def test_window_loop_matches_official_reconciliation_and_midi(tmp_path, failures):
    with _window_runtime(tmp_path, failures) as (transcriber, tokenizer, calls, events):
        results = {}

        class Queue:
            def put(self, task):
                results[task[1]] = upstream.process_segments(tasks=[task])[0]

        random.seed(17)
        with patch.object(upstream, "get_wav_segments", return_value=range(5)):
            official_sequence = upstream.transcribe_file(
                "input.wav",
                Queue(),
                results,
                0,
                tokenizer,
            )
        official_calls = copy.deepcopy(calls)
        calls.clear()
        random.seed(17)
        transcriber._run_transcription_windows_single_file(tmp_path / "input.wav", tmp_path)

    last_onset = next(
        tok[1]
        for tok in reversed(official_sequence)
        if isinstance(tok, tuple) and tok[0] == "onset"
    )
    official_midi = tokenizer.detokenize(official_sequence, last_onset)
    official_midi.remove_redundant_pedals()
    official_midi.to_midi().save(tmp_path / "official.mid")
    assert calls == official_calls
    assert len(calls) == 5
    assert (tmp_path / "input.mid").read_bytes() == (tmp_path / "official.mid").read_bytes()
    assert [event["completed"] for event in events] == [
        i + 1 for i in range(5) if i not in failures
    ]
    assert events[-1]["frontier_seconds"] == 60.0
    for failed in failures:
        restored_start = (failed + 1) * 10
        assert any(
            restored_start <= note["start"] < restored_start + 10 for note in events[-1]["notes"]
        )


@pytest.mark.parametrize(
    "failures,expected_calls,interval",
    [((0,), 1, "0.000–10.000"), ((1, 2), 3, "20.000–30.000"), ((4,), 5, "50.000–60.000")],
)
def test_unrecoverable_windows_cannot_publish_a_complete_midi(
    tmp_path,
    failures,
    expected_calls,
    interval,
):
    with _window_runtime(tmp_path, failures) as (transcriber, tokenizer, calls, events):
        with pytest.raises(RuntimeError, match=interval) as error:
            transcriber._run_transcription_windows_single_file(tmp_path / "input.wav", tmp_path)
    assert "Unexpected token order" in str(error.value)
    assert len(calls) == expected_calls
    assert not (tmp_path / "input.mid").exists()
    assert all(event["frontier_seconds"] < 60 for event in events)


def test_final_failure_range_is_clipped_to_real_audio_duration(tmp_path):
    with _window_runtime(tmp_path, (4,), duration=50.125) as (transcriber, *_):
        with pytest.raises(RuntimeError, match="50.000–50.125"):
            transcriber._run_transcription_windows_single_file(tmp_path / "input.wav", tmp_path)
    assert not (tmp_path / "input.mid").exists()


def test_generation_error_is_not_treated_as_reconciliation(tmp_path):
    with _window_runtime(tmp_path) as (transcriber, *_):
        with patch.object(
            upstream, "process_segments", side_effect=RuntimeError("GPU_FAILURE")
        ) as run:
            with pytest.raises(RuntimeError, match="GPU_FAILURE"):
                transcriber._run_transcription_windows_single_file(tmp_path / "input.wav", tmp_path)
    run.assert_called_once()
    assert not (tmp_path / "input.mid").exists()


def test_cancel_during_final_generation_does_not_write_midi(tmp_path):
    with _window_runtime(tmp_path) as (transcriber, tokenizer, calls, events):
        process = upstream.process_segments.side_effect

        def cancel_on_final_window(*args, **kwargs):
            generated = process(*args, **kwargs)
            if len(calls) == 5:
                transcriber.cancel()
            return generated

        with patch.object(upstream, "process_segments", side_effect=cancel_on_final_window):
            with pytest.raises(InterruptedError):
                transcriber._run_transcription_windows_single_file(tmp_path / "input.wav", tmp_path)
    assert len(calls) == 5
    assert not (tmp_path / "input.mid").exists()
    assert all(event["frontier_seconds"] < 60 for event in events)


def test_valid_but_context_exhausted_sequence_is_not_complete():
    tokenizer = AmtTokenizer()
    stitcher = AriaSequenceStitcher(upstream, tokenizer, 40, Translator("en_US").t)
    prefix = [tokenizer.bos_tok]
    first = _generated_window(0, prefix, (), tokenizer)
    assert stitcher.append(0, first, prefix)
    prefix = copy.deepcopy(stitcher.sequence)
    partial = _generated_window(1, prefix, (), tokenizer)[:-1]
    partial += [tokenizer.pad_tok] * (upstream.MAX_BLOCK_LEN - len(partial))
    assert not stitcher.append(1, partial, prefix)
    assert stitcher.covered_until_ms == 20000
    with pytest.raises(RuntimeError, match="20.000–40.000.*token limit"):
        stitcher.finish()


def test_silence_processing_cannot_hide_context_exhaustion(tmp_path):
    with _window_runtime(tmp_path) as (transcriber, tokenizer, calls, events):
        process = upstream.process_segments.side_effect

        def exhausted_window(*args, **kwargs):
            generated = process(*args, **kwargs)
            if len(calls) == 2:
                generated[0] += [tokenizer.pad_tok] * (upstream.MAX_BLOCK_LEN - len(generated[0]))
            return generated

        with (
            patch.object(upstream, "process_segments", side_effect=exhausted_window),
            patch.object(
                upstream,
                "_process_silent_intervals",
                side_effect=lambda seq, **kw: [tok for tok in seq if tok != tokenizer.pad_tok],
            ),
        ):
            transcriber._run_transcription_windows_single_file(tmp_path / "input.wav", tmp_path)
    assert len(calls) == 5
    assert [event["completed"] for event in events] == [1, 3, 4, 5]
    assert (tmp_path / "input.mid").is_file()


@pytest.mark.parametrize("language", list(Translator.AVAILABLE_LANGUAGES))
def test_incomplete_error_translations_have_matching_placeholders(language):
    translator = Translator(language)
    template = translator.translations[language]["aria_amt"]["incomplete"]
    message = translator.t("aria_amt.incomplete", start="20.000", end="30.000", error="TEST_ERROR")
    assert all("{" + name + "}" in template for name in ("start", "end", "error"))
    assert "20.000–30.000" in message and "TEST_ERROR" in message
    assert "{" not in message
    assert translator.t("aria_amt.context_exhausted") != "aria_amt.context_exhausted"
