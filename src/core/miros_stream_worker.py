"""Project-owned MIROS launcher that exposes real inference-batch snapshots.

The pinned external source remains untouched and still performs its own final
post-processing and official MIDI write.  This module only wraps the model's
batch loop so the parent process can see stable decoded prefixes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.core.transcription_stream import (
    append_jsonl_event,
    model_notes_payload,
    snapshot_event,
)


def run_miros_stream_worker(
    repo_dir: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    events_path: str | Path,
) -> str:
    repo = Path(repo_dir).resolve()
    source = Path(audio_path).resolve()
    destination = Path(output_path).resolve()
    event_file = Path(events_path).resolve()
    if not (repo / "transcribe.py").is_file():
        raise FileNotFoundError(f"MIROS transcribe.py is missing: {repo}")

    sys.path.insert(0, str(repo))
    previous_cwd = Path.cwd()
    try:
        os.chdir(repo)
        import soundfile as sf
        from model.ymt3 import YourMT3
        from utils.event2note import merge_zipped_note_events_and_ties_to_notes
        from utils.note2event import mix_notes
        import transcribe as transcribe_module

        info = sf.info(str(source))
        duration_seconds = info.frames / float(info.samplerate)
        original_inference_file = YourMT3.inference_file

        def inference_file_with_events(
            model,
            bsz,
            audio_segments,
            note_token_array=None,
            task_token_array=None,
        ):
            # The pinned transcribe.py calls this with both optional arrays
            # absent.  Refuse an unverified training/evaluation invocation.
            if note_token_array is not None or task_token_array is not None:
                raise RuntimeError(
                    "MIROS streamed worker only supports the official inference-only call"
                )
            n_items = int(audio_segments.shape[0])
            segment_seconds = model.audio_cfg["input_frames"] / model.audio_cfg["sample_rate"]
            predicted_batches = []
            for batch_start in range(0, n_items, bsz):
                batch_end = min(batch_start + bsz, n_items)
                x = audio_segments[batch_start:batch_end].to(model.device)
                task_tokens = None
                if model.test_pitch_shift_layer is not None:
                    predictions, _shifted = model.inference(x, task_tokens)
                else:
                    predictions = model.inference(x, task_tokens)
                predictions = predictions.detach().cpu().numpy()
                if len(predictions) != len(x):
                    raise ValueError(
                        "MIROS streamed inference length mismatch: "
                        f"predictions={len(predictions)}, audio={len(x)}"
                    )
                predicted_batches.append(predictions)

                completed = batch_end
                start_seconds = [segment_seconds * index for index in range(completed)]
                notes_by_channel = []
                for channel in range(model.task_manager.num_decoding_channels):
                    channel_batches = [batch[:, channel, :] for batch in predicted_batches]
                    zipped, _, _ = model.task_manager.detokenize_list_batches(
                        channel_batches,
                        start_seconds,
                        return_events=True,
                    )
                    channel_notes, _ = merge_zipped_note_events_and_ties_to_notes(zipped)
                    notes_by_channel.append(channel_notes)
                decoded_notes = mix_notes(notes_by_channel)
                frontier = (
                    duration_seconds
                    if completed == n_items
                    else max(0.0, (completed - 1) * segment_seconds)
                )
                append_jsonl_event(
                    event_file,
                    snapshot_event(
                        backend="MIROS (MusicFM)",
                        completed=completed,
                        total=n_items,
                        frontier_seconds=frontier,
                        duration_seconds=duration_seconds,
                        notes=model_notes_payload(
                            decoded_notes,
                            frontier_seconds=frontier,
                            inverse_vocab=model.midi_output_inverse_vocab,
                        ),
                    ),
                )
            return predicted_batches, None

        YourMT3.inference_file = inference_file_with_events
        try:
            result = transcribe_module.transcribe(str(source), str(destination))
        finally:
            YourMT3.inference_file = original_inference_file
        return str(result)
    finally:
        os.chdir(previous_cwd)
        try:
            sys.path.remove(str(repo))
        except ValueError:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--events-jsonl", required=True)
    args = parser.parse_args(argv)
    run_miros_stream_worker(
        args.repo_dir,
        args.input,
        args.output,
        args.events_jsonl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
