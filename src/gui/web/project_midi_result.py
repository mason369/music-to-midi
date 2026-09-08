"""Open checkpoint MIDI files in the existing shared Gradio piano-roll editor."""

from dataclasses import asdict

from src.core.midi_tempo import rewrite_midi_tempo_preserving_ticks, validated_midi_time_signature
from src.core.muscriptor_result_assets import prepare_midi_playback_assets
from src.gui.web.muscriptor_result_runtime import build_muscriptor_result_html
from src.gui.web.project_files import cache_project_file
from src.projects.store import exclusive


def render_saved_midi(store, song_id, signature, translator, previews, sheets):
    with exclusive(store.run_lock):
        doc = store.load()
        song = next(s for s in doc.songs if s.id == song_id)
        step = song.checkpoints[signature]
        if step.status != "succeeded":
            raise ValueError("MIDI 步骤尚未完成")
        midi = next(a for a in step.artifacts if a.kind == "midi")
        source = song.source
        if step.kind == "manual_midi":
            primary = song.checkpoints[song.primary_key]
            source = song.extra_tracks.get(midi.track_id) or next(
                a
                for a in primary.artifacts
                if a.kind == "audio_track" and a.track_id == midi.track_id
            )
        original, midi_path = store.verify(source), store.verify(midi)
        beat = step.result.get("beat")
        if not beat:
            raise ValueError("项目缺少 MIDI 节拍信息，无法打开编辑器")
        reference_bpm = beat["source_bpm"] if beat.get("source_bpm") is not None else beat["bpm"]
        output = store.root / "previews" / signature
        output.mkdir(parents=True, exist_ok=True)
        playback = rewrite_midi_tempo_preserving_ticks(
            midi_path, output / "source-tempo.mid", reference_bpm, label="Project MIDI playback"
        )
        groups = step.result.get("transcription_backend") == "muscriptor"
        assets = prepare_midi_playback_assets(playback, original, output, muscriptor_groups=groups)
        detected = list(dict.fromkeys(n.instrument for n in assets.notes))
        state = dict(
            notes=[asdict(n) for n in assets.notes],
            duration=assets.duration,
            reference_bpm=reference_bpm,
            target_bpm=beat["bpm"],
            time_signature=validated_midi_time_signature(beat.get("time_signature") or (4, 4)),
            fixed_tempo_reliable=beat.get("fixed_tempo_reliable"),
            tempo_warning=beat.get("tempo_warning"),
            audio_path=str(original),
            playback_audio_path=cache_project_file(assets.original_wav),
            midi_path=cache_project_file(midi_path),
            transcription_wav=cache_project_file(assets.transcription_wav),
            stereo_mix_wav=cache_project_file(assets.stereo_mix_wav),
            instrument_wavs={
                name: cache_project_file(p) for name, p in assets.instrument_wavs.items()
            },
            selected_instruments=step.result.get("selected_instruments") or detected,
            detected_instruments=detected,
            backend_label=step.result.get("transcription_backend", ""),
            source_track_name=midi.track_id or "",
            repeat_tempo_per_note_track=groups,
            beat_times=beat.get("beat_times", []),
            downbeats=beat.get("downbeats", []),
            preview_api="./api/render_edited_midi_preview",
            audio_export_api="./api/render_edited_midi_audio_export",
            audio_stem_export_api="./api/render_edited_midi_stem_export",
            sheet_api="./api/render_sheet_music_export",
            preview_token=previews.register(
                request_dir=store.root,
                source_midi_path=midi_path,
                original_audio_path=original,
                reference_bpm=reference_bpm,
                muscriptor_groups=groups,
                duration_seconds=assets.duration,
            ),
            sheet_token=sheets.register(request_dir=store.root, source_midi_path=midi_path),
        )
        return build_muscriptor_result_html(state, translator.t, language=translator.get_language())
