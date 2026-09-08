"""Shared Gradio project workflow for Space and Colab.

The JSON and checkpoint runner are identical to desktop/CLI/API. Cloud sessions
can transfer a verified project archive when their storage is ephemeral.
"""

from __future__ import annotations

import queue
import threading
import uuid
import json
import secrets
import hashlib
from contextvars import copy_context
from pathlib import Path

from src.core.manual_midi import MANUAL_MIDI_ROUTES
from src.gui.web.track_mixer_runtime import TRACK_COLORS, build_track_mixer_html
from src.gui.web.project_files import cache_project_file
from src.i18n.translator import Translator
from src.models.data_models import (
    MultiInstrumentModel,
    MuscriptorModel,
    ProcessingMode,
    YourMT3Model,
)
from src.projects import ProjectRunner, ProjectStore, Workflow
from src.projects.labels import labels
from src.projects.service import ProjectService
from src.projects.store import SPLIT_STEMS, TrackView, state_access
from src.web_api.schemas import InferenceOptions, ManualMidiOptions


def build_project_workbench(
    root, language="zh_CN", *, engine_factory=None, preview_registry=None, sheet_registry=None
):
    import gradio as gr

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    key_path = root / ".browser-state-key"
    with state_access(root):
        if not key_path.exists():
            with key_path.open("x", encoding="utf-8") as handle:
                handle.write(secrets.token_urlsafe(32))
        browser_secret = key_path.read_text(encoding="utf-8").strip()
        if len(browser_secret) < 32:
            raise RuntimeError("项目库的浏览器状态密钥无效")
    storage_key = "music-to-midi-project-" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    service = ProjectService(root, engine_factory)
    words = labels(language)
    translator = Translator(language)
    runs = {}
    run_guard = threading.Lock()

    def store_for(token):
        if not token:
            raise gr.Error(words["empty"])
        return service.store(token)

    def load(token):
        if not token:
            return {}, words["empty"], gr.update(choices=[], value=None)
        snapshot = service.snapshot(token)
        names = [
            (f"{song['name']} · {words[song['status']]}", song["id"]) for song in snapshot["songs"]
        ]
        return (
            snapshot,
            f"{snapshot['name']} · {words[snapshot['status']]} · {snapshot['counts']['succeeded']}/{snapshot['counts']['total']}",
            gr.update(choices=names, value=names[0][1] if names else None),
        )

    def create(name):
        snapshot = service.create(name)
        return snapshot["storage_id"], *load(snapshot["storage_id"])

    def add(token, files):
        store_for(token).add(files or [])
        return load(token)

    def export(token):
        store = store_for(token)
        return str(store.export_archive(store.root / "project.mtmproject"))

    def import_project(file):
        if not file:
            raise gr.Error(words["empty"])
        token = uuid.uuid4().hex
        ProjectStore.import_archive(file, root / token)
        return token, *load(token)

    def configure(token, song_id, scope, mode, backend, checkpoint, size, selected_stems, *routes):
        store = store_for(token)
        document = store.load()
        selected = next((s for s in document.songs if s.id == song_id), None)
        base = (selected.workflow if selected else document.workflow).model_dump(mode="json")
        base["primary"].update(
            processing_mode=mode,
            transcription_backend=backend,
            yourmt3_model=checkpoint,
            muscriptor_model=size,
            language=language,
        )
        all_stems = ("vocals", "accompaniment", "bass", "drums", "guitar", "piano", "other")
        base["stems"] = {
            **{
                key: value
                for key, value in base["stems"].items()
                if mode in SPLIT_STEMS and key.startswith("custom-")
            },
            **{
                stem: {"route": route, "language": language}
                for stem, route in zip(all_stems, routes)
                if stem in (selected_stems or [])
            },
        }
        workflow = Workflow.model_validate(base)
        store.configure(workflow, None if scope == "all" else [song_id] if song_id else [])
        return load(token)

    def import_profile(token, song_id, scope, file):
        if not file:
            raise gr.Error(words["empty"])
        store_for(token).configure(
            Workflow.load(file), None if scope == "all" else [song_id] if song_id else []
        )
        return load(token)

    def export_profile(token, song_id):
        store = store_for(token)
        doc = store.load()
        song = next((s for s in doc.songs if s.id == song_id), None)
        path = store.root / "workflow.json"
        (song.workflow if song else doc.workflow).save(path)
        return str(path)

    def execute(token, song_ids=None, track_ids=None):
        store = store_for(token)
        events = queue.Queue()
        runner = ProjectRunner(store, engine_factory, events.put)
        with run_guard:
            if token in runs:
                raise gr.Error(words["processing_hint"])
            runs[token] = runner

        def work():
            try:
                events.put(("done", runner.run(song_ids, track_ids=track_ids)))
            except Exception as exc:
                events.put(("error", exc))
            finally:
                with run_guard:
                    runs.pop(token, None)

        thread = threading.Thread(
            target=copy_context().run, args=(work,), name=f"gradio-project-{token}"
        )
        thread.start()
        try:
            while True:
                event = events.get()
                if isinstance(event, tuple):
                    if event[0] == "error":
                        raise gr.Error(str(event[1]))
                    snapshot, text, songs = load(token)
                    yield snapshot, text, songs
                    break
                progress = event["progress"]
                text = f"{progress.get('index', 0)}/{progress.get('total', len(event['songs']))} · {progress.get('song_name', '')} · {progress.get('percent', 0)}% · {progress.get('message', '')}"
                yield gr.skip(), text, gr.skip()
        finally:
            if thread.is_alive():
                runner.cancel()
            thread.join()

    def stop(token):
        with run_guard:
            runner = runs.get(token)
        if runner:
            runner.cancel()
        return words["cancelling"] if runner else load(token)[1]

    def run_all(token):
        yield from execute(token)

    def midi_editor_button(store, song_id, signature):
        if preview_registry is None or sheet_registry is None:
            return
        from src.gui.web.project_midi_result import render_saved_midi

        button = gr.Button(words["result"])
        editor = gr.HTML()

        def open_editor():
            return render_saved_midi(
                store, song_id, signature, translator, preview_registry, sheet_registry
            )

        button.click(
            open_editor, outputs=[editor], concurrency_id="music-to-midi-gpu", concurrency_limit=1
        )

    with gr.Accordion(words["title"], open=True):
        gr.Markdown(words["portable_hint"])
        token = gr.BrowserState("", storage_key=storage_key, secret=browser_secret)
        snapshot = gr.State({})
        name = gr.Textbox(label=words["name"], value=words["title"])
        with gr.Row():
            new = gr.Button(words["new"])
            refresh = gr.Button(words["open"])
        status = gr.Markdown(words["empty"])
        selected_song = gr.Dropdown(label=words["song"], choices=[], interactive=True)
        audio_files = gr.File(
            label=words["files"],
            file_count="multiple",
            type="filepath",
            file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"],
        )
        add_button = gr.Button(words["add"])
        scope = gr.Radio(
            [(words["selected"], "selected"), (words["all"], "all")],
            value="all",
            label=words["scope"],
        )
        modes = [
            (translator.t(f"main.mode.{m.value}"), m.value)
            for m in ProcessingMode
            if m is not ProcessingMode.PIANO
        ]
        mode = gr.Dropdown(modes, value="smart", label=words["mode"], interactive=True)
        backend = gr.Dropdown(
            [m.value for m in MultiInstrumentModel],
            value="yourmt3",
            label=words["route"],
            interactive=True,
        )
        checkpoint = gr.Dropdown(
            [m.value for m in YourMT3Model if m is not YourMT3Model.LEGACY_MC13],
            value="yptf_moe_multi_nops",
            label="YourMT3+",
            interactive=True,
        )
        size = gr.Dropdown(
            [m.value for m in MuscriptorModel], value="large", label="MuScriptor", interactive=True
        )
        stem_ids = ("vocals", "accompaniment", "bass", "drums", "guitar", "piano", "other")
        selected_stems = gr.CheckboxGroup(
            [], value=[], label=words["stems"], visible=False, interactive=True
        )
        routes = [
            gr.Dropdown(
                list(MANUAL_MIDI_ROUTES),
                value=MANUAL_MIDI_ROUTES[0],
                label=stem,
                visible=False,
                interactive=True,
            )
            for stem in stem_ids
        ]

        def restore_settings(current, song_id):
            song = next((s for s in current.get("songs", []) if s["id"] == song_id), None)
            workflow = song["workflow"] if song else current.get("workflow")
            if not workflow:
                return [gr.skip()] * (5 + len(routes))
            primary = workflow["primary"]
            return [
                primary["processing_mode"],
                primary["transcription_backend"],
                primary["yourmt3_model"],
                primary["muscriptor_model"],
                gr.update(
                    choices=list(SPLIT_STEMS.get(primary["processing_mode"], ())),
                    value=[key for key in workflow["stems"] if key in stem_ids],
                    visible=primary["processing_mode"] in SPLIT_STEMS,
                ),
                *[
                    gr.update(
                        value=workflow["stems"].get(stem, {}).get("route", MANUAL_MIDI_ROUTES[0]),
                        visible=stem in workflow["stems"],
                    )
                    for stem in stem_ids
                ],
            ]

        def show_stems(mode_value, selected):
            allowed = SPLIT_STEMS.get(mode_value, ())
            chosen = [stem for stem in selected or [] if stem in allowed]
            return [
                gr.update(choices=list(allowed), value=chosen, visible=bool(allowed)),
                *[gr.update(visible=stem in chosen) for stem in stem_ids],
            ]

        selected_song.change(
            restore_settings,
            [snapshot, selected_song],
            [mode, backend, checkpoint, size, selected_stems, *routes],
            queue=False,
        )
        snapshot.change(
            restore_settings,
            [snapshot, selected_song],
            [mode, backend, checkpoint, size, selected_stems, *routes],
            queue=False,
        )
        mode.input(show_stems, [mode, selected_stems], [selected_stems, *routes], queue=False)
        selected_stems.input(
            lambda chosen: [gr.update(visible=stem in chosen) for stem in stem_ids],
            [selected_stems],
            routes,
            queue=False,
        )
        apply_button = gr.Button(words["save"])
        with gr.Row():
            start_button = gr.Button(words["run"], variant="primary")
            stop_button = gr.Button(words["stop"])
        archive_upload = gr.File(label=words["import"], file_types=[".mtmproject"], type="filepath")
        import_button = gr.Button(words["import"])
        export_button = gr.Button(words["export"])
        archive_output = gr.File(label=words["export"], interactive=False)
        profile_upload = gr.File(label=words["profile_load"], file_types=[".json"], type="filepath")
        profile_load = gr.Button(words["profile_load"])
        profile_save = gr.Button(words["profile_save"])
        profile_output = gr.File(label=words["profile_save"], interactive=False)
        outputs = [snapshot, status, selected_song]
        new.click(create, [name], [token, *outputs], queue=False)
        refresh.click(load, [token], outputs, queue=False)
        add_button.click(add, [token, audio_files], outputs, queue=False)
        import_button.click(import_project, [archive_upload], [token, *outputs], queue=False)
        export_button.click(export, [token], [archive_output], queue=False)
        apply_button.click(
            configure,
            [token, selected_song, scope, mode, backend, checkpoint, size, selected_stems, *routes],
            outputs,
            queue=False,
        )
        profile_load.click(
            import_profile, [token, selected_song, scope, profile_upload], outputs, queue=False
        )
        profile_save.click(export_profile, [token, selected_song], [profile_output], queue=False)
        start_button.click(
            run_all, [token], outputs, concurrency_id="music-to-midi-gpu", concurrency_limit=1
        )
        stop_button.click(stop, [token], [status], queue=False)
        token.change(load, [token], outputs, queue=False)

        @gr.render(inputs=[snapshot, selected_song])
        def render_project_result(current, song_id):
            if not current or not song_id:
                return
            store = store_for(current["storage_id"])
            song = next((s for s in current["songs"] if s["id"] == song_id), None)
            if not song or not song["primary_key"]:
                return
            primary = song["checkpoints"][song["primary_key"]]
            if primary["status"] != "succeeded":
                return
            from src.projects.store import Asset

            files = [
                cache_project_file(store.verify(Asset.model_validate(a)))
                for a in primary["artifacts"]
            ]
            gr.File(value=files, file_count="multiple", label=words["download"], interactive=False)
            if any(a["kind"] == "midi" for a in primary["artifacts"]):
                midi_editor_button(store, song_id, primary["signature"])
            audio_assets = [a for a in primary["artifacts"] if a["kind"] == "audio_track"] + list(
                song.get("extra_tracks", {}).values()
            )
            tracks = [
                dict(
                    id=a["track_id"],
                    name=a["track_id"],
                    audio_path=cache_project_file(store.verify(Asset.model_validate(a))),
                    color=TRACK_COLORS[i % len(TRACK_COLORS)],
                    view=song["mixer"].get(a["track_id"], {}),
                )
                for i, a in enumerate(audio_assets)
            ]
            if tracks:
                mixer_id = f"project-mixer-{song_id}"
                gr.HTML(build_track_mixer_html(tracks, translator.t), elem_id=mixer_id)
                mix_json = gr.Textbox(visible=False)
                save_mix = gr.Button(words["save_mix"])

                def save_mixer(raw, selected_id=song_id):
                    latest = store.load()
                    selected = next(s for s in latest.songs if s.id == selected_id)
                    views = {
                        key: TrackView.model_validate(value)
                        for key, value in json.loads(raw).items()
                    }
                    store.update_song_state(
                        selected_id, views, selected.workflow.stems, latest.revision
                    )
                    return load(current["storage_id"])

                mixer_js = (
                    "() => { const rows = document.querySelectorAll('#"
                    + mixer_id
                    + " .mtm-track'); return [JSON.stringify(Object.fromEntries([...rows].map(row => [row.dataset.trackId, {muted:row.querySelector('.mtm-mute').classList.contains('mtm-active'),solo:row.querySelector('.mtm-solo').classList.contains('mtm-active'),gain_db:Number(row.querySelector('.mtm-volume input').value),offset:Number(row.querySelector('.mtm-offset input').value)}])))]; }"
                )
                save_mix.click(save_mixer, [mix_json], outputs, js=mixer_js, queue=False)
                extra_files = gr.File(
                    label=words["add_tracks"], file_count="multiple", type="filepath"
                )
                add_tracks = gr.Button(words["add_tracks"])

                def add_extra(files, selected_id=song_id):
                    for path in files or []:
                        store.add_track(selected_id, path)
                    return load(current["storage_id"])

                add_tracks.click(add_extra, [extra_files], outputs, queue=False)
            for track in tracks:
                stem = track["id"]
                gr.Markdown(f"**{stem}**")
                chosen = song["workflow"]["stems"].get(stem)
                route = gr.Dropdown(
                    list(MANUAL_MIDI_ROUTES),
                    value=chosen["route"] if chosen else MANUAL_MIDI_ROUTES[0],
                    label=words["route"],
                    interactive=True,
                    key=f"project-route-{song_id}-{stem}",
                )
                convert = gr.Button(
                    translator.t("toolbar.start_convert"), key=f"project-convert-{song_id}-{stem}"
                )

                def convert_one(
                    current_token, selected_route, selected_stem=stem, selected_id=song_id
                ):
                    target = store_for(current_token)
                    doc = target.load()
                    current_song = next(s for s in doc.songs if s.id == selected_id)
                    workflow = current_song.workflow.model_copy(deep=True)
                    workflow.stems[selected_stem] = ManualMidiOptions(
                        route=selected_route, language=language
                    )
                    target.configure(workflow, [selected_id])
                    yield from execute(current_token, [selected_id], [selected_stem])

                convert.click(
                    convert_one,
                    [token, route],
                    outputs,
                    concurrency_id="music-to-midi-gpu",
                    concurrency_limit=1,
                )
                key = song["track_keys"].get(stem)
                step = song["checkpoints"].get(key)
                if step and step["status"] == "succeeded":
                    gr.File(
                        value=[
                            cache_project_file(store.verify(Asset.model_validate(a)))
                            for a in step["artifacts"]
                        ],
                        file_count="multiple",
                        label=words["download"],
                        interactive=False,
                    )
                    midi_editor_button(store, song_id, step["signature"])
                elif step and step["error"]:
                    gr.Markdown(f"{words['error']}: {step['error']}")

    return service
