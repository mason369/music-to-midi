"""Native project controls embedded in the existing desktop workspace."""

from __future__ import annotations

from pathlib import Path
import uuid

from PyQt6.QtCore import QSignalBlocker, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.manual_midi import MANUAL_MIDI_ROUTES
from src.gui.layouts.flow_layout import FlowLayout
from src.gui.widgets.audio_track_mixer import NoWheelComboBox, midi_route_label
from src.projects import ProjectRunner, ProjectStore, Workflow
from src.projects.labels import labels
from src.projects.store import (
    PROJECT_FILE,
    SPLIT_STEMS,
    TrackView,
    exclusive,
    select_checkpoints,
    plan_complete,
)
from src.web_api.schemas import InferenceOptions, ManualMidiOptions


class ProjectWorker(QThread):
    updated = pyqtSignal(dict)
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        store,
        song_ids=None,
        track_ids=None,
        parent=None,
        engine_factory=None,
        primary_only=False,
    ):
        super().__init__(parent)
        self.runner = ProjectRunner(
            store, engine_factory=engine_factory, on_update=self.updated.emit
        )
        self.song_ids, self.track_ids = song_ids, track_ids
        self.primary_only = primary_only

    def run(self):
        try:
            self.completed.emit(
                self.runner.run(
                    self.song_ids, track_ids=self.track_ids, primary_only=self.primary_only
                )
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def cancel(self):
        self.runner.cancel()


class ProjectPanel(QGroupBox):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.store = None
        self.current_song_id = None
        self._restoring = False
        self.words = labels(owner.config.language)
        self.setTitle(self.words["title"])
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        buttons = FlowLayout(horizontal_spacing=8, vertical_spacing=6)
        self.edit_buttons = []
        self.translated_buttons = {}
        for key, callback in (
            ("new", self.new),
            ("open", self.open),
            ("export", self.export),
            ("add", self.browse),
            ("profile_load", self.load_profile),
            ("profile_save", self.save_profile),
        ):
            button = QPushButton(self.words[key])
            button.setMinimumWidth(0)
            button.clicked.connect(lambda _checked=False, fn=callback: self.guard(fn))
            buttons.addWidget(button)
            self.edit_buttons.append(button)
            self.translated_buttons[key] = button
        layout.addLayout(buttons)
        self.songs = NoWheelComboBox()
        self.songs.setMinimumWidth(0)
        self.songs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.songs.currentIndexChanged.connect(self.select_song)
        layout.addWidget(self.songs)
        self.scope = NoWheelComboBox()
        self.scope.addItem(self.words["selected"], "selected")
        self.scope.addItem(self.words["all"], "all")
        layout.addWidget(self.scope)
        self.stem_group = QGroupBox()
        self.stem_group.setAccessibleName(self.words["stems"])
        stems_layout = QVBoxLayout(self.stem_group)
        self.stem_heading = QLabel(self.words["stems"])
        self.stem_heading.setWordWrap(True)
        self.stem_heading.setMinimumWidth(0)
        self.stem_heading.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        stems_layout.addWidget(self.stem_heading)
        self.stem_controls = {}
        for stem in ("vocals", "accompaniment", "bass", "drums", "guitar", "piano", "other"):
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            check = QCheckBox(stem)
            route = NoWheelComboBox()
            route.setMinimumWidth(0)
            route.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            for value in MANUAL_MIDI_ROUTES:
                route.addItem(midi_route_label(value), value)
            row_layout.addWidget(check)
            row_layout.addWidget(route)
            stems_layout.addWidget(row)
            self.stem_controls[stem] = (row, check, route)
        layout.addWidget(self.stem_group)
        self.stem_group.hide()
        actions = FlowLayout(horizontal_spacing=8, vertical_spacing=6)
        for key, callback in (
            ("apply", self.apply),
            ("result", self.restore_result),
            ("run", self.start_all),
            ("stop", owner._stop_processing),
        ):
            button = QPushButton(self.words[key])
            button.setMinimumWidth(0)
            button.clicked.connect(lambda _checked=False, fn=callback: self.guard(fn))
            actions.addWidget(button)
            self.translated_buttons[key] = button
            if key != "stop":
                self.edit_buttons.append(button)
        layout.addLayout(actions)
        self.status = QLabel(self.words["empty"])
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(0)
        layout.addWidget(self.status)
        (
            owner.track_panel.mode_combo.currentIndexChanged.connect(self.update_stem_visibility)
            if hasattr(owner.track_panel, "mode_combo")
            else None
        )

    def guard(self, callback):
        try:
            return callback()
        except Exception as exc:
            self.status.setText(f"{self.words['error']}：{exc}")
            self.owner.status_label.setText(f"{self.words['error']}：{exc}")
            return None

    def update_translations(self):
        self.words = labels(self.owner.config.language)
        self.setTitle(self.words["title"])
        self.stem_heading.setText(self.words["stems"])
        self.stem_group.setAccessibleName(self.words["stems"])
        for key, button in self.translated_buttons.items():
            button.setText(self.words[key])
        for index, key in enumerate(("selected", "all")):
            self.scope.setItemText(index, self.words[key])
        for _stem, (_, _, route) in self.stem_controls.items():
            for index in range(route.count()):
                route.setItemText(index, midi_route_label(route.itemData(index)))
        if self.store:
            self.refresh()
        else:
            self.status.setText(self.words["empty"])

    def busy(self):
        return self.owner.worker is not None and self.owner.worker.isRunning()

    def require_idle(self):
        if self.busy():
            raise RuntimeError(self.words["processing_hint"])

    def current_workflow(self):
        controls = self.owner.track_panel
        values = {
            name: getattr(self.owner.config, name)
            for name in InferenceOptions.model_fields
            if hasattr(self.owner.config, name)
        }
        values.update(
            processing_mode=controls.get_processing_mode(),
            transcription_backend=controls.get_multi_instrument_model(),
            yourmt3_model=controls.get_yourmt3_model(),
            muscriptor_model=controls.get_muscriptor_model(),
            midi_track_mode=controls.get_midi_track_mode(),
            tempo_mode=controls.get_tempo_mode(),
            custom_bpm=controls.get_custom_bpm(),
            muscriptor_processing_chain=controls.get_muscriptor_processing_chain(),
            muscriptor_instruments=controls.get_muscriptor_instruments(),
            quantize_notes=False,
        )
        primary = InferenceOptions.model_validate(values)
        stems = {}
        if self.store and self.current_song_id:
            saved = self.song()
            if saved.workflow.primary.processing_mode == primary.processing_mode:
                stems.update(
                    {
                        key: value
                        for key, value in saved.workflow.stems.items()
                        if key in saved.extra_tracks
                    }
                )
        for stem in SPLIT_STEMS.get(primary.processing_mode, ()):
            _, check, route = self.stem_controls[stem]
            if check.isChecked():
                common = {
                    key: value
                    for key, value in primary.model_dump().items()
                    if key in ManualMidiOptions.model_fields
                }
                stems[stem] = ManualMidiOptions(route=route.currentData(), **common)
        return Workflow(primary=primary, stems=stems)

    def update_stem_visibility(self, *_args):
        allowed = SPLIT_STEMS.get(self.owner.track_panel.get_processing_mode(), ())
        self.stem_group.setVisible(bool(allowed))
        for stem, (row, _, _) in self.stem_controls.items():
            row.setVisible(stem in allowed)

    def ensure_store(self):
        if self.store is None:
            self.store = ProjectStore(
                Path(self.owner.output_dir_edit.text()) / "Projects" / "Library"
            )
            if not self.store.path.exists():
                self.store.create(workflow=self.current_workflow())

    def add_audio(self, paths):
        self.require_idle()
        self.ensure_store()
        ids = self.store.add(paths, self.current_workflow())
        self.current_song_id = ids[0] if ids else None
        self.refresh()
        self.restore_result()

    def browse(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, self.words["add"], "", "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma)"
        )
        if paths:
            self.add_audio(paths)

    def new(self):
        self.require_idle()
        path = QFileDialog.getExistingDirectory(self, self.words["new"])
        if path:
            store = ProjectStore(Path(path) / f"Project-{uuid.uuid4().hex[:8]}")
            store.create(workflow=self.current_workflow())
            self.store, self.current_song_id = store, None
            self.refresh()

    def open(self):
        self.require_idle()
        path, _ = QFileDialog.getOpenFileName(
            self, self.words["open"], "", f"Music to MIDI ({PROJECT_FILE} *.mtmproject)"
        )
        if not path:
            return
        if Path(path).suffix == ".mtmproject":
            root = Path(self.owner.output_dir_edit.text()) / "Projects" / uuid.uuid4().hex
            self.store = ProjectStore.import_archive(path, root)
        else:
            self.store = ProjectStore(path)
            self.store.load()
        self.current_song_id = None
        self.refresh()
        self.restore_result()

    def export(self):
        self.require_idle()
        if not self.store:
            raise ValueError(self.words["empty"])
        path, _ = QFileDialog.getSaveFileName(
            self, self.words["export"], "project.mtmproject", "Music to MIDI (*.mtmproject)"
        )
        if path:
            self.store.export_archive(path)

    def load_profile(self):
        self.require_idle()
        path, _ = QFileDialog.getOpenFileName(self, self.words["profile_load"], "", "JSON (*.json)")
        if path:
            workflow = Workflow.load(path)
            self.ensure_store()
            self.store.configure(workflow, self.configuration_scope())
            self.refresh()

    def save_profile(self):
        path, _ = QFileDialog.getSaveFileName(
            self, self.words["profile_save"], "workflow.json", "JSON (*.json)"
        )
        if path:
            self.current_workflow().save(path)

    def configuration_scope(self):
        return (
            None
            if self.scope.currentData() == "all"
            else [self.current_song_id] if self.current_song_id else []
        )

    def apply(self):
        self.require_idle()
        self.ensure_store()
        self.store.configure(self.current_workflow(), self.configuration_scope())
        self.refresh()
        self.status.setText(self.words["saved"])

    def refresh(self):
        if not self.store:
            return
        document = self.store.load()
        blocker = QSignalBlocker(self.songs)
        self.songs.clear()
        for index, song in enumerate(document.songs, 1):
            self.songs.addItem(f"{index}. {song.name} · {self.words[song.status]}", song.id)
        self.songs.setCurrentIndex(max(0, self.songs.findData(self.current_song_id)))
        del blocker
        self.current_song_id = self.songs.currentData()
        self.status.setText(
            f"{document.name} · {len(document.songs)} · {self.words[document.status]}\n{document.error or ''}"
        )
        if not self.busy():
            self.select_song()

    def select_song(self, *_args):
        if self.busy() or not self.store:
            return
        self.current_song_id = self.songs.currentData()
        if not self.current_song_id:
            return
        song = self.song()
        self.owner.current_file = str(self.store.verify(song.source))
        controls = self.owner.track_panel
        primary = song.workflow.primary
        controls.set_processing_mode(primary.processing_mode)
        controls.set_multi_instrument_model(primary.transcription_backend)
        controls.set_yourmt3_model(primary.yourmt3_model)
        controls.set_muscriptor_model(primary.muscriptor_model)
        controls.set_midi_track_mode(primary.midi_track_mode)
        controls.set_muscriptor_processing_chain(primary.muscriptor_processing_chain)
        controls.set_muscriptor_instruments(primary.muscriptor_instruments)
        controls.set_custom_bpm(primary.custom_bpm)
        controls.set_tempo_mode(primary.tempo_mode)
        controls.set_processing_controls_enabled(True)
        for stem, (_, check, route) in self.stem_controls.items():
            option = song.workflow.stems.get(stem)
            check.setChecked(option is not None)
            if option:
                route.setCurrentIndex(route.findData(option.route))
        self.update_stem_visibility()
        self.owner.start_btn.setEnabled(True)

    def song(self):
        return next(s for s in self.store.load().songs if s.id == self.current_song_id)

    def handles_current(self):
        return bool(
            self.store
            and self.current_song_id
            and self.owner.current_file
            and Path(self.owner.current_file).resolve()
            == self.store.resolve(self.song().source.path)
        )

    def start_all(self):
        self.start(None)

    def start(self, song_ids=None, track_ids=None, capture=False):
        self.require_idle()
        if not self.store:
            raise ValueError(self.words["empty"])
        if capture:
            self.store.configure(self.current_workflow(), [self.current_song_id])
        worker = ProjectWorker(self.store, song_ids, track_ids, self.owner, primary_only=capture)
        self.owner.worker = worker
        self.owner._stopping = False
        self.owner.start_btn.setEnabled(False)
        self.owner.stop_btn.setEnabled(True)
        self.owner.track_panel.set_processing_controls_enabled(False)
        for control in [self.songs, self.scope, self.stem_group, *self.edit_buttons]:
            control.setEnabled(False)
        if self.owner.audio_mixer:
            self.owner.audio_mixer.set_midi_controls_enabled(False)
        worker.updated.connect(self.progress)
        worker.completed.connect(self.completed)
        worker.failed.connect(
            lambda message: self.status.setText(f"{self.words['error']}：{message}")
        )
        worker.finished.connect(lambda: self.owner._on_worker_thread_finished(worker))
        worker.finished.connect(self.finished)
        worker.start()

    def progress(self, snapshot):
        p = snapshot["progress"]
        message = f"[{p.get('index', 0)}/{p.get('total', len(snapshot['songs']))}] {p.get('song_name', '')} · {p.get('percent', 0)}% {p.get('message', '')}"
        self.status.setText(message)
        self.owner.status_label.setText(message)

    def completed(self, snapshot):
        self.status.setText(f"{self.words[snapshot['status']]}\n{snapshot['error'] or ''}")

    def finished(self):
        for control in [self.songs, self.scope, self.stem_group, *self.edit_buttons]:
            control.setEnabled(True)
        self.guard(self.refresh)
        self.guard(self.restore_result)

    def convert_track(self, stem, route):
        self.require_idle()
        instruments = self.owner.audio_mixer.track_muscriptor_instruments(stem)
        if (
            stem not in SPLIT_STEMS.get(self.song().workflow.primary.processing_mode, ())
            and stem not in self.song().extra_tracks
        ):
            stem = self.store.add_track(
                self.current_song_id, self.owner.audio_mixer.track_state(stem).path
            )
        document = self.store.load()
        song = next(s for s in document.songs if s.id == self.current_song_id)
        workflow = song.workflow.model_copy(deep=True)
        common = {
            key: value
            for key, value in self.current_workflow().primary.model_dump().items()
            if key in ManualMidiOptions.model_fields
        }
        common["muscriptor_instruments"] = instruments
        workflow.stems[stem] = ManualMidiOptions(route=route, **common)
        self.store.configure(workflow, [song.id])
        self.start([song.id], [stem])

    def save_track_controls(self):
        if (
            self._restoring
            or self.busy()
            or not self.handles_current()
            or not self.owner.audio_mixer
        ):
            return
        with exclusive(self.store.run_lock):
            document = self.store.load()
            song = next(s for s in document.songs if s.id == self.current_song_id)
            mixer = self.owner.audio_mixer
            for stem, backend in mixer._backends.items():
                if (
                    stem not in SPLIT_STEMS.get(song.workflow.primary.processing_mode, ())
                    and stem not in song.extra_tracks
                ):
                    continue
                row = backend.row
                state = mixer.track_state(stem)
                song.mixer[stem] = TrackView(
                    muted=state.muted,
                    solo=state.solo,
                    gain_db=state.volume_db,
                    offset=state.offset_ms / 1000,
                )
                route = row.midi_model_selector.currentData()
                if row.midi_enabled_checkbox.isChecked() and route:
                    previous = song.workflow.stems.get(stem, ManualMidiOptions(route=route))
                    song.workflow.stems[stem] = previous.model_copy(
                        update={
                            "route": route,
                            "muscriptor_instruments": mixer.track_muscriptor_instruments(stem),
                        }
                    )
                else:
                    song.workflow.stems.pop(stem, None)
            select_checkpoints(song)
            song.status = "succeeded" if plan_complete(song) else "pending"
            document.status = (
                "succeeded" if all(s.status == "succeeded" for s in document.songs) else "pending"
            )
            self.store.save(document)

    def restore_result(self):
        if not self.store or not self.current_song_id or self.busy():
            return
        song = self.song()
        if not song.primary_key:
            return
        primary = song.checkpoints[song.primary_key]
        if primary.status != "succeeded":
            return
        for asset in primary.artifacts:
            self.store.verify(asset)
        self._restoring = True
        try:
            self.owner._clear_completed_result()
            source = str(self.store.verify(song.source))
            self.owner.current_file = source
            if primary.result.get("manual_midi_required"):
                from src.core.separation_service import SeparationResult

                separated = {
                    a.track_id: str(self.store.verify(a))
                    for a in primary.artifacts
                    if a.kind == "audio_track"
                }
                self.owner._show_separation_result(
                    SeparationResult(
                        primary.options["processing_mode"],
                        source,
                        str(self.store.resolve(primary.output_dir)),
                        separated,
                        primary.result.get("processing_time", 0),
                    )
                )
                mixer = self.owner.audio_mixer
                if mixer:
                    if song.extra_tracks:
                        mixer.add_tracks(
                            {key: str(self.store.verify(a)) for key, a in song.extra_tracks.items()}
                        )
                    completed_tracks = []
                    for stem, backend in mixer._backends.items():
                        row = backend.row
                        option = song.workflow.stems.get(stem)
                        if option:
                            row.midi_enabled_checkbox.setChecked(True)
                            row.midi_model_selector.setCurrentIndex(
                                row.midi_model_selector.findData(option.route)
                            )
                        if stem in song.mixer:
                            state = song.mixer[stem]
                            mixer.set_track_muted(stem, state.muted)
                            mixer.set_track_solo(stem, state.solo)
                            mixer.set_track_volume_db(stem, state.gain_db)
                            mixer.set_track_offset_ms(stem, round(state.offset * 1000))
                        key = song.track_keys.get(stem)
                        step = song.checkpoints.get(key) if key else None
                        if step and step.status == "succeeded":
                            asset = next(a for a in step.artifacts if a.kind == "midi")
                            mixer.set_track_midi_succeeded(
                                stem, step.options["route"], str(self.store.verify(asset))
                            )
                            completed_tracks.append((step.finished_at or "", stem))
                            self.owner._add_result_button(
                                f"{self.words['result']} · {stem} MIDI",
                                f"projectMidiResult-{stem}",
                                lambda _checked=False, track=stem: self.guard(
                                    lambda: self.open_midi(track)
                                ),
                            )
                        elif step and step.error:
                            mixer.set_track_midi_failed(stem, step.error)
                        for signal in (
                            row.midi_enabled_checkbox.toggled,
                            row.midi_model_selector.currentIndexChanged,
                            row.mute_changed,
                            row.solo_changed,
                            row.volume_changed,
                            row.offset_committed,
                        ):
                            signal.connect(lambda *_args: self.guard(self.save_track_controls))
                    if completed_tracks:
                        self.open_midi(max(completed_tracks)[1])
            else:
                self.open_midi()
        finally:
            self._restoring = False

    def open_midi(self, stem=None):
        """Open a verified checkpoint in the shared editor without running a model."""
        from dataclasses import fields
        from src.models.data_models import BeatInfo, ProcessingResult, Track, TrackType

        self.require_idle()
        song = self.song()
        key = song.track_keys.get(stem) if stem else song.primary_key
        step = song.checkpoints.get(key) if key else None
        if step is None or step.status != "succeeded":
            raise ValueError(self.words["empty"])
        asset = next(a for a in step.artifacts if a.kind == "midi")
        source = song.source
        if stem:
            source = song.extra_tracks.get(stem)
            if source is None:
                source = next(
                    a for a in song.checkpoints[song.primary_key].artifacts
                    if a.kind == "audio_track" and a.track_id == stem
                )
        beat = step.result.get("beat")
        beat_info = (
            BeatInfo(**{k: v for k, v in beat.items() if k in {f.name for f in fields(BeatInfo)}})
            if beat else None
        )
        result = ProcessingResult(
            midi_path=str(self.store.verify(asset)),
            tracks=[Track(TrackType.OTHER, str(self.store.verify(source)))],
            processing_time=step.result.get("processing_time", 0),
            total_notes=step.result.get("total_notes", 0),
            beat_info=beat_info,
            selected_instruments=step.result.get("selected_instruments", []),
            detected_instruments=step.result.get("detected_instruments", []),
            transcription_backend=step.result.get("transcription_backend"),
        )
        if stem:
            route = step.options["route"]
            self.owner._show_muscriptor_streaming(
                result.tracks[0].audio_path,
                result.selected_instruments,
                backend_label=midi_route_label(route),
                muscriptor_groups=route.startswith("muscriptor"),
                preserve_mixer=True,
                source_track_name=stem,
            )
            self.owner.muscriptor_result_widget.finalize_result(result)
        else:
            self.owner._show_muscriptor_result(result, reveal=True)
