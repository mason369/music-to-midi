# Projects, workflows and resuming

A project owns input audio, model choices, processing state and verified outputs. `workflow.json` is a reusable configuration template. `music-to-midi-project.json` records one project's work; a `.mtmproject` archive includes the sources, stems, MIDI files and manifest for transfer between machines and interfaces.

## Desktop and standalone Web

Select or drop multiple files. The desktop's ordinary file input uses `Projects/Library` under the output directory; use **New project** for a separate folder. Web projects live under `projects` in the API data directory.

Choose a mode, select the stems to transcribe, and choose a model for each. Apply the settings to the current song or all songs. Selection and configuration never start inference. **Continue project** executes each song's saved plan. The primary **Start separation** action only separates the current song; each track's convert action only transcribes that track.

Reopen the project or add the same source again to resume. Completed separation and matching MIDI steps are verified and reused. Desktop and Web preserve per-track model selections, mute/solo, gain and offsets. Use **Open song result** to restore the result workspace. External tracks can also be retained; the desktop imports an external track when its conversion is first requested.

External track plans belong to their song. Applying a workflow to all songs preserves each song's own external track plan; reusable workflow JSON excludes those track IDs. Selecting a direct transcription mode or using CLI `--clear-stems` clears the corresponding track plan while retaining completed artifacts.

## Space and Colab

Use **Projects and batch** at the top of the page: create a project, add multiple files, save the mode and per-stem model plan, then continue. **Open project** restores the browser's saved project identifier. Archives can be imported in another session. Completed MIDI opens in the shared playback/editor/export workspace. Click **Save mixer settings** after changing the mix.

Cloud session storage may be deleted when the runtime ends. Browser storage only holds the identifier, not the audio: export the project archive before leaving and import it next time. Projects do not bypass ZeroGPU duration or quota limits. The existing single-file quick conversion remains available; use the project section for persistent or batch work.

## CLI

```text
python -m src.cli project create ./project song1.wav song2.wav --mode six_stem_split --stem piano=piano_transkun --stem bass=miros
python -m src.cli project run ./project
python -m src.cli project status ./project --json
python -m src.cli project configure ./project --stem piano=piano_aria_amt --save-profile workflow.json
python -m src.cli project export ./project project.mtmproject
python -m src.cli project import ./restored project.mtmproject
```

Use `--profile` to import a workflow, `--song` to target song IDs, and `--fail-fast` to stop the queue on its first failure. Otherwise independent songs continue and the final status and exit code still report failure. Ctrl+C preserves successful steps. Existing `batch --resume` remains supported.

Resume checkpoints cover one separation or one track transcription, not intermediate neural inference frames. A cancelled or interrupted step restarts; completed matching steps are reused. Source SHA-256 deduplicates songs within a project. Model changes invalidate the affected transcription while retaining earlier outputs; MIDI choices do not invalidate separation. Missing or changed assets fail explicitly.

Atomic JSON commits and OS locks coordinate readers and writers. The archive validates its exact manifest, hashes and relative paths and rejects symlinks. Move the whole project or use its archive, not the JSON alone. Persist the API data volume in Docker, and preserve projects when updating a portable installation.

Contract tests cover the supported interfaces and packaging paths. Actual cloud runtimes, Linux GPU, Intel XPU and newly built portable executables require separate acceptance; local NVIDIA validation does not establish those results.
