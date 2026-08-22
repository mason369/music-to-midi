# Music to MIDI Converter (AI Audio to MIDI)

<p align="center">
  <a href="../README.md">中文</a> | English
</p>

Music to MIDI is a local-first AI audio-to-MIDI converter for music producers, transcription hobbyists, piano learners, sampling workflows, and automatic music transcription (AMT) experiments. Drop in an `MP3`, `WAV`, `FLAC`, `OGG`, or `M4A` file, then generate editable MIDI from the PyQt6 desktop app, the Gradio Web interface, or the Google Colab notebook.

The current product surface syncs seven processing modes: full-mix transcription, two separation workflows, and four dedicated piano routes.

Every one of the seven modes and every per-track conversion uses Beat This `final0` as its only beat/downbeat detector. Competing marks are cleaned, missed musical positions are counted, and global BPM is fitted by least squares; downbeats determine meter independently, and meter remains unknown when the resulting grid is unreliable. The default writes the detected global BPM as the only tempo event; an explicit adaptive mode writes a stable section-level tempo map. A manual 30–300 BPM value overrides the project tempo while preserving the musical ticks established at the detected BPM. Event payloads remain unchanged; the detector does not quantize or clean notes.

Note quantization is explicit and disabled by default. The desktop, Space, and Colab result editors expose `1/4`, `1/8`, `1/16`, `1/32`, and `1/64`; their scope defaults to All tracks and can be changed to Selected notes. Changing either selector does not alter notes until Quantize is pressed. The standalone Web/API, including Docker deployment, exposes the same grids behind an all-tracks Quantize notes checkbox. That post-write pass snaps paired note starts and durations once while verifying that tempo, meter, controller, and other non-note events retain their absolute ticks.

The Standard MIDI tempo, meter, and every non-tempo event are verified after publication. Correct interpretation in a DAW depends on tempo-map import being enabled. MuseScore 3/4 may classify an unquantized performance MIDI as human performance, run its own beat tracker, and replace the tempo shown on the notation page. That displayed value is produced by MuseScore's MIDI importer; the project does not quantize or move model notes merely to force a notation application to display the file tempo.

## Standalone Web API And Browser Frontend

The standalone Web API and browser frontend call the same `MusicToMidiPipeline` and provide the same seven modes as the desktop application. The Windows release contains three purpose-specific folders:

| Package | Executable | Purpose |
|---------|------------|---------|
| `MusicToMidi-App` | `MusicToMidi.exe` | Local desktop application; it does not host the Web service |
| `MusicToMidi-WebBackend` | `MusicToMidiBackend.exe` | GPU inference, job queue, output files, and API |
| `MusicToMidi-WebFrontend` | `MusicToMidiFrontend.exe` | Browser UI and static assets; no inference models |

Run the source Web application from the repository root with one command:

```powershell
.\venv\Scripts\python.exe -m src.web
```

The entry point detects the server computer's primary LAN IPv4 address, starts WebBackend followed by WebFrontend, verifies that both are ready, and opens an Edge app window. The terminal prints the frontend address that other computers on the same LAN can use. `Ctrl+C` or closing the Edge app window stops both processes. On a multi-adapter or VPN host, pass the concrete address explicitly, for example `--host 192.168.1.50`. See [web/README.md](../web/README.md) for all options and the scoped firewall configuration.

For packaged use, the startup order is `MusicToMidiBackend.exe` from WebBackend followed by `MusicToMidiFrontend.exe` from WebFrontend. Defaults work on one computer. For LAN use, `MusicToMidiBackend.json` and `MusicToMidiFrontend.json`, created next to their executables on first launch, hold the network settings used after restart. Client computers need only a browser. The health endpoint is `http://<backend-address>:8765/api/v1/health`; OpenAPI documentation is at `http://<backend-address>:8765/docs`.

The browser submits multipart jobs, polls `GET /api/v1/jobs/<job-id>` for the real terminal state, and downloads MIDI or separated WAV files through the returned `download_url` values. Frontend and backend verify API 2.0 compatibility, a five-second heartbeat keeps the connection indicator current, terminal jobs default to a 30-day/200-record retention policy, and deleting a job and its related files requires confirmation. Desktop, Web, Space, and Colab run one accelerator task at a time while other tasks wait. Failures remain visible; no reduced-quality algorithm or silent fallback is used.

The standalone Web deployment is intentionally limited to a trusted LAN and does not include authentication, authorization, or TLS. Exposing ports `5173` or `8765` to the Internet would publish an unauthenticated service. See [web/README.md](../web/README.md) for complete JSON examples, scoped firewall commands, connectivity checks, shutdown instructions, and split-host deployment.

Issue #9 is delivered as a complete Docker self-host distribution rather than a hosted website or a lone public image. Each stable release includes model-free backend and gateway images, a digest-pinned Compose file, an environment template, Windows/Linux management scripts, image identities, and SHA-256 checksums. The default stack binds only to `127.0.0.1:7860`, so it needs no public domain, ACME email, application password, or SSH credential. Operators select model profiles in `.env`, run the explicit one-shot `model-init`, and then keep the inference container offline. See the [Docker self-host guide](docker-deployment.md) for model selection, persistence, validation, upgrades, rollback, and the separate authenticated public-deployment option.

## Unified Interface Gallery

The desktop app, Gradio Web interface, and Google Colab use the same seven-mode workflow and interaction semantics. The gallery follows the core flow from the main interface to completed separation, per-track processing, and MuScriptor's progressive MIDI preview.

### 1. Main interface and full-mix transcription

![Main interface and full-mix transcription](../resources/screenshots/01-main-interface.png)

### 2. Completed six-stem separation

![Completed six-stem separation](../resources/screenshots/02-six-stem-separation-result.png)

### 3. Six-stem waveforms and per-track MIDI controls

![Six-stem waveforms and per-track MIDI controls](../resources/screenshots/03-six-stem-track-controls.png)

### 4. MuScriptor progressive transcription and MIDI preview

![MuScriptor chunked transcription, playable progress, and piano-roll preview](../resources/screenshots/04-muscriptor-progressive-midi-preview.png)

## Use Cases

The application supports turning a vocal line, piano recording, full mix, or separated stem into MIDI that remains editable in a DAW. It is designed for users who want more control than a simple upload-and-download converter while keeping the common audio-to-MIDI path approachable.

## Current Capabilities

| Area | Current behavior |
|------|------------------|
| Full mix | `SMART` sends the song to YourMT3+, MIROS, or MuScriptor Large / Medium / Small and exports MIDI with notes, drums, and instrument groups. The default is the official YourMT3+ `YPTF.MoE+Multi (noPS)` checkpoint. |
| Separation | `VOCAL_SPLIT` produces vocals and accompaniment WAV files with Leap XE 90-band and PolarFormer. `SIX_STEM_SPLIT` produces `bass / drums / guitar / piano / vocals / other` WAV files with `BS-Rofo-SW-Fixed.ckpt`. MIDI conversion starts per track from the result workbench. |
| Per-track routes | Each separated WAV can use five YourMT3+ checkpoints, MIROS, MuScriptor Large / Medium / Small, or one of four piano models, for 13 routes in total. |
| Piano | `PIANO_TRANSKUN`, `PIANO_TRANSKUN_V2_AUG`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL` use TransKun default V2, the official V2 Aug checkpoint, Aria-AMT, or ByteDance's pedal-aware model. |
| MuScriptor selection | An empty instrument list enables model detection. A selection is passed to the official `instruments` and `prelude_forcing` path, masks unselected tokens during generation, and is checked against streamed events and final MIDI. |
| MIDI content | YourMT3+ and MIROS retain their official writer results. MuScriptor uses its official events and writer, followed by selected-instrument validation. The project adds no quantization, de-duplication, short-note filtering, velocity smoothing, polyphony limiting, or local `NoteEvent` regeneration. |
| Input and interfaces | `MP3`, `WAV`, `FLAC`, `OGG`, and `M4A` are accepted; FFmpeg converts non-WAV input to 44.1 kHz PCM WAV. Desktop, Space, and Colab expose the same seven modes. |

## Interface Matrix

| Interface | Modes | Backend Selection | Best For |
|-----------|-------|-------------------|----------|
| PyQt6 desktop | `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_TRANSKUN_V2_AUG`, `PIANO_ARIA_AMT`, `PIANO_BYTEDANCE_PEDAL` | `SMART` selects YourMT3+ / MIROS / MuScriptor; separated WAV tracks expose 13 routes; piano modes use their dedicated backend | Local GPU use, persistent output folders, and dedicated piano transcription |
| Standalone Web API | Same seven modes as desktop | Multipart jobs, terminal-state polling, and output-file downloads; processing still runs through the same `MusicToMidiPipeline` | Custom Web clients, LAN service, or system integration |
| Gradio Space | Same seven modes as desktop | MuScriptor instrument search/multi-select, hard decoding constraint, and real MIDI workbench are synchronized | Browser-based use or hosted demos |
| Google Colab | Same seven modes as desktop | Same transcription and separation semantics as Space | Temporary Colab GPU sessions |

## Entry And Dependency Sync Status

This repository keeps project workflows separate from official YourMT3+ checkpoint modes:

- `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_TRANSKUN_V2_AUG`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL` are this project's seven processing workflows.
- `YMT3+`, `YPTF+Single (noPS)`, `YPTF+Multi (PS)`, `YPTF.MoE+Multi (noPS)`, and `YPTF.MoE+Multi (PS)` are the five checkpoint / architecture modes exposed by the official YourMT3 demo.
- Desktop, Gradio Space, and Colab expose the same seven workflows. `SMART` selects YourMT3+, MIROS, or MuScriptor Large / Medium / Small; both separation workflows first deliver WAV tracks, and each track then exposes 13 explicit MIDI routes.

Current synchronization coverage:

| Location | Synced Content | Notes |
|----------|----------------|-------|
| `download_sota_models.py` | Prepares Beat This `final0`; all five official YourMT3+ checkpoints; pinned MIROS source plus both weights; MuScriptor Large / Medium / Small; `BS-Rofo-SW-Fixed.ckpt`; Leap XE; PolarFormer; TransKun V2 Aug; Aria-AMT; ByteDance; MuseScore General SoundFont; and FluidSynth, while strictly validating the default TransKun 2.0.1 package and bundled V2 resources | Fixed-source resources are validated by known size/SHA256 or their explicit source/runtime identity; any required-resource failure stops the command. |
| `run.ps1` / `run_xpu.ps1` / `run.sh` | Checks actual accelerator execution, all official YourMT3+ modes, MuScriptor Large / Medium / Small, BS-RoFormer SW Fixed, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance Pedal, MIROS, SoundFont, FluidSynth, and separator availability before launch | Missing or invalid required resources and CPU fallback are reported explicitly. |
| `install.ps1` / `install_xpu.ps1` / `install.sh` | Installs an isolated NVIDIA PyTorch 2.7 or Intel XPU PyTorch 2.11 runtime, NumPy 1.26, audio-separator 0.44.1, the identity-verified official MuScriptor v0.3.0 runtime, and every required model/runtime asset | NVIDIA uses `venv` + CUDA 12.8; Windows Intel uses `venv-xpu` + native PyTorch XPU + OpenVINO GPU. Mixed runtimes are rejected. |
| `.github/workflows/build.yml` | Push/PR jobs run Linux and Windows source, test, and packaging checks only | They produce no portable package; empty directories and fake models fail bundle validation. |
| `.github/workflows/release.yml` | The complete portable-build pipeline; it downloads and strictly verifies every YourMT3+, separator, MIROS, MuScriptor, TransKun, Aria-AMT, ByteDance, playback, and runtime asset | The 29-component gate currently records 25 `VERIFIED` and four explicitly documented `OWNER_ACCEPTED` items. The target GPU runtime is PyTorch 2.7 + CUDA 12.8; an owner acceptance is a revocable distribution decision, not a claim that upstream granted a license. |
| `colab_notebook.ipynb` | Keeps Colab's preinstalled Torch, installs pinned Web/runtime dependencies, and synchronizes all seven modes | `SMART` and the per-track workbench expose YourMT3+, MIROS, and MuScriptor Large / Medium / Small; the per-track menu contains 13 routes in total. |

## Processing Modes

| Mode | Processing Flow | Main Output | Notes |
|------|-----------------|-------------|-------|
| `SMART` | Audio -> selected YourMT3+ / MIROS / MuScriptor Large, Medium, or Small -> MIDI | `<song>.mid` | No source separation. A non-empty MuScriptor instrument selection is a real decoding constraint. |
| `VOCAL_SPLIT` | Audio -> Leap XE vocals + PolarFormer accompaniment -> two WAV tracks -> explicit per-track MIDI | `<song>_vocals.wav`, `<song>_accompaniment.wav`; per-track MIDI on request | Separation does not auto-transcribe. Each WAV independently selects one of 13 routes. |
| `SIX_STEM_SPLIT` | Audio -> `BS-Rofo-SW-Fixed.ckpt` -> six WAV tracks -> explicit per-track MIDI | `<song>_<stem>.wav`; per-track MIDI on request | Each real WAV independently selects its route and whether to convert; MIDI is not auto-merged. |
| `PIANO_TRANSKUN` | Audio -> TransKun default V2 model -> MIDI | `<song>_piano_transkun.mid` | Pure-piano route using the checkpoint resources bundled with the PyPI package. |
| `PIANO_TRANSKUN_V2_AUG` | Audio -> official TransKun V2 Aug checkpoint -> MIDI | `<song>_piano_transkun_v2_aug.mid` | Independent mode with a separately downloaded and verified checkpoint; it is not a fallback for default V2. |
| `PIANO_ARIA_AMT` | Audio -> Aria-AMT piano model -> MIDI | `<song>_piano_aria.mid` | Pure-piano route, available when the Aria-AMT checkpoint is bundled or present in the model directory. |
| `PIANO_BYTEDANCE_PEDAL` | Audio -> ByteDance pedal-aware piano model -> MIDI | `<song>_piano_bytedance_pedal.mid` | Pure-piano route that retains sustain-pedal CC64, available when the ByteDance Piano checkpoint is bundled or present in the model directory. |

## Output Files

The desktop app writes to:

```text
MidiOutput/<audio-file-name>/
```

If the folder already exists, the app chooses `<audio-file-name>_2`, `<audio-file-name>_3`, and so on.

Common outputs:

```text
song.mid
song_accompaniment.mid
song_vocal.mid
song_vocal_accompaniment_merged.mid
song_bass.mid
song_drums.mid
song_guitar.mid
song_piano.mid
song_vocals.mid
song_other.mid
song_all_stems_merged.mid
song_piano_transkun.mid
song_piano_transkun_v2_aug.mid
song_piano_aria.mid
song_piano_bytedance_pedal.mid
song_vocals.wav
song_accompaniment.wav
song_bass.wav
song_drums.wav
song_guitar.wav
song_piano.wav
song_other.wav
```

The exact files depend on the selected mode and the per-track conversions the user explicitly starts. Vocal split exposes canonical `vocals` and `accompaniment` WAV files; six-stem mode delivers six real separated WAV files. MIDI is generated only for tracks whose conversion action is triggered.

## Backends

### YourMT3+

YourMT3+ is the default multi-instrument backend. `download_sota_models.py` prepares Beat This `final0`, all five official YourMT3+ checkpoints, pinned MIROS source and both weights, MuScriptor Large / Medium / Small, `BS-Rofo-SW-Fixed.ckpt`, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance, MuseScore General SoundFont, and FluidSynth, and strictly validates the default TransKun 2.0.1 package and bundled V2 resources. YourMT3 inference imports the repository-controlled `YourMT3/amt/src` tree through `src/core/yourmt3_transcriber.py`.

The source runtime depends on:

```text
YourMT3/amt/src/model/ymt3.py
YourMT3/amt/src/utils/task_manager.py
YourMT3/amt/src/config/config.py
```

The complete project checkout includes a compatibility-patched `YourMT3/amt/src` tree protected by a fixed manifest. When it is missing, the supported recovery source is the current project revision. Mutable upstream `master` is suitable for a separate experiment but does not satisfy this project's three-interface source-parity or portable-build identity contract.

Model-weight command:

The source-maintenance commands below use the repository's `venv`. Without activation, the complete paths are `.\venv\Scripts\python.exe` / `.\venv\Scripts\hf.exe` on Windows and `./venv/bin/python` / `./venv/bin/hf` on Linux/WSL2. A global Python interpreter is outside the supported source runtime.

```bash
python download_sota_models.py
```

Default model search roots include:

```text
~/.cache/music_ai_models/yourmt3_all
runtime/models/yourmt3_all
models/yourmt3_all
```

### MuScriptor Large / Medium / Small

The project pins the unmodified upstream `v0.3.0` commit `d73147e75e5b9b0c0a79ebe154587db4fd603e0c` and validates seven runtime source files by SHA-256. The single project-wide Beat This `final0` analysis passes the complete grid, BPM, reliable meter, and first downbeat into the official v0.3.0 BeatGrid. Official onset-phase correction is applied only when its sample-count and concentration thresholds are met; otherwise a zero correction is recorded explicitly. No second beat detector or placeholder 120 BPM is accepted. Normalized tempo metadata is repeated on every note-bearing track for MuseScore compatibility, and desktop, Space, and Colab render the same beat, downbeat, and alternating-bar piano-roll grid.

The three gated checkpoints are pinned independently: [`muscriptor-large`](https://huggingface.co/MuScriptor/muscriptor-large) revision `8809fdfbed2affa7ade94a7059e746e3880720e7`, [`muscriptor-medium`](https://huggingface.co/MuScriptor/muscriptor-medium) revision `f32236969308476e01fd3aae67357de5feb05a2d`, and [`muscriptor-small`](https://huggingface.co/MuScriptor/muscriptor-small) revision `8c127f603b807520fa465c838e9bfee8a91ada4e`. All use CC BY-NC 4.0 with additional lawful-use terms. Valid access comes from one Hugging Face account accepting all three repository conditions in a browser and then authenticating the CLI. `hf auth login` cannot accept the web terms on the user's behalf, and source installs, Colab sessions, and self-hosted Spaces cannot download these weights anonymously:

```bash
hf auth login
python download_muscriptor_model.py --size all
```

All three are explicit choices and do not silently replace one another. Large prioritizes quality, Medium trades speed against quality, and Small has the fewest parameters and fastest runtime. Every tier uses the official 5-second window, prelude forcing, and single-generation path. Large is a decoder-only Transformer with roughly 1.3B parameters (the current code README rounds it to 1.4B), 48 layers, and hidden dimension 1536. It consumes 5-second 16 kHz mono chunks and emits MT3-style onset, offset, pitch, and 36-group instrument events. Training combines about 1.45 million MIDI files for synthetic pretraining, 170,000 real recordings / about 11,000 hours for fine-tuning, and 300 curated tracks for RL post-training.

The official model card reports the following scores on the authors' 372-track real multi-instrument `D_Test` set, using the full training pipeline and CFG=2:

| Model | Onset F1 | Frame F1 | Offset F1 | Drums F1 | Multi F1 |
|---|---:|---:|---:|---:|---:|
| YourMT3+ `YPTF.MoE+Multi (noPS)` | 32.5 | 45.5 | 17.8 | 41.4 | 21.9 |
| MuScriptor Large | **60.4** | **72.4** | **48.6** | **49.6** | **47.8** |

This is strong evidence that MuScriptor is a leading public full-mix candidate, but not proof of universal SOTA: `D_Test` is an author-held set without a public download path, and MuScriptor wins Multi F1 on six of the paper's eight public cross-domain datasets while losing on RWC-C and RWC-R. It also does not emit velocity, uses a fixed 36-group instrument taxonomy, and the weights are non-commercial.

The release chronology is also explicit: the [Hugging Face API](https://huggingface.co/api/models/MuScriptor/muscriptor-large) records repository creation on 2026-06-30; the [paper](https://arxiv.org/abs/2607.08168) and [Mirelo article](https://mirelo.ai/blog/turning-audio-to-midi) were published on 2026-07-09; the current public weight revisions were updated on 2026-07-10; and official source `v0.3.0` was released on 2026-08-05. Repository timestamps are publishing metadata, not model-training dates.

Mirelo separately says that Studio hosts a more accurate version trained on more data. No public checkpoint, revision, parameter count, or comparable score has been published for that service model, so it is not the same verifiable artifact as any public MuScriptor checkpoint and is not integrated here. Full ablations, all eight public-dataset comparisons, scale results, conditioning gains, and the frontier watchlist are documented in [the MuScriptor research note](muscriptor-model.md).

### MIROS

MIROS is the optional pinned MusicFM / AI4Musician Challenge SOTA backend for `SMART`, `VOCAL_SPLIT`, and `SIX_STEM_SPLIT` in the desktop app, Space, and Colab. It is not integrated as a PyPI package; the wrapper requires the verified upstream source and weights, runs its entrypoint to produce temporary MIDI, then converts that MIDI into the app's internal note format.

Supported locations:

```text
ai4m-miros/
external/ai4m-miros/
MIROS/
external/MIROS/
```

The wrapper checks for:

```text
main.py
transcribe.py
model/musicfm/data/pretrained_msd.pt
logs/Multi_longer_seq_length_frozen_enc_silu/le2bzt53/checkpoints/last.ckpt
```

MIROS also needs its upstream runtime dependencies. `requirements.txt` installs this project; it does not guarantee a complete MIROS environment.

The downloader checks out a pinned `amt-os/ai4m-miros` source commit and applies controlled compatibility patches. `pretrained_msd.pt` is fetched from the official Hugging Face `minzwon/MusicFM` repository, while `last.ckpt` still follows the official Google Drive file ID used by upstream `main.py`. GitHub Actions release packaging does not depend on the live Google Drive quota: it streams the already packaged and verified `external/ai4m-miros` directory from this repository's `v1.0.16` Linux portable release assets. If those portable assets are missing, extraction fails, or the checkpoint container is incomplete, the release job fails explicitly instead of using an unknown source or silently skipping the model.

### Vocal Separation: Leap XE + PolarFormer

`VOCAL_SPLIT` uses two independent separation models on the original mix:

- [BS-RoFormer Leap XE](https://huggingface.co/pcunwa/BS-Roformer-Leap) uses `Xe/bs_leap_xe_voc.ckpt` with `Xe/leap_xe_config_voc.yaml` to produce vocals.
- [BS PolarFormer](https://huggingface.co/bgkb/bs_polarformer) uses the official `bs_polarformer_fp16.onnx` with `model_bs_polarformer_float16.yaml` to produce accompaniment.

The canonical separated outputs are `vocals` and `accompaniment`. Each enters the track workbench with 13 explicit routes: five YourMT3+ checkpoints, MIROS, MuScriptor Large / Medium / Small, and four piano-specialized backends. The two separation calls are not substitutes for one another, and a failure in either route is surfaced instead of synthesizing a missing stem.

TelkNet boundary: with authorization, this audit inspected private `mason369/telknet` dev commit `52be6fec179be492f5229ba149545ac2833b284a`. This project only aligns its core YourMT3/MIROS rule: official writer output followed only by tempo metadata, with no generic note cleanup. Both separation workflows likewise deliver WAV first; MIDI is explicitly triggered in this project's per-track workbench. There is no evidence that this dev commit is the deployed production revision, and no line-for-line routing, environment, or bit-identical-output claim is made.

Asset preparation:

```bash
python download_vocal_model.py
python download_accompaniment_model.py
```

### TransKun Default V2

TransKun default V2 is a dedicated piano transcription backend for pure or piano-forward audio. The project calls the pretrained resources bundled with the `transkun` PyPI package through `src/core/transkun_transcriber.py`:

```bash
python -m pip install "transkun==2.0.1"
```

Availability checks confirm that `transkun.transcribe`, `pretrained/2.0.pt`, and `pretrained/2.0.conf` exist. Missing packaged resources correspond to this repair command:

```bash
python -m pip install --force-reinstall "transkun==2.0.1"
```

### TransKun V2 Aug

`PIANO_TRANSKUN_V2_AUG` is a separate route backed by the official `checkpointTransformerAug.zip` archive. The downloader verifies the archive and loads `checkpointMSimplerAug/checkpoint.pt` with `model.conf`; neither V2 Aug nor default V2 silently replaces the other route.

```bash
python download_transkun_v2_aug_model.py
```

Default search roots include:

```text
~/.cache/music_ai_models/transkun_v2_aug
models/transkun_v2_aug
```

### Aria-AMT

Aria-AMT is another dedicated piano backend. The upstream README documents the `aria-amt transcribe` CLI; this project's wrapper currently calls `amt.run transcribe` through `src/core/aria_amt_transcriber.py`. The default checkpoint is:

```text
piano-medium-double-1.0.safetensors
```

Backend installation:

```bash
python -m pip install --no-deps --force-reinstall "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip"
```

Prepare the checkpoint:

```bash
python download_aria_amt_model.py
```

Default search roots include:

```text
~/.cache/music_ai_models/aria_amt
models/aria_amt
```

### ByteDance Pedal

ByteDance Pedal is a dedicated pedal-aware piano transcription backend for solo piano or clean piano stems. It comes from ByteDance's High-Resolution Piano Transcription with Pedals system. This project wraps it through `piano-transcription-inference` and preserves sustain pedal `CC64` events from the upstream MIDI output.

Dependencies:

```bash
python -m pip install "piano-transcription-inference==0.0.6" "torchlibrosa>=0.1.0,<0.2"
```

Prepare the checkpoint:

```bash
python download_bytedance_piano_model.py
```

Default search roots include:

```text
~/.cache/music_ai_models/bytedance_piano
models/bytedance_piano
```

## Piano Backend Selection Guide

All four piano routes are piano-specialized models. They do not perform full-mix multi-instrument recognition. The target determines the route:

| Goal | Recommended Mode | Notes |
|------|------------------|-------|
| Project-default TransKun route | `PIANO_TRANSKUN` | Uses the V2 resources bundled with the PyPI package. |
| Compare the official augmented checkpoint explicitly | `PIANO_TRANSKUN_V2_AUG` | Separately downloaded and verified V2 Aug assets; no fallback substitution for default V2. |
| Alternative modern piano AMT backend | `PIANO_ARIA_AMT` | Suitable for A/B testing on the same pure-piano inputs. |
| Output needs sustain pedal CC64, especially classical, lyrical, or legato-heavy piano | `PIANO_BYTEDANCE_PEDAL` | Preserves sustain pedal control events. The upstream ByteDance repository is archived, so validate it once in the target runtime. |

These piano results are not directly comparable with `YourMT3+` / `MIROS` multi-instrument outputs: piano backends model 88-key piano performance details, while multi-instrument backends handle instrument recognition and multi-track output for full mixes.

## Models and Public Comparisons

This section separates public benchmark claims from project integration status. The current published entry points expose `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_TRANSKUN_V2_AUG`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL`.

#### Integrated Backend Overview

| Backend / Model | Type | Project Entry | Public Quality Signal | Selection Notes |
|-----------------|------|---------------|-----------------------|-----------------|
| YourMT3+ | Multi-instrument AMT | Selectable directly in `SMART` and as five official checkpoint routes per separated WAV; conversion preserves official-writer notes and only adds required tempo metadata | Official Space default noPS result: Slakh `multi_f = 0.7398`; YourMT3+ paper table: Slakh2100 `Multi F1 = 74.84`, same table `MT3 = 62.0` | Default multi-instrument backend; the project default checkpoint is `YPTF.MoE+Multi (noPS)`, aligned with the official Hugging Face Space default. |
| MuScriptor Large | Multi-instrument AMT | Selectable in `SMART` and per separated WAV, with model-native hard instrument constraints and the official writer | Author `D_Test`: Onset / Frame / Offset / Drums / Multi F1 = **60.4 / 72.4 / 48.6 / 49.6 / 47.8**; YourMT3+ Multi F1 is 21.9 in the same table | Strong public full-mix candidate; author-set scores do not form a universal leaderboard, and weights are non-commercial. |
| MIROS | Multi-instrument AMT | Selectable in `SMART` and per separated WAV | 2025 AMT Challenge F1 **0.5998**, versus YourMT3-YPTF-MoE-M 0.5938 and MT3 0.3932 | Pinned MusicFM backend; the challenge used 76 constrained synthetic clips, so its score is not comparable to MuScriptor `D_Test` or Slakh. |
| TransKun default V2 | Piano-specialized | `PIANO_TRANSKUN` | The V2 / pip checkpoints publish MAESTRO V3 F1 values | Project default TransKun route with package-bundled resources. |
| TransKun V2 Aug | Piano-specialized | `PIANO_TRANSKUN_V2_AUG` | Official augmented checkpoint; this README does not transfer metrics from a different checkpoint | Separate, fixed-asset A/B route with no fallback substitution for default V2. |
| Aria-AMT | Piano-specialized | `PIANO_ARIA_AMT` | Public checkpoint; this README does not invent a missing same-protocol F1 score | Integrated pure-piano A/B option. |
| ByteDance Pedal | Piano-specialized / pedal-aware | `PIANO_BYTEDANCE_PEDAL` | MAESTRO note onset F1 / pedal onset F1 = 96.72% / 91.86% | Prefer when the output needs sustain pedal CC64; it is not a silent substitute for other piano backends. |
| Leap XE + PolarFormer | Vocal/accompaniment separation | Pre-separation for `VOCAL_SPLIT` | The two public models target different outputs, so no combined benchmark is claimed | Leap XE produces vocals; PolarFormer produces accompaniment; both stems then use the selected transcription backend. |
| BS-RoFormer SW Fixed | Six-stem separation | Pre-separation for `SIX_STEM_SPLIT` | MVSEP 6-stem SDR protocol | `BS-Rofo-SW-Fixed.ckpt` produces six WAV stems; separation SDR is not end-to-end MIDI F1. |

YourMT3+ / MuScriptor / MIROS are multi-instrument backends, TransKun / Aria-AMT / ByteDance Pedal are piano-specialized backends, and Leap XE / PolarFormer / BS-RoFormer SW Fixed are source-separation backends. Their public metrics use different tasks and protocols, so one combined leaderboard would be invalid.

#### MuScriptor and Frontier Watchlist (verified 2026-08-08)

| Model / Direction | Public Evidence | Status | Project Decision |
|---|---|---|---|
| MuScriptor Small / Medium | Official 103M / 307M weights; `D_Real`-only Multi F1 38.2 / 39.7, versus Large 40.5 in the same scale ablation | Integrated | Pinned independent selectors for lower VRAM and faster inference. They do not silently replace Large; quality, speed, latency, and memory are validated on the same real audio. |
| Mirelo Studio improved model | Mirelo says it uses more training data and is more accurate | Private service | Watch only. No public weights, revision, license mapping, or comparable score; it cannot be relabeled as `muscriptor-large`. |
| MIROS / MusicFM | 2025 AMT Challenge winner at F1 0.5998 on its own 76-clip protocol | Integrated | Keep as a separate backend and protocol, not as a numeric MuScriptor replacement. |
| Dense polyphony and instrument detection | The challenge paper reports MIROS F-measure dropping from 0.7193 for one instrument to 0.4367 for three and identifies leakage, similar timbres, and polyphonic confusion as persistent failures | Research priority | A complete model report includes instrument-aware F1, leakage, polyphony degradation, real jazz/pop coverage, weights, licensing, speed, and VRAM alongside the note score. |

#### YourMT3+ Official Checkpoint Modes

The official Hugging Face / Colab demo's model selector is a checkpoint / architecture selector, not a processing-workflow selector. This project aligns the YourMT3+ selector with the official list, then embeds the chosen checkpoint into workflows such as `SMART`, `VOCAL_SPLIT`, and `SIX_STEM_SPLIT`.

| Model | MoE | Pitch Shift | Notes |
|-------|-----|-------------|-------|
| YMT3+ | No | No | Baseline checkpoint from the official YourMT3+ model family. |
| YPTF+Single (noPS) | No | No | Perceiver-TF with a single decoder and no pitch-shift augmentation. |
| YPTF+Multi (PS) | No | Yes | Perceiver-TF with multi-t5 / multi-channel decoding. |
| YPTF.MoE+Multi (noPS) | 8 experts | No | Project default and official Hugging Face Space default; the Space result file reports Slakh `multi_f = 0.7398`. |
| YPTF.MoE+Multi (PS) | 8 experts | Yes | Optional pitch-shift MoE checkpoint; the YourMT3+ paper table reports Slakh `Multi F1 = 74.84` for its final model line. |

Main alignment points:

- The five official mode names, checkpoint directory mappings, and UI order match the official demo.
- `YPTF.MoE+Multi (noPS)` is the project default because it is the official Hugging Face Space default.
- All five checkpoints use the official Space argument table and official `update_config` path to build tokenizer, model, and audio configuration; older checkpoints no longer depend on guessed missing metadata.
- Older T5 checkpoints that do not store `ff_layer_type` are loaded with the standard T5 feed-forward layer type `t5_gmlp`.

Known differences from the official demo:

- The official demo runs one selected YourMT3 checkpoint; this project also adds separation workflows, dedicated piano models, tempo metadata, and stem-MIDI merging, without applying a second note-cleanup pass to official writer output.
- The official GPU Space usually runs 16-bit inference; this project defaults to full precision for better stability across Windows / CUDA environments.
- The product route uses the official non-overlapping slices and `inference_file(bsz=8)`; environment variables no longer alter the batch size of this official path.

#### Piano Model Quality Comparison

| Model | Current Project Entry | Same-Type Quality Protocol | Public Result | How To Read It |
|-------|-----------------------|----------------------------|---------------|----------------|
| TransKun V2 | Research checkpoint | MAESTRO V3 `note onset F1 / onset+offset F1 / onset+offset+velocity F1` | **0.9832 / 0.9349 / 0.9296** | Strong public piano AMT reference. |
| TransKun pip checkpoint (No Ext) | `PIANO_TRANSKUN` | MAESTRO V3 No Ext, same three metrics | **0.9833 / 0.8149 / 0.8109** | Project default route; upstream documents it as `without pedal extension of notes`. |
| TransKun V2 Aug | `PIANO_TRANSKUN_V2_AUG` | Official augmented checkpoint; metrics from other V2 checkpoints are not copied here | No cross-checkpoint F1 claimed | Compare default V2 and V2 Aug on the same local piano set. |
| Aria-AMT | `PIANO_ARIA_AMT` | Public checkpoint, but no fully matching published TransKun-style benchmark table | No unified F1 written here | Compare with local A/B audio. |
| ByteDance Pedal | `PIANO_BYTEDANCE_PEDAL` | MAESTRO `note onset F1 / pedal onset F1` | **96.72% / 91.86%** | Its same-type advantage is pedal output; generated MIDI preserves sustain pedal `CC64`. |

YourMT3+ / MuScriptor / MIROS are multi-instrument backends and are not directly rankable against the piano-specialized F1 scores above. ByteDance Pedal's `pedal onset F1` is also not equivalent to TransKun's `onset+offset+velocity F1`.

## Default Processing Strategy

The desktop, Space, Colab, and standalone Web interfaces no longer expose a user-adjustable quality preset. YourMT3+ uses official non-overlapping slices, fixed `bsz=8`, per-channel detokenization/merge, `mix_notes`, and its MIDI writer; MIROS preserves the official CLI writer result. MuScriptor keeps the pinned upstream v0.3.0 source, weights, five-second windows, and MIDI writer while exposing two explicit segment-state paths: the default “Official processing path” retains upstream segment behavior, while the opt-in “Segment-boundary continuity fix path” restores verified cross-segment continuous notes for same-input A/B comparison. This switch is independent of the tempo mode, and neither path receives project-level note quantization, filtering, or local `NoteEvent` regeneration.

For `SIX_STEM_SPLIT`, `BS-Rofo-SW-Fixed.ckpt` produces six real WAV stems. Each stem keeps an independent route selector and explicit conversion action; no MIDI backend is invoked merely because separation completed.

## Requirements

| Item | Requirement |
|------|-------------|
| Python | 3.11+; the Windows installer prefers 3.11-3.12 |
| PyTorch | Desktop/portable baseline: 2.7.0 with matching `torchaudio==2.7.0` and `torchvision==0.22.0` |
| Git | Required for source installation; installers stop explicitly when Git is unavailable |
| FFmpeg | Required for reliable MP3/M4A/FLAC/OGG handling |
| GPU | The complete Windows runtime supports NVIDIA CUDA 12.8 or Intel XPU in isolated environments. Linux/WSL2 remains NVIDIA-only; no entry point automatically downgrades to CPU/AMD |
| Disk | A complete cold install retains wheels, models, and download caches; the measured working set was about 32.36 GB, with at least 40 GB free space recommended at startup |
| OS | Windows 10/11, Linux, WSL2 |

Each platform has its own pinned compatibility envelope; cross-platform NumPy/Torch replacement is outside the supported combinations:

| Platform | Python / Torch | NumPy and GPU runtime | Release status |
|----------|----------------|-----------------------|----------------|
| Windows / NVIDIA desktop and portable target | Python 3.11-3.12; Torch 2.7.0 / torchaudio 2.7.0 / torchvision 0.22.0 | NumPy 1.26.4; CUDA 12.8 wheels | Source launchers verify this combination; `release.yml` revalidates the closed third-party inventory, exact model identities, and finished portable smoke tests before publishing |
| Windows / Intel XPU desktop and local portable target | Python 3.11-3.12; native Torch 2.11.0 XPU / torchaudio 2.11.0 XPU / torchvision 0.26.0 XPU | NumPy 1.26.4; `onnxruntime-openvino==1.24.1` + `openvino==2025.4.1`; the startup gate verifies FFT/STFT, BF16, and matrix probes remain on XPU, while PolarFormer uses `OpenVINOExecutionProvider` on `GPU.0` | The newest coherent PyTorch XPU trio covers Arc B-Series (Battlemage) and Core Ultra Series 3 (Panther Lake) in the official hardware matrix; Panther Lake requires Windows 11. Uses isolated `venv-xpu`; IPEX, CUDA ORT mixing, and CPU EP fallback are rejected. Official GitHub releases remain CUDA-only for now |
| Linux / NVIDIA source | Python 3.11+; Torch 2.7.0 / torchaudio 2.7.0 / torchvision 0.22.0 | NumPy 1.26.4; NVIDIA driver compatible with CUDA 12.8; `cu128` only | `install.sh` / `run.sh` verify the complete seven-mode runtime; `build.yml` performs source, test, and packaging checks only |
| Linux / AMD/ROCm | No complete seven-mode compatibility runtime | PolarFormer requires ONNX Runtime `CUDAExecutionProvider` | Currently unsupported; the installer stops explicitly instead of silently switching to CPU |
| Hugging Face Space | Python 3.12.12; Torch 2.8.0 / torchaudio 2.8.0 / torchvision 0.23.0 | NumPy `>=2,<2.5`; ZeroGPU | Uses `space/requirements.txt`; the desktop NumPy 1.26 pin is not part of the Space compatibility set |
| Google Colab | Current Colab Python and preinstalled Torch | Keeps preinstalled Torch; installs only pinned Web/runtime dependencies | Avoids replacing Torch and breaking its CUDA runtime |

The supported Windows source location is a local-disk, plain-ASCII path with no spaces:

```text
C:\MusicToMidi
D:\Projects\music-to-midi
```

Paths containing non-ASCII characters, spaces, or parentheses can cause PyTorch DLL loading failures. Mapped drives and UNC/SMB paths have not passed the source-environment checks and can make the path recorded by `venv` differ from the launch path; a local NTFS directory is the currently supported source location.

## Quick Start

### MuScriptor gated authorization

MuScriptor Small, Medium, and Large are Hugging Face gated models and cannot be downloaded anonymously. Source installation is available when one Hugging Face account has accepted the browser terms for [Small](https://huggingface.co/MuScriptor/muscriptor-small), [Medium](https://huggingface.co/MuScriptor/muscriptor-medium), and [Large](https://huggingface.co/MuScriptor/muscriptor-large), and that approved account authenticates from the repository's isolated environment:

```powershell
# Windows bootstrap without venv: minimal environment with the official HF CLI
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install "huggingface_hub>=0.20,<2" "hf_xet==1.5.2"
.\venv\Scripts\hf.exe auth login
```

```bash
# Linux / WSL2
python3.11 -m venv venv
./venv/bin/python -m pip install "huggingface_hub>=0.20,<2" "hf_xet==1.5.2"
./venv/bin/hf auth login
```

Authentication uses a personal token with read access to all three gated repositories. The repository, README, command-line arguments, and logs are not supported token locations; unattended environments use a protected `HF_TOKEN` secret. Before any other large model download, the aggregate downloader performs three metadata-only permission checks and stops immediately if one repository is not accepted or the account is not authenticated. A complete portable release already contains identity-verified weights and does not download them from the Hub at startup; the CC BY-NC 4.0 license and additional terms still apply.

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

You can also double-click `run.bat`. `run.ps1` checks the virtual environment, Beat This `final0`, all five YourMT3+ modes, MuScriptor Large / Medium / Small, BS-RoFormer SW Fixed, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance Pedal, MIROS, SoundFont, and FluidSynth, then calls `install.ps1` if something is missing or invalid.

An Intel GPU uses the isolated native XPU environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_xpu.ps1
powershell -ExecutionPolicy Bypass -File .\run_xpu.ps1
```

Installation and every launch perform real `torch.xpu` matrix, FFT, STFT, and BF16 convolution operations while rejecting XPU-to-CPU operator fallback. They then run a minimal ONNX MatMul graph through `OpenVINOExecutionProvider` on `GPU.0` with CPU EP fallback disabled. ORT may list its built-in CPU provider, but `session.disable_cpu_ep_fallback=1` makes any CPU-assigned node fail session creation. A failed gate stops instead of switching to IPEX, DirectML, CUDA, or CPU.

PolarFormer caps the model's 882000-sample window at 441000 by default to control peak device memory; this default completed a real two-model split on the 16 GiB NVIDIA baseline. `POLARFORMER_MAX_CHUNK_SIZE=220500` selects an explicitly lower peak, while `0` removes the cap. An OOM does not trigger a silent window-size retry.

On XPU, Leap XE retains the official full approximately 20-second audio window, all keys/values, checkpoint, and post-processing. Only the attention query axis is evaluated in fixed 128-row slices and concatenated. Every query still attends to the complete context, so this is a numerically equivalent inference-time memory bound rather than a shorter window, smaller model, or CPU fallback; accidental training-mode use fails explicitly. This path avoids relying on the XPU Flash Attention kernel, whose architecture coverage is narrower than the full PyTorch XPU hardware matrix, and completed a real two-track split on a 16 GB Arc 140T.

### Linux / WSL2

```bash
chmod +x run.sh
./run.sh
```

`run.sh` checks the virtual environment, core imports, Beat This `final0`, YourMT3+ source and all five model modes, MuScriptor Large / Medium / Small, BS-RoFormer SW Fixed, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance Pedal, MIROS, SoundFont, and FluidSynth, then calls `install.sh` if something is missing or invalid.

### Direct Source Run

The supported source runtime is the repository's isolated `venv`. Before importing GUI or model dependencies, the entry point strictly validates the interpreter path, `include-system-site-packages=false`, the MuScriptor package location and version, and all seven pinned source SHA-256 hashes. A bare global `python -m src.main` invocation is rejected.

```powershell
# Windows desktop app
.\venv\Scripts\python.exe -m src.main

# Windows Web backend and frontend; detects LAN IPv4 and opens the frontend
.\venv\Scripts\python.exe -m src.web
```

```bash
# Linux / WSL2 desktop app
./venv/bin/python -m src.main

# Linux / WSL2 Web backend and frontend; open the printed URL manually
./venv/bin/python -m src.web --no-window
```

## Manual Setup

### 1. Create a Virtual Environment

Windows:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip wheel "setuptools==80.10.2"
```

Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel "setuptools==80.10.2"
```

### 2. Install PyTorch

CUDA 12.8 (the supported complete seven-mode runtime, checked strictly by the launchers):

```bash
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

`cu118` / CUDA 11 is outside the current one-click launcher and complete seven-mode acceptance contract; launchers do not silently treat it as an aligned runtime.

The standard Windows Intel XPU installer is `install_xpu.ps1`. Manual preparation corresponds to the isolated `venv-xpu` and the exact versions in `requirements-xpu.txt`; overwriting CUDA wheels in `venv` breaks the environment-isolation contract. The project selects the newest coherent trio rather than the highest standalone Torch version: Torch XPU wheels currently extend beyond 2.11, but the newest matching torchaudio XPU wheel is 2.11, so the contract is `torch/torchaudio==2.11.0+xpu` plus `torchvision==0.26.0+xpu`. The [official PyTorch 2.11 matrix](https://docs.pytorch.org/docs/2.11/notes/get_start_xpu.html) includes Arc B-Series and Panther Lake / Core Ultra Series 3. PolarFormer uses [ONNX Runtime OpenVINO 1.24.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1), aligned to OpenVINO 2025.4.1.

Intel XPU has no project-level compatibility number directly equivalent to NVIDIA `sm_XX`. The stack discovers devices through oneAPI/Level Zero, and the normal JIT path lets the Intel Graphics Compiler generate code for the detected hardware. The real support boundary is therefore the pinned PyTorch release's hardware matrix plus its OS/driver requirements and this project's launch-time operator probes; an arbitrary Intel GPU is not automatically compatible.

On XPU, the 5.1 GiB MuScriptor Large checkpoint is read with the official `pread` backend from `safetensors==0.8.0`, in file-offset order and one tensor at a time. The lazy `pread` file handle is opened before allocating the roughly 5.50 GiB unified-memory model, avoiding the Windows `os error 1455` observed when the checkpoint was opened only after model allocation consumed system commit. This does not expose the complete checkpoint as a PyTorch memory mapping or change the official weights, precision, or writer. A read failure stops explicitly; it does not switch to mmap, resize the page file, or retry silently.

The local Intel XPU Web backend starts a fresh processing process for every GPU job and lets the operating system reclaim that process's complete address space at the terminal state. The persistent HTTP process therefore retains no YourMT3, MIROS, MuScriptor, or separator model between jobs, preventing unified-memory models from accumulating against the Windows system commit limit during long sessions. Stop terminates and reaps the active process; a non-zero exit, missing response manifest, or missing output file is reported as a failure, with no retry, CPU switch, or fabricated success.

`torchaudio` 2.11 delegates `load/save` to TorchCodec, whose Windows wheels require full-shared FFmpeg DLLs. The project does not hide missing DLLs behind a fallback: public inputs are first converted to WAV by the bundled FFmpeg, then a pinned libsndfile PCM reader creates channels-first float32 tensors; resampling and inference remain on the validated XPU route.

AMD/ROCm cannot currently run the complete seven-mode surface: the fixed separator contracts validate either NVIDIA `CUDAExecutionProvider` or Intel `OpenVINOExecutionProvider/GPU.0`, with no strict AMD GPU provider. The installer stops explicitly instead of silently switching to CPU.

`release.yml` produces a CUDA 12.8 GPU portable build only; it does not publish a CPU variant. The current closed inventory contains 29 third-party components: 25 are `VERIFIED`, 4 are `OWNER_ACCEPTED` with named maintainer responsibility and a revocation contact, and 0 are `BLOCKED`. Every release revalidates that inventory, model identities, the SBOM, the packaged FFmpeg build, and the finished application smoke test; any failed requirement stops the release. Push/PR `build.yml` jobs validate source, tests, and packaging contracts but produce no portable artifact. For local source development, CPU-only PyTorch remains a manual choice with slower inference and different dependency compatibility.

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
python -m pip install --no-deps "audio-separator==0.44.1"
python -m pip install --no-deps --force-reinstall "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip"
python -m pip install --no-deps --force-reinstall "muscriptor @ https://github.com/muscriptor/muscriptor/archive/d73147e75e5b9b0c0a79ebe154587db4fd603e0c.zip"
python -m src.utils.source_runtime
```

`requirements.txt` intentionally prevents audio-separator's NumPy 2 metadata and Aria-AMT's older torchaudio constraint from replacing the desktop compatibility stack. The pinned `--no-deps` commands above provide their separate installation. MuScriptor is also installed from the exact official v0.3.0 commit without dependency resolution, then its environment, package path, version, and source hashes are validated together. `install.ps1` / `install.sh` provide the complete pinned companion dependency set.

### 4. Prepare YourMT3+ Source and Weights

```bash
python download_sota_models.py
```

The repository already includes the controlled, compatibility-patched `YourMT3/amt/src`; mutable upstream `master` does not satisfy its source-identity check. `download_sota_models.py` prepares Beat This `final0`, all five official YourMT3+ checkpoints, pinned MIROS source and both weights, MuScriptor Large / Medium / Small, `BS-Rofo-SW-Fixed.ckpt`, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance, MuseScore General SoundFont, and FluidSynth, and strictly verifies the default TransKun 2.0.1 package and bundled V2 resources.

### 5. Prepare Separation and Piano Models

```bash
python download_vocal_model.py
python download_multistem_model.py
python download_accompaniment_model.py
python download_transkun_v2_aug_model.py
python download_aria_amt_model.py
python download_bytedance_piano_model.py
python download_miros_model.py
```

The default cache location is:

```text
~/.cache/music_ai_models/yourmt3_all
~/.music-to-midi/models/beat_this
~/.music-to-midi/models/audio-separator
~/.cache/music_ai_models/transkun_v2_aug
~/.cache/music_ai_models/aria_amt
~/.cache/music_ai_models/bytedance_piano
~/.cache/music_ai_models/fluidsynth/2.5.6
${HF_HOME:-~/.cache/huggingface}/hub  # three MuScriptor tiers and MuseScore SoundFont
external/ai4m-miros
```

On Windows, `~` means `%USERPROFILE%`. Source runs use the repository's accelerator-specific virtual environment: `venv` for NVIDIA CUDA and `venv-xpu` for Windows Intel XPU, selected by `run.ps1` or `run_xpu.ps1`. Desktop output defaults to `MidiOutput\<audio-file-name>` inside the repository, and logs are written to `%USERPROFILE%\.music-to-midi\logs`. Default TransKun V2 resources live inside the selected environment's `transkun==2.0.1` package. Portable builds instead read their packaged `models`, `runtime`, and `tools` directories and do not depend on these source caches.

Default TransKun V2 resources are bundled with `transkun==2.0.1`. Missing or mismatched `PIANO_TRANSKUN` resources correspond to `python -m pip install --force-reinstall "transkun==2.0.1"`. `PIANO_TRANSKUN_V2_AUG` uses a separate cache prepared by `python download_transkun_v2_aug_model.py`.

### 6. Launch

```powershell
# Windows
.\venv\Scripts\python.exe -m src.main
```

```bash
# Linux / WSL2
./venv/bin/python -m src.main
```

## Google Colab

Notebook entry:

```text
colab_notebook.ipynb
```

Steps:

1. Open the notebook.
2. Select a GPU runtime.
3. To use MuScriptor, first accept all three gated repository terms, store the token as the private Colab secret `HF_TOKEN`, and enable `ENABLE_MUSCRIPTOR` in code cell 3; that cell verifies all three permissions before launch.
4. Run the remaining cells in order.
5. The final cell launches Gradio and prints a public URL.

The Colab setup preserves the preinstalled PyTorch package to avoid CUDA runtime conflicts.

## Gradio Space

Space entry:

```text
space/app.py
```

Local launch:

```bash
cd space
python app.py
```

The Space deployment bundles the project's verified, compatibility-patched `YourMT3/amt/src` tree, identical to the desktop and Colab source; it does not switch to mutable Hugging Face Space source at runtime. During conversion it checks or prepares only the resources required by the selected mode: the selected official YourMT3+ checkpoint or MIROS, BS-RoFormer SW Fixed, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, or ByteDance Pedal. Missing resources or identity mismatches are surfaced explicitly.

The ZeroGPU deployment is a short-clip demo, not a promise that full songs complete end to end. The [Hugging Face ZeroGPU documentation](https://huggingface.co/docs/hub/main/en/spaces-zerogpu) currently lists daily quotas of 2 GPU minutes for anonymous users and 5 minutes for logged-in free accounts. The conservative minimum request already exceeds the anonymous allowance after the platform's `large` GPU multiplier, so conversion currently requires sign-in. The Space estimates each mode/backend/model combination, applies the pinned `spaces==0.51.1` multiplier upper bound, and rejects requests above one 300-second logged-in free-account window before downloading models. The estimate is an admission ceiling, not a guarantee of remaining daily quota or queue capacity; Colab, the desktop build, or dedicated GPU hardware are better suited to long songs.

Under the current formula, the maximum input lengths below are exact admission thresholds rather than measured-runtime promises. They apply to the default `YPTF.MoE+Multi (noPS)` checkpoint and MIROS; other YourMT3 checkpoints use their own factors.

| ZeroGPU route | YourMT3 default noPS | MIROS |
|---------------|---------------------:|------:|
| `SMART` | 2.00 s | 1.00 s |
| `VOCAL_SPLIT` | 0.53 s | 0.27 s |
| `SIX_STEM_SPLIT` | 0.22 s | 0.11 s |
| Any dedicated piano mode | 2.50 s | N/A |

Failed Space requests delete their request directory immediately. Successful outputs remain available for Gradio download and become eligible for expiration cleanup after 24 hours by default; they are removed on a later cleanup pass or normal Space process exit. Colab also removes failed request directories immediately and retains successful outputs until that runtime exits; its Gradio cache makes files eligible for cleanup after 24 hours.

## Portable Build

The Windows CUDA package command produces the desktop App, Web backend, and Web frontend:

The portable gate accepts components recorded as `VERIFIED` or explicitly `OWNER_ACCEPTED`. The current `THIRD_PARTY_NOTICES.md` records 25 verified and four owner-accepted components. Owner acceptance is revocable and does not represent an upstream license grant; running the local command below does not create redistribution rights.

```powershell
powershell -ExecutionPolicy Bypass -File .\build_web_executables.ps1
```

The Intel XPU package command uses the separately validated `venv-xpu`:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable_xpu.ps1
```

CUDA outputs `dist\MusicToMidi-App`, `dist\MusicToMidi-WebBackend`, and `dist\MusicToMidi-WebFrontend`. XPU outputs `dist\MusicToMidi-XPU-App`, `dist\MusicToMidi-XPU-WebBackend`, and the same accelerator-neutral `dist\MusicToMidi-WebFrontend`. Each `_internal` directory belongs to its corresponding role.

The complete XPU package exceeds 20 GB. When the repository volume is too small, both roots can use another NTFS volume with enough free space; missing models or an incomplete package fail packaged-file validation:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable_xpu.ps1 `
  -BuildRoot C:\MusicToMidi-XPU-build `
  -DistRoot C:\MusicToMidi-XPU-dist
```

The XPU wrapper hard-links staged assets only when source and staging are on the same volume; cross-volume assets are copied. Staged and packaged SHA-256/manifest validation runs in both cases. A hard-link error stops the build without falling back to copying. Valid `BuildRoot` and `DistRoot` values are disjoint ordinary directories; drive roots and the project root are rejected by argument validation.

Explicit Python or FFmpeg paths:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_web_executables.ps1 `
  -PythonExe .\venv\Scripts\python.exe `
  -FfmpegDir C:\ffmpeg\bin
```

The build script requires and strictly validates every resource below. Missing assets or mismatched size, SHA256, source manifest, or runtime-package identity stop the build before PyInstaller runs:

```text
YourMT3/amt/src
YourMT3 model cache -> models/yourmt3_all
audio-separator model cache -> models/audio-separator
transkun==2.0.1 package and its bundled default V2 resources
TransKun V2 Aug model cache -> models/transkun_v2_aug
Aria-AMT model cache -> models/aria_amt
ByteDance Piano model cache -> models/bytedance_piano
pinned, compatibility-patched MIROS source and both weights
ffmpeg.exe / ffprobe.exe
```

Portable asset source priority:

```text
MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR or ~/.cache/music_ai_models/yourmt3_all or checkpoints/yourmt3_all
MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR or ~/.music-to-midi/models/audio-separator or checkpoints/audio-separator
MUSIC_TO_MIDI_BUNDLE_TRANSKUN_V2_AUG_DIR or ~/.cache/music_ai_models/transkun_v2_aug or checkpoints/transkun_v2_aug
MUSIC_TO_MIDI_BUNDLE_ARIA_AMT_DIR or ~/.cache/music_ai_models/aria_amt or checkpoints/aria_amt
MUSIC_TO_MIDI_BUNDLE_BYTEDANCE_PIANO_DIR or ~/.cache/music_ai_models/bytedance_piano or checkpoints/bytedance_piano
MUSIC_TO_MIDI_BUNDLE_MIROS_DIR or external/ai4m-miros / ai4m-miros / .tmp/ai4m-miros
MUSIC_TO_MIDI_BUNDLE_FFMPEG_DIR or tools/ffmpeg / ffmpeg
```

Complete output directories:

```text
dist/MusicToMidi-App/
dist/MusicToMidi-WebBackend/
dist/MusicToMidi-WebFrontend/
```

Each directory is the complete unit for its role. Desktop users use the App folder; Web users use both WebBackend and WebFrontend.

## Project Structure

```text
src/
  core/
    pipeline.py              # Main processing pipeline
    yourmt3_transcriber.py   # YourMT3+ backend
    miros_transcriber.py     # Local MIROS wrapper
    transkun_transcriber.py  # TransKun default V2 piano backend
    transkun_v2_aug_transcriber.py # TransKun V2 Aug piano backend
    aria_amt_transcriber.py  # Aria-AMT piano backend
    bytedance_piano_transcriber.py # ByteDance Pedal piano backend
    vocal_separator.py       # Vocal/accompaniment separation
    multi_stem_separator.py  # Six-stem separation
    midi_generator.py        # MIDI generation and post-processing
    beat_detector.py         # BPM/beat detection
  gui/
    main_window.py           # PyQt6 main window
    widgets/track_panel.py   # Mode, backend, and layout selector
    workers/processing_worker.py
  models/
    data_models.py           # Config, ProcessingResult, NoteEvent, etc.
    gm_instruments.py        # GM 128 instrument mapping
  utils/
    runtime_paths.py         # Runtime resource paths
    yourmt3_downloader.py    # YourMT3+ model path and download helpers

space/app.py                 # Gradio Web UI
colab_notebook.ipynb         # Colab entry
download_sota_models.py      # Beat This + five YourMT3 + MIROS + three MuScriptor + separation + four piano + playback assets
download_vocal_model.py      # Leap XE vocals asset download
download_accompaniment_model.py # PolarFormer accompaniment asset download
download_multistem_model.py  # BS-RoFormer SW Fixed six-stem asset download
download_transkun_v2_aug_model.py # TransKun V2 Aug download and validation
download_aria_amt_model.py   # Aria-AMT model download
download_bytedance_piano_model.py # ByteDance Pedal model download
download_vocal_harmony_model.py # Historical compatibility entry for PolarFormer accompaniment
MusicToMidi.spec             # PyInstaller configuration
```

## Development Commands

These commands use the repository's activated `venv`.

```bash
pytest
pytest tests/test_yourmt3_integration.py -v
black src/
isort src/
flake8 src/
mypy src/
pyinstaller MusicToMidi.spec
```

Useful self-checks:

```bash
python -m src.main --self-test
python -c "from src.utils.gpu_utils import print_gpu_diagnosis; print_gpu_diagnosis()"
python -c "from src.core.yourmt3_transcriber import YourMT3Transcriber; print(YourMT3Transcriber.is_available())"
```

## Troubleshooting

### PyTorch DLL Loading Fails

Check:

- Whether the project path contains non-ASCII characters, spaces, or parentheses.
- Whether Visual C++ Redistributable 2022 x64 is installed.
- Whether PyTorch, torchaudio, and torchvision versions match.

On Windows, rerun:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

### FFmpeg Is Unavailable

The Windows installer can install FFmpeg automatically, or you can install it manually and add it to PATH. Linux:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### YourMT3+ Is Unavailable

Check source:

```text
YourMT3/amt/src
```

Check model:

```bash
python -c "from src.utils.yourmt3_downloader import get_model_path; print(get_model_path())"
```

If missing:

```bash
python download_sota_models.py
```

If the controlled `YourMT3/amt/src` tree is missing, its supported recovery source is the current project revision. Mutable upstream `master` cannot satisfy three-interface source parity or the portable source-manifest check.

### Vocal Separation Is Unavailable

Dependency and model check:

Windows / Linux NVIDIA CUDA:

```bash
python -m pip install --no-deps "audio-separator==0.44.1" "onnxruntime-gpu==1.23.2"
python download_vocal_model.py
python download_accompaniment_model.py
```

On macOS or an explicitly CPU-only environment, replace `onnxruntime-gpu==1.23.2` with `onnxruntime==1.23.2`.

### Six-Stem Separation Is Unavailable

This route depends on `audio-separator==0.44.1` and the BS-RoFormer SW Fixed resources:

```bash
python download_multistem_model.py
```

### Dedicated Piano Transcription Is Unavailable

Default TransKun V2 mode needs the `transkun` package and its bundled pretrained resources:

```bash
python -m pip install --force-reinstall "transkun==2.0.1"
```

TransKun V2 Aug mode uses a separate, fixed-asset checkpoint:

```bash
python download_transkun_v2_aug_model.py
```

Aria-AMT mode needs the `aria-amt` package and checkpoint:

```bash
python -m pip install --no-deps --force-reinstall "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip"
python download_aria_amt_model.py
```

ByteDance Pedal mode needs `piano-transcription-inference`, `torchlibrosa`, and the ByteDance Piano checkpoint:

```bash
python -m pip install "piano-transcription-inference==0.0.6" "torchlibrosa>=0.1.0,<0.2"
python download_bytedance_piano_model.py
```

### MIROS Is Unavailable

Check local repository files:

```text
ai4m-miros/main.py
ai4m-miros/transcribe.py
```

If the error lists missing Python modules, the dependency source of record is the upstream MIROS repository.

## License

This project uses the MIT License. Third-party models, datasets, and upstream repositories remain governed by their own licenses and terms; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for adapted-code notices and full license text.
