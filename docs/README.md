# Music to MIDI Converter (AI Audio to MIDI)

<p align="center">
  <a href="../README.md">中文</a> | English
</p>

Music to MIDI is a local-first AI audio-to-MIDI converter for music producers, transcription hobbyists, piano learners, sampling workflows, and automatic music transcription (AMT) experiments. Drop in an `MP3`, `WAV`, `FLAC`, `OGG`, or `M4A` file, then generate editable MIDI from the PyQt6 desktop app, the Gradio Web interface, or the Google Colab notebook.

The current product surface syncs seven processing modes: full-mix multi-instrument transcription, vocal/accompaniment WAV separation, six-stem WAV separation, and four dedicated piano routes through TransKun default V2, TransKun V2 Aug, Aria-AMT, or ByteDance Pedal. Both separation buttons deliver WAV files only; after separation, each track can explicitly use one of 13 MIDI routes in the same result workbench. The project is more than a one-note melody extractor: it brings multi-instrument AI music transcription, stem separation, piano-to-MIDI conversion, and BPM/tempo metadata into one workflow.

Every one of the seven modes and every per-track conversion uses Beat This `final0` as its only beat/downbeat detector. Competing marks are cleaned, missed musical positions are counted, and global BPM is fitted by least squares; meter is inferred independently and remains unknown rather than inventing 4/4. Constant recordings receive one tempo event, expressive recordings automatically receive a beat-level tempo map. A manual 4–400 BPM value remains an explicit project-tempo override. This is not quantization or note cleanup: event payloads and musical tick positions are preserved.

The Standard MIDI tempo, meter, and every non-tempo event are verified after publication. DAWs must enable tempo-map import. MuseScore 3/4 may classify an unquantized performance MIDI as human performance, run its own beat tracker, and replace the tempo shown on the notation page. That displayed value is produced by MuseScore's MIDI importer; the project does not quantize or move model notes merely to force a notation application to display the file tempo.

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

Use it when you want to turn a vocal line, piano recording, full mix, or separated stem into MIDI you can edit in a DAW. It is designed for users who want more control than a simple upload-and-download converter, while still keeping the common audio-to-MIDI path approachable.

## Current Capabilities

- **Full-mix transcription**: `SMART` mode sends the whole song to YourMT3+, MIROS, or MuScriptor Large / Medium / Small and exports MIDI with notes, drums, and instrument tracks.
- **Vocal/accompaniment separation and per-track transcription**: `VOCAL_SPLIT` uses BS-RoFormer Leap XE 90-band for vocals and BS PolarFormer for accompaniment, and first delivers two real WAV tracks. Each track can then select one of 13 explicit MIDI routes in the track workbench.
- **Six-stem separation and per-track transcription**: `SIX_STEM_SPLIT` uses `BS-Rofo-SW-Fixed.ckpt` to deliver six real `bass / drums / guitar / piano / vocals / other` WAV tracks. Separation does not silently transcribe or merge MIDI.
- **Dedicated piano transcription**: `PIANO_TRANSKUN`, `PIANO_TRANSKUN_V2_AUG`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL` target pure piano audio through TransKun default V2, the official V2 Aug checkpoint, Aria-AMT, or ByteDance's pedal-aware model.
- **Default backend semantics**: the multi-instrument default remains the official YourMT3+ `YPTF.MoE+Multi (noPS)` checkpoint. `SMART` can explicitly select YourMT3+, MIROS, or one of the three MuScriptor tiers, and every separated WAV can select the same multi-instrument families independently.
- **Real MuScriptor constraints**: an empty instrument list enables model detection. A non-empty list is passed to the official `instruments` plus `prelude_forcing` decoding path, masks every unselected instrument token during generation, and is validated again against streamed events and final MIDI.
- **Explicit multi-instrument routes**: `SMART` and every separated WAV can select five YourMT3+ checkpoints, MIROS, or MuScriptor Large / Medium / Small. The per-track menu also exposes four piano-specialized routes, for 13 routes in total.
- **Official transcription output**: YourMT3+ and MIROS preserve their official writer results; MuScriptor uses its official events and MIDI writer plus strict selected-instrument validation. The project does not add quantization, de-duplication, short-note filtering, velocity smoothing, polyphony limiting, or local `NoteEvent` regeneration.
- **Common audio formats**: `MP3`, `WAV`, `FLAC`, `OGG`, and `M4A` are accepted. Non-WAV input must be converted to 44.1 kHz PCM WAV through FFmpeg; FFmpeg failures stop processing and show the stderr root cause.
- **Consistent mode set**: desktop, Space, and Colab expose the same seven processing modes.

## Interface Matrix

| Interface | Modes | Backend Selection | Best For |
|-----------|-------|-------------------|----------|
| PyQt6 desktop | `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_TRANSKUN_V2_AUG`, `PIANO_ARIA_AMT`, `PIANO_BYTEDANCE_PEDAL` | `SMART` selects YourMT3+ / MIROS / MuScriptor; separated WAV tracks expose 13 routes; piano modes use their dedicated backend | Local GPU use, persistent output folders, and dedicated piano transcription |
| Gradio Space | Same seven modes as desktop | MuScriptor instrument search/multi-select, hard decoding constraint, and real MIDI workbench are synchronized | Browser-based use or hosted demos |
| Google Colab | Same seven modes as desktop | Same MuScriptor constraint and linked WAV/MIDI result workbench as Space | Temporary Colab GPU sessions |

## Entry And Dependency Sync Status

This repository keeps project workflows separate from official YourMT3+ checkpoint modes:

- `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_TRANSKUN_V2_AUG`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL` are this project's seven processing workflows.
- `YMT3+`, `YPTF+Single (noPS)`, `YPTF+Multi (PS)`, `YPTF.MoE+Multi (noPS)`, and `YPTF.MoE+Multi (PS)` are the five checkpoint / architecture modes exposed by the official YourMT3 demo.
- Desktop, Gradio Space, and Colab expose the same seven workflows. `SMART` selects YourMT3+, MIROS, or MuScriptor Large / Medium / Small; both separation workflows first deliver WAV tracks, and each track then exposes 13 explicit MIDI routes.

Current synchronization coverage:

| Location | Synced Content | Notes |
|----------|----------------|-------|
| `download_sota_models.py` | Prepares Beat This `final0`; all five official YourMT3+ checkpoints; pinned MIROS source plus both weights; MuScriptor Large / Medium / Small; `BS-Rofo-SW-Fixed.ckpt`; Leap XE; PolarFormer; TransKun V2 Aug; Aria-AMT; ByteDance; MuseScore General SoundFont; and FluidSynth, while strictly validating the default TransKun 2.0.1 package and bundled V2 resources | Fixed-source resources are validated by known size/SHA256 or their explicit source/runtime identity; any required-resource failure stops the command. |
| `run.ps1` / `run.sh` | Checks all official YourMT3+ modes, MuScriptor Large / Medium / Small, BS-RoFormer SW Fixed, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance Pedal, MIROS, SoundFont, FluidSynth, and separator availability before launch | Missing or invalid required resources are reported explicitly. |
| `install.ps1` / `install.sh` | Installs PyTorch 2.7, NumPy 1.26, audio-separator 0.44.1 runtime pins, the identity-verified official MuScriptor v0.3.0 runtime, and every required model/runtime asset | `audio-separator` is installed with `--no-deps` to avoid pulling NumPy 2 into the current PyTorch / desktop stack. |
| `.github/workflows/build.yml` | Push/PR jobs run Linux and Windows source, test, and packaging-contract checks only | They produce no portable artifact and never use empty directories or fake models to bypass mandatory bundle validation. |
| `.github/workflows/release.yml` | The complete portable-build pipeline; it downloads and strictly verifies every YourMT3+, separator, MIROS, MuScriptor, TransKun, Aria-AMT, ByteDance, playback, and runtime asset | The 28-component gate currently records 24 `VERIFIED` and four explicitly documented `OWNER_ACCEPTED` items. The target GPU runtime is PyTorch 2.7 + CUDA 12.8; an owner acceptance is a revocable distribution decision, not a claim that upstream granted a license. |
| `colab_notebook.ipynb` | Keeps Colab's preinstalled Torch, installs pinned Web/runtime dependencies, and synchronizes all seven modes | `SMART` and the per-track workbench expose YourMT3+, MIROS, and MuScriptor Large / Medium / Small; the per-track menu contains 13 routes in total. |

## Processing Modes

| Mode | Internal Pipeline | Main Output | Notes |
|------|-------------------|-------------|-------|
| `SMART` | Audio -> selected YourMT3+ / MIROS / MuScriptor Large, Medium, or Small -> MIDI | `<song>.mid` | No source separation. A non-empty MuScriptor instrument selection is a real decoding constraint. |
| `VOCAL_SPLIT` | Audio -> Leap XE vocals + PolarFormer accompaniment -> two WAV tracks -> explicit per-track MIDI | `<song>_vocals.wav`, `<song>_accompaniment.wav`; per-track MIDI on request | Separation does not auto-transcribe. Each WAV independently selects one of 13 routes. |
| `SIX_STEM_SPLIT` | Audio -> `BS-Rofo-SW-Fixed.ckpt` -> six WAV tracks -> explicit per-track MIDI | `<song>_<stem>.wav`; per-track MIDI on request | Each real WAV independently selects its route and whether to convert; MIDI is not auto-merged. |
| `PIANO_TRANSKUN` | Audio -> TransKun default V2 model -> MIDI | `<song>_piano_transkun.mid` | Best for pure piano audio; uses the checkpoint resources bundled with the PyPI package. |
| `PIANO_TRANSKUN_V2_AUG` | Audio -> official TransKun V2 Aug checkpoint -> MIDI | `<song>_piano_transkun_v2_aug.mid` | Independent mode with a separately downloaded and verified checkpoint; it is not a fallback for default V2. |
| `PIANO_ARIA_AMT` | Audio -> Aria-AMT piano model -> MIDI | `<song>_piano_aria.mid` | Best for pure piano audio; expects the Aria-AMT checkpoint to be bundled or present in the model directory. |
| `PIANO_BYTEDANCE_PEDAL` | Audio -> ByteDance pedal-aware piano model -> MIDI | `<song>_piano_bytedance_pedal.mid` | Best for pure piano audio when the output needs sustain pedal CC64; expects the ByteDance Piano checkpoint to be bundled or present in the model directory. |

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

The source tree must include:

```text
YourMT3/amt/src/model/ymt3.py
YourMT3/amt/src/utils/task_manager.py
YourMT3/amt/src/config/config.py
```

The complete project checkout includes a compatibility-patched `YourMT3/amt/src` tree protected by a fixed manifest. If it is missing, restore it from the current project revision; do not overwrite it with mutable upstream `master`. A separate upstream clone is suitable for experiments but does not satisfy this project's three-interface source-parity or portable-build identity contract.

Download model weights:

The source-maintenance commands below assume that the repository's `venv` is active. Otherwise, replace `python` / `hf` with `.\venv\Scripts\python.exe` / `.\venv\Scripts\hf.exe` on Windows, or `./venv/bin/python` / `./venv/bin/hf` on Linux/WSL2. Do not use a global Python interpreter.

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

The three gated checkpoints are pinned independently: [`muscriptor-large`](https://huggingface.co/MuScriptor/muscriptor-large) revision `8809fdfbed2affa7ade94a7059e746e3880720e7`, [`muscriptor-medium`](https://huggingface.co/MuScriptor/muscriptor-medium) revision `f32236969308476e01fd3aae67357de5feb05a2d`, and [`muscriptor-small`](https://huggingface.co/MuScriptor/muscriptor-small) revision `8c127f603b807520fa465c838e9bfee8a91ada4e`. All use CC BY-NC 4.0 with additional lawful-use terms. The same Hugging Face account must accept all three repository conditions in a browser and then authenticate the CLI. `hf auth login` cannot accept the web terms on the user's behalf, and source installs, Colab sessions, and self-hosted Spaces cannot download these weights anonymously:

```bash
hf auth login
python download_muscriptor_model.py --size all
```

All three are explicit choices and never silently replace one another. Large prioritizes quality, Medium trades speed against quality, and Small has the fewest parameters and fastest runtime. Every tier uses the official 5-second window, prelude forcing, and single-generation path. Large is a decoder-only Transformer with roughly 1.3B parameters (the current code README rounds it to 1.4B), 48 layers, and hidden dimension 1536. It consumes 5-second 16 kHz mono chunks and emits MT3-style onset, offset, pitch, and 36-group instrument events. Training combines about 1.45 million MIDI files for synthetic pretraining, 170,000 real recordings / about 11,000 hours for fine-tuning, and 300 curated tracks for RL post-training.

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

TelkNet boundary: with authorization, this audit inspected private `mason369/telknet` dev commit `52be6fec179be492f5229ba149545ac2833b284a`. This project only aligns its core YourMT3/MIROS rule—official writer output followed only by tempo metadata, with no generic note cleanup. Both separation workflows likewise deliver WAV first; MIDI is explicitly triggered in this project's per-track workbench. There is no evidence that this dev commit is the deployed production revision, and no line-for-line routing, environment, or bit-identical-output claim is made.

Prepare the assets explicitly:

```bash
python download_vocal_model.py
python download_accompaniment_model.py
```

### TransKun Default V2

TransKun default V2 is a dedicated piano transcription backend for pure or piano-forward audio. The project calls the pretrained resources bundled with the `transkun` PyPI package through `src/core/transkun_transcriber.py`:

```bash
python -m pip install "transkun==2.0.1"
```

Availability checks confirm that `transkun.transcribe`, `pretrained/2.0.pt`, and `pretrained/2.0.conf` exist. If the packaged resources are missing, reinstall:

```bash
python -m pip install --force-reinstall "transkun==2.0.1"
```

### TransKun V2 Aug

`PIANO_TRANSKUN_V2_AUG` is a separate route backed by the official `checkpointTransformerAug.zip` archive. The downloader verifies the archive and loads `checkpointMSimplerAug/checkpoint.pt` with `model.conf`; V2 Aug never silently replaces the default V2 route, and default V2 never silently replaces V2 Aug.

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

Install the backend:

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

Install dependencies:

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

All four piano routes are piano-specialized models. They do not perform full-mix multi-instrument recognition. Choose by target:

| Goal | Recommended Mode | Notes |
|------|------------------|-------|
| Use the project's default TransKun route | `PIANO_TRANSKUN` | Uses the V2 resources bundled with the PyPI package. |
| Compare the official augmented checkpoint explicitly | `PIANO_TRANSKUN_V2_AUG` | Separately downloaded and verified V2 Aug assets; never a silent fallback for default V2. |
| Use another modern piano AMT backend | `PIANO_ARIA_AMT` | Suitable for A/B testing on the same pure-piano inputs. |
| Output needs sustain pedal CC64, especially classical, lyrical, or legato-heavy piano | `PIANO_BYTEDANCE_PEDAL` | Preserves sustain pedal control events. The upstream ByteDance repository is archived, so validate it once in the target runtime. |

These piano results should not be directly compared with `YourMT3+` / `MIROS` multi-instrument outputs: piano backends model 88-key piano performance details, while multi-instrument backends handle instrument recognition and multi-track output for full mixes.

## Models and Public Comparisons

This section separates public benchmark claims from project integration status. The current published entry points expose `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_TRANSKUN_V2_AUG`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL`.

#### Integrated Backend Overview

| Backend / Model | Type | Project Entry | Public Quality Signal | Selection Notes |
|-----------------|------|---------------|-----------------------|-----------------|
| YourMT3+ | Multi-instrument AMT | Selectable directly in `SMART` and as five official checkpoint routes per separated WAV; conversion preserves official-writer notes and only adds required tempo metadata | Official Space default noPS result: Slakh `multi_f = 0.7398`; YourMT3+ paper table: Slakh2100 `Multi F1 = 74.84`, same table `MT3 = 62.0` | Default multi-instrument backend; the project default checkpoint is `YPTF.MoE+Multi (noPS)`, aligned with the official Hugging Face Space default. |
| MuScriptor Large | Multi-instrument AMT | Selectable in `SMART` and per separated WAV, with model-native hard instrument constraints and the official writer | Author `D_Test`: Onset / Frame / Offset / Drums / Multi F1 = **60.4 / 72.4 / 48.6 / 49.6 / 47.8**; YourMT3+ Multi F1 is 21.9 in the same table | Strong public full-mix candidate; author-set scores do not form a universal leaderboard, and weights are non-commercial. |
| MIROS | Multi-instrument AMT | Selectable in `SMART` and per separated WAV | 2025 AMT Challenge F1 **0.5998**, versus YourMT3-YPTF-MoE-M 0.5938 and MT3 0.3932 | Pinned MusicFM backend; the challenge used 76 constrained synthetic clips, so its score is not comparable to MuScriptor `D_Test` or Slakh. |
| TransKun default V2 | Piano-specialized | `PIANO_TRANSKUN` | The V2 / pip checkpoints publish MAESTRO V3 F1 values | Project default TransKun route with package-bundled resources. |
| TransKun V2 Aug | Piano-specialized | `PIANO_TRANSKUN_V2_AUG` | Official augmented checkpoint; this README does not transfer metrics from a different checkpoint | Separate, fixed-asset A/B route; never a fallback for default V2. |
| Aria-AMT | Piano-specialized | `PIANO_ARIA_AMT` | Public checkpoint; this README does not invent a missing same-protocol F1 score | Integrated pure-piano A/B option. |
| ByteDance Pedal | Piano-specialized / pedal-aware | `PIANO_BYTEDANCE_PEDAL` | MAESTRO note onset F1 / pedal onset F1 = 96.72% / 91.86% | Prefer when the output needs sustain pedal CC64; never used as a silent substitute for other piano backends. |
| Leap XE + PolarFormer | Vocal/accompaniment separation | Pre-separation for `VOCAL_SPLIT` | The two public models target different outputs, so no combined benchmark is claimed | Leap XE produces vocals; PolarFormer produces accompaniment; both stems then use the selected transcription backend. |
| BS-RoFormer SW Fixed | Six-stem separation | Pre-separation for `SIX_STEM_SPLIT` | MVSEP 6-stem SDR protocol | `BS-Rofo-SW-Fixed.ckpt` produces six WAV stems; separation SDR is not end-to-end MIDI F1. |

YourMT3+ / MuScriptor / MIROS are multi-instrument backends, TransKun / Aria-AMT / ByteDance Pedal are piano-specialized backends, and Leap XE / PolarFormer / BS-RoFormer SW Fixed are source-separation backends. Their public metrics must not be collapsed into one leaderboard.

#### MuScriptor and Frontier Watchlist (verified 2026-08-08)

| Model / Direction | Public Evidence | Status | Project Decision |
|---|---|---|---|
| MuScriptor Small / Medium | Official 103M / 307M weights; `D_Real`-only Multi F1 38.2 / 39.7, versus Large 40.5 in the same scale ablation | Integrated | Pinned independent selectors for lower VRAM and faster inference. They do not silently replace Large; quality, speed, latency, and memory are validated on the same real audio. |
| Mirelo Studio improved model | Mirelo says it uses more training data and is more accurate | Private service | Watch only. No public weights, revision, license mapping, or comparable score; it cannot be relabeled as `muscriptor-large`. |
| MIROS / MusicFM | 2025 AMT Challenge winner at F1 0.5998 on its own 76-clip protocol | Integrated | Keep as a separate backend and protocol, not as a numeric MuScriptor replacement. |
| Dense polyphony and instrument detection | The challenge paper reports MIROS F-measure dropping from 0.7193 for one instrument to 0.4367 for three and identifies leakage, similar timbres, and polyphonic confusion as persistent failures | Research priority | Prefer future models that publish instrument-aware F1, leakage, polyphony degradation, real jazz/pop coverage, weights, licensing, speed, and VRAM—not just one incompatible note score. |

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

YourMT3+ / MuScriptor / MIROS are multi-instrument backends and should not be directly ranked against the piano-specialized F1 scores above. ByteDance Pedal's `pedal onset F1` is also not equivalent to TransKun's `onset+offset+velocity F1`.

## Default Processing Strategy

The desktop, Space, and Colab interfaces no longer expose a user-adjustable quality preset. YourMT3+ uses official non-overlapping slices, fixed `bsz=8`, per-channel detokenization/merge, `mix_notes`, and its MIDI writer; MIROS preserves the official CLI writer result; MuScriptor strictly uses the official v0.3.0 5-second window, prelude forcing, single generation, events, and MIDI writer. None of the three backends receives project-level note quantization, filtering, or local `NoteEvent` regeneration.

For `SIX_STEM_SPLIT`, `BS-Rofo-SW-Fixed.ckpt` produces six real WAV stems. Each stem keeps an independent route selector and explicit conversion action; no MIDI backend is invoked merely because separation completed.

## Requirements

| Item | Requirement |
|------|-------------|
| Python | 3.11+; the Windows installer prefers 3.11-3.12 |
| PyTorch | Desktop/portable baseline: 2.7.0 with matching `torchaudio==2.7.0` and `torchvision==0.22.0` |
| Git | Required for source installation; installers stop explicitly when Git is unavailable |
| FFmpeg | Required for reliable MP3/M4A/FLAC/OGG handling |
| GPU | One-click source installation and the complete seven-mode runtime require NVIDIA, a driver compatible with CUDA 12.8, and working `nvidia-smi`; there is no automatic CPU/AMD downgrade |
| Disk | A complete cold install retains wheels, models, and download caches; the measured working set was about 32.36 GB, so keep at least 40 GB free before starting |
| OS | Windows 10/11, Linux, WSL2 |

Each platform has its own pinned compatibility envelope. Do not force one platform's NumPy/Torch combination onto another:

| Platform | Python / Torch | NumPy and GPU runtime | Release boundary |
|----------|----------------|-----------------------|------------------|
| Windows / NVIDIA desktop and portable target | Python 3.11-3.12; Torch 2.7.0 / torchaudio 2.7.0 / torchvision 0.22.0 | NumPy 1.26.4; CUDA 12.8 wheels | Source launchers enforce this contract; `release.yml` revalidates the closed third-party inventory, exact model identities, and finished portable smoke tests before publishing |
| Linux / NVIDIA source | Python 3.11+; Torch 2.7.0 / torchaudio 2.7.0 / torchvision 0.22.0 | NumPy 1.26.4; NVIDIA driver compatible with CUDA 12.8; `cu128` only | `install.sh` / `run.sh` verify the exact complete seven-mode runtime; `build.yml` performs source, test, and packaging-contract checks only |
| Linux / AMD/ROCm | No complete seven-mode compatibility runtime | PolarFormer requires ONNX Runtime `CUDAExecutionProvider` | Currently unsupported; the installer stops explicitly instead of silently switching to CPU |
| Hugging Face Space | Python 3.12.12; Torch 2.8.0 / torchaudio 2.8.0 / torchvision 0.23.0 | NumPy `>=2,<2.5`; ZeroGPU | Uses `space/requirements.txt`; do not apply the desktop NumPy 1.26 pin |
| Google Colab | Current Colab Python and preinstalled Torch | Keeps preinstalled Torch; installs only pinned Web/runtime dependencies | Avoids replacing Torch and breaking its CUDA runtime |

On Windows, put the source checkout on a local disk in a plain ASCII path with no spaces:

```text
C:\MusicToMidi
D:\Projects\music-to-midi
```

Paths containing non-ASCII characters, spaces, or parentheses can cause PyTorch DLL loading failures. Mapped drives and UNC/SMB paths have not passed the source virtual-environment identity contract and can make the path recorded by `venv` differ from the launch path; copy the checkout to a local NTFS directory first.

## Quick Start

### MuScriptor authorization (required before the first source install)

MuScriptor Small, Medium, and Large are Hugging Face gated models and cannot be downloaded anonymously. Sign in with one Hugging Face account and accept the terms in the browser for [Small](https://huggingface.co/MuScriptor/muscriptor-small), [Medium](https://huggingface.co/MuScriptor/muscriptor-medium), and [Large](https://huggingface.co/MuScriptor/muscriptor-large). After access is approved, authenticate from the repository's isolated environment:

```powershell
# Windows; if venv does not exist, install only the official HF CLI first
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

Use a personal token with read access to all three gated repositories. Do not place the token in the repository, README, command-line arguments, or logs; unattended environments must use a protected `HF_TOKEN` secret. Before any other large model download, the aggregate downloader performs three metadata-only permission checks and stops immediately if one repository is not accepted or the account is not authenticated. A complete portable release already contains identity-verified weights and does not download them from the Hub at startup; the CC BY-NC 4.0 license and additional terms still apply.

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

You can also double-click `run.bat`. `run.ps1` checks the virtual environment, Beat This `final0`, all five YourMT3+ modes, MuScriptor Large / Medium / Small, BS-RoFormer SW Fixed, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance Pedal, MIROS, SoundFont, and FluidSynth, then calls `install.ps1` if something is missing or invalid.

### Linux / WSL2

```bash
chmod +x run.sh
./run.sh
```

`run.sh` checks the virtual environment, core imports, Beat This `final0`, YourMT3+ source and all five model modes, MuScriptor Large / Medium / Small, BS-RoFormer SW Fixed, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance Pedal, MIROS, SoundFont, and FluidSynth, then calls `install.sh` if something is missing or invalid.

### Direct Source Run

Source runs must use the repository's isolated `venv`. Before importing GUI or model dependencies, the entry point strictly validates the interpreter path, `include-system-site-packages=false`, the MuScriptor package location and version, and all seven pinned source SHA-256 hashes. A bare global `python -m src.main` invocation is rejected.

```powershell
# Windows
.\venv\Scripts\python.exe -m src.main
```

```bash
# Linux / WSL2
./venv/bin/python -m src.main
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

AMD/ROCm cannot currently run the complete seven-mode surface: even though PyTorch publishes ROCm wheels, PolarFormer requires ONNX Runtime `CUDAExecutionProvider`. The installer stops explicitly instead of silently switching to CPU. The complete seven-mode path is currently validated only on NVIDIA CUDA.

`release.yml` produces a CUDA 12.8 GPU portable build only; it does not publish a CPU variant. The current closed inventory contains 28 third-party components: 24 are `VERIFIED`, 4 are `OWNER_ACCEPTED` with named maintainer responsibility and a revocation contact, and 0 are `BLOCKED`. Every release revalidates that inventory, model identities, the SBOM, the packaged FFmpeg build, and the finished application smoke test; any failed requirement stops the release. Push/PR `build.yml` jobs validate source, tests, and packaging contracts but produce no portable artifact. For local source development, CPU-only PyTorch remains a manual choice with slower inference and different dependency compatibility.

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
python -m pip install --no-deps "audio-separator==0.44.1"
python -m pip install --no-deps --force-reinstall "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip"
python -m pip install --no-deps --force-reinstall "muscriptor @ https://github.com/muscriptor/muscriptor/archive/d73147e75e5b9b0c0a79ebe154587db4fd603e0c.zip"
python -m src.utils.source_runtime
```

`requirements.txt` intentionally prevents audio-separator's NumPy 2 metadata and Aria-AMT's older torchaudio constraint from replacing the desktop compatibility stack. Install those packages separately with the pinned `--no-deps` commands above. MuScriptor is also installed from the exact official v0.3.0 commit without dependency resolution, then its environment, package path, version, and source hashes are validated together. Prefer `install.ps1` / `install.sh` when you also need the complete pinned companion dependency set.

### 4. Prepare YourMT3+ Source and Weights

```bash
python download_sota_models.py
```

The repository already includes the controlled, compatibility-patched `YourMT3/amt/src`; do not overwrite it with mutable upstream `master`. `download_sota_models.py` prepares Beat This `final0`, all five official YourMT3+ checkpoints, pinned MIROS source and both weights, MuScriptor Large / Medium / Small, `BS-Rofo-SW-Fixed.ckpt`, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance, MuseScore General SoundFont, and FluidSynth, and strictly verifies the default TransKun 2.0.1 package and bundled V2 resources.

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

On Windows, `~` means `%USERPROFILE%`. The source virtual environment is always the repository's `venv`, desktop output defaults to `MidiOutput\<audio-file-name>` inside the repository, and logs are written to `%USERPROFILE%\.music-to-midi\logs`. Default TransKun V2 resources live inside the `transkun==2.0.1` package in `venv`. Portable builds instead read their packaged `models`, `runtime`, and `tools` directories and do not depend on these source caches.

Default TransKun V2 resources are bundled with `transkun==2.0.1`. If `PIANO_TRANSKUN` reports missing or mismatched resources, run `python -m pip install --force-reinstall "transkun==2.0.1"`. `PIANO_TRANSKUN_V2_AUG` uses its separate cache and requires `python download_transkun_v2_aug_model.py`.

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

The ZeroGPU deployment is a short-clip demo, not a promise that full songs complete end to end. The [Hugging Face ZeroGPU documentation](https://huggingface.co/docs/hub/main/en/spaces-zerogpu) currently lists daily quotas of 2 GPU minutes for anonymous users and 5 minutes for logged-in free accounts. The conservative minimum request already exceeds the anonymous allowance after the platform's `large` GPU multiplier, so conversion currently requires sign-in. The Space estimates each mode/backend/model combination, applies the pinned `spaces==0.51.1` multiplier upper bound, and rejects requests above one 300-second logged-in free-account window before downloading models. The estimate is an admission ceiling, not a guarantee of remaining daily quota or queue capacity; use Colab, the desktop build, or dedicated GPU hardware for long songs.

Under the current formula, the maximum input lengths below are exact admission thresholds rather than measured-runtime promises. They apply to the default `YPTF.MoE+Multi (noPS)` checkpoint and MIROS; other YourMT3 checkpoints use their own factors.

| ZeroGPU route | YourMT3 default noPS | MIROS |
|---------------|---------------------:|------:|
| `SMART` | 2.00 s | 1.00 s |
| `VOCAL_SPLIT` | 0.53 s | 0.27 s |
| `SIX_STEM_SPLIT` | 0.22 s | 0.11 s |
| Any dedicated piano mode | 2.50 s | N/A |

Failed Space requests delete their request directory immediately. Successful outputs remain available for Gradio download and become eligible for expiration cleanup after 24 hours by default; they are removed on a later cleanup pass or normal Space process exit. Colab also removes failed request directories immediately and retains successful outputs until that runtime exits; its Gradio cache makes files eligible for cleanup after 24 hours.

## Portable Build

Windows directory-style portable build:

> Current release status: the portable gate requires all 28 components to be either `VERIFIED` or explicitly `OWNER_ACCEPTED`. It currently records 24 verified and four owner-accepted components in `THIRD_PARTY_NOTICES.md`. Owner acceptance is revocable and does not represent an upstream license grant; running the local command below does not create redistribution rights.

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable.ps1
```

Specify Python or FFmpeg:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable.ps1 `
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

Distribute the entire folder:

```text
dist/MusicToMidi/
```

Do not distribute only the single executable.

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

Run these commands only after activating the repository's `venv`.

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

If the controlled `YourMT3/amt/src` tree is missing, restore it from the current project revision. Do not overwrite it with mutable upstream `master`, which cannot satisfy three-interface source parity or the portable source-manifest check.

### Vocal Separation Is Unavailable

Confirm dependency and model:

Windows / Linux NVIDIA CUDA:

```bash
python -m pip install --no-deps "audio-separator==0.44.1" "onnxruntime-gpu==1.23.2"
python download_vocal_model.py
python download_accompaniment_model.py
```

On macOS or an explicitly CPU-only environment, replace `onnxruntime-gpu==1.23.2` with `onnxruntime==1.23.2`.

### Six-Stem Separation Is Unavailable

Confirm `audio-separator==0.44.1` is installed and download the BS-RoFormer SW Fixed resources:

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

If the error lists missing Python modules, install the dependencies required by the upstream MIROS repository.

## License

This project uses the MIT License. Third-party models, datasets, and upstream repositories remain governed by their own licenses and terms; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for adapted-code notices and full license text.
