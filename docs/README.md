# Music to MIDI Converter (AI Audio to MIDI)

<p align="center">
  <a href="../README.md">中文</a> | English
</p>

Music to MIDI is a local-first AI audio-to-MIDI converter for music producers, transcription hobbyists, piano learners, sampling workflows, and automatic music transcription (AMT) experiments. Drop in an `MP3`, `WAV`, `FLAC`, `OGG`, or `M4A` file, then generate editable MIDI from the PyQt6 desktop app, the Gradio Web interface, or the Google Colab notebook.

The current product surface syncs seven processing modes: full-mix multi-instrument transcription, vocal/accompaniment split transcription, six-stem split transcription, and four dedicated piano routes through TransKun default V2, TransKun V2 Aug, Aria-AMT, or ByteDance Pedal. The project is more than a one-note melody extractor: it brings multi-instrument AI music transcription, stem separation, piano-to-MIDI conversion, BPM/tempo metadata, and MIDI merging into one workflow.

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

- **Full-mix transcription**: `SMART` mode sends the whole song to YourMT3+, MIROS, or MuScriptor Large and exports MIDI with notes, drums, and instrument tracks.
- **Vocal/accompaniment separation and per-track transcription**: `VOCAL_SPLIT` uses BS-RoFormer Leap XE 90-band for vocals and BS PolarFormer for accompaniment, and first delivers two real WAV tracks. Each track can then select one of 11 explicit MIDI routes in the track workbench.
- **Six-stem separation and per-track transcription**: `SIX_STEM_SPLIT` uses `BS-Rofo-SW-Fixed.ckpt` to deliver six real `bass / drums / guitar / piano / vocals / other` WAV tracks. Separation does not silently transcribe or merge MIDI.
- **Dedicated piano transcription**: `PIANO_TRANSKUN`, `PIANO_TRANSKUN_V2_AUG`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL` target pure piano audio through TransKun default V2, the official V2 Aug checkpoint, Aria-AMT, or ByteDance's pedal-aware model.
- **Default backend semantics**: the multi-instrument default remains the official YourMT3+ `YPTF.MoE+Multi (noPS)` checkpoint. `SMART` can explicitly select YourMT3+, MIROS, or MuScriptor Large, and every separated WAV can select the same three multi-instrument families independently.
- **Real MuScriptor constraints**: an empty instrument list enables model detection. A non-empty list is passed to the official `instruments` plus `prelude_forcing` decoding path, masks every unselected instrument token during generation, and is validated again against streamed events and final MIDI.
- **Explicit multi-instrument routes**: `SMART` and every separated WAV can select YourMT3+, MIROS, or MuScriptor Large. The per-track menu also exposes four piano-specialized routes, for 11 routes in total.
- **Official transcription output**: YourMT3+ and MIROS preserve their official writer results; MuScriptor uses its official events and MIDI writer plus strict selected-instrument validation. The project does not add quantization, de-duplication, short-note filtering, velocity smoothing, polyphony limiting, or local `NoteEvent` regeneration.
- **Common audio formats**: `MP3`, `WAV`, `FLAC`, `OGG`, and `M4A` are accepted. Non-WAV input must be converted to 44.1 kHz PCM WAV through FFmpeg; FFmpeg failures stop processing and show the stderr root cause.
- **Consistent mode set**: desktop, Space, and Colab expose the same seven processing modes.

## Interface Matrix

| Interface | Modes | Backend Selection | Best For |
|-----------|-------|-------------------|----------|
| PyQt6 desktop | `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_TRANSKUN_V2_AUG`, `PIANO_ARIA_AMT`, `PIANO_BYTEDANCE_PEDAL` | `SMART` selects YourMT3+ / MIROS / MuScriptor; separated WAV tracks expose 11 routes; piano modes use their dedicated backend | Local GPU use, persistent output folders, and dedicated piano transcription |
| Gradio Space | Same seven modes as desktop | MuScriptor instrument search/multi-select, hard decoding constraint, and real MIDI workbench are synchronized | Browser-based use or hosted demos |
| Google Colab | Same seven modes as desktop | Same MuScriptor constraint and linked WAV/MIDI result workbench as Space | Temporary Colab GPU sessions |

## Entry And Dependency Sync Status

This repository keeps project workflows separate from official YourMT3+ checkpoint modes:

- `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_TRANSKUN_V2_AUG`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL` are this project's seven processing workflows.
- `YMT3+`, `YPTF+Single (noPS)`, `YPTF+Multi (PS)`, `YPTF.MoE+Multi (noPS)`, and `YPTF.MoE+Multi (PS)` are the five checkpoint / architecture modes exposed by the official YourMT3 demo.
- Desktop, Gradio Space, and Colab expose the same seven workflows. `SMART` selects YourMT3+, MIROS, or MuScriptor Large; both separation workflows first deliver WAV tracks, and each track then exposes 11 explicit MIDI routes.

Current synchronization coverage:

| Location | Synced Content | Notes |
|----------|----------------|-------|
| `download_sota_models.py` | Prepares all five official YourMT3+ checkpoints; pinned MIROS source plus both weights; `BS-Rofo-SW-Fixed.ckpt`; Leap XE; PolarFormer; TransKun V2 Aug; Aria-AMT; and ByteDance, while strictly validating the default TransKun 2.0.1 package and bundled V2 resources | Fixed-source resources are validated by known size/SHA256 or their explicit source/runtime identity; any required-resource failure stops the command. |
| `run.ps1` / `run.sh` | Checks all official YourMT3+ modes, BS-RoFormer SW Fixed, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance Pedal, MIROS, and separator availability before launch | Missing or invalid required resources are reported explicitly. |
| `install.ps1` / `install.sh` | Installs PyTorch 2.7, NumPy 1.26, audio-separator 0.44.1 runtime pins, and required models | `audio-separator` is installed with `--no-deps` to avoid pulling NumPy 2 into the current PyTorch / desktop stack. |
| `.github/workflows/build.yml` | Push/PR jobs run Linux and Windows source, test, and packaging-contract checks only | They produce no portable artifact and never use empty directories or fake models to bypass mandatory bundle validation. |
| `.github/workflows/release.yml` | The target complete portable-build pipeline; it is designed to download and strictly verify every YourMT3+, separator, MIROS, TransKun, Aria-AMT, and ByteDance asset | A closed third-party-license gate currently stops the workflow before any build; no portable artifact is produced or published until every component is cleared. The target GPU runtime is PyTorch 2.7 + CUDA 12.8. |
| `colab_notebook.ipynb` | Keeps Colab's preinstalled Torch, installs pinned Web/runtime dependencies, and synchronizes all seven modes | `SMART` and the per-track workbench expose YourMT3+, MIROS, and MuScriptor Large; the per-track menu also includes four piano routes. |

## Processing Modes

| Mode | Internal Pipeline | Main Output | Notes |
|------|-------------------|-------------|-------|
| `SMART` | Audio -> selected YourMT3+ / MIROS / MuScriptor Large -> MIDI | `<song>.mid` | No source separation. A non-empty MuScriptor instrument selection is a real decoding constraint. |
| `VOCAL_SPLIT` | Audio -> Leap XE vocals + PolarFormer accompaniment -> two WAV tracks -> explicit per-track MIDI | `<song>_vocals.wav`, `<song>_accompaniment.wav`; per-track MIDI on request | Separation does not auto-transcribe. Each WAV independently selects one of 11 routes. |
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

YourMT3+ is the default multi-instrument backend. `download_sota_models.py` prepares all five official YourMT3+ checkpoints, pinned MIROS source and both weights, `BS-Rofo-SW-Fixed.ckpt`, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, and ByteDance, and strictly validates the default TransKun 2.0.1 package and bundled V2 resources. YourMT3 inference imports the repository-controlled `YourMT3/amt/src` tree through `src/core/yourmt3_transcriber.py`.

The source tree must include:

```text
YourMT3/amt/src/model/ymt3.py
YourMT3/amt/src/utils/task_manager.py
YourMT3/amt/src/config/config.py
```

The complete project checkout includes a compatibility-patched `YourMT3/amt/src` tree protected by a fixed manifest. If it is missing, restore it from the current project revision; do not overwrite it with mutable upstream `master`. A separate upstream clone is suitable for experiments but does not satisfy this project's three-interface source-parity or portable-build identity contract.

Download model weights:

```bash
python download_sota_models.py
```

Default model search roots include:

```text
~/.cache/music_ai_models/yourmt3_all
runtime/models/yourmt3_all
models/yourmt3_all
```

### MuScriptor Large

The project pins public `muscriptor` commit `302343e8992bdfc619f77f1988168374ed5d675d` (package `0.2.2a1`) and gated [`MuScriptor/muscriptor-large`](https://huggingface.co/MuScriptor/muscriptor-large) revision `8809fdfbed2affa7ade94a7059e746e3880720e7`. The weight file is 5,465,642,136 bytes. It is licensed under CC BY-NC 4.0 with additional lawful-use terms, so the user must accept the Hugging Face conditions and authenticate before download:

```bash
hf auth login
python download_muscriptor_model.py
```

Large is a decoder-only Transformer with roughly 1.3B parameters (the current code README rounds it to 1.4B), 48 layers, and hidden dimension 1536. It consumes 5-second 16 kHz mono chunks and emits MT3-style onset, offset, pitch, and 36-group instrument events. Training combines about 1.45 million MIDI files for synthetic pretraining, 170,000 real recordings / about 11,000 hours for fine-tuning, and 300 curated tracks for RL post-training.

The official model card reports the following scores on the authors' 372-track real multi-instrument `D_Test` set, using the full training pipeline and CFG=2:

| Model | Onset F1 | Frame F1 | Offset F1 | Drums F1 | Multi F1 |
|---|---:|---:|---:|---:|---:|
| YourMT3+ `YPTF.MoE+Multi (noPS)` | 32.5 | 45.5 | 17.8 | 41.4 | 21.9 |
| MuScriptor Large | **60.4** | **72.4** | **48.6** | **49.6** | **47.8** |

This is strong evidence that MuScriptor is a leading public full-mix candidate, but not proof of universal SOTA: `D_Test` is an author-held set without a public download path, and MuScriptor wins Multi F1 on six of the paper's eight public cross-domain datasets while losing on RWC-C and RWC-R. It also does not emit velocity, uses a fixed 36-group instrument taxonomy, and the weights are non-commercial.

The release chronology is also explicit: the [Hugging Face API](https://huggingface.co/api/models/MuScriptor/muscriptor-large) records repository creation on 2026-06-30; the [paper](https://arxiv.org/abs/2607.08168) and [Mirelo article](https://mirelo.ai/blog/turning-audio-to-midi) were published on 2026-07-09; the GitHub release and current public weight revision followed on 2026-07-10. Repository timestamps are publishing metadata, not model-training dates.

Mirelo separately says that Studio hosts a more accurate version trained on more data. No public checkpoint, revision, parameter count, or comparable score has been published for that service model, so it is not the same verifiable artifact as `muscriptor-large` and is not integrated here. Full ablations, all eight public-dataset comparisons, scale results, conditioning gains, and the frontier watchlist are documented in [the MuScriptor research note](muscriptor-model.md).

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
- [BS PolarFormer](https://huggingface.co/bgkb/bs_polarformer) uses `bs_polarformer.onnx` with `model_bs_polarformer_float16.yaml` to produce accompaniment.

The canonical separated outputs are `vocals` and `accompaniment`. Each enters the track workbench with 11 explicit routes: five YourMT3+ checkpoints, MIROS, MuScriptor Large, and four piano-specialized backends. The two separation calls are not substitutes for one another, and a failure in either route is surfaced instead of synthesizing a missing stem.

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

#### MuScriptor and Frontier Watchlist (verified 2026-07-19)

| Model / Direction | Public Evidence | Status | Project Decision |
|---|---|---|---|
| MuScriptor Small / Medium | Official 103M / 307M weights; `D_Real`-only Multi F1 38.2 / 39.7, versus Large 40.5 in the same scale ablation | Public, not integrated | Closest deployable future candidates for lower VRAM or CPU. Require same-audio local quality, speed, latency, and memory tests before adding a selector. |
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

The desktop, Space, and Colab interfaces no longer expose a user-adjustable quality preset. YourMT3+ uses official non-overlapping slices, fixed `bsz=8`, per-channel detokenization/merge, `mix_notes`, and its MIDI writer; MIROS preserves the official CLI writer result; MuScriptor uses its official segmented generation, events, and MIDI writer. The project does not add overlap de-duplication, sparse-program filtering, or local MIDI regeneration.

For `SIX_STEM_SPLIT`, `BS-Rofo-SW-Fixed.ckpt` produces six real WAV stems. Each stem keeps an independent route selector and explicit conversion action; no MIDI backend is invoked merely because separation completed.

## Requirements

| Item | Requirement |
|------|-------------|
| Python | 3.11+; the Windows installer prefers 3.11-3.12 |
| PyTorch | Desktop/portable baseline: 2.7.0 with matching `torchaudio==2.7.0` and `torchvision==0.22.0` |
| FFmpeg | Required for reliable MP3/M4A/FLAC/OGG handling |
| GPU | Some source transcription routes can run on CPU; the complete seven-mode experience and complete portable artifact require a compatible GPU runtime |
| OS | Windows 10/11, Linux, WSL2 |

Each platform has its own pinned compatibility envelope. Do not force one platform's NumPy/Torch combination onto another:

| Platform | Python / Torch | NumPy and GPU runtime | Release boundary |
|----------|----------------|-----------------------|------------------|
| Windows / NVIDIA desktop and portable target | Python 3.11-3.12; Torch 2.7.0 / torchaudio 2.7.0 / torchvision 0.22.0 | NumPy 1.26.4; CUDA 12.8 wheels | Source launchers enforce this contract; `release.yml` currently produces no artifact because third-party redistribution review is incomplete |
| Linux / NVIDIA source | Python 3.11+; Torch 2.7.0 / torchaudio 2.7.0 / torchvision 0.22.0 | NumPy 1.26.4; NVIDIA driver compatible with CUDA 12.8; `cu128` only | `install.sh` / `run.sh` verify the exact complete seven-mode runtime; `build.yml` performs source, test, and packaging-contract checks only |
| Linux / AMD/ROCm | No complete seven-mode compatibility runtime | PolarFormer requires ONNX Runtime `CUDAExecutionProvider` | Currently unsupported; the installer stops explicitly instead of silently switching to CPU |
| Hugging Face Space | Python 3.12.12; Torch 2.8.0 / torchaudio 2.8.0 / torchvision 0.23.0 | NumPy `>=2,<2.5`; ZeroGPU | Uses `space/requirements.txt`; do not apply the desktop NumPy 1.26 pin |
| Google Colab | Current Colab Python and preinstalled Torch | Keeps preinstalled Torch; installs only pinned Web/runtime dependencies | Avoids replacing Torch and breaking its CUDA runtime |

On Windows, use a plain ASCII path with no spaces where possible:

```text
C:\MusicToMidi
D:\Projects\music-to-midi
```

Paths containing non-ASCII characters, spaces, or parentheses can cause PyTorch DLL loading failures.

## Quick Start

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

You can also double-click `run.bat`. `run.ps1` checks the virtual environment, all five YourMT3+ modes, BS-RoFormer SW Fixed, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance Pedal, and MIROS, then calls `install.ps1` if something is missing or invalid.

### Linux / WSL2

```bash
chmod +x run.sh
./run.sh
```

`run.sh` checks the virtual environment, core imports, YourMT3+ source and all five model modes, BS-RoFormer SW Fixed, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, ByteDance Pedal, and MIROS, then calls `install.sh` if something is missing or invalid.

### Direct Source Run

```bash
python -m src.main
```

## Manual Setup

### 1. Create a Virtual Environment

Windows:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch

CUDA 12.8 (the supported complete seven-mode runtime, checked strictly by the launchers):

```bash
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

`cu118` / CUDA 11 is outside the current one-click launcher and complete seven-mode acceptance contract; launchers do not silently treat it as an aligned runtime.

AMD/ROCm cannot currently run the complete seven-mode surface: even though PyTorch publishes ROCm wheels, PolarFormer requires ONNX Runtime `CUDAExecutionProvider`. The installer stops explicitly instead of silently switching to CPU. The complete seven-mode path is currently validated only on NVIDIA CUDA.

The target `release.yml` artifact is CUDA 12.8 GPU-only; no CPU variant is planned. However, 13 of the 22 components in the closed third-party inventory are currently `BLOCKED`, so the workflow fails before building and there is no current portable artifact that can be described as redistribution-cleared. Push/PR `build.yml` jobs validate source, tests, and packaging contracts but produce no portable artifact. For local source development, CPU-only PyTorch remains a manual choice with slower inference and different dependency compatibility.

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
python -m pip install --no-deps "audio-separator==0.44.1"
python -m pip install --no-deps --force-reinstall "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip"
```

`requirements.txt` intentionally prevents audio-separator's NumPy 2 metadata and Aria-AMT's older torchaudio constraint from replacing the desktop compatibility stack. Install those two packages separately with the pinned `--no-deps` commands above; prefer `install.ps1` / `install.sh` when you also need the complete pinned companion dependency set.

### 4. Prepare YourMT3+ Source and Weights

```bash
python download_sota_models.py
```

The repository already includes the controlled, compatibility-patched `YourMT3/amt/src`; do not overwrite it with mutable upstream `master`. `download_sota_models.py` prepares all five official YourMT3+ checkpoints, pinned MIROS source and both weights, `BS-Rofo-SW-Fixed.ckpt`, Leap XE, PolarFormer, TransKun V2 Aug, Aria-AMT, and ByteDance, and strictly verifies the default TransKun 2.0.1 package and bundled V2 resources.

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
~/.music-to-midi/models/audio-separator
~/.cache/music_ai_models/transkun_v2_aug
~/.cache/music_ai_models/aria_amt
~/.cache/music_ai_models/bytedance_piano
external/ai4m-miros
```

Default TransKun V2 resources are bundled with `transkun==2.0.1`. If `PIANO_TRANSKUN` reports missing or mismatched resources, run `python -m pip install --force-reinstall "transkun==2.0.1"`. `PIANO_TRANSKUN_V2_AUG` uses its separate cache and requires `python download_transkun_v2_aug_model.py`.

### 6. Launch

```bash
python -m src.main
```

## Google Colab

Notebook entry:

```text
colab_notebook.ipynb
```

Steps:

1. Open the notebook.
2. Select a GPU runtime.
3. Run the cells in order.
4. The final cell launches Gradio and prints a public URL.

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

The ZeroGPU deployment is a short-clip demo, not a promise that full songs complete end to end. The [Hugging Face ZeroGPU documentation](https://huggingface.co/docs/hub/main/en/spaces-zerogpu) currently lists daily quotas of 2 GPU minutes for anonymous users and 5 minutes for logged-in free accounts. The conservative minimum request already exceeds the anonymous allowance after the platform's `large` GPU multiplier, so conversion currently requires sign-in. The Space estimates each mode/backend/model combination, applies the pinned `spaces==0.51.0` multiplier upper bound, and rejects requests above one 300-second logged-in free-account window before downloading models. The estimate is an admission ceiling, not a guarantee of remaining daily quota or queue capacity; use Colab, the desktop build, or dedicated GPU hardware for long songs.

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

> Current release status: the future portable-build path is retained and tested, but the official release gate remains closed by unresolved model, patched-source, and runtime redistribution terms listed in `THIRD_PARTY_NOTICES.md`. Running the local command below does not grant redistribution permission.

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
download_sota_models.py      # All five YourMT3 + MIROS + separation + four piano model contracts
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
