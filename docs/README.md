# Music to MIDI Converter (AI Audio to MIDI)

<p align="center">
  <a href="../README.md">中文</a> | English
</p>

Music to MIDI is a local-first AI audio-to-MIDI converter for music producers, transcription hobbyists, piano learners, sampling workflows, and automatic music transcription (AMT) experiments. Drop in an `MP3`, `WAV`, `FLAC`, `OGG`, or `M4A` file, then generate editable MIDI from the PyQt6 desktop app, the Gradio Web interface, or the Google Colab notebook.

The current product surface syncs six processing modes: full-mix multi-instrument transcription, vocal/accompaniment split transcription, six-stem split transcription, and dedicated Transkun / Aria-AMT / ByteDance Pedal piano transcription. The project is more than a one-note melody extractor: it brings multi-instrument AI music transcription, stem separation, piano-to-MIDI conversion, BPM detection for split-mode MIDI extensions, and MIDI post-processing into one workflow.

## Screenshots

| Windows | Linux |
|---------|-------|
| ![Windows demo](../resources/icons/Windows演示.png) | ![Linux demo](../resources/icons/Linux演示.png) |

## Use Cases

Use it when you want to turn a vocal line, piano recording, full mix, or separated stem into MIDI you can edit in a DAW. It is designed for users who want more control than a simple upload-and-download converter, while still keeping the common audio-to-MIDI path approachable.

## Current Capabilities

- **Full-mix transcription**: `SMART` mode sends the whole song to the selected multi-instrument backend and exports MIDI with notes, drums, and GM instrument tracks.
- **Vocal/accompaniment split transcription**: `VOCAL_SPLIT` separates vocals and accompaniment, transcribes both, and can optionally export one merged MIDI for a fuller arrangement sketch.
- **Six-stem split transcription**: `SIX_STEM_SPLIT` separates `bass / drums / guitar / piano / vocals / other`, then exports stem MIDI files and one merged MIDI for DAW cleanup or re-arrangement.
- **Dedicated piano transcription**: `PIANO_TRANSKUN`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL` target pure piano audio through Transkun, Aria-AMT, and ByteDance's pedal-aware piano model.
- **Default backend semantics**: the multi-instrument default is the official YourMT3+ `YPTF.MoE+Multi (noPS)` checkpoint; the desktop app can switch to MIROS, while dedicated piano modes use Transkun, Aria-AMT, or ByteDance Pedal by entry.
- **Optional MIROS backend**: the desktop app can route transcription through a local `ai4m-miros` checkout as an experimental backend.
- **Beat and post-processing**: `SMART` mode keeps the official YourMT3+ / MIROS MIDI output. Split-mode MIDI extensions run BPM detection before local MIDI generation; if beat detection fails, processing stops instead of writing a fake default tempo. Quantization, duplicate removal, velocity smoothing, and polyphony limiting are available as post-processing.
- **Common audio formats**: `MP3`, `WAV`, `FLAC`, `OGG`, and `M4A` are accepted. Non-WAV input must be converted to 44.1 kHz PCM WAV through FFmpeg; FFmpeg failures stop processing and show the stderr root cause.
- **Consistent mode set**: desktop, Space, and Colab expose the same six processing modes.

## Interface Matrix

| Interface | Modes | Backend Selection | Best For |
|-----------|-------|-------------------|----------|
| PyQt6 desktop | `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_ARIA_AMT`, `PIANO_BYTEDANCE_PEDAL` | Multi-instrument default is YourMT3+ noPS, with optional MIROS; piano modes use Transkun / Aria-AMT / ByteDance Pedal by entry | Local GPU use, persistent output folders, and dedicated piano transcription |
| Gradio Space | Same six modes as desktop | Fixed processing routes; multi-instrument default is YourMT3+ noPS; piano modes use their corresponding backend | Browser-based use or hosted demos |
| Google Colab | Same six modes as desktop | SMART can select an official YourMT3+ checkpoint; split-mode MIDI extension uses default noPS; piano modes use their corresponding backend | Temporary Colab GPU sessions |

## Entry And Dependency Sync Status

This repository keeps project workflows separate from official YourMT3+ checkpoint modes:

- `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL` are this project's six processing workflows.
- `YMT3+`, `YPTF+Single (noPS)`, `YPTF+Multi (PS)`, `YPTF.MoE+Multi (noPS)`, and `YPTF.MoE+Multi (PS)` are the five checkpoint / architecture modes exposed by the official YourMT3 demo.
- Desktop, Gradio Space, and Colab expose the same six workflows; workflows that need multi-instrument transcription then choose one of the official YourMT3+ checkpoint modes.

Current synchronization coverage:

| Location | Synced Content | Notes |
|----------|----------------|-------|
| `download_sota_models.py` | Downloads all five official YourMT3+ checkpoint modes and prepares BS-RoFormer SW six-stem assets plus RoFormer vocal_rvc/karaoke vocal ensembles | YourMT3 uses `OFFICIAL_YOURMT3_MODEL_KEYS`; the six-stem checkpoint is validated by fixed size and SHA256; vocal split uses audio-separator 0.44.1 ensemble presets. |
| `run.ps1` / `run.sh` | Checks all official YourMT3+ modes, BS-RoFormer SW six-stem assets, RoFormer vocal ensembles, Aria-AMT, ByteDance Pedal, MIROS, and separator availability before launch | Missing or invalid required resources are reported explicitly. |
| `install.ps1` / `install.sh` | Installs PyTorch 2.7, NumPy 1.26, audio-separator 0.44.1 runtime pins, and required models | `audio-separator` is installed with `--no-deps` to avoid pulling NumPy 2 into the current PyTorch / desktop stack. |
| `.github/workflows/build.yml` | CI build and test jobs install the same pinned runtime dependencies | Test failures are not masked with `|| true`. |
| `.github/workflows/release.yml` | Release packages download and bundle all official YourMT3+ modes, BS-RoFormer SW, RoFormer vocal ensembles, Aria-AMT, ByteDance Pedal, and MIROS | GPU builds use PyTorch 2.7 + CUDA 12.8 wheels. |
| `colab_notebook.ipynb` | Keeps Colab's preinstalled Torch, installs pinned Web/runtime dependencies, and downloads all official YourMT3+ modes | The Colab UI exposes the YourMT3 model selector only for SMART; split modes use the default noPS backend for MIDI extension. |

## Processing Modes

| Mode | Internal Pipeline | Main Output | Notes |
|------|-------------------|-------------|-------|
| `SMART` | Audio -> official YourMT3+ / MIROS MIDI output | `<song>.mid` | No source separation. Suitable for most full mixes, instrumentals, and short multi-instrument clips. |
| `VOCAL_SPLIT` | Audio -> RoFormer `vocal_rvc` vocal/accompaniment split -> RoFormer `karaoke` lead/backing split -> vocal/accompaniment MIDI transcription -> MIDI generation | `<song>_accompaniment.mid`, `<song>_vocal.mid`, optional `<song>_vocal_accompaniment_merged.mid` | The separation stage also writes `vocals_with_harmony`, `original_vocals`, `backing_vocals`, and `accompaniment_with_harmony` WAV files; MIDI transcription still uses the selected multi-instrument backend. |
| `SIX_STEM_SPLIT` | Audio -> BS-RoFormer SW six-stem WAV separation -> one full-mix multi-instrument transcription -> route notes to stem MIDI by GM family -> stem MIDI merge | `<song>_<stem>.mid`, `<song>_all_stems_merged.mid` | The six WAV stems come from the separator. Stem MIDI is not produced by running AMT separately on each stem; it is routed from the full-mix transcription. The piano stem can prefer Aria-AMT when that backend and checkpoint are available. |
| `PIANO_TRANSKUN` | Audio -> Transkun piano model -> MIDI | `<song>_piano_transkun.mid` | Best for pure piano audio; quality presets do not change Transkun checkpoint inference. |
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
song_piano_aria.mid
song_piano_bytedance_pedal.mid
song_vocals_with_harmony.wav
song_original_vocals.wav
song_backing_vocals.wav
song_accompaniment.wav
song_accompaniment_with_harmony.wav
song_bass.wav
song_drums.wav
song_guitar.wav
song_piano.wav
song_vocals.wav
song_other.wav
```

The exact files depend on the selected mode, whether merged MIDI is enabled, and what the separator returns. Vocal split creates `_vocal_rvc/` and `_karaoke/` subfolders for intermediate separation outputs.

## Backends

### YourMT3+

YourMT3+ is the default multi-instrument backend. `download_sota_models.py` downloads all official YourMT3+ checkpoint modes and prepares the BS-RoFormer SW six-stem assets plus RoFormer `vocal_rvc` / `karaoke` vocal ensembles; YourMT3 inference imports the local `YourMT3/amt/src` source tree through `src/core/yourmt3_transcriber.py`.

The source tree must include:

```text
YourMT3/amt/src/model/ymt3.py
YourMT3/amt/src/utils/task_manager.py
YourMT3/amt/src/config/config.py
```

If your checkout does not include `YourMT3/amt/src`, place the YourMT3 source at the repository root:

```bash
git clone https://github.com/mimbres/YourMT3.git
```

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

### MIROS

MIROS is an optional experimental backend in the desktop app. It is not integrated as a PyPI package; the wrapper expects a local upstream checkout, runs its entrypoint to produce temporary MIDI, then converts that MIDI into the app's internal note format.

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

The downloader clones the upstream `amt-os/ai4m-miros` checkout. `pretrained_msd.pt` is fetched from the official Hugging Face `minzwon/MusicFM` repository, while `last.ckpt` still follows the official Google Drive file ID used by upstream `main.py`. GitHub Actions release packaging does not depend on the live Google Drive quota: it streams the already packaged and verified `external/ai4m-miros` directory from this repository's `v1.0.16` Linux portable release assets. If those portable assets are missing, extraction fails, or the checkpoint container is incomplete, the release job fails explicitly instead of using an unknown source or silently skipping the model.

### Transkun

Transkun is a dedicated piano transcription backend for pure or piano-forward audio. The project calls the pretrained resources bundled with the `transkun` PyPI package through `src/core/transkun_transcriber.py`:

```bash
pip install "transkun>=2.0.1"
```

Availability checks confirm that `transkun.transcribe`, `pretrained/2.0.pt`, and `pretrained/2.0.conf` exist. If the packaged resources are missing, reinstall:

```bash
python -m pip install --force-reinstall transkun
```

### Aria-AMT

Aria-AMT is another dedicated piano backend. The upstream README documents the `aria-amt transcribe` CLI; this project's wrapper currently calls `amt.run transcribe` through `src/core/aria_amt_transcriber.py`. The default checkpoint is:

```text
piano-medium-double-1.0.safetensors
```

Install the backend:

```bash
python -m pip install git+https://github.com/EleutherAI/aria-amt.git
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
python -m pip install "piano-transcription-inference>=0.0.6,<0.1" "torchlibrosa>=0.1.0,<0.2"
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

All three piano backends are piano-specialized models. They do not perform full-mix multi-instrument recognition. Choose by target:

| Goal | Recommended Mode | Notes |
|------|------------------|-------|
| General pure-piano note transcription with the current default piano route | `PIANO_ARIA_AMT` | Modern piano AMT backend and a good default candidate for normal pure-piano input. |
| Mature, lightweight piano backend with bundled weights | `PIANO_TRANSKUN` | Clear package and checkpoint boundaries, good for quick local validation. |
| Output needs sustain pedal CC64, especially classical, lyrical, or legato-heavy piano | `PIANO_BYTEDANCE_PEDAL` | Preserves sustain pedal control events. The upstream ByteDance repository is archived, so validate it once in the target runtime. |

These three results should not be directly compared with `YourMT3+` / `MIROS` multi-instrument outputs: piano backends model 88-key piano performance details, while multi-instrument backends handle instrument recognition and multi-track output for full mixes.

## Models and Public Comparisons

This section separates public benchmark claims from project integration status. The current published entry points expose `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL`.

#### Integrated Backend Overview

| Backend / Model | Type | Project Entry | Public Quality Signal | Selection Notes |
|-----------------|------|---------------|-----------------------|-----------------|
| YourMT3+ | Multi-instrument AMT | `SMART` keeps the official MIDI output directly; `VOCAL_SPLIT` and `SIX_STEM_SPLIT` MIDI extensions use multi-instrument inference and project-side MIDI generation/routing | Official Space default noPS result: Slakh `multi_f = 0.7398`; YourMT3+ paper table: Slakh2100 `Multi F1 = 74.84`, same table `MT3 = 62.0` | Default full-mix route; the project default checkpoint is `YPTF.MoE+Multi (noPS)`, aligned with the official Hugging Face Space default. |
| MIROS | Multi-instrument AMT | Optional desktop multi-instrument backend | Upstream repository describes it as a 2025 AI4Musician winning model | Experimental local backend for A/B testing against YourMT3+ on the same task. |
| Transkun | Piano-specialized | `PIANO_TRANSKUN` | Transkun V2 and pip checkpoints publish MAESTRO V3 F1 values | Mature and lightweight; the current pip checkpoint is documented as not using pedal note extension. |
| Aria-AMT | Piano-specialized | `PIANO_ARIA_AMT` | Public checkpoint; this README does not invent a missing same-protocol F1 score | Current default piano candidate for normal pure-piano transcription. |
| ByteDance Pedal | Piano-specialized / pedal-aware | `PIANO_BYTEDANCE_PEDAL` | MAESTRO note onset F1 / pedal onset F1 = 96.72% / 91.86% | Prefer when the output needs sustain pedal CC64; never used as a silent substitute for other piano backends. |
| RoFormer vocal_rvc / karaoke ensemble | Vocal/accompaniment plus lead/backing-vocal separation | Pre-separation for `VOCAL_SPLIT` | audio-separator 0.44.1 preset registry | Separation only; final MIDI quality still depends on the transcription backend. |
| BS-RoFormer SW | Six-stem separation | Pre-separation for `SIX_STEM_SPLIT` | MVSEP 6-stem SDR protocol | Separation SDR is not end-to-end MIDI F1. |

YourMT3+ / MIROS are multi-instrument backends, Transkun / Aria-AMT / ByteDance Pedal are piano-specialized backends, and RoFormer separation models are source-separation backends. Their public metrics must not be collapsed into one leaderboard.

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
- Older checkpoints that do not store `TOKENIZER.max_shift_steps` are loaded with the official MT3 tokenizer default of `206`.
- Older T5 checkpoints that do not store `ff_layer_type` are loaded with the standard T5 feed-forward layer type `t5_gmlp`.

Known differences from the official demo:

- The official demo runs one selected YourMT3 checkpoint; this project also adds separation workflows, dedicated piano models, MIDI post-processing, and track-layout control.
- The official GPU Space usually runs 16-bit inference; this project defaults to full precision for better stability across Windows / CUDA environments.
- To avoid long-audio stalls with oversized batches, this project caps YourMT3+ batch size conservatively: older official modes default to 1, MoE modes default to 2. Set `MUSIC_TO_MIDI_YOURMT3_BATCH_SIZE` to override this explicitly for experiments.

#### Piano Model Quality Comparison

| Model | Current Project Entry | Same-Type Quality Protocol | Public Result | How To Read It |
|-------|-----------------------|----------------------------|---------------|----------------|
| Transkun V2 | Research checkpoint, not the current pip default entry | MAESTRO V3 `note onset F1 / onset+offset F1 / onset+offset+velocity F1` | **0.9832 / 0.9349 / 0.9296** | Strong public piano AMT reference. |
| Transkun pip checkpoint (No Ext) | `PIANO_TRANSKUN` | MAESTRO V3 No Ext, same three metrics | **0.9833 / 0.8149 / 0.8109** | Lightweight and bundled, but upstream documents it as `without pedal extension of notes`. |
| Aria-AMT | `PIANO_ARIA_AMT` | Public checkpoint, but no fully matching published Transkun-style benchmark table | No unified F1 written here | Use as the current default piano candidate; compare with local A/B audio. |
| ByteDance Pedal | `PIANO_BYTEDANCE_PEDAL` | MAESTRO `note onset F1 / pedal onset F1` | **96.72% / 91.86%** | Its same-type advantage is pedal output; generated MIDI preserves sustain pedal `CC64`. |

YourMT3+ / MIROS are multi-instrument backends and should not be directly ranked against the piano-specialized F1 scores above. ByteDance Pedal's `pedal onset F1` is also not equivalent to Transkun's `onset+offset+velocity F1`.

## Default Processing Strategy

The desktop, Space, and Colab interfaces no longer expose a user-adjustable quality preset. All modes use a fixed high-quality strategy. YourMT3+ still uses overlap, de-duplication, and MIDI post-processing; MIROS and dedicated piano backends run their own fixed checkpoint inference.

For `SIX_STEM_SPLIT`, the WAV stems come from BS-RoFormer SW, while stem MIDI is routed from one full-mix multi-instrument transcription by GM instrument family. The piano stem can use fixed-checkpoint Aria-AMT when that route is selected and the model is available.

## Requirements

| Item | Requirement |
|------|-------------|
| Python | 3.10+; the Windows installer prefers 3.10-3.12 |
| PyTorch | 2.7.0 with matching `torchaudio==2.7.0` and `torchvision==0.22.0` in the installer/release path |
| FFmpeg | Required for reliable MP3/M4A/FLAC/OGG handling |
| GPU | NVIDIA CUDA recommended; CPU works but is slow |
| OS | Windows 10/11, Linux, WSL2 |

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

You can also double-click `run.bat`. `run.ps1` checks the virtual environment, core imports, YourMT3+ weights, vocal separation resources, and ByteDance Piano resources, then calls `install.ps1` if something is missing.

### Linux / WSL2

```bash
chmod +x run.sh
./run.sh
```

`run.sh` checks the virtual environment, core imports, YourMT3+ source, YourMT3+ weights, vocal separation resources, and ByteDance Piano resources, then calls `install.sh` if something is missing.

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

CUDA 12.8 (recommended, required for newer NVIDIA CUDA 12 class environments):

```bash
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

CUDA 11.8:

```bash
pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
```

CI/CD and portable release packages no longer build CPU-only variants. For local source development, CPU-only PyTorch is a manual choice with slower inference and different dependency compatibility.

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare YourMT3+ Source and Weights

```bash
git clone https://github.com/mimbres/YourMT3.git
python download_sota_models.py
```

If `YourMT3/` already exists, you only need the model download step. `download_sota_models.py` prepares all five official YourMT3+ checkpoint modes, the BS-RoFormer SW six-stem assets, and the RoFormer `vocal_rvc` / `karaoke` vocal ensembles.

### 5. Prepare Separation and Piano Models

```bash
python download_vocal_model.py
python download_multistem_model.py
python download_vocal_harmony_model.py
python download_aria_amt_model.py
python download_bytedance_piano_model.py
python download_miros_model.py
```

The default cache location is:

```text
~/.cache/music_ai_models/yourmt3_all
~/.music-to-midi/models/audio-separator
~/.cache/music_ai_models/aria_amt
~/.cache/music_ai_models/bytedance_piano
external/ai4m-miros
```

Transkun resources are bundled with the `transkun` package. If Transkun mode reports missing resources, run `python -m pip install --force-reinstall transkun`.

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

The Space app tries to sync YourMT3 source from the Hugging Face Space repository. During conversion it checks/prepares the resources required by the selected mode: official YourMT3+ checkpoints, BS-RoFormer SW, RoFormer `vocal_rvc` / `karaoke`, Aria-AMT, or ByteDance Pedal. Resource preparation failures are surfaced by the processing flow.

## Portable Build

Windows directory-style portable build:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable.ps1
```

Specify Python or FFmpeg:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable.ps1 `
  -PythonExe .\venv\Scripts\python.exe `
  -FfmpegDir C:\ffmpeg\bin
```

The build script attempts to collect:

```text
YourMT3/amt/src
YourMT3 model cache -> models/yourmt3_all
audio-separator model cache -> models/audio-separator
Aria-AMT model cache -> models/aria_amt
ByteDance Piano model cache -> models/bytedance_piano
optional local MIROS checkout
ffmpeg.exe / ffprobe.exe
```

Portable asset source priority:

```text
MUSIC_TO_MIDI_BUNDLE_YOURMT3_DIR or ~/.cache/music_ai_models/yourmt3_all or checkpoints/yourmt3_all
MUSIC_TO_MIDI_BUNDLE_AUDIO_SEPARATOR_DIR or ~/.music-to-midi/models/audio-separator or checkpoints/audio-separator
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
    transkun_transcriber.py  # Transkun piano backend
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
download_sota_models.py      # Official YourMT3+ modes + BS-RoFormer SW six-stem assets
download_vocal_model.py      # Vocal separation model download
download_multistem_model.py  # Six-stem separation model download
download_aria_amt_model.py   # Aria-AMT model download
download_bytedance_piano_model.py # ByteDance Pedal model download
download_vocal_harmony_model.py # RoFormer karaoke vocal model download
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
git clone https://github.com/mimbres/YourMT3.git
python download_sota_models.py
```

### Vocal Separation Is Unavailable

Confirm dependency and model:

```bash
pip install "audio-separator==0.44.1" "onnxruntime==1.23.2" --no-deps
python download_vocal_model.py
python download_vocal_harmony_model.py
```

### Six-Stem Separation Is Unavailable

Confirm `audio-separator==0.44.1` is installed and download the BS-RoFormer SW resources:

```bash
python download_multistem_model.py
```

### Dedicated Piano Transcription Is Unavailable

Transkun mode needs the `transkun` package and its bundled pretrained resources:

```bash
python -m pip install --force-reinstall transkun
```

Aria-AMT mode needs the `aria-amt` package and checkpoint:

```bash
python -m pip install git+https://github.com/EleutherAI/aria-amt.git
python download_aria_amt_model.py
```

ByteDance Pedal mode needs `piano-transcription-inference`, `torchlibrosa`, and the ByteDance Piano checkpoint:

```bash
python -m pip install "piano-transcription-inference>=0.0.6,<0.1" "torchlibrosa>=0.1.0,<0.2"
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

This project uses the MIT License. Third-party models, datasets, and upstream repositories remain governed by their own licenses and terms.
