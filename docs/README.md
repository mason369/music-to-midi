# Music to MIDI Converter

<p align="center">
  <a href="./README_zh.md">中文</a> | English
</p>

Music to MIDI is an AI-assisted audio-to-MIDI application with a PyQt6 desktop app, a Gradio Web interface, and a Google Colab notebook. The current product surface syncs six processing modes: full-mix multi-instrument transcription, vocal/accompaniment split transcription, six-stem split transcription, and dedicated Transkun / Aria-AMT / ByteDance Pedal piano transcription.

## Screenshots

| Windows | Linux |
|---------|-------|
| ![Windows demo](../resources/icons/Windows演示.png) | ![Linux demo](../resources/icons/Linux演示.png) |

## Current Capabilities

- **Full-mix transcription**: `SMART` mode sends the whole audio file to the selected multi-instrument backend.
- **Vocal/accompaniment split transcription**: `VOCAL_SPLIT` separates vocals and accompaniment, transcribes both, and can optionally export one merged MIDI.
- **Six-stem split transcription**: `SIX_STEM_SPLIT` separates `bass / drums / guitar / piano / vocals / other`, then exports stem MIDI files and one merged MIDI.
- **Dedicated piano transcription**: `PIANO_TRANSKUN`, `PIANO_ARIA_AMT`, and `PIANO_BYTEDANCE_PEDAL` target pure piano audio through Transkun, Aria-AMT, and ByteDance's pedal-aware piano model.
- **Default backend semantics**: the config default prefers the `Aria-AMT` piano backend; `SMART`, `VOCAL_SPLIT`, and non-piano stems still use YourMT3+ or MIROS as multi-instrument backends.
- **Optional MIROS backend**: the desktop app can route transcription through a local `ai4m-miros` checkout as an experimental backend.
- **MIDI layout control**: YourMT3+ can export by GM instrument, or merge non-drum notes into one melodic track while keeping drums separate.
- **Beat and post-processing**: MIDI generation includes tempo metadata after BPM detection succeeds. If beat detection fails, processing stops instead of writing a fake default tempo. Quantization, duplicate removal, velocity smoothing, and polyphony limiting are available as post-processing.
- **Common audio formats**: `MP3`, `WAV`, `FLAC`, `OGG`, and `M4A` are accepted. Non-WAV input must be converted to 44.1 kHz PCM WAV through FFmpeg; FFmpeg failures stop processing and show the stderr root cause.
- **Consistent mode set**: desktop, Space, and Colab expose the same six processing modes.

## Interface Matrix

| Interface | Modes | Backend Selection | Best For |
|-----------|-------|-------------------|----------|
| PyQt6 desktop | `SMART`, `VOCAL_SPLIT`, `SIX_STEM_SPLIT`, `PIANO_TRANSKUN`, `PIANO_ARIA_AMT`, `PIANO_BYTEDANCE_PEDAL` | `Aria-AMT`, `ByteDance Pedal`, `YourMT3+`, `MIROS` | Local GPU use, persistent output folders, and dedicated piano transcription |
| Gradio Space | Same six modes as desktop | Default `Aria-AMT` / `ByteDance Pedal` piano backend + `YourMT3+` multi-instrument backend | Browser-based use or hosted demos |
| Google Colab | Same six modes as desktop | Default `Aria-AMT` / `ByteDance Pedal` piano backend + `YourMT3+` multi-instrument backend | Temporary Colab GPU sessions |

## Processing Modes

| Mode | Internal Pipeline | Main Output | Notes |
|------|-------------------|-------------|-------|
| `SMART` | Audio -> multi-instrument backend -> MIDI generation | `<song>.mid` | No source separation. Suitable for most full mixes, instrumentals, and short multi-instrument clips. |
| `VOCAL_SPLIT` | Audio -> vocal/accompaniment separation -> accompaniment transcription -> vocal transcription -> MIDI generation | `<song>_accompaniment.mid`, `<song>_vocal.mid`, optional `<song>_vocal_accompaniment_merged.mid` | The vocal MIDI path filters the backend output toward a vocal melody track to reduce accompaniment hallucinations. |
| `SIX_STEM_SPLIT` | Audio -> six-stem separation -> stem transcription -> stem MIDI merge | `<song>_<stem>.mid`, `<song>_all_stems_merged.mid` or `<song>_selected_stems_merged.mid` | Can transcribe only selected stems; the piano stem prefers Aria-AMT when that backend and checkpoint are available. |
| `PIANO_TRANSKUN` | Audio -> Transkun piano model -> MIDI | `<song>_piano_transkun.mid` | Best for pure piano audio; quality presets do not change Transkun checkpoint inference. |
| `PIANO_ARIA_AMT` | Audio -> Aria-AMT piano model -> MIDI | `<song>_piano_aria_amt.mid` | Best for pure piano audio; requires the Aria-AMT checkpoint. |
| `PIANO_BYTEDANCE_PEDAL` | Audio -> ByteDance pedal-aware piano model -> MIDI | `<song>_piano_bytedance_pedal.mid` | Best for pure piano audio when the output needs sustain pedal CC64; requires the ByteDance Piano checkpoint. |

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
song_piano_aria_amt.mid
song_piano_bytedance_pedal.mid
song_(Vocals).wav
song_(Instrumental).wav
```

The exact files depend on the selected mode, whether merged MIDI is enabled, and what the separator returns.

## Backends

### YourMT3+

YourMT3+ is the default backend. `download_sota_models.py` downloads the default checkpoint, and `src/core/yourmt3_transcriber.py` imports a local `YourMT3/amt/src` source tree.

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

Download the checkpoint:

```bash
python download_aria_amt_model.py
```

Default search roots include:

```text
~/.cache/music_ai_models/aria_amt
runtime/models/aria_amt
models/aria_amt
```

### ByteDance Pedal

ByteDance Pedal is a dedicated pedal-aware piano transcription backend for solo piano or clean piano stems. It comes from ByteDance's High-Resolution Piano Transcription with Pedals system. This project wraps it through `piano-transcription-inference` and preserves sustain pedal `CC64` events from the upstream MIDI output.

Install dependencies:

```bash
python -m pip install "piano-transcription-inference>=0.0.6,<0.1" "torchlibrosa>=0.1.0,<0.2"
```

Download the checkpoint:

```bash
python download_bytedance_piano_model.py
```

Default search roots include:

```text
~/.cache/music_ai_models/bytedance_piano
runtime/models/bytedance_piano
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
| YourMT3+ | Multi-instrument AMT | Multi-instrument or stem transcription in `SMART`, `VOCAL_SPLIT`, and `SIX_STEM_SPLIT` | Slakh2100 `Multi (Onset-Offset) F1 = 74.84` | Default full-mix route for multi-instrument and GM-track output. |
| MIROS | Multi-instrument AMT | Optional desktop multi-instrument backend | Upstream repository describes it as a 2025 AI4Musician winning model | Experimental local backend for A/B testing against YourMT3+ on the same task. |
| Transkun | Piano-specialized | `PIANO_TRANSKUN` | Transkun V2 and pip checkpoints publish MAESTRO V3 F1 values | Mature and lightweight; the current pip checkpoint is documented as not using pedal note extension. |
| Aria-AMT | Piano-specialized | `PIANO_ARIA_AMT` | Public checkpoint; this README does not invent a missing same-protocol F1 score | Current default piano candidate for normal pure-piano transcription. |
| ByteDance Pedal | Piano-specialized / pedal-aware | `PIANO_BYTEDANCE_PEDAL` | MAESTRO note onset F1 / pedal onset F1 = 96.72% / 91.86% | Prefer when the output needs sustain pedal CC64; never used as a silent substitute for other piano backends. |
| BS-RoFormer | Vocal/accompaniment separation | Pre-separation for `VOCAL_SPLIT` | Checkpoint filename includes a training score label, not a unified benchmark | Separation only; final MIDI quality still depends on the transcription backend. |
| BS-RoFormer SW | Six-stem separation | Pre-separation for `SIX_STEM_SPLIT` | MVSEP 6-stem SDR protocol | Separation SDR is not end-to-end MIDI F1. |

YourMT3+ / MIROS are multi-instrument backends, Transkun / Aria-AMT / ByteDance Pedal are piano-specialized backends, and BS-RoFormer models are source-separation backends. Their public metrics must not be collapsed into one leaderboard.

#### Piano Model Quality Comparison

| Model | Current Project Entry | Same-Type Quality Protocol | Public Result | How To Read It |
|-------|-----------------------|----------------------------|---------------|----------------|
| Transkun V2 | Research checkpoint, not the current pip default entry | MAESTRO V3 `note onset F1 / onset+offset F1 / onset+offset+velocity F1` | **0.9832 / 0.9349 / 0.9296** | Strong public piano AMT reference. |
| Transkun pip checkpoint (No Ext) | `PIANO_TRANSKUN` | MAESTRO V3 No Ext, same three metrics | **0.9833 / 0.8149 / 0.8109** | Lightweight and bundled, but upstream documents it as `without pedal extension of notes`. |
| Aria-AMT | `PIANO_ARIA_AMT` | Public checkpoint, but no fully matching published Transkun-style benchmark table | No unified F1 written here | Use as the current default piano candidate; compare with local A/B audio. |
| ByteDance Pedal | `PIANO_BYTEDANCE_PEDAL` | MAESTRO `note onset F1 / pedal onset F1` | **96.72% / 91.86%** | Its same-type advantage is pedal output; generated MIDI preserves sustain pedal `CC64`. |

YourMT3+ / MIROS are multi-instrument backends and should not be directly ranked against the piano-specialized F1 scores above. ByteDance Pedal's `pedal onset F1` is also not equivalent to Transkun's `onset+offset+velocity F1`.

## MIDI Track Layout

YourMT3+ provides two output layouts:

| Layout | Behavior |
|--------|----------|
| Multi-track by GM instrument | Each recognized GM program is kept as separate as possible; drums use the GM drum channel. |
| Single melodic track, drums separate | Non-drum notes are written into one melodic track while keeping channel/program changes; drums remain separate. |

If the detected instrument count exceeds available non-drum MIDI channels, the generator merges related instrument families instead of dropping notes outright.

## Quality Presets

The desktop and Web interfaces expose:

```text
fast
balanced
best
```

- For `YourMT3+`, the preset affects post-processing.
- For `MIROS`, the current wrapper uses fixed checkpoint quality; the preset does not change MIROS inference.
- For `Transkun` / `Aria-AMT` / `ByteDance Pedal`, dedicated piano modes use fixed checkpoint quality.
- For `SIX_STEM_SPLIT`, multi-instrument stems follow the active multi-instrument backend; the piano stem uses fixed checkpoint quality when routed through Aria-AMT.

## Requirements

| Item | Requirement |
|------|-------------|
| Python | 3.10+; the Windows installer prefers 3.10-3.12 |
| PyTorch | 2.4.0 or newer, with matching `torchaudio` and `torchvision` |
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

CUDA 12.1:

```bash
pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.8:

```bash
pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
```

CPU:

```bash
pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare YourMT3+ Source and Weights

```bash
git clone https://github.com/mimbres/YourMT3.git
python download_sota_models.py
```

If `YourMT3/` already exists, you only need the model download step.

### 5. Prepare Separation and Piano Models

```bash
python download_vocal_model.py
python download_multistem_model.py
python download_aria_amt_model.py
python download_bytedance_piano_model.py
```

The default cache location is:

```text
~/.music-to-midi/models/audio-separator
~/.cache/music_ai_models/aria_amt
~/.cache/music_ai_models/bytedance_piano
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

The Space app tries to sync YourMT3 source from the Hugging Face Space repository and checks default YourMT3+, Aria-AMT, and ByteDance Piano model weights automatically.

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
YourMT3 model cache
audio-separator model cache
Aria-AMT model cache
ByteDance Piano model cache
optional local MIROS checkout
ffmpeg.exe / ffprobe.exe
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
    vocal_harmony_separator.py # Experimental lead/harmony vocal separation
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
download_sota_models.py      # Default YourMT3+ model download
download_vocal_model.py      # Vocal separation model download
download_multistem_model.py  # Six-stem separation model download
download_aria_amt_model.py   # Aria-AMT model download
download_bytedance_piano_model.py # ByteDance Pedal model download
download_vocal_harmony_model.py # Experimental lead/harmony model download
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
pip install "audio-separator>=0.38.0" "onnxruntime>=1.16.0,<2"
python download_vocal_model.py
```

### Six-Stem Separation Is Unavailable

Confirm `audio-separator>=0.38.0` is installed and download the BS-RoFormer SW resources:

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
