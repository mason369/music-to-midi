# Music to MIDI Converter

<p align="center">
  <a href="./README_zh.md">中文</a> | English
</p>

Music to MIDI is an AI-assisted audio-to-MIDI application with a PyQt6 desktop app, a Gradio Web interface, and a Google Colab notebook. The current product surface intentionally focuses on two supported workflows: full-mix multi-instrument transcription, and vocal/accompaniment split transcription.

## Screenshots

| Windows | Linux |
|---------|-------|
| ![Windows demo](../resources/icons/Windows演示.png) | ![Linux demo](../resources/icons/Linux演示.png) |

## Current Capabilities

- **Full-mix transcription**: `SMART` mode sends the whole audio file to the selected multi-instrument backend.
- **Vocal/accompaniment split transcription**: `VOCAL_SPLIT` separates vocals and accompaniment, transcribes both, and can optionally export one merged MIDI.
- **YourMT3+ default backend**: YourMT3+ MoE is the default path for multi-instrument notes, GM programs, drums, and multi-track MIDI output.
- **Optional MIROS backend**: the desktop app can route transcription through a local `ai4m-miros` checkout as an experimental backend.
- **MIDI layout control**: YourMT3+ can export by GM instrument, or merge non-drum notes into one melodic track while keeping drums separate.
- **Beat and post-processing**: BPM is detected automatically; MIDI generation includes tempo metadata, quantization, duplicate removal, velocity smoothing, and polyphony limiting.
- **Common audio formats**: `MP3`, `WAV`, `FLAC`, `OGG`, and `M4A` are accepted. Non-WAV input is converted to 44.1 kHz PCM WAV, preferably with FFmpeg.
- **Consistent mode set**: desktop, Space, and Colab expose only the two supported processing modes.

## Interface Matrix

| Interface | Modes | Backend Selection | Best For |
|-----------|-------|-------------------|----------|
| PyQt6 desktop | `SMART`, `VOCAL_SPLIT` | `YourMT3+`, `MIROS` | Local GPU use and persistent output folders |
| Gradio Space | `SMART`, `VOCAL_SPLIT` | Default `YourMT3+` | Browser-based use or hosted demos |
| Google Colab | `SMART`, `VOCAL_SPLIT` | Default `YourMT3+` | Temporary Colab GPU sessions |

## Processing Modes

| Mode | Internal Pipeline | Main Output | Notes |
|------|-------------------|-------------|-------|
| `SMART` | Audio -> multi-instrument backend -> MIDI generation | `<song>.mid` | No source separation. Suitable for most full mixes, instrumentals, and short multi-instrument clips. |
| `VOCAL_SPLIT` | Audio -> vocal/accompaniment separation -> accompaniment transcription -> vocal transcription -> MIDI generation | `<song>_accompaniment.mid`, `<song>_vocal.mid`, optional `<song>_vocal_accompaniment_merged.mid` | The vocal MIDI path filters the backend output toward a vocal melody track to reduce accompaniment hallucinations. |

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

You can also double-click `run.bat`. `run.ps1` checks the virtual environment, core imports, YourMT3+ weights, and the vocal separation model, then calls `install.ps1` if something is missing.

### Linux / WSL2

```bash
chmod +x run.sh
./run.sh
```

`run.sh` checks the virtual environment, core imports, YourMT3+ source, YourMT3+ weights, and the vocal separation model, then calls `install.sh` if something is missing.

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

### 5. Prepare the Vocal Separation Model

```bash
python download_vocal_model.py
```

The default cache location is:

```text
~/.music-to-midi/models/audio-separator
```

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

The Space app tries to sync YourMT3 source from the Hugging Face Space repository and checks the default model weights automatically.

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
    vocal_separator.py       # Vocal/accompaniment separation
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

### MIROS Is Unavailable

Check local repository files:

```text
ai4m-miros/main.py
ai4m-miros/transcribe.py
```

If the error lists missing Python modules, install the dependencies required by the upstream MIROS repository.

## License

This project uses the MIT License. Third-party models, datasets, and upstream repositories remain governed by their own licenses and terms.
