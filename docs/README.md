# Music to MIDI Converter

<p align="center">
  <a href="./README_zh.md">中文</a> | English
</p>

Convert audio files to multi-track MIDI with automatic 128 GM instrument recognition.

**Platform Support: Windows / Linux / WSL2**

## Screenshots

| Windows | Linux |
|---------|-------|
| ![Windows Demo](../resources/icons/Windows演示.png) | ![Linux Demo](../resources/icons/Linux演示.png) |

## Features

- **Multi-Instrument Transcription**: Uses a high-performance YourMT3+ MoE model for direct multi-instrument recognition from mixed audio
- **Vocal Separation Mode**: BS-RoFormer separates vocals/accompaniment and transcribes each to an independent MIDI file (default checkpoint: `model_bs_roformer_ep_317_sdr_12.9755.ckpt`)
- **128 GM Instruments**: Outputs standard General MIDI multi-track MIDI, accurately distinguishing drums, bass, guitar, piano, etc.
- **MIDI Post-processing**: Note quantization, velocity smoothing, deduplication, polyphony limiting
- **GPU Acceleration**: Auto-detects and uses CUDA (NVIDIA) / ROCm (AMD) / CPU
- **Multi-language UI**: Support for English and Chinese interface
- **Professional Dark Theme**: Modern audio software-style interface design

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Windows 10/11 (x64) | ✅ Supported | Double-click `run.bat` to launch |
| Linux (Ubuntu/Debian) | ✅ Supported | Full functionality, Ubuntu 22.04+ recommended |
| WSL2 (Windows 11) | ✅ Supported | Requires WSLg (built-in on Win11) |
| macOS | 🚧 Planned | Apple Silicon MPS support in development |

## Quick Start

### Windows

```
1. Clone or download the repository
2. Double-click run.bat (auto-installs all dependencies on first run)
```

Or use PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File run.ps1
```

### Linux

```bash
# 1. Clone the repository
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# 2. Run directly (auto-installs all dependencies on first run)
./run.sh
```

## Installation

### Prerequisites

- **Python 3.10+** (3.10 or 3.11 recommended, 3.12 may have compatibility issues)
- **FFmpeg**: Required for audio processing
  - Windows: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Linux: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo dnf install ffmpeg` (Fedora)
  - macOS: `brew install ffmpeg`
- **NVIDIA GPU + CUDA** (recommended): For significantly faster processing

### Git LFS Installation

- Windows:
  - `choco install git-lfs` or `winget install GitHub.GitLFS`
  - After install: `git lfs install`
- macOS:
  - `brew install git-lfs`
  - After install: `git lfs install`
- Linux:
  - Ubuntu/Debian: `sudo apt-get install git-lfs`
  - Fedora: `sudo dnf install git-lfs`
  - After install: `git lfs install`

### Dependency Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| PyTorch | 2.1.0 - 2.4.x | YourMT3+ compatibility |
| torchaudio | 2.1.0 - 2.4.x | Must match PyTorch version |
| NumPy | < 2.0 | numba compatibility |
| CUDA | 11.8 or 12.1 | GPU acceleration (optional) |
| Python | 3.10+ | 3.10 or 3.11 recommended |

### Linux Installation (Recommended)

Linux is the recommended platform for running this project - environment setup is simpler and GPU acceleration is more stable.

```bash
# 1. Clone the repository
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# 2. Create virtual environment (conda recommended)
conda create -n music2midi python=3.10
conda activate music2midi

# Or use venv
python -m venv venv
source venv/bin/activate

# 3. Install PyTorch (choose based on your CUDA version)
# CUDA 11.8
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended, slower)
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Download YourMT3+ models
python download_sota_models.py

# 6. Run the application
python -m src.main
```

### Windows Installation

Note: `tflite-runtime` is not available on Windows. `requirements.txt` uses a platform marker to install `tensorflow` instead, so make sure you are using a recent version of `pip`.

```bash
# Clone the repository
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch (choose one)
# CUDA 11.8
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only (no GPU or no CUDA needed)
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Download YourMT3+ models
python download_sota_models.py

# Run the application
python -m src.main
```

### CUDA Installation Guide

#### Linux (Ubuntu/Debian)

```bash
# Method 1: Using NVIDIA official repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1

# Method 2: Using conda (recommended, automatic management)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

#### Windows

1. Download CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Choose custom installation, ensure cuDNN is selected
3. Restart and verify: `nvidia-smi`

### Install from Release

Download the latest release from the [Releases](https://github.com/mason369/music-to-midi/releases) page.

## Usage

1. **Open Audio File**: Drag and drop an audio file (MP3, WAV, FLAC, OGG) or click to browse
2. **Configure Output**: Choose output directory and options (MIDI, lyrics, separated tracks)
3. **Start Processing**: Click "Start" to begin conversion
4. **Get Results**: Find MIDI file, LRC lyrics, and separated audio tracks in output directory

## Supported Formats

### Input
- MP3, WAV, FLAC, OGG, M4A, AAC, WMA

### Output
- MIDI (.mid) - Multi-track MIDI

## Technical Details

### Architecture Overview

```
Audio Input (16kHz mono)
    ↓
MusicToMidiPipeline
    ├── BeatDetector (librosa) → BPM / beat grid
    └── YourMT3Transcriber
            ↓
        ┌─────────────────────────────────────────────┐
        │  YourMT3+ MoE (YPTF.MoE+Multi PS)          │
        │                                             │
        │  Audio → Mel Spectrogram (256×512)           │
        │    ↓                                        │
        │  Pre-Encoder (Conv2d Res3B, 1→C, /8 freq)   │
        │    ↓                                        │
        │  PerceiverTF Encoder (26 latents, MoE×8)    │
        │    ↓                                        │
        │  Multi-T5 Decoder (13 channels, AR greedy)  │
        │    ↓                                        │
        │  Token → Note Events (per channel)          │
        └─────────────────────────────────────────────┘
            ↓
        Smart deduplication (overlapping segment merge)
            ↓
        MIDI post-processing (quantize / velocity / polyphony)
            ↓
        Multi-track MIDI Output (up to 128 GM instruments)
```

### Source Code Structure

```
src/
├── main.py                          # Application entry point
├── core/                            # Core processing engine
│   ├── pipeline.py                  # Main processing pipeline (MusicToMidiPipeline)
│   ├── yourmt3_transcriber.py       # YourMT3+ transcriber wrapper
│   ├── beat_detector.py             # Beat/BPM detection
│   ├── midi_generator.py            # MIDI generation & post-processing
│   └── vocal_separator.py           # BS-RoFormer vocal separation
├── gui/                             # PyQt6 graphical interface
│   ├── main_window.py               # Main window (MainWindow)
│   ├── widgets/
│   │   ├── dropzone.py              # File drag-and-drop zone
│   │   ├── track_panel.py           # Track/mode configuration panel
│   │   └── progress_widget.py       # Progress display component
│   └── workers/
│       └── processing_worker.py     # QThread background worker
├── models/                          # Data models
│   ├── data_models.py               # All core data classes (Config, NoteEvent, BeatInfo...)
│   └── gm_instruments.py            # 128 GM instrument definitions & mappings
├── i18n/                            # Internationalization
│   └── translator.py                # Translation system (zh_CN / en_US)
└── utils/                           # Utility modules
    ├── gpu_utils.py                 # GPU detection/device selection/memory management
    ├── yourmt3_downloader.py        # Model weight download & caching
    ├── audio_utils.py               # Audio loading/resampling/format conversion
    ├── logger.py                    # Colored logging system
    └── warnings_filter.py           # Third-party library warning suppression
```

---

### Application Entry & Startup Flow

`src/main.py` (~150 lines) handles environment initialization and GUI launch:

```
main()
├── Environment variables
│   ├── OMP_NUM_THREADS = physical core count (dynamically detected)
│   ├── MKL_NUM_THREADS = same
│   └── TF_CPP_MIN_LOG_LEVEL = 3 (suppress TensorFlow logs)
├── Windows-specific handling
│   ├── DLL path fix (CJK paths → 8.3 short paths)
│   └── PyTorch DLL preloading (torch.dll, torch_cpu.dll)
├── Third-party warning suppression (warnings_filter.py)
├── Logging system initialization (logger.py)
└── PyQt6 application launch
    ├── High DPI scaling: Floor policy (prevent oversized UI)
    ├── Style: Fusion (cross-platform consistent)
    ├── CJK font: Microsoft YaHei / Noto Sans CJK
    ├── MainWindow creation
    └── Event loop (app.exec())
```

---

### Processing Pipeline: MusicToMidiPipeline

`src/core/pipeline.py` (~400 lines) is the core orchestrator coordinating all processing stages.

Two processing modes are supported:

#### Mode 1: SMART (Default)

```
Audio file
    ↓
BeatDetector.detect() → BPM / beat grid          [0-10%]
    ↓
YourMT3Transcriber.transcribe_precise()           [10-85%]
    → instrument_notes: Dict[program, List[NoteEvent]]
    → drum_notes: Dict[program, List[NoteEvent]]
    ↓
MidiGenerator.generate_from_precise_instruments_v2()  [85-95%]
    ↓
Multi-track MIDI file                              [95-100%]
```

#### Mode 2: VOCAL_SPLIT

```
Audio file
    ↓
BeatDetector.detect() → BPM                       [0-5%]
    ↓
VocalSeparator.separate()                          [5-35%]
    → vocals.wav + no_vocals.wav
    ↓
YourMT3Transcriber.transcribe_precise(accompaniment)  [35-60%]
    ↓
YourMT3Transcriber.transcribe_precise(vocals)      [60-85%]
    ↓
MidiGenerator × 2 (accompaniment MIDI + vocal MIDI)  [85-95%]
    ↓
Two MIDI files                                     [95-100%]
```

Key design decisions:
- **Cancel propagation**: `pipeline.cancel()` propagates to all submodules, supports abort at any stage
- **Progress callbacks**: each stage sends `ProcessingProgress` via `_report()`, containing both stage and overall progress
- **Error isolation**: each stage has independent try/catch, finally blocks ensure model unloading and GPU memory cleanup

---

### Beat Detection: BeatDetector

`src/core/beat_detector.py` (~380 lines) uses a multi-algorithm fusion strategy with librosa for BPM detection.

```
Audio file
    ↓
librosa.load(sr=22050, mono=True)
    ↓
4 algorithms in parallel:
    ├── librosa.beat.beat_track() → tempo₁
    ├── librosa.beat.tempo(onset_envelope) → tempo₂
    ├── librosa.beat.tempo(aggregate=np.mean) → tempo₃
    └── librosa.feature.tempogram() → tempo₄
    ↓
Octave correction: _correct_octave_error()
    Constrain candidates to 60-200 BPM range
    (e.g., 240 BPM → 120 BPM, 50 BPM → 100 BPM)
    ↓
Cluster voting: _vote_best_tempo()
    Clustering threshold: 8 BPM
    Select cluster center containing the most candidates
    ↓
BeatInfo(bpm, beat_times, downbeats, time_signature)
```

Design highlights:
- Multi-algorithm voting is more robust than any single algorithm, avoiding extreme values
- Octave correction solves the common "double/half BPM" misdetection problem
- Graceful degradation to default 120 BPM on failure

---

### Vocal Separation Model: BS-RoFormer

This project uses **BS-RoFormer** (Band-Split Rotary Transformer) for vocal/accompaniment separation, wrapped via the [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) library.

| Item | Details |
|------|---------|
| Full Name | Band-Split RoFormer |
| Paper | [Music Source Separation with Band-Split RoFormer](https://arxiv.org/abs/2309.02612) (ISMIR 2023 Workshop) |
| Checkpoint | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (epoch 317) |
| Trainer | [ZFTurbo](https://github.com/ZFTurbo) / [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) |
| License | MIT |
| Metric note | Checkpoint filename includes `sdr_12.9755` (training-time score tag, not a universal benchmark score) |
| Public comparison (vocal SDR) | Multisong: BS-RoFormer 10.87 / MelBand-RoFormer(Kim) 10.98 / HTDemucs_ft 8.38 |
| Model Size | ~500 MB |
| First Use | Auto-downloaded from HuggingFace to `~/.music-to-midi/models/audio-separator/` |

#### Architecture Overview

BS-RoFormer's core idea is to split the spectrogram into frequency bands for independent modeling, enhanced with Rotary Position Embedding (RoPE) for temporal modeling:

```
Audio waveform (44.1kHz stereo)
    ↓
STFT → Complex spectrogram (F × T)
    ↓
Band-Split Module
    Split frequency axis into K sub-bands by predefined boundaries
    Each band independently mapped to D-dim embedding via MLP
    → (K, T, D)
    ↓
N × Band-Split RoFormer Blocks
    ├── Band-level Self-Attention (inter-band interaction)
    │   K bands as tokens, captures cross-band harmonic relationships
    │   RoPE encodes temporal position
    ├── Temporal Self-Attention (temporal modeling)
    │   T time frames as tokens, captures temporal dependencies
    │   RoPE encodes temporal position
    └── Feed-Forward Network
    ↓
Band-Merge Module
    Map K band embeddings back to frequency dimension
    → Complex mask (F × T)
    ↓
Mask × Original spectrogram → iSTFT
    ↓
Separated waveform (vocals / instrumental)
```

#### Why BS-RoFormer (Replacing Demucs)

The project originally used Meta's Demucs (htdemucs) for vocal separation, replaced entirely with BS-RoFormer in commit `d6309de`:

| Dimension | Demucs (htdemucs) | BS-RoFormer |
|-----------|-------------------|-------------|
| Public comparison (Multisong vocal SDR) | 8.38 (HTDemucs_ft) | 10.87 (BS-RoFormer_ep_317) |
| Architecture | Hybrid U-Net + Transformer | Pure Transformer (band-split) |
| Integration | Manual model management | audio-separator 3-step API |
| Dependencies | Heavy (demucs' own deps) | Lightweight (audio-separator + onnxruntime) |
| Packaging | Poor PyInstaller compat | Good |

Separation quality directly affects downstream YourMT3+ transcription quality. This document now annotates source + dataset + metric together to avoid cross-benchmark misinterpretation.

#### Processing Flow

`src/core/vocal_separator.py` (~200 lines):

```
Audio file
    ↓
Separator(output_dir, model_file_dir, output_format="WAV")
    ↓
separator.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    ↓
separator.separate(audio_path)
    ↓
2 stems:
    ├── {stem}_(Vocals).wav   → renamed to {stem}_vocals.wav
    └── {stem}_(Instrumental).wav → renamed to {stem}_accompaniment.wav
    ↓
Output: {"vocals": vocals_path, "no_vocals": accompaniment_path}
```

Key design decisions:
- **Auto GPU detection**: audio-separator's Separator class detects GPU devices internally, does not accept external device parameter
- **Blocking call**: `separator.separate()` is a blocking call; cancellation can only be checked before/after the call, not during
- **Background progress thread**: estimates progress every 3 seconds based on elapsed time (audio-separator provides no fine-grained callbacks)
- **Speed estimation**: GPU ~3x real-time, CPU ~0.15x real-time

#### Frontier Vocal Separation Models (Verified on 2026-03-01)

| Model/Direction | Source | Type | Status | Notes |
|-----------------|--------|------|--------|-------|
| BS-RoFormer ep317 (current) | Current project default checkpoint | Local drop-in (audio-separator) | ✅ In use | Multisong vocal SDR 10.87 (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`) |
| SCNet XL IHF | [ZFTurbo pretrained list](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) | Local drop-in (audio-separator) | ✅ Drop-in | Multisong vocal SDR 11.11 (`model_scnet_xl_2ep_..._musdb18hq.ckpt`), stronger practical local replacement |
| MelBand-RoFormer (Kim) | [ZFTurbo pretrained list](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) | Local drop-in (audio-separator) | ✅ Drop-in | Multisong vocal SDR 10.98 (`Kim_Vocal_2.onnx`) |
| Ensemble (vocals,instrum) | [MVSEP Algorithms](https://mvsep.com/algorithms) | Frontier leaderboard (service/ensemble) | 🌐 Available (not local drop-in) | MVSEP vocals SDR 11.93 (ver 2025.06), among the highest public entries |
| BS Roformer (vocals,instrum) | [MVSEP Algorithms](https://mvsep.com/algorithms) | Frontier leaderboard (service/single model) | 🌐 Available (not local drop-in) | MVSEP vocals SDR 11.89 (ver 2025.07) |
| BS Roformer SW (vocals,instrum, 6 stem) | [MVSEP Quality Checker](https://mvsep.ru/quality_checker/synth_mutlitrack/create_table) | Frontier leaderboard (service/multi-stem) | 🌐 Available (not local drop-in) | MVSEP vocals SDR 11.30; multi-stem oriented, not equal to this project's 2-stem local replacement path |
| Mel-RoFormer (ISMIR 2024) | [arXiv:2409.04702](https://arxiv.org/abs/2409.04702) / [ar5iv Table 2](https://ar5iv.org/html/2409.04702v1) | Paper-stage (research model) | 📄 Published paper | MUSDB18-HQ (Table 2, Scenario ⓑ, with extra data): Vocals SDR 13.29; same table reports BS-RoFormer 12.82 |
| Mamba2 Meets Silence (v2, 2025) | [arXiv:2508.14556](https://arxiv.org/abs/2508.14556) | Paper-stage (research model) | 📄 Paper | Abstract reports cSDR 11.03 dB (claimed as best reported), focused on sparse-vocal robustness |
| Windowed Sink Attention (2025) | [arXiv:2510.25745](https://arxiv.org/abs/2510.25745) | Paper-stage (efficiency direction) | 📄 Paper + open code | Under fine-tuning, recovers ~92% of original model SDR with ~44.5x FLOPs reduction (primarily efficiency gain) |

> **Conclusion (by protocol):**  
> - For **practical local open replacement**, SCNet XL IHF is currently the stronger option than the default ep317 in documented Multisong SDR.  
> - For **MVSEP leaderboard**, Ensemble (11.93) and BS Roformer (11.89) are higher.  
> - For **paper-specific MUSDB18-HQ setting (Table 2, Scenario ⓑ)**, Mel-RoFormer reports 13.29.  
> **Protocol note:** Do not directly compare numbers across different datasets/protocols (Multisong, MUSDB, MVSEP, cSDR/uSDR).

---

### MIDI Generation & Post-Processing: MidiGenerator

`src/core/midi_generator.py` (~1100 lines) converts note events into standard MIDI files.

#### Core Method: generate_from_precise_instruments_v2()

```
instrument_notes: Dict[int, List[NoteEvent]]  (program → note list)
drum_notes: Dict[int, List[NoteEvent]]        (program → drum note list)
    ↓
Post-processing chain (selected by quality mode):
    ↓
MIDI channel allocation:
    ├── Channels 0-8, 10-15: melodic instruments (max 15 types)
    ├── Channel 9: drums (GM standard, fixed)
    └── Over 15 instruments: smart merge of same-family instruments
    ↓
Vocal special handling:
    program 100/101 (YourMT3 vocals) → mapped to piano timbre
    ↓
mido.MidiFile write:
    ├── Track 0: tempo track (tempo meta event)
    └── Track 1-N: instrument tracks
        ├── program_change (timbre selection)
        ├── note_on / note_off (note events)
        └── control_change (pedals etc.)
    ↓
.mid file output
```

#### Post-Processing Chain (7 Steps)

Different post-processing intensity based on `transcription_quality`:

| Step | Method | best | balanced | fast |
|------|--------|------|----------|------|
| 1 | `_smooth_vibrato()` vibrato smoothing | ❌ | ✅ | ❌ |
| 2 | `_remove_duplicate_notes(25ms)` dedup | ❌ | ✅ | ❌ |
| 3 | `_merge_close_notes(10ms)` merge fragments | ❌ | ✅ | ❌ |
| 4 | `_quantize_notes(1/32)` grid quantization | ❌ | ✅ | ❌ |
| 5 | `_smooth_velocity(window=5)` velocity smoothing | ❌ | ✅ | ❌ |
| 6 | `_normalize_velocity(mean=80)` velocity normalization | ❌ | ✅ | ❌ |
| 7 | `_limit_polyphony(max=40)` polyphony limit | ❌ | ✅ | ❌ |
| Special | `post_process_minimal()` remove <10ms only | ✅ | ❌ | ❌ |

`best` mode intentionally skips most post-processing, preserving the AI model's raw output and only removing obvious noise notes (<10ms). This is because YourMT3+'s output quality is already high — excessive post-processing would lose detail.

---

### GUI Layer

#### Main Window: MainWindow

`src/gui/main_window.py` (~900 lines) built with PyQt6, dark theme Fusion style.

```
┌─────────────────────────────────────────┐
│  🎵 Music to MIDI                       │  ← _create_header()
│  Convert audio to multi-track MIDI      │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐    │
│  │  🎵 Drag & drop audio here      │    │  ← DropZoneWidget
│  │     or click to browse          │    │
│  └─────────────────────────────────┘    │
├─────────────────────────────────────────┤
│  Mode: [Smart Mode ▼]                   │  ← TrackPanel
│  YourMT3+ direct multi-instrument       │
├─────────────────────────────────────────┤
│  ● Preprocess → ● Transcribe → ○ Gen   │  ← ProgressWidget
│  ████████████████░░░░░░░░  65%          │     (StageIndicator × N)
│  Transcribing: segment 15/28...         │
├─────────────────────────────────────────┤
│  Output: [/path/to/output] [Browse]     │  ← _create_output_settings()
├─────────────────────────────────────────┤
│  [  Start  ]  [  Stop  ]               │  ← _create_action_buttons()
├─────────────────────────────────────────┤
│  GPU: NVIDIA RTX 4090 | VRAM: 8.2/24GB │  ← Status bar
└─────────────────────────────────────────┘
```

Key design:
- **Background GPU detection**: GPU detected in a separate thread at startup to avoid blocking UI
- **Signal-slot communication**: `ProcessingWorker` sends progress/result/error via Qt signals, thread-safe
- **Shadow effects**: `QGraphicsDropShadowEffect` adds depth to card components
- **Menu bar**: File (Open/Exit), Edit (Settings), View (Language switch), Help (About)

#### Background Worker: ProcessingWorker

`src/gui/workers/processing_worker.py` (~70 lines) extends `QThread`.

```
MainWindow._start_processing()
    ↓
ProcessingWorker(QThread)
    ├── Signals:
    │   ├── progress_updated(ProcessingProgress)  → update progress bar
    │   ├── processing_finished(ProcessingResult) → show completion dialog
    │   └── error_occurred(str)                   → show error dialog
    ├── run():
    │   ├── MusicToMidiPipeline(config)
    │   ├── pipeline.process(audio_path, output_dir)
    │   └── finally: clear_gpu_memory()
    └── cancel():
        └── pipeline.cancel()
```

#### Custom Widgets

| Widget | File | Function |
|--------|------|----------|
| `DropZoneWidget` | `widgets/dropzone.py` (~200 lines) | Drag-drop + browse dual-channel file input, supports MP3/WAV/FLAC/OGG/M4A/AAC/WMA |
| `TrackPanel` | `widgets/track_panel.py` (~120 lines) | Processing mode selection (Smart/Vocal Split), emits `mode_changed` signal |
| `ProgressWidget` | `widgets/progress_widget.py` (~280 lines) | Stage flow indicators + progress bar, dynamically rebuilds stages per mode |
| `StageIndicator` | `widgets/progress_widget.py` | Single stage status indicator (pending/current/done), with color and icon transitions |

---

### Data Models

`src/models/data_models.py` (~650 lines) defines all core data structures.

#### Enum Types

```python
ProcessingMode        # SMART | VOCAL_SPLIT | PIANO(deprecated→SMART)
TranscriptionQuality  # FAST | BALANCED | BEST
ProcessingStage       # PREPROCESSING | SEPARATION | TRANSCRIPTION |
                      # VOCAL_TRANSCRIPTION | SYNTHESIS | COMPLETE
InstrumentType        # PIANO | DRUMS | BASS | GUITAR | VOCALS | STRINGS |
                      # BRASS | WOODWIND | SYNTH | ORGAN | ... (25 types)
                      # Provides to_program_number() / get_display_name(lang)
```

#### Core Data Classes

```python
@dataclass
class NoteEvent:
    pitch: int          # MIDI pitch (0-127)
    start_time: float   # Start time (seconds)
    end_time: float     # End time (seconds)
    velocity: int       # Velocity (0-127)
    program: int        # GM program number (0-127, 128=drums)

@dataclass
class BeatInfo:
    bpm: float                          # Detected BPM
    beat_times: List[float]             # Beat time points
    downbeats: Optional[List[float]]    # Downbeat time points
    time_signature: tuple               # Time signature (4, 4)

@dataclass
class ProcessingProgress:
    stage: ProcessingStage      # Current stage
    stage_progress: float       # Stage progress (0-1)
    overall_progress: float     # Overall progress (0-1)
    message: str                # Display message

@dataclass
class ProcessingResult:
    midi_path: str                              # Main MIDI file path
    tracks: List[Track]                         # Track list
    beat_info: Optional[BeatInfo]               # Beat info
    processing_time: float                      # Processing time (seconds)
    total_notes: int                            # Total note count
    vocal_midi_path: Optional[str]              # Vocal MIDI (VOCAL_SPLIT only)
    accompaniment_midi_path: Optional[str]      # Accompaniment MIDI (VOCAL_SPLIT only)
    separated_audio: Optional[Dict[str, str]]   # Separated audio paths

@dataclass
class Config:
    # Interface
    language: str = "zh_CN"             # UI language
    theme: str = "dark"                 # Theme
    # GPU
    use_gpu: bool = True                # Enable GPU
    gpu_device: int = 0                 # GPU device index
    # Processing
    processing_mode: str = "smart"      # Processing mode
    transcription_quality: str = "best" # Transcription quality
    # MIDI post-processing
    quantize_notes: bool = True         # Note quantization
    quantize_grid: str = "1/32"         # Quantization grid
    max_polyphony: int = 40             # Max polyphony
    aggressive_post_processing: bool = False  # Aggressive post-processing
    # ... more fields in source
```

#### GM Instrument Mapping

`src/models/gm_instruments.py` (~450 lines) defines the complete 128 General MIDI instruments:

```
16 instrument families (GMFamily):
├── PIANO (0-7):       Acoustic Grand → Clavinet
├── CHROMATIC (8-15):  Celesta → Dulcimer
├── ORGAN (16-23):     Drawbar Organ → Tango Accordion
├── GUITAR (24-31):    Nylon Guitar → Guitar Harmonics
├── BASS (32-39):      Acoustic Bass → Synth Bass 2
├── STRINGS (40-47):   Violin → Tremolo Strings
├── ENSEMBLE (48-55):  String Ensemble → Orchestra Hit
├── BRASS (56-63):     Trumpet → Tuba
├── REED (64-71):      Soprano Sax → Bassoon
├── PIPE (72-79):      Piccolo → Ocarina
├── SYNTH_LEAD (80-87): Square → Charang
├── SYNTH_PAD (88-95): New Age → Sweep
├── SYNTH_FX (96-103): Rain → Sci-fi
├── ETHNIC (104-111):  Sitar → Shanai
├── PERCUSSIVE (112-119): Tinkle Bell → Reverse Cymbal
└── SOUND_FX (120-127): Guitar Fret → Gunshot

YourMT3 extensions:
├── Program 100: Singing Voice (female)
└── Program 101: Singing Voice (male)
```

Bilingual instrument name lookup: `get_instrument_name(program=0, language="zh_CN")` → `"原声大钢琴"`

---

### Utility Modules

#### GPU Management: gpu_utils.py

`src/utils/gpu_utils.py` (~900 lines) manages detection, selection, and memory optimization for multiple accelerators.

```
Supported accelerators:
├── CUDA (NVIDIA)     — Primary support, auto-detects CUDA version
├── ROCm (AMD)        — Via torch.cuda (ROCm compatibility layer)
├── MPS (Apple)       — macOS Apple Silicon
├── XPU (Intel)       — Requires intel_extension_for_pytorch
├── DirectML          — Windows universal GPU (experimental)
└── CPU               — Fallback

Key functions:
├── get_device(prefer_gpu, gpu_index) → "cuda:0" / "cpu" / ...
├── get_optimal_batch_size(n_segments, quality, device, ultra_quality)
│   ├── Performance tier detection: high (≥8GB) / medium (4-8GB) / low (<4GB)
│   ├── best mode: bsz = 4/2/1
│   ├── balanced mode: bsz = 8/4/2
│   └── fast mode: bsz = 16/8/4
├── clear_gpu_memory()
│   ├── torch.cuda.empty_cache()
│   ├── gc.collect()
│   └── Platform-specific cleanup
└── diagnose_gpu() → full diagnostic report (dict)
```

Special handling:
- Windows DLL path fix: CJK username/special character paths → 8.3 short paths
- OOM auto-fallback: CUDA OOM during inference → halve batch size and retry
- Dynamic thread count: `OMP_NUM_THREADS` set to physical core count (not logical)

#### Model Download: yourmt3_downloader.py

`src/utils/yourmt3_downloader.py` (~550 lines) manages YourMT3+ model weight download and caching.

```
Model cache structure:
~/.cache/music_ai_models/yourmt3_all/
└── amt/logs/2024/{checkpoint_name}/checkpoints/model.ckpt

Download flow:
├── Check local cache → hit: return path directly
├── Try Hugging Face main site download
├── Fail → try HF mirror (hf-mirror.com, China acceleration)
├── SSL certificate fix (corporate network/proxy environments)
├── Resume download support
└── Recursive path search (compatible with repo structure changes)

Checkpoint name mapping:
├── "YPTF.MoE+Multi (PS)" → mc13_256_g4_all_v7_mt3f_...
├── "YPTF.MoE+Multi (noPS)" → ...
├── "YPTF+Multi (PS)" → ...
└── "YPTF+Multi (noPS)" → ...
```

#### Internationalization: translator.py

`src/i18n/translator.py` (~200 lines) provides the global translation function `t(key)`.

```
Design:
├── Singleton pattern: get_translator() returns global instance
├── JSON storage: src/i18n/zh_CN.json, src/i18n/en_US.json
├── Nested keys: t("menu.file.open") → lookup {"menu": {"file": {"open": "Open"}}}
├── Parameter substitution: t("progress.segment", current=5, total=28) → "Segment 5/28"
└── Fallback: current language missing → try en_US → return raw key

Supported languages:
├── zh_CN: Simplified Chinese (default)
└── en_US: English
```

#### Other Utilities

| Module | Lines | Function |
|--------|-------|----------|
| `audio_utils.py` | ~100 | Audio load/save/resample/format detection, supports 7 formats |
| `logger.py` | ~150 | ANSI colored console logging + plain text file logging, color-coded by level |
| `warnings_filter.py` | ~250 | Suppress TensorFlow/Keras/librosa warnings, custom stderr filter |

---

### Core AI Model: YourMT3+ YPTF.MoE+Multi (PS)

The sole transcription engine used in this project. Developed by Sungkyun Chang at KAIST (Korea Advanced Institute of Science and Technology), published at IEEE MLSP 2024.

| Item | Details |
|------|---------|
| Full Name | YPTF.MoE+Multi (PS) |
| Checkpoint | `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2` |
| Source | [KAIST - YourMT3+](https://huggingface.co/spaces/mimbres/YourMT3) ([arXiv:2407.04822](https://arxiv.org/abs/2407.04822)) |
| License | Apache 2.0 |
| Model Size | ~724 MB |
| Task Type | `mc13_full_plus_256` (34 MT3 instrument classes → 13 decoding channels, max 256 tokens/channel) |

#### Encoder: PerceiverTF + MoE

PerceiverTF is a hierarchical attention Transformer. Its core idea is to use a small set of learnable latent vectors to extract information from high-dimensional input via cross-attention, avoiding the quadratic complexity of standard Transformers on long sequences.

```
Mel Spectrogram (B, 256, 512)
    ↓
Pre-Encoder: Conv2d Res3B
    3 residual conv blocks, kernel=(3,3), frequency dim /8
    Output: (B, 256, 64, C) → reshape → (B, 256, 64*C)
    ↓
PerceiverTF Encoder
    ├── 26 learnable Latent vectors (d_latent)
    ├── 3 PerceiverTF Blocks, each containing:
    │   ├── SCA (Stochastic Cross-Attention)
    │   │   Latents as Query, spectrogram features as Key/Value
    │   │   attention_to_channel=True: attention along frequency channel dim
    │   │   sca_use_query_residual=True: query residual connection
    │   ├── 2 × Local Self-Attention
    │   │   Local self-attention among latents
    │   └── 2 × Temporal Self-Attention
    │       Global self-attention along time dimension
    ├── Position Encoding: RoPE (Rotary Position Embedding)
    │   rope_partial_pe=True: rotation applied to partial dimensions only
    ├── Normalization: RMSNorm (more efficient than LayerNorm)
    └── Feed-Forward: MoE (Mixture of Experts)
        ├── 8 expert networks (num_experts=8)
        ├── Top-2 routing (topk=2): each token activates 2 experts
        ├── Widening factor: 4 (ff_widening_factor=4)
        └── Activation: SiLU (Sigmoid Linear Unit)
```

Key advantage of MoE: only 2 of 8 experts are activated per token, so the model has 8× the parameters but only 2× the compute of a single expert. Different experts naturally specialize in different instrument families, achieving implicit instrument specialization.

#### Decoder: Multi-Channel T5

The Multi-T5 decoder maps 128 GM instruments to 13 independent decoding channels, each responsible for one instrument family:

| Channel | Instrument Family | GM Program Range |
|---------|------------------|-----------------|
| 0 | Piano | 0-7 |
| 1 | Chromatic Percussion | 8-15 |
| 2 | Organ | 16-23 |
| 3 | Guitar | 24-31 |
| 4 | Bass | 32-39 |
| 5 | Strings (+ Ensemble) | 40-55 |
| 6 | Brass | 56-63 |
| 7 | Reed | 64-71 |
| 8 | Pipe | 72-79 |
| 9 | Synth Lead | 80-87 |
| 10 | Synth Pad | 88-95 |
| 11 | Singing Voice | 100-101 |
| 12 | Drums | 128 (internal) |

Each channel performs independent autoregressive decoding with max 256 tokens. Decoding strategy is greedy (`logits.argmax(-1)`) with KV-cache acceleration.

```
Encoder Hidden States (B, T, D)
    ↓
Multi-Channel T5 Decoder
    ├── 13 independent decoding channels, shared weights
    ├── Based on T5-small architecture (google/t5-v1_1-small)
    ├── Autoregressive generation: <BOS> → token₁ → token₂ → ... → <EOS>
    ├── Each step: embed → decoder(+KV-cache) → lm_head → argmax
    └── Max length: 256 tokens/channel
        ↓
    Token sequence (B, 13, ≤256)
        ↓
    TaskManager.detokenize() → NoteEvent / TieEvent
        ↓
    merge_zipped_note_events_and_ties_to_notes()
        ↓
    mix_notes() → merge 13 channels
```

#### Training Configuration

```
Training command (published by author):
python train.py mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2 \
  -p slakh2024 -d all_cross_final -it 320000 -vit 20000 \
  -enc perceiver-tf -dec multi-t5 -nl 26 \
  -ff moe -wf 4 -nmoe 8 -kmoe 2 -act silu \
  -epe rope -rp 1 -sqr 1 -atc 1 \
  -ac spec -hop 300 -bsz 10 10 -xk 5 \
  -tk mc13_full_plus_256 \
  -edr 0.05 -ddr 0.05 -sb 1 -ps -2 2 \
  -st ddp -wb online
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Dataset | Slakh2024 + all_cross_final | Multi-dataset cross-training |
| Iterations | 320,000 | Total training steps |
| Global batch size | 80 | 10×10 (2 GPU × 5 accumulation) |
| Pitch shift | [-2, +2] semitones | Data augmentation |
| Precision | bf16-mixed | Mixed precision training |
| Sample rate | 16,000 Hz | Input audio |
| Hop length | 300 | Spectrogram frame step |
| Input frames | 32,767 (~2.05s) | Audio segment length |

#### Inference Pipeline (This Project)

```
Full audio (any length)
    ↓
Resample to 16kHz, convert to mono
    ↓
Segmentation: slice_padded_array()
    25% overlap (best mode), 32,767 frames (~2.05s) per segment
    ↓
Batch inference: inference_file(bsz=auto)
    Auto mixed precision (bf16/fp16)
    OOM auto-fallback (halve bsz and retry)
    ↓
13-channel token decoding
    ↓
Smart dedup: _deduplicate_overlapping_notes_smart()
    Group by (pitch, program, is_drum)
    Cluster-merge notes with onset diff < 10ms
    Keep the longest duration
    ↓
MIDI post-processing (by quality mode):
    best:     only remove <10ms noise notes
    balanced: light dedup + velocity smoothing + polyphony limit
    fast:     no post-processing
```

#### Benchmark (Slakh2100 Dataset)

| Metric | YPTF.MoE+Multi (PS) | MT3 (Google Baseline) |
|--------|---------------------|----------------------|
| Multi F1 | **0.7484** | 0.62 |
| Frame F1 | 0.8487 | — |
| Onset F1 | 0.8419 | — |
| Offset F1 | 0.6961 | — |
| Drum Onset F1 | 0.9113 | — |

Per-instrument Onset F1: Bass 0.93 / Piano 0.88 / Guitar 0.82 / Synth Lead 0.82 / Brass 0.73 / Strings 0.73

#### Available Model Variants

| Model | MoE | Pitch Shift | Size | Notes |
|-------|-----|-------------|------|-------|
| YPTF.MoE+Multi (PS) | 8 experts | Yes | 724 MB | **Default, highest performance** |
| YPTF.MoE+Multi (noPS) | 8 experts | No | 724 MB | Without pitch shift augmentation |
| YPTF+Multi (PS) | No | Yes | 2.0 GB | Standard Perceiver |
| YPTF+Multi (noPS) | No | No | 2.0 GB | Standard Perceiver, no augmentation |

---

### Encoder Architecture Comparison: PerceiverTF vs MusicFM

The winning solution of the 2025 AI4Musician AMT Challenge replaced PerceiverTF with a MusicFM encoder. Below is an in-depth comparison of both architectures.

#### MusicFM Encoder

MusicFM is a music foundation model proposed by Minz Won et al. (ICASSP 2024, [arXiv:2311.03318](https://arxiv.org/abs/2311.03318)), pre-trained on 160,000 hours of unlabeled music from the Million Song Dataset via self-supervised learning.

```
Audio (24kHz)
    ↓
128-band Mel Spectrogram
    ↓
2-layer Residual Conv2dSubsampling (downsample to 25Hz frame rate)
    ↓
12-layer Wav2Vec2 Conformer
    ├── Each layer: Multi-Head Self-Attention + Convolution Module
    ├── RoPE position encoding
    ├── Output dimension: 1024
    └── Frame rate: 25 Hz
    ↓
Dense music embeddings (B, T, 1024) @ 25Hz
```

Training: BEST-RQ style masked token modeling with random projection quantizer. The model learns to predict original features from masked audio segments, capturing rich musical structure information.

#### Competition Winner Full Architecture (amt-os/ai4m-miros)

```
Audio (16kHz → resample to 24kHz)
    ↓
MusicFM 25Hz Encoder (frozen, 12-layer Conformer, 1024-dim)
    ↓
MusicFMAdapter (novel component)
    ├── 13 learnable view embeddings (512-dim)
    ├── 4 recurrent iterations:
    │   concat(encoder_output, recurrent_state)
    │   → Linear → 3-layer Self-Attention (RoPE, QK-norm, SiLU)
    └── Output: (B, 13, T, 512) — 13 instrument views
    ↓
TemporalUpsample (2× ConvTranspose1d, 25Hz → 100Hz)
    ↓
Multi-Dec Decoder (Llama-style, not T5)
    ├── 13 channels, 8 layers, 8 heads
    ├── RoPE + RMSNorm + SiLU
    ├── torch.compile support
    └── Max token length: 1024
    ↓
LM Head → Token predictions
```

| Dimension | PerceiverTF + MoE (Current) | MusicFM + Multi-Dec (Winner) |
|-----------|---------------------------|------------------------------|
| Encoder type | End-to-end from scratch | Self-supervised pretrained (160K hrs) |
| Encoder arch | PerceiverTF (cross-attn + MoE) | Conformer (self-attn + conv) |
| Encoder params | Smaller (MoE sparse activation) | ~300M (full activation) |
| Output dim | ~512 | 1024 |
| Decoder | Multi-T5 (T5-small arch) | Multi-Dec (Llama-style) |
| Input window | 32,767 frames (~2.05s) | 87,381 frames (~5.46s) |
| Token length | 256 / channel | 1024 / channel |
| Frame rate | ~53 Hz | 25 Hz → 100 Hz (upsampled) |
| Inference speed | Fast (MoE sparse + short window) | Slow (full + long window) |
| VRAM requirement | Low (~2GB) | High (~6GB+) |
| Music understanding | Transcription task only | Broad musical structure |
| Dense polyphony | 256 tokens may truncate | 1024 tokens sufficient |

#### Core Difference Analysis

**PerceiverTF + MoE (Specialist Model):**
- Strengths: Extremely efficient — MoE sparse activation means 8 experts' parameters with only 2 experts' compute; short window = fast inference; VRAM-friendly
- Weaknesses: Narrow knowledge — only learns from labeled data; 2.05s window may cause boundary errors on long notes and slow pieces; 256 tokens may truncate in dense polyphonic passages

**MusicFM (Generalist Model):**
- Strengths: 160K hours of pretraining provides deep musical understanding (chord structure, timbre features, rhythmic patterns); 5.46s window drastically reduces boundary errors; 1024 tokens handles any density
- Weaknesses: Slow inference (full Conformer); high VRAM; requires additional Adapter layer to bridge encoder and decoder

**Analogy:** PerceiverTF is like a professional stenographer who only learned shorthand — efficient but narrow. MusicFM is like a conservatory graduate — deep musical literacy but needs to learn shorthand before they can do the job.

---

### AI4Musician 2025 AMT Challenge

| Item | Details |
|------|---------|
| Name | 2025 Automatic Music Transcription (AMT) Challenge |
| Organizer | AI4Musicians (Purdue University affiliated) |
| Date | April 2025 |
| Paper | "Advancing Multi-Instrument Music Transcription: Results from the 2025 AMT Challenge", NeurIPS 2025 Datasets & Benchmarks Track |
| Scale | 8 teams submitted valid solutions, 2 outperformed baseline (MT3) |
| Winner | amt-os (University of Osnabrück) |
| Repository | [amt-os/ai4m-miros](https://github.com/amt-os/ai4m-miros) |

#### Winner Model Integration Feasibility: Not Currently Viable

| Blocker | Status | Details |
|---------|--------|---------|
| Model weights | ⚠️ Needs verification | Repository includes a `checkpoints/` path; completeness, usability, and reproducibility still require manual validation |
| License | ❌ None | No LICENSE file, legally all rights reserved by default |
| Documentation | ❌ None | No README, no usage instructions |
| Performance comparison | ❓ Unknown | Competition baseline was MT3, not YourMT3+; no direct comparison data |
| Community | ⚠️ Minimal | 1 star, 0 forks |
| MusicFM pretrained weights | ✅ Available | [HuggingFace](https://huggingface.co/minzwon/MusicFM), MIT license, ~1.3GB |

The MusicFM encoder itself is available (MIT license), but an encoder alone cannot perform transcription — it also requires an Adapter + Decoder + fine-tuning on AMT datasets, which is a research-project-level effort.

---

### Frontier Models & Research Directions

| Model / Direction | Source | Type | Status | Notes |
|-------------------|--------|------|--------|-------|
| YPTF.MoE+Multi (PS) (current) | [YourMT3+ paper](https://arxiv.org/abs/2407.04822) / [KAIST HF](https://huggingface.co/spaces/mimbres/YourMT3) | Multi-instrument | ✅ In use | Slakh2100: Multi F1 0.7484 (same-protocol baseline used in this README) |
| AI4Musician 2025 winner (ai4m-miros) | [AI4Musicians 2025](https://ai4musicians.org/transcription/2025transcription.html) / [amt-os/ai4m-miros](https://github.com/amt-os/ai4m-miros) | Multi-instrument | 🔬 Paper/repo stage | Winning approach published with code; publicly reproducible, same-protocol comparison vs YourMT3+ remains limited |
| [Aria-AMT](https://github.com/EleutherAI/aria-amt) | EleutherAI | Piano | ✅ Open source | Seq-to-seq piano AMT implementation with released checkpoints |
| MusicFM encoder + AMT decoder | [MusicFM paper](https://arxiv.org/abs/2311.03318) / [MusicFM HF](https://huggingface.co/minzwon/MusicFM) | Multi-instrument (research direction) | 📄 Paper/components available | Strong pretrained encoder; full AMT still needs Adapter + Decoder + fine-tuning |
| CountEM (weakly-supervised AMT) | [arXiv:2511.14250](https://arxiv.org/abs/2511.14250) | AMT training method | 📄 Paper | Uses note histogram supervision to reduce aligned-label requirements |

| Model | Source | Type | Notes |
|-------|--------|------|-------|
| [MT3](https://github.com/magenta/mt3) | Google Magenta | Multi-instrument | Transformer enc-dec, base architecture of YourMT3+, Multi F1=0.62 (Slakh) |
| [Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) | MCT Lab | Multi-task | Piano/drums/vocal/chord transcription |
| [Basic Pitch](https://github.com/spotify/basic-pitch) | Spotify | General | Lightweight mono/polyphonic, fast inference, lower accuracy than MT3 family |

> **Trend Summary**: As of early 2026, multi-instrument AMT continues along two paths: MT3/YourMT3+ style task-specialized models and pretrained-encoder-enhanced systems. Piano AMT has stronger open-source maturity than full multi-instrument SOTA pipelines.

## Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run specific tests
pytest tests/test_yourmt3_integration.py -v

# Optional: coverage report (generates htmlcov/ and .coverage)
pytest --cov=src --cov-report=html

# Format code
black src/
isort src/

# Type checking
mypy src/
```

### Build Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Build using project spec file (recommended)
pyinstaller MusicToMidi.spec

# Build output is in dist/MusicToMidi/ directory
```

### GPU Diagnostics

```bash
# Check GPU status
python -c "from src.utils.gpu_utils import print_gpu_diagnosis; print_gpu_diagnosis()"

# Check YourMT3+ availability
python -c "from src.core.yourmt3_transcriber import YourMT3Transcriber; print(YourMT3Transcriber.is_available())"
```

## Troubleshooting

### Linux Environment Issues

**Q: Missing libGL.so.1**
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx

# CentOS/RHEL
sudo yum install mesa-libGL
```

**Q: PyQt6 won't start, shows "could not load Qt platform plugin"**
```bash
# Install Qt dependencies
sudo apt install libxcb-xinerama0 libxkbcommon-x11-0

# If running on headless server, need virtual display
sudo apt install xvfb
xvfb-run python -m src.main
```

**Q: CUDA not available**
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If returns False, reinstall correct PyTorch version
pip uninstall torch torchaudio
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

**Q: YourMT3+ not available**
```bash
# Download models
python download_sota_models.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- [YourMT3+](https://huggingface.co/spaces/mimbres/YourMT3) - Core multi-instrument transcription model
- [BS-RoFormer](https://arxiv.org/abs/2309.02612) - Vocal separation architecture paper
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) - Source separation inference framework
- [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) - BS-RoFormer pretrained weights
- [mido](https://github.com/mido/mido) - MIDI file handling
- [librosa](https://librosa.org/) - Audio analysis

## Support

If you encounter any issues, please [open an issue](https://github.com/mason369/music-to-midi/issues).
