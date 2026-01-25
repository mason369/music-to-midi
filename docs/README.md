# Music to MIDI Converter

<p align="center">
  <a href="./README_zh.md">中文</a> | English
</p>

Convert audio files to multi-track MIDI with automatic lyrics embedding. Supports 128 GM instrument recognition.

## Features

- **Dual Processing Modes**:
  - **Piano Mode**: Skip source separation, use ByteDance professional piano model to convert audio to multi-track piano MIDI (ideal for solo piano pieces)
  - **Smart Mode**: Use YourMT3+ MoE model for direct multi-instrument recognition, supporting 128 GM instruments
- **Source Separation**: Automatically separate audio into 6 tracks (vocals, drums, bass, guitar, piano, other) using Demucs v4
- **Instrument Recognition**: Smart instrument detection and classification using PANNs
- **Multi-Instrument Transcription**:
  - **YourMT3+ MoE** (2025 AMT Challenge SOTA): Hierarchical attention Transformer + Mixture of Experts, supports 128 GM instruments
  - **Basic Pitch** (Spotify): Polyphonic pitch detection as fallback
  - **ByteDance Piano Transcription**: Professional piano transcription with pedal detection
- **MIDI Post-processing**: Note quantization, velocity smoothing, deduplication, polyphony limiting
- **Lyrics Recognition**: Recognize lyrics from vocals and embed them into MIDI with word-level timestamps
- **Multi-language UI**: Support for English and Chinese interface
- **Professional Dark Theme**: Modern audio software-style interface design

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Windows | ✅ Supported | Full functionality, CUDA recommended |
| Linux | ✅ Supported | Full functionality, Ubuntu 22.04+ recommended |
| macOS | 🚧 Planned | Apple Silicon MPS support in development |

## Screenshots

Coming soon...

## Installation

### Prerequisites

- **Python 3.10+** (3.10 or 3.11 recommended, 3.12 may have compatibility issues)
- **FFmpeg**: Required for audio processing
  - Windows: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Linux: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo dnf install ffmpeg` (Fedora)
  - macOS: `brew install ffmpeg`
- **NVIDIA GPU + CUDA** (recommended): For significantly faster processing

### Dependency Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| PyTorch | 2.1.0 - 2.4.x | pyannote.audio compatibility |
| torchaudio | 2.1.0 - 2.4.x | Must match PyTorch version |
| NumPy | < 2.0 | numba/JAX compatibility |
| CUDA | 11.8 or 12.1 | GPU acceleration (optional) |

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

# 5. Install YourMT3+ code (optional, for 128 instrument recognition)
git clone https://github.com/mimbres/YourMT3.git
# Or run the install script
bash install_yourmt3_code.sh

# 6. Download YourMT3+ models (optional)
python download_sota_models.py

# 7. Run the application
python -m src.main
```

### Windows Installation

```bash
# Clone the repository
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch (CUDA 11.8)
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

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
- MIDI (.mid) - Multi-track MIDI with lyrics embedded
- LRC (.lrc) - Synchronized lyrics file
- WAV - Separated audio tracks

## Technical Details

### AI Models Used

| Model | Source | Purpose | Description |
|-------|--------|---------|-------------|
| **YourMT3+ MoE** | KAIST | Multi-instrument transcription | 2025 AMT Challenge SOTA, Mixture of Experts, supports 128 GM instruments |
| **Demucs v4** | Meta | Source separation | State-of-the-art music source separation, supports 4/6 track modes |
| **PANNs** | KAIST | Instrument recognition | Audio pattern analysis and classification |
| **Basic Pitch** | Spotify | Polyphonic pitch detection | Lightweight audio-to-MIDI |
| **Piano Transcription** | ByteDance | Piano transcription | Professional piano transcription with pedal detection |
| **Whisper + WhisperX** | OpenAI | Speech recognition | Lyrics recognition with word-level alignment |

### Processing Modes

| Mode | Description | Use Case | Output Tracks |
|------|-------------|----------|---------------|
| Piano Mode | Skip separation, generate multi-track piano MIDI | Solo piano pieces, simple melodies | 1-6 tracks (auto-detected) |
| Smart Mode (Standard) | 6-track separation + instrument recognition | Full arrangements | Up to 6 tracks |
| Smart Mode (Precise) | YourMT3+ direct transcription | Complex multi-instrument works | Up to 128 GM instruments |

### Architecture

```
Audio Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Mode Selection                                       │
│  ├─ Piano Mode ──→ ByteDance Piano Transcription    │
│  └─ Smart Mode ──→ YourMT3+ MoE or Demucs+Basic Pitch│
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Smart Mode Processing Pipeline                       │
│  ├──→ YourMT3+ MoE (preferred): Direct multi-instrument│
│  │    └── Supports 128 GM instruments               │
│  ├──→ Fallback: Demucs 6-track + Basic Pitch        │
│  ├──→ Beat Detection (librosa)                      │
│  └──→ Lyrics Recognition (Whisper + WhisperX)       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ MIDI Post-processing                                 │
│  ├──→ Note Quantization (optional)                  │
│  ├──→ Velocity Smoothing                            │
│  ├──→ Smart Deduplication (handles overlap segments)│
│  └──→ Polyphony Limiting                            │
└─────────────────────────────────────────────────────┘
    │
    ▼
Output: MIDI + LRC + WAV
```

## Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run specific tests
pytest tests/test_yourmt3_integration.py -v

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
# Ensure YourMT3 code exists
ls YourMT3/

# If not exists, clone the repository
git clone https://github.com/mimbres/YourMT3.git

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

- [YourMT3+](https://github.com/mimbres/YourMT3) - 2025 AMT Challenge SOTA multi-instrument transcription
- [Demucs](https://github.com/facebookresearch/demucs) - Music source separation
- [PANNs](https://github.com/qiuqiangkong/panns_inference) - Audio pattern analysis and instrument recognition
- [Basic Pitch](https://github.com/spotify/basic-pitch) - Audio to MIDI transcription
- [Piano Transcription](https://github.com/bytedance/piano_transcription) - ByteDance piano transcription
- [Whisper](https://github.com/openai/whisper) - Speech recognition
- [WhisperX](https://github.com/m-bain/whisperX) - Word-level alignment
- [mido](https://github.com/mido/mido) - MIDI file handling

## Support

If you encounter any issues, please [open an issue](https://github.com/mason369/music-to-midi/issues).
