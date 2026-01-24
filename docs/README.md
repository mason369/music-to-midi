# Music to MIDI Converter

<p align="center">
  <a href="./README_zh.md">中文</a> | English
</p>

Convert audio files to multi-track MIDI with automatic lyrics embedding.

## Features

- **Source Separation**: Automatically separate audio into 4 tracks (vocals, drums, bass, other) using Demucs v4
- **Audio to MIDI**: Convert each track to MIDI using AI-powered pitch detection (Basic Pitch)
- **Lyrics Recognition**: Recognize lyrics from vocals and embed them into MIDI with word-level timestamps
- **Multi-language UI**: Support for English and Chinese interface
- **Cross-platform**: Windows, macOS, and Linux support

## Screenshots

Coming soon...

## Installation

### Prerequisites

- **Python 3.10+**
- **FFmpeg**: Required for audio processing
  - Windows: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- **NVIDIA GPU** (recommended): For faster processing with CUDA

### Install from Source

```bash
# Clone the repository
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m src.main
```

### Install from Release

Download the latest release for your platform from the [Releases](https://github.com/mason369/music-to-midi/releases) page.

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
- **Demucs v4** (Meta): State-of-the-art source separation
- **Basic Pitch** (Spotify): Polyphonic pitch detection
- **Whisper + WhisperX** (OpenAI): Speech recognition with word-level alignment

### Architecture

```
Audio Input
    │
    ▼
Source Separation (Demucs) ──→ 4 tracks (vocals/drums/bass/other)
    │
    ├──→ Beat Detection (librosa)
    │
    ├──→ Audio to MIDI (Basic Pitch)
    │
    └──→ Lyrics Recognition (Whisper) ──→ Word Alignment (WhisperX)
                                              │
                                              ▼
                                    MIDI Generation (mido)
                                              │
                                              ▼
                                    Output: MIDI + LRC + WAV
```

## Configuration

Settings are stored in `~/.music-to-midi/config.yaml`:

```yaml
# General
language: zh_CN  # or en_US
theme: dark

# Processing
use_gpu: true
whisper_model: medium  # tiny, base, small, medium, large

# MIDI
ticks_per_beat: 480
default_velocity: 80
```

## Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

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

# Build
pyinstaller music-to-midi.spec
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

- [Demucs](https://github.com/facebookresearch/demucs) - Music source separation
- [Basic Pitch](https://github.com/spotify/basic-pitch) - Audio to MIDI transcription
- [Whisper](https://github.com/openai/whisper) - Speech recognition
- [WhisperX](https://github.com/m-bain/whisperX) - Word-level alignment
- [mido](https://github.com/mido/mido) - MIDI file handling

## Support

If you encounter any issues, please [open an issue](https://github.com/mason369/music-to-midi/issues).
