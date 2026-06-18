---
title: Music to MIDI
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
suggested_hardware: zero-a10g
short_description: AI 音频转 MIDI，支持多乐器、人声分离、六声部分离和钢琴转写
tags:
  - audio-to-midi
  - midi
  - music-transcription
  - ai-music
  - stem-separation
  - piano-transcription
---

# Music to MIDI - AI Audio to MIDI

将 `MP3`、`WAV`、`FLAC`、`OGG`、`M4A` 音频转换为可编辑 MIDI。Space 版与桌面版、Colab 版保持一致，提供六种处理模式：完整混音转写、人声/伴奏分离转写、六声部分离转写、Transkun 钢琴专用转写、Aria-AMT 钢琴专用转写和 ByteDance Pedal 带踏板钢琴转写。

它适合扒谱、编曲草稿、钢琴录音转 MIDI、歌曲 stem 拆解，以及想快速试用 AI music transcription 的用户。

## 功能

- **YourMT3+ 多乐器转写**：直接识别 128 种 GM 乐器并输出 MIDI，适合完整混音和纯音乐片段。
- **人声分离 + 分别转写**：先分离人声与伴奏，再分别转写；可选生成合并 MIDI，方便继续进 DAW 整理。
- **六声部分离 + 分别转写**：分离 bass / drums / guitar / piano / vocals / other 六个 stem，再输出 stem MIDI 与合并 MIDI。
- **钢琴专用转写 (Transkun)**：使用 Transkun 处理纯钢琴音频。
- **钢琴专用转写 (Aria-AMT)**：使用 Aria-AMT 处理纯钢琴音频，首次使用会检查 checkpoint。
- **钢琴专用转写 (ByteDance Pedal)**：使用 ByteDance 带踏板钢琴模型处理纯钢琴音频，保留延音踏板 CC64。
- **节拍检测**：自动检测 BPM 并写入 MIDI。
- **多格式输入**：支持 MP3、WAV、FLAC、OGG、M4A。

## 使用方法

1. 上传音频文件。
2. 选择处理模式。
3. 点击“开始转换”。
4. 转换完成后下载 MIDI 或分离音频文件。

## 技术栈

- **转写引擎**: YourMT3+、Transkun、Aria-AMT、ByteDance Pedal
- **人声/伴奏/六声部分离**: `audio-separator`
- **节拍检测**: librosa
- **Web 框架**: Gradio

## 链接

- [GitHub 仓库](https://github.com/mason369/music-to-midi)
- [桌面版](https://github.com/mason369/music-to-midi/releases)
