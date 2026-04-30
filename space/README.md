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
short_description: 将音乐智能转换为多轨道 MIDI 文件
---

# Music to MIDI

将音频文件转换为 MIDI。Space 版与桌面版保持一致，只提供完整混音转写和人声/伴奏分离转写。

## 功能

- **YourMT3+ 多乐器转写**：直接识别 128 种 GM 乐器并输出 MIDI。
- **人声分离 + 分别转写**：先分离人声与伴奏，再分别转写；可选生成合并 MIDI。
- **节拍检测**：自动检测 BPM 并写入 MIDI。
- **多格式输入**：支持 MP3、WAV、FLAC、OGG、M4A。

## 使用方法

1. 上传音频文件。
2. 选择处理模式和转写质量。
3. 点击“开始转换”。
4. 转换完成后下载 MIDI 或分离音频文件。

## 技术栈

- **转写引擎**: YourMT3+ MoE
- **人声/伴奏分离**: `audio-separator`
- **节拍检测**: librosa
- **Web 框架**: Gradio

## 链接

- [GitHub 仓库](https://github.com/mason369/music-to-midi)
- [桌面版](https://github.com/mason369/music-to-midi/releases)
