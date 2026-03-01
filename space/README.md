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

# 🎵 Music to MIDI

将音频文件智能转换为多轨道 MIDI，基于 **YourMT3+ MoE** 深度学习模型。

## 功能

- **多乐器转写**：直接识别 128 种 GM 乐器，自动分配轨道
- **人声分离模式**：先用 BS-RoFormer 分离人声与伴奏，再分别转写为独立 MIDI
- **节拍检测**：自动检测 BPM 并嵌入 MIDI
- **支持多种格式**：MP3, WAV, FLAC, OGG, M4A

## 使用方法

1. 上传音频文件
2. 选择处理模式和转写质量
3. 点击"开始转换"
4. 转换完成后下载 MIDI 文件

## 技术栈

- **转写引擎**: [YourMT3+](https://github.com/mimbres/YourMT3) MoE (Mixture of Experts)
- **人声分离**: [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) BS-RoFormer（默认 checkpoint: `model_bs_roformer_ep_317_sdr_12.9755.ckpt`）
- **节拍检测**: librosa
- **Web 框架**: Gradio

## 链接

- [GitHub 仓库](https://github.com/mason369/music-to-midi)
- [桌面版 (PyQt6)](https://github.com/mason369/music-to-midi/releases)
