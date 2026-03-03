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
- **六声部分离模式**：BS-RoFormer SW 输出 bass/drums/guitar/piano/vocals/other，再分别转写并合并
- **主唱/和声（实验近似）**：六声部分离下可将 vocals 继续拆分为 lead+harmony proxy
- **钢琴专用模式**：Aria-AMT 钢琴模型专门转写钢琴曲为钢琴 MIDI
- **节拍检测**：自动检测 BPM 并嵌入 MIDI
- **支持多种格式**：MP3, WAV, FLAC, OGG, M4A

## 使用方法

1. 上传音频文件
2. 选择处理模式和转写质量
3. 点击"开始转换"
4. 转换完成后下载 MIDI 文件

## 技术栈

- **转写引擎**: [YourMT3+](https://github.com/mimbres/YourMT3) MoE (Mixture of Experts)
- **钢琴转写**: [Aria-AMT](https://github.com/EleutherAI/aria-amt) (`piano-medium-double-1.0.safetensors`)
- **人声分离**: [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) BS-RoFormer（默认 checkpoint: `model_bs_roformer_ep_368_sdr_12.9628.ckpt`）
- **六声部分离**: BS-RoFormer SW (`BS-Rofo-SW-Fixed.ckpt`)
- **节拍检测**: librosa
- **Web 框架**: Gradio

## 链接

- [GitHub 仓库](https://github.com/mason369/music-to-midi)
- [桌面版 (PyQt6)](https://github.com/mason369/music-to-midi/releases)
