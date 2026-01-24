# 音乐转MIDI转换器

<p align="center">
  中文 | <a href="./README.md">English</a>
</p>

将音频文件转换为多轨道MIDI，自动嵌入歌词。

## 功能特点

- **双模式处理**：
  - **钢琴模式**：跳过音源分离，直接将音频转换为多轨钢琴MIDI（适合纯钢琴曲）
  - **智能模式**：自动检测乐器类型，分离并转换为对应乐器的MIDI轨道
- **音源分离**：使用Demucs v4自动将音频分离为6个轨道（人声、鼓、贝斯、吉他、钢琴、其他）
- **乐器识别**：使用PANNs进行智能乐器检测和分类
- **音频转MIDI**：使用AI驱动的音高检测（Basic Pitch）将每个轨道转换为MIDI
- **MIDI后处理**：音符量化、力度平滑、去重、复音限制等优化
- **歌词识别**：识别人声中的歌词，并以单词级时间戳嵌入MIDI
- **多语言界面**：支持中文和英文界面切换
- **专业深色主题**：现代化音频软件风格界面设计

## 平台支持

| 平台 | 状态 | 说明 |
|------|------|------|
| Windows | ✅ 已支持 | 完整功能 |
| macOS | 🚧 计划中 | 开发中 |
| Linux | 🚧 计划中 | 开发中 |

## 截图

即将推出...

## 安装

### 前置要求

- **Python 3.10+**
- **FFmpeg**：音频处理必需
  - Windows: `choco install ffmpeg` 或从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- **NVIDIA GPU**（推荐）：使用CUDA加速处理

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows上: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行应用
python -m src.main
```

### 从发布版安装

从 [Releases](https://github.com/mason369/music-to-midi/releases) 页面下载Windows版本。

## 使用方法

1. **打开音频文件**：拖放音频文件（MP3、WAV、FLAC、OGG）或点击浏览选择
2. **配置输出**：选择输出目录和选项（MIDI、歌词、分离音轨）
3. **开始处理**：点击"开始"按钮开始转换
4. **获取结果**：在输出目录中找到MIDI文件、LRC歌词和分离的音轨

## 支持的格式

### 输入
- MP3, WAV, FLAC, OGG, M4A, AAC, WMA

### 输出
- MIDI (.mid) - 嵌入歌词的多轨道MIDI
- LRC (.lrc) - 同步歌词文件
- WAV - 分离的音频轨道

## 技术细节

### 使用的AI模型
- **Demucs v4** (Meta)：最先进的音源分离（支持4轨和6轨模式）
- **PANNs** (Audio Pattern Analysis)：乐器识别和音频分类
- **Basic Pitch** (Spotify)：多音高检测
- **Whisper + WhisperX** (OpenAI)：带单词级对齐的语音识别

### 处理模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| 钢琴模式 | 跳过分离，生成多轨钢琴MIDI | 纯钢琴曲、简单旋律 |
| 智能模式 | 6轨分离 + 乐器识别 | 完整编曲、多乐器作品 |

### 架构

```
音频输入
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 模式选择                                              │
│  ├─ 钢琴模式 ──→ 跳过分离，直接转写                    │
│  └─ 智能模式 ──→ 6轨分离 (Demucs htdemucs_6s)         │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 智能模式处理流程                                       │
│  ├──→ 乐器识别 (PANNs) ──→ 轨道布局建议                │
│  ├──→ 节拍检测 (librosa)                              │
│  ├──→ 音频转MIDI (Basic Pitch)                        │
│  └──→ 歌词识别 (Whisper) ──→ 单词对齐 (WhisperX)       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ MIDI后处理                                            │
│  ├──→ 音符量化                                        │
│  ├──→ 力度平滑                                        │
│  ├──→ 重复音符去除                                    │
│  └──→ 复音数限制                                      │
└─────────────────────────────────────────────────────┘
    │
    ▼
输出: MIDI + LRC + WAV
```

## 开发

### 设置开发环境

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest

# 格式化代码
black src/
isort src/

# 类型检查
mypy src/
```

### 构建可执行文件

```bash
# 安装PyInstaller
pip install pyinstaller

# 使用项目配置文件构建（推荐）
pyinstaller MusicToMidi.spec

# 构建产物在 dist/MusicToMidi/ 目录下
```

## 贡献

欢迎贡献！请随时提交Pull Request。

1. Fork本仓库
2. 创建功能分支（`git checkout -b feature/amazing-feature`）
3. 提交更改（`git commit -m 'Add amazing feature'`）
4. 推送到分支（`git push origin feature/amazing-feature`）
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](../LICENSE) 文件。

## 致谢

- [Demucs](https://github.com/facebookresearch/demucs) - 音乐源分离
- [PANNs](https://github.com/qiuqiangkong/panns_inference) - 音频模式分析与乐器识别
- [Basic Pitch](https://github.com/spotify/basic-pitch) - 音频转MIDI转录
- [Whisper](https://github.com/openai/whisper) - 语音识别
- [WhisperX](https://github.com/m-bain/whisperX) - 单词级对齐
- [mido](https://github.com/mido/mido) - MIDI文件处理

## 支持

如果您遇到任何问题，请 [创建issue](https://github.com/mason369/music-to-midi/issues)。
