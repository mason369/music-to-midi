# 音乐转MIDI转换器

<p align="center">
  中文 | <a href="./docs/README.md">English</a>
</p>

将音频文件转换为多轨道MIDI，支持 128 种 GM 乐器精确识别。

**平台支持：Windows / Linux / WSL2**

## 截图演示

| Windows | Linux |
|---------|-------|
| ![Windows 演示](resources/icons/Windows演示.png) | ![Linux 演示](resources/icons/Linux演示.png) |

## 功能特点

- **多乐器转写**：使用 YourMT3+ MoE 高性能模型直接识别混音中的多种乐器
- **人声分离模式**：BS-RoFormer 分离人声与伴奏，默认输出两个独立 MIDI（可选额外输出 1 个合并 MIDI，默认 checkpoint：`model_bs_roformer_ep_368_sdr_12.9628.ckpt`）
- **六声部分离模式**：BS-RoFormer SW 分离 `bass/drums/guitar/piano/vocals/other`，默认输出 6 个 stem MIDI + 1 个合并 MIDI（可切换仅转写选中 stem，输出 N 个 stem MIDI）
- **主唱/和声（实验近似）**：六声部分离模式可将 `vocals` 再分离为主唱+和声代理 stem（基于公开 male/female 模型）
- **钢琴专用转写模式**：Aria-AMT 针对钢琴场景输出专用钢琴 MIDI（本地可下载 checkpoint）
- **128 种 GM 乐器**：输出标准 General MIDI 多轨道 MIDI，精确区分鼓、贝斯、吉他、钢琴等
- **MIDI 后处理**：音符量化、力度平滑、去重、复音限制等优化
- **GPU 加速**：自动检测并使用 CUDA（NVIDIA）/ ROCm（AMD）/ CPU
- **多语言界面**：支持中文和英文界面切换
- **深色主题**：现代化音频软件风格界面

## 平台支持

| 平台 | 状态 | 说明 |
|------|------|------|
| Windows 10/11 (x64) | ✅ 已支持 | 双击 `run.bat` 一键启动 |
| Linux (Ubuntu/Debian) | ✅ 已支持 | 推荐 Ubuntu 22.04+，完整功能 |
| WSL2 (Windows 11) | ✅ 已支持 | 需要 WSLg（Win11 内置）|
| macOS | 🚧 计划中 | Apple Silicon MPS 支持开发中 |

## 快速开始

### Windows

```
1. 克隆或下载仓库
2. 双击 run.bat（首次自动安装所有依赖）
```

或使用 PowerShell：
```powershell
powershell -ExecutionPolicy Bypass -File run.ps1
```

### Linux / WSL2

```bash
# 1. 克隆仓库
git clone https://github.com/mason369/music-to-midi.git
cd music-to-midi

# 2. 直接运行（首次自动安装所有依赖）
./run.sh
```

启动脚本会自动检测依赖完整性，首次运行或依赖缺失时自动完成：
- 安装系统包（FFmpeg、PyQt6 所需库、中文/Emoji 字体等）
- 创建 Python 虚拟环境
- 安装 PyTorch（根据 GPU 自动选择 CUDA/ROCm/CPU 版本）
- 安装所有 Python 依赖
- 安装 Aria-AMT（钢琴专用转写，可选）
- 下载 YPTF.MoE+Multi (PS) 模型权重（约 800MB）
- 下载 BS-RoFormer ep368 人声分离模型（约 600MB）
- 下载 BS-RoFormer SW 六声部分离资源（约 500MB）
- 下载 Aria-AMT 钢琴 checkpoint（约 426MB）

## 手动安装

如需手动安装，请按以下步骤操作：

### 1. 系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libgl1 \
    fonts-ubuntu \
    fonts-dejavu-core \
    fonts-noto-cjk \
    fonts-noto-color-emoji
```

### 2. Python 环境

```bash
# 创建虚拟环境（推荐 Python 3.10 或 3.11）
python3.10 -m venv venv
source venv/bin/activate

# 安装 PyTorch（按 GPU 选择）
# CUDA 12.1
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 仅 CPU（速度较慢）
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# 安装项目依赖
pip install -r requirements.txt
```

### 3. 下载模型权重

```bash
# 下载 YourMT3+ 转写模型（约 800MB）
python download_sota_models.py

# 下载 BS-RoFormer ep368（人声/伴奏）
python download_vocal_model.py

# 下载 BS-RoFormer SW（六声部分离）
python download_multistem_model.py

# 下载 Aria-AMT（钢琴专用转写）
python download_aria_amt_model.py
```

### 4. 运行

```bash
source venv/bin/activate
python -m src.main
```

## WSL2 配置说明

WSL2 下运行 GUI 需要 WSLg（Windows 11 内置），环境变量会自动设置：

```bash
# 如果 DISPLAY 未自动设置，添加到 ~/.bashrc
export DISPLAY=:0
```

验证显示环境：

```bash
echo $DISPLAY        # 应显示 :0
echo $WAYLAND_DISPLAY  # 应显示 wayland-0（WSLg 环境）
```

如果 WSLg 不可用（Windows 10），可安装 VcXsrv：
1. 在 Windows 上安装 [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
2. 启动 XLaunch（勾选 "Disable access control"）
3. 在 WSL 中：`export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0`

## 依赖版本说明

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| PyTorch | 2.1.0 - 2.4.x | YourMT3+ 兼容性要求 |
| torchaudio | 2.1.0 - 2.4.x | 与 PyTorch 版本对应 |
| NumPy | < 2.0 | numba 兼容性 |
| CUDA | 11.8 或 12.1 | GPU 加速（可选） |
| Python | 3.10+ | 推荐 3.10 或 3.11 |

## 使用方法

1. **打开音频文件**：拖放音频文件（MP3、WAV、FLAC、OGG 等）或点击浏览选择
2. **选择处理模式**：SMART / 人声分离 / 六声部分离 / 钢琴专用(Aria-AMT)
3. **可选（人声分离）**：勾选“输出 1 个人声+伴奏合并 MIDI”
4. **可选（六声部分离）**：勾选“仅转写选中的 stem”，并勾选要转写的 stem（如仅鼓/仅贝斯）
5. **可选（六声部分离）**：勾选“将 vocals 进一步分离为主唱 + 和声（实验近似）”
6. **配置输出**：选择输出目录
7. **开始处理**：点击"开始"按钮
8. **获取结果**：在输出目录中找到 MIDI 与分离音频文件

## 支持的格式

**输入**：MP3, WAV, FLAC, OGG, M4A, AAC, WMA

**输出**：MIDI (.mid)

## 技术架构

```
音频输入 → MusicToMidiPipeline
              ↓
          ┌─ SMART 模式（默认）──────────────────────────┐
          │  YourMT3+ MoE 直接对完整混音进行多乐器转写    │
          └──────────────────────────────────────────────┘
              ↓
          ┌─ VOCAL_SPLIT 模式 ──────────────────────────┐
          │  BS-RoFormer 分离人声与伴奏                   │
          │      ↓                                      │
          │  YourMT3+ 分别转写 → 两个独立 MIDI（可选 +1 合并） │
          └──────────────────────────────────────────────┘
              ↓
          ┌─ SIX_STEM_SPLIT 模式 ───────────────────────┐
          │  BS-RoFormer SW 分离 6 个 stem               │
          │  (bass/drums/guitar/piano/vocals/other)      │
          │      ↓                                      │
          │  可选：vocals → 主唱/和声（实验近似）         │
          │      ↓                                      │
          │  YourMT3+ 分别转写 → N 个 MIDI + 1 合并 MIDI │
          └──────────────────────────────────────────────┘
              ↓
          ┌─ PIANO_ARIA_AMT 模式 ───────────────────────┐
          │  Aria-AMT 钢琴专用模型                        │
          │      ↓                                      │
          │  钢琴 MIDI 输出（单文件）                     │
          └──────────────────────────────────────────────┘
              ↓
          MIDI 后处理（量化 / 去重 / 复音限制）
              ↓
          多轨道 MIDI 输出（最多 128 种 GM 乐器）
```

### 使用的 AI 模型

本项目使用 **YPTF.MoE+Multi (PS)** — YourMT3+ 系列中的最高性能变体。

| 项目 | 详情 |
|------|------|
| 模型全称 | YPTF.MoE+Multi (PS) |
| 检查点 | `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2` |
| 来源 | [KAIST - YourMT3+](https://huggingface.co/spaces/mimbres/YourMT3)（[arXiv:2407.04822](https://arxiv.org/abs/2407.04822)） |
| 架构 | Perceiver Transformer 编码器 + Multi-T5 解码器 |
| MoE | 8 专家, Top-2 路由, SiLU 激活 |
| 位置编码 | RoPE（部分旋转位置编码） |
| 归一化 | RMSNorm |
| 训练增强 | Pitch Shift 音高偏移增强（PS） |
| 模型大小 | ~724 MB |
| 任务类型 | `mt3_full_plus`（128 种 GM 乐器 + 鼓） |

#### 性能基准（Slakh2100 数据集）

| 指标 | YPTF.MoE+Multi (PS) | MT3 (Google 基线) |
|------|---------------------|-------------------|
| Multi F1 | **0.7484** | 0.62 |
| Frame F1 | 0.8487 | — |
| Onset F1 | 0.8419 | — |
| Offset F1 | 0.6961 | — |
| Drum Onset F1 | 0.9113 | — |

各乐器 Onset F1：Bass 0.93 / Piano 0.88 / Guitar 0.82 / Synth Lead 0.82 / Brass 0.73 / Strings 0.73

#### 仓库内可用模型变体

| 模型 | MoE | Pitch Shift | 大小 | 说明 |
|------|-----|-------------|------|------|
| YPTF.MoE+Multi (PS) | ✅ 8专家 | ✅ | 724 MB | **默认，最高性能** |
| YPTF.MoE+Multi (noPS) | ✅ 8专家 | ❌ | 724 MB | 无音高偏移增强 |
| YPTF+Multi (PS) | ❌ | ✅ | 2.0 GB | 标准 Perceiver，较轻量 |
| YPTF+Multi (noPS) | ❌ | ❌ | 2.0 GB | 标准 Perceiver，无增强 |
| YourMT3+ 传统版 | ❌ | ❌ | 2.0 GB | 旧版兼容 |

### 人声分离模型：BS-RoFormer

VOCAL_SPLIT 模式使用 **BS-RoFormer**（Band-Split Rotary Transformer）进行人声/伴奏分离。

| 项目 | 详情 |
|------|------|
| 模型全称 | Band-Split RoFormer |
| 论文 | [Music Source Separation with Band-Split RoFormer](https://arxiv.org/abs/2309.02612) (ISMIR 2023 Workshop) |
| 检查点 | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` (epoch 368) |
| 训练者 | [ZFTurbo](https://github.com/ZFTurbo) / [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) |
| 许可证 | MIT |
| 指标说明 | checkpoint 文件名包含 `sdr_12.9628`（训练过程分数标签，不是统一评测口径） |
| 公开对比（Multisong） | BS-RoFormer(ep317/viperx): Vocals 10.87, Instrum 17.17；MelBand-RoFormer(Kim): Vocals 10.98（公开列表）/ 11.01, Instrum 17.32（MVSEP）；HTDemucs_ft: Vocals 8.38 |
| 模型大小 | ~500 MB |
| 调用方式 | 通过 [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 库封装 |
| 首次使用 | 自动从 HuggingFace 下载到 `~/.music-to-midi/models/audio-separator/` |
| 输出选项 | 默认 2 个 MIDI（伴奏 + 人声）；可勾选额外输出 1 个合并 MIDI |

核心思想：将频谱按频段（band）拆分后独立建模，用 RoPE 增强时序建模，通过 Band-level 和 Temporal Self-Attention 交替捕获跨频段谐波关系和时序依赖。

### 六声部分离模型：BS-RoFormer SW（新功能主后端）

`SIX_STEM_SPLIT` 模式默认使用 **BS-RoFormer SW** 作为主分离后端，输出 `bass/drums/guitar/piano/vocals/other` 六个 stem。

| 项目 | 详情 |
|------|------|
| 模型文件 | `BS-Rofo-SW-Fixed.ckpt` |
| 配置文件 | `config_bs_roformer_sw.yaml` |
| 下载脚本 | `download_multistem_model.py` |
| 缓存目录 | `~/.music-to-midi/models/audio-separator/` |
| 指定转写开关 | 可勾选“仅转写选中的 stem（六声部模式）”，只对选中 stem 跑 YourMT3+ |
| 输出结果 | 6 个 WAV + N 个 stem MIDI + 1 个合并 MIDI（默认 N=6） |
| 回退策略 | BS-RoFormer SW 加载失败时自动回退 `htdemucs_6s` |

#### 为什么选择 BS-RoFormer（替代 Demucs）

| 维度 | Demucs (htdemucs) | BS-RoFormer |
|------|-------------------|-------------|
| 公开对比（Multisong 人声/伴奏 SDR） | 8.38 / N/A (HTDemucs_ft, vocals-only) | 10.87 / 17.17 (BS-RoFormer_ep_317, viperx) |
| 架构 | 混合 U-Net + Transformer | 纯 Transformer (频段拆分) |
| 集成方式 | 需手动管理模型加载 | audio-separator 三步调用 |
| 依赖链 | 较重 | 轻量（audio-separator + onnxruntime） |

#### 前沿人声分离模型（2026-03-02 核实）

> 注：本表中 `a / b` 统一表示 **Vocals SDR（人声）/ Instrum SDR（伴奏）**，单位 dB。  
> 若写明“仅 Vocals”，表示该来源未给出伴奏口径。  
> 若标注“API 调用（权重未公开）”，表示可在线调用，但权重未提供公开下载链接（不属于本地开源可直替）。

| 模型/方向 | 来源 | 类型 | 状态 | 说明 |
|-----------|------|------|------|------|
| BS-RoFormer ep368（当前） | [TRvlvr download_checks](https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json) / [TRvlvr model_repo](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models) / [audio-separator models-scores](https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/models-scores.json) | 本地直替（audio-separator） | ✅ 使用中 | 公开可下载；audio-separator 内置中位分数约 **12.10（人声）/ 16.31（伴奏）**；当前工程默认 checkpoint：`model_bs_roformer_ep_368_sdr_12.9628.ckpt` |
| BS-RoFormer ep317（公开可下载） | [MVSEP News (2024-03-29)](https://www.mvsep.com/news/41) / [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) | 本地直替（audio-separator） | ✅ 可替换（权衡） | Multisong（vocal models 表）：**10.87（仅人声）**；在 audio-separator 内置口径下相对 ep368 为“人声回落、伴奏略升” |
| MelBand-RoFormer (KimberleyJensen) | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [Hugging Face](https://huggingface.co/KimberleyJSN/melbandroformer) / [MVSEP Full API](https://mvsep.ru/full_api) | 本地可用（vocals/other） | ✅ 可用（偏人声） | 公开权重 `MelBandRoformer.ckpt`；Multisong（公开列表）为 **10.98（仅人声）**。MVSEP API 另列 11.01/17.32 条目，但未给出与公开 checkpoint 的一一映射说明 |
| SCNet XL IHF（开源权重） | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [ZFTurbo Release v1.0.15](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.15) | 开源可下载（4-stem） | ⚠️ 需改造接入 | 公开权重是 4-stem 模型（`model_scnet_ep_36_sdr_10.0891.ckpt`）；同表 Multisong vocals 为 9.68，不是本项目现有 2-stem 直替 |
| SCNet XL IHF（MVSEP） | [MVSEP Full API](https://mvsep.ru/full_api) | API 调用（权重映射未公开） | 🌐 可用（非本地直替） | MVSEP 页面列出 **11.11（人声）/ 17.41（伴奏）**；但公开下载清单未见可直接对应的 checkpoint 名称 |
| Ensemble (vocals,instrum) | [MVSEP News](https://www.mvsep.com/news) | API 调用（权重未公开） | 🌐 可用（非本地直替） | Multisong：**11.93（人声）/ 18.23（伴奏）**（ver 2025.06） |
| BS Roformer (vocals,instrum) | [MVSEP News](https://www.mvsep.com/news) / [MVSEP Full API](https://mvsep.ru/full_api) | API 调用（权重未公开） | 🌐 可用（非本地直替） | Multisong：**11.89（人声）/ 18.20（伴奏）**（ver 2025.07）；当前公开下载列表未见该版本 checkpoint |
| BS Roformer SW（6-stem） | [python-audio-separator model-configs](https://github.com/nomadkaraoke/python-audio-separator/releases/tag/model-configs) / [MVSEP News](https://www.mvsep.com/news) | 开源可下载（6-stem） + API | ✅ 已接入（六声部模式） | 本项目已将其接入 `SIX_STEM_SPLIT` 主后端，输出六路音频并分别转写 |
| 主唱/和声（实验近似） | [audio-separator models-scores](https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/models-scores.json) | 开源可下载（male/female） | ✅ 已接入（实验） | 可下载 checkpoint：`model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt`；该模型公开标签为 male/female，不是严格意义 lead/back benchmark |
| Mel-RoFormer (ISMIR 2024) | [arXiv:2409.04702](https://arxiv.org/abs/2409.04702) / [ar5iv 表2](https://ar5iv.org/html/2409.04702v1) | 论文阶段（研究模型） | 📄 论文已发表 | MUSDB18-HQ（表2，场景ⓑ，含额外数据）**仅报告 Vocals SDR**：Mel-RoFormer 13.29；同表 BS-RoFormer 12.82 |
| Mamba2 Meets Silence (v2, 2025) | [arXiv:2508.14556](https://arxiv.org/abs/2508.14556) | 论文阶段（研究模型） | 📄 论文 | 摘要报告 cSDR 11.03 dB（作者称 best reported），强调稀疏人声段鲁棒性 |
| Windowed Sink Attention (2025) | [arXiv:2510.25745](https://arxiv.org/abs/2510.25745) | 论文阶段（效率优化方向） | 📄 论文 + 开源代码 | 在微调设定下恢复原模型约 92% SDR，同时 FLOPs 降低约 44.5x（偏效率收益） |

> **结论（按口径）**：  
> - 若只看 **本地开源 + 2-stem 直替 + 双指标都高于 ep368**：截至 2026-03-02，未检出明确公开 checkpoint。  
> - 若优先 **人声质量**：`MelBandRoformer.ckpt`（Kim）在公开列表给出 Vocals SDR 10.98（高于 ep317 的 10.87），但与 ep368 的 `audio-separator` 口径并非同一评测协议，不能直接横比。  
> - 若看 **MVSEP 榜单**：Ensemble（11.93 / 18.23）和 BS Roformer ver.2025.07（11.89 / 18.20）更高，但当前属于 API 调用，公开下载清单未见对应 checkpoint。  
> - 若看 **论文特定协议（MUSDB18-HQ 表2 场景ⓑ）**：Mel-RoFormer 报告 13.29，BS-RoFormer 为 12.82（该表仅有 vocals 指标）。  
> - **10.87 vs 12.82 不矛盾**：前者是 Multisong（viperx ep317），后者是 MUSDB18-HQ 场景ⓑ（含额外数据）的论文口径。  
> - **主唱/和声**：当前公开可下载里更常见的是 male/female 拆分；MVSEP 的 lead/back 主要在在线 API 集成中可见，权重映射未公开。  
> **口径提醒**：不同榜单/数据集/评测协议（Multisong、MUSDB、MVSEP、cSDR/uSDR）不可直接横比。

#### 六声部与细粒度乐器分离：前沿模型（2026-03-03 核实）

> 注：本表聚焦 **6-stem/细粒度乐器分离**，并明确区分“本地可下载”与“在线服务/API”。

| 模型/方向 | 来源 | 类型 | 状态 | 说明 |
|-----------|------|------|------|------|
| BS Roformer SW（当前六声部主后端） | [MVSEP Algorithms #77](https://mvsep.com/algorithms/77) / [HF: jarredou/BS-ROFO-SW-Fixed](https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/blob/main/BS-Rofo-SW-Fixed.ckpt) / [openmirlab config](https://raw.githubusercontent.com/openmirlab/bs-roformer-infer/main/src/bs_roformer/configs/config_bs_roformer_sw.yaml) | 本地可下载（6-stem） | ✅ 使用中 | MVSEP 页面口径给出：**vocals 11.30 / instrum 17.50 / bass 14.62 / drums 14.11 / guitar 9.05 / piano 7.83 / other 8.71**；本项目已接入并默认用于 `SIX_STEM_SPLIT` |
| HTDemucs4 (6 stems) | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) | 本地可下载（6-stem 基线） | ✅ 可替换（回退） | Multisong（同表）示例：bass 11.22 / drums 10.22 / vocals 8.05；适合作为通用回退 |
| DrumSep mdx23c（jarredou，5 stems） | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [jarredou models](https://github.com/jarredou/models/releases) | 本地可下载（鼓细分） | ✅ 可用 | 鼓细分公开指标：kick 16.66 / snare 11.53 / toms 12.33 / hh 4.04 / cymbals 6.36；适合对 `drums` 二次拆分 |
| DrumSep mdx23c（aufr33+jarredou，6 stems） | [ZFTurbo 预训练列表](https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/docs/pretrained_models.md) / [jarredou models](https://github.com/jarredou/models/releases) | 本地可下载（鼓细分） | ✅ 可用（实验） | 鼓细分公开指标：kick 14.54 / snare 9.79 / toms 10.63 / hh 3.19 / ride+crash 6.08 |
| MVSep Bass/Drums/Piano/Guitar 专项路线 | [MVSep Bass #37](https://mvsep.com/algorithms/37) / [Drums #43](https://mvsep.com/algorithms/43) / [Piano #14](https://mvsep.com/algorithms/14) / [Guitar #17](https://mvsep.com/algorithms/17) | 在线服务/榜单（专项） | 🌐 服务可用（非本地直替） | 页面展示 BS Roformer SW 在专项上的单模型值较高（如 Bass 14.62、Drums 14.11、Piano 7.83、Guitar 9.05），更高分通常来自融合 |
| Ensemble All-In（vocals,bass,drums,piano,guitar,lead/back,other） | [MVSEP Algorithms #47](https://mvsep.com/algorithms/47) | 在线融合 | 🌐 服务可用（非开源） | 覆盖面最广，适合“一次出多类 stem”；但当前未见完整公开 checkpoint 对应 |
| 主唱/和声（lead/back）前沿路线 | [MVSEP Karaoke #76](https://mvsep.com/algorithms/76) / [QC #8211](https://mvsep.com/quality_checker/entry/8211) / [QC #7845](https://mvsep.com/quality_checker/entry/7845) | 在线+部分开源 | 🔬 混合（开源 + 服务） | 公开可下载常见为 karaoke 或 male/female 路线；公开页上更高 lead/back 组合多为服务侧流程或融合 |

> **落地建议（本地开源优先）**：  
> - 6-stem 主链路：`BS Roformer SW`。  
> - 若要更细粒度鼓：在 `drums` 上追加 `DrumSep mdx23c`。  
> - 若要主唱/和声：在 `vocals` 上追加 karaoke/chorus 路线，并在文档中标明“近似 lead/back、非统一 benchmark”。  
> - 任何 API/服务模型均应单独标注“非公开权重、不可本地直替”。

#### 未来可关注的转写模型

以下为截至 2026 年初值得关注的新兴模型和研究方向（含当前项目基线）：

| 模型/方向 | 来源 | 类型 | 状态 | 说明 |
|-----------|------|------|------|------|
| YPTF.MoE+Multi (PS)（当前） | [YourMT3+ 论文](https://arxiv.org/abs/2407.04822) / [KAIST HF](https://huggingface.co/spaces/mimbres/YourMT3) | 多乐器 | ✅ 使用中 | Slakh2100 Multi F1 = **74.84%（0.7484）**；同表给出的 MT3 为 62.0（统一口径） |
| 2025 AMT Challenge 冠军方案（ai4m-miros） | [ICME 2025 Workshop 结果](https://ai4musicians.org/2025icme.html) / [Challenge 页](https://ai4musicians.org/transcription/2025transcription.html) / [amt-os/ai4m-miros](https://github.com/amt-os/ai4m-miros) | 多乐器 | ✅ 结果公布 / 代码可见 | ICME 页面列出 MIROS 第 1、YourMT3-YPTF-MoE-M 第 2；仓库说明方案基于 YourMT3+ + 预训练编码器 |
| [Aria-AMT](https://github.com/EleutherAI/aria-amt) | EleutherAI | 钢琴 | ✅ 开源可用（本项目已接入） | README 提供公开 checkpoint `piano-medium-double-1.0.safetensors`；适合钢琴专用转写，当前缺少与多乐器口径完全一致的统一公开榜单 |
| MusicFM 编码器 + 转写解码器 | [MusicFM 论文](https://arxiv.org/abs/2311.03318) / [MusicFM 仓库](https://github.com/minzwon/musicfm) | 多乐器（研究方向） | 📄 论文/组件可用 | 开源了基础编码器；仓库说明下游任务的微调模型与完整评测流水线未公开，仍需自建 decoder/训练流程 |
| CountEM（弱监督 AMT） | [arXiv:2511.14250](https://arxiv.org/abs/2511.14250) | AMT 训练方法 | 📄 论文阶段 | 以音符直方图监督替代对齐监督，降低标注依赖 |

#### 同领域已有模型对比

| 模型 | 来源 | 类型 | 说明 |
|------|------|------|------|
| [MT3](https://github.com/magenta/mt3) | Google Magenta | 多乐器 | Transformer 编码-解码，YourMT3+ 的基础架构；YourMT3+ 论文对比表中 MT3 在 Slakh Multi F1 为 62.0 |
| [Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) | MCT Lab | 多任务 | 支持钢琴/鼓/人声/和弦转写；GitHub latest release 仍为 2021-12-09（0.5.0） |
| [Basic Pitch](https://github.com/spotify/basic-pitch) | Spotify | 通用 | 轻量级单音/复音转写（<20MB、<17K 参数），官方定位为轻量与“comparable accuracy”；与 MT3/YourMT3 多乐器口径不宜直接横比 |

> **趋势总结**：截至 2026 年初，多乐器 AMT 仍以 MT3/YourMT3+ 系架构与“预训练编码器增强”路线并行发展；钢琴转写开源成熟度更高；挑战赛冠军方案虽已公开仓库，但跨数据集、跨口径的可复现对比仍在完善中。

## 开发

```bash
source venv/bin/activate

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/ -v

# 代码格式化
black src/
isort src/

# 类型检查
mypy src/

# 代码检查
flake8 src/ --max-line-length=100
```

## 常见问题

**Q: 中文/图标显示为方块（字体渲染问题）**
```bash
sudo apt-get install -y fonts-noto-cjk fonts-wqy-zenhei fonts-noto-color-emoji fonts-symbola
fc-cache -f
```
`./run.sh` 首次运行时会自动处理此问题。

**Q: PyQt6 提示 "could not load Qt platform plugin"**
```bash
sudo apt-get install libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor0
```

**Q: 提示找不到 libGL.so.1**
```bash
sudo apt-get install libgl1-mesa-glx
```

**Q: WSL2 窗口无法显示**

确认 WSLg 已启用（Windows 11 22000+ 默认内置），并检查：
```bash
echo $DISPLAY   # 应为 :0
ls /mnt/wslg    # 应有文件
```

**Q: CUDA 不可用**
```bash
nvidia-smi                                    # 检查驱动
python -c "import torch; print(torch.cuda.is_available())"
```

**Q: YourMT3+ 不可用**
```bash
python download_sota_models.py # 重新下载模型
```

**Q: 人声分离或六声部分离模型缺失**
```bash
python download_vocal_model.py      # 重新下载 BS-RoFormer ep368
python download_multistem_model.py  # 重新下载 BS-RoFormer SW
python download_aria_amt_model.py   # 重新下载 Aria-AMT 钢琴模型
```

## GPU 诊断

```bash
python -c "from src.utils.gpu_utils import print_gpu_diagnosis; print_gpu_diagnosis()"
python -c "from src.core.yourmt3_transcriber import YourMT3Transcriber; print(YourMT3Transcriber.is_available())"
```

## 贡献

欢迎提交 Pull Request。

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

## 许可证

MIT License - 详见 [LICENSE](./LICENSE) 文件。

## 致谢

- [YourMT3+](https://huggingface.co/spaces/mimbres/YourMT3) - 多乐器转写核心模型
- [BS-RoFormer](https://arxiv.org/abs/2309.02612) - 人声分离模型架构论文
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) - 音源分离推理框架
- [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) - BS-RoFormer 预训练权重
- [mido](https://github.com/mido/mido) - MIDI 文件处理
- [librosa](https://librosa.org/) - 音频分析
