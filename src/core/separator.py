"""
音源分离模块 - 使用 Demucs v4
将音频分离为鼓、贝斯、人声和其他乐器轨道
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Callable
import torch

from src.models.data_models import Config, TrackType
from src.utils.gpu_utils import get_device, clear_gpu_memory

logger = logging.getLogger(__name__)


class SourceSeparator:
    """
    使用 Demucs v4 (htdemucs) 进行音源分离

    将音频分离为4个音轨：
    - drums: 鼓和打击乐器
    - bass: 贝斯
    - vocals: 人声
    - other: 其他乐器（吉他、键盘等）
    """

    STEMS = ["drums", "bass", "vocals", "other"]
    MODEL_NAME = "htdemucs"

    def __init__(self, config: Config):
        """
        初始化音源分离器

        参数:
            config: 应用配置
        """
        self.config = config
        self.model = None
        self.device = get_device(config.use_gpu, config.gpu_device)
        self.segment_size = config.segment_size

    def load_model(self) -> None:
        """加载 Demucs 模型（延迟加载以节省内存）"""
        if self.model is not None:
            return

        logger.info(f"正在加载 Demucs 模型: {self.MODEL_NAME}")

        try:
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels

            self.model = get_model(self.MODEL_NAME)

            if isinstance(self.model, BagOfModels):
                logger.info(f"已加载 {len(self.model.models)} 个模型组合")
            else:
                logger.info("已加载单个模型")

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"模型已加载到设备: {self.device}")

        except ImportError as e:
            logger.error("Demucs 未安装，请运行: pip install demucs")
            raise ImportError("音源分离需要 Demucs 库") from e

    def unload_model(self) -> None:
        """卸载模型以释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
            clear_gpu_memory()
            logger.info("模型已卸载")

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, str]:
        """
        将音频文件分离为多个音轨

        参数:
            audio_path: 输入音频文件路径
            output_dir: 保存分离音轨的目录
            progress_callback: 可选的进度回调函数 (进度, 消息)

        返回:
            音轨名称到输出文件路径的字典
        """
        import soundfile as sf
        import numpy as np
        import torchaudio.transforms
        from demucs.apply import apply_model

        self.load_model()

        if progress_callback:
            progress_callback(0.0, "正在加载音频...")

        # 使用 soundfile 直接加载音频（避免 torchaudio 的 torchcodec 依赖问题）
        logger.info(f"正在加载音频: {audio_path}")
        audio_data, sr = sf.read(audio_path, dtype='float32')

        # 转换为 torch tensor (soundfile 返回 [samples, channels]，需要转置为 [channels, samples])
        if audio_data.ndim == 1:
            # 单声道
            wav = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            # 多声道，转置
            wav = torch.from_numpy(audio_data.T)

        # 确保是立体声
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        # 添加批次维度
        wav = wav.unsqueeze(0)

        # 如需要则重采样
        if sr != self.model.samplerate:
            logger.info(f"正在从 {sr} 重采样到 {self.model.samplerate}")
            wav = torchaudio.transforms.Resample(sr, self.model.samplerate)(wav)

        wav = wav.to(self.device)

        if progress_callback:
            progress_callback(0.2, "正在分离音源...")

        # 应用模型
        logger.info("正在应用音源分离模型...")
        with torch.no_grad():
            sources = apply_model(
                self.model,
                wav,
                device=self.device,
                segment=self.segment_size,
                overlap=0.25,
                progress=True
            )

        if progress_callback:
            progress_callback(0.8, "正在保存分离的音轨...")

        # 保存音轨
        output_paths = {}
        os.makedirs(output_dir, exist_ok=True)

        input_name = Path(audio_path).stem

        for i, stem in enumerate(self.model.sources):
            if stem in self.STEMS:
                output_path = os.path.join(output_dir, f"{input_name}_{stem}.wav")
                stem_audio = sources[0, i].cpu().numpy().T  # 转置为 [samples, channels]

                sf.write(output_path, stem_audio, self.model.samplerate)

                output_paths[stem] = output_path
                logger.info(f"已保存 {stem}: {output_path}")

        if progress_callback:
            progress_callback(1.0, "分离完成")

        return output_paths

    def get_stem_for_track_type(self, track_type: TrackType) -> str:
        """获取轨道类型对应的音轨名称"""
        return track_type.value
