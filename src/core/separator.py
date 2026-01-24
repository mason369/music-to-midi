"""
音源分离模块 - 使用 Demucs v4
支持钢琴模式（跳过分离）和智能模式（6轨分离）
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional, Callable, List
import torch

from src.models.data_models import (
    Config, TrackType, InstrumentType, ProcessingMode, TrackLayout
)
from src.utils.gpu_utils import get_device, clear_gpu_memory

logger = logging.getLogger(__name__)


class SourceSeparator:
    """
    使用 Demucs v4 进行音源分离

    支持两种模式:
    - 钢琴模式: 跳过分离，直接使用原始音频
    - 智能模式: 使用 htdemucs_6s 分离6轨（drums, bass, vocals, guitar, piano, other）
    """

    # 4轨模型（传统模式）
    STEMS_4 = ["drums", "bass", "vocals", "other"]
    MODEL_4S = "htdemucs"

    # 6轨模型（智能模式）
    STEMS_6 = ["drums", "bass", "vocals", "guitar", "piano", "other"]
    MODEL_6S = "htdemucs_6s"

    def __init__(self, config: Config):
        """
        初始化音源分离器

        参数:
            config: 应用配置
        """
        self.config = config
        self.model = None
        self.current_model_name = None
        self.device = get_device(config.use_gpu, config.gpu_device)
        self.segment_size = config.segment_size

    def load_model(self, model_name: str = None) -> None:
        """
        加载 Demucs 模型（延迟加载以节省内存）

        参数:
            model_name: 模型名称，默认使用4轨模型
        """
        if model_name is None:
            model_name = self.MODEL_4S

        # 如果已加载同一模型，直接返回
        if self.model is not None and self.current_model_name == model_name:
            return

        # 卸载之前的模型
        if self.model is not None:
            self.unload_model()

        logger.info(f"正在加载 Demucs 模型: {model_name}")

        try:
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels

            self.model = get_model(model_name)
            self.current_model_name = model_name

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
            self.current_model_name = None
            clear_gpu_memory()
            logger.info("模型已卸载")

    def _load_audio(self, audio_path: str) -> tuple:
        """
        加载音频文件并转换为适合 Demucs 的格式

        返回:
            (wav tensor [1, channels, samples], sample_rate)
        """
        import soundfile as sf

        logger.info(f"正在加载音频: {audio_path}")
        audio_data, sr = sf.read(audio_path, dtype='float32')

        # 转换为 torch tensor
        if audio_data.ndim == 1:
            wav = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            wav = torch.from_numpy(audio_data.T)

        # 确保是立体声
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        # 添加批次维度
        wav = wav.unsqueeze(0)

        return wav, sr

    def separate(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, str]:
        """
        将音频文件分离为多个音轨（4轨模式）

        参数:
            audio_path: 输入音频文件路径
            output_dir: 保存分离音轨的目录
            progress_callback: 可选的进度回调函数 (进度, 消息)

        返回:
            音轨名称到输出文件路径的字典
        """
        return self._separate_internal(
            audio_path, output_dir, self.MODEL_4S, self.STEMS_4, progress_callback
        )

    def separate_6s(
        self,
        audio_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, str]:
        """
        将音频文件分离为6个音轨（智能模式）

        参数:
            audio_path: 输入音频文件路径
            output_dir: 保存分离音轨的目录
            progress_callback: 可选的进度回调函数 (进度, 消息)

        返回:
            音轨名称到输出文件路径的字典
            包含: drums, bass, vocals, guitar, piano, other
        """
        return self._separate_internal(
            audio_path, output_dir, self.MODEL_6S, self.STEMS_6, progress_callback
        )

    def separate_v2(
        self,
        audio_path: str,
        output_dir: str,
        track_layout: TrackLayout,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, str]:
        """
        根据轨道布局分离音频

        参数:
            audio_path: 输入音频文件路径
            output_dir: 保存分离音轨的目录
            track_layout: 轨道布局配置
            progress_callback: 可选的进度回调函数 (进度, 消息)

        返回:
            轨道ID到输出文件路径的字典
        """
        if track_layout.mode == ProcessingMode.PIANO:
            # 钢琴模式：跳过分离，所有轨道使用原始音频
            logger.info("钢琴模式：跳过音源分离")
            if progress_callback:
                progress_callback(1.0, "钢琴模式：使用原始音频")
            return {track.id: audio_path for track in track_layout.get_enabled_tracks()}

        elif track_layout.mode == ProcessingMode.SMART:
            # 智能模式：使用6轨分离
            stem_paths = self.separate_6s(audio_path, output_dir, progress_callback)

            # 将分离结果映射到轨道ID
            result = {}
            for track in track_layout.get_enabled_tracks():
                # 根据乐器类型找到对应的分离轨道
                stem_name = self._instrument_to_stem(track.instrument)
                if stem_name in stem_paths:
                    result[track.id] = stem_paths[stem_name]
                else:
                    # 如果没有对应的分离轨道，使用 other
                    result[track.id] = stem_paths.get("other", audio_path)
            return result

        else:
            raise ValueError(f"未知的处理模式: {track_layout.mode}")

    def _instrument_to_stem(self, instrument: InstrumentType) -> str:
        """将乐器类型映射到分离轨道名称"""
        mapping = {
            InstrumentType.DRUMS: "drums",
            InstrumentType.BASS: "bass",
            InstrumentType.VOCALS: "vocals",
            InstrumentType.GUITAR: "guitar",
            InstrumentType.PIANO: "piano",
            InstrumentType.STRINGS: "other",
            InstrumentType.OTHER: "other",
        }
        return mapping.get(instrument, "other")

    def _separate_internal(
        self,
        audio_path: str,
        output_dir: str,
        model_name: str,
        stems: List[str],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, str]:
        """
        内部分离方法

        参数:
            audio_path: 输入音频文件路径
            output_dir: 保存分离音轨的目录
            model_name: 模型名称
            stems: 要提取的轨道列表
            progress_callback: 可选的进度回调函数 (进度, 消息)

        返回:
            音轨名称到输出文件路径的字典
        """
        import soundfile as sf
        import torchaudio.transforms
        from demucs.apply import apply_model

        self.load_model(model_name)

        if progress_callback:
            progress_callback(0.0, "正在加载音频...")

        wav, sr = self._load_audio(audio_path)

        # 如需要则重采样
        if sr != self.model.samplerate:
            logger.info(f"正在从 {sr} 重采样到 {self.model.samplerate}")
            wav = torchaudio.transforms.Resample(sr, self.model.samplerate)(wav)

        wav = wav.to(self.device)

        if progress_callback:
            progress_callback(0.2, "正在分离音源...")

        # 应用模型
        logger.info(f"正在应用音源分离模型 ({model_name})...")
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
            if stem in stems:
                output_path = os.path.join(output_dir, f"{input_name}_{stem}.wav")
                stem_audio = sources[0, i].cpu().numpy().T  # 转置为 [samples, channels]

                sf.write(output_path, stem_audio, self.model.samplerate)

                output_paths[stem] = output_path
                logger.info(f"已保存 {stem}: {output_path}")

        if progress_callback:
            progress_callback(1.0, "分离完成")

        return output_paths

    def get_stem_for_track_type(self, track_type: TrackType) -> str:
        """获取轨道类型对应的音轨名称（已弃用，保留向后兼容）"""
        return track_type.value

    def get_stem_for_instrument(self, instrument: InstrumentType) -> str:
        """获取乐器类型对应的音轨名称"""
        return self._instrument_to_stem(instrument)
