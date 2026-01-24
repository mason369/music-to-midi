"""
乐器识别模块 - 使用 Demucs 6s + PANNs 进行智能乐器检测

通过分析音频文件自动检测其中包含的乐器类型
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

import numpy as np

from src.models.data_models import (
    Config, InstrumentType, TrackLayout, TrackConfig, ProcessingMode
)
from src.utils.gpu_utils import get_device
from src.utils.panns_downloader import ensure_panns_files, get_panns_data_dir

logger = logging.getLogger(__name__)


@dataclass
class InstrumentPrediction:
    """乐器预测结果"""
    instrument: InstrumentType
    confidence: float  # 0-1 之间的置信度
    source: str        # 预测来源: "demucs" 或 "panns"


class InstrumentClassifier:
    """
    使用 Demucs 6s + PANNs 进行乐器识别

    工作流程:
    1. 使用 Demucs 6s 分离音轨，检测各轨道是否有内容
    2. 使用 PANNs 对分离后的轨道进行验证和细化识别
    """

    # PANNs 标签到乐器类型的映射
    PANNS_LABEL_MAPPING = {
        # 钢琴相关
        "Piano": InstrumentType.PIANO,
        "Electric piano": InstrumentType.PIANO,
        "Keyboard (musical)": InstrumentType.PIANO,

        # 鼓相关
        "Drum": InstrumentType.DRUMS,
        "Drum kit": InstrumentType.DRUMS,
        "Snare drum": InstrumentType.DRUMS,
        "Bass drum": InstrumentType.DRUMS,
        "Hi-hat": InstrumentType.DRUMS,
        "Cymbal": InstrumentType.DRUMS,
        "Percussion": InstrumentType.DRUMS,
        "Drum machine": InstrumentType.DRUMS,
        "Timpani": InstrumentType.DRUMS,
        "Tabla": InstrumentType.DRUMS,

        # 贝斯相关
        "Bass guitar": InstrumentType.BASS,
        "Electric bass": InstrumentType.BASS,
        "Double bass": InstrumentType.BASS,
        "Bass": InstrumentType.BASS,

        # 吉他相关
        "Guitar": InstrumentType.GUITAR,
        "Electric guitar": InstrumentType.GUITAR,
        "Acoustic guitar": InstrumentType.GUITAR,
        "Steel guitar, slide guitar": InstrumentType.GUITAR,
        "Banjo": InstrumentType.GUITAR,
        "Mandolin": InstrumentType.GUITAR,
        "Ukulele": InstrumentType.GUITAR,

        # 人声相关
        "Singing": InstrumentType.VOCALS,
        "Male singing": InstrumentType.VOCALS,
        "Female singing": InstrumentType.VOCALS,
        "Choir": InstrumentType.VOCALS,
        "Speech": InstrumentType.VOCALS,
        "Vocal music": InstrumentType.VOCALS,
        "Rapping": InstrumentType.VOCALS,
        "Humming": InstrumentType.VOCALS,

        # 弦乐相关
        "Violin, fiddle": InstrumentType.STRINGS,
        "Cello": InstrumentType.STRINGS,
        "Viola": InstrumentType.STRINGS,
        "String section": InstrumentType.STRINGS,
        "Orchestral music": InstrumentType.STRINGS,
        "Bowed string instrument": InstrumentType.STRINGS,
        "Pizzicato": InstrumentType.STRINGS,

        # 铜管乐器
        "Trumpet": InstrumentType.BRASS,
        "Trombone": InstrumentType.BRASS,
        "French horn": InstrumentType.BRASS,
        "Brass instrument": InstrumentType.BRASS,
        "Tuba": InstrumentType.BRASS,

        # 木管乐器
        "Flute": InstrumentType.WOODWIND,
        "Saxophone": InstrumentType.WOODWIND,
        "Clarinet": InstrumentType.WOODWIND,
        "Oboe": InstrumentType.WOODWIND,
        "Bassoon": InstrumentType.WOODWIND,
        "Wind instrument, woodwind instrument": InstrumentType.WOODWIND,
        "Recorder": InstrumentType.WOODWIND,

        # 合成器
        "Synthesizer": InstrumentType.SYNTH,
        "Electronic music": InstrumentType.SYNTH,
        "Sampler": InstrumentType.SYNTH,

        # 风琴
        "Organ": InstrumentType.ORGAN,
        "Electronic organ": InstrumentType.ORGAN,
        "Hammond organ": InstrumentType.ORGAN,
        "Accordion": InstrumentType.ORGAN,
        "Harmonica": InstrumentType.ORGAN,

        # 竖琴
        "Harp": InstrumentType.HARP,
        "Zither": InstrumentType.HARP,
        "Sitar": InstrumentType.HARP,
    }

    # 最小能量阈值（用于判断轨道是否有内容）
    # 降低绝对阈值，适应更多音乐类型（正常音乐轨道能量通常在 0.001-0.01 范围）
    MIN_ENERGY_THRESHOLD = 0.001

    # 相对能量阈值（轨道能量占最大能量的比例）
    # 降低到3%以接受更多弱轨道
    MIN_RELATIVE_ENERGY_THRESHOLD = 0.03

    # 最小置信度阈值（低于此值的预测将被忽略）
    MIN_CONFIDENCE_THRESHOLD = 0.25  # 从0.1提高到0.25，减少误检

    def __init__(self, config: Config):
        """
        初始化乐器识别器

        参数:
            config: 应用配置
        """
        self.config = config
        self.device = get_device(config.use_gpu, config.gpu_device)
        self.panns_model = None
        self.panns_labels = None

    def load_panns_model(self) -> None:
        """加载 PANNs 模型（延迟加载）"""
        if self.panns_model is not None:
            return

        logger.info("正在加载 PANNs 模型...")

        try:
            # 首先确保 PANNs 所需文件已下载（解决 Windows 下 wget 不可用的问题）
            logger.info("正在检查 PANNs 依赖文件...")
            if not ensure_panns_files():
                logger.warning("PANNs 文件下载失败，将仅使用 Demucs 进行基础识别")
                self.panns_model = None
                return

            # 获取模型路径
            panns_data_dir = get_panns_data_dir()
            checkpoint_path = panns_data_dir / "Cnn14_mAP=0.431.pth"

            # 关键：在导入 panns_inference 之前，先注入修补的 config 模块
            # 这样可以避免 panns_inference/config.py 中的 wget 调用
            self._inject_panns_config()

            # 抑制 PANNs 的 print 输出（它使用 print 而非 logging）
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                from panns_inference import AudioTagging

                logger.debug(f"PANNs 检查点路径: {checkpoint_path}")
                logger.debug(f"PANNs 使用设备: {self.device}")

                self.panns_model = AudioTagging(
                    checkpoint_path=str(checkpoint_path),
                    device=str(self.device)
                )
            finally:
                # 获取 PANNs 的输出并记录到日志
                panns_output = sys.stdout.getvalue()
                sys.stdout = old_stdout
                if panns_output.strip():
                    for line in panns_output.strip().split('\n'):
                        logger.debug(f"[PANNs] {line}")

            logger.info("PANNs 模型已加载")
            logger.debug(f"PANNs 模型类型: {type(self.panns_model)}")

        except ImportError:
            logger.warning("PANNs 未安装，将仅使用 Demucs 进行基础识别")
            self.panns_model = None
        except Exception as e:
            logger.warning(f"PANNs 加载失败: {e}，将仅使用 Demucs 进行基础识别")
            self.panns_model = None

    def _inject_panns_config(self) -> None:
        """
        在 sys.modules 中注入修补的 panns_inference.config 模块

        这个方法必须在 import panns_inference 之前调用，
        这样当 panns_inference/__init__.py 尝试导入 config 时，
        会使用我们注入的版本，而不是执行原始的 config.py（其中包含 wget 调用）
        """
        import sys

        # 如果已经导入过，不需要再处理
        if 'panns_inference.config' in sys.modules:
            logger.info("PANNs config 模块已存在，跳过注入")
            return

        try:
            import csv
            from types import ModuleType

            panns_data_dir = get_panns_data_dir()
            csv_path = panns_data_dir / "class_labels_indices.csv"

            if not csv_path.exists():
                logger.warning(f"PANNs 标签文件不存在: {csv_path}")
                return

            # 读取标签
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=',')
                lines = list(reader)

            labels = []
            ids = []
            for i in range(1, len(lines)):
                if len(lines[i]) >= 3:
                    ids.append(lines[i][1])
                    labels.append(lines[i][2])

            classes_num = len(labels)

            # 创建一个模拟的 config 模块
            config_module = ModuleType('panns_inference.config')
            config_module.labels = labels
            config_module.ids = ids
            config_module.classes_num = classes_num
            config_module.sample_rate = 32000
            config_module.lb_to_ix = {label: i for i, label in enumerate(labels)}
            config_module.ix_to_lb = {i: label for i, label in enumerate(labels)}
            config_module.id_to_ix = {id_: i for i, id_ in enumerate(ids)}
            config_module.ix_to_id = {i: id_ for i, id_ in enumerate(ids)}

            # 注入到 sys.modules（关键步骤！）
            sys.modules['panns_inference.config'] = config_module

            logger.info(f"PANNs config 模块已注入 (共 {classes_num} 个类别)")

        except Exception as e:
            logger.warning(f"注入 PANNs config 失败: {e}")

    def classify_from_stems(
        self,
        stem_paths: Dict[str, str],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[InstrumentPrediction]:
        """
        从分离的音轨中识别乐器

        参数:
            stem_paths: 轨道名称到文件路径的字典
            progress_callback: 可选的进度回调函数

        返回:
            识别到的乐器预测列表
        """
        predictions = []
        total_stems = len(stem_paths)

        logger.debug(f"开始分析 {total_stems} 个分离轨道")

        # 首先计算所有轨道的能量，用于相对比较
        stem_energies = {}
        for stem_name, stem_path in stem_paths.items():
            energy = self._calculate_energy(stem_path)
            stem_energies[stem_name] = energy
            logger.debug(f"  {stem_name}: 能量 = {energy:.8f}")

        max_energy = max(stem_energies.values()) if stem_energies else 1.0
        if max_energy < 0.0001:
            max_energy = 0.0001  # 避免除零

        logger.debug(f"轨道能量范围: 最大 = {max_energy:.8f}")

        for i, (stem_name, stem_path) in enumerate(stem_paths.items()):
            if progress_callback:
                progress = (i + 1) / total_stems
                progress_callback(progress, f"正在分析 {stem_name} 轨道...")

            # 检查轨道是否有内容（使用增强的三重检测）
            if not self._has_content(stem_path, max_energy):
                logger.info(f"{stem_name} 轨道无内容，跳过")
                continue

            # 根据轨道名称确定基础乐器类型
            instrument = self._stem_to_instrument(stem_name)
            if instrument is None:
                continue

            # 使用相对能量计算置信度
            # Demucs 分离的轨道只要有内容，就应该有较高的基础置信度
            energy = stem_energies[stem_name]
            relative_energy = energy / max_energy

            # 基础置信度：有内容的轨道至少 0.5，根据相对能量增加到最高 1.0
            base_confidence = 0.5 + (relative_energy * 0.5)

            logger.info(f"{stem_name}: 能量={energy:.6f}, 相对能量={relative_energy:.2f}, 置信度={base_confidence:.2f}")

            predictions.append(InstrumentPrediction(
                instrument=instrument,
                confidence=base_confidence,
                source="demucs"
            ))

            logger.info(f"检测到 {instrument.value}: 置信度 {base_confidence:.2f}")

        return predictions

    def refine_with_panns(
        self,
        audio_path: str,
        stem_paths: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[InstrumentPrediction]:
        """
        使用 PANNs 对音频进行更精确的乐器识别

        增强功能：
        - 分析原始音频检测所有乐器
        - 对 other 轨道进行额外分析，识别其中包含的具体乐器（如弦乐、管乐等）
        - 不受 Demucs 6轨限制，可以识别更多乐器类型

        参数:
            audio_path: 原始音频路径
            stem_paths: 可选的分离轨道路径（用于细化识别）
            progress_callback: 可选的进度回调函数

        返回:
            识别到的乐器预测列表
        """
        self.load_panns_model()

        if self.panns_model is None:
            logger.warning("PANNs 模型不可用，跳过细化识别")
            return []

        predictions = []

        if progress_callback:
            progress_callback(0.1, "正在使用 PANNs 分析音频...")

        try:
            import librosa

            # 1. 分析原始音频
            logger.debug(f"正在加载音频进行 PANNs 分析: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=32000, mono=True)
            logger.debug(f"音频加载完成: {len(audio)} 采样点, 采样率 {sr}Hz, 时长 {len(audio)/sr:.2f}秒")

            logger.debug("正在运行 PANNs 推理...")
            clipwise_output, _ = self.panns_model.inference(audio[None, :])
            logger.debug(f"PANNs 输出形状: {clipwise_output.shape}")

            if progress_callback:
                progress_callback(0.4, "正在解析识别结果...")

            # 获取前 N 个预测
            top_k = 30  # 增加检测数量以捕获更多乐器
            top_indices = np.argsort(clipwise_output[0])[::-1][:top_k]
            labels = self._get_panns_labels()

            logger.debug(f"PANNs 前 {top_k} 个预测结果:")
            for rank, idx in enumerate(top_indices):
                label = labels[idx] if idx < len(labels) else f"Unknown_{idx}"
                confidence = float(clipwise_output[0, idx])
                logger.debug(f"  #{rank+1}: {label} = {confidence:.4f}")

            for idx in top_indices:
                label = labels[idx] if idx < len(labels) else f"Unknown_{idx}"
                confidence = float(clipwise_output[0, idx])

                if confidence < self.MIN_CONFIDENCE_THRESHOLD:
                    continue

                if label in self.PANNS_LABEL_MAPPING:
                    instrument = self.PANNS_LABEL_MAPPING[label]
                    predictions.append(InstrumentPrediction(
                        instrument=instrument,
                        confidence=confidence,
                        source="panns"
                    ))
                    logger.info(f"PANNs 检测到 {label} -> {instrument.value}: {confidence:.2f}")

            # 2. 对 other 轨道进行额外分析（如果存在）
            if stem_paths and "other" in stem_paths:
                if progress_callback:
                    progress_callback(0.6, "正在分析 other 轨道中的额外乐器...")

                other_predictions = self._analyze_other_stem(stem_paths["other"])
                predictions.extend(other_predictions)

            if progress_callback:
                progress_callback(1.0, "PANNs 分析完成")

        except Exception as e:
            logger.error(f"PANNs 分析失败: {e}")

        return predictions

    def _analyze_other_stem(self, other_path: str) -> List[InstrumentPrediction]:
        """
        对 other 轨道进行深度分析，识别其中包含的具体乐器

        Demucs 的 other 轨道通常包含：弦乐、管乐、合成器、打击乐等

        参数:
            other_path: other 轨道的音频路径

        返回:
            额外检测到的乐器预测列表
        """
        predictions = []

        if self.panns_model is None:
            return predictions

        try:
            import librosa

            # 检查 other 轨道是否有足够内容
            if not self._has_content(other_path):
                logger.info("other 轨道无内容，跳过额外分析")
                return predictions

            audio, sr = librosa.load(other_path, sr=32000, mono=True)
            clipwise_output, _ = self.panns_model.inference(audio[None, :])

            top_k = 20
            top_indices = np.argsort(clipwise_output[0])[::-1][:top_k]
            labels = self._get_panns_labels()

            # 特别关注 other 轨道中可能的乐器
            # 这些乐器不在 Demucs 6轨中但 PANNs 可以识别
            other_instrument_labels = {
                # 弦乐
                "Violin, fiddle", "Cello", "Viola", "String section",
                "Bowed string instrument", "Pizzicato",
                # 管乐
                "Flute", "Saxophone", "Clarinet", "Trumpet", "Trombone",
                "French horn", "Brass instrument",
                # 其他
                "Harp", "Organ", "Synthesizer", "Harmonica", "Accordion",
            }

            for idx in top_indices:
                label = labels[idx] if idx < len(labels) else f"Unknown_{idx}"
                confidence = float(clipwise_output[0, idx])

                # 对 other 轨道使用更低的阈值，因为这些乐器可能较弱
                if confidence < 0.08:
                    continue

                if label in other_instrument_labels and label in self.PANNS_LABEL_MAPPING:
                    instrument = self.PANNS_LABEL_MAPPING[label]

                    # 提升置信度，因为这是从 other 轨道单独检测的
                    boosted_confidence = min(1.0, confidence * 1.5)

                    predictions.append(InstrumentPrediction(
                        instrument=instrument,
                        confidence=boosted_confidence,
                        source="panns_other"
                    ))
                    logger.info(f"other轨道检测到 {label} -> {instrument.value}: {boosted_confidence:.2f}")

        except Exception as e:
            logger.warning(f"other 轨道分析失败: {e}")

        return predictions

    def suggest_track_layout(
        self,
        predictions: List[InstrumentPrediction],
        min_confidence: float = 0.30  # 从0.15提高到0.30，减少低置信度轨道
    ) -> TrackLayout:
        """
        根据预测结果建议轨道布局

        增强功能：
        - 支持超过6个乐器（不受 Demucs 限制）
        - 来自 other 轨道的乐器也会创建独立轨道
        - 智能合并同类乐器

        参数:
            predictions: 乐器预测列表
            min_confidence: 最小置信度阈值（降低以接受更多轨道）

        返回:
            建议的轨道布局
        """
        # 按乐器类型聚合预测，取最高置信度
        instrument_scores: Dict[InstrumentType, float] = {}
        instrument_sources: Dict[InstrumentType, str] = {}

        logger.info(f"收到 {len(predictions)} 个预测结果:")
        for pred in predictions:
            logger.info(f"  - {pred.instrument.value}: {pred.confidence:.2f} (来源: {pred.source})")

            current_score = instrument_scores.get(pred.instrument, 0)
            # 使用加权平均
            # panns_other 的结果权重最高（从 other 轨道单独检测的）
            # panns 次之
            # demucs 最低
            if pred.source == "panns_other":
                weight = 1.8
            elif pred.source == "panns":
                weight = 1.5
            else:
                weight = 1.0

            new_score = pred.confidence * weight
            if new_score > current_score:
                instrument_scores[pred.instrument] = new_score
                instrument_sources[pred.instrument] = pred.source

        # 过滤低置信度的乐器，但对 other 轨道检测的乐器使用更低阈值
        filtered_instruments = {}
        for inst, score in instrument_scores.items():
            source = instrument_sources.get(inst, "unknown")
            # 来自 other 轨道的乐器使用更低阈值
            effective_threshold = min_confidence * 0.6 if source == "panns_other" else min_confidence

            if score >= effective_threshold:
                filtered_instruments[inst] = score
            else:
                logger.warning(f"乐器 {inst.value} 因置信度不足被过滤 ({score:.2f} < {effective_threshold:.2f})")

        # 按置信度排序
        sorted_instruments = sorted(
            filtered_instruments.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 创建轨道配置
        tracks = []
        channel = 0

        for instrument, score in sorted_instruments:
            # 跳过 OTHER 类型 - 让具体乐器替代
            # 只有当没有其他任何乐器时才保留 OTHER
            if instrument == InstrumentType.OTHER:
                # 检查是否有其他具体乐器
                has_concrete_instruments = any(
                    inst != InstrumentType.OTHER
                    for inst, _ in sorted_instruments
                )
                if has_concrete_instruments:
                    logger.info("跳过 OTHER 轨道（已有具体乐器）")
                    continue

            # 鼓使用固定通道
            if instrument == InstrumentType.DRUMS:
                midi_channel = 9
            else:
                midi_channel = channel
                channel += 1
                if channel == 9:
                    channel = 10

            tracks.append(TrackConfig(
                id=f"{instrument.value}_{len(tracks) + 1}",
                instrument=instrument,
                name=instrument.get_display_name(self.config.language),
                enabled=True,
                midi_channel=midi_channel,
                program=instrument.to_program_number()
            ))

            source = instrument_sources.get(instrument, "unknown")
            logger.info(f"添加轨道: {instrument.value} (置信度: {score:.2f}, 来源: {source}, 通道: {midi_channel})")

        # 如果没有检测到任何乐器，尝试降低阈值再次检测
        if not tracks:
            # 如果有预测结果但都被过滤掉了，尝试降低阈值再次检测
            if predictions:
                logger.info("首次检测无结果，尝试降低阈值...")
                # 使用 50% 的阈值重新过滤
                lower_threshold = min_confidence * 0.5
                for inst, score in instrument_scores.items():
                    source = instrument_sources.get(inst, "unknown")
                    effective_threshold = lower_threshold * 0.6 if source == "panns_other" else lower_threshold
                    if score >= effective_threshold and inst != InstrumentType.OTHER:
                        midi_channel = 9 if inst == InstrumentType.DRUMS else len(tracks)
                        if len(tracks) == 9:
                            midi_channel = 10
                        tracks.append(TrackConfig(
                            id=f"{inst.value}_{len(tracks) + 1}",
                            instrument=inst,
                            name=inst.get_display_name(self.config.language),
                            enabled=True,
                            midi_channel=midi_channel,
                            program=inst.to_program_number()
                        ))
                        logger.info(f"降低阈值后添加轨道: {inst.value} (置信度: {score:.2f})")

            # 如果仍然没有轨道，创建一个通用轨道（保持 SMART 模式）
            if not tracks:
                logger.warning("未检测到任何乐器，创建通用轨道")
                # 使用置信度最高的乐器，或者使用 OTHER
                if instrument_scores:
                    best_inst = max(instrument_scores.items(), key=lambda x: x[1])
                    inst, score = best_inst
                    tracks.append(TrackConfig(
                        id=f"{inst.value}_1",
                        instrument=inst,
                        name=inst.get_display_name(self.config.language),
                        enabled=True,
                        midi_channel=0,
                        program=inst.to_program_number()
                    ))
                    logger.warning(f"使用最高置信度乐器: {inst.value} ({score:.2f})")
                else:
                    # 真正没有任何检测结果，使用 OTHER 轨道
                    tracks.append(TrackConfig(
                        id="other_1",
                        instrument=InstrumentType.OTHER,
                        name=InstrumentType.OTHER.get_display_name(self.config.language),
                        enabled=True,
                        midi_channel=0,
                        program=0
                    ))

        logger.info(f"最终轨道布局: {len(tracks)} 个轨道")
        return TrackLayout(mode=ProcessingMode.SMART, tracks=tracks)  # 始终保持 SMART 模式

    def _stem_to_instrument(self, stem_name: str) -> Optional[InstrumentType]:
        """将分离轨道名称转换为乐器类型"""
        mapping = {
            "drums": InstrumentType.DRUMS,
            "bass": InstrumentType.BASS,
            "vocals": InstrumentType.VOCALS,
            "guitar": InstrumentType.GUITAR,
            "piano": InstrumentType.PIANO,
            "other": InstrumentType.OTHER,
        }
        return mapping.get(stem_name.lower())

    def _has_content(self, audio_path: str, max_energy: float = None) -> bool:
        """
        检查音频文件是否有内容（非静音）

        使用三重检测：
        1. 绝对能量阈值 - 过滤完全静音
        2. 相对能量阈值 - 过滤相对于最响轨道太弱的轨道
        3. Crest Factor - 区分噪声和音乐（音乐有更大的动态范围）
        """
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)

            # 如果是多声道，转为单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # 1. 绝对能量检测
            energy = np.mean(audio ** 2)
            if energy < self.MIN_ENERGY_THRESHOLD:
                logger.debug(f"轨道能量过低: {energy:.8f} < {self.MIN_ENERGY_THRESHOLD}")
                return False

            # 2. 相对能量检测（如果提供了最大能量）
            if max_energy is not None and max_energy > 0:
                relative_energy = energy / max_energy
                if relative_energy < self.MIN_RELATIVE_ENERGY_THRESHOLD:
                    logger.debug(f"轨道相对能量过低: {relative_energy:.4f} < {self.MIN_RELATIVE_ENERGY_THRESHOLD}")
                    return False

            # 3. Crest Factor 检测（峰值/RMS比）
            # 音乐通常有较高的 Crest Factor（6-20），纯噪声较低（2-4）
            rms = np.sqrt(energy)
            peak = np.max(np.abs(audio))
            if rms > 0:
                crest_factor = peak / rms
                # 如果 Crest Factor 太低，可能只是噪声
                if crest_factor < 2.0:  # 更宽松的阈值，避免误判
                    logger.debug(f"轨道可能是噪声 (Crest Factor: {crest_factor:.2f})")
                    return False

            return True
        except Exception as e:
            logger.warning(f"检查音频内容时出错: {e}")
            return True  # 默认假设有内容

    def _calculate_energy(self, audio_path: str) -> float:
        """计算音频的平均能量"""
        try:
            import soundfile as sf
            audio, _ = sf.read(audio_path)
            return float(np.mean(audio ** 2))
        except Exception:
            return 0.5  # 默认中等能量

    def _get_panns_labels(self) -> List[str]:
        """获取 PANNs 的标签列表"""
        if self.panns_labels is not None:
            return self.panns_labels

        # AudioSet 527 类标签的前几个常见乐器标签
        # 完整列表可从 AudioSet 获取
        self.panns_labels = [
            "Speech", "Male speech, man speaking", "Female speech, woman speaking",
            "Child speech, kid speaking", "Conversation", "Narration, monologue",
            "Babbling", "Speech synthesizer", "Shout", "Bellow", "Whoop",
            "Yell", "Children shouting", "Screaming", "Whispering", "Laughter",
            "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle",
            "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan",
            "Sigh", "Singing", "Choir", "Yodeling", "Chant", "Mantra",
            "Male singing", "Female singing", "Child singing", "Synthetic singing",
            "Rapping", "Humming", "Groan", "Grunt", "Whistling", "Breathing",
            "Wheeze", "Snoring", "Gasp", "Pant", "Snort", "Cough", "Throat clearing",
            "Sneeze", "Sniff", "Run", "Shuffle", "Walk, footsteps", "Chewing, mastication",
            "Biting", "Gargling", "Stomach rumble", "Burping, eructation",
            "Hiccup", "Fart", "Hands", "Finger snapping", "Clapping", "Heart sounds, heartbeat",
            "Heart murmur", "Cheering", "Applause", "Chatter", "Crowd",
            "Hubbub, speech noise, speech babble", "Children playing",
            "Animal", "Domestic animals, pets", "Dog", "Bark", "Yip", "Howl",
            "Bow-wow", "Growling", "Whimper (dog)", "Cat", "Purr", "Meow",
            "Hiss", "Caterwaul", "Livestock, farm animals, working animals",
            "Horse", "Clip-clop", "Neigh, whinny", "Cattle, bovinae", "Moo",
            "Cowbell", "Pig", "Oink", "Goat", "Bleat", "Sheep", "Fowl",
            "Chicken, rooster", "Cluck", "Crowing, cock-a-doodle-doo",
            "Turkey", "Gobble", "Duck", "Quack", "Goose", "Honk",
            "Wild animals", "Roaring cats (lions, tigers)", "Roar",
            "Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet",
            "Squawk", "Pigeon, dove", "Coo", "Crow", "Caw", "Owl", "Hoot",
            "Bird flight, flapping wings", "Canidae, dogs, wolves",
            "Rodents, rats, mice", "Mouse", "Patter", "Insect", "Cricket",
            "Mosquito", "Fly, housefly", "Buzz", "Bee, wasp, etc.", "Frog",
            "Croak", "Snake", "Rattle", "Whale vocalization", "Music",
            "Musical instrument", "Plucked string instrument",
            "Guitar", "Electric guitar", "Bass guitar", "Acoustic guitar",
            "Steel guitar, slide guitar", "Tapping (guitar technique)",
            "Strum", "Banjo", "Sitar", "Mandolin", "Zither", "Ukulele",
            "Keyboard (musical)", "Piano", "Electric piano",
            "Organ", "Electronic organ", "Hammond organ", "Synthesizer",
            "Sampler", "Harpsichord", "Percussion", "Drum kit", "Drum machine",
            "Drum", "Snare drum", "Rimshot", "Drum roll", "Bass drum",
            "Timpani", "Tabla", "Cymbal", "Hi-hat", "Wood block",
            "Tambourine", "Rattle (instrument)", "Maraca", "Gong",
            "Tubular bells", "Mallet percussion", "Marimba, xylophone",
            "Glockenspiel", "Vibraphone", "Steelpan", "Orchestra",
            "Brass instrument", "French horn", "Trumpet", "Trombone",
            "Bowed string instrument", "String section", "Violin, fiddle",
            "Pizzicato", "Cello", "Double bass", "Wind instrument, woodwind instrument",
            "Flute", "Saxophone", "Clarinet", "Harp", "Bell", "Church bell",
            "Jingle bell", "Bicycle bell", "Tuning fork", "Chime", "Wind chime",
            "Change ringing (campanology)", "Harmonica", "Accordion",
            "Bagpipes", "Didgeridoo", "Shofar", "Theremin", "Singing bowl",
            "Scratching (performance technique)", "Pop music", "Hip hop music",
            "Beatboxing", "Rock music", "Heavy metal", "Punk rock",
            "Grunge", "Progressive rock", "Rock and roll", "Psychedelic rock",
            "Rhythm and blues", "Soul music", "Reggae", "Country",
            "Swing music", "Bluegrass", "Funk", "Folk music", "Middle Eastern music",
            "Jazz", "Disco", "Classical music", "Opera", "Electronic music",
            "House music", "Techno", "Dubstep", "Drum and bass", "Electronica",
            "Electronic dance music", "Ambient music", "Trance music",
            "Music of Latin America", "Salsa music", "Flamenco", "Blues",
            "Music for children", "New-age music", "Vocal music",
            "A capella", "Music of Africa", "Afrobeat", "Christian music",
            "Gospel music", "Music of Asia", "Carnatic music",
            "Music of Bollywood", "Ska", "Traditional music", "Independent music",
            "Song", "Background music", "Theme music", "Jingle (music)",
            "Soundtrack music", "Lullaby", "Video game music",
            "Christmas music", "Dance music", "Wedding music",
            "Happy music", "Sad music", "Tender music", "Exciting music",
            "Angry music", "Scary music", "Wind", "Rustling leaves",
            "Wind noise (microphone)", "Thunderstorm", "Thunder", "Water",
            "Rain", "Raindrop", "Rain on surface", "Stream", "Waterfall",
            "Ocean", "Waves, surf", "Steam", "Gurgling", "Fire", "Crackle",
            "Vehicle", "Boat, Water vehicle", "Sailboat, sailing ship",
            "Rowboat, canoe, kayak", "Motorboat, speedboat", "Ship",
            "Motor vehicle (road)", "Car", "Vehicle horn, car horn, honking",
            "Toot", "Car alarm", "Power windows, electric windows",
            "Skidding", "Tire squeal", "Car passing by", "Race car, auto racing",
            "Truck", "Air brake", "Air horn, truck horn",
            "Reversing beeps", "Ice cream truck, ice cream van",
            "Bus", "Emergency vehicle", "Police car (siren)",
            "Ambulance (siren)", "Fire engine, fire truck (siren)",
            "Motorcycle", "Traffic noise, roadway noise",
            "Rail transport", "Train", "Train whistle", "Train horn",
            "Railroad car, train wagon", "Train wheels squealing",
            "Subway, metro, underground", "Aircraft", "Aircraft engine",
            "Jet engine", "Propeller, airscrew", "Helicopter",
            "Fixed-wing aircraft, airplane", "Bicycle", "Skateboard",
            "Engine", "Light engine (high frequency)",
            "Dental drill, dentist's drill", "Lawn mower",
            "Chainsaw", "Medium engine (mid frequency)",
            "Heavy engine (low frequency)", "Engine knocking",
            "Engine starting", "Idling", "Accelerating, revving, vroom",
            "Door", "Doorbell", "Ding-dong", "Sliding door",
            "Slam", "Knock", "Tap", "Squeak", "Cupboard open or close",
            "Drawer open or close", "Dishes, pots, and pans",
            "Cutlery, silverware", "Chopping (food)", "Frying (food)",
            "Microwave oven", "Blender", "Water tap, faucet",
            "Sink (filling or washing)", "Bathtub (filling or washing)",
            "Hair dryer", "Toilet flush", "Toothbrush",
            "Electric toothbrush", "Vacuum cleaner", "Zipper (clothing)",
            "Keys jangling", "Coin (dropping)", "Scissors", "Electric shaver, electric razor",
            "Shuffling cards", "Typing", "Typewriter", "Computer keyboard",
            "Writing", "Alarm", "Telephone", "Telephone bell ringing",
            "Ringtone", "Telephone dialing, DTMF", "Dial tone",
            "Busy signal", "Alarm clock", "Siren", "Civil defense siren",
            "Buzzer", "Smoke detector, smoke alarm", "Fire alarm",
            "Foghorn", "Whistle", "Steam whistle",
            "Mechanisms", "Ratchet, pawl", "Clock", "Tick", "Tick-tock",
            "Gears", "Pulleys", "Sewing machine", "Mechanical fan",
            "Air conditioning", "Cash register", "Printer",
            "Camera", "Single-lens reflex camera", "Tools",
            "Hammer", "Jackhammer", "Sawing", "Filing (rasp)",
            "Sanding", "Power tool", "Drill", "Explosion", "Gunshot, gunfire",
            "Machine gun", "Fusillade", "Artillery fire",
            "Cap gun", "Fireworks", "Firecracker", "Burst, pop",
            "Eruption", "Boom", "Wood", "Chop", "Splinter",
            "Crack", "Glass", "Chink, clink", "Shatter",
            "Liquid", "Splash, splatter", "Slosh", "Squish",
            "Drip", "Pour", "Trickle, dribble", "Gush", "Fill (with liquid)",
            "Spray", "Pump (liquid)", "Stir", "Boiling", "Sonar",
            "Arrow", "Whoosh, swoosh, swish", "Thump, thud", "Thunk",
            "Electronic tuner", "Effects unit", "Chorus effect",
            "Basketball bounce", "Bang", "Slap, smack", "Whack, thwack",
            "Smash, crash", "Breaking", "Bouncing", "Whip", "Flap",
            "Scratch", "Scrape", "Rub", "Roll", "Crushing",
            "Crumpling, crinkling", "Tearing", "Beep, bleep",
            "Ping", "Ding", "Clang", "Squeal", "Creak",
            "Rustle", "Whir", "Clatter", "Sizzle", "Clicking",
            "Clickety-clack", "Rumble", "Plop", "Jingle, tinkle",
            "Hum", "Zing", "Boing", "Crunch", "Silence", "Sine wave",
            "Harmonic", "Chirp tone", "Sound effect", "Pulse",
            "Inside, small room", "Inside, large room or hall",
            "Inside, public space", "Outside, urban or manmade",
            "Outside, rural or natural", "Reverberation", "Echo",
            "Noise", "Environmental noise", "Static",
            "Mains hum", "Distortion", "Sidetone", "Cacophony",
            "White noise", "Pink noise", "Throbbing", "Vibration", "Television",
            "Radio", "Field recording"
        ]

        return self.panns_labels

    def classify_audio(
        self,
        audio_path: str,
        stem_paths: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> TrackLayout:
        """
        分析音频并返回建议的轨道布局

        参数:
            audio_path: 原始音频路径
            stem_paths: 可选的分离轨道路径
            progress_callback: 可选的进度回调函数

        返回:
            建议的轨道布局
        """
        all_predictions = []

        # 如果有分离轨道，先分析它们
        if stem_paths:
            if progress_callback:
                progress_callback(0.0, "正在分析分离轨道...")

            demucs_predictions = self.classify_from_stems(
                stem_paths,
                lambda p, m: progress_callback(p * 0.5, m) if progress_callback else None
            )
            all_predictions.extend(demucs_predictions)

        # 使用 PANNs 细化
        if progress_callback:
            progress_callback(0.5, "正在使用 AI 细化识别...")

        panns_predictions = self.refine_with_panns(
            audio_path,
            stem_paths,
            lambda p, m: progress_callback(0.5 + p * 0.5, m) if progress_callback else None
        )
        all_predictions.extend(panns_predictions)

        # 生成轨道布局建议
        layout = self.suggest_track_layout(all_predictions)

        logger.info(f"建议轨道布局: {len(layout.tracks)} 个轨道")
        for track in layout.tracks:
            logger.info(f"  - {track.name} ({track.instrument.value})")

        return layout

    def analyze_piano_track_count(
        self,
        audio_path: str,
        notes: Optional[List] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> int:
        """
        分析钢琴音频并确定最佳轨道数

        通过分析以下特征来确定轨道数：
        1. 音高分布范围 - 宽范围可能需要更多轨道
        2. 同时发声的复音程度 - 高复音需要更多轨道
        3. 音高聚类分析 - 识别明显的分层

        参数:
            audio_path: 音频文件路径
            notes: 可选的已转写音符列表（如果已转写过）
            progress_callback: 可选的进度回调函数

        返回:
            建议的轨道数 (1-6)
        """
        if progress_callback:
            progress_callback(0.0, "正在分析钢琴轨道数...")

        try:
            import librosa
            import numpy as np

            # 如果没有提供音符，进行简单的音频分析
            if notes is None:
                return self._analyze_track_count_from_audio(audio_path, progress_callback)

            # 基于转写音符分析
            return self._analyze_track_count_from_notes(notes, progress_callback)

        except Exception as e:
            logger.warning(f"轨道数分析失败: {e}，使用默认值2")
            return 2

    def _analyze_track_count_from_audio(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> int:
        """通过音频频谱分析确定轨道数"""
        try:
            import librosa
            import numpy as np

            if progress_callback:
                progress_callback(0.2, "正在加载音频...")

            # 加载音频
            y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)

            if progress_callback:
                progress_callback(0.4, "正在分析频谱...")

            # 计算常数Q变换（更适合音乐分析）
            C = np.abs(librosa.cqt(y, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12))

            if progress_callback:
                progress_callback(0.6, "正在分析音高分布...")

            # 计算每个时间帧的活跃音符数
            threshold = np.percentile(C, 70)  # 只考虑较强的音符
            active_notes = np.sum(C > threshold, axis=0)

            # 计算平均复音度
            avg_polyphony = np.mean(active_notes)
            max_polyphony = np.percentile(active_notes, 95)  # 95百分位

            if progress_callback:
                progress_callback(0.8, "正在确定轨道数...")

            # 分析音高范围
            # 找出有能量的频率范围
            energy_per_bin = np.sum(C, axis=1)
            active_bins = np.where(energy_per_bin > np.percentile(energy_per_bin, 20))[0]

            if len(active_bins) > 0:
                pitch_range = active_bins[-1] - active_bins[0]  # 音高范围（半音数）
            else:
                pitch_range = 48  # 默认4个八度

            logger.info(f"音频分析: 平均复音度={avg_polyphony:.1f}, 最大复音度={max_polyphony:.1f}, 音高范围={pitch_range}半音")

            # 根据分析结果确定轨道数
            track_count = self._determine_track_count(avg_polyphony, max_polyphony, pitch_range)

            if progress_callback:
                progress_callback(1.0, f"建议使用 {track_count} 个轨道")

            return track_count

        except Exception as e:
            logger.warning(f"音频分析失败: {e}")
            return 2

    def _analyze_track_count_from_notes(
        self,
        notes: List,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> int:
        """通过转写的音符分析确定轨道数"""
        if not notes:
            return 2

        if progress_callback:
            progress_callback(0.3, "正在分析音符分布...")

        import numpy as np

        # 收集音高信息
        pitches = [n.pitch for n in notes]
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        pitch_range = max_pitch - min_pitch

        if progress_callback:
            progress_callback(0.5, "正在分析复音程度...")

        # 计算复音度（同时发声的音符数）
        # 将时间轴离散化
        time_resolution = 0.05  # 50ms
        max_time = max(n.end_time for n in notes)
        time_slots = int(max_time / time_resolution) + 1

        polyphony_count = np.zeros(time_slots)
        for note in notes:
            start_slot = int(note.start_time / time_resolution)
            end_slot = int(note.end_time / time_resolution)
            polyphony_count[start_slot:end_slot + 1] += 1

        avg_polyphony = np.mean(polyphony_count[polyphony_count > 0])
        max_polyphony = np.percentile(polyphony_count, 95)

        if progress_callback:
            progress_callback(0.8, "正在确定最佳轨道数...")

        logger.info(f"音符分析: 平均复音度={avg_polyphony:.1f}, 最大复音度={max_polyphony:.1f}, 音高范围={pitch_range}半音")

        track_count = self._determine_track_count(avg_polyphony, max_polyphony, pitch_range)

        if progress_callback:
            progress_callback(1.0, f"建议使用 {track_count} 个轨道")

        return track_count

    def _determine_track_count(
        self,
        avg_polyphony: float,
        max_polyphony: float,
        pitch_range: int
    ) -> int:
        """
        根据分析结果确定轨道数

        规则:
        - 1轨: 单旋律或极低复音度
        - 2轨: 标准钢琴（左右手）
        - 3轨: 中等复杂度（旋律+和弦+贝斯）
        - 4轨: 高复杂度（多层次编排）
        - 5-6轨: 极高复杂度（管弦乐改编等）
        """
        # 基于复音度评分
        if avg_polyphony < 1.5:
            polyphony_score = 1
        elif avg_polyphony < 3:
            polyphony_score = 2
        elif avg_polyphony < 5:
            polyphony_score = 3
        elif avg_polyphony < 8:
            polyphony_score = 4
        else:
            polyphony_score = 5

        # 基于音高范围评分 (MIDI: 每个八度12个半音)
        octaves = pitch_range / 12
        if octaves < 2:
            range_score = 1
        elif octaves < 3:
            range_score = 2
        elif octaves < 4:
            range_score = 3
        elif octaves < 5:
            range_score = 4
        else:
            range_score = 5

        # 基于最大复音度调整
        if max_polyphony > 10:
            max_poly_boost = 2
        elif max_polyphony > 6:
            max_poly_boost = 1
        else:
            max_poly_boost = 0

        # 综合评分
        total_score = (polyphony_score + range_score) / 2 + max_poly_boost

        # 映射到轨道数
        if total_score < 1.5:
            track_count = 1
        elif total_score < 2.5:
            track_count = 2
        elif total_score < 3.5:
            track_count = 3
        elif total_score < 4.5:
            track_count = 4
        elif total_score < 5.5:
            track_count = 5
        else:
            track_count = 6

        logger.info(f"轨道数评分: 复音度={polyphony_score}, 音高范围={range_score}, "
                   f"最大复音加成={max_poly_boost}, 总分={total_score:.1f}, 轨道数={track_count}")

        return track_count
