"""
歌词识别模块 - 使用 Whisper 和 WhisperX
"""
import logging
from typing import List, Optional, Callable
import warnings

from src.models.data_models import Config, LyricEvent
from src.utils.gpu_utils import get_device, clear_gpu_memory

logger = logging.getLogger(__name__)

# 抑制 whisper 警告
warnings.filterwarnings("ignore", category=UserWarning)


class LyricsRecognizer:
    """
    使用 Whisper 和 WhisperX 进行歌词识别和对齐

    功能特点:
    - 自动语音识别
    - 单词级时间戳对齐
    - 多语言支持
    """

    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, config: Config):
        """
        初始化歌词识别器

        参数:
            config: 应用配置
        """
        self.config = config
        self.model_size = config.whisper_model
        self.language = config.lyrics_language
        self.device = get_device(config.use_gpu, config.gpu_device)
        self.model = None
        self.align_model = None

    def load_model(self) -> None:
        """加载 Whisper 模型"""
        if self.model is not None:
            return

        logger.info(f"正在加载 Whisper 模型: {self.model_size}")

        try:
            import whisperx

            compute_type = "float16" if "cuda" in self.device else "int8"

            self.model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type=compute_type
            )

            logger.info(f"Whisper 模型已加载到 {self.device}")

        except ImportError as e:
            logger.error("WhisperX 未安装，请运行: pip install whisperx")
            raise ImportError("歌词识别需要 WhisperX 库") from e

    def unload_model(self) -> None:
        """卸载模型以释放内存"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.align_model is not None:
            del self.align_model
            self.align_model = None

        clear_gpu_memory()
        logger.info("Whisper 模型已卸载")

    def recognize(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[LyricEvent]:
        """
        从音频识别歌词并获取单词级时间戳

        参数:
            audio_path: 音频文件路径（最好是分离后的人声）
            progress_callback: 可选的进度回调

        返回:
            带时间戳的 LyricEvent 对象列表
        """
        import whisperx

        self.load_model()

        if progress_callback:
            progress_callback(0.0, "正在加载音频...")

        # 加载音频
        logger.info(f"正在识别歌词: {audio_path}")
        audio = whisperx.load_audio(audio_path)

        if progress_callback:
            progress_callback(0.2, "正在转录...")

        # 转录
        result = self.model.transcribe(
            audio,
            batch_size=16,
            language=self.language
        )

        detected_language = result.get("language", "en")
        logger.info(f"检测到语言: {detected_language}")

        if progress_callback:
            progress_callback(0.5, "正在对齐单词...")

        # 加载检测到语言的对齐模型
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=self.device
            )
        except Exception as e:
            logger.warning(f"无法加载对齐模型: {e}")
            # 回退到段落级时间戳
            return self._extract_segment_lyrics(result)

        # 对齐
        aligned = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False
        )

        if progress_callback:
            progress_callback(0.9, "正在处理歌词...")

        # 提取歌词
        lyrics = self._extract_word_lyrics(aligned)

        if progress_callback:
            progress_callback(1.0, f"发现 {len(lyrics)} 个单词")

        logger.info(f"识别了 {len(lyrics)} 个单词/音节")
        return lyrics

    def _extract_word_lyrics(self, aligned_result: dict) -> List[LyricEvent]:
        """从对齐结果中提取单词级歌词"""
        lyrics = []

        for segment in aligned_result.get("segments", []):
            for word_info in segment.get("words", []):
                word = word_info.get("word", "").strip()
                start = word_info.get("start", 0)
                end = word_info.get("end", start + 0.1)
                score = word_info.get("score", 1.0)

                if word:
                    lyrics.append(LyricEvent(
                        text=word,
                        start_time=float(start),
                        end_time=float(end),
                        confidence=float(score) if score else 1.0
                    ))

        # 按开始时间排序
        lyrics.sort(key=lambda l: l.start_time)

        return lyrics

    def _extract_segment_lyrics(self, result: dict) -> List[LyricEvent]:
        """提取段落级歌词（回退方案）"""
        lyrics = []

        for segment in result.get("segments", []):
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            end = segment.get("end", start + 1)

            if text:
                # 分割为单词
                words = text.split()
                duration = (end - start) / len(words) if words else end - start

                for i, word in enumerate(words):
                    word_start = start + i * duration
                    word_end = word_start + duration

                    lyrics.append(LyricEvent(
                        text=word,
                        start_time=float(word_start),
                        end_time=float(word_end),
                        confidence=0.8  # 段落级置信度较低
                    ))

        return lyrics

    def get_full_text(self, lyrics: List[LyricEvent]) -> str:
        """从事件中获取完整歌词文本"""
        return " ".join(lyric.text for lyric in lyrics)
