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

        # 人声相关
        "Singing": InstrumentType.VOCALS,
        "Male singing": InstrumentType.VOCALS,
        "Female singing": InstrumentType.VOCALS,
        "Choir": InstrumentType.VOCALS,
        "Speech": InstrumentType.VOCALS,
        "Vocal music": InstrumentType.VOCALS,

        # 弦乐相关
        "Violin, fiddle": InstrumentType.STRINGS,
        "Cello": InstrumentType.STRINGS,
        "Viola": InstrumentType.STRINGS,
        "String section": InstrumentType.STRINGS,
        "Orchestral music": InstrumentType.STRINGS,
        "Bowed string instrument": InstrumentType.STRINGS,
    }

    # 最小能量阈值（用于判断轨道是否有内容）
    MIN_ENERGY_THRESHOLD = 0.001

    # 最小置信度阈值（低于此值的预测将被忽略）
    MIN_CONFIDENCE_THRESHOLD = 0.1

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
            from panns_inference import AudioTagging

            self.panns_model = AudioTagging(
                checkpoint_path=None,  # 使用默认模型
                device=str(self.device)
            )
            logger.info("PANNs 模型已加载")

        except ImportError:
            logger.warning("PANNs 未安装，将仅使用 Demucs 进行基础识别")
            self.panns_model = None
        except Exception as e:
            logger.warning(f"PANNs 加载失败: {e}，将仅使用 Demucs 进行基础识别")
            self.panns_model = None

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

        for i, (stem_name, stem_path) in enumerate(stem_paths.items()):
            if progress_callback:
                progress = (i + 1) / total_stems
                progress_callback(progress, f"正在分析 {stem_name} 轨道...")

            # 检查轨道是否有内容
            if not self._has_content(stem_path):
                logger.info(f"{stem_name} 轨道无内容，跳过")
                continue

            # 根据轨道名称确定基础乐器类型
            instrument = self._stem_to_instrument(stem_name)
            if instrument is None:
                continue

            # 计算轨道能量作为置信度的基础
            energy = self._calculate_energy(stem_path)
            base_confidence = min(1.0, energy * 10)  # 归一化

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

            # 加载音频
            audio, sr = librosa.load(audio_path, sr=32000, mono=True)

            # 使用 PANNs 进行预测
            clipwise_output, _ = self.panns_model.inference(audio[None, :])

            if progress_callback:
                progress_callback(0.8, "正在解析识别结果...")

            # 获取前 N 个预测
            top_k = 20
            top_indices = np.argsort(clipwise_output[0])[::-1][:top_k]

            # 从 PANNs 标签文件获取标签名称
            labels = self._get_panns_labels()

            for idx in top_indices:
                label = labels[idx] if idx < len(labels) else f"Unknown_{idx}"
                confidence = float(clipwise_output[0, idx])

                if confidence < self.MIN_CONFIDENCE_THRESHOLD:
                    continue

                # 检查是否在映射表中
                if label in self.PANNS_LABEL_MAPPING:
                    instrument = self.PANNS_LABEL_MAPPING[label]
                    predictions.append(InstrumentPrediction(
                        instrument=instrument,
                        confidence=confidence,
                        source="panns"
                    ))
                    logger.info(f"PANNs 检测到 {label} -> {instrument.value}: {confidence:.2f}")

            if progress_callback:
                progress_callback(1.0, "PANNs 分析完成")

        except Exception as e:
            logger.error(f"PANNs 分析失败: {e}")

        return predictions

    def suggest_track_layout(
        self,
        predictions: List[InstrumentPrediction],
        min_confidence: float = 0.3
    ) -> TrackLayout:
        """
        根据预测结果建议轨道布局

        参数:
            predictions: 乐器预测列表
            min_confidence: 最小置信度阈值

        返回:
            建议的轨道布局
        """
        # 按乐器类型聚合预测，取最高置信度
        instrument_scores: Dict[InstrumentType, float] = {}

        for pred in predictions:
            if pred.confidence < min_confidence:
                continue

            current_score = instrument_scores.get(pred.instrument, 0)
            # 使用加权平均，PANNs 的结果权重更高
            weight = 1.5 if pred.source == "panns" else 1.0
            new_score = max(current_score, pred.confidence * weight)
            instrument_scores[pred.instrument] = new_score

        # 按置信度排序
        sorted_instruments = sorted(
            instrument_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 创建轨道配置
        tracks = []
        channel = 0

        for instrument, score in sorted_instruments:
            if score < min_confidence:
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

        # 如果没有检测到任何乐器，返回默认的钢琴布局
        if not tracks:
            logger.warning("未检测到任何乐器，使用默认钢琴布局")
            return TrackLayout.default_piano(2)

        return TrackLayout(mode=ProcessingMode.SMART, tracks=tracks)

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

    def _has_content(self, audio_path: str) -> bool:
        """检查音频文件是否有内容（非静音）"""
        try:
            import soundfile as sf
            audio, _ = sf.read(audio_path)
            energy = np.mean(audio ** 2)
            return energy > self.MIN_ENERGY_THRESHOLD
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
