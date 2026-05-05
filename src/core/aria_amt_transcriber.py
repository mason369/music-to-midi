"""Aria-AMT piano transcription wrapper."""

from __future__ import annotations

import importlib.util
import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import torchaudio

from src.i18n.translator import Translator
from src.utils.runtime_paths import get_aria_amt_dir, is_frozen_app

logger = logging.getLogger(__name__)

ARIA_AMT_CHECKPOINT_NAME = "piano-medium-double-1.0.safetensors"
ARIA_AMT_MODEL_CONFIG_NAME = "medium-double"
ARIA_AMT_CHECKPOINT_URL = (
    "https://huggingface.co/datasets/loubb/aria-midi/resolve/main/"
    "piano-medium-double-1.0.safetensors?download=true"
)
ARIA_AMT_CACHE_DIR = Path.home() / ".cache" / "music_ai_models" / "aria_amt"


class AriaAmtTranscriber:
    def __init__(self, checkpoint_path: Optional[Path] = None, language: str = Translator.DEFAULT_LANGUAGE):
        if checkpoint_path is None:
            checkpoint_path = self.default_checkpoint_path()
        self.checkpoint_path = Path(checkpoint_path)
        self._cancelled = False
        self._process: Optional[subprocess.Popen[str]] = None
        self._translator = Translator(language)

    def _pt(self, key: str, **kwargs) -> str:
        return self._translator.t(key, **kwargs)

    @staticmethod
    def default_checkpoint_path() -> Path:
        return get_aria_amt_dir() / ARIA_AMT_CHECKPOINT_NAME

    @staticmethod
    def is_available() -> bool:
        try:
            return importlib.util.find_spec("amt.run") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def is_model_available(self) -> bool:
        return self.checkpoint_path.exists() and self.checkpoint_path.stat().st_size > 0

    @staticmethod
    def _guess_output_midi(save_dir: Path, audio_path: Path) -> Optional[Path]:
        direct = save_dir / f"{audio_path.stem}.mid"
        if direct.exists():
            return direct
        midis = sorted(save_dir.glob("*.mid"))
        return midis[0] if midis else None

    @staticmethod
    def _format_missing_output_error(out_path: Path, temp_dir: Path) -> str:
        lines = [
            "Aria-AMT 未生成 MIDI 输出",
            f"期望输出: {out_path.resolve()}",
            f"临时输出目录: {temp_dir.resolve()}",
        ]
        if temp_dir.exists():
            entries = sorted(path.resolve() for path in temp_dir.iterdir())
            if entries:
                lines.append("临时输出目录内容:")
                lines.extend(f"  {entry}" for entry in entries[:20])
                if len(entries) > 20:
                    lines.append(f"  ... 另外 {len(entries) - 20} 个")
            else:
                lines.append("临时输出目录为空")
        else:
            lines.append("临时输出目录不存在")
        return "\n".join(lines)

    @staticmethod
    def _save_token_sequence_as_midi(tokenizer, sequence: list, save_path: Path) -> None:
        last_onset = None
        for token in reversed(sequence):
            if isinstance(token, tuple) and token[0] == "onset":
                last_onset = token[1]
                break
        if last_onset is None:
            raise RuntimeError("Aria-AMT 未生成有效的 onset token，无法保存 MIDI")

        midi_dict = tokenizer.detokenize(tokenized_seq=sequence, len_ms=last_onset)
        midi_dict.remove_redundant_pedals()
        midi = midi_dict.to_midi()
        midi.save(str(save_path))

    def _load_aria_model(self):
        from amt.config import load_model_config
        from amt.inference.model import AmtEncoderDecoder, ModelConfig
        from amt.tokenizer import AmtTokenizer
        from amt.utils import _load_weight

        tokenizer = AmtTokenizer()
        model_config = ModelConfig(**load_model_config(ARIA_AMT_MODEL_CONFIG_NAME))
        model_config.set_vocab_size(tokenizer.vocab_size)
        model = AmtEncoderDecoder(model_config)
        model_state = _load_weight(ckpt_path=str(self.checkpoint_path))

        normalized_state = {}
        for key, value in model_state.items():
            if key.startswith("_orig_mod."):
                normalized_state[key[len("_orig_mod."):]] = value
            else:
                normalized_state[key] = value
        model.load_state_dict(normalized_state)
        return model, tokenizer

    @staticmethod
    def _iter_windows_wav_segments(
        input_path: Path,
        sample_rate: int,
        chunk_len_seconds: int,
        stride_factor: int,
    ):
        import torch
        import torch.nn.functional as torch_functional

        waveform, original_sample_rate = torchaudio.load(str(input_path))
        waveform = waveform.mean(0)
        if original_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=original_sample_rate,
                new_freq=sample_rate,
            )

        chunk_samples = int(sample_rate * chunk_len_seconds)
        stride_samples = int(chunk_samples // stride_factor)
        if len(waveform) <= chunk_samples:
            yield torch_functional.pad(waveform, (0, chunk_samples - len(waveform)))
            return

        buffer = torch.tensor([], dtype=torch.float32)
        for start in range(0, len(waveform), stride_samples):
            stride_segment = waveform[start:start + stride_samples]
            if stride_segment.shape[0] < stride_samples:
                stride_segment = torch_functional.pad(
                    stride_segment,
                    (0, stride_samples - stride_segment.shape[0]),
                    mode="constant",
                    value=0.0,
                )

            buffer = torch.cat((buffer, stride_segment))
            if len(buffer) < chunk_samples:
                continue
            yield buffer[:chunk_samples]
            buffer = buffer[stride_samples:]

            if start + stride_samples >= len(waveform):
                break

        if len(buffer) > 0:
            yield torch_functional.pad(buffer, (0, chunk_samples - len(buffer)))

    def _run_transcription_windows_single_file(
        self,
        input_path: Path,
        temp_dir: Path,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        try:
            import torch
            from torch.cuda import is_available as cuda_is_available
            from torch.cuda import is_bf16_supported

            from amt.audio import AudioTransform
            from amt.config import load_config
            from amt.inference import transcribe as transcribe_module

            if not cuda_is_available():
                raise RuntimeError("CUDA device not found")

            model, tokenizer = self._load_aria_model()
            model.decoder.setup_cache(
                batch_size=1,
                max_seq_len=transcribe_module.MAX_BLOCK_LEN,
                dtype=torch.bfloat16 if is_bf16_supported() else torch.float,
            )
            model.cuda()
            model.eval()
            audio_transform = AudioTransform().cuda()
            audio_config = load_config()["audio"]

            sequence = [tokenizer.bos_tok]
            concat_sequence = [tokenizer.bos_tok]
            for index, audio_segment in enumerate(
                self._iter_windows_wav_segments(
                    input_path=input_path,
                    sample_rate=audio_config["sample_rate"],
                    chunk_len_seconds=audio_config["chunk_len"],
                    stride_factor=transcribe_module.STRIDE_FACTOR,
                )
            ):
                if self._cancelled:
                    raise InterruptedError("Aria-AMT 转写处理已取消")

                if progress_callback:
                    progress_callback(0.10 + min(index, 8) * 0.08, self._pt("progress.running_aria_amt"))

                init_index = len(sequence)
                silent_intervals = transcribe_module._get_silent_intervals(audio_segment)
                input_sequence = list(sequence)
                try:
                    (sequence,) = transcribe_module.process_segments(
                        tasks=[((audio_segment, sequence), 0)],
                        model=model,
                        audio_transform=audio_transform,
                        tokenizer=tokenizer,
                        logger=logger,
                    )
                    adjusted_sequence = transcribe_module._process_silent_intervals(
                        sequence,
                        intervals=silent_intervals,
                        tokenizer=tokenizer,
                    )
                    if len(adjusted_sequence) < len(sequence) - 15:
                        sequence = adjusted_sequence
                    next_sequence = transcribe_module._truncate_seq(
                        sequence,
                        transcribe_module.CHUNK_LEN_MS,
                        transcribe_module.LEN_MS - transcribe_module.CHUNK_LEN_MS,
                    )
                except Exception:
                    logger.info("Aria-AMT chunk reconciliation failed for %s", input_path, exc_info=True)
                    try:
                        sequence = transcribe_module._truncate_seq(
                            input_sequence,
                            transcribe_module.CHUNK_LEN_MS - 2,
                            transcribe_module.CHUNK_LEN_MS,
                        )
                    except Exception:
                        sequence = [tokenizer.bos_tok]
                    continue

                if sequence[-1] == tokenizer.eos_tok:
                    sequence = sequence[:-1]
                concat_sequence += transcribe_module._shift_onset(
                    sequence[init_index:],
                    index * transcribe_module.CHUNK_LEN_MS,
                )
                sequence = [tokenizer.bos_tok] if len(next_sequence) == 1 else next_sequence

            if len(concat_sequence) < 10:
                raise RuntimeError("Aria-AMT 推理结果为空或过短，未生成可保存的 MIDI")

            self._save_token_sequence_as_midi(
                tokenizer,
                concat_sequence,
                temp_dir / f"{input_path.stem}.mid",
            )
        except InterruptedError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Aria-AMT 转写失败:\n{exc}") from exc

    def _run_transcription_in_process(self, input_path: Path, temp_dir: Path) -> None:
        try:
            run_module = importlib.import_module("amt.run")
            run_module.transcribe(
                model_name=ARIA_AMT_MODEL_CONFIG_NAME,
                checkpoint_path=str(self.checkpoint_path),
                load_path=str(input_path),
                load_dir=None,
                save_dir=str(temp_dir),
                batch_size=1,
            )
        except Exception as exc:
            raise RuntimeError(f"Aria-AMT 转写失败:\n{exc}") from exc

    def _run_transcription_subprocess(self, input_path: Path, temp_dir: Path) -> None:
        command = [
            sys.executable,
            "-m",
            "amt.run",
            "transcribe",
            ARIA_AMT_MODEL_CONFIG_NAME,
            str(self.checkpoint_path),
            "-load_path",
            str(input_path),
            "-save_dir",
            str(temp_dir),
            "-bs",
            "1",
        ]

        logger.info("Running Aria-AMT transcription: %s", " ".join(command))
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._process = process
        try:
            stdout, stderr = process.communicate()
        finally:
            self._process = None

        if process.returncode != 0:
            raise RuntimeError(
                "Aria-AMT 转写失败:\n"
                f"{stdout}\n{stderr}"
            )

    def transcribe(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        input_path = Path(audio_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            raise RuntimeError(
                "Aria-AMT 未安装。请执行: "
                "python -m pip install git+https://github.com/EleutherAI/aria-amt.git"
            )
        if not self.is_model_available():
            raise RuntimeError(
                "Aria-AMT 模型权重缺失。请执行: "
                "python download_aria_amt_model.py"
            )

        temp_dir = out_path.parent / ".aria_amt_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(0.05, self._pt("progress.loading_aria_amt"))

        self._cancelled = False
        if platform.system() == "Windows":
            self._run_transcription_windows_single_file(input_path, temp_dir, progress_callback)
        elif is_frozen_app():
            self._run_transcription_in_process(input_path, temp_dir)
        else:
            self._run_transcription_subprocess(input_path, temp_dir)

        if self._cancelled:
            raise InterruptedError("Aria-AMT 转写处理已取消")

        midi_path = self._guess_output_midi(temp_dir, input_path)
        if midi_path is None or not midi_path.exists():
            raise RuntimeError(self._format_missing_output_error(out_path, temp_dir))

        shutil.move(str(midi_path), str(out_path))
        shutil.rmtree(temp_dir, ignore_errors=True)

        if progress_callback:
            progress_callback(1.0, self._pt("progress.aria_amt_complete"))

        logger.info("Aria-AMT output: %s", out_path)
        return str(out_path)

    def cancel(self) -> None:
        self._cancelled = True
        process = self._process
        if process is not None and process.poll() is None:
            logger.info("正在终止 Aria-AMT 子进程...")
            process.terminate()
