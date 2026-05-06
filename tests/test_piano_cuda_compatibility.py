import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import src.utils.gpu_utils as gpu_utils
from src.core.aria_amt_transcriber import AriaAmtTranscriber
from src.models.data_models import Config


NO_KERNEL_IMAGE = "CUDA error: no kernel image is available for execution on the device"


class PianoCudaCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self._reset_torch_cache()

    def tearDown(self):
        self._reset_torch_cache()

    @staticmethod
    def _reset_torch_cache():
        gpu_utils._torch_module = None
        gpu_utils._torch_checked = False

    def test_transkun_preflights_cuda_runtime_before_worker_process(self):
        from src.core.transkun_transcriber import TranskunTranscriber

        transcriber = TranskunTranscriber(Config(use_gpu=True))

        with patch(
            "src.core.transkun_transcriber.get_device",
            return_value="cuda:0",
        ), patch(
            "src.core.transkun_transcriber.ensure_cuda_runtime_compatibility",
            side_effect=RuntimeError("friendly cuda architecture error"),
        ) as ensure_compat:
            with self.assertRaisesRegex(RuntimeError, "friendly cuda architecture error"):
                transcriber._resolve_runtime_device()

        ensure_compat.assert_called_once_with("cuda:0")

    def test_bytedance_piano_preflights_cuda_runtime_before_model_load(self):
        from src.core.bytedance_piano_transcriber import ByteDancePianoTranscriber

        transcriber = ByteDancePianoTranscriber(Config(use_gpu=True))

        with patch(
            "src.core.bytedance_piano_transcriber.get_device",
            return_value="cuda:0",
        ), patch(
            "src.core.bytedance_piano_transcriber.ensure_cuda_runtime_compatibility",
            side_effect=RuntimeError("friendly cuda architecture error"),
        ) as ensure_compat:
            with self.assertRaisesRegex(RuntimeError, "friendly cuda architecture error"):
                transcriber._resolve_runtime_device()

        ensure_compat.assert_called_once_with("cuda:0")

    def test_aria_amt_rewrites_cuda_architecture_error_in_windows_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "piano-medium-double-1.0.safetensors"
            checkpoint_path.write_bytes(b"weights")
            transcriber = AriaAmtTranscriber(checkpoint_path=checkpoint_path)

            fake_torch = types.ModuleType("torch")
            fake_torch.bfloat16 = object()
            fake_torch.float = object()
            fake_cuda = types.ModuleType("torch.cuda")
            fake_cuda.is_available = lambda: True
            fake_cuda.is_bf16_supported = lambda: False
            fake_torch.cuda = fake_cuda

            fake_amt = types.ModuleType("amt")
            fake_audio = types.ModuleType("amt.audio")
            fake_audio.AudioTransform = Mock()
            fake_config = types.ModuleType("amt.config")
            fake_config.load_config = lambda: {
                "audio": {"sample_rate": 16000, "chunk_len": 1}
            }
            fake_inference = types.ModuleType("amt.inference")
            fake_transcribe = types.ModuleType("amt.inference.transcribe")
            fake_transcribe.MAX_BLOCK_LEN = 128
            fake_transcribe.STRIDE_FACTOR = 2
            fake_transcribe.CHUNK_LEN_MS = 1000
            fake_transcribe.LEN_MS = 2000
            fake_inference.transcribe = fake_transcribe

            with patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "torch.cuda": fake_cuda,
                    "amt": fake_amt,
                    "amt.audio": fake_audio,
                    "amt.config": fake_config,
                    "amt.inference": fake_inference,
                    "amt.inference.transcribe": fake_transcribe,
                },
            ), patch(
                "src.core.aria_amt_transcriber.ensure_cuda_runtime_compatibility",
                side_effect=RuntimeError("friendly cuda architecture error"),
            ) as ensure_compat, patch.object(
                transcriber,
                "_load_aria_model",
                side_effect=AssertionError("model should not load when CUDA preflight fails"),
            ):
                with self.assertRaisesRegex(RuntimeError, "friendly cuda architecture error") as cm:
                    transcriber._run_transcription_windows_single_file(
                        tmp_path / "song.wav",
                        tmp_path,
                    )

        self.assertIn("Aria-AMT 转写失败", str(cm.exception))
        ensure_compat.assert_called_once_with("cuda:0")

    def test_aria_amt_does_not_convert_cuda_chunk_failure_to_empty_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "piano-medium-double-1.0.safetensors"
            checkpoint_path.write_bytes(b"weights")
            transcriber = AriaAmtTranscriber(checkpoint_path=checkpoint_path)

            class FakeModel:
                def __init__(self):
                    self.decoder = types.SimpleNamespace(setup_cache=lambda **_kwargs: None)

                def cuda(self):
                    return self

                def eval(self):
                    return self

            class FakeAudioTransform:
                def cuda(self):
                    return self

            fake_tokenizer = types.SimpleNamespace(bos_tok=("bos",), eos_tok=("eos",))
            fake_torch = types.ModuleType("torch")
            fake_torch.bfloat16 = object()
            fake_torch.float = object()
            fake_cuda = types.ModuleType("torch.cuda")
            fake_cuda.is_available = lambda: True
            fake_cuda.is_bf16_supported = lambda: False
            fake_torch.cuda = fake_cuda

            fake_amt = types.ModuleType("amt")
            fake_audio = types.ModuleType("amt.audio")
            fake_audio.AudioTransform = FakeAudioTransform
            fake_config = types.ModuleType("amt.config")
            fake_config.load_config = lambda: {
                "audio": {"sample_rate": 16000, "chunk_len": 1}
            }
            fake_inference = types.ModuleType("amt.inference")
            fake_transcribe = types.ModuleType("amt.inference.transcribe")
            fake_transcribe.MAX_BLOCK_LEN = 128
            fake_transcribe.STRIDE_FACTOR = 2
            fake_transcribe.CHUNK_LEN_MS = 1000
            fake_transcribe.LEN_MS = 2000
            fake_transcribe._get_silent_intervals = lambda _segment: []
            fake_transcribe.process_segments = Mock(side_effect=RuntimeError(NO_KERNEL_IMAGE))
            fake_transcribe._process_silent_intervals = lambda sequence, **_kwargs: sequence
            fake_transcribe._truncate_seq = lambda sequence, *_args: sequence
            fake_transcribe._shift_onset = lambda sequence, _offset: sequence
            fake_inference.transcribe = fake_transcribe

            with patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "torch.cuda": fake_cuda,
                    "amt": fake_amt,
                    "amt.audio": fake_audio,
                    "amt.config": fake_config,
                    "amt.inference": fake_inference,
                    "amt.inference.transcribe": fake_transcribe,
                },
            ), patch(
                "src.core.aria_amt_transcriber.ensure_cuda_runtime_compatibility",
                return_value=None,
            ), patch.object(
                transcriber,
                "_load_aria_model",
                return_value=(FakeModel(), fake_tokenizer),
            ), patch.object(
                AriaAmtTranscriber,
                "_iter_windows_wav_segments",
                return_value=[object()],
            ):
                with self.assertRaises(RuntimeError) as cm:
                    transcriber._run_transcription_windows_single_file(
                        tmp_path / "song.wav",
                        tmp_path,
                    )

        message = str(cm.exception)
        self.assertIn("显卡架构", message)
        self.assertNotIn("推理结果为空或过短", message)
        self.assertNotIn(NO_KERNEL_IMAGE, message)


if __name__ == "__main__":
    unittest.main()
