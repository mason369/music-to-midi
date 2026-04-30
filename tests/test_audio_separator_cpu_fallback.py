import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.vocal_separator import VocalSeparator


_NO_KERNEL_IMAGE = RuntimeError(
    "CUDA error: no kernel image is available for execution on the device"
)


class _BaseFallbackSeparator:
    load_devices = []
    separate_devices = []

    def __init__(self, output_dir, model_file_dir, output_format):
        self.output_dir = Path(output_dir)
        self.model_file_dir = Path(model_file_dir)
        self.output_format = output_format
        self.torch_device = "cuda"
        self.torch_device_cpu = "cpu"
        self.onnx_execution_provider = ["CUDAExecutionProvider"]

    @classmethod
    def reset(cls):
        cls.load_devices = []
        cls.separate_devices = []

    def load_model(self, _model_name):
        device = str(self.torch_device)
        self.__class__.load_devices.append(device)
        if device != "cpu":
            raise _NO_KERNEL_IMAGE

    def separate(self, _audio_path):
        device = str(self.torch_device)
        self.__class__.separate_devices.append(device)
        if device != "cpu":
            raise _NO_KERNEL_IMAGE
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self._write_outputs()

    def _write_outputs(self):
        raise NotImplementedError


class _VocalSeparatorFallbackFake(_BaseFallbackSeparator):
    def _write_outputs(self):
        files = [
            self.output_dir / "song_(Vocals).wav",
            self.output_dir / "song_(Instrumental).wav",
        ]
        for path in files:
            path.write_bytes(b"wav")
        return [str(path) for path in files]


class AudioSeparatorCpuFallbackTests(unittest.TestCase):
    def test_vocal_separator_preflights_unsupported_cuda_and_retries_on_cpu(self):
        _VocalSeparatorFallbackFake.reset()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            audio_path.write_bytes(b"audio")
            out_dir = tmp_path / "out"

            with patch(
                "src.core.vocal_separator.get_separator_cls",
                return_value=_VocalSeparatorFallbackFake,
            ), patch(
                "src.utils.audio_separator_compat.get_device",
                return_value="cuda:0",
                create=True,
            ), patch(
                "src.utils.audio_separator_compat.ensure_cuda_runtime_compatibility",
                side_effect=_NO_KERNEL_IMAGE,
                create=True,
            ):
                result = VocalSeparator().separate(str(audio_path), str(out_dir))
                self.assertEqual(_VocalSeparatorFallbackFake.load_devices, ["cpu"])
                self.assertEqual(_VocalSeparatorFallbackFake.separate_devices, ["cpu"])
                self.assertTrue(Path(result["vocals"]).exists())
                self.assertTrue(Path(result["no_vocals"]).exists())

if __name__ == "__main__":
    unittest.main()
