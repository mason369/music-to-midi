import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.multi_stem_separator import SixStemSeparator
from src.core.vocal_separator import VocalSeparator


_NO_KERNEL_IMAGE = RuntimeError(
    "CUDA error: no kernel image is available for execution on the device"
)


class _BaseFallbackSeparator:
    load_devices = []
    separate_devices = []
    init_kwargs = []

    def __init__(self, output_dir, model_file_dir, output_format, **kwargs):
        self.output_dir = Path(output_dir)
        self.model_file_dir = Path(model_file_dir)
        self.output_format = output_format
        self.__class__.init_kwargs.append(kwargs)
        self.torch_device = "cuda"
        self.torch_device_cpu = "cpu"
        self.onnx_execution_provider = ["CUDAExecutionProvider"]

    @classmethod
    def reset(cls):
        cls.load_devices = []
        cls.separate_devices = []
        cls.init_kwargs = []

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


class _SixStemSeparatorFallbackFake(_BaseFallbackSeparator):
    def _write_outputs(self):
        files = []
        for stem in ("bass", "drums", "guitar", "piano", "vocals", "other"):
            path = self.output_dir / f"song_({stem}).wav"
            path.write_bytes(b"wav")
            files.append(str(path))
        return files


class _SixStemSeparatorNoFailFake(_SixStemSeparatorFallbackFake):
    def load_model(self, _model_name):
        self.__class__.load_devices.append(str(self.torch_device))

    def separate(self, _audio_path):
        self.__class__.separate_devices.append(str(self.torch_device))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self._write_outputs()


class AudioSeparatorCpuFallbackTests(unittest.TestCase):
    def test_vocal_separator_reports_unsupported_cuda_without_cpu_retry(self):
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
                with self.assertRaisesRegex(RuntimeError, "GPU 不兼容"):
                    VocalSeparator().separate(str(audio_path), str(out_dir))
                self.assertEqual(_VocalSeparatorFallbackFake.load_devices, [])
                self.assertEqual(_VocalSeparatorFallbackFake.separate_devices, [])

    def test_six_stem_separator_reports_unsupported_cuda_without_cpu_retry(self):
        _SixStemSeparatorFallbackFake.reset()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            audio_path.write_bytes(b"audio")
            out_dir = tmp_path / "out"

            with patch(
                "src.utils.audio_separator_compat.get_device",
                return_value="cuda:0",
                create=True,
            ), patch(
                "src.utils.audio_separator_compat.ensure_cuda_runtime_compatibility",
                side_effect=_NO_KERNEL_IMAGE,
                create=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "GPU 不兼容|CUDA"):
                    SixStemSeparator(
                        separator_cls=_SixStemSeparatorFallbackFake,
                        ensure_assets_fn=lambda **_kwargs: tmp_path / "model.ckpt",
                        cache_dir=tmp_path / "cache",
                    ).separate(str(audio_path), str(out_dir))
                self.assertEqual(_SixStemSeparatorFallbackFake.load_devices, [])
                self.assertEqual(_SixStemSeparatorFallbackFake.separate_devices, [])

    def test_six_stem_separator_forces_small_mdxc_segments(self):
        _SixStemSeparatorNoFailFake.reset()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio_path = tmp_path / "song.wav"
            audio_path.write_bytes(b"audio")
            out_dir = tmp_path / "out"

            SixStemSeparator(
                separator_cls=_SixStemSeparatorNoFailFake,
                ensure_assets_fn=lambda **_kwargs: tmp_path / "model.ckpt",
                cache_dir=tmp_path / "cache",
            ).separate(str(audio_path), str(out_dir))

        mdxc_params = _SixStemSeparatorNoFailFake.init_kwargs[0]["mdxc_params"]
        self.assertTrue(mdxc_params["override_model_segment_size"])
        self.assertEqual(mdxc_params["segment_size"], 128)
        self.assertEqual(mdxc_params["batch_size"], 1)

if __name__ == "__main__":
    unittest.main()
