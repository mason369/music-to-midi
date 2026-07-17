import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.core import vocal_separator
from src.core.vocal_separator import (
    KARAOKE_REQUIRED_MODELS,
    ROFORMER_REQUIRED_MODELS,
    VocalSeparator,
)
from src.i18n.translator import Translator


class VocalSeparatorTwoLegTests(unittest.TestCase):
    def test_model_available_requires_leap_and_polarformer_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)

            with (
                patch(
                    "src.core.vocal_separator.get_audio_separator_model_dir",
                    return_value=cache_dir,
                ),
                patch(
                    "src.core.vocal_separator.is_vocal_model_available",
                    side_effect=[False, True, True],
                ),
                patch(
                    "src.core.vocal_separator.is_accompaniment_model_available",
                    side_effect=[False, True],
                ),
            ):
                self.assertFalse(VocalSeparator.is_model_available())
                self.assertFalse(VocalSeparator.is_model_available())
                self.assertTrue(VocalSeparator.is_model_available())

    def test_separate_runs_two_independent_legs_on_original_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "models"
            cache_dir.mkdir()
            for model_name in ROFORMER_REQUIRED_MODELS + KARAOKE_REQUIRED_MODELS:
                (cache_dir / model_name).write_bytes(b"asset")
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"input")
            output_dir = root / "out"

            vocals = np.full((2, 32), 0.25, dtype=np.float32)
            accompaniment = np.full((2, 32), -0.25, dtype=np.float32)
            with (
                patch(
                    "src.core.vocal_separator.get_audio_separator_model_dir",
                    return_value=cache_dir,
                ),
                patch(
                    "src.core.vocal_separator._resolve_verified_model_assets",
                    return_value=tuple(
                        cache_dir / name
                        for name in ROFORMER_REQUIRED_MODELS + KARAOKE_REQUIRED_MODELS
                    ),
                ),
                patch(
                    "src.core.vocal_separator._run_leap_vocals_leg",
                    return_value=(vocals, 44_100),
                ) as leap_leg,
                patch(
                    "src.core.vocal_separator._run_polarformer_accompaniment_leg",
                    return_value=(accompaniment, 44_100),
                ) as polar_leg,
            ):
                outputs = VocalSeparator().separate(str(audio_path), str(output_dir))

            self.assertEqual(leap_leg.call_args.kwargs["audio_path"], str(audio_path))
            self.assertEqual(polar_leg.call_args.kwargs["audio_path"], str(audio_path))
            self.assertTrue(callable(leap_leg.call_args.kwargs["translate"]))
            self.assertTrue(callable(polar_leg.call_args.kwargs["translate"]))
            self.assertEqual(outputs["no_vocals"], outputs["accompaniment"])
            self.assertNotEqual(outputs["vocals"], outputs["accompaniment"])
            self.assertEqual(
                set(outputs),
                {"vocals", "accompaniment", "no_vocals"},
            )
            for path in {Path(value) for value in outputs.values()}:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_polarformer_leg_is_not_run_when_leap_leg_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "models"
            cache_dir.mkdir()
            for model_name in ROFORMER_REQUIRED_MODELS + KARAOKE_REQUIRED_MODELS:
                (cache_dir / model_name).write_bytes(b"asset")
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"input")

            with (
                patch(
                    "src.core.vocal_separator.get_audio_separator_model_dir",
                    return_value=cache_dir,
                ),
                patch(
                    "src.core.vocal_separator._resolve_verified_model_assets",
                    return_value=tuple(
                        cache_dir / name
                        for name in ROFORMER_REQUIRED_MODELS + KARAOKE_REQUIRED_MODELS
                    ),
                ),
                patch(
                    "src.core.vocal_separator._run_leap_vocals_leg",
                    side_effect=RuntimeError("leap failed"),
                ),
                patch("src.core.vocal_separator._run_polarformer_accompaniment_leg") as polar_leg,
            ):
                with self.assertRaisesRegex(RuntimeError, "leap failed"):
                    VocalSeparator().separate(str(audio_path), str(root / "out"))
            polar_leg.assert_not_called()

    def test_leap_reports_model_and_batch_before_slow_calls(self):
        import torch

        class StopAfterFirstBatch(Exception):
            pass

        config = {
            "audio": {"sample_rate": 44_100, "chunk_size": 20},
            "model": {"num_stems": 1},
            "training": {"use_amp": True},
            "inference": {"num_overlap": 2, "batch_size": 2},
        }
        audio = np.zeros((2, 30), dtype=np.float32)
        events = []

        class BlockingModel(torch.nn.Module):
            def forward(self, _batch):
                events.append(("forward", ""))
                raise StopAfterFirstBatch

        def build_model(*_args, **_kwargs):
            events.append(("build", ""))
            return BlockingModel()

        def report_progress(_progress, message):
            events.append(("progress", message))

        with (
            patch("src.core.vocal_separator._load_yaml", return_value=config),
            patch("src.core.vocal_separator._load_stereo_audio", return_value=audio),
            patch(
                "src.core.vocal_separator._resolve_torch_device",
                return_value=torch.device("cpu"),
            ),
            patch(
                "src.core.vocal_separator._build_leap_model",
                side_effect=build_model,
            ),
            patch("src.core.vocal_separator.clear_gpu_memory", return_value=None),
        ):
            with self.assertRaises(StopAfterFirstBatch):
                vocal_separator._run_leap_vocals_leg(
                    audio_path="song.wav",
                    checkpoint_path=Path("bs_leap_xe_voc.ckpt"),
                    config_path=Path("config.yaml"),
                    requested_device="cpu",
                    progress_callback=report_progress,
                    translate=Translator("en_US").t,
                    cancel_check=lambda: None,
                )

        loading_index = next(
            index
            for index, event in enumerate(events)
            if event[0] == "progress" and "Loading the Leap XE (vocals) model" in event[1]
        )
        build_index = next(index for index, event in enumerate(events) if event[0] == "build")
        running_index = next(
            index
            for index, event in enumerate(events)
            if event[0] == "progress" and "Leap XE (vocals) audio chunk 1-2/5" in event[1]
        )
        forward_index = next(index for index, event in enumerate(events) if event[0] == "forward")

        self.assertLess(loading_index, build_index)
        self.assertLess(build_index, running_index)
        self.assertLess(running_index, forward_index)
        self.assertIn("bs_leap_xe_voc.ckpt", events[loading_index][1])
        self.assertIn("PyTorch · cpu · FP32 · batch=2", events[running_index][1])

    def test_polarformer_reports_model_and_chunk_before_slow_calls(self):
        class FakeOrtFailure(Exception):
            pass

        config = {
            "audio": {"sample_rate": 44_100},
            "model": {
                "stereo": True,
                "stft_n_fft": 4,
                "stft_hop_length": 2,
                "stft_win_length": 4,
                "stft_normalized": False,
            },
            "inference": {"chunk_size": 4, "num_overlap": 1},
        }
        audio = np.zeros((2, 4), dtype=np.float32)
        events = []

        class RunOptions:
            def __init__(self):
                self.terminate = False

        class Features:
            @staticmethod
            def numpy():
                return np.zeros((1,), dtype=np.float32)

        class Session:
            def __init__(self, *_args, **_kwargs):
                events.append(("session", ""))

            @staticmethod
            def get_inputs():
                return [SimpleNamespace(name="stft_features")]

            @staticmethod
            def get_providers():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

            @staticmethod
            def run(_outputs, _feed, run_options=None):
                events.append(("run", ""))
                self.assertIsNotNone(run_options)
                raise FakeOrtFailure("stop after observing progress order")

        fake_ort = SimpleNamespace(
            InferenceSession=Session,
            RunOptions=RunOptions,
        )

        def prepare_stft(*_args, **_kwargs):
            events.append(("prepare", ""))
            return Features(), object(), object(), 4

        def report_progress(_progress, message):
            events.append(("progress", message))

        with (
            patch("src.core.vocal_separator.activate_audio_separator_runtime"),
            patch("src.core.vocal_separator._load_yaml", return_value=config),
            patch("src.core.vocal_separator._load_stereo_audio", return_value=audio),
            patch(
                "src.core.vocal_separator._resolve_onnx_providers",
                return_value=[("CUDAExecutionProvider", {"device_id": 0})],
            ),
            patch.dict("sys.modules", {"onnxruntime": fake_ort}),
            patch(
                "src.core.vocal_separator._prepare_polar_stft",
                side_effect=prepare_stft,
            ),
        ):
            with self.assertRaisesRegex(
                FakeOrtFailure,
                "stop after observing progress order",
            ):
                vocal_separator._run_polarformer_accompaniment_leg(
                    audio_path="song.wav",
                    onnx_path=Path("bs_polarformer.onnx"),
                    config_path=Path("config.yaml"),
                    requested_device="cuda:0",
                    progress_callback=report_progress,
                    translate=Translator("en_US").t,
                    cancel_check=lambda: None,
                )

        loading_index = next(
            index
            for index, event in enumerate(events)
            if event[0] == "progress"
            and "Loading the PolarFormer (accompaniment) model" in event[1]
        )
        session_index = next(index for index, event in enumerate(events) if event[0] == "session")
        running_index = next(
            index
            for index, event in enumerate(events)
            if event[0] == "progress" and "PolarFormer (accompaniment) audio chunk 1/1" in event[1]
        )
        prepare_index = next(index for index, event in enumerate(events) if event[0] == "prepare")
        run_index = next(index for index, event in enumerate(events) if event[0] == "run")

        self.assertLess(loading_index, session_index)
        self.assertLess(session_index, running_index)
        self.assertLess(running_index, prepare_index)
        self.assertLess(prepare_index, run_index)
        self.assertIn("bs_polarformer.onnx", events[running_index][1])
        self.assertIn(
            "ONNX Runtime · CUDAExecutionProvider + CPUExecutionProvider",
            events[running_index][1],
        )

    def test_separate_reports_switch_before_saving_and_polarformer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"input")
            assets = tuple(
                root / name for name in ROFORMER_REQUIRED_MODELS + KARAOKE_REQUIRED_MODELS
            )
            audio = np.zeros((2, 8), dtype=np.float32)
            events = []

            def report_progress(_progress, message):
                events.append(("progress", message))

            def leap_leg(**_kwargs):
                events.append(("leap", ""))
                return audio, 44_100

            def polar_leg(**_kwargs):
                events.append(("polar", ""))
                return audio, 44_100

            def write_wav(path, *_args):
                events.append(("write", Path(path).name))

            with (
                patch(
                    "src.core.vocal_separator.get_audio_separator_model_dir",
                    return_value=root,
                ),
                patch(
                    "src.core.vocal_separator._resolve_verified_model_assets",
                    return_value=assets,
                ),
                patch(
                    "src.core.vocal_separator._run_leap_vocals_leg",
                    side_effect=leap_leg,
                ),
                patch(
                    "src.core.vocal_separator._run_polarformer_accompaniment_leg",
                    side_effect=polar_leg,
                ),
                patch(
                    "src.core.vocal_separator._write_and_validate_wav",
                    side_effect=write_wav,
                ),
                patch("src.core.vocal_separator.clear_gpu_memory", return_value=None),
            ):
                VocalSeparator(language="zh_CN").separate(
                    str(audio_path),
                    str(root / "out"),
                    progress_callback=report_progress,
                )

        leap_index = next(index for index, event in enumerate(events) if event[0] == "leap")
        switching_index = next(
            index
            for index, event in enumerate(events)
            if event[0] == "progress" and "Leap XE（人声）已完成" in event[1]
        )
        first_write_index = next(index for index, event in enumerate(events) if event[0] == "write")
        polar_index = next(index for index, event in enumerate(events) if event[0] == "polar")

        self.assertLess(leap_index, switching_index)
        self.assertLess(switching_index, first_write_index)
        self.assertLess(first_write_index, polar_index)
        self.assertIn("bs_polarformer.onnx", events[switching_index][1])

    def test_polarformer_cancel_terminates_active_onnx_run(self):
        config = {
            "audio": {"sample_rate": 44_100},
            "model": {
                "stereo": True,
                "stft_n_fft": 4,
                "stft_hop_length": 2,
                "stft_win_length": 4,
                "stft_normalized": False,
            },
            "inference": {"chunk_size": 4, "num_overlap": 1},
        }
        audio = np.zeros((2, 4), dtype=np.float32)
        run_started = threading.Event()
        observed = {}
        errors = []

        class FakeOrtFailure(Exception):
            pass

        class RunOptions:
            def __init__(self):
                self.terminate = False

        class Features:
            @staticmethod
            def numpy():
                return np.zeros((1,), dtype=np.float32)

        class BlockingSession:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def get_inputs():
                return [SimpleNamespace(name="stft_features")]

            @staticmethod
            def get_providers():
                return ["CUDAExecutionProvider"]

            @staticmethod
            def run(_outputs, _feed, run_options=None):
                observed["run_options"] = run_options
                run_started.set()
                deadline = time.monotonic() + 2.0
                while not run_options.terminate:
                    if time.monotonic() >= deadline:
                        raise AssertionError("cancel did not terminate the active ONNX run")
                    time.sleep(0.005)
                raise FakeOrtFailure("Exiting due to terminate flag being set to true.")

        fake_ort = SimpleNamespace(
            InferenceSession=BlockingSession,
            RunOptions=RunOptions,
        )

        separator = VocalSeparator()

        def run_leg():
            try:
                vocal_separator._run_polarformer_accompaniment_leg(
                    audio_path="song.wav",
                    onnx_path=Path("model.onnx"),
                    config_path=Path("config.yaml"),
                    requested_device="cuda:0",
                    progress_callback=None,
                    translate=separator._pt,
                    cancel_check=separator._check_cancelled,
                    active_run_options_callback=(separator._set_active_onnx_run_options),
                )
            except BaseException as exc:
                errors.append(exc)

        with (
            patch("src.core.vocal_separator.activate_audio_separator_runtime"),
            patch("src.core.vocal_separator._load_yaml", return_value=config),
            patch(
                "src.core.vocal_separator._load_stereo_audio",
                return_value=audio,
            ),
            patch(
                "src.core.vocal_separator._resolve_onnx_providers",
                return_value=["CUDAExecutionProvider"],
            ),
            patch.dict("sys.modules", {"onnxruntime": fake_ort}),
            patch(
                "src.core.vocal_separator._prepare_polar_stft",
                return_value=(Features(), object(), object(), 4),
            ),
        ):
            thread = threading.Thread(target=run_leg)
            thread.start()
            self.assertTrue(run_started.wait(1.0))
            separator.cancel()
            thread.join(1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InterruptedError)
        self.assertTrue(observed["run_options"].terminate)
        self.assertIsNone(separator._active_onnx_run_options)

    def test_polarformer_preserves_non_cancel_onnx_error(self):
        config = {
            "audio": {"sample_rate": 44_100},
            "model": {
                "stereo": True,
                "stft_n_fft": 4,
                "stft_hop_length": 2,
                "stft_win_length": 4,
                "stft_normalized": False,
            },
            "inference": {"chunk_size": 4, "num_overlap": 1},
        }
        audio = np.zeros((2, 4), dtype=np.float32)

        class FakeOrtFailure(Exception):
            pass

        class RunOptions:
            def __init__(self):
                self.terminate = False

        class Features:
            @staticmethod
            def numpy():
                return np.zeros((1,), dtype=np.float32)

        class FailingSession:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def get_inputs():
                return [SimpleNamespace(name="stft_features")]

            @staticmethod
            def get_providers():
                return ["CUDAExecutionProvider"]

            @staticmethod
            def run(_outputs, _feed, run_options=None):
                assert run_options is not None
                raise FakeOrtFailure("real ONNX inference failure")

        fake_ort = SimpleNamespace(
            InferenceSession=FailingSession,
            RunOptions=RunOptions,
        )

        separator = VocalSeparator()
        with (
            patch("src.core.vocal_separator.activate_audio_separator_runtime"),
            patch("src.core.vocal_separator._load_yaml", return_value=config),
            patch(
                "src.core.vocal_separator._load_stereo_audio",
                return_value=audio,
            ),
            patch(
                "src.core.vocal_separator._resolve_onnx_providers",
                return_value=["CUDAExecutionProvider"],
            ),
            patch.dict("sys.modules", {"onnxruntime": fake_ort}),
            patch(
                "src.core.vocal_separator._prepare_polar_stft",
                return_value=(Features(), object(), object(), 4),
            ),
        ):
            with self.assertRaisesRegex(
                FakeOrtFailure,
                "real ONNX inference failure",
            ):
                vocal_separator._run_polarformer_accompaniment_leg(
                    audio_path="song.wav",
                    onnx_path=Path("model.onnx"),
                    config_path=Path("config.yaml"),
                    requested_device="cuda:0",
                    progress_callback=None,
                    translate=separator._pt,
                    cancel_check=separator._check_cancelled,
                    active_run_options_callback=(separator._set_active_onnx_run_options),
                )

        self.assertIsNone(separator._active_onnx_run_options)

    def test_leap_chunk_starts_match_reference_partial_chunk_schedule(self):
        self.assertEqual(vocal_separator._leap_chunk_starts(100, 200, 50), [0, 50])
        self.assertEqual(
            vocal_separator._leap_chunk_starts(1_000, 400, 200),
            [0, 200, 400, 600, 800],
        )
        self.assertEqual(
            vocal_separator._leap_chunk_starts(1_050, 400, 200),
            [0, 200, 400, 600, 800, 1_000],
        )

    def test_audio_chunk_progress_explains_duration_overlap_and_non_stage_semantics(self):
        kwargs = {
            "model": "PolarFormer",
            "role_key": "progress.audio_chunk_role_accompaniment",
            "done": 2,
            "total": 24,
            "chunk_size": 882_000,
            "step": 441_000,
            "sample_rate": 44_100,
        }

        self.assertEqual(
            vocal_separator._audio_chunk_progress_message(
                Translator("zh_CN").t,
                **kwargs,
            ),
            "音频分片进度：PolarFormer 伴奏 2/24（约 20 秒/片，相邻重叠约 10 秒；"
            "不是处理阶段或重试）",
        )
        self.assertEqual(
            vocal_separator._audio_chunk_progress_message(
                Translator("en_US").t,
                **kwargs,
            ),
            "Audio chunk progress: PolarFormer accompaniment 2/24 "
            "(~20s/chunk, ~10s adjacent overlap; not workflow stages or retries)",
        )

    def test_leap_reference_demix_uses_configured_overlap_batching_and_reconstructs(self):
        import torch

        audio = np.linspace(-0.8, 0.8, 60, dtype=np.float32).reshape(2, 30)
        observed_batch_sizes = []
        progress_messages = []

        class IdentitySeparator(torch.nn.Module):
            def forward(self, batch):
                observed_batch_sizes.append(batch.shape[0])
                # audio-separator BSRoformer squeezes the stem axis when num_stems == 1.
                return batch

        config = {
            "audio": {"sample_rate": 44_100, "chunk_size": 20},
            "model": {"num_stems": 1},
            "training": {"use_amp": True},
            "inference": {"num_overlap": 2, "batch_size": 2},
        }
        with (
            patch("src.core.vocal_separator._load_yaml", return_value=config),
            patch("src.core.vocal_separator._load_stereo_audio", return_value=audio.copy()),
            patch(
                "src.core.vocal_separator._resolve_torch_device",
                return_value=torch.device("cpu"),
            ),
            patch(
                "src.core.vocal_separator._build_leap_model",
                return_value=IdentitySeparator(),
            ),
            patch("src.core.vocal_separator.clear_gpu_memory", return_value=None),
        ):
            vocals, sample_rate = vocal_separator._run_leap_vocals_leg(
                audio_path="song.wav",
                checkpoint_path=Path("model.ckpt"),
                config_path=Path("config.yaml"),
                requested_device="cpu",
                progress_callback=lambda _progress, message: progress_messages.append(message),
                translate=Translator("en_US").t,
                cancel_check=lambda: None,
            )

        self.assertEqual(sample_rate, 44_100)
        self.assertEqual(observed_batch_sizes, [2, 2, 1])
        self.assertIn("Running inference: Leap XE (vocals) audio chunk 5/5", progress_messages[-1])
        self.assertIn("model model.ckpt", progress_messages[-1])
        self.assertIn("not workflow stages or retries", progress_messages[-1])
        np.testing.assert_allclose(vocals, audio, atol=1e-6, rtol=0.0)

    def test_leap_reference_forward_preserves_length_and_zeroes_dc(self):
        from types import SimpleNamespace

        import torch

        class IdentityBandSplit(torch.nn.Module):
            def forward(self, features):
                return features.unsqueeze(-2)

        class IdentityComplexMask(torch.nn.Module):
            def forward(self, features):
                batch, frames = features.shape[:2]
                mask = torch.zeros((batch, frames, 12), device=features.device)
                mask[..., 0::2] = 1.0
                return mask

        model = SimpleNamespace(
            audio_channels=2,
            num_stems=1,
            stft_kwargs={
                "n_fft": 4,
                "hop_length": 2,
                "win_length": 4,
                "normalized": False,
            },
            stft_window_fn=lambda *, device: torch.hann_window(4, device=device),
            use_torch_checkpoint=False,
            skip_connection=False,
            band_split=IdentityBandSplit(),
            layers=torch.nn.ModuleList(),
            final_norm=torch.nn.Identity(),
            mask_estimators=torch.nn.ModuleList([IdentityComplexMask()]),
            zero_dc=True,
        )
        audio = torch.linspace(-1.0, 1.0, 14).reshape(1, 2, 7)
        captured = {}
        real_istft = torch.istft

        def inspect_istft(stft, **kwargs):
            captured["dc"] = stft[:, 0, :].detach().clone()
            captured["length"] = kwargs.get("length")
            return real_istft(stft, **kwargs)

        with patch("torch.istft", side_effect=inspect_istft):
            output = vocal_separator._leap_reference_forward(model, audio)

        self.assertEqual(tuple(output.shape), (1, 1, 2, 7))
        self.assertEqual(captured["length"], 7)
        self.assertTrue(torch.equal(captured["dc"], torch.zeros_like(captured["dc"])))


if __name__ == "__main__":
    unittest.main()
