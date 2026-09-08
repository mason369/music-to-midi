import tempfile
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
                    "src.core.vocal_separator._run_leap_accompaniment_leg",
                    return_value=(accompaniment, 44_100),
                ) as polar_leg,
            ):
                outputs = VocalSeparator().separate(str(audio_path), str(output_dir))

            self.assertEqual(leap_leg.call_args.kwargs["audio_path"], str(audio_path))
            self.assertEqual(polar_leg.call_args.kwargs["audio_path"], str(audio_path))
            self.assertTrue(callable(leap_leg.call_args.kwargs["translate"]))
            self.assertTrue(callable(polar_leg.call_args.kwargs["translate"]))
            self.assertNotEqual(outputs["vocals"], outputs["accompaniment"])
            self.assertEqual(
                set(outputs),
                {"vocals", "accompaniment"},
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
                patch("src.core.vocal_separator._run_leap_accompaniment_leg") as polar_leg,
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
            def __init__(self):
                super().__init__()
                self.precision_contract_weight = torch.nn.Parameter(torch.ones(1))

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
            if event[0] == "progress" and "Loading Leap XE (vocals)" in event[1]
        )
        build_index = next(index for index, event in enumerate(events) if event[0] == "build")
        running_index = next(
            index
            for index, event in enumerate(events)
            if event[0] == "progress" and "Leap XE is processing vocals · chunk 1-2/5" in event[1]
        )
        forward_index = next(index for index, event in enumerate(events) if event[0] == "forward")

        self.assertLess(loading_index, build_index)
        self.assertLess(build_index, running_index)
        self.assertLess(running_index, forward_index)
        self.assertEqual("Loading Leap XE (vocals)...", events[loading_index][1])
        self.assertIn("about 0s each", events[running_index][1])


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
                    "src.core.vocal_separator._run_leap_accompaniment_leg",
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
        self.assertIn("正在保存 WAV", events[switching_index][1])
        self.assertIn("准备 Leap Instrumental（伴奏）", events[switching_index][1])



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

    def test_leap_xpu_uses_native_fp16_amp_and_single_chunk_batches(self):
        self.assertEqual(
            vocal_separator._resolve_leap_execution_settings("xpu", True, 2),
            (True, 1),
        )
        self.assertEqual(
            vocal_separator._resolve_leap_execution_settings("xpu", False, 2),
            (False, 1),
        )
        self.assertEqual(
            vocal_separator._resolve_leap_execution_settings("cuda", True, 2),
            (True, 2),
        )
        self.assertEqual(
            vocal_separator._resolve_leap_execution_settings("cpu", True, 2),
            (False, 2),
        )
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            vocal_separator._resolve_leap_execution_settings("xpu", True, 0)

    def test_leap_xpu_query_chunking_is_numerically_equivalent_and_inference_only(self):
        import torch

        class Attend(torch.nn.Module):
            def forward(self, q, k, v):
                scale = q.shape[-1] ** -0.5
                weights = torch.matmul(q, k.transpose(-1, -2)) * scale
                return torch.matmul(weights.softmax(dim=-1), v)

        class AttentionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = Attend()

        torch.manual_seed(7)
        model = AttentionModel().eval()
        q = torch.randn(2, 3, 11, 5)
        k = torch.randn(2, 3, 13, 5)
        v = torch.randn(2, 3, 13, 7)
        expected = model.attention(q, k, v)

        enabled = vocal_separator._enable_leap_xpu_exact_query_chunking(
            model,
            SimpleNamespace(type="xpu"),
            query_chunk_size=4,
            attention_module_type=Attend,
        )
        self.assertEqual(enabled, 1)
        torch.testing.assert_close(model.attention(q, k, v), expected)

        model.train()
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            model.attention(q, k, v)

    def test_leap_query_chunking_leaves_non_xpu_model_untouched(self):
        class Model:
            @staticmethod
            def modules():
                raise AssertionError("non-XPU path must not inspect the model")

        self.assertEqual(
            vocal_separator._enable_leap_xpu_exact_query_chunking(
                Model(),
                SimpleNamespace(type="cuda"),
            ),
            0,
        )



    def test_audio_chunk_progress_explains_duration_overlap_and_non_stage_semantics(self):
        kwargs = {
            "model": "Leap Instrumental",
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
            "Leap Instrumental 伴奏 · 分片 2/24 · 每片约 20s",
        )
        self.assertEqual(
            vocal_separator._audio_chunk_progress_message(
                Translator("en_US").t,
                **kwargs,
            ),
            "Leap Instrumental accompaniment · chunk 2/24 · about 20s each",
        )

    def test_leap_reference_demix_uses_configured_overlap_batching_and_reconstructs(self):
        import torch

        audio = np.linspace(-0.8, 0.8, 60, dtype=np.float32).reshape(2, 30)
        observed_batch_sizes = []
        progress_messages = []

        class IdentitySeparator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.precision_contract_weight = torch.nn.Parameter(torch.ones(1))

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
        self.assertEqual(
            "Leap XE is processing vocals · chunk 5/5 · about 0s each",
            progress_messages[-1],
        )
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
