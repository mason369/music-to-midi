import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.yourmt3_transcriber import YourMT3Transcriber
from src.models.data_models import YourMT3Model
from src.models.data_models import Config


class YourMT3ModelSelectionTests(unittest.TestCase):
    def test_availability_uses_selected_checkpoint_for_bound_instance(self):
        selected_model = YourMT3Model.YPTF_MOE_MULTI_PS.value
        fake_lightning = types.ModuleType("pytorch_lightning")
        transcriber = YourMT3Transcriber(Config(yourmt3_model=selected_model))

        with tempfile.TemporaryDirectory() as tmp_dir:
            selected_checkpoint = Path(tmp_dir) / "selected.ckpt"
            selected_checkpoint.touch()

            def fake_get_model_path(model_name):
                if model_name == selected_model:
                    return selected_checkpoint
                return None

            with patch.dict(sys.modules, {"pytorch_lightning": fake_lightning}), patch(
                "src.core.yourmt3_transcriber._import_torch", return_value=object()
            ), patch(
                "src.core.yourmt3_transcriber._get_yourmt3_amt_src_path",
                return_value=Path(tmp_dir),
            ), patch(
                "src.utils.yourmt3_downloader.get_model_path",
                side_effect=fake_get_model_path,
            ) as get_model_path:
                self.assertTrue(transcriber.is_selected_model_available())
                self.assertFalse(YourMT3Transcriber.is_available())

            self.assertEqual(
                [call.args[0] for call in get_model_path.call_args_list],
                [selected_model, "yptf_moe_multi_nops"],
            )

    def test_prepare_and_infer_passes_configured_yourmt3_model_to_loader(self):
        transcriber = YourMT3Transcriber(Config(yourmt3_model="yptf_moe_multi_nops"))

        with patch.object(transcriber, "_load_model", side_effect=InterruptedError("stop")) as load_model:
            with self.assertRaises(InterruptedError):
                transcriber._prepare_and_infer("unused.wav")

        self.assertEqual(load_model.call_args.kwargs["model_name"], "yptf_moe_multi_nops")

    def test_resolve_max_shift_steps_rejects_checkpoint_without_tokenizer_cfg(self):
        with self.assertRaisesRegex(RuntimeError, "TOKENIZER.max_shift_steps"):
            YourMT3Transcriber._resolve_max_shift_steps({}, None)

    def test_resolve_max_shift_steps_rejects_checkpoint_metadata_mismatch(self):
        task_manager = type("TaskManagerWithShift", (), {"max_shift_steps": 206})()

        with self.assertRaisesRegex(RuntimeError, "disagree"):
            YourMT3Transcriber._resolve_max_shift_steps(
                {"TOKENIZER": {"max_shift_steps": 127}}, task_manager
            )

    def test_resolve_max_shift_steps_uses_resolved_official_checkpoint_value(self):
        task_manager = type("TaskManagerWithShift", (), {"max_shift_steps": 206})()

        self.assertEqual(
            YourMT3Transcriber._resolve_max_shift_steps(
                {"TOKENIZER": {"max_shift_steps": 206}}, task_manager
            ),
            206,
        )

    def test_official_midi_vocab_must_be_initialized_by_model_loader(self):
        model_without_vocab = types.SimpleNamespace()

        with self.assertRaisesRegex(RuntimeError, "midi_output_inverse_vocab"):
            YourMT3Transcriber._ensure_midi_output_vocab(model_without_vocab)

    def test_legacy_official_modes_cap_batch_size_to_single_segment(self):
        for model in (
            YourMT3Model.YMT3_PLUS.value,
            YourMT3Model.YPTF_SINGLE_NOPS.value,
            YourMT3Model.YPTF_MULTI_PS.value,
        ):
            with self.subTest(model=model):
                self.assertEqual(
                    YourMT3Transcriber._cap_batch_size_for_model(model, 8),
                    1,
                )

    def test_moe_official_modes_cap_batch_size_to_two_segments(self):
        for model in (
            YourMT3Model.YPTF_MOE_MULTI_NOPS.value,
            YourMT3Model.YPTF_MOE_MULTI_PS.value,
        ):
            with self.subTest(model=model):
                self.assertEqual(
                    YourMT3Transcriber._cap_batch_size_for_model(model, 8),
                    2,
                )
                self.assertEqual(
                    YourMT3Transcriber._cap_batch_size_for_model(model, 1),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
