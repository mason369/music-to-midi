import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import T5Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOURMT3_SRC = PROJECT_ROOT / "YourMT3" / "amt" / "src"
if str(YOURMT3_SRC) not in sys.path:
    sys.path.insert(0, str(YOURMT3_SRC))

from model.t5mod import T5StackYMT3  # noqa: E402


class YourMT3TransformersCompatTests(unittest.TestCase):
    def _tiny_decoder_config(self) -> T5Config:
        config = T5Config(
            d_model=4,
            d_ff=8,
            num_layers=1,
            num_heads=1,
            dropout_rate=0.0,
            is_decoder=True,
            use_cache=True,
            num_max_positions=16,
        )
        config.ff_layer_type = "t5_gmlp"
        config.position_encoding_type = "sinusoidal"
        return config

    def test_yourmt3_import_does_not_leave_onnxruntime_loaded(self):
        self.assertNotIn("onnxruntime", sys.modules)

    def test_t5_stack_initializes_attention_layers_with_layer_index(self):
        stack = T5StackYMT3(self._tiny_decoder_config())

        self_attn = stack.block[0].layer[0].SelfAttention
        cross_attn = stack.block[0].layer[1].EncDecAttention

        self.assertEqual(self_attn.layer_idx, 0)
        self.assertEqual(cross_attn.layer_idx, 0)

    def test_t5_stack_defaults_missing_ff_layer_type_to_standard_t5_layer(self):
        config = self._tiny_decoder_config()
        delattr(config, "ff_layer_type")

        stack = T5StackYMT3(config)

        self.assertEqual(stack.block[0].ff_layer_type, "t5_gmlp")

    def test_t5_stack_provides_cache_position_for_transformers_448_cache_api(self):
        captured = {}

        class CaptureBlock(nn.Module):
            def forward(
                self,
                hidden_states,
                attention_mask=None,
                position_bias=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                encoder_decoder_position_bias=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
                cache_position=None,
            ):
                captured["cache_position"] = (
                    None if cache_position is None else cache_position.detach().cpu().tolist()
                )
                return hidden_states, past_key_value, None

        stack = T5StackYMT3(self._tiny_decoder_config())
        stack.block = nn.ModuleList([CaptureBlock()])

        past = (
            (
                torch.zeros(1, 1, 3, 4),
                torch.zeros(1, 1, 3, 4),
            ),
        )
        stack(
            inputs_embeds=torch.zeros(1, 1, 4),
            past_key_values=past,
            use_cache=True,
            return_dict=False,
        )

        self.assertEqual(captured["cache_position"], [3])

    def test_t5_stack_returns_legacy_cache_tensors_for_autoregressive_generation(self):
        stack = T5StackYMT3(self._tiny_decoder_config())

        outputs = stack(
            inputs_embeds=torch.zeros(1, 1, 4),
            use_cache=True,
            return_dict=False,
        )

        past_key_values = outputs[1]
        self.assertIsNotNone(past_key_values[0][0])
        self.assertEqual(past_key_values[0][0].shape[2], 1)

    def test_t5_stack_cross_attention_uses_cache_object_without_tuple_indexing(self):
        stack = T5StackYMT3(self._tiny_decoder_config())

        outputs = stack(
            inputs_embeds=torch.zeros(1, 1, 4),
            encoder_hidden_states=torch.zeros(1, 2, 4),
            use_cache=True,
            return_dict=False,
        )

        past_key_values = outputs[1]
        self.assertEqual(len(past_key_values[0]), 4)
        self.assertIsNotNone(past_key_values[0][2])


if __name__ == "__main__":
    unittest.main()
