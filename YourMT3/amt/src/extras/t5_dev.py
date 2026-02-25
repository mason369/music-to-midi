import torch
from transformers import T5Config
from model.t5mod import T5ForConditionalGeneration

a = {
    "architectures": ["T5ForConditionalGeneration"],
    "d_ff": 1024,  # size of the intermediate feed forward layer in each T5Block
    "d_kv": 64,  # d_kv has to be equal to d_model // num_heads.
    # "d_model": 512,  # encoder hiddnen size, defined by model_cfg
    "decoder_start_token_id": 0,
    "dense_act_fn": "gelu_new",
    # "dropout_rate": 0.05,  # can be overwritten by args in ymt3
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "initializer_factor": 1.0,
    # "is_encoder_decoder": True,
    "is_gated_act": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    # "num_decoder_layers": 8,
    "num_heads": 6,
    "num_layers": 8,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "use_cache": True,
    "vocab_size": 1391  # vocab_size is automatically set by the task manager...
}
cfg = T5Config(**a)
cfg.num_decoder_layers = 4
cfg.num_layers = 0

model = T5ForConditionalGeneration(cfg)
print(model)

x = torch.rand(((2, 256, 512)))
out = model.encoder.forward(inputs_embeds=x)

enc_hs = torch.rand((2, 256, 512))
labels = torch.randint(0, 1391, (2, 256))
pred = model(encoder_outputs=(enc_hs,), labels=labels)  # important (enc_hs,) comma!
