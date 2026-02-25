# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
""" test.py """
import os
import pprint
import argparse
import torch

from utils.data_modules import AMTDataModule
from utils.task_manager import TaskManager
from model.init_train import initialize_trainer, update_config
from model.ymt3 import YourMT3
from config.data_presets import data_preset_single_cfg, data_preset_multi_cfg
from config.vocabulary import drum_vocab_presets
from utils.utils import str2bool

# yapf: disable
parser = argparse.ArgumentParser(description="YourMT3")
# General
parser.add_argument('exp_id', type=str, help='A unique identifier for the experiment is used to resume training. The "@" symbol can be used to load a specific checkpoint.')
parser.add_argument('-p', '--project', type=str, default='ymt3', help='project name')
parser.add_argument('-d', '--data-preset', type=str, default='musicnet_thickstun_ext_em', help='dataset preset (default=musicnet_thickstun_ext_em). See config/data.py for more options.')
# Audio configurations
parser.add_argument('-ac', '--audio-codec', type=str, default=None, help='audio codec (default=None). {"spec", "melspec"}. If None, default value defined in config.py will be used.')
parser.add_argument('-hop', '--hop-length', type=int, default=None, help='hop length in frames (default=None). {128, 300} 128 for MT3, 300 for PerceiverTFIf None, default value defined in config.py will be used.')
parser.add_argument('-nmel', '--n-mels', type=int, default=None, help='number of mel bins (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-if', '--input-frames', type=int, default=None, help='number of audio frames for input segment (default=None). If None, default value defined in config.py will be used.')
# Model configurations
parser.add_argument('-sqr', '--sca-use-query-residual', type=str2bool, default=None, help='sca use query residual flag. Default follows config.py')
parser.add_argument('-enc', '--encoder-type', type=str, default=None, help="Encoder type. 't5' or 'perceiver-tf' or 'conformer'. Default is 't5', following config.py.")
parser.add_argument('-dec', '--decoder-type', type=str, default=None, help="Decoder type. 't5' or 'multi-t5'. Default is 't5', following config.py.")
parser.add_argument('-preenc', '--pre-encoder-type', type=str, default='default', help="Pre-encoder type. None or 'conv' or 'default'. By default, t5_enc:None, perceiver_tf_enc:conv, conformer:None")
parser.add_argument('-predec', '--pre-decoder-type', type=str, default='default', help="Pre-decoder type. {None, 'linear', 'conv1', 'mlp', 'group_linear'} or 'default'. Default is {'t5': None, 'perceiver-tf': 'linear', 'conformer': None}.")
parser.add_argument('-cout', '--conv-out-channels', type=int, default=None, help='Number of filters for pre-encoder conv layer. Default follows "model_cfg" of config.py.')
parser.add_argument('-tenc', '--task-cond-encoder', type=str2bool, default=True, help='task conditional encoder (default=True). True or False')
parser.add_argument('-tdec', '--task-cond-decoder', type=str2bool, default=True, help='task conditional decoder (default=True). True or False')
parser.add_argument('-df', '--d-feat', type=int, default=None, help='Audio feature will be projected to this dimension for Q,K,V of T5 or K,V of Perceiver (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-pt', '--pretrained', type=str2bool, default=False, help='pretrained T5(default=False). True or False')
parser.add_argument('-b', '--base-name', type=str, default="google/t5-v1_1-small", help='base model name (default="google/t5-v1_1-small")')
parser.add_argument('-epe', '--encoder-position-encoding-type', type=str, default='default', help="Positional encoding type of encoder. By default, pre-defined PE for T5 or Perceiver-TF encoder in config.py. For T5: {'sinusoidal', 'trainable'}, conformer: {'rotary', 'trainable'}, Perceiver-TF: {'trainable', 'rope', 'alibi', 'alibit', 'None', '0', 'none', 'tkd', 'td', 'tk', 'kdt'}.")
parser.add_argument('-dpe', '--decoder-position-encoding-type', type=str, default='default', help="Positional encoding type of decoder. By default, pre-defined PE for T5 in config.py. {'sinusoidal', 'trainable'}.")
parser.add_argument('-twe', '--tie-word-embedding', type=str2bool, default=None, help='tie word embedding (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-el', '--event-length', type=int, default=None, help='event length (default=None). If None, default value defined in model cfg of config.py will be used.')
# Perceiver-TF configurations
parser.add_argument('-dl', '--d-latent', type=int, default=None, help='Latent dimension of Perceiver. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-nl', '--num-latents', type=int, default=None, help='Number of latents of Perceiver. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-dpm', '--perceiver-tf-d-model', type=int, default=None, help='Perceiver-TF d_model (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-npb', '--num-perceiver-tf-blocks', type=int, default=None, help='Number of blocks of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py.')
parser.add_argument('-npl', '--num-perceiver-tf-local-transformers-per-block', type=int, default=None, help='Number of local layers per block of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-npt', '--num-perceiver-tf-temporal-transformers-per-block', type=int, default=None, help='Number of temporal layers per block of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-atc', '--attention-to-channel', type=str2bool, default=None, help='Attention to channel flag of Perceiver-TF. On T5, this will be ignored (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-ln', '--layer-norm-type', type=str, default=None, help='Layer normalization type (default=None). {"layer_norm", "rms_norm"}. If None, default value defined in config.py will be used.')
parser.add_argument('-ff', '--ff-layer-type', type=str, default=None, help='Feed forward layer type (default=None). {"mlp", "moe", "gmlp"}. If None, default value defined in config.py will be used.')
parser.add_argument('-wf', '--ff-widening-factor', type=int, default=None, help='Feed forward layer widening factor for MLP/MoE/gMLP (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-nmoe', '--moe-num-experts', type=int, default=None, help='Number of experts for MoE (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-kmoe', '--moe-topk', type=int, default=None, help='Top-k for MoE (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-act', '--hidden-act', type=str, default=None, help='Hidden activation function (default=None). {"gelu", "silu", "relu", "tanh"}. If None, default value defined in config.py will be used.')
parser.add_argument('-rt', '--rotary-type', type=str, default=None, help='Rotary embedding type expressed in three letters. e.g. ppl: "pixel" for SCA and latents, "lang" for temporal transformer. If None, use config.')
parser.add_argument('-rk', '--rope-apply-to-keys', type=str2bool, default=None, help='Apply rope to keys (default=None). If None, use config.')
parser.add_argument('-rp', '--rope-partial-pe', type=str2bool, default=None, help='Whether to apply RoPE to partial positions (default=None). If None, use config.')
# Decoder configurations
parser.add_argument('-dff', '--decoder-ff-layer-type', type=str, default=None, help='Feed forward layer type of decoder (default=None). {"mlp", "moe", "gmlp"}. If None, default value defined in config.py will be used.')
parser.add_argument('-dwf', '--decoder-ff-widening-factor', type=int, default=None, help='Feed forward layer widening factor for decoder MLP/MoE/gMLP (default=None). If None, default value defined in config.py will be used.')
# Task and Evaluation configurations
parser.add_argument('-tk', '--task', type=str, default='mt3_full_plus', help='tokenizer type (default=mt3_full_plus). See config/task.py for more options.')
parser.add_argument('-epv', '--eval-program-vocab', type=str, default=None, help='evaluation vocabulary (default=None). If None, default vocabulary of the data preset will be used.')
parser.add_argument('-edv', '--eval-drum-vocab', type=str, default=None, help='evaluation vocabulary for drum (default=None). If None, default vocabulary of the data preset will be used.')
parser.add_argument('-etk', '--eval-subtask-key', type=str, default='default', help='evaluation subtask key (default=default). See config/task.py for more options.')
parser.add_argument('-t', '--onset-tolerance', type=float, default=0.05, help='onset tolerance (default=0.05).')
parser.add_argument('-os', '--test-octave-shift', type=str2bool, default=False, help='test optimal octave shift (default=False). True or False')
parser.add_argument('-w', '--write-model-output', type=str2bool, default=False, help='write model test output to file (default=False). True or False')
# Trainer configurations
parser.add_argument('-pr','--precision', type=str, default="bf16-mixed", help='precision (default="bf16-mixed") {32, 16, bf16, bf16-mixed}')
parser.add_argument('-st', '--strategy', type=str, default='auto', help='strategy (default=auto). auto or deepspeed or ddp')
parser.add_argument('-n', '--num-nodes', type=int, default=1, help='number of nodes (default=1)')
parser.add_argument('-g', '--num-gpus', type=str, default='auto', help='number of gpus (default="auto")')
parser.add_argument('-wb', '--wandb-mode', type=str, default=None, help='wandb mode for logging (default=None). "disabled" or "online" or "offline". If None, default value defined in config.py will be used.')
# Debug
parser.add_argument('-debug', '--debug-mode', type=str2bool, default=False, help='debug mode (default=False). True or False')
parser.add_argument('-tps', '--test-pitch-shift', type=int, default=None, help='use pitch shift when testing. debug-purpose only. (default=None). semitone in int.')
args = parser.parse_args()
# yapf: enable
if torch.__version__ >= "1.13":
    torch.set_float32_matmul_precision("high")
args.epochs = None

# Initialize trainer
trainer, wandb_logger, dir_info, shared_cfg = initialize_trainer(args, stage='test')

# Update config with args, including augmentation settings
shared_cfg, audio_cfg, model_cfg = update_config(args, shared_cfg, stage='test')


def main():
    # Data preset
    if args.data_preset in data_preset_single_cfg:
        # convert single preset into multi preset format
        data_preset = {
            "presets": [args.data_preset],
            "eval_vocab": data_preset_single_cfg[args.data_preset]["eval_vocab"],
        }
        for k in data_preset_single_cfg[args.data_preset].keys():
            if k in ["eval_drum_vocab", "add_pitch_class_metric"]:
                data_preset[k] = data_preset_single_cfg[args.data_preset][k]
    elif args.data_preset in data_preset_multi_cfg:
        data_preset = data_preset_multi_cfg[args.data_preset]
    else:
        raise ValueError("Invalid data preset")
    eval_drum_vocab = data_preset.get("eval_drum_vocab", None)

    if args.eval_drum_vocab != None:  # override eval_drum_vocab
        eval_drum_vocab = drum_vocab_presets[args.eval_drum_vocab]

    # Task manager
    tm = TaskManager(task_name=args.task,
                     max_shift_steps=int(shared_cfg["TOKENIZER"]["max_shift_steps"]),
                     debug_mode=args.debug_mode)
    print(f"Task: {tm.task_name}, Max Shift Steps: {tm.max_shift_steps}")

    results = []
    for i, preset in enumerate(data_preset["presets"]):
        # sdp: unpacking multi preset as a list of single presets
        sdp = {
            "presets": [preset],
            "eval_vocab": [data_preset["eval_vocab"][i]],
            "eval_drum_vocab": eval_drum_vocab,
        }
        for k in data_preset.keys():
            if k not in ["presets", "eval_vocab"]:
                sdp[k] = data_preset[k]

        dm = AMTDataModule(data_preset_multi=sdp, task_manager=tm, audio_cfg=audio_cfg)

        model = YourMT3(
            audio_cfg=audio_cfg,
            model_cfg=model_cfg,
            shared_cfg=shared_cfg,
            optimizer=None,
            task_manager=tm,  # tokenizer is a member of task_manager
            eval_subtask_key=args.eval_subtask_key,
            eval_vocab=args.eval_program_vocab if args.eval_program_vocab != None else sdp["eval_vocab"],
            eval_drum_vocab=sdp["eval_drum_vocab"],
            write_output_dir=dir_info["lightning_dir"] if args.write_model_output or args.test_octave_shift else None,
            onset_tolerance=float(args.onset_tolerance),
            add_pitch_class_metric=sdp.get("add_pitch_class_metric", None),
            test_optimal_octave_shift=args.test_octave_shift,
            test_pitch_shift_layer=args.test_pitch_shift)

        # load checkpoint & drop pitchshift from state_dict
        checkpoint = torch.load(dir_info["last_ckpt_path"])
        state_dict = checkpoint['state_dict']
        new_state_dict = {k: v for k, v in state_dict.items() if 'pitchshift' not in k}
        model.load_state_dict(new_state_dict, strict=False)
        # if args.test_pitch_shift is None:
        #     new_state_dict = {k: v for k, v in state_dict.items() if 'pitchshift' not in k}
        #     model.load_state_dict(new_state_dict, strict=False)
        # else:
        #     model.load_state_dict(state_dict, strict=False)

        results.append("-----------------------------------------------------------------")
        results.append(sdp)
        results.append(trainer.test(model, datamodule=dm))
        # TODO: directly load checkpoint including hyperparmeters https://lightning.ai/docs/pytorch/1.6.2/common/hyperparameters.html

    # save result
    pp = pprint.PrettyPrinter(indent=4)
    results_str = pp.pformat(results)
    result_file = os.path.join(dir_info["lightning_dir"],
                               f"result_{args.task}_{args.eval_subtask_key}_{args.data_preset}.json")
    with open(result_file, 'w') as f:
        f.write(results_str)
    print(f"Result is saved to {result_file}")


if __name__ == "__main__":
    main()
