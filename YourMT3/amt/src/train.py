# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
import argparse
# from packaging.version import parse as VersionParse

import torch
from utils.data_modules import AMTDataModule
from utils.task_manager import TaskManager
from model.init_train import initialize_trainer, update_config
from model.ymt3 import YourMT3
from config.data_presets import data_preset_single_cfg, data_preset_multi_cfg
from config.vocabulary import program_vocab_presets
from utils.utils import str2bool

# yapf: disable
parser = argparse.ArgumentParser(description="YourMT3")
# General
parser.add_argument('exp_id', type=str, help='A unique identifier for the experiment is used to resume training. The "@" symbol can be used to load a specific checkpoint.')
parser.add_argument('-p', '--project', type=str, default='ymt3', help='project name')
parser.add_argument('-d', '--data-preset', type=str, default='musicnet_thickstun_ext_em', help='dataset preset (default=musicnet_thickstun_ext_em). See config/data.py for more options.')
# Intra-stem augmentation
parser.add_argument('-amp', '--random-amp-range', nargs=2, type=float, default=None, help='random amp range for audio augmentation (default=None, using default value defined in config.py). In command line, use -amp 0.6 1.2')
parser.add_argument('-iaug', '--stem-iaug-prob', type=float, default=None, help='intra-stem augmentation probability (default follow config.py). p=1.0 means no intra-stem augmentation (no stems are dropped)')
# Cross-stem augmentation policy
parser.add_argument('-xk', '--xaug-max-k', type=int, default=None, help='max number of external sources used for cross-stem augmentations. Default follows config.py.')
parser.add_argument('-xtau', '--xaug-tau', type=float, default=None, help='exponential decay rate for cross-stem augmentation. Default follows config.py')
parser.add_argument('-xalpha', '--xaug-alpha', type=float, default=None, help='shape parameter for Weibull distribution. set 1.0 for exponential. Default follows config.py')
parser.add_argument('-xnio', '--xaug-no-instr-overlap', type=str2bool, default=None, help='No instrument overlap flag. Default follows config.py')
parser.add_argument('-xndo', '--xaug-no-drum-overlap', type=str2bool, default=None, help='No drum overlap flag. Default follows config.py')
parser.add_argument('-xiaug', '--uhat-intra-stem_augment', type=str2bool, default=None, help='uhat intra-stem augmentation flag. Default follows config.py')
# Post-mix augmentation (post-mixing)
parser.add_argument('-ps', '--pitch-shift-range', nargs=2, type=int, default=None, help='pitch shift range in semitones (default=None). If None, default value defined in config.py. [0, 0] disables pitch shift. In command line, use -ps -2 2')
# Audio configurations
parser.add_argument('-ac', '--audio-codec', type=str, default=None, help='audio codec (default=None). {"spec", "melspec"}. If None, default value defined in config.py will be used.')
parser.add_argument('-hop', '--hop-length', type=int, default=None, help='hop length in frames (default=None). {128, 300} 128 for MT3, 300 for PerceiverTFIf None, default value defined in config.py will be used.')
parser.add_argument('-nmel', '--n-mels', type=int, default=None, help='number of mel bins (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-if', '--input-frames', type=int, default=None, help='number of audio frames for input segment (default=None). If None, default value defined in config.py will be used.')
# Model configurations
parser.add_argument('-sqr', '--sca-use-query-residual', type=str2bool, default=None, help='sca use query residual flag. Default follows config.py')
parser.add_argument('-enc', '--encoder-type', type=str, default=None, help="Encoder type. 't5' or 'perceiver-tf' or 'conformer'. Default is 't5', following config.py.")
parser.add_argument('-dec', '--decoder-type', type=str, default=None, help="Decoder type. 't5' or 'multi-t5'. Default is 't5', following config.py.")
parser.add_argument('-preenc', '--pre-encoder-type', type=str, default='default', help="Pre-encoder type. None or 'conv' or 'default' or 'conv1d_t' or 'conv1d_f' or 'hftt' or 'res3b_hftt'. By default, t5_enc:None, perceiver_tf_enc:conv, conformer:None")
parser.add_argument('-predec', '--pre-decoder-type', type=str, default='default', help="Pre-decoder type. {None, 'linear', 'conv1', 'mlp', 'group_linear'} or 'default'. Default is {'t5': None, 'perceiver-tf': 'linear', 'conformer': None}.")
parser.add_argument('-cout', '--conv-out-channels', type=int, default=None, help='Number of filters for pre-encoder conv layer. Default follows "model_cfg" of config.py.')
parser.add_argument('-tenc', '--task-cond-encoder', type=str2bool, default=True, help='task conditional encoder (default=True). True or False')
parser.add_argument('-tdec', '--task-cond-decoder', type=str2bool, default=True, help='task conditional decoder (default=True). True or False')
parser.add_argument('-df', '--d-feat', type=int, default=None, help='Audio feature will be projected to this dimension for Q,K,V of T5 or K,V of Perceiver (default=None). If None, default value defined in config.py will be used.')
parser.add_argument('-pt', '--pretrained', type=str2bool, default=False, help='pretrained T5(default=False). True or False')
parser.add_argument('-b', '--base-name', type=str, default="google/t5-v1_1-small", help='base model name (default="google/t5-v1_1-small")')
parser.add_argument('-edr', '--encoder-dropout-rate', type=float, default=None, help='encoder dropout rate (default=None). If None, default rate defined in config will be used.')
parser.add_argument('-ddr', '--decoder-dropout-rate', type=float, default=None, help='decoder dropout rate (default=None). If None, default rate defined in config will be used.')
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
parser.add_argument('-tk', '--task', type=str, default='mt3_full_plus', help='tokenizer type (default=gm_ext_plus). See config/task.py for more options.')
parser.add_argument('-epv', '--eval-program-vocab', type=str, default=None, help='evaluation program vocabulary (default=None). If None, default vocabulary of the data preset will be used.')
parser.add_argument('-w', '--write-model-output', type=str2bool, default=False, help='write model test output to file (default=False). True or False')
# Trainer configurations
parser.add_argument('-bsz', '--train-batch-size', nargs=2, type=int, default=None, help='train batch size for sub and local (default=None) per GPU. e.g. "-bsz 6 12". If None, default value defined in config.py will be used.')
parser.add_argument('-pr','--precision', type=str, default="bf16-mixed", help='precision (default="bf16-mixed") {32, 16, bf16, bf16-mixed}')
parser.add_argument('-st', '--strategy', type=str, default='auto', help='strategy (default=auto). auto or deepspeed or ddp')
parser.add_argument('-sb', '--sync-batchnorm', type=str2bool, default=False, help='sync batchnorm (default=True). True or False')
parser.add_argument('-se', '--train-num-samples-per-epoch', type=int, default=90000, help='number of samples per epoch (default=96000). If None, use the total number of files in multi datasets.')
parser.add_argument('-e', '--max-epochs', type=int, default=None, help='number of max epochs (default is None, which is 1000).')
parser.add_argument('-it', '--max-steps', type=int, default=-1, help='number of max steps (default is -1, disabled). This overrides the number of total steps defined in config.')
parser.add_argument('-vit', '--val-interval', type=int, default=None, help='validation interval (default=None). If None, use the check_val_every_n_epoch defined in config.py')
parser.add_argument('-lr', '--base-learning-rate', type=float, default=None, help='base learning rate (default is 1e-03 for AdamW, and auto for AdaFactor)')
parser.add_argument('-o', '--optimizer', type=str, default='AdamWScale', help='optimizer (default=AdamWScale) or AdaFactor or AdamW or CPUAdam or DAdaptAdam. Only check lowercase.')
parser.add_argument('-s', '--scheduler', type=str, default='cosine', help='scheduler name (default=legacy), constant or legacy or cosine')
parser.add_argument('-n', '--num-nodes', type=int, default=1, help='number of nodes (default=1)')
parser.add_argument('-g', '--num-gpus', type=str, default='auto', help='number of gpus (default="auto")')
parser.add_argument('-wb', '--wandb-mode', type=str, default=None, help='wandb mode for logging (default=None). "disabled" or "online" or "offline". If None, default value defined in config.py will be used.')
args = parser.parse_args()
# yapf: enable
if torch.__version__ >= "1.13":
    torch.set_float32_matmul_precision("high")

# Initialize trainer
trainer, wandb_logger, dir_info, shared_cfg = initialize_trainer(args, stage='train')

# Update config with args, including augmentation settings
shared_cfg, audio_cfg, model_cfg = update_config(args, shared_cfg, stage='train')


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
        raise ValueError(f"Invalid data preset: {args.data_preset}")

    # Task manager
    tm = TaskManager(task_name=args.task, max_shift_steps=int(shared_cfg["TOKENIZER"]["max_shift_steps"]))
    print(f"Task: {tm.task_name}, Max Shift Steps: {tm.max_shift_steps}")

    # Vocabulary for validation
    if args.eval_program_vocab != None:
        eval_program_vocab = program_vocab_presets[args.eval_program_vocab]
    else:
        eval_program_vocab = data_preset["eval_vocab"]
    eval_drum_vocab = data_preset.get("eval_drum_vocab", None)

    dm = AMTDataModule(data_preset_multi=data_preset,
                       task_manager=tm,
                       train_num_samples_per_epoch=args.train_num_samples_per_epoch,
                       audio_cfg=audio_cfg,
                       **shared_cfg["AUGMENTATION"])

    model = YourMT3(
        audio_cfg=audio_cfg,
        model_cfg=model_cfg,
        shared_cfg=shared_cfg,
        pretrained=args.pretrained,
        optimizer_name="CPUAdam" if "offload" in args.strategy.lower() else args.optimizer,
        scheduler_name=args.scheduler.lower(),
        base_lr=float(args.base_learning_rate) if args.base_learning_rate != None else None,
        max_steps=int(args.max_steps),
        task_manager=tm,  # tokenizer is a member of task_manager
        eval_vocab=eval_program_vocab,
        eval_drum_vocab=eval_drum_vocab,
        write_output_dir=dir_info["lightning_dir"] if args.write_model_output else None,
        add_pitch_class_metric=data_preset.get("add_pitch_class_metric", None))

    # if VersionParse(torch.__version__) >= VersionParse("2.1"):
    #     model = torch.compile(model, mode="reduce-overhead")

    # Logging config updated by args
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update({"audio_cfg": model.audio_cfg}, allow_val_change=True)
        wandb_logger.experiment.config.update({"model_cfg": model.model_cfg}, allow_val_change=True)
        wandb_logger.experiment.config.update(model.shared_cfg, allow_val_change=True)

    wandb_logger.watch(model, log='gradients', log_freq=5000)

    # last_ckpt_path can be None
    if dir_info["last_ckpt_path"] is not None:
        checkpoint = torch.load(dir_info["last_ckpt_path"])
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        trainer.fit(model, datamodule=dm)
    else:
        trainer.fit(model, ckpt_path=dir_info["last_ckpt_path"], datamodule=dm)


if __name__ == "__main__":
    main()
