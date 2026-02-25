# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""init_train.py"""
from typing import Tuple, Literal, Any
from copy import deepcopy
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from config.config import shared_cfg as default_shared_cfg
from config.config import audio_cfg as default_audio_cfg
from config.config import model_cfg as default_model_cfg
from config.config import DEEPSPEED_CFG


def initialize_trainer(args: argparse.Namespace,
                       stage: Literal['train', 'test'] = 'train') -> Tuple[pl.Trainer, WandbLogger, dict]:
    """Initialize trainer and logger"""
    shared_cfg = deepcopy(default_shared_cfg)

    # create save dir
    os.makedirs(shared_cfg["WANDB"]["save_dir"], exist_ok=True)

    # collecting specific checkpoint from exp_id with extension (@xxx where xxx is checkpoint name)
    if "@" in args.exp_id:
        args.exp_id, checkpoint_name = args.exp_id.split("@")
    else:
        checkpoint_name = "last.ckpt"

    # checkpoint dir
    lightning_dir = os.path.join(shared_cfg["WANDB"]["save_dir"], args.project, args.exp_id)

    # create logger
    if args.wandb_mode is not None:
        shared_cfg["WANDB"]["mode"] = str(args.wandb_mode)
    if shared_cfg["WANDB"].get("cache_dir", None) is not None:
        os.environ["WANDB_CACHE_DIR"] = shared_cfg["WANDB"].get("cache_dir")
        del shared_cfg["WANDB"]["cache_dir"]  # remove cache_dir from shared_cfg
    # wandb_logger = WandbLogger(log_model="all",
    #                            project=args.project,
    #                            id=args.exp_id,
    #                            allow_val_change=True,
    #                            **shared_cfg['WANDB'])
    wandb_logger = None

    # check if any checkpoint exists
    last_ckpt_path = os.path.join(lightning_dir, "checkpoints", checkpoint_name)
    if os.path.exists(os.path.join(last_ckpt_path)):
        print(f'Resuming from {last_ckpt_path}')
    elif stage == 'train':
        print(f'No checkpoint found in {last_ckpt_path}. Starting from scratch')
        last_ckpt_path = None
    else:
        raise ValueError(f'No checkpoint found in {last_ckpt_path}. Quit...')

    # add info
    dir_info = dict(lightning_dir=lightning_dir, last_ckpt_path=last_ckpt_path)

    # define checkpoint callback
    checkpoint_callback = ModelCheckpoint(**shared_cfg["CHECKPOINT"],)

    # define lr scheduler monitor callback
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # deepspeed strategy
    if args.strategy == 'deepspeed':
        strategy = pl.strategies.DeepSpeedStrategy(config=DEEPSPEED_CFG)

    # validation interval
    if stage == 'train' and args.val_interval is not None:
        shared_cfg["TRAINER"]["check_val_every_n_epoch"] = None
        shared_cfg["TRAINER"]["val_check_interval"] = int(args.val_interval)

    # define trainer
    sync_batchnorm = False
    if stage == 'train':
        # train batch size
        if args.train_batch_size is not None:
            train_sub_bsz = int(args.train_batch_size[0])
            train_local_bsz = int(args.train_batch_size[1])
            if train_local_bsz % train_sub_bsz == 0:
                shared_cfg["BSZ"]["train_sub"] = train_sub_bsz
                shared_cfg["BSZ"]["train_local"] = train_local_bsz
            else:
                raise ValueError(
                    f'Local batch size {train_local_bsz} must be divisible by sub batch size {train_sub_bsz}')

        # ddp strategy
        if args.strategy == 'ddp':
            args.strategy = 'ddp_find_unused_parameters_true'  # fix for conformer or pitchshifter having unused parameter issue

            # sync-batchnorm
            if args.sync_batchnorm is True:
                sync_batchnorm = True

    train_params = dict(**shared_cfg["TRAINER"],
                        devices=args.num_gpus if args.num_gpus == 'auto' else int(args.num_gpus),
                        num_nodes=int(args.num_nodes),
                        strategy=strategy if args.strategy == 'deepspeed' else args.strategy,
                        precision=args.precision,
                        max_epochs=args.max_epochs if stage == 'train' else None,
                        max_steps=args.max_steps if stage == 'train' else -1,
                        # logger=wandb_logger,
                        callbacks=[checkpoint_callback, lr_monitor],
                        sync_batchnorm=sync_batchnorm)
    trainer = pl.trainer.trainer.Trainer(**train_params)

    # # Update wandb logger (for DDP)
    # if trainer.global_rank == 0:
    #     wandb_logger.experiment.config.update(args, allow_val_change=True)

    return trainer, wandb_logger, dir_info, shared_cfg


def update_config(args, shared_cfg, stage: Literal['train', 'test'] = 'train'):
    """Update audio/model/shared configurations with args"""
    audio_cfg = default_audio_cfg
    model_cfg = default_model_cfg

    # Only update config when training
    if stage == 'train':
        # Augmentation parameters
        if args.random_amp_range is not None:
            shared_cfg["AUGMENTATION"]["train_random_amp_range"] = list(
                (float(args.random_amp_range[0]), float(args.random_amp_range[1])))
        if args.stem_iaug_prob is not None:
            shared_cfg["AUGMENTATION"]["train_stem_iaug_prob"] = float(args.stem_iaug_prob)

        if args.xaug_max_k is not None:
            shared_cfg["AUGMENTATION"]["train_stem_xaug_policy"]["max_k"] = int(args.xaug_max_k)
        if args.xaug_tau is not None:
            shared_cfg["AUGMENTATION"]["train_stem_xaug_policy"]["tau"] = float(args.xaug_tau)
        if args.xaug_alpha is not None:
            shared_cfg["AUGMENTATION"]["train_stem_xaug_policy"]["alpha"] = float(args.xaug_alpha)
        if args.xaug_no_instr_overlap is not None:
            shared_cfg["AUGMENTATION"]["train_stem_xaug_policy"]["no_instr_overlap"] = bool(args.xaug_no_instr_overlap)
        if args.xaug_no_drum_overlap is not None:
            shared_cfg["AUGMENTATION"]["train_stem_xaug_policy"]["no_drum_overlap"] = bool(args.xaug_no_drum_overlap)
        if args.uhat_intra_stem_augment is not None:
            shared_cfg["AUGMENTATION"]["train_stem_xaug_policy"]["uhat_intra_stem_augment"] = bool(
                args.uhat_intra_stem_augment)

        if args.pitch_shift_range is not None:
            if args.pitch_shift_range in [["0", "0"], [0, 0]]:
                shared_cfg["AUGMENTATION"]["train_pitch_shift_range"] = None
            else:
                shared_cfg["AUGMENTATION"]["train_pitch_shift_range"] = list(
                    (int(args.pitch_shift_range[0]), int(args.pitch_shift_range[1])))

        train_stem_iaug_prob = shared_cfg["AUGMENTATION"]["train_stem_iaug_prob"]
        random_amp_range = shared_cfg["AUGMENTATION"]["train_random_amp_range"]
        train_stem_xaug_policy = shared_cfg["AUGMENTATION"]["train_stem_xaug_policy"]
        print(f'Random amp range: {random_amp_range}\n' +
              f'Intra-stem augmentation probability: {train_stem_iaug_prob}\n' +
              f'Stem augmentation policy: {train_stem_xaug_policy}\n' +
              f'Pitch shift range: {shared_cfg["AUGMENTATION"]["train_pitch_shift_range"]}\n')

    # Update audio config
    if args.audio_codec != None:
        assert args.audio_codec in ['spec', 'melspec']
        audio_cfg["codec"] = str(args.audio_codec)
    if args.hop_length != None:
        audio_cfg["hop_length"] = int(args.hop_length)
    if args.n_mels != None:
        audio_cfg["n_mels"] = int(args.n_mels)
    if args.input_frames != None:
        audio_cfg["input_frames"] = int(args.input_frames)

    # Update shared config
    if shared_cfg["TOKENIZER"]["max_shift_steps"] == "auto":
        shift_steps_ms = shared_cfg["TOKENIZER"]["shift_step_ms"]
        input_frames = audio_cfg["input_frames"]
        fs = audio_cfg["sample_rate"]
        max_shift_steps = (input_frames / fs) // (shift_steps_ms / 1000) + 2  # 206 by default
        shared_cfg["TOKENIZER"]["max_shift_steps"] = int(max_shift_steps)

    # Update model config
    if args.encoder_type != None:
        model_cfg["encoder_type"] = str(args.encoder_type)
    if args.decoder_type != None:
        model_cfg["decoder_type"] = str(args.decoder_type)
    if args.pre_encoder_type != "default":
        model_cfg["pre_encoder_type"] = str(args.pre_encoder_type)
    if args.pre_decoder_type != 'default':
        model_cfg["pre_decoder_type"] = str(args.pre_decoder_type)
    if args.conv_out_channels != None:
        model_cfg["conv_out_channels"] = int(args.conv_out_channels)
    assert isinstance(args.task_cond_decoder, bool) and isinstance(args.task_cond_encoder, bool)
    model_cfg["use_task_conditional_encoder"] = args.task_cond_encoder
    model_cfg["use_task_conditional_decoder"] = args.task_cond_decoder

    if args.encoder_position_encoding_type != 'default':
        if args.encoder_position_encoding_type in ['None', 'none', '0']:
            model_cfg["encoder"][model_cfg["encoder_type"]]["position_encoding_type"] = None
        elif args.encoder_position_encoding_type in [
                'sinusoidal', 'rope', 'trainable', 'alibi', 'alibit', 'tkd', 'td', 'tk', 'kdt'
        ]:
            model_cfg["encoder"][model_cfg["encoder_type"]]["position_encoding_type"] = str(
                args.encoder_position_encoding_type)
        else:
            raise ValueError(f'Encoder PE type {args.encoder_position_encoding_type} not supported')
    if args.decoder_position_encoding_type != 'default':
        if args.decoder_position_encoding_type in ['None', 'none', '0']:
            raise ValueError('Decoder PE type cannot be None')
        elif args.decoder_position_encoding_type in ['sinusoidal', 'trainable']:
            model_cfg["decoder"][model_cfg["decoder_type"]]["position_encoding_type"] = str(
                args.decoder_position_encoding_type)
        else:
            raise ValueError(f'Decoder PE {args.decoder_position_encoding_type} not supported')

    if args.tie_word_embedding is not None:
        model_cfg["tie_word_embedding"] = bool(args.tie_word_embedding)

    if args.d_feat != None:
        model_cfg["d_feat"] = int(args.d_feat)
    if args.d_latent != None:
        model_cfg['encoder']['perceiver-tf']["d_latent"] = int(args.d_latent)
    if args.num_latents != None:
        model_cfg['encoder']['perceiver-tf']['num_latents'] = int(args.num_latents)
    if args.perceiver_tf_d_model != None:
        model_cfg['encoder']['perceiver-tf']['d_model'] = int(args.perceiver_tf_d_model)
    if args.num_perceiver_tf_blocks != None:
        model_cfg["encoder"]["perceiver-tf"]["num_blocks"] = int(args.num_perceiver_tf_blocks)
    if args.num_perceiver_tf_local_transformers_per_block != None:
        model_cfg["encoder"]["perceiver-tf"]["num_local_transformers_per_block"] = int(
            args.num_perceiver_tf_local_transformers_per_block)
    if args.num_perceiver_tf_temporal_transformers_per_block != None:
        model_cfg["encoder"]["perceiver-tf"]["num_temporal_transformers_per_block"] = int(
            args.num_perceiver_tf_temporal_transformers_per_block)
    if args.attention_to_channel != None:
        model_cfg["encoder"]["perceiver-tf"]["attention_to_channel"] = bool(args.attention_to_channel)
    if args.sca_use_query_residual != None:
        model_cfg["encoder"]["perceiver-tf"]["sca_use_query_residual"] = bool(args.sca_use_query_residual)
    if args.layer_norm_type != None:
        model_cfg["encoder"]["perceiver-tf"]["layer_norm"] = str(args.layer_norm_type)
    if args.ff_layer_type != None:
        model_cfg["encoder"]["perceiver-tf"]["ff_layer_type"] = str(args.ff_layer_type)
    if args.ff_widening_factor != None:
        model_cfg["encoder"]["perceiver-tf"]["ff_widening_factor"] = int(args.ff_widening_factor)
    if args.moe_num_experts != None:
        model_cfg["encoder"]["perceiver-tf"]["moe_num_experts"] = int(args.moe_num_experts)
    if args.moe_topk != None:
        model_cfg["encoder"]["perceiver-tf"]["moe_topk"] = int(args.moe_topk)
    if args.hidden_act != None:
        model_cfg["encoder"]["perceiver-tf"]["hidden_act"] = str(args.hidden_act)
    if args.rotary_type != None:
        assert len(
            args.rotary_type
        ) == 3, "rotary_type must be a 3-letter string (e.g. 'ppl': 'pixel' for SCA, 'pixel' for latent, 'lang' for temporal transformer)"
        model_cfg["encoder"]["perceiver-tf"]["rotary_type_sca"] = str(args.rotary_type)[0]
        model_cfg["encoder"]["perceiver-tf"]["rotary_type_latent"] = str(args.rotary_type)[1]
        model_cfg["encoder"]["perceiver-tf"]["rotary_type_temporal"] = str(args.rotary_type)[2]
    if args.rope_apply_to_keys != None:
        model_cfg["encoder"]["perceiver-tf"]["rope_apply_to_keys"] = bool(args.rope_apply_to_keys)
    if args.rope_partial_pe != None:
        model_cfg["encoder"]["perceiver-tf"]["rope_partial_pe"] = bool(args.rope_partial_pe)

    if args.decoder_ff_layer_type != None:
        model_cfg["decoder"][model_cfg["decoder_type"]]["ff_layer_type"] = str(args.decoder_ff_layer_type)
    if args.decoder_ff_widening_factor != None:
        model_cfg["decoder"][model_cfg["decoder_type"]]["ff_widening_factor"] = int(args.decoder_ff_widening_factor)

    if args.event_length != None:
        model_cfg["event_length"] = int(args.event_length)

    if stage == 'train':
        if args.encoder_dropout_rate != None:
            model_cfg["encoder"][model_cfg["encoder_type"]]["dropout_rate"] = float(args.encoder_dropout_rate)
        if args.decoder_dropout_rate != None:
            model_cfg["decoder"][model_cfg["decoder_type"]]["dropout_rate"] = float(args.decoder_dropout_rate)

    return shared_cfg, audio_cfg, model_cfg  # return updated configs
