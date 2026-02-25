# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""ymt3.py"""
import os
from typing import Union, Optional, Tuple, Dict, List, Any
from collections import Counter

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torchaudio  # for debugging audio
import pytorch_lightning as pl
import numpy as np
import wandb
from einops import rearrange

from transformers import T5Config
from model.t5mod import T5EncoderYMT3, T5DecoderYMT3, MultiChannelT5Decoder
from model.t5mod_helper import task_cond_dec_generate
from model.perceiver_mod import PerceiverTFEncoder
from model.perceiver_helper import PerceiverTFConfig
from model.conformer_mod import ConformerYMT3Encoder
from model.conformer_helper import ConformerYMT3Config
from model.lm_head import LMHead
from model.pitchshift_layer import PitchShiftLayer
from model.spectrogram import get_spectrogram_layer_from_audio_cfg
from model.conv_block import PreEncoderBlockRes3B
from model.conv_block import PreEncoderBlockHFTT, PreEncoderBlockRes3BHFTT  # added for hFTT-like pre-encoder
from model.projection_layer import get_projection_layer, get_multi_channel_projection_layer
from model.optimizers import get_optimizer
from model.lr_scheduler import get_lr_scheduler

from utils.note_event_dataclasses import Note
from utils.note2event import mix_notes
from utils.event2note import merge_zipped_note_events_and_ties_to_notes, DECODING_ERR_TYPES
from utils.metrics import compute_track_metrics
from utils.metrics import AMTMetrics
# from utils.utils import write_model_output_as_npy
from utils.utils import write_model_output_as_midi, create_inverse_vocab, write_err_cnt_as_json
from utils.utils import Timer
from utils.task_manager import TaskManager

from config.config import audio_cfg as default_audio_cfg
from config.config import model_cfg as default_model_cfg
from config.config import shared_cfg as default_shared_cfg
from config.config import T5_BASE_CFG


class YourMT3(pl.LightningModule):
    """YourMT3:
    
    Lightning wrapper for multi-task music transcription Transformer.
    
    """

    def __init__(
            self,
            audio_cfg: Optional[Dict] = None,
            model_cfg: Optional[Dict] = None,
            shared_cfg: Optional[Dict] = None,
            pretrained: bool = False,
            optimizer_name: str = 'adamwscale',
            scheduler_name: str = 'cosine',
            base_lr: float = None,  # None: 'auto' for AdaFactor, 1e-3 for constant, 1e-2 for cosine
            max_steps: Optional[int] = None,
            weight_decay: float = 0.0,
            init_factor: Optional[Union[str, float]] = None,
            task_manager: TaskManager = TaskManager(),
            eval_subtask_key: Optional[str] = "default",
            eval_vocab: Optional[Dict] = None,
            eval_drum_vocab: Optional[Dict] = None,
            write_output_dir: Optional[str] = None,
            write_output_vocab: Optional[Dict] = None,
            onset_tolerance: float = 0.05,
            add_pitch_class_metric: Optional[List[str]] = None,
            add_melody_metric_to_singing: bool = True,
            test_optimal_octave_shift: bool = False,
            test_pitch_shift_layer: Optional[str] = None,
            **kwargs: Any) -> None:
        super().__init__()
        if pretrained is True:
            raise NotImplementedError("Pretrained model is not supported in this version.")
        self.test_pitch_shift_layer = test_pitch_shift_layer  # debug only

        # Config
        if model_cfg is None:
            model_cfg = default_model_cfg  # default config, not overwritten by args of trainer
        if audio_cfg is None:
            audio_cfg = default_audio_cfg  # default config, not overwritten by args of trainer
        if shared_cfg is None:
            shared_cfg = default_shared_cfg  # default config, not overwritten by args of trainer

        # Spec Layer (need to define here to infer max token length)
        self.spectrogram, spec_output_shape = get_spectrogram_layer_from_audio_cfg(
            audio_cfg)  # can be spec or melspec; output_shape is (T, F)
        model_cfg["feat_length"] = spec_output_shape[0]  # T of (T, F)

        # Task manger and Tokens
        self.task_manager = task_manager
        self.max_total_token_length = self.task_manager.max_total_token_length

        # Task Conditioning
        self.use_task_cond_encoder = bool(model_cfg["use_task_conditional_encoder"])
        self.use_task_cond_decoder = bool(model_cfg["use_task_conditional_decoder"])

        # Select Encoder type, Model-specific Config
        assert model_cfg["encoder_type"] in ["t5", "perceiver-tf", "conformer"]
        assert model_cfg["decoder_type"] in ["t5", "multi-t5"]
        self.encoder_type = model_cfg["encoder_type"]  # {"t5", "perceiver-tf", "conformer"}
        self.decoder_type = model_cfg["decoder_type"]  # {"t5", "multi-t5"}
        encoder_config = model_cfg["encoder"][self.encoder_type]  # mutable
        decoder_config = model_cfg["decoder"][self.decoder_type]  # mutable

        # Positional Encoding
        if isinstance(model_cfg["num_max_positions"], str) and model_cfg["num_max_positions"] == 'auto':
            encoder_config["num_max_positions"] = int(model_cfg["feat_length"] +
                                                      self.task_manager.max_task_token_length + 10)
            decoder_config["num_max_positions"] = int(self.max_total_token_length + 10)
        else:
            assert isinstance(model_cfg["num_max_positions"], int)
            encoder_config["num_max_positions"] = model_cfg["num_max_positions"]
            decoder_config["num_max_positions"] = model_cfg["num_max_positions"]

        # Select Pre-Encoder and Pre-Decoder type
        if model_cfg["pre_encoder_type"] == "default":
            model_cfg["pre_encoder_type"] = model_cfg["pre_encoder_type_default"].get(model_cfg["encoder_type"], None)
        elif model_cfg["pre_encoder_type"] in [None, "none", "None", "0"]:
            model_cfg["pre_encoder_type"] = None
        if model_cfg["pre_decoder_type"] == "default":
            model_cfg["pre_decoder_type"] = model_cfg["pre_decoder_type_default"].get(model_cfg["encoder_type"]).get(
                model_cfg["decoder_type"], None)
        elif model_cfg["pre_decoder_type"] in [None, "none", "None", "0"]:
            model_cfg["pre_decoder_type"] = None
        self.pre_encoder_type = model_cfg["pre_encoder_type"]
        self.pre_decoder_type = model_cfg["pre_decoder_type"]

        # Pre-encoder
        self.pre_encoder = nn.Sequential()
        if self.pre_encoder_type in ["conv", "conv1d_t", "conv1d_f"]:
            kernel_size = (3, 3)
            avp_kernel_size = (1, 2)
            if self.pre_encoder_type == "conv1d_t":
                kernel_size = (3, 1)
            elif self.pre_encoder_type == "conv1d_f":
                kernel_size = (1, 3)
            self.pre_encoder.append(
                PreEncoderBlockRes3B(1,
                                     model_cfg["conv_out_channels"],
                                     kernel_size=kernel_size,
                                     avp_kernerl_size=avp_kernel_size,
                                     activation="relu"))
            pre_enc_output_shape = (spec_output_shape[0], spec_output_shape[1] // 2**3, model_cfg["conv_out_channels"]
                                   )  # (T, F, C) excluding batch dim
        elif self.pre_encoder_type == "hftt":
            self.pre_encoder.append(PreEncoderBlockHFTT())
            pre_enc_output_shape = (spec_output_shape[0], spec_output_shape[1], 128)  # (T, F, C) excluding batch dim
        elif self.pre_encoder_type == "res3b_hftt":
            self.pre_encoder.append(PreEncoderBlockRes3BHFTT())
            pre_enc_output_shape = (spec_output_shape[0], spec_output_shape[1] // 2**3, 128)
        else:
            pre_enc_output_shape = spec_output_shape  # (T, F) excluding batch dim

        # Auto-infer `d_feat` and `d_model`, `vocab_size`, and `num_max_positions`
        if isinstance(model_cfg["d_feat"], str) and model_cfg["d_feat"] == 'auto':
            if self.encoder_type == "perceiver-tf" and encoder_config["attention_to_channel"] is True:
                model_cfg["d_feat"] = pre_enc_output_shape[-2]  # TODO: better readablity
            else:
                model_cfg["d_feat"] = pre_enc_output_shape[-1]  # C of (T, F, C) or F or (T, F)

        if self.encoder_type == "perceiver-tf" and isinstance(encoder_config["d_model"], str):
            if encoder_config["d_model"] == 'q':
                encoder_config["d_model"] = encoder_config["d_latent"]
            elif encoder_config["d_model"] == 'kv':
                encoder_config["d_model"] = model_cfg["d_feat"]
            else:
                raise ValueError(f"Unknown d_model: {encoder_config['d_model']}")

        # # required for PerceiverTF with attention_to_channel option
        # if self.encoder_type == "perceiver-tf":
        #     if encoder_config["attention_to_channel"] is True:
        #         encoder_config["kv_dim"] = model_cfg["d_feat"]  # TODO: better readablity
        #     else:
        #         encoder_config["kv_dim"] = model_cfg["conv_out_channels"]

        if isinstance(model_cfg["vocab_size"], str) and model_cfg["vocab_size"] == 'auto':
            model_cfg["vocab_size"] = task_manager.num_tokens

        if isinstance(model_cfg["num_max_positions"], str) and model_cfg["num_max_positions"] == 'auto':
            model_cfg["num_max_positions"] = int(
                max(model_cfg["feat_length"], model_cfg["event_length"]) + self.task_manager.max_task_token_length + 10)

        # Pre-decoder
        self.pre_decoder = nn.Sequential()
        if self.encoder_type == "perceiver-tf" and self.decoder_type == "t5":
            t, f, c = pre_enc_output_shape  # perceiver-tf: (110, 128, 128) for 2s
            encoder_output_shape = (t, encoder_config["num_latents"], encoder_config["d_latent"])  # (T, K, D_source)
            decoder_input_shape = (t, decoder_config["d_model"])  # (T, D_target)
            proj_layer = get_projection_layer(input_shape=encoder_output_shape,
                                              output_shape=decoder_input_shape,
                                              proj_type=self.pre_decoder_type)
            self.pre_encoder_output_shape = pre_enc_output_shape
            self.encoder_output_shape = encoder_output_shape
            self.decoder_input_shape = decoder_input_shape
            self.pre_decoder.append(proj_layer)
        elif self.encoder_type in ["t5", "conformer"] and self.decoder_type == "t5":
            pass
        elif self.encoder_type == "perceiver-tf" and self.decoder_type == "multi-t5":
            # NOTE: this is experiemental, only for multi-channel decoding with 13 classes
            assert encoder_config["num_latents"] % decoder_config["num_channels"] == 0
            encoder_output_shape = (encoder_config["num_latents"], encoder_config["d_model"])
            decoder_input_shape = (decoder_config["num_channels"], decoder_config["d_model"])
            proj_layer = get_multi_channel_projection_layer(input_shape=encoder_output_shape,
                                                            output_shape=decoder_input_shape,
                                                            proj_type=self.pre_decoder_type)
            self.pre_decoder.append(proj_layer)
        else:
            raise NotImplementedError(
                f"Encoder type {self.encoder_type} and decoder type {self.decoder_type} is not implemented yet.")

        # Positional Encoding, Vocab, etc.
        if self.encoder_type in ["t5", "conformer"]:
            encoder_config["num_max_positions"] = decoder_config["num_max_positions"] = model_cfg["num_max_positions"]
        else:  # perceiver-tf uses separate positional encoding
            encoder_config["num_max_positions"] = model_cfg["feat_length"]
            decoder_config["num_max_positions"] = model_cfg["num_max_positions"]
        encoder_config["vocab_size"] = decoder_config["vocab_size"] = model_cfg["vocab_size"]

        # Print and save updated configs
        self.audio_cfg = audio_cfg
        self.model_cfg = model_cfg
        self.shared_cfg = shared_cfg
        self.save_hyperparameters()
        if self.global_rank == 0:
            print(self.hparams)

        # Encoder and Decoder and LM-head
        self.encoder = None
        self.decoder = None
        self.lm_head = LMHead(decoder_config, 1.0, model_cfg["tie_word_embeddings"])
        self.embed_tokens = nn.Embedding(decoder_config["vocab_size"], decoder_config["d_model"])
        self.embed_tokens.weight.data.normal_(mean=0.0, std=1.0)
        self.shift_right_fn = None
        self.set_encoder_decoder()  # shift_right_fn is also set here

        # Model as ModuleDict
        # self.model = nn.ModuleDict({
        #     "pitchshift": self.pitchshift,   # no grad; created in setup() only for training,
        #                                        and called by training_step()
        #     "spectrogram": self.spectrogram,  # no grad
        #     "pre_encoder": self.pre_encoder,
        #     "encoder": self.encoder,
        #     "pre_decoder": self.pre_decoder,
        #     "decoder": self.decoder,
        #     "embed_tokens": self.embed_tokens,
        #     "lm_head": self.lm_head,
        # })

        # Tables (for logging)
        columns = ['Ep', 'Track ID', 'Pred Events', 'Actual Events', 'Pred Notes', 'Actual Notes']
        self.sample_table = wandb.Table(columns=columns)

        # Output MIDI
        if write_output_dir is not None:
            if write_output_vocab is None:
                from config.vocabulary import program_vocab_presets
                self.midi_output_vocab = program_vocab_presets["gm_ext_plus"]
            else:
                self.midi_output_vocab = write_output_vocab
            self.midi_output_inverse_vocab = create_inverse_vocab(self.midi_output_vocab)

    def set_encoder_decoder(self) -> None:
        """Set encoder, decoder, lm_head and emb_tokens from self.model_cfg"""

        # Generate and update T5Config
        t5_basename = self.model_cfg["t5_basename"]
        if t5_basename in T5_BASE_CFG.keys():
            # Load from pre-defined config in config.py
            t5_config = T5Config(**T5_BASE_CFG[t5_basename])
        else:
            # Load from HuggingFace hub
            t5_config = T5Config.from_pretrained(t5_basename)

        # Create encoder, decoder, lm_head and embed_tokens
        if self.encoder_type == "t5":
            self.encoder = T5EncoderYMT3(self.model_cfg["encoder"]["t5"], t5_config)
        elif self.encoder_type == "perceiver-tf":
            perceivertf_config = PerceiverTFConfig()
            perceivertf_config.update(self.model_cfg["encoder"]["perceiver-tf"])
            self.encoder = PerceiverTFEncoder(perceivertf_config)
        elif self.encoder_type == "conformer":
            conformer_config = ConformerYMT3Config()
            conformer_config.update(self.model_cfg["encoder"]["conformer"])
            self.encoder = ConformerYMT3Encoder(conformer_config)

        if self.decoder_type == "t5":
            self.decoder = T5DecoderYMT3(self.model_cfg["decoder"]["t5"], t5_config)
        elif self.decoder_type == "multi-t5":
            self.decoder = MultiChannelT5Decoder(self.model_cfg["decoder"]["multi-t5"], t5_config)

        # `shift_right` function for decoding
        self.shift_right_fn = self.decoder._shift_right

    def setup(self, stage: str) -> None:
        # Defining metrics
        if self.hparams.eval_vocab is None:
            extra_classes_per_dataset = [None]
        else:
            extra_classes_per_dataset = [
                list(v.keys()) if v is not None else None for v in self.hparams.eval_vocab
            ]  # e.g. [['Piano'], ['Guitar'], ['Piano'], ['Piano', 'Strings', 'Winds'], None]

        # For direct addition of extra metrics using full metric name
        extra_metrics = None
        if self.hparams.add_melody_metric_to_singing is True:
            extra_metrics = ["melody_rpa_Singing Voice", "melody_rca_Singing Voice", "melody_oa_Singing Voice"]

        # Add pitch class metric
        if self.hparams.add_pitch_class_metric is not None:
            for sublist in extra_classes_per_dataset:
                for name in self.hparams.add_pitch_class_metric:
                    if sublist is not None and name in sublist:
                        sublist += [name + "_pc"]

        extra_classes_unique = list(
            set(item for sublist in extra_classes_per_dataset if sublist is not None
                for item in sublist))  # e.g. ['Strings', 'Winds', 'Guitar', 'Piano']
        dm = self.trainer.datamodule

        # Train/Vaidation-only
        if stage == "fit":
            self.val_metrics_macro = AMTMetrics(prefix=f'validation/macro_', extra_classes=extra_classes_unique)
            self.val_metrics = nn.ModuleList()  # val_metric is a list of AMTMetrics objects
            for i in range(dm.num_val_dataloaders):
                self.val_metrics.append(
                    AMTMetrics(prefix=f'validation/({dm.get_val_dataset_name(i)})',
                               extra_classes=extra_classes_per_dataset[i],
                               error_types=DECODING_ERR_TYPES))

            # Add pitchshift layer
            if self.shared_cfg["AUGMENTATION"]["train_pitch_shift_range"] in [None, [0, 0]]:
                self.pitchshift = None
            else:
                # torchaudio pitchshifter requires a dummy input for initialization in DDP
                input_shape = (self.shared_cfg["BSZ"]["train_local"], 1, self.audio_cfg["input_frames"])
                self.pitchshift = PitchShiftLayer(
                    pshift_range=self.shared_cfg["AUGMENTATION"]["train_pitch_shift_range"],
                    expected_input_shape=input_shape,
                    device=self.device)

        # Test-only
        elif stage == "test":
            # self.test_metrics_macro = AMTMetrics(
            #     prefix=f'test/macro_', extra_classes=extra_classes_unique)
            self.test_metrics = nn.ModuleList()
            for i in range(dm.num_test_dataloaders):
                self.test_metrics.append(
                    AMTMetrics(prefix=f'test/({dm.get_test_dataset_name(i)})',
                               extra_classes=extra_classes_per_dataset[i],
                               extra_metrics=extra_metrics,
                               error_types=DECODING_ERR_TYPES))

            # Test pitch shift layer: debug only
            if self.test_pitch_shift_layer is not None:
                self.test_pitch_shift_semitone = int(self.test_pitch_shift_layer)
                self.pitchshift = PitchShiftLayer(
                    pshift_range=[self.test_pitch_shift_semitone, self.test_pitch_shift_semitone])

    def configure_optimizers(self) -> None:
        """Configure optimizer and scheduler"""
        optimizer, base_lr = get_optimizer(models_dict=self.named_parameters(),
                                           optimizer_name=self.hparams.optimizer_name,
                                           base_lr=self.hparams.base_lr,
                                           weight_decay=self.hparams.weight_decay)

        if self.hparams.optimizer_name.lower() == 'adafactor' and self.hparams.base_lr == None:
            print("Using AdaFactor with auto learning rate and no scheduler")
            return [optimizer]
        if self.hparams.optimizer_name.lower() == 'dadaptadam':
            print("Using dAdaptAdam with auto learning rate and no scheduler")
            return [optimizer]
        elif self.hparams.base_lr == None:
            print(f"Using default learning rate {base_lr} of {self.hparams.optimizer_name} as base learning rate.")
            self.hparams.base_lr = base_lr

        scheduler_cfg = self.shared_cfg["LR_SCHEDULE"]
        if self.hparams.max_steps != -1:
            # overwrite total_steps
            scheduler_cfg["total_steps"] = self.hparams.max_steps
        _lr_scheduler = get_lr_scheduler(optimizer,
                                         scheduler_name=self.hparams.scheduler_name,
                                         base_lr=base_lr,
                                         scheduler_cfg=scheduler_cfg)

        lr_scheduler = {'scheduler': _lr_scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [lr_scheduler]

    def forward(
            self,
            x: torch.FloatTensor,
            target_tokens: torch.LongTensor,
            # task_tokens: Optional[torch.LongTensor] = None,
            **kwargs) -> Dict:
        """ Forward pass with teacher-forcing for training and validation.
        Args:
            x: (B, 1, T) waveform with default T=32767
            target_tokens: (B, C, N) tokenized sequence of length N=event_length
            task_tokens: (B, C, task_len) tokenized task

        Returns:
            {
                'logits': (B, N + task_len + 1, vocab_size)
                'loss': (1, )
            }

        NOTE: all the commented shapes are in the case of original MT3 setup.
        """
        x = self.spectrogram(x)  # mel-/spectrogram: (b, 256, 512) or (B, T, F)
        x = self.pre_encoder(x)  # projection to d_model: (B, 256, 512)

        # TODO: task_cond_encoder would not work properly because of 3-d task_tokens
        # if task_tokens is not None and task_tokens.numel() > 0 and self.use_task_cond_encoder is True:
        #     # append task embedding to encoder input
        #     task_embed = self.embed_tokens(task_tokens)  # (B, task_len, 512)
        #     x = torch.cat([task_embed, x], dim=1)  # (B, task_len + 256, 512)
        enc_hs = self.encoder(inputs_embeds=x)["last_hidden_state"]  # (B, T', D)
        enc_hs = self.pre_decoder(enc_hs)  # (B, T', D) or (B, K, T, D)

        # if task_tokens is not None and task_tokens.numel() > 0 and self.use_task_cond_decoder is True:
        #     # append task token to decoder input and output label
        #     labels = torch.cat([task_tokens, target_tokens], dim=2)  # (B, C, task_len + N)
        # else:
        #     labels = target_tokens  # (B, C, N)
        labels = target_tokens  # (B, C, N)
        if labels.shape[1] == 1:  # for single-channel decoders, e.g. t5.
            labels = labels.squeeze(1)  # (B, N)

        dec_input_ids = self.shift_right_fn(labels)  # t5:(B, N), multi-t5:(B, C, N)
        dec_inputs_embeds = self.embed_tokens(dec_input_ids)  # t5:(B, N, D), multi-t5:(B, C, N, D)
        dec_hs, _ = self.decoder(inputs_embeds=dec_inputs_embeds, encoder_hidden_states=enc_hs, return_dict=False)

        if self.model_cfg["tie_word_embeddings"] is True:
            dec_hs = dec_hs * (self.model_cfg["decoder"][self.decoder_type]["d_model"]**-0.5)

        logits = self.lm_head(dec_hs)

        loss = None
        labels = labels.masked_fill(labels == 0, value=-100)  # ignore pad tokens for loss
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"logits": logits, "loss": loss}

    def inference(self,
                  x: torch.FloatTensor,
                  task_tokens: Optional[torch.LongTensor] = None,
                  max_token_length: Optional[int] = None,
                  **kwargs: Any) -> torch.Tensor:
        """ Inference from audio batch by cached autoregressive decoding.
        Args:
            x: (b, 1, t) waveform with t=32767
            task_token: (b, c, task_len) tokenized task. If None, will not append task embeddings (from task_tokens) to input.
            max_length: Maximum length of generated sequence. If None, self.max_total_token_length.
            **kwargs: https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/text_generation#transformers.GenerationMixin.generate
            
        Returns:
            res_tokens: (b, n) resulting tokenized sequence of variable length < max_length
        """
        if self.test_pitch_shift_layer is not None:
            x_ps = self.pitchshift(x, self.test_pitch_shift_semitone)
            x = x_ps

        # From spectrogram to pre-decoder is the same pipeline as in forward()
        x = self.spectrogram(x)  # mel-/spectrogram: (b, 256, 512) or (B, T, F)
        x = self.pre_encoder(x)  # projection to d_model: (B, 256, 512)
        if task_tokens is not None and task_tokens.numel() > 0 and self.use_task_cond_encoder is True:
            # append task embedding to encoder input
            task_embed = self.embed_tokens(task_tokens)  # (B, task_len, 512)
            x = torch.cat([task_embed, x], dim=1)  # (B, task_len + 256, 512)
        enc_hs = self.encoder(inputs_embeds=x)["last_hidden_state"]  # (B, task_len + 256, 512)
        enc_hs = self.pre_decoder(enc_hs)  # (B, task_len + 256, 512)

        # Cached-autoregressive decoding with task token (can be None) as prefix
        if max_token_length is None:
            max_token_length = self.max_total_token_length

        pred_ids = task_cond_dec_generate(decoder=self.decoder,
                                          decoder_type=self.decoder_type,
                                          embed_tokens=self.embed_tokens,
                                          lm_head=self.lm_head,
                                          encoder_hidden_states=enc_hs,
                                          shift_right_fn=self.shift_right_fn,
                                          prefix_ids=task_tokens,
                                          max_length=max_token_length)  # (B, task_len + N) or (B, C, task_len + N)
        if pred_ids.dim() == 2:
            pred_ids = pred_ids.unsqueeze(1)  # (B, 1, task_len + N)

        if self.test_pitch_shift_layer is None:
            return pred_ids
        else:
            return pred_ids, x_ps

    def inference_file(
        self,
        bsz: int,
        audio_segments: torch.FloatTensor,  # (n_items, 1, segment_len): from a single file
        note_token_array: Optional[torch.LongTensor] = None,
        task_token_array: Optional[torch.LongTensor] = None,
        # subtask_key: Optional[str] = "default"
    ) -> Tuple[List[np.ndarray], Optional[torch.Tensor]]:
        """ Inference from audio batch by autoregressive decoding:
        Args:
            bsz: batch size
            audio_segments: (n_items, 1, segment_len): segmented audio from a single file
            note_token_array: (n_items, max_token_len): Optional. If token_array is None, will not return loss.
            subtask_key: (str): If None, not using subtask prefix. By default, using "default" defined in task manager.
        """
        # if subtask_key is not None:
        #     _subtask_token = torch.LongTensor(
        #         self.task_manager.get_eval_subtask_prefix_dict()[subtask_key]).to(self.device)

        n_items = audio_segments.shape[0]
        loss = 0.
        pred_token_array_file = []  # each element is (B, C, L) np.ndarray
        x_ps_concat = []

        for i in range(0, n_items, bsz):
            if i + bsz > n_items:  # last batch can be smaller
                x = audio_segments[i:n_items].to(self.device)
                # if subtask_key is not None:
                #     b = n_items - i  # bsz for the last batch
                #     task_tokens = _subtask_token.expand((b, -1))  # (b, task_len)
                if note_token_array is not None:
                    target_tokens = note_token_array[i:n_items].to(self.device)
                if task_token_array is not None and task_token_array.numel() > 0:
                    task_tokens = task_token_array[i:n_items].to(self.device)
                else:
                    task_tokens = None
            else:
                x = audio_segments[i:i + bsz].to(self.device)  # (bsz, 1, segment_len)
                # if subtask_key is not None:
                #     task_tokens = _subtask_token.expand((bsz, -1))  # (bsz, task_len)
                if note_token_array is not None:
                    target_tokens = note_token_array[i:i + bsz].to(self.device)  # (bsz, token_len)
                if task_token_array is not None and task_token_array.numel() > 0:
                    task_tokens = task_token_array[i:i + bsz].to(self.device)
                else:
                    task_tokens = None

            # token prediction (fast-autoregressive decoding)
            # if subtask_key is not None:
            #     preds = self.inference(x, task_tokens).detach().cpu().numpy()
            # else:
            #     preds = self.inference(x).detach().cpu().numpy()

            if self.test_pitch_shift_layer is not None:  # debug only
                preds, x_ps = self.inference(x, task_tokens)
                preds = preds.detach().cpu().numpy()
                x_ps_concat.append(x_ps.detach().cpu())
            else:
                preds = self.inference(x, task_tokens).detach().cpu().numpy()
            if len(preds) != len(x):
                raise ValueError(f'preds: {len(preds)}, x: {len(x)}')
            pred_token_array_file.append(preds)

            # validation loss (by teacher forcing)
            if note_token_array is not None:
                loss_weight = x.shape[0] / n_items
                loss += self(x, target_tokens)['loss'] * loss_weight
                # loss += self(x, target_tokens, task_tokens)['loss'] * loss_weight
            else:
                loss = None

        if self.test_pitch_shift_layer is not None:  # debug only
            if self.hparams.write_output_dir is not None:
                x_ps_concat = torch.cat(x_ps_concat, dim=0)
                return pred_token_array_file, loss, x_ps_concat.flatten().unsqueeze(0)
        else:
            return pred_token_array_file, loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # batch: {
        # 'dataset1': [Tuple[audio_segments(b, 1, t), tokens(b, max_token_len), ...]]
        # 'dataset2': [Tuple[audio_segments(b, 1, t), tokens(b, max_token_len), ...]]
        # 'dataset3': ...
        # }
        audio_segments, note_tokens, pshift_steps = [torch.cat(t, dim=0) for t in zip(*batch.values())]

        if self.pitchshift is not None:
            # Pitch shift
            n_groups = len(batch)
            audio_segments = torch.chunk(audio_segments, n_groups, dim=0)
            pshift_steps = torch.chunk(pshift_steps, n_groups, dim=0)
            for p in pshift_steps:
                assert p.eq(p[0]).all().item()

            audio_segments = torch.cat([self.pitchshift(a, p[0].item()) for a, p in zip(audio_segments, pshift_steps)],
                                       dim=0)

        loss = self(audio_segments, note_tokens)['loss']
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=note_tokens.shape[0],
                 sync_dist=True)
        # print('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Dict:
        # File-wise validation
        if self.task_manager.num_decoding_channels == 1:
            bsz = self.shared_cfg["BSZ"]["validation"]
        else:
            bsz = self.shared_cfg["BSZ"]["validation"] // self.task_manager.num_decoding_channels * 3
        # audio_segments, notes_dict, note_token_array, task_token_array = batch
        audio_segments, notes_dict, note_token_array = batch
        task_token_array = None

        # Loop through the tensor in chunks of bsz (=subbsz actually)
        n_items = audio_segments.shape[0]
        start_secs_file = [32767 * i / 16000 for i in range(n_items)]
        with Timer() as t:
            pred_token_array_file, loss = self.inference_file(bsz, audio_segments, note_token_array, task_token_array)
            """
            notes_dict: # Ground truth notes
                {
                    'mtrack_id': int,
                    'program': List[int],
                    'is_drum': bool,
                    'duration_sec': float,
                    'notes': List[Note],
                }
            """
            # Process a list of channel-wise token arrays for a file
            num_channels = self.task_manager.num_decoding_channels
            pred_notes_in_file = []
            n_err_cnt = Counter()
            for ch in range(num_channels):
                pred_token_array_ch = [arr[:, ch, :] for arr in pred_token_array_file]  # (B, L)
                zipped_note_events_and_tie, list_events, ne_err_cnt = self.task_manager.detokenize_list_batches(
                    pred_token_array_ch, start_secs_file, return_events=True)
                pred_notes_ch, n_err_cnt_ch = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
                pred_notes_in_file.append(pred_notes_ch)
                n_err_cnt += n_err_cnt_ch
            pred_notes = mix_notes(pred_notes_in_file)  # This is the mixed notes from all channels

            if self.hparams.write_output_dir is not None:
                track_info = [notes_dict[k] for k in notes_dict.keys() if k.endswith("_id")][0]
                dataset_info = [k for k in notes_dict.keys() if k.endswith('_id')][0][:-3]
                # write_model_output_as_npy(zipped_note_events_and_tie, self.hparams.write_output_dir,
                #                           track_info)
                write_model_output_as_midi(pred_notes,
                                           self.hparams.write_output_dir,
                                           track_info,
                                           self.midi_output_inverse_vocab,
                                           output_dir_suffix=str(dataset_info) + '_' +
                                           str(self.hparams.eval_subtask_key))
            # generate sample text to display in log table
            # pred_events_text = [str([list_events[0][:200]])]
            # pred_notes_text = [str([pred_notes[:200]])]

            # this is local GPU metric per file, not global metric in DDP
            drum_metric, non_drum_metric, instr_metric = compute_track_metrics(
                pred_notes,
                notes_dict['notes'],
                eval_vocab=self.hparams.eval_vocab[dataloader_idx],
                eval_drum_vocab=self.hparams.eval_drum_vocab,
                onset_tolerance=self.hparams.onset_tolerance,
                add_pitch_class_metric=self.hparams.add_pitch_class_metric)
            self.val_metrics[dataloader_idx].bulk_update(drum_metric)
            self.val_metrics[dataloader_idx].bulk_update(non_drum_metric)
            self.val_metrics[dataloader_idx].bulk_update(instr_metric)
            self.val_metrics_macro.bulk_update(drum_metric)
            self.val_metrics_macro.bulk_update(non_drum_metric)
            self.val_metrics_macro.bulk_update(instr_metric)

        # Log sample table: predicted notes and ground truth notes
        # if batch_idx in (0, 1) and self.global_rank == 0:
        #     actual_notes_text = [str([notes_dict['notes'][:200]])]
        #     actual_tokens = token_array[0, :200].detach().cpu().numpy().tolist()
        #     actual_events_text = [str(self.tokenizer._decode(actual_tokens))]
        #     track_info = [notes_dict[k] for k in notes_dict.keys() if k.endswith("_id")]
        #     self.sample_table.add_data(self.current_epoch, track_info, pred_events_text,
        #                                actual_events_text, pred_notes_text, actual_notes_text)
        #     self.logger.log_table('Samples', self.sample_table.columns, self.sample_table.data)

        decoding_time_sec = t.elapsed_time()
        self.log('val_loss', loss, prog_bar=True, batch_size=n_items, sync_dist=True)
        # self.val_metrics[dataloader_idx].bulk_update_errors({'decoding_time': decoding_time_sec})

    def on_validation_epoch_end(self) -> None:
        for val_metrics in self.val_metrics:
            self.log_dict(val_metrics.bulk_compute(), sync_dist=True)
            val_metrics.bulk_reset()
        self.log_dict(self.val_metrics_macro.bulk_compute(), sync_dist=True)
        self.val_metrics_macro.bulk_reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Dict:
        # File-wise evaluation
        if self.task_manager.num_decoding_channels == 1:
            bsz = self.shared_cfg["BSZ"]["validation"]
        else:
            bsz = self.shared_cfg["BSZ"]["validation"] // self.task_manager.num_decoding_channels * 3
        # audio_segments, notes_dict, note_token_array, task_token_array = batch
        audio_segments, notes_dict, note_token_array = batch
        task_token_array = None

        # Test pitch shift layer: debug only
        if self.test_pitch_shift_layer is not None and self.test_pitch_shift_semitone != 0:
            for n in notes_dict['notes']:
                if n.is_drum == False:
                    n.pitch = n.pitch + self.test_pitch_shift_semitone

        # Loop through the tensor in chunks of bsz (=subbsz actually)
        n_items = audio_segments.shape[0]
        start_secs_file = [32767 * i / 16000 for i in range(n_items)]

        if self.test_pitch_shift_layer is not None and self.hparams.write_output_dir is not None:
            pred_token_array_file, loss, x_ps = self.inference_file(bsz, audio_segments, None, None)
        else:
            pred_token_array_file, loss = self.inference_file(bsz, audio_segments, None, None)
        if len(pred_token_array_file) > 0:

            # Process a list of channel-wise token arrays for a file
            num_channels = self.task_manager.num_decoding_channels
            pred_notes_in_file = []
            n_err_cnt = Counter()
            for ch in range(num_channels):
                pred_token_array_ch = [arr[:, ch, :] for arr in pred_token_array_file]  # (B, L)
                zipped_note_events_and_tie, list_events, ne_err_cnt = self.task_manager.detokenize_list_batches(
                    pred_token_array_ch, start_secs_file, return_events=True)
                pred_notes_ch, n_err_cnt_ch = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie)
                pred_notes_in_file.append(pred_notes_ch)
                n_err_cnt += n_err_cnt_ch
            pred_notes = mix_notes(pred_notes_in_file)  # This is the mixed notes from all channels

            if self.test_pitch_shift_layer is not None and self.hparams.write_output_dir is not None:
                # debug only
                wav_output_dir = os.path.join(self.hparams.write_output_dir, f"model_output_{dataset_info}")
                os.makedirs(wav_output_dir, exist_ok=True)
                wav_output_file = os.path.join(wav_output_dir, f"{track_info}_ps_{self.test_pitch_shift_semitone}.wav")
                torchaudio.save(wav_output_file, x_ps.squeeze(1), 16000, bits_per_sample=16)

            drum_metric, non_drum_metric, instr_metric = compute_track_metrics(
                pred_notes,
                notes_dict['notes'],
                eval_vocab=self.hparams.eval_vocab[dataloader_idx],
                eval_drum_vocab=self.hparams.eval_drum_vocab,
                onset_tolerance=self.hparams.onset_tolerance,
                add_pitch_class_metric=self.hparams.add_pitch_class_metric,
                add_melody_metric=['Singing Voice'] if self.hparams.add_melody_metric_to_singing else None,
                add_frame_metric=True,
                add_micro_metric=True,
                add_multi_f_metric=True)

            if self.hparams.write_output_dir is not None and self.global_rank == 0:
                # write model output to file
                track_info = [notes_dict[k] for k in notes_dict.keys() if k.endswith("_id")][0]
                dataset_info = [k for k in notes_dict.keys() if k.endswith('_id')][0][:-3]
                f_score = f"OnF{non_drum_metric['onset_f']:.2f}_MulF{instr_metric['multi_f']:.2f}"
                write_model_output_as_midi(pred_notes,
                                           self.hparams.write_output_dir,
                                           track_info,
                                           self.midi_output_inverse_vocab,
                                           output_dir_suffix=str(dataset_info) + '_' +
                                           str(self.hparams.eval_subtask_key) + '_' + f_score)
                write_err_cnt_as_json(track_info, self.hparams.write_output_dir,
                                      str(dataset_info) + '_' + str(self.hparams.eval_subtask_key) + '_' + f_score,
                                      n_err_cnt, ne_err_cnt)

            # Test with optimal octave shift
            if self.hparams.test_optimal_octave_shift:
                track_info = [notes_dict[k] for k in notes_dict.keys() if k.endswith("_id")][0]
                dataset_info = [k for k in notes_dict.keys() if k.endswith('_id')][0][:-3]
                score = [instr_metric['onset_f_Bass']]
                ref_notes_plus = []
                ref_notes_minus = []
                for note in notes_dict['notes']:
                    if note.is_drum == True:
                        ref_notes_plus.append(note)
                        ref_notes_minus.append(note)
                    else:
                        ref_notes_plus.append(
                            Note(is_drum=note.is_drum,
                                 program=note.program,
                                 onset=note.onset,
                                 offset=note.offset,
                                 pitch=note.pitch + 12,
                                 velocity=note.velocity))
                        ref_notes_minus.append(
                            Note(is_drum=note.is_drum,
                                 program=note.program,
                                 onset=note.onset,
                                 offset=note.offset,
                                 pitch=note.pitch - 12,
                                 velocity=note.velocity))

                drum_metric_plus, non_drum_metric_plus, instr_metric_plus = compute_track_metrics(
                    pred_notes,
                    ref_notes_plus,
                    eval_vocab=self.hparams.eval_vocab[dataloader_idx],
                    eval_drum_vocab=self.hparams.eval_drum_vocab,
                    onset_tolerance=self.hparams.onset_tolerance,
                    add_pitch_class_metric=self.hparams.add_pitch_class_metric)
                drum_metric_minus, non_drum_metric_minus, instr_metric_minus = compute_track_metrics(
                    ref_notes_minus,
                    notes_dict['notes'],
                    eval_vocab=self.hparams.eval_vocab[dataloader_idx],
                    eval_drum_vocab=self.hparams.eval_drum_vocab,
                    onset_tolerance=self.hparams.onset_tolerance,
                    add_pitch_class_metric=self.hparams.add_pitch_class_metric)

                score.append(instr_metric_plus['onset_f_Bass'])
                score.append(instr_metric_minus['onset_f_Bass'])
                max_index = score.index(max(score))
                if max_index == 0:
                    print(f"ZERO: {track_info}, z/p/m: {score[0]:.2f}/{score[1]:.2f}/{score[2]:.2f}")
                elif max_index == 1:
                    # plus
                    instr_metric['onset_f_Bass'] = instr_metric_plus['onset_f_Bass']
                    print(f"PLUS: {track_info}, z/p/m: {score[0]:.2f}/{score[1]:.2f}/{score[2]:.2f}")
                    write_model_output_as_midi(ref_notes_plus,
                                               self.hparams.write_output_dir,
                                               track_info + '_ref_octave_plus',
                                               self.midi_output_inverse_vocab,
                                               output_dir_suffix=str(dataset_info) + '_' +
                                               str(self.hparams.eval_subtask_key))
                else:
                    # minus
                    instr_metric['onset_f_Bass'] = instr_metric_minus['onset_f_Bass']
                    print(f"MINUS: {track_info}, z/p/m: {score[0]:.2f}/{score[1]:.2f}/{score[2]:.2f}")
                    write_model_output_as_midi(ref_notes_minus,
                                               self.hparams.write_output_dir,
                                               track_info + '_ref_octave_minus',
                                               self.midi_output_,
                                               output_dir_suffix=str(dataset_info) + '_' +
                                               str(self.hparams.eval_subtask_key))

            self.test_metrics[dataloader_idx].bulk_update(drum_metric)
            self.test_metrics[dataloader_idx].bulk_update(non_drum_metric)
            self.test_metrics[dataloader_idx].bulk_update(instr_metric)
            # self.test_metrics_macro.bulk_update(drum_metric)
            # self.test_metrics_macro.bulk_update(non_drum_metric)
            # self.test_metrics_macro.bulk_update(instr_metric)

    def on_test_epoch_end(self) -> None:
        # all_gather is done seeminglesly by torchmetrics
        for test_metrics in self.test_metrics:
            self.log_dict(test_metrics.bulk_compute(), sync_dist=True)
            test_metrics.bulk_reset()
        # self.log_dict(self.test_metrics_macro.bulk_compute(), sync_dist=True)
        # self.test_metrics_macro.bulk_reset()


def test_case_forward_mt3():
    import torch
    from config.config import audio_cfg, model_cfg, shared_cfg
    from model.ymt3 import YourMT3
    model = YourMT3()
    model.eval()
    x = torch.randn(2, 1, 32767)
    labels = torch.randint(0, 596, (2, 1, 1024), requires_grad=False)  # (B, C=1, T)
    task_tokens = torch.LongTensor([])
    output = model.forward(x, labels, task_tokens)
    logits, loss = output['logits'], output['loss']
    assert logits.shape == (2, 1024, 596)  # (B, N, vocab_size)


def test_case_inference_mt3():
    import torch
    from config.config import audio_cfg, model_cfg, shared_cfg
    from model.ymt3 import YourMT3
    model_cfg["num_max_positions"] = 1024 + 3 + 1
    model = YourMT3(model_cfg=model_cfg)
    model.eval()
    x = torch.randn(2, 1, 32767)
    task_tokens = torch.randint(0, 596, (2, 3), requires_grad=False)
    pred_ids = model.inference(x, task_tokens, max_token_length=10)  # (2, 3, 9) (B, C, L-task_len)
    # TODO: need to check the length of pred_ids when task_tokens is not None


def test_case_forward_enc_perceiver_tf_dec_t5():
    import torch
    from model.ymt3 import YourMT3
    from config.config import audio_cfg, model_cfg, shared_cfg
    model_cfg["encoder_type"] = "perceiver-tf"
    audio_cfg["codec"] = "spec"
    audio_cfg["hop_length"] = 300

    model = YourMT3(audio_cfg=audio_cfg, model_cfg=model_cfg)
    model.eval()

    x = torch.randn(2, 1, 32767)
    labels = torch.randint(0, 596, (2, 1, 1024), requires_grad=False)

    # forward
    output = model.forward(x, labels)
    logits, loss = output['logits'], output['loss']  # logits: (2, 1024, 596) (B, N, vocab_size)

    # inference
    pred_ids = model.inference(x, None, max_token_length=3)  # (2, 1, 3) (B, C, L)


def test_case_forward_enc_conformer_dec_t5():
    import torch
    from model.ymt3 import YourMT3
    from config.config import audio_cfg, model_cfg, shared_cfg
    model_cfg["encoder_type"] = "conformer"
    audio_cfg["codec"] = "melspec"
    audio_cfg["hop_length"] = 128
    model = YourMT3(audio_cfg=audio_cfg, model_cfg=model_cfg)
    model.eval()

    x = torch.randn(2, 1, 32767)
    labels = torch.randint(0, 596, (2, 1024), requires_grad=False)

    # forward
    output = model.forward(x, labels)
    logits, loss = output['logits'], output['loss']  # logits: (2, 1024, 596) (B, N, vocab_size)

    # inference
    pred_ids = model.inference(x, None, 20)  # (2, 1, 20) (B, C, L)


def test_case_enc_perceiver_tf_dec_multi_t5():
    import torch
    from model.ymt3 import YourMT3
    from config.config import audio_cfg, model_cfg, shared_cfg
    model_cfg["encoder_type"] = "perceiver-tf"
    model_cfg["decoder_type"] = "multi-t5"
    model_cfg["encoder"]["perceiver-tf"]["attention_to_channel"] = True
    model_cfg["encoder"]["perceiver-tf"]["num_latents"] = 26
    audio_cfg["codec"] = "spec"
    audio_cfg["hop_length"] = 300
    model = YourMT3(audio_cfg=audio_cfg, model_cfg=model_cfg)
    model.eval()

    x = torch.randn(2, 1, 32767)
    labels = torch.randint(0, 596, (2, 13, 200), requires_grad=False)  # (B, C, T)

    # x = model.spectrogram(x)
    # x = model.pre_encoder(x)  # (2, 110, 128, 128) (B, T, C, D)
    # enc_hs = model.encoder(inputs_embeds=x)["last_hidden_state"]  # (2, 110, 128, 128) (B, T, C, D)
    # enc_hs = model.pre_decoder(enc_hs)  # (2, 13, 110, 512) (B, C, T, D)

    # dec_input_ids = model.shift_right_fn(labels)  # (2, 13, 200) (B, C, T)
    # dec_inputs_embeds = model.embed_tokens(dec_input_ids)  # (2, 13, 200, 512) (B, C, T, D)
    # dec_hs, _ = model.decoder(
    #     inputs_embeds=dec_inputs_embeds, encoder_hidden_states=enc_hs, return_dict=False)
    # logits = model.lm_head(dec_hs)  # (2, 13, 200, 596) (B, C, T, vocab_size)

    # forward
    x = torch.randn(2, 1, 32767)
    labels = torch.randint(0, 596, (2, 13, 200), requires_grad=False)  # (B, C, T)
    output = model.forward(x, labels)
    logits, loss = output['logits'], output['loss']  # (2, 13, 200, 596) (B, C, T, vocab_size)

    # inference
    model.max_total_token_length = 123  # to save time..
    pred_ids = model.inference(x, None)  # (2, 13, 123) (B, C, L)
