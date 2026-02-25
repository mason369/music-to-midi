# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
from typing import Dict, Tuple
from copy import deepcopy
import soundfile as sf
import torch
from utils.data_modules import AMTDataModule
from config.data_presets import data_preset_single_cfg, data_preset_multi_cfg
from utils.augment import intra_stem_augment_processor


def get_ds(data_preset_multi: Dict, train_num_samples_per_epoch: int = 90000):
    dm = AMTDataModule(data_preset_multi=data_preset_multi, train_num_samples_per_epoch=train_num_samples_per_epoch)
    dm.setup('fit')
    dl = dm.train_dataloader()
    ds = dl.flattened[0].dataset
    return ds


def debug_func(num_segments: int = 10):
    sampled_data, sampled_ids = ds._get_rand_segments_from_cache(num_segments)
    ux_sampled_data, _ = ds._get_rand_segments_from_cache(ux_count_sum, False, sampled_ids)
    s = deepcopy(sampled_data)
    intra_stem_augment_processor(sampled_data, submix_audio=False)


def gen_audio(index: int = 0):
    # audio_arr: (b, 1, nframe), note_token_arr: (b, l), task_token_arr: (b, task_l)
    audio_arr, note_token_arr, task_token_arr = ds.__getitem__(index)

    # merge all the segments into one audio file
    audio = audio_arr.permute(0, 2, 1).reshape(-1).squeeze().numpy()

    # save the audio file
    sf.write('xaug_demo_audio.wav', audio, 16000, subtype='PCM_16')


data_preset_multi = data_preset_multi_cfg["all_cross_rebal5"]
ds = get_ds(data_preset_multi)
ds.random_amp_range = [0.8, 1.1]
ds.stem_xaug_policy = {
    "max_k": 5,
    "tau": 0.3,
    "alpha": 1.0,
    "max_subunit_stems": 12,
    "no_instr_overlap": True,
    "no_drum_overlap": True,
    "uhat_intra_stem_augment": True,
}
gen_audio(3)

# for k in ds.cache.keys():
#     arr = ds.cache[k]['audio_array']
#     arr = np.sum(arr, axis=1).reshape(-1)
#     # sf.write(f'xxx/{k}.wav', arr, 16000, subtype='PCM_16')
#     if np.min(arr) > -0.5:
#         print(k)

# arr = ds.cache[52]['audio_array']
# for i in range(arr.shape[1]):
#     a = arr[:, i, :].reshape(-1)
#     sf.write(f'xxx52/52_{i}.wav', a, 16000, subtype='PCM_16')
