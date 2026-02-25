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
from collections import Counter
import numpy as np
import torch
from utils.data_modules import AMTDataModule
from utils.task_manager import TaskManager
from config.data_presets import data_preset_single_cfg, data_preset_multi_cfg
from utils.augment import intra_stem_augment_processor


def get_ds(data_preset_multi: Dict, task_name: str, train_num_samples_per_epoch: int = 90000):
    tm = TaskManager(task_name=task_name)
    tm.max_note_token_length_per_ch = 1024  # only to check the max length
    dm = AMTDataModule(data_preset_multi=data_preset_multi,
                       task_manager=tm,
                       train_num_samples_per_epoch=train_num_samples_per_epoch)
    dm.setup('fit')
    dl = dm.train_dataloader()
    ds = dl.flattened[0].dataset
    return ds


data_preset_multi = data_preset_multi_cfg["all_cross_v6"]
task_name = "mc13"  # "mt3_full_plus"
ds = get_ds(data_preset_multi, task_name=task_name)
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

length_all = []
for i in range(40000):
    if i % 5000 == 0:
        print(i)
    audio_arr, note_token_arr, task_totken_arr, pshift_steps = ds.__getitem__(i)
    lengths = torch.sum(note_token_arr != 0, dim=2).flatten().cpu().tolist()
    length_all.extend(lengths)

length_all = np.asarray(length_all)

# stats
empty_sequence = np.sum(length_all < 3) / len(length_all) * 100
print("empty_sequences:", f"{empty_sequence:.2f}", "%")

mean_except_empty = np.mean(length_all[length_all > 2])
print("mean_except_empty:", mean_except_empty)

median_except_empty = np.median(length_all[length_all > 2])
print("median_except_empty:", median_except_empty)

ch_less_than_768 = np.sum(length_all < 768) / len(length_all) * 100
print("ch_less_than_768:", f"{ch_less_than_768:.2f}", "%")

ch_larger_than_512 = np.sum(length_all > 512) / len(length_all) * 100
print("ch_larger_than_512:", f"{ch_larger_than_512:.6f}", "%")

ch_larger_than_256 = np.sum(length_all > 256) / len(length_all) * 100
print("ch_larger_than_256:", f"{ch_larger_than_256:.6f}", "%")

ch_larger_than_128 = np.sum(length_all > 128) / len(length_all) * 100
print("ch_larger_than_128:", f"{ch_larger_than_128:.6f}", "%")

ch_larger_than_64 = np.sum(length_all > 64) / len(length_all) * 100
print("ch_larger_than_64:", f"{ch_larger_than_64:.6f}", "%")

song_length_all = length_all.reshape(-1, 13)
song_larger_than_512 = 0
song_larger_than_256 = 0
song_larger_than_128 = 0
song_larger_than_64 = 0
for l in song_length_all:
    if np.sum(l > 512) > 0:
        song_larger_than_512 += 1
    if np.sum(l > 256) > 0:
        song_larger_than_256 += 1
    if np.sum(l > 128) > 0:
        song_larger_than_128 += 1
    if np.sum(l > 64) > 0:
        song_larger_than_64 += 1
num_songs = len(song_length_all)
print("song_larger_than_512:", f"{song_larger_than_512/num_songs*100:.4f}", "%")
print("song_larger_than_256:", f"{song_larger_than_256/num_songs*100:.4f}", "%")
print("song_larger_than_128:", f"{song_larger_than_128/num_songs*100:.4f}", "%")
print("song_larger_than_64:", f"{song_larger_than_64/num_songs*100:.4f}", "%")

instr_dict = {
    0: "Piano",
    1: "Chromatic Percussion",
    2: "Organ",
    3: "Guitar",
    4: "Bass",
    5: "Strings + Ensemble",
    6: "Brass",
    7: "Reed",
    8: "Pipe",
    9: "Synth Lead",
    10: "Synth Pad",
    11: "Singing",
    12: "Drums",
}
cnt_larger_than_512 = Counter()
for i in np.where(length_all > 512)[0] % 13:
    cnt_larger_than_512[i] += 1
print("larger_than_512:")
for k, v in cnt_larger_than_512.items():
    print(f"    - {instr_dict[k]}: {v}")

cnt_larger_than_256 = Counter()
for i in np.where(length_all > 256)[0] % 13:
    cnt_larger_than_256[i] += 1
print("larger_than_256:")
for k, v in cnt_larger_than_256.items():
    print(f"    - {instr_dict[k]}: {v}")

cnt_larger_than_128 = Counter()
for i in np.where(length_all > 128)[0] % 13:
    cnt_larger_than_128[i] += 1
print("larger_than_128:")
for k, v in cnt_larger_than_128.items():
    print(f"    - {instr_dict[k]}: {v}")
"""
empty_sequences: 91.06 %
mean_except_empty: 36.68976799156269
median_except_empty: 31.0
ch_less_than_768: 100.00 %
ch_larger_than_512: 0.000158 %
ch_larger_than_256: 0.015132 %
ch_larger_than_128: 0.192061 %
ch_larger_than_64: 0.661260 %
song_larger_than_512: 0.0021 %
song_larger_than_256: 0.1926 %
song_larger_than_128: 2.2280 %
song_larger_than_64: 6.1033 %

larger_than_512:
    - Guitar: 7
    - Strings + Ensemble: 3
larger_than_256:
    - Piano: 177
    - Guitar: 680
    - Strings + Ensemble: 79
    - Organ: 2
    - Chromatic Percussion: 11
    - Bass: 1
    - Synth Lead: 2
    - Brass: 1
    - Reed: 5
larger_than_128:
    - Guitar: 4711
    - Strings + Ensemble: 1280
    - Piano: 5548
    - Bass: 211
    - Synth Pad: 22
    - Pipe: 18
    - Chromatic Percussion: 55
    - Synth Lead: 22
    - Organ: 75
    - Reed: 161
    - Brass: 45
    - Drums: 11
"""
