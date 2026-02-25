# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
import numpy as np
import torch
import json
import soundfile as sf
from utils.datasets_train import get_cache_data_loader


def get_filelist(track_id: int) -> dict:
    filelist = '../../data/yourmt3_indexes/slakh_train_file_list.json'
    with open(filelist, 'r') as f:
        fl = json.load(f)
    new_filelist = dict()
    for key, value in fl.items():
        if int(key) == track_id:
            new_filelist[0] = value
    return new_filelist


def get_ds(track_id: int, random_amp_range: list = [1., 1.], stem_aug_prob: float = 0.8):
    filelist = get_filelist(track_id)
    dl = get_cache_data_loader(filelist,
                               'train',
                               1,
                               1,
                               random_amp_range=random_amp_range,
                               stem_aug_prob=stem_aug_prob,
                               shuffle=False)
    ds = dl.dataset
    return ds


def gen_audio(track_id: int, n_segments: int = 30, random_amp_range: list = [1., 1.], stem_aug_prob: float = 0.8):
    ds = get_ds(track_id, random_amp_range, stem_aug_prob)
    audio = []
    for i in range(n_segments):
        audio.append(ds.__getitem__(0)[0])
        # audio.append(ds.__getitem__(i)[0])

    audio = torch.concat(audio, dim=2).numpy()[0, 0, :]
    sf.write('audio.wav', audio, 16000, subtype='PCM_16')


gen_audio(1, 20)
