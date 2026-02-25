# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
import json
import os
import warnings
import random
from collections import OrderedDict
from itertools import cycle
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
import scipy.stats as stats
from torch.utils.data import DataLoader, Dataset, Sampler

from config.config import shared_cfg
from config.config import audio_cfg as default_audio_cfg
from utils.audio import get_segments_from_numpy_array, load_audio_file
from utils.augment import (audio_random_submix_processor, combined_survival_and_stop, cross_stem_augment_processor,
                           intra_stem_augment_processor)
from utils.note2event import slice_multiple_note_events_and_ties_to_bundle, slice_note_events_and_ties
from utils.note2event import pitch_shift_note_events
from utils.note_event_dataclasses import NoteEventListsBundle
from utils.task_manager import TaskManager
from utils.utils import Timer

UNANNOTATED_PROGRAM = 129


class FixedSizeOrderedDict(OrderedDict):
    """
    Dequeue-dict: If the dictionary reaches its maximum size, it will 
    automatically remove the oldest key-value pair. 
    """

    def __init__(self, max_size: int):
        super().__init__()
        self.max_size: int = max_size
        self._id_set: set = set()
        self._id_counter: int = 0

    def __setitem__(self, key: Any, value: Any) -> None:
        if key not in self:
            if len(self) >= self.max_size:
                oldest_key, _ = self.popitem(last=False)
                self._id_set.remove(oldest_key)
        super().__setitem__(key, value)
        self._id_set.add(key)

    def generate_unique_id(self) -> int:
        while self._id_counter in self._id_set:
            self._id_counter = (self._id_counter + 1) % (self.max_size * 100)
            # max_size * 100 is arbitrary, but to ensure that there are enough
            # unique ids available when the dictionary is full.
        unique_id: int = self._id_counter
        return unique_id


class CachedAudioDataset(Dataset):
    """
    🎧 CachedAudioDataset:

    This dataset subsamples from a temporal cache of audio data to improve efficiency
    during training.

    - The dataset uses a fixed size cache and subsamples from the N most recent batches
      stored in the cache.
    - This design can help alleviate the disk I/O bottleneck that can occur during
      random access of audio multi-track data for augmentation.

    Tips:
    - The '__getitem__()' method returns a sub-batch of samples from the cache with a
      size specified by the 'subbatch_size' parameter.
    - Use 'collate_fn' in your dataloader to get the final batch size
      (num_workers * subbatch_size).
    - Larger 'subbatch_size' will result in more efficient parallelization.
 
    👀 See '_update_cache()' for customized data processing.

    """

    def __init__(
        self,
        file_list: Union[str, os.PathLike, Dict],
        task_manager: TaskManager = TaskManager(),
        num_samples_per_epoch: Optional[int] = None,
        fs: int = 16000,
        seg_len_frame: int = 32767,
        sub_batch_size: int = 16,
        num_files_cache: Optional[int] = None,
        sample_index_for_init_cache: Optional[List[int]] = None,
        random_amp_range: Optional[List[float]] = [0.6, 1.2],
        pitch_shift_range: Optional[List[int]] = None,
        stem_iaug_prob: Optional[float] = 0.7,
        stem_xaug_policy: Optional[Dict] = {
            "max_k": 3,  # max number of external sources used for cross-stem augmentations
            "tau": 0.3,  # exponential decay rate for cross-stem augmentation
            "alpha": 1.0,  # shape parameter for Weibull distribution. Set to 1.0 for exponential distribution.
            "max_subunit_stems": 12,  # the number of subunit stems to be reduced to this number of stems
            "p_include_singing":
                0.8,  # probability of including singing for cross augmented examples. if None, use base probaility.
            "no_instr_overlap": True,
            "no_drum_overlap": True,
            "uhat_intra_stem_augment": True,
        }
    ) -> None:
        """
        Args:
            file_list: Path to the file list, or a dictionary of file list. e.g. "../../data/yourmt3_indexes/slakh_train_file_list.json",
            task_manager: Task manager.
            fs: Sampling frequency.
            seg_len_frame: Length of the audio segment in frames.
            sub_batch_size: Number of segments per sub-batch.
            num_files_cache: Number of files to cache.
                -  If None, max(4, cross_stem_aug_max_k) * sub_batch_size files will be cached.
                -  When manually setting, it is recommended to use a number larger than the sub_batch_size.
                -  When using `cross_stem_aug`, it is recommended to set num_files_cache to a
                   multiple of sub_batch_size for diversity of cross-batch samples.
            random_amp_range: Random amplitude range. Default: [0.6, 1.2].
            pitch_shift_range: Pitch shift range. Default: [-2, 2]. If None or [0, 0], pitch shift is disabled.
            stem_iaug_prob: Probability of intra-stem augmentation. Bernoulli(p). Default: 0.7.
              If None or 1, intra-stem augmentation is disabled. If 0, only one stem is randomly
              selected.
            stem_xaug_policy: Policy for cross-stem augmentation. If None, cross-stem augmentation
              is disabled. Default: {
                "max_k": 5, (Max number of external sources used for cross-stem augmentations. If 0, no cross-stem augmentation) 
                "no_instr_overlap": True, 
                "no_drum_overlap": True,
                "uhat_intra_stem_augment": False,
              }
        """

        # load the file list
        if isinstance(file_list, dict):
            self.file_list = file_list
        elif isinstance(file_list, str) or isinstance(file_list, os.PathLike):
            with open(file_list, 'r') as f:
                fl = json.load(f)
            self.file_list = {int(key): value for key, value in fl.items()}
        else:
            raise ValueError(f'📕 file_list must be a dictionary or a path to \
                              a json file.')

        self.num_samples_per_epoch = num_samples_per_epoch
        self.fs = fs
        self.seg_len_frame = seg_len_frame
        self.seg_len_sec = seg_len_frame / fs

        # Task manager
        self.task_manager = task_manager  # task_manager includes the tokenizer
        self.num_decoding_channels = task_manager.num_decoding_channels  # By default 1, but can be > 1 for multi-channel decoding

        # Augmentation
        self.random_amp_range = random_amp_range
        self.stem_iaug_prob = stem_iaug_prob
        self.stem_xaug_policy = stem_xaug_policy
        if stem_xaug_policy is not None:
            # precompute the probability distribution of stopping at each k
            self.precomputed_prob_stop_at_k = combined_survival_and_stop(max_k=stem_xaug_policy["max_k"],
                                                                         tau=stem_xaug_policy["tau"],
                                                                         alpha=stem_xaug_policy["alpha"])[1]
        if pitch_shift_range is not None or pitch_shift_range != [0, 0]:
            self.pitch_shift_range = pitch_shift_range
        else:
            self.pitch_shift_range = None

        # determine the number of samples per file & the number of files to cache
        self.sub_batch_size = sub_batch_size
        if num_files_cache is None:
            if stem_xaug_policy is None:
                self.num_files_cache = 4 * sub_batch_size
            else:
                self.num_files_cache = max(4, stem_xaug_policy["max_k"] + 1) * sub_batch_size
        elif isinstance(num_files_cache, int):
            if sub_batch_size > num_files_cache:
                raise ValueError(
                    f'📙 num_files_cache {num_files_cache} must be equal or larger than sub_batch_size {sub_batch_size}.'
                )  # currently, we do not support sub_batch_size > num_files_cache
            if stem_xaug_policy is not None and (sub_batch_size * 2 > num_files_cache):
                warnings.warn(
                    f'📙 When cross_stem_aug_k is not None, sub_batch_size {sub_batch_size} * 2 > num_files_cache {num_files_cache} will decrease diversity in training examples.'
                )
            self.num_files_cache = num_files_cache
        else:
            raise ValueError(f'📙 num_files_cache must be an integer or None. Got {num_files_cache}.')

        self.seg_read_size = 1  # np.ceil(sub_batch_size / num_files_cache).astype(int)
        self.num_cached_seg_per_file = sub_batch_size
        print(f'📘 caching {self.num_cached_seg_per_file} segments per file.')

        # initialize cache
        self._init_cache(index_for_init_cache=sample_index_for_init_cache)

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        # update cache with new stem and token segments
        self._update_cache(index)

        # get sub-batch note_events and audio for segments from the cache
        sampled_data, sampled_ids = self._get_rand_segments_from_cache(
            num_segments=self.sub_batch_size)  # sampled_data is deepcopy of sampled cached instances

        # Stem augmentation and audio submix: processing sampled_data in-place
        self._augment_stems_and_audio_submix_in_place(sampled_data, sampled_ids)
        # assert "processed_audio_array" in sampled_data.keys()

        # Post-mix augmentation: pitch shift (per-batch)
        self._post_mix_augment(sampled_data)
        # assert "pitch_shift_steps" in sampled_data.keys()

        # Prepare sub-batch
        processed_audio_array = sampled_data['processed_audio_array']
        token_array = self.task_manager.tokenize_task_and_note_events_batch(
            programs_segments=sampled_data['programs_segments'],
            has_unannotated_segments=sampled_data['has_unannotated_segments'],
            note_event_segments=sampled_data['note_event_segments'],
            subunit_programs_segments=None,  # using subunit is TODO
            subunit_note_event_segments=None  # using subunit is TODO
        )

        # note_token_array = self.task_manager.tokenize_note_events_batch(sampled_data['note_event_segments'])
        # task_token_array = self.task_manager.tokenize_task_events_batch(sampled_data['programs_segments'],
        #                                                                 sampled_data['has_unannotated_segments'])
        pitch_shift_steps = sampled_data['pitch_shift_steps']

        # Shape:
        #   processed_audio_array: (sub_b, 1, nframe)
        #   note_token_array: (sub_b, decoding_ch, max_note_token_len)
        #   task_token_array: (sub_b, decoding_ch, max_task_token_len)
        #   pitch_shift_steps: (sub_b,)
        return torch.FloatTensor(processed_audio_array), torch.LongTensor(token_array), torch.LongTensor(
            pitch_shift_steps)

        # Shape:
        #   processed_audio_array: (sub_b, 1, nframe)
        #   note_token_array: (sub_b, decoding_ch, max_note_token_len)
        #   task_token_array: (sub_b, decoding_ch, max_task_token_len)
        #   pitch_shift_steps: (sub_b,)
        # return torch.FloatTensor(processed_audio_array), torch.LongTensor(note_token_array), torch.LongTensor(
        #     task_token_array), torch.LongTensor(pitch_shift_steps)

    def _post_mix_augment(self, sampled_data: Dict[str, Any]) -> None:
        """Post-mix augmentation"""

        if self.pitch_shift_range is None:
            sampled_data['pitch_shift_steps'] = [0] * self.sub_batch_size
            return
        else:
            """random pitch shift on note events only. audio will be transformer in the model's layer"""
            # random pitch shift steps
            sampled_data['pitch_shift_steps'] = np.random.randint(
                self.pitch_shift_range[0], self.pitch_shift_range[1] + 1) * np.ones(self.sub_batch_size)
            # n_choices = self.pitch_shift_range[1] - self.pitch_shift_range[
            #     0] + 1
            # zero_index = np.argmax(
            #     np.arange(self.pitch_shift_range[0],
            #               self.pitch_shift_range[1] + 1) == 0)
            # p = np.ones(n_choices)
            # p[zero_index] = n_choices * 2
            # sampled_data['pitch_shift_steps'] = np.full(
            #     self.sub_batch_size,
            #     np.random.choice(n_choices, 1, p=p / p.sum())[0] +
            #     self.pitch_shift_range[0],
            #     dtype=np.int32
            # )  # p = [0.07142857 0.07142857 0.71428571 0.07142857 0.07142857]

            # apply pitch shift to note events and tie note events (in-place)
            note_event_segments = sampled_data['note_event_segments']
            for i, (note_events, tie_note_events, start_time) in enumerate(list(zip(*note_event_segments.values()))):
                note_events = pitch_shift_note_events(note_events,
                                                      sampled_data['pitch_shift_steps'][i],
                                                      use_deepcopy=True)
                tie_note_events = pitch_shift_note_events(tie_note_events,
                                                          sampled_data['pitch_shift_steps'][i],
                                                          use_deepcopy=True)

    def _augment_stems_and_audio_submix_in_place(self, sampled_data: Dict[str, Any], sampled_ids: np.ndarray) -> None:
        """Augment stems and submix audio"""

        if self.stem_iaug_prob is None or self.stem_iaug_prob == 1.:
            # no augmentation at all
            audio_random_submix_processor(sampled_data=sampled_data, random_amp_range=self.random_amp_range)
            return
        elif self.stem_xaug_policy is None or self.stem_xaug_policy["max_k"] == 0:
            # intra-stem augmentation only
            intra_stem_augment_processor(sampled_data=sampled_data,
                                         random_amp_range=self.random_amp_range,
                                         prob=self.stem_iaug_prob,
                                         submix_audio=True)
            return
        elif self.stem_xaug_policy is not None and self.stem_xaug_policy["max_k"] > 0:
            intra_stem_augment_processor(
                sampled_data=sampled_data,
                random_amp_range=self.random_amp_range,
                prob=self.stem_iaug_prob,
                submix_audio=False)  # submix_audio=False to postpone audio mixing until cross-stem augmentation
            cross_stem_augment_processor(
                sampled_data=sampled_data,  # X_hat
                sampled_ids=sampled_ids,  # indices of X, to exclude X from U
                get_rand_segments_from_cache_fn=self._get_rand_segments_from_cache,
                random_amp_range=self.random_amp_range,
                stem_iaug_prob=self.stem_iaug_prob,
                stem_xaug_policy=self.stem_xaug_policy,
                max_l=self.task_manager.max_note_token_length,
                precomputed_prob_stop_at_k=self.precomputed_prob_stop_at_k,
                mix_audio=True,
                create_subunit_note_events=False)
            # assert "subunit_programs_segments" in sampled_data.keys()
            # assert "subunit_audio_array" in sampled_data.keys()
            # assert "subunit_note_event_segments" in sampled_data.keys()
            # assert "programs_segments" in sampled_data.keys()
            # assert "note_event_segments" in sampled_data.keys()
            # assert "has_unannotated_segments" in sampled_data.keys()
            # assert "processed_audio_array" in sampled_data.keys()
        else:
            raise ValueError(f"Invalid stem_xaug_policy: {self.stem_xaug_policy}")

    def __len__(self):
        return len(self.file_list)

    def _get_rand_segments_from_cache(
            self,
            num_segments: Union[int, Literal["max"]],
            use_ordered_read_pos: bool = True,
            sample_excluding_ids: Optional[np.ndarray] = None) -> Tuple[NoteEventListsBundle, np.ndarray]:
        """Get sampled segments from the cache, accessed by file ids and read positions.
        Args:
            use_ordered_read_pos: Whether to use the oredered read position generator. Default: True.
              If False, the read position is randomly selected. This is used for cross-stem augmentation
              source samples. 
            sample_excluding_ids: IDs to exclude files from sampling.
            num_segments: Number of segments to sample. If None, sub_batch_size * cross_stem_aug_max_k.
        Returns:
            sampled_data: Dict

        Function execution time: 60 µs for sub_bsz=36 with single worker

        NOTE: This function returns mutable instances of the cached data. If you want to modify the
            data, make sure to deepcopy the returned data, as in the augment.py/drop_random_stems_from_bundle()
        """
        # construct output dict
        sampled_data = {
            'audio_segments': [],  # list of (1, n_stems, n_frame) with len = sub_batch_size
            'note_event_segments': {
                'note_events': [],  # list of List[NoteEvent]
                'tie_note_events': [],  # list of List[NoteEvent]
                'start_times': [],  # [float, float, ...]
            },  # NoteEventBundle dataclass
            'programs_segments': [],  # list of List[int]
            'is_drum_segments': [],  # list of List[bool]
            'has_stems_segments': [],  # List[bool]
            'has_unannotated_segments': [],  # List[bool]
        }

        # random choice of files from cache
        if num_segments == "max":
            n = self.sub_batch_size * self.stem_xaug_policy["max_k"]
        elif isinstance(num_segments, int):
            n = num_segments
        else:
            raise ValueError(f"num_segments must be int or 'max', but got {num_segments}")
        cache_values = np.array(list(self.cache.values()))
        if sample_excluding_ids is None:
            sampled_ids = np.random.choice(
                self.num_files_cache, n, replace=False
            )  # The ids are not exactly the keys() of cache, since we reindexed them in the range(0,N) by np.array(dict.values())
        else:
            sampled_ids = np.random.permutation(list(set(np.arange(self.num_files_cache)) -
                                                     set(sample_excluding_ids)))[:n]
        selected_files = cache_values[sampled_ids]

        if use_ordered_read_pos is True:
            start = self._get_read_pos()
            end = start + self.seg_read_size
        for d in selected_files:
            if use_ordered_read_pos is False:
                start = np.random.randint(0, self.num_cached_seg_per_file - self.seg_read_size + 1)
                end = start + self.seg_read_size
            sampled_data['audio_segments'].append(d['audio_array'][start:end])
            sampled_data['note_event_segments']['note_events'].extend(
                d['note_event_segments']['note_events'][start:end])
            sampled_data['note_event_segments']['tie_note_events'].extend(
                d['note_event_segments']['tie_note_events'][start:end])
            sampled_data['note_event_segments']['start_times'].extend(
                d['note_event_segments']['start_times'][start:end])
            sampled_data['programs_segments'].append(d['programs'])
            sampled_data['is_drum_segments'].append(d['is_drum'])
            sampled_data['has_stems_segments'].append(d['has_stems'])
            sampled_data['has_unannotated_segments'].append(d['has_unannotated'])
        return sampled_data, sampled_ids  # Note that the data returned is mutable instance.

    def _update_cache(self, index) -> None:
        data = {
            'programs': None,
            'is_drum': None,
            'has_stems': None,
            'has_unannotated': None,
            'audio_array': None,  # (n_segs, n_stems, n_frames): non-stem dataset has n_stems=1
            'note_event_segments': None,  # NoteEventBundle dataclass
        }

        # Load Audio stems -> slice -> (audio_segments, start_times)
        if 'stem_file' in self.file_list[index].keys() and \
            self.file_list[index]['stem_file'] != None:
            audio_data = np.load(self.file_list[index]['stem_file'],
                                 allow_pickle=True).tolist()  # dict with 'audio_array' having shape (n_stems, n_frames)
            data['has_stems'] = True
        elif 'mix_audio_file' in self.file_list[index].keys():
            wav_data = load_audio_file(self.file_list[index]['mix_audio_file'], fs=self.fs, dtype=np.float32)
            audio_data = {
                'audio_array': wav_data[np.newaxis, :],  # (1, n_frames)
                'n_frames': len(wav_data),
                'program': np.array(self.file_list[index]['program'], dtype=np.int32),
                'is_drum': np.array(self.file_list[index]['is_drum'], dtype=np.int32),
            }
            data['has_stems'] = False
        else:
            raise ValueError(f'📕 No stem_file or mix_audio_file found in the file list.')

        if UNANNOTATED_PROGRAM in audio_data['program']:
            data['has_unannotated'] = True

        # Pad audio data shorter than the segment length
        if audio_data['audio_array'].shape[1] < self.seg_len_frame + 2000:
            audio_data['audio_array'] = np.pad(audio_data['audio_array'],
                                               ((0, 0),
                                                (0, self.seg_len_frame + 2000 - audio_data['audio_array'].shape[1])),
                                               mode='constant')
            audio_data['n_frames'] = audio_data['audio_array'].shape[1]

        data['programs'] = audio_data['program']
        data['is_drum'] = audio_data['is_drum']

        # Randomly select start frame indices and filtering out empty note_event segments
        note_event_data = np.load(self.file_list[index]['note_events_file'], allow_pickle=True).tolist()
        note_event_segments = NoteEventListsBundle({'note_events': [], 'tie_note_events': [], 'start_times': []})
        start_frame_indices = []
        attempt = 0
        while len(start_frame_indices) < self.num_cached_seg_per_file and attempt < 5:
            sampled_indices = random.sample(range(audio_data['n_frames'] - self.seg_len_frame),
                                            self.num_cached_seg_per_file)
            for idx in sampled_indices:
                _start_time = idx / self.fs
                _end_time = _start_time + self.seg_len_sec
                sliced_note_events, sliced_tie_note_events, _ = slice_note_events_and_ties(
                    note_event_data['note_events'], _start_time, _end_time, False)
                if len(sliced_note_events) + len(sliced_tie_note_events) > 0 or attempt == 4:
                    # non-empty segment or last attempt
                    start_frame_indices.append(idx)
                    note_event_segments['note_events'].append(sliced_note_events)
                    note_event_segments['tie_note_events'].append(sliced_tie_note_events)
                    note_event_segments['start_times'].append(_start_time)
                if len(start_frame_indices) == self.num_cached_seg_per_file:
                    break
            attempt += 1
        assert len(start_frame_indices) == self.num_cached_seg_per_file

        # start_frame_indices = np.random.choice(audio_data['n_frames'] - self.seg_len_frame,
        #                                        size=self.num_cached_seg_per_file,
        #                                        replace=False)
        # start_times = start_frame_indices / self.fs

        # # Load Note events -> slice -> note_event_segments, tie_note_event_segments
        # note_event_data = np.load(self.file_list[index]['note_events_file'], allow_pickle=True).tolist()

        # # Extract note event segments for the audio segments, returning a dictionary
        # # with keys: 'note_events', 'tie_note_events', and 'start_times'.
        # note_event_segments = slice_multiple_note_events_and_ties_to_bundle(
        #     note_event_data['note_events'],
        #     start_times,
        #     self.seg_len_sec,
        # )  # note_event_segments: see NoteEventBundle dataclass...

        audio_segments = get_segments_from_numpy_array(audio_data['audio_array'],
                                                       self.seg_len_frame,
                                                       start_frame_indices=start_frame_indices,
                                                       dtype=np.float32)  # audio_segments: (n_segs, n_stems, n_frames)

        # Add audio and note events of the sliced segments to data
        data['audio_array'] = audio_segments  # (n_segs, n_stems, n_frames)
        data['note_event_segments'] = note_event_segments  # NoteEventBundle dataclass

        # Update the cache
        unique_id = self.cache.generate_unique_id()
        self.cache[unique_id] = data  # push

    def _init_cache(self, index_for_init_cache: Optional[List[int]] = None):
        with Timer() as t:
            self.cache = FixedSizeOrderedDict(max_size=self.num_files_cache)
            print(f'💿 Initializing cache with max_size={self.cache.max_size}')
            if index_for_init_cache is not None:
                assert len(index_for_init_cache) >= self.num_files_cache
                for i in index_for_init_cache[-self.num_files_cache:]:
                    self._update_cache(i)
            else:
                rand_ids = np.random.choice(np.arange(len(self)), size=self.num_files_cache, replace=False)
                for i in rand_ids:
                    self._update_cache(i)

            # Initialize an infinite cache read position generator
            self._cache_read_pos_generator = cycle(np.arange(0, self.num_cached_seg_per_file, self.seg_read_size))
        t.print_elapsed_time()

    def _get_read_pos(self):
        return next(self._cache_read_pos_generator)


def collate_fn(batch: Tuple[torch.FloatTensor, torch.LongTensor],
               local_batch_size: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """
    This function is used to get the final batch size 
    batch: (np.ndarray of shape (B, b, 1, T), np.ndarray of shape (B, b, T))
           where b is the sub-batch size and B is the batch size.
    """
    audio_segments = torch.vstack([b[0] for b in batch])
    note_tokens = torch.vstack([b[1] for b in batch])
    return (audio_segments, note_tokens)


# def collate_fn(batch: Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor],
#                local_batch_size: int) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
#     """
#     This function is used to get the final batch size
#     batch: (np.ndarray of shape (B, b, 1, T), np.ndarray of shape (B, b, T))
#            where b is the sub-batch size and B is the batch size.
#     """
#     audio_segments = torch.vstack([b[0] for b in batch])
#     note_tokens = torch.vstack([b[1] for b in batch])
#     task_tokens = torch.vstack([b[2] for b in batch])
#     return (audio_segments, note_tokens, task_tokens)


def get_cache_data_loader(
        dataset_name: Optional[str] = None,
        split: Optional[str] = None,
        file_list: Optional[Dict] = None,
        sub_batch_size: int = 32,
        task_manager: TaskManager = TaskManager(),
        stem_iaug_prob: Optional[float] = 0.7,
        stem_xaug_policy: Optional[Dict] = {
            "max_k": 3,
            "tau": 0.3,
            "alpha": 1.0,
            "max_subunit_stems": 12,
            "p_include_singing": 0.8,
            "no_instr_overlap": True,
            "no_drum_overlap": True,
            "uhat_intra_stem_augment": True,
        },
        random_amp_range: Optional[List[float]] = [0.6, 1.2],
        pitch_shift_range: Optional[List[int]] = None,
        shuffle: Optional[bool] = True,
        sampler: Optional[Sampler] = None,
        audio_cfg: Optional[Dict] = None,
        dataloader_config: Dict = {"num_workers": 0}) -> DataLoader:
    """
    This function returns a DataLoader object that can be used to iterate over the dataset.
    Args:
        dataset_name: str, name of the dataset.
        split: str, name of the split.
        - dataset_name and split are used to load the file list.
        - if file_list is not None, and dataset_name and split should be None, it will be used to load the dataset.
        file_list: dict, file list of the dataset.
        sub_batch_size: int, number of segments per sub-batch.
        task_manager: TaskManager, See `utils/task_manager.py`.
        stem_iaug_prob: float, probability of intra-stem augmentation. Bernoulli(p). Default: 0.7.
            If None or 1, intra-stem augmentation is disabled. If 0, only one stem is randomly selected.
        stem_xaug_policy: dict, policy for cross-stem augmentation. If None, cross-stem augmentation
            is disabled. 
        random_amp_range: list, random amplitude range. Default: [0.6, 1.2].
        pitch_shift_range: list, pitch shift range. Default: [-2, 2]. None or [0, 0] for no pitch shift.
        shuffle (bool): whether to shuffle the dataset. Default: True. However, shuffle is ignored when sampler is specified.
        sampler: Sampler, defines the strategy to draw samples from the dataset. If specified, shuffle must be False.
        audio_cfg: dict, audio configuration.
        dataloader_config: dict, other arguments for PyTorch native DataLoader class.

    Returns:
        DataLoader object.
    """
    if dataset_name is None and split is None and file_list is None:
        raise ValueError("Error: all arguments cannot be None.")
    elif (dataset_name is not None and split is not None and file_list is None) and isinstance(
            split, str) and isinstance(dataset_name, str):
        data_home = shared_cfg["PATH"]["data_home"]
        file_list = f"{data_home}/yourmt3_indexes/{dataset_name}_{split}_file_list.json"
        assert os.path.exists(file_list)
    elif (dataset_name is None and split is None and file_list is not None) and isinstance(file_list, dict):
        pass
    else:
        raise ValueError("Error: invalid combination of arguments.")

    # If sampler is specified, initialize cache using sampler, otherwise random initialization.
    if sampler is not None:
        sample_index_for_init_cache = list(sampler)
    else:
        sample_index_for_init_cache = None

    if audio_cfg is None:
        audio_cfg = default_audio_cfg

    ds = CachedAudioDataset(
        file_list,
        task_manager=task_manager,
        seg_len_frame=int(audio_cfg['input_frames']),
        sub_batch_size=sub_batch_size,
        num_files_cache=None,  # auto
        random_amp_range=random_amp_range,
        pitch_shift_range=pitch_shift_range,
        stem_iaug_prob=stem_iaug_prob,
        stem_xaug_policy=stem_xaug_policy,
        sample_index_for_init_cache=sample_index_for_init_cache,
    )
    batch_size = None
    _collate_fn = None

    return DataLoader(ds,
                      batch_size=batch_size,
                      collate_fn=_collate_fn,
                      sampler=sampler,
                      shuffle=None if sampler is not None else shuffle,
                      **dataloader_config)


# def speed_benchmark_cache_audio_dataset():
#     # ds = CachedAudioDataset(sub_batch_size=32, num_files_cache=8)
#     #
#     # Audio-only w/ single worker:
#     #   %timeit ds.__getitem__(0) # 61.9ms b16 c16; 76.1ms b16 c4;  77.2ms b16 c1
#     #   %timeit ds.__getitem__(0) # 133ms b64 c64; 118ms b64 c32; 114ms b64 c16
#     #   %timeit ds.__getitem__(0) # 371ms b128 c128; 205ms b128 c64; 200ms b128 c32; 165ms b128 c16
#     #
#     ds = CachedAudioDataset(sub_batch_size=128, num_files_cache=16, tokenizer=NoteEventTokenizer())
#     # Audio + Tokenization w/ single worker:
#     # %timeit ds.__getitem__(0) # 91.2ms b16 c16; 90.8ms b16 c4; 98.6ms b16 c1
#     # %timeit ds.__getitem__(0) # 172ms b64 c64; 158ms b64 c32; 158ms b64 c16
#     # %timeit ds.__getitem__(0) # 422ms b128 c128; 278ms b128 c64; 280ms b128 c32; 269ms b128 c16

#     # dl = DataLoader(
#     #     ds, batch_size=None, shuffle=True, collate_fn=collate_fn, num_workers=0)

#     dl = get_cache_data_loader(
#         'slakh',
#         tokenizer=NoteEventTokenizer('mt3'),
#         sub_batch_size=32,
#         global_batch_size=32,
#         num_workers=0)

# with Timer() as t:
#     for i, data in enumerate(dl):
#         if i > 4:
#             break
#         print(i)
# t.print_elapsed_time()
