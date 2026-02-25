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
from typing import Dict, Any, Union, Tuple, Optional

import torch
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from utils.audio import load_audio_file, slice_padded_array
from utils.tokenizer import EventTokenizerBase, NoteEventTokenizer
from utils.note2event import slice_multiple_note_events_and_ties_to_bundle
from utils.note_event_dataclasses import Note, NoteEvent, NoteEventListsBundle
from utils.task_manager import TaskManager
from config.config import shared_cfg
from config.config import audio_cfg as default_audio_cfg

UNANNOTATED_PROGRAM = 129


class AudioFileDataset(Dataset):
    """
    🎧 AudioFileDataset for validation/test:
    
    This dataset class is designed to be used ONLY with `batch_size=None` and 
    returns sliced audio segments and unsliced notes and sliced note events for
     a single song when `__getitem__` is called.

    Args:
        file_list (Union[str, bytes, os.PathLike], optional):
            Path to the file list. e.g. "../../data/yourmt3_indexes/slakh_validation_file_list.json"
        task_manager (TaskManager, optional): TaskManager instance. Defaults to TaskManager().
        fs (int, optional): Sampling rate. Defaults to 16000.
        seg_len_frame (int, optional): Segment length in frames. Defaults to 32767.
        seg_hop_frame (int, optional): Segment hop in frames. Defaults to 32767.
        sub_batch_size (int, optional): Sub-batch size that will be used in 
            generation of tokens. Defaults to 32.
        max_num_files (int, optional): Maximum number of files to be loaded. Defaults to None.
        
    
    Variables:
        file_list:
            '{dataset_name}_{split}_file_list.json' has the following keys:
            {
                'index':
                    {
                        'mtrack_id': mtrack_id,
                        'n_frames': n of audio frames
                        'stem_file': Dict of stem audio file info
                        'mix_audio_file': mtrack.mix_path,
                        'notes_file': available only for 'validation' and 'test'
                        'note_events_file': available only for 'train' and 'validation'
                        'midi_file': mtrack.midi_path
                    }
            }
            
    __getitem__(index) returns:

        audio_segment:
            torch.FloatTensor: (nearest_N_divisable_by_sub_batch_size, 1, seg_len_frame)

        notes_dict:
            {
                'mtrack_id': int,
                'program': List[int],
                'is_drum': bool, 
                'duration_sec': float, 
                'notes': List[Note], 
            }
            
        token_array:
            torch.LongTensor: (n_segments, seg_len_frame)

    """

    def __init__(
            self,
            file_list: Union[str, bytes, os.PathLike],
            task_manager: TaskManager = TaskManager(),
            #  tokenizer: Optional[EventTokenizerBase] = None,
            fs: int = 16000,
            seg_len_frame: int = 32767,
            seg_hop_frame: int = 32767,
            max_num_files: Optional[int] = None) -> None:

        # load the file list
        with open(file_list, 'r') as f:
            fl = json.load(f)
        file_list = {int(key): value for key, value in fl.items()}
        if max_num_files:  # reduce the number of files
            self.file_list = dict(list(file_list.items())[:max_num_files])
        else:
            self.file_list = file_list

        self.fs = fs
        self.seg_len_frame = seg_len_frame
        self.seg_len_sec = seg_len_frame / fs
        self.seg_hop_frame = seg_hop_frame
        self.task_manager = task_manager

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict, NoteEventListsBundle]:
        # get metadata
        metadata = self.file_list[index]
        audio_file = metadata['mix_audio_file']
        notes_file = metadata['notes_file']
        note_events_file = metadata['note_events_file']

        # load the audio
        audio = load_audio_file(audio_file, dtype=np.int16)  # returns bytes
        audio = audio / 2**15
        audio = audio.astype(np.float32)
        audio = audio.reshape(1, -1)
        audio_segments = slice_padded_array(
            audio,
            self.seg_len_frame,
            self.seg_hop_frame,
            pad=True,
        )  # (n_segs, seg_len_frame)
        audio_segments = rearrange(audio_segments, 'n t -> n 1 t').astype(np.float32)
        num_segs = audio_segments.shape[0]

        # load all notes and from a file (of a single song)
        notes_dict = np.load(notes_file, allow_pickle=True, fix_imports=False).tolist()

        # TODO: add midi_file path in preprocessing instead of here
        notes_dict['midi_file'] = metadata['midi_file']

        # tokenize note_events
        note_events_dict = np.load(note_events_file, allow_pickle=True, fix_imports=False).tolist()

        if self.task_manager.tokenizer is not None:
            # not using seg_len_sec to avoid accumulated rounding errors
            start_times = [i * self.seg_hop_frame / self.fs for i in range(num_segs)]
            note_event_segments = slice_multiple_note_events_and_ties_to_bundle(
                note_events_dict['note_events'],
                start_times,
                self.seg_len_sec,
            )

            # Support for multi-channel decoding
            if UNANNOTATED_PROGRAM in notes_dict['program']:
                has_unannotated_segments = [True] * num_segs
            else:
                has_unannotated_segments = [False] * num_segs

            token_array = self.task_manager.tokenize_note_events_batch(note_event_segments,
                                                                       start_time_to_zero=False,
                                                                       sort=True)
            # note_token_array = self.task_manager.tokenize_note_events_batch(note_event_segments,
            #                                                                 start_time_to_zero=False,
            #                                                                 sort=True)
            # task_token_array = self.task_manager.tokenize_task_events_batch(note_event_segments,
            #                                                                 has_unannotated_segments)

        # Shape:
        #   processed_audio_array: (num_segs, 1, nframe)
        #   notes_dict: Dict
        #   note_token_array: (num_segs, decoding_ch, max_note_token_len)
        #   task_token_array: (num_segs, decoding_ch, max_task_token_len)
        # return torch.from_numpy(audio_segments), notes_dict, torch.from_numpy(
        #     note_token_array).long(), torch.from_numpy(task_token_array).long()
        return torch.from_numpy(audio_segments), notes_dict, torch.from_numpy(token_array).long()

        # # Tokenize/pad note_event_segments -> array of token and mask
        # max_len = self.tokenizer.max_length
        # token_array = np.zeros((num_segs, max_len), dtype=np.int32)

        # for i, tup in enumerate(list(zip(*note_event_segments.values()))):
        #     padded_tokens = self.tokenizer.encode_plus(*tup)
        #     token_array[i, :] = padded_tokens
        # return torch.from_numpy(audio_segments), notes_dict, torch.from_numpy(token_array).long()

    def __len__(self) -> int:
        return len(self.file_list)


def get_eval_dataloader(
    dataset_name: str,
    split: str = 'validation',
    dataloader_config: Dict = {"num_workers": 0},
    task_manager: TaskManager = TaskManager(),
    # tokenizer: Optional[EventTokenizerBase] = NoteEventTokenizer('mt3'),
    max_num_files: Optional[int] = None,
    audio_cfg: Optional[Dict] = None,
) -> DataLoader:
    """
    🎧 get_audio_file_dataloader:
    
    This function returns a dataloader for AudioFileDataset that returns padded slices
    of audio samples with the divisable number of sub-batch size.
    """
    data_home = shared_cfg["PATH"]["data_home"]
    file_list = f"{data_home}/yourmt3_indexes/{dataset_name}_{split}_file_list.json"

    if audio_cfg is None:
        audio_cfg = default_audio_cfg

    ds = AudioFileDataset(
        file_list,
        task_manager=task_manager,
        # tokenizer=tokenizer,
        seg_len_frame=int(audio_cfg["input_frames"]),  # Default: 32767
        seg_hop_frame=int(audio_cfg["input_frames"]),  # Default: 32767
        max_num_files=max_num_files)
    dl = DataLoader(ds, batch_size=None, collate_fn=lambda k: k, **dataloader_config)
    return dl
