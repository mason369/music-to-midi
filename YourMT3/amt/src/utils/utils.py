# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
import os
import json
import time
import hashlib
import requests
import tarfile
import warnings
import argparse
from typing import Tuple, Union, Optional, List, Dict, Any
from tqdm import tqdm
import numpy as np
from collections import Counter
from utils.note_event_dataclasses import Note
from utils.note2event import note2note_event
from utils.midi import note_event2midi
from utils.note2event import slice_multiple_note_events_and_ties_to_bundle
from utils.event2note import merge_zipped_note_events_and_ties_to_notes
from utils.metrics import compute_track_metrics
from utils.tokenizer import EventTokenizer, NoteEventTokenizer
from utils.note_event_dataclasses import Note, NoteEvent, Event
from config.vocabulary import GM_INSTR_FULL, GM_INSTR_CLASS_PLUS
from config.config import shared_cfg


def get_checksum(file_path: os.PathLike, buffer_size: int = 65536) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def download_and_extract(data_home: os.PathLike,
                         url: str,
                         remove_tar_file: bool = True,
                         check_sum: Optional[str] = None,
                         zenodo_token: Optional[str] = None) -> None:

    file_name = url.split("/")[-1].split("?")[0]
    tar_path = os.path.join(data_home, file_name)

    if not os.path.exists(data_home):
        os.makedirs(data_home)

    if zenodo_token is not None:
        url_with_token = f"{url}&token={zenodo_token}" if "?download=1" in url else f"{url}?token={zenodo_token}"
    else:
        url_with_token = url

    response = requests.get(url_with_token, stream=True)

    # Check HTTP Status
    if response.status_code != 200:
        print(f"Failed to download file. Status code: {response.status_code}")
        return

    total_size = int(response.headers.get('content-length', 0))

    with open(tar_path, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size // 8192, unit='KB', desc=file_name):
            f.write(chunk)

    _check_sum = get_checksum(tar_path)
    print(f"Checksum (md5): {_check_sum}")

    if check_sum is not None and check_sum != _check_sum:
        raise ValueError(f"Checksum doesn't match! Expected: {check_sum}, Actual: {_check_sum}")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_home)

    if remove_tar_file:
        os.remove(tar_path)


def create_inverse_vocab(vocab: Dict) -> Dict:
    inverse_vocab = {}
    for k, vnp in vocab.items():
        for v in vnp:
            inverse_vocab[v] = (vnp[0], k)  # (program, str_instrument_name)
    return inverse_vocab


def create_program2channel_vocab(program_vocab: Dict, drum_program: int = 128, force_assign_13_ch: bool = False):
    """
    Create a direct map for programs to indices, instrument groups, and primary programs.
    
    Args:
        program_vocab (dict): A dictionary of program vocabularies.
        drum_program (int): The program number for drums. Default: 128.
    
    Returns:
        program2channel_vocab (dict): A dictionary of program to indices, instrument groups, and primary programs.
            e.g. {
                0: {'channel': 0, 'instrument_group': 'Piano', 'primary_program': 0},
                1: {'channel': 1, 'instrument_group': 'Chromatic Percussion', 'primary_program': 8},
                ...
                100: {'channel': 11, 'instrument_group': 'Singing Voice', 'primary_program': 100},
                128: {'channel': 12, 'instrument_group': 'Drums', 'primary_program': 128}
                }
            "primary_program" is not used now.
        
        num_channels (int): The number of channels. Typically length of program vocab + 1 (for drums)
    
    """
    num_channels = len(program_vocab) + 1
    program2channel_vocab = {}
    for idx, (instrument_group, programs) in enumerate(program_vocab.items()):
        if idx > num_channels:
            raise ValueError(
                f"📕 The number of channels ({num_channels}) is less than the number of instrument groups ({idx})")
        for program in programs:
            if program in program2channel_vocab:
                raise ValueError(f"📕 program {program} is duplicated in program_vocab")
            else:
                program2channel_vocab[program] = {
                    "channel": int(idx),
                    "instrument_group": str(instrument_group),
                    "primary_program": int(programs[0]),
                }

    # Add drums
    if drum_program in program2channel_vocab.keys():
        raise ValueError(
            f"📕 drum_program {drum_program} is duplicated in program_vocab. program_vocab should not include drum or program 128"
        )
    else:
        program2channel_vocab[drum_program] = {
            "channel": idx + 1,
            "instrument_group": "Drums",
            "primary_program": drum_program,
        }
    return program2channel_vocab, num_channels


def write_model_output_as_npy(data, output_dir, track_id):
    output_dir = os.path.join(output_dir, "model_output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"output_{track_id}.npy")
    np.save(output_file, data, allow_pickle=True)


def write_model_output_as_midi(notes: List[Note],
                               output_dir: os.PathLike,
                               track_id: str,
                               output_inverse_vocab: Optional[Dict] = None,
                               output_dir_suffix: Optional[str] = None) -> None:

    if output_dir_suffix is not None:
        output_dir = os.path.join(output_dir, f"model_output/{output_dir_suffix}")
    else:
        output_dir = os.path.join(output_dir, "model_output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{track_id}.mid")

    if output_inverse_vocab is not None:
        # Convert the note events to the output vocabulary
        new_notes = []
        for note in notes:
            if note.is_drum:
                new_notes.append(note)
            else:
                new_notes.append(
                    Note(is_drum=note.is_drum,
                         program=output_inverse_vocab.get(note.program, [note.program])[0],
                         onset=note.onset,
                         offset=note.offset,
                         pitch=note.pitch,
                         velocity=note.velocity))

    note_events = note2note_event(new_notes, return_activity=False)
    note_event2midi(note_events, output_file, output_inverse_vocab=output_inverse_vocab)


def write_err_cnt_as_json(
    track_id: str,
    output_dir: os.PathLike,
    output_dir_suffix: Optional[str] = None,
    note_err_cnt: Optional[Counter] = None,
    note_event_err_cnt: Optional[Counter] = None,
):

    if output_dir_suffix is not None:
        output_dir = os.path.join(output_dir, f"model_output/{output_dir_suffix}")
    else:
        output_dir = os.path.join(output_dir, "model_output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"error_count_{track_id}.json")

    output_dict = {}
    if note_err_cnt is not None:
        output_dict['note_err_cnt'] = dict(note_err_cnt)
    if note_event_err_cnt is not None:
        output_dict['note_event_err_cnt'] = dict(note_event_err_cnt)
    output_str = json.dumps(output_dict, indent=4)

    with open(output_file, 'w') as json_file:
        json_file.write(output_str)


class Timer:
    """A simple timer class to measure elapsed time.
    Usage:

    with Timer() as t:
        # Your code here
        time.sleep(2)
    t.print_elapsed_time()

    """

    def __init__(self) -> None:
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> None:
        self.end_time = time.time()

    def elapsed_time(self) -> float:
        if self.start_time is None:
            raise ValueError("Timer has not been started yet.")
        if self.end_time is None:
            raise ValueError("Timer has not been stopped yet.")
        return self.end_time - self.start_time

    def print_elapsed_time(self, message: Optional[str] = None) -> float:
        elapsed_seconds = self.elapsed_time()
        minutes, seconds = divmod(elapsed_seconds, 60)
        milliseconds = (elapsed_seconds % 1) * 1000
        if message is not None:
            text = f"⏰ {message}: {int(minutes)}m {int(seconds)}s {milliseconds:.2f}ms"
        else:
            text = f"⏰ elapse time: {int(minutes)}m {int(seconds)}s {milliseconds:.2f}ms"
        print(text)
        return elapsed_seconds

    def reset(self) -> None:
        self.start_time = None
        self.end_time = None

    def __enter__(self) -> 'Timer':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def merge_file_lists(file_lists: List[Dict]) -> Dict[int, Any]:
    """ Merge file lists from different datasets, and return a reindexed 
    dictionary of file list."""
    merged_file_list = {}
    index = 0
    for file_list in file_lists:
        for v in file_list.values():
            merged_file_list[index] = v
            index += 1
    return merged_file_list


def merge_splits(splits: List[str], dataset_name: Union[str, List[str]]) -> Dict[int, Any]:
    """ 
    merge_splits:
    - Merge multiple splits from different datasets, and return a reindexed 
    dictionary of file list. 
    - It is also possible to merge splits from different datasets.
    
    """
    n_splits = len(splits)
    if n_splits > 1 and isinstance(dataset_name, str):
        dataset_name = [dataset_name] * n_splits
    elif n_splits > 1 and isinstance(dataset_name, list) and len(dataset_name) != n_splits:
        raise ValueError("The number of dataset names in list must be equal to the number of splits.")
    else:
        pass

    # load file_list dictionaries
    data_home = shared_cfg['PATH']['data_home']
    file_lists = []  # list of dictionaries
    for i, s in enumerate(splits):
        json_file = (f"{data_home}/yourmt3_indexes/{dataset_name[i]}_{s}_file_list.json")

        # Fix for missing file_list with incomplete dataset package
        if not os.path.exists(json_file):
            warnings.warn(
                f"File list {json_file} does not exist. If you don't have a complete package of dataset, ignore this warning..."
            )
            return {}

        with open(json_file, 'r') as j:
            file_lists.append(json.load(j))
    merged_file_list = merge_file_lists(file_lists)  # reindexed, merged file list
    return merged_file_list


def reindex_file_list_keys(file_list: Dict[str, Any]) -> Dict[int, Any]:
    """ Reindex file list keys from 0 to total count."""
    reindexed_file_list = {}
    for i, (k, v) in enumerate(file_list.items()):
        reindexed_file_list[i] = v
    return reindexed_file_list


def remove_ids_from_file_list(file_list: Dict[str, Any],
                              selected_ids: List[int],
                              reindex: bool = True) -> Dict[int, Any]:
    """ Remove selected ids from file list."""
    key = None
    for v in file_list.values():
        # search keys that contain 'id'
        for k in v.keys():
            if 'id' in k:
                key = k
                break
        if key:
            break

    if key is None:
        raise ValueError("No key contains 'id'.")

    # generate new filelist by removing selected ids
    selected_ids = [str(id) for id in selected_ids]  # ids to str
    file_list = {k: v for k, v in file_list.items() if str(v[key]) not in selected_ids}
    if reindex:
        return reindex_file_list_keys(file_list)
    else:
        return file_list


def deduplicate_splits(split_a: Union[str, Dict],
                       split_b: Union[str, Dict],
                       dataset_name: Optional[str] = None,
                       reindex: bool = True) -> Dict[int, Any]:
    """Remove overlapping splits in file_list A with splits from file_list B,
    and return a reindexed dictionary of files."""
    data_home = shared_cfg['PATH']['data_home']

    if isinstance(split_a, str):
        json_file_a = (f"{data_home}/yourmt3_indexes/{dataset_name}_{split_a}_file_list.json")
        with open(json_file_a, 'r') as j:
            file_list_a = json.load(j)
    elif isinstance(split_a, dict):
        file_list_a = split_a

    if isinstance(split_b, str):
        json_file_b = (f"{data_home}/yourmt3_indexes/{dataset_name}_{split_b}_file_list.json")
        with open(json_file_b, 'r') as j:
            file_list_b = json.load(j)
    elif isinstance(split_b, dict):
        file_list_b = split_b

    # Get the key that contains 'id' from file_list_a splits
    id_key = None
    for v in file_list_a.values():
        for k in v.keys():
            if 'id' in k:
                id_key = k
                break
        if id_key:
            break
    if id_key is None:
        raise ValueError("No key contains 'id' in file_list_a.")

    # Get IDs from file_list_b splits
    ids_b = set(str(v.get(id_key, '')) for v in file_list_b.values())

    # Extract IDs from file_list_a splits
    ids_a = [str(v.get(id_key, '')) for v in file_list_a.values()]

    # Remove IDs from file_list_a that are also in file_list_b
    ids_to_remove = list(set(ids_a).intersection(ids_b))
    filtered_file_list_a = remove_ids_from_file_list(file_list_a, ids_to_remove, reindex)

    return filtered_file_list_a


def merge_vocab(vocab_list: List[Dict]) -> Dict[str, Any]:
    """ Merge file lists from different datasets, and return a reindexed 
    dictionary of file list."""
    merged_vocab = {}
    for vocab in vocab_list:
        for k, v in vocab.items():
            if k not in merged_vocab.keys():
                merged_vocab[k] = v
    return merged_vocab


def assert_note_events_almost_equal(actual_note_events,
                                    predicted_note_events,
                                    ignore_time=False,
                                    ignore_activity=True,
                                    delta=5.1e-3):
    """
    Asserts that the given lists of Note instances are equal up to a small
    floating-point tolerance, similar to `assertAlmostEqual` of `unittest`.
    Tolerance is 5e-3 by default, which is 5 ms for 100 ticks-per-second.

    If `ignore_time` is True, then the time field is ignored. (useful for 
    comparing tie note events, default is False)

    If `ignore_activity` is True, then the activity field is ignored (default
    is True).
    """
    assert len(actual_note_events) == len(predicted_note_events)
    for j, (actual_note_event, predicted_note_event) in enumerate(zip(actual_note_events, predicted_note_events)):
        if ignore_time is False:
            assert abs(actual_note_event.time - predicted_note_event.time) <= delta, (j, actual_note_event,
                                                                                      predicted_note_event)
        assert actual_note_event.is_drum == predicted_note_event.is_drum, (j, actual_note_event, predicted_note_event)
        assert actual_note_event.program == predicted_note_event.program, (j, actual_note_event, predicted_note_event)
        assert actual_note_event.pitch == predicted_note_event.pitch, (j, actual_note_event, predicted_note_event)
        assert actual_note_event.velocity == predicted_note_event.velocity, (j, actual_note_event, predicted_note_event)
        if ignore_activity is False:
            assert actual_note_event.activity == predicted_note_event.activity, (j, actual_note_event,
                                                                                 predicted_note_event)


def note_event2token2note_event_sanity_check(note_events: List[NoteEvent],
                                             notes: List[Note],
                                             report_err_cnt=False) -> Counter:
    # slice note events
    max_time = note_events[-1].time
    num_segs = int(max_time * 16000 // 32757 + 1)
    seg_len_sec = 32767 / 16000
    start_times = [i * seg_len_sec for i in range(num_segs)]
    note_event_segments = slice_multiple_note_events_and_ties_to_bundle(
        note_events,
        start_times,
        seg_len_sec,
    )

    # encode
    tokenizer = NoteEventTokenizer()
    token_array = np.zeros((num_segs, 1024), dtype=np.int32)
    for i, tup in enumerate(list(zip(*note_event_segments.values()))):
        padded_tokens = tokenizer.encode_plus(*tup)
        token_array[i, :] = padded_tokens

    # decode: warning: Invalid pitch event without program or velocity --> solved
    zipped_note_events_and_tie, list_events, err_cnt = tokenizer.decode_list_batches([token_array],
                                                                                     start_times,
                                                                                     return_events=True)
    if report_err_cnt:
        # report error and do not break..
        err_cnt_all = err_cnt
    else:
        assert len(err_cnt) == 0
        err_cnt_all = Counter()

    # First check, the number of empty note_events and tie_note_events
    cnt_org_empty = 0
    cnt_recon_empty = 0
    for i, (recon_note_events, recon_tie_note_events, _, _) in enumerate(zipped_note_events_and_tie):
        org_note_events = note_event_segments['note_events'][i]
        org_tie_note_events = note_event_segments['tie_note_events'][i]
        if org_note_events == []:
            cnt_org_empty += 1
        if recon_note_events == []:
            cnt_recon_empty += 1

    # assert len(org_note_events) == len(recon_note_events)  # passed after bug fix

    # Check the reconstruction of note_events
    for i, (recon_note_events, recon_tie_note_events, _, _) in enumerate(zipped_note_events_and_tie):
        org_note_events = note_event_segments['note_events'][i]
        org_tie_note_events = note_event_segments['tie_note_events'][i]

        org_note_events.sort(key=lambda n_ev: (n_ev.time, n_ev.is_drum, n_ev.program, n_ev.velocity, n_ev.pitch))
        org_tie_note_events.sort(key=lambda n_ev: (n_ev.program, n_ev.pitch))
        recon_note_events.sort(key=lambda n_ev: (n_ev.time, n_ev.is_drum, n_ev.program, n_ev.velocity, n_ev.pitch))
        recon_tie_note_events.sort(key=lambda n_ev: (n_ev.program, n_ev.pitch))

        #assert_note_events_almost_equal(org_note_events, recon_note_events)
        # assert_note_events_almost_equal(
        #     org_tie_note_events, recon_tie_note_events, ignore_time=True)

    # Check notes: of course this fails.. and a lot of warning for cut off 20s
    recon_notes, err_cnt = merge_zipped_note_events_and_ties_to_notes(zipped_note_events_and_tie, fix_offset=False)
    # assert len(err_cnt) == 0  # this error is due to the cut off 5 seconds...

    # Check metric
    drum_metric, non_drum_metric, instr_metric = compute_track_metrics(recon_notes,
                                                                       notes,
                                                                       eval_vocab=GM_INSTR_FULL,
                                                                       onset_tolerance=0.005)  # 5ms
    if not np.isnan(non_drum_metric['offset_f']) and non_drum_metric['offset_f'] != 1.0:
        warnings.warn(f"non_drum_metric['offset_f'] = {non_drum_metric['offset_f']}")
        assert non_drum_metric['onset_f'] > 0.99
    if not np.isnan(drum_metric['onset_f_drum']) and non_drum_metric['offset_f'] != 1.0:
        warnings.warn(f"drum_metric['offset_f'] = {drum_metric['offset_f']}")
        assert drum_metric['offset_f'] > 0.99
    return err_cnt_all + Counter(err_cnt)


def str2bool(v):
    """
    Converts a string value to a boolean value.

    Args:
        v: The string value to convert.

    Returns:
        The boolean value equivalent of the input string.

    Raises:
        ArgumentTypeError: If the input string is not a valid boolean value.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def freq_to_midi(freq):
    return round(69 + 12 * np.log2(freq / 440))


def dict_iterator(d: Dict):
    """
    This function is used to iterate over a dictionary of lists.
    As an output, it yields a newly created instance of a dictionary 
    """
    for values in zip(*d.values()):
        yield {k: [v] for k, v in zip(d.keys(), values)}


def extend_dict(dict1: dict, dict2: dict) -> None:
    """
    Extends the lists in dict1 with the corresponding lists in dict2. 
    Modifies dict1 in-place and does not return anything.
    
    Args:
        dict1 (dict): The dictionary to be extended.
        dict2 (dict): The dictionary with lists to extend dict1.
        
    Example:
        dict1 = {'a': [1,2,3], 'b':[4,5,6]}
        dict2 = {'a':[10], 'b':[17]}
        extend_dict_in_place(dict1, dict2)
        print(dict1)  # Outputs: {'a': [1, 2, 3, 10], 'b': [4, 5, 6, 17]}
    """
    for key in dict1:
        dict1[key].extend(dict2[key])
