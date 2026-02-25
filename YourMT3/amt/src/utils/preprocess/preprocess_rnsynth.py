# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""preprocess_rnsynth.py 

RNSynth: Randomly generated note sequences using the NSynth dataset.

"""
import os
import random
import glob
import json
import logging
import numpy as np
from typing import Dict, Literal, Optional
from utils.note_event_dataclasses import Note
from utils.audio import get_audio_file_info, load_audio_file, write_wav_file, guess_onset_offset_by_amp_envelope
from utils.midi import note_event2midi
from utils.note2event import note2note_event, sort_notes, validate_notes, trim_overlapping_notes, mix_notes

# yapf: disable
QUALITY_VOCAB = [
    'bright', 'dark', 'distortion', 'fast_decay', 'long_release', 'multiphonic', 'nonlinear_env',
    'percussive', 'reverb', 'tempo-synced'
]

INSTRUMENT_FAMILY_VOCAB = [
    'bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal',
    'synth_lead'
]

INSTRUMENT_SOURCE_VOCAB = ['acoustic', 'electronic', 'synthetic']

INSTRUMENT_MAPPING = {
    # key: (instrument_family, instrument_source)
    ('bass', 'acoustic'): {'program': 32, 'channel': 0, 'allow_poly': False,},
    ('bass', 'electronic'): {'program': 33, 'channel': 0, 'allow_poly': False,},
    ('bass', 'synthetic'): {'program': 38, 'channel': 0, 'allow_poly': False,},
    ('brass', 'acoustic'): {'program': 61, 'channel': 1, 'allow_poly': True,},
    ('brass', 'electronic'): {'program': 62, 'channel': 1, 'allow_poly': True,},
    ('brass', 'synthetic'): {'program': 62, 'channel': 1, 'allow_poly': True, },
    ('flute', 'acoustic'): {'program': 73, 'channel': 2, 'allow_poly': False,},
    ('flute', 'electronic'): {'program': 76, 'channel': 2, 'allow_poly': False,},
    ('flute', 'synthetic'): {'program': 76, 'channel': 2, 'allow_poly': False,},
    ('guitar', 'acoustic'): {'program': 24, 'channel': 3, 'allow_poly': True,},
    ('guitar', 'electronic'): {'program': 27, 'channel': 3, 'allow_poly': True,},
    ('guitar', 'synthetic'): {'program': 27, 'channel': 3, 'allow_poly': True,},
    ('keyboard', 'acoustic'): {'program': 0, 'channel': 4, 'allow_poly': True,},
    ('keyboard', 'electronic'): {'program': 4, 'channel': 4, 'allow_poly': True,},
    ('keyboard', 'synthetic'): {'program': 80, 'channel': 4, 'allow_poly': True,},
    ('mallet', 'acoustic'): {'program': 12, 'channel': 5, 'allow_poly': True,},
    ('mallet', 'electronic'): {'program': 12, 'channel': 5, 'allow_poly': True,},
    ('mallet', 'synthetic'): {'program': 12, 'channel': 5, 'allow_poly': True,},
    ('organ', 'acoustic'): {'program': 16, 'channel': 6, 'allow_poly': True,},
    ('organ', 'electronic'): {'program': 18, 'channel': 6, 'allow_poly': True,},
    ('organ', 'synthetic'): {'program': 18, 'channel': 6, 'allow_poly': True,},
    ('reed', 'acoustic'): {'program': 65, 'channel': 7, 'allow_poly': True,},
    ('reed', 'electronic'): {'program': 83, 'channel': 7, 'allow_poly': True,},
    ('reed', 'synthetic'): {'program': 83, 'channel': 7, 'allow_poly': True,},
    ('string', 'acoustic'): {'program': 48, 'channel': 8, 'allow_poly': True,},
    ('string', 'electronic'): {'program': 50, 'channel': 8, 'allow_poly': True,},
    ('string', 'synthetic'): {'program': 50, 'channel': 8, 'allow_poly': True,},
    # ('vocal', 'acoustic'): [56],
    # ('vocal', 'electronic'): [56],
    # ('vocal', 'synthetic'): [56],
    ('synth_lead', 'acoustic'): {'program': 80, 'channel': 9, 'allow_poly': True,},
    ('synth_lead', 'electronic'): {'program': 80, 'channel': 9, 'allow_poly': True,},
    ('synth_lead', 'synthetic'): {'program': 80, 'channel': 9, 'allow_poly': True,},
}


CHANNEL_INFO = {
    0: {'name': 'bass', 'max_poly': 1},
    1: {'name': 'brass', 'max_poly': 4},
    2: {'name': 'flute', 'max_poly': 1},
    3: {'name': 'guitar', 'max_poly': 6},
    4: {'name': 'keyboard', 'max_poly': 8},
    5: {'name': 'mallet', 'max_poly': 4},
    6: {'name': 'organ', 'max_poly': 8},
    7: {'name': 'reed', 'max_poly': 2},
    8: {'name': 'string', 'max_poly': 4},
    9: {'name': 'synth_lead', 'max_poly': 2},
}
# yapf: enable


class RandomNSynthGenerator(object):

    def __init__(self, channel_info: Dict=CHANNEL_INFO):
        self.num_channels = len(channel_info)
        self.channel_info = channel_info
        self.channel_max_poly = [channel_info[ch]['max_poly'] for ch in range(self.num_channels)]
        
        # channel_space_left[ch]: current state of empty space for notes left in channel
        self.channel_space_left = [0] * self.num_channels
        for ch in range(self.num_channels):
            self.reset_space_left(ch)

    def reset_space_left(self, ch: int):
        max_poly = self.channel_max_poly[ch]
        if max_poly == 1:
            self.channel_space_left[ch] = 1
        else:
            self.channel_space_left[ch] = np.random.randint(1, max_poly + 1 )




def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


def get_duration_by_detecting_offset(audio_file: os.PathLike,
                                     side_info: Optional[str] = None,
                                     offset_threshold: float = 0.02) -> float:
    fs, n_frames, _ = get_audio_file_info(audio_file)
    x = load_audio_file(audio_file, fs=fs)
    if side_info is not None and 'fast_decay' in side_info or 'percussive' in side_info:
        x = x[:int(fs * 2.0)]  # limit to 1.5 sec

    _, offset, _ = guess_onset_offset_by_amp_envelope(
        x, fs=fs, onset_threshold=0., offset_threshold=offset_threshold, frame_size=128)
    offset = min(offset, n_frames)
    dur_sec = np.floor((offset / fs) * 1000) / 1000
    return dur_sec


def random_key_cycle(d: Dict):
    keys = list(d.keys())
    while True:
        random.shuffle(keys)
        for i, key in enumerate(keys):
            is_last_element = (i == len(keys) - 1)  # Check if it's the last element in the cycle
            yield (d[key], is_last_element)


def create_sound_info(base_dir: os.PathLike, logger: logging.Logger,
                      split: Literal['train', 'validation', 'test'], metadata_file: os.PathLike):
    """Create a dictionary of sound info from the metadata file."""
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    logger.info(f'Loaded {metadata_file}. Number of examples: {len(metadata)}')

    # Create a sound_info dictionary
    sound_info = {}  # key: nsynth_id, value: dictionary of sound info
    count_skipped = 0
    skipped_instrument_family = set()
    for i, (k, v) in enumerate(metadata.items()):
        if i % 5000 == 0:
            print(f'Creating sound info {i} / {len(metadata)}')
        nsynth_id = v['note']
        instrument_family = v['instrument_family_str']
        instrument_source = v['instrument_source_str']
        audio_file = os.path.join(base_dir, split, 'audio', k + '.wav')
        if not os.path.exists(audio_file):
            raise FileNotFoundError(audio_file)
        dur_sec = get_duration_by_detecting_offset(
            audio_file, side_info=v['qualities_str'], offset_threshold=0.001)

        if INSTRUMENT_MAPPING.get((instrument_family, instrument_source), None) is not None:
            sound_info[nsynth_id] = {
                'audio_file':
                    audio_file,
                'program':
                    INSTRUMENT_MAPPING[instrument_family, instrument_source]['program'],
                'pitch':
                    int(v['pitch']),
                'velocity':
                    int(v['velocity']),
                'channel_group':
                    INSTRUMENT_MAPPING[instrument_family, instrument_source]['channel'],
                'dur_sec':
                    dur_sec,
            }
        else:
            count_skipped += 1
            skipped_instrument_family.add(instrument_family)
    logger.info(f'Created sound info. Number of examples: {len(sound_info)}')
    logger.info(f'Number of skipped examples: {count_skipped}, {skipped_instrument_family}')
    del metadata

    # Regroup sound_info by channel_group
    sound_info_by_channel_group = {}  # key: channel_group, value: list of sound_info
    num_channel_groups = 10
    for i in range(num_channel_groups):
        sound_info_by_channel_group[i] = {}
    for nsynth_id, info in sound_info.items():
        channel_group = info['channel_group']
        sound_info_by_channel_group[channel_group][nsynth_id] = info
    del sound_info
    channel_group_counts = [
        (CHANNEL_INFO[k]['name'], len(v)) for k, v in sound_info_by_channel_group.items()
    ]
    logger.info('Count of sound_info in each channel_group: {}'.format(channel_group_counts))
    return sound_info_by_channel_group, num_channel_groups






def random_nsynth_generator(data_home: os.PathLike,
                            dataset_name: str = 'random_nsynth',
                            generation_minutes_per_file: float = 4.0) -> None:
    """
    Splits:
        'train'
        'validation'
        'test'

    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
        {
            index:
            {
                'random_nsynth_id': random_nsynth_id, # = nsynth_id
                'n_frames': (int),
                'stem_file': 'path/to/stem.npy',
                'mix_audio_file': 'path/to/mix.wav',
                'notes_file': 'path/to/notes.npy',
                'note_events_file': 'path/to/note_events.npy',
                'midi_file': 'path/to/midi.mid', # this is 120bpm converted midi file from note_events
                'program': List[int],   
                'is_drum': List[int], # [0] or [1]
            }
        }
    """
    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # Setup logger
    log_file = os.path.join(base_dir, 'sound_genetation_log.txt')
    logger = setup_logger(log_file)

    # Load annotation json file as dictionary
    split = 'validation'
    metadata_file = os.path.join(base_dir, split, 'examples.json')

    # Create a sound_info dictionary
    sound_info_by_channel_group, num_channel_groups = create_sound_info(
        base_dir, logger, split, metadata_file)

    # Gnenerate random note sequences
    max_frames_per_file = int(generation_minutes_per_file * 60 * 16000)
    sound_gens = [
        random_key_cycle(sound_info_by_channel_group[key])
        for key in sorted(sound_info_by_channel_group.keys())
    ]

    # 5-minute audio generation
    notes = []
    y = np.zeros((num_channel_groups, max_frames_per_file), dtype=np.float32) # (C, L)
    bass_channel = 0  # loop for a cycle of bass channel generation
    cur_frame = 0
    # is_last_element_bass = False
    #while cur_frame < max_frames_per_file and is_last_element_bass == False:

        # x: source audio, y: target audio for each channel
        x_info, is_last_element = next(sound_gens[ch])
        if ch == bass_channel:
            is_last_element = is_last_element_bass
        
        # info about this channel
        onset_in_frame = cur_frame
        offset_in_frame = cur_frame + int(x_info['dur_sec'] * 16000)

        x = load_audio_file(x_info['audio_file'], fs=16000)
        x = x[:int(x_info['dur_sec'] * 16000)]
        y[ch, :] = 0




def preprocess_random_nsynth_16k(data_home=os.PathLike, dataset_name='random_nsynth') -> None:
    """
    Splits:
        'train'
        'validation'
        'test'

    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
        {
            index:
            {
                'random_nsynth_id': random_nsynth_id, # = nsynth_id
                'n_frames': (int),
                'stem_file': 'path/to/stem.npy',
                'mix_audio_file': 'path/to/mix.wav',
                'notes_file': 'path/to/notes.npy',
                'note_events_file': 'path/to/note_events.npy',
                'midi_file': 'path/to/midi.mid', # this is 120bpm converted midi file from note_events
                'program': List[int],   
                'is_drum': List[int], # [0] or [1]
            }
        }
    """
    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # Setup logger
    log_file = os.path.join(base_dir, 'log.txt')
    logger = setup_logger(log_file)

    # Load annotation json file as dictionary
    split = 'validation'
    metadata_file = os.path.join(base_dir, split, 'examples.json')
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    logger.info(f'Loaded {metadata_file}. Number of examples: {len(metadata)}')