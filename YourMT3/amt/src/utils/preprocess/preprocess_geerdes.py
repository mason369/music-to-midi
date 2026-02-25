"""preprocess_geerdes.py"""
import os
import glob
import re
import json
import csv
import logging
import random
from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
from utils.audio import get_audio_file_info, load_audio_file
from utils.midi import midi2note, note_event2midi
from utils.note2event import (note2note_event, sort_notes, validate_notes, trim_overlapping_notes,
                              extract_program_from_notes, extract_notes_selected_by_programs)
from utils.event2note import event2note_event
from utils.note_event_dataclasses import Note, NoteEvent
from utils.utils import note_event2token2note_event_sanity_check, create_inverse_vocab
from config.vocabulary import MT3_FULL_PLUS

GEERDES_DATA_CSV_FILENAME = 'geerdes_data_final.csv'
DRUM_CHANNEL = 9  # all drums are in channel 9 in geerdes dataset
DRUM_PROGRAM = 128
SINGING_VOICE_PROGRAM = 100
SINGING_VOICE_CHORUS_PROGRAM = 101  # representing backup vocals and choir
TRACK_NAME_TO_PROGRAM_MAP = { # compared by exact match of lowercase
    "vocal": SINGING_VOICE_PROGRAM,
    "vocalist": SINGING_VOICE_PROGRAM,
    "2nd Vocals/backings/harmony": SINGING_VOICE_CHORUS_PROGRAM,
    "backvocals": SINGING_VOICE_CHORUS_PROGRAM,
}


def format_number(n, width=5):
    """
    Format a number to a fixed width string, padding with leading zeros if needed.
    
    Parameters:
    - n (int): The number to be formatted.
    - width (int, optional): The desired fixed width for the resulting string. Default is 5.
    
    Returns:
    - str: The formatted string representation of the number.
    
    Example:
    >>> format_number(123)
    '00123'
    >>> format_number(7, 3)
    '007'
    """
    return f"{int(n):0{width}}"


def find_index_with_key(lst, key):
    # only checks alphanumeric characters, ignoring upper/lower case
    def filter_string(s):
        return re.sub(r'[^a-zA-Z0-9]', '', s)

    filtered_key = filter_string(key).lower()
    indices = [
        index for index, value in enumerate(lst) if filtered_key in filter_string(value.lower())
    ]

    if len(indices) > 1:
        raise ValueError(f"'{key}'has more than two matching song titles.")
    elif len(indices) == 1:
        return indices[0]
    else:
        return None


"""Code below was used to generate the "geerdes_data_final.csv" file for the Geerdes dataset split info."""
# def split_and_generate_data_info_csv(data_home=os.PathLike, dataset_name='geerdes') -> None:
#     """Preprocess Geerdes dataset."""
#     # Directory and file paths
#     base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
#     output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
#     os.makedirs(output_index_dir, exist_ok=True)

#     # Setup logger
#     log_file = os.path.join(base_dir, 'log.txt')
#     logger = logging.getLogger('my_logger')
#     logger.setLevel(logging.DEBUG)
#     file_handler = logging.FileHandler(log_file)
#     formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
#     file_handler.setFormatter(formatter)
#     if not logger.handlers:
#         logger.addHandler(file_handler)
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.DEBUG)
#     console_formatter = logging.Formatter('%(levelname)s - %(message)s')
#     console_handler.setFormatter(console_formatter)
#     logger.addHandler(console_handler)

#     # Load CSV: construct id to midi/wav dictionary
#     csv_file = os.path.join(base_dir, 'tracks_title_corrected.csv')
#     tracks_all = {}
#     with open(csv_file, 'r') as f:
#         reader = csv.reader(f)
#         next(reader)  # skip header

#         for row in reader:
#             geerdes_id = format_number(row[0])
#             title = row[1]
#             artist = row[2]
#             link = row[6]
#             tracks_all[geerdes_id] = {'title': title}
#             tracks_all[geerdes_id]['artist'] = artist
#             tracks_all[geerdes_id]['link'] = link
#         logger.info(f'Loaded {len(tracks_all)} tracks from {csv_file}.')

#     # Search existing audio files
#     audio_dir = os.path.join(base_dir, 'audio_16k_final')
#     _audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
#     audio_files = [
#         file for file in _audio_files
#         if not file.endswith('_vocals.wav') and not file.endswith('_accompaniment.wav')
#     ]
#     gid_no_audio = []
#     gid_has_audio = []
#     audio_matched = set()
#     audio_no_match = set()

#     for geerdes_id in tracks_all.keys():
#         title = tracks_all[geerdes_id]['title']
#         artist = tracks_all[geerdes_id]['artist']
#         # Find matching audio file
#         audio_file_id = find_index_with_key(audio_files, title)
#         if audio_file_id is not None:
#             # add audio file to tracks_all
#             audio_file = audio_files[audio_file_id]
#             tracks_all[geerdes_id]['audio_file'] = audio_file
#             gid_has_audio.append(geerdes_id)
#             audio_matched.add(audio_file)
#         else:
#             logger.info(f'No matching audio file found for {artist} - {title}.')
#             gid_no_audio.append(geerdes_id)
#             continue

#     audio_no_match = set(audio_files) - audio_matched
#     logger.info(
#         f'Found {len(audio_files)} audio files. {len(gid_no_audio)} geerdes_ids have no audio files. {gid_no_audio}'
#     )
#     logging.warning(
#         f'{len(audio_no_match)} audio files have no matching geerdes_id. {audio_no_match}')

#     # Search existing midi files
#     midi_dir = os.path.join(base_dir, 'aligned_midifiles_corrected')
#     midi_files = glob.glob(os.path.join(midi_dir, '*.mid')) + glob.glob(
#         os.path.join(midi_dir, '*.MID'))
#     logger.info(f'Found {len(midi_files)} midi files in {midi_dir}.')

#     # Construct id to midi/wav dictionary
#     gid_no_midi = []
#     gid_has_midi = []
#     for geerdes_id in tracks_all.keys():
#         expected_midi_file = os.path.join(midi_dir, geerdes_id + 'T.MID')
#         if os.path.exists(expected_midi_file):
#             gid_has_midi.append(geerdes_id)
#             tracks_all[geerdes_id]['midi_file'] = expected_midi_file
#         else:
#             artist = tracks_all[geerdes_id]['artist']
#             title = tracks_all[geerdes_id]['title']
#             logging.warning(
#                 f'No matching midi file found for {expected_midi_file}, {artist} - {title}')
#             tracks_all[geerdes_id]['midi_file'] = expected_midi_file
#             gid_no_midi.append(geerdes_id)

#     # Final dictionary where audio and midi files are matched
#     gid_has_midi_and_audio = set(gid_has_midi) & set(gid_has_audio)
#     gid_midi_or_audio_missing = set(gid_no_midi).union(set(gid_no_audio))
#     assert len(gid_has_midi_and_audio) + len(gid_midi_or_audio_missing) == len(tracks_all)
#     logger.info(f'Found {len(gid_has_midi_and_audio)} tracks with both midi and audio files.')
#     logging.warning(
#         f'Found {len(gid_midi_or_audio_missing)} tracks with either midi or audio files missing.')

#     for gid in gid_midi_or_audio_missing:
#         tracks_all.pop(gid)
#     logger.info(f'Final number of tracks: {len(tracks_all)}.')

#     # Stratified split using artist name 5:5
#     artist_groups = {}
#     for id, info in tracks_all.items():
#         artist = info['artist']
#         if artist not in artist_groups:
#             artist_groups[artist] = []
#         artist_groups[artist].append((id, info))

#     train_set = {}
#     test_set = {}
#     for artist, tracks in artist_groups.items():
#         if len(tracks) == 1:
#             if random.random() < 0.5:
#                 train_set[tracks[0][0]] = tracks[0][1]
#             else:
#                 test_set[tracks[0][0]] = tracks[0][1]
#         else:
#             split_index = len(tracks) // 2
#             for id, info in tracks[:split_index]:
#                 train_set[id] = info
#             for id, info in tracks[split_index:]:
#                 test_set[id] = info
#     logger.info("Train Set:", len(train_set))
#     logger.info("Test Set:", len(test_set))
#     gid_train = list(train_set.keys())
#     gid_validation = list(test_set.keys())

#     # Create split information
#     gid_all = np.random.permutation(list(tracks_all.keys()))
#     gid_train = gid_all[:50]
#     gid_validation = gid_all[50:]
#     for k, v in tracks_all.items():
#         if k in gid_train:
#             v['split_half'] = 'train'
#         elif k in gid_validation:
#             v['split_half'] = 'validation'
#         else:
#             raise ValueError(f'Invalid split for {k}.')
#     logger.info(
#         f'Split information created.\ngid_train: {gid_train}\n gid_validation: {gid_validation}.')

#     # Remove base_dir from audio_file and midi_file
# for v in tracks_all.values():
#     v['audio_file'] = v['audio_file'].replace(base_dir + '/', '')
#     v['midi_file'] = v['midi_file'].replace(base_dir + '/', '')

# Write a new csv file
# output_csv_file = os.path.join(base_dir, 'geerdes_data_final.csv')
# with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     headers = ['id', 'split_half', 'title', 'artist', 'audio_file', 'midi_file', 'link']
#     writer.writerow(headers)

#     for id, info in tracks_all.items():
#         row = [
#             id, info['split_half'], info['title'], info['artist'], info['audio_file'],
#             info['midi_file'], info['link']
#         ]
#         writer.writerow(row)
#     logger.info(f'Wrote {len(tracks_all)} rows to {output_csv_file}.')
#     logger.info(f'Finished creating split and basic info file.')


def create_note_event_and_note_from_midi(mid_file: str,
                                         id: str,
                                         ch_9_as_drum: bool = True,
                                         track_name_to_program: Dict = None,
                                         ignore_pedal: bool = False) -> Tuple[Dict, Dict]:
    """Create note_events and notes from midi file."""

    # Load midi file
    notes, dur_sec, program = midi2note(
        mid_file,
        ch_9_as_drum=ch_9_as_drum,
        track_name_to_program=track_name_to_program,
        binary_velocity=True,
        ignore_pedal=ignore_pedal,
        return_programs=True)
    program = [x for x in set(program) if x is not None]  # remove None and duplicates
    return { # notes
        'geerdes_id': id,
        'program': program,
        'is_drum': [1 if p == DRUM_PROGRAM else 0 for p in program],
        'duration_sec': dur_sec,
        'notes': notes,
    }, { # note_events
        'geerdes_id': id,
        'program': program,
        'is_drum': [1 if p == DRUM_PROGRAM else 0 for p in program],
        'duration_sec': dur_sec,
        'note_events': note2note_event(notes),
    }


def preprocess_geerdes16k(data_home=os.PathLike,
                          dataset_name='geerdes',
                          sanity_check=False) -> None:
    """Preprocess Geerdes dataset."""
    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # Setup logger
    log_file = os.path.join(base_dir, 'log.txt')
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

    # Load CSV: construct id to midi/wav dictionary
    ymt3_geerdes_csv_file = os.path.join(base_dir, GEERDES_DATA_CSV_FILENAME)
    tracks_all = {}
    with open(ymt3_geerdes_csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            geerdes_id = row['id']
            tracks_all[geerdes_id] = row
    # append base_dir to audio_file and midi_file
    for v in tracks_all.values():
        v['audio_file'] = os.path.join(base_dir, v['audio_file'])
        v['midi_file'] = os.path.join(base_dir, v['midi_file'])
    logger.info(f'Loaded {len(tracks_all)} tracks from {ymt3_geerdes_csv_file}.')

    # Process midi files
    note_processed_dir = os.path.join(base_dir, 'note_processed')
    os.makedirs(note_processed_dir, exist_ok=True)

    for geerdes_id, v in tracks_all.items():
        midi_file = v['midi_file']

        # create notes and note_events
        notes, note_events = create_note_event_and_note_from_midi(
            mid_file=midi_file,
            id=geerdes_id,
            ch_9_as_drum=True,
            track_name_to_program=TRACK_NAME_TO_PROGRAM_MAP,
            ignore_pedal=False)

        # sanity check
        if sanity_check is True:
            err_cnt = note_event2token2note_event_sanity_check(note_events['note_events'],
                                                               notes['notes'])
            if len(err_cnt) > 0:
                logging.warning(f'Found {err_cnt} errors in {geerdes_id}.')

        # save notes and note_events
        notes_file = os.path.join(note_processed_dir, geerdes_id + '_notes.npy')
        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        logger.info(f'Created {notes_file}.')

        note_events_file = os.path.join(note_processed_dir, geerdes_id + '_note_events.npy')
        np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
        logger.info(f'Created {note_events_file}.')

        # save reconstructed midi file
        recon_midi_file = os.path.join(note_processed_dir, geerdes_id + '_recon.mid')
        inverse_vocab = create_inverse_vocab(MT3_FULL_PLUS)
        note_event2midi(
            note_events['note_events'], recon_midi_file, output_inverse_vocab=inverse_vocab)
        logger.info(f'Created {recon_midi_file}.')

        # add file paths and info to tracks_all
        tracks_all[geerdes_id]['notes_file'] = notes_file
        tracks_all[geerdes_id]['note_events_file'] = note_events_file
        tracks_all[geerdes_id]['recon_midi_file'] = recon_midi_file
        tracks_all[geerdes_id]['program'] = notes['program']
        tracks_all[geerdes_id]['is_drum'] = notes['is_drum']

        # save extract main_vocal/vocal_and_chorus/accompaniment only notes and note_events
        notes_voc = deepcopy(notes)
        notes_voc['notes'] = extract_notes_selected_by_programs(
            notes['notes'], [SINGING_VOICE_PROGRAM, SINGING_VOICE_CHORUS_PROGRAM])
        notes_voc['program'] = list(extract_program_from_notes(notes_voc['notes']))
        notes_voc['is_drum'] = [1 if p == DRUM_PROGRAM else 0 for p in notes_voc['program']]
        notes_voc_file = os.path.join(note_processed_dir, geerdes_id + '_notes_voc.npy')
        np.save(notes_voc_file, notes_voc, allow_pickle=True, fix_imports=False)

        note_events_voc = deepcopy(note_events)
        note_events_voc['note_events'] = note2note_event(notes_voc['notes'])
        note_events_voc['program'] = deepcopy(notes_voc['program'])
        note_events_voc['is_drum'] = deepcopy(notes_voc['is_drum'])
        note_events_voc_file = os.path.join(note_processed_dir, geerdes_id + '_note_events_voc.npy')
        np.save(note_events_voc_file, note_events_voc, allow_pickle=True, fix_imports=False)

        notes_acc = deepcopy(notes)
        notes_acc['notes'] = extract_notes_selected_by_programs(notes['notes'], [
            p for p in notes['program']
            if p not in [SINGING_VOICE_PROGRAM, SINGING_VOICE_CHORUS_PROGRAM]
        ])
        notes_acc['program'] = list(extract_program_from_notes(notes_acc['notes']))
        notes_acc['is_drum'] = [1 if p == DRUM_PROGRAM else 0 for p in notes_acc['program']]
        notes_acc_file = os.path.join(note_processed_dir, geerdes_id + '_notes_acc.npy')
        np.save(notes_acc_file, notes_acc, allow_pickle=True, fix_imports=False)

        note_events_acc = deepcopy(note_events)
        note_events_acc['note_events'] = note2note_event(notes_acc['notes'])
        note_events_acc['program'] = deepcopy(notes_acc['program'])
        note_events_acc['is_drum'] = deepcopy(notes_acc['is_drum'])
        note_events_acc_file = os.path.join(note_processed_dir, geerdes_id + '_note_events_acc.npy')
        np.save(note_events_acc_file, note_events_acc, allow_pickle=True, fix_imports=False)

        tracks_all[geerdes_id]['notes_file_voc'] = notes_voc_file
        tracks_all[geerdes_id]['note_events_file_voc'] = note_events_voc_file
        tracks_all[geerdes_id]['program_voc'] = notes_voc['program']
        tracks_all[geerdes_id]['is_drum_voc'] = notes_voc['is_drum']
        tracks_all[geerdes_id]['notes_file_acc'] = notes_acc_file
        tracks_all[geerdes_id]['note_events_file_acc'] = note_events_acc_file
        tracks_all[geerdes_id]['program_acc'] = notes_acc['program']
        tracks_all[geerdes_id]['is_drum_acc'] = notes_acc['is_drum']

    # Process or check audio files
    for geerdes_id, v in tracks_all.items():
        v['mix_audio_file'] = v['audio_file']
        v['mix_audio_file_voc'] = v['audio_file'].replace('.wav', '_vocals.wav')
        v['mix_audio_file_acc'] = v['audio_file'].replace('.wav', '_accompaniment.wav')
        assert os.path.exists(v['mix_audio_file'])
        assert os.path.exists(v['mix_audio_file_voc'])
        assert os.path.exists(v['mix_audio_file_acc'])
        v['n_frames'] = get_audio_file_info(v['mix_audio_file'])[1]
    logger.info(f'Checked audio files. All audio files exist.')

    # Create file_list.json
    splits = ['train', 'validation', 'all']
    task_suffixes = ['', '_sep']

    for task_suffix in task_suffixes:
        for split in splits:
            # NOTE: We use spleeter files as the mix audio files, since partial stems (for accomp.) are not implemented yet
            file_list = {}
            cur_idx = 0
            for geerdes_id, v in tracks_all.items():
                if v['split_half'] == split or split == 'all':
                    if task_suffix == '':
                        file_list[cur_idx] = {
                            'geerdes_id': geerdes_id,
                            'n_frames': v['n_frames'],
                            'mix_audio_file': v['mix_audio_file'],
                            'notes_file': v['notes_file'],
                            'note_events_file': v['note_events_file'],
                            'midi_file': v['midi_file'],
                            'program': v['program'],
                            'is_drum': v['is_drum'],
                        }
                        cur_idx += 1
                    elif task_suffix == '_sep':
                        file_list[cur_idx] = {
                            'geerdes_id': geerdes_id,
                            'n_frames': v['n_frames'],
                            'mix_audio_file': v['mix_audio_file_voc'],
                            'notes_file': v['notes_file_voc'],
                            'note_events_file': v['note_events_file_voc'],
                            'midi_file': v['midi_file'],
                            'program': v['program_voc'],
                            'is_drum': v['is_drum_voc'],
                        }
                        cur_idx += 1
                        file_list[cur_idx] = {
                            'geerdes_id': geerdes_id,
                            'n_frames': v['n_frames'],
                            'mix_audio_file': v['mix_audio_file_acc'],
                            'notes_file': v['notes_file_acc'],
                            'note_events_file': v['note_events_file_acc'],
                            'midi_file': v['midi_file'],
                            'program': v['program_acc'],
                            'is_drum': v['is_drum_acc'],
                        }
                        cur_idx += 1

            file_list_file = os.path.join(output_index_dir,
                                          f'{dataset_name}_{split}{task_suffix}_file_list.json')
            with open(file_list_file, 'w') as f:
                json.dump(file_list, f, indent=4)
            logger.info(f'Created {file_list_file}.')