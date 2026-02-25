"""preprocess_musicnet.py"""
import os
import glob
import csv
import json
from typing import Dict, List, Tuple
import numpy as np
from utils.audio import get_audio_file_info
from utils.midi import midi2note
from utils.note2event import note2note_event
from utils.note_event_dataclasses import Note

# yapf: disable
MUSICNET_SPLIT_INFO = {
    'train_mt3': [], # the first 300 songs are synth dataset, while the remaining 300 songs are acoustic dataset. 
    'train_mt3_synth' : [], # Note: this is not the synthetic dataset of EM (MIDI Pop 80K) nor pitch-augmented. Just recording of MusicNet MIDI, split by MT3 author's split. But not sure if they used this (maybe not).
    'train_mt3_acoustic': [],
    'validation_mt3': [1733, 1765, 1790, 1818, 2160, 2198, 2289, 2300, 2308, 2315, 2336, 2466, 2477, 2504, 2611],
    'validation_mt3_synth': [1733, 1765, 1790, 1818, 2160, 2198, 2289, 2300, 2308, 2315, 2336, 2466, 2477, 2504, 2611],
    'validation_mt3_acoustic': [1733, 1765, 1790, 1818, 2160, 2198, 2289, 2300, 2308, 2315, 2336, 2466, 2477, 2504, 2611],
    'test_mt3_acoustic': [1729, 1776, 1813, 1893, 2118, 2186, 2296, 2431, 2432, 2487, 2497, 2501, 2507, 2537, 2621],
    'train_thickstun': [], # the first 320 songs are synth dataset, while the remaining 320 songs are acoustic dataset.  
    'test_thickstun': [1819, 2303, 2382],
    'test_thickstun_em': [1819, 2303, 2382],
    'test_thickstun_ext': [1759, 1819, 2106, 2191, 2298, 2303, 2382, 2416, 2556, 2628],
    'test_thickstun_ext_em': [1759, 1819, 2106, 2191, 2298, 2303, 2382, 2416, 2556, 2628],
    'train_mt3_em': [], # 300 synth + 293 tracks for MT3 acoustic train set - 7 EM tracks are missing: [2194, 2211, 2227, 2230, 2292, 2305, 2310].
    'train_thickstun_em': [], # 320 synth + 313 tracks for Thickstun acoustic train set - 7 EM tracks are missing.
    'validation_mt3_em': [1733, 1765, 1790, 1818, 2160, 2198, 2289, 2300, 2308, 2315, 2336, 2466, 2477, 2504, 2611], # ours
    'test_mt3_em': [1729, 1776, 1813, 1893, 2118, 2186, 2296, 2431, 2432, 2487, 2497, 2501, 2507, 2537, 2621], # ours
    'test_em_table2' : [2191, 2628, 2106, 2298, 1819, 2416], # strings and winds from Cheuk's split, using EM annotations
    'test_cheuk_table2' : [2191, 2628, 2106, 2298, 1819, 2416], # strings and winds from Cheuk's split, using Thickstun's annotations
    'test_thickstun_ext_em': [1759, 1819, 2106, 2191, 2298, 2303, 2382, 2416, 2556, 2628],
}
# Table 4 of EM is not included here.

# yapf: enable
MUSICNET_DISCARD_INFO = ['test_labels_midi/1759.mid',
                         'test_labels_midi/1819.mid']  # duplicated midi files
MUSICNET_EM_MISSING_IDS = set(['2194', '2211', '2227', '2230', '2292', '2305', '2310'])

MUSICNET_FS = 44100


def create_note_event_and_note_from_label(label_file: str, id: str):
    """Extracts note or note_event and metadata from a label file.

    Returns:
        notes (dict): note events and metadata.
        note_events (dict): note events and metadata.
    """
    program_numbers = set()
    notes = []
    with open(label_file, 'r', newline='', encoding='utf-8') as c:
        csv_reader = csv.reader(c)
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            start_frame, end_frame, program, pitch, _, _, _ = row
            new_note = Note(
                is_drum=False,
                program=int(program),
                onset=float(start_frame) / MUSICNET_FS,
                offset=float(end_frame) / MUSICNET_FS,
                pitch=int(pitch),
                velocity=1)
            notes.append(new_note)
            program_numbers.add(int(program))
    program_numbers = list(program_numbers)

    return { # notes
        'musicnet_id': id,
        'program': program_numbers,
        'is_drum': [0]*len(program_numbers),
        'duration_sec': notes[0].offset,
        'notes': notes,
    }, { # note_events
        'musicnet_id': id,
        'program': program_numbers,
        'is_drum': [0]*len(program_numbers),
        'duration_sec': notes[0].offset,
        'note_events': note2note_event(notes),
    }


def create_note_event_and_note_from_midi(mid_file: str, id: str) -> Tuple[Dict, Dict]:
    """Extracts note or note_event and metadata from midi:

    Returns:
        notes (dict): note events and metadata.
        note_events (dict): note events and metadata.
    """
    notes, dur_sec = midi2note(
        mid_file,
        binary_velocity=True,
        ch_9_as_drum=False,
        force_all_drum=False,
        force_all_program_to=None,
        trim_overlap=True,
        fix_offset=True,
        quantize=True,
        verbose=0,
        minimum_offset_sec=0.01,
        drum_offset_sec=0.01)
    return {  # notes
        'musicnet_id': id,
        'program': [],
        'is_drum': [],
        'duration_sec': dur_sec,
        'notes': notes,
    }, {  # note_events
        'musicnet_id': id,
        'program': [],
        'is_drum': [],
        'duration_sec': dur_sec,
        'note_events': note2note_event(notes),
    }


def preprocess_musicnet16k(data_home=os.PathLike, dataset_name='musicnet') -> None:
    """
    
    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
        {
            index:
            {
                'musicnet_id': musicnet_id,
                'n_frames': (int),
                'mix_audio_file': 'path/to/mix.wav',
                'notes_file': 'path/to/notes.npy',
                'note_events_file': 'path/to/note_events.npy',
                'midi_file': 'path/to/midi.mid',
                'program': List[int],
                'is_drum': List[int], # 0 or 1
            }
        }
    """

    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # Search for files with .mid and .wav (synth / acoustic) extensions
    label_pattern = os.path.join(base_dir, '*_labels', '*.csv')
    mid_em_pattern = os.path.join(base_dir, '*_em',
                                  '*.mid')  # EM annotations for real performances (wav)
    mid_pattern = os.path.join(base_dir, '*_midi', '*.mid')
    wav_synth_pattern = os.path.join(base_dir, '*_synth', '*.wav')
    wav_acoustic_pattern = os.path.join(base_dir, '*_data', '*.wav')

    label_files = glob.glob(label_pattern, recursive=True)
    mid_em_files = glob.glob(mid_em_pattern, recursive=True)  # 323 files, not 330!
    mid_files = glob.glob(mid_pattern, recursive=True)
    wav_synth_files = glob.glob(wav_synth_pattern, recursive=True)
    wav_acoustic_files = glob.glob(wav_acoustic_pattern, recursive=True)

    # Discard duplicated files
    for file in MUSICNET_DISCARD_INFO:
        mid_files.remove(os.path.join(base_dir, file))
    assert (len(mid_files) == len(label_files) == len(wav_synth_files) == len(wav_acoustic_files) ==
            330)

    # Sort files by id
    musicnet_ids = []
    for label_file in label_files:
        musicnet_ids.append(os.path.basename(label_file).split('.')[0])
    musicnet_ids.sort()
    assert (len(musicnet_ids) == 330)

    musicnet_em_ids = []
    for mid_em_file in mid_em_files:
        musicnet_em_ids.append(os.path.basename(mid_em_file).split('.')[0])
    assert (len(musicnet_em_ids) == 323)

    def search_file_by_musicnet_id(musicnet_id, files):
        file_found = [f for f in files if musicnet_id in f
                     ]  # this only works in 4-digits file names of MusicNet
        assert (len(file_found) == 1)
        return file_found[0]

    # yapf: disable
    musicnet_dict = {}
    for i in musicnet_ids:
        musicnet_dict[i] = {
            'wav_acoustic_file': search_file_by_musicnet_id(i, wav_acoustic_files),
            'wav_synth_file': search_file_by_musicnet_id(i, wav_synth_files),
            'mid_file': search_file_by_musicnet_id(i, mid_files),
            'mid_em_file': search_file_by_musicnet_id(i, mid_em_files) if i in musicnet_em_ids else None,
            'label_file': search_file_by_musicnet_id(i, label_files),
            'program': [],
            'is_drum': [],
            'duration_sec': 0.,
            'notes_file_acoustic': '',
            'note_events_file_acoustic': '',
            'notes_file_synth': '',
            'note_events_file_synth': '',
            'notes_file_em': '',
            'note_events_file_em': '',
        }
    # yapf: enable

    # Process label files
    for i in musicnet_ids:
        notes, note_events = create_note_event_and_note_from_label(
            label_file=musicnet_dict[i]['label_file'], id=i)

        notes_file = os.path.join(musicnet_dict[i]['label_file'][:-4] + '_notes.npy')
        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        print(f'Created {notes_file}')

        note_events_file = os.path.join(musicnet_dict[i]['label_file'][:-4] + '_note_events.npy')
        np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
        print(f'Created {note_events_file}')

        # update musicnet_dict
        musicnet_dict[i]['program'] = notes['program']
        musicnet_dict[i]['is_drum'] = notes['is_drum']
        musicnet_dict[i]['duration_sec'] = notes['duration_sec']
        musicnet_dict[i]['notes_file_acoustic'] = notes_file
        musicnet_dict[i]['note_events_file_acoustic'] = note_events_file

    # Process MIDI files
    for i in musicnet_ids:
        # musicnet
        notes, note_events = create_note_event_and_note_from_midi(
            mid_file=musicnet_dict[i]['mid_file'], id=i)
        notes['program'] = musicnet_dict[i]['program'].copy()
        notes['is_drum'] = musicnet_dict[i]['is_drum'].copy()
        notes_file = os.path.join(musicnet_dict[i]['mid_file'][:-4] + '_notes.npy')
        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        print(f'Created {notes_file}')

        note_events['program'] = musicnet_dict[i]['program'].copy()
        note_events['is_drum'] = musicnet_dict[i]['is_drum'].copy()
        note_events_file = os.path.join(musicnet_dict[i]['mid_file'][:-4] + '_note_events.npy')
        np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
        print(f'Created {note_events_file}')

        # update musicnet_dict
        musicnet_dict[i]['duration_sec'] = max(notes['duration_sec'],
                                               musicnet_dict[i]['duration_sec'])
        musicnet_dict[i]['notes_file_synth'] = notes_file
        musicnet_dict[i]['note_events_file_synth'] = note_events_file

        # musicnet_em
        if i in musicnet_em_ids:
            notes, note_events = create_note_event_and_note_from_midi(
                mid_file=musicnet_dict[i]['mid_em_file'], id=i)
            notes['program'] = musicnet_dict[i]['program'].copy()
            notes['is_drum'] = musicnet_dict[i]['is_drum'].copy()
            notes_file = os.path.join(musicnet_dict[i]['mid_em_file'][:-4] + '_notes.npy')
            np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
            print(f'Created {notes_file}')

            note_events['program'] = musicnet_dict[i]['program'].copy()
            note_events['is_drum'] = musicnet_dict[i]['is_drum'].copy()
            note_events_file = os.path.join(musicnet_dict[i]['mid_em_file'][:-4] +
                                            '_note_events.npy')
            np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
            print(f'Created {note_events_file}')

            # update musicnet_dict: use the longest duration
            musicnet_dict[i]['duration_sec'] = max(notes['duration_sec'],
                                                   musicnet_dict[i]['duration_sec'])
            musicnet_dict[i]['notes_file_em'] = notes_file
            musicnet_dict[i]['note_events_file_em'] = note_events_file

    # Process audio files
    pass

    # Complete split dictionary
    split_dict = MUSICNET_SPLIT_INFO.copy()

    # Convert each list in the dictionary to a list of strings
    for key in split_dict:
        split_dict[key] = [str(item) for item in split_dict[key]]

    # Convert each list to a sorted tuple of strings to preserve the original order
    for key in split_dict:
        split_dict[key] = tuple(sorted(split_dict[key]))

    # Create sets and subtract sets to create new sets
    whole_set = set(musicnet_ids)
    split_dict['train_mt3'] = whole_set - set(split_dict['validation_mt3']) - set(
        split_dict['test_mt3_acoustic'])
    split_dict['train_mt3_synth'] = split_dict['train_mt3']
    split_dict['train_mt3_acoustic'] = split_dict['train_mt3']
    split_dict['train_thickstun'] = whole_set - set(split_dict['test_thickstun_ext'])
    split_dict['train_thickstun_synth'] = split_dict['train_thickstun']
    split_dict['train_mt3_em'] = whole_set - set(split_dict['validation_mt3']) - set(
        split_dict['test_mt3_acoustic']) - MUSICNET_EM_MISSING_IDS
    split_dict['train_thickstun_em'] = whole_set - set(
        split_dict['test_thickstun_ext']) - MUSICNET_EM_MISSING_IDS
    # Convert each tuple back to a list of strings
    for key in split_dict:
        split_dict[key] = [str(item) for item in split_dict[key]]

    # Write MT3 file_list
    for split in ('train_mt3_synth', 'validation_mt3_synth'):
        file_list = {}
        for i, musicnet_id in enumerate(split_dict[split]):
            file_list[i] = {
                'musicnet_id': musicnet_id,
                'n_frames': get_audio_file_info(musicnet_dict[musicnet_id]['wav_synth_file'])[1],
                'mix_audio_file': musicnet_dict[musicnet_id]['wav_synth_file'],
                'notes_file': musicnet_dict[musicnet_id]['notes_file_synth'],
                'note_events_file': musicnet_dict[musicnet_id]['note_events_file_synth'],
                'midi_file': musicnet_dict[musicnet_id]['mid_file'],
                'program': musicnet_dict[musicnet_id]['program'],
                'is_drum': musicnet_dict[musicnet_id]['is_drum'],
            }
        assert (len(file_list) == len(split_dict[split]))
        output_index_file = os.path.join(output_index_dir, f'musicnet_{split}_file_list.json')
        with open(output_index_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Created {output_index_file}')

    for split in ('train_mt3_acoustic', 'validation_mt3_acoustic', 'test_mt3_acoustic'):
        file_list = {}
        for i, musicnet_id in enumerate(split_dict[split]):
            file_list[i] = {
                'musicnet_id': musicnet_id,
                'n_frames': get_audio_file_info(musicnet_dict[musicnet_id]['wav_acoustic_file'])[1],
                'mix_audio_file': musicnet_dict[musicnet_id]['wav_acoustic_file'],
                'notes_file': musicnet_dict[musicnet_id]['notes_file_acoustic'],
                'note_events_file': musicnet_dict[musicnet_id]['note_events_file_acoustic'],
                'midi_file': musicnet_dict[musicnet_id]['mid_file'],
                'program': musicnet_dict[musicnet_id]['program'],
                'is_drum': musicnet_dict[musicnet_id]['is_drum'],
            }
        assert (len(file_list) == len(split_dict[split]))
        output_index_file = os.path.join(output_index_dir, f'musicnet_{split}_file_list.json')
        with open(output_index_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Created {output_index_file}')

    split = 'train_mt3'
    merged_file_list = {}
    index = 0
    file_list_train_mt3_synth = json.load(
        open(os.path.join(output_index_dir, 'musicnet_train_mt3_synth_file_list.json')))
    file_list_train_mt3_acoustic = json.load(
        open(os.path.join(output_index_dir, 'musicnet_train_mt3_acoustic_file_list.json')))
    for d in [file_list_train_mt3_synth, file_list_train_mt3_acoustic]:
        for key, value in d.items():
            new_key = f'{index}'
            merged_file_list[new_key] = value
            index += 1
    assert (len(merged_file_list) == 600)
    output_index_file = os.path.join(output_index_dir, f'musicnet_{split}_file_list.json')
    with open(output_index_file, 'w') as f:
        json.dump(merged_file_list, f, indent=4)
    print(f'Created {output_index_file}')

    # Write ThickStun file_list
    split = 'train_thickstun'
    file_list = {}
    for i, musicnet_id in enumerate(split_dict[split]):
        file_list[i] = {
            'musicnet_id': musicnet_id,
            'n_frames': get_audio_file_info(musicnet_dict[musicnet_id]['wav_synth_file'])[1],
            'mix_audio_file': musicnet_dict[musicnet_id]['wav_synth_file'],
            'notes_file': musicnet_dict[musicnet_id]['notes_file_synth'],
            'note_events_file': musicnet_dict[musicnet_id]['note_events_file_synth'],
            'midi_file': musicnet_dict[musicnet_id]['mid_file'],
            'program': musicnet_dict[musicnet_id]['program'],
            'is_drum': musicnet_dict[musicnet_id]['is_drum'],
        }
        file_list[i + 327] = {
            'musicnet_id': musicnet_id,
            'n_frames': get_audio_file_info(musicnet_dict[musicnet_id]['wav_acoustic_file'])[1],
            'mix_audio_file': musicnet_dict[musicnet_id]['wav_acoustic_file'],
            'notes_file': musicnet_dict[musicnet_id]['notes_file_acoustic'],
            'note_events_file': musicnet_dict[musicnet_id]['note_events_file_acoustic'],
            'midi_file': musicnet_dict[musicnet_id]['mid_file'],
            'program': musicnet_dict[musicnet_id]['program'],
            'is_drum': musicnet_dict[musicnet_id]['is_drum'],
        }
    assert (len(file_list) == len(split_dict[split]) * 2)
    output_index_file = os.path.join(output_index_dir, f'musicnet_{split}_file_list.json')
    with open(output_index_file, 'w') as f:
        json.dump(file_list, f, indent=4)
    print(f'Created {output_index_file}')

    for split in ('test_thickstun', 'test_thickstun_ext'):
        file_list = {}
        for i, musicnet_id in enumerate(split_dict[split]):
            file_list[i] = {
                'musicnet_id': musicnet_id,
                'n_frames': get_audio_file_info(musicnet_dict[musicnet_id]['wav_acoustic_file'])[1],
                'mix_audio_file': musicnet_dict[musicnet_id]['wav_acoustic_file'],
                'notes_file': musicnet_dict[musicnet_id]['notes_file_acoustic'],
                'note_events_file': musicnet_dict[musicnet_id]['note_events_file_acoustic'],
                'midi_file': musicnet_dict[musicnet_id]['mid_file'],
                'program': musicnet_dict[musicnet_id]['program'],
                'is_drum': musicnet_dict[musicnet_id]['is_drum'],
            }
        assert (len(file_list) == len(split_dict[split]))
        output_index_file = os.path.join(output_index_dir, f'musicnet_{split}_file_list.json')
        with open(output_index_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Created {output_index_file}')

    # Write EM file_list
    for split in ('train_thickstun_em', 'train_mt3_em'):
        file_list = {}
        for i, musicnet_id in enumerate(split_dict[split]):
            file_list[i] = {
                'musicnet_id': musicnet_id,
                'n_frames': get_audio_file_info(musicnet_dict[musicnet_id]['wav_acoustic_file'])[1],
                'mix_audio_file': musicnet_dict[musicnet_id]['wav_acoustic_file'],
                'notes_file': musicnet_dict[musicnet_id]['notes_file_em'],
                'note_events_file': musicnet_dict[musicnet_id]['note_events_file_em'],
                'midi_file': musicnet_dict[musicnet_id]['mid_em_file'],
                'program': musicnet_dict[musicnet_id]['program'],
                'is_drum': musicnet_dict[musicnet_id]['is_drum'],
            }
        synth_ids = split_dict['train_mt3'] if split == 'train_mt3_em' else split_dict[
            'train_thickstun']
        for i, musicnet_id in enumerate(synth_ids):
            file_list[i + len(split_dict[split])] = {
                'musicnet_id': musicnet_id,
                'n_frames': get_audio_file_info(musicnet_dict[musicnet_id]['wav_synth_file'])[1],
                'mix_audio_file': musicnet_dict[musicnet_id]['wav_synth_file'],
                'notes_file': musicnet_dict[musicnet_id]['notes_file_synth'],
                'note_events_file': musicnet_dict[musicnet_id]['note_events_file_synth'],
                'midi_file': musicnet_dict[musicnet_id]['mid_file'],
                'program': musicnet_dict[musicnet_id]['program'],
                'is_drum': musicnet_dict[musicnet_id]['is_drum'],
            }
        if split == 'train_thickstun_em':
            assert (len(file_list) == 320 + 313)
        if split == 'train_mt3_em':
            assert (len(file_list) == 300 + 293)
        output_index_file = os.path.join(output_index_dir, f'musicnet_{split}_file_list.json')
        with open(output_index_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Created {output_index_file}')

    for split in ('validation_mt3_em', 'test_mt3_em', 'test_em_table2', 'test_thickstun_em',
                  'test_thickstun_ext_em'):
        file_list = {}
        for i, musicnet_id in enumerate(split_dict[split]):
            file_list[i] = {
                'musicnet_id': musicnet_id,
                'n_frames': get_audio_file_info(musicnet_dict[musicnet_id]['wav_acoustic_file'])[1],
                'mix_audio_file': musicnet_dict[musicnet_id]['wav_acoustic_file'],
                'notes_file': musicnet_dict[musicnet_id]['notes_file_em'],
                'note_events_file': musicnet_dict[musicnet_id]['note_events_file_em'],
                'midi_file': musicnet_dict[musicnet_id]['mid_em_file'],
                'program': musicnet_dict[musicnet_id]['program'],
                'is_drum': musicnet_dict[musicnet_id]['is_drum'],
            }
        assert (len(file_list) == len(split_dict[split]))
        output_index_file = os.path.join(output_index_dir, f'musicnet_{split}_file_list.json')
        with open(output_index_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Created {output_index_file}')

    # Write Cheuk file_list
    for split in ['test_cheuk_table2']:
        file_list = {}
        for i, musicnet_id in enumerate(split_dict[split]):
            file_list[i] = {
                'musicnet_id': musicnet_id,
                'n_frames': get_audio_file_info(musicnet_dict[musicnet_id]['wav_acoustic_file'])[1],
                'mix_audio_file': musicnet_dict[musicnet_id]['wav_acoustic_file'],
                'notes_file': musicnet_dict[musicnet_id]['notes_file_acoustic'],
                'note_events_file': musicnet_dict[musicnet_id]['note_events_file_acoustic'],
                'midi_file': musicnet_dict[musicnet_id]['mid_file'],
                'program': musicnet_dict[musicnet_id]['program'],
                'is_drum': musicnet_dict[musicnet_id]['is_drum'],
            }
        assert (len(file_list) == len(split_dict[split]))
        output_index_file = os.path.join(output_index_dir, f'musicnet_{split}_file_list.json')
        with open(output_index_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Created {output_index_file}')


if __name__ == '__main__':
    from config.config import shared_cfg
    data_home = shared_cfg['PATH']['data_home']
    preprocess_musicnet16k(data_home=data_home, dataset_name='musicnet')