"""preprocess_egmd.py"""
import os
import csv
import glob
import re
import json
from typing import Dict, List, Tuple
import numpy as np
from utils.audio import get_audio_file_info
from utils.midi import midi2note, note_event2midi
from utils.note2event import note2note_event, note_event2event
from utils.event2note import event2note_event
from utils.note_event_dataclasses import Note, NoteEvent
from utils.utils import note_event2token2note_event_sanity_check
# from utils.utils import assert_note_events_almost_equal


def create_note_event_and_note_from_midi(mid_file: str, id: str) -> Tuple[Dict, Dict]:
    """Extracts note or note_event and metadata from midi:

    Returns:
        notes (dict): note events and metadata.
        note_events (dict): note events and metadata.
    """
    notes, dur_sec = midi2note(
        mid_file,
        binary_velocity=True,
        ch_9_as_drum=True,
        force_all_drum=True,
        trim_overlap=True,
        fix_offset=True,
        quantize=True,
        verbose=0,
        minimum_offset_sec=0.01,
        drum_offset_sec=0.01,
        ignore_pedal=True)
    return {  # notes
        'egmd_id': id,
        'program': [128],
        'is_drum': [1],
        'duration_sec': dur_sec,
        'notes': notes,
    }, {  # note_events
        'maps_id': id,
        'program': [128],
        'is_drum': [1],
        'duration_sec': dur_sec,
        'note_events': note2note_event(notes),
    }


def preprocess_egmd16k(data_home: os.PathLike, dataset_name='egmd') -> None:
    """
    Splits:
        - train: 35217 files
        - validation: 5031 files
        - test: 5289 files
        - test_reduced: 246 files that contain '_5.midi' or '_10.midi' in the filename


    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
        {
            index:
            {
                'egmd_id': egmd_id, # filename wihout extension
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

    # Load csv file and create a dictionary
    csv_file = os.path.join(base_dir, 'e-gmd-v1.0.0.csv')
    with open(csv_file, 'r') as f:
        csv_dict_reader = csv.DictReader(f)
        egmd_dict_list_all = list(csv_dict_reader)
    assert len(egmd_dict_list_all) == 45537

    # Process MIDI files
    for d in egmd_dict_list_all:
        emgd_id = d['midi_filename'].split('.')[0]
        midi_file = os.path.join(base_dir, d['midi_filename'])
        notes, note_events = create_note_event_and_note_from_midi(midi_file, emgd_id)

        # Write notes and note_events
        notes_file = midi_file.replace('.midi', '_notes.npy')
        note_events_file = midi_file.replace('.midi', '_note_events.npy')
        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        print(f"Created {notes_file}")
        np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
        print(f"Created {note_events_file}")

        # rewrite 120 bpm quantized midi file
        quantized_midi_file = midi_file.replace('.midi', '_quantized_120bpm.mid')
        note_event2midi(note_events['note_events'], quantized_midi_file)
        print(f'Wrote {quantized_midi_file}')

    # Process audio files
    pass

    # Create index files
    for split in ['train', 'validation', 'test']:
        file_list = {}
        i = 0
        for d in egmd_dict_list_all:
            if d['split'] == split:
                egmd_id = d['midi_filename'].split('.')[0]
                mix_audio_file = os.path.join(base_dir, d['audio_filename'])
                n_frames = get_audio_file_info(mix_audio_file)[1]
                midi_file = os.path.join(base_dir, d['midi_filename'])
                notes_file = midi_file.replace('.midi', '_notes.npy')
                note_events_file = midi_file.replace('.midi', '_note_events.npy')

                # check file existence
                assert os.path.exists(mix_audio_file)
                assert os.path.exists(midi_file)
                assert os.path.exists(notes_file)
                assert os.path.exists(note_events_file)

                # create file list
                file_list[i] = {
                    'egmd_id': egmd_id,
                    'n_frames': n_frames,
                    'mix_audio_file': mix_audio_file,
                    'notes_file': notes_file,
                    'note_events_file': note_events_file,
                    'midi_file': midi_file,
                    'program': [128],
                    'is_drum': [1],
                }
                i += 1
            else:
                pass

        # Write file list
        output_file = os.path.join(output_index_dir, f'{dataset_name}_{split}_file_list.json')
        with open(output_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Wrote {output_file}')
        if split == 'train':
            assert len(file_list) == 35217
        elif split == 'validation':
            assert len(file_list) == 5031
        elif split == 'test':
            assert len(file_list) == 5289

    # Create reduced test index file
    split = 'test_reduced'
    file_list = {}
    i = 0
    for d in egmd_dict_list_all:
        if d['split'] == 'test':
            midi_file = os.path.join(base_dir, d['midi_filename'])
            if '_5.midi' in midi_file or '_10.midi' in midi_file:
                egmd_id = d['midi_filename'].split('.')[0]
                mix_audio_file = os.path.join(base_dir, d['audio_filename'])
                n_frames = get_audio_file_info(mix_audio_file)[1]
                notes_file = midi_file.replace('.midi', '_notes.npy')
                note_events_file = midi_file.replace('.midi', '_note_events.npy')
                file_list[i] = {
                    'egmd_id': egmd_id,
                    'n_frames': n_frames,
                    'mix_audio_file': mix_audio_file,
                    'notes_file': notes_file,
                    'note_events_file': note_events_file,
                    'midi_file': midi_file,
                    'program': [128],
                    'is_drum': [1],
                }
                i += 1
    output_file = os.path.join(output_index_dir, f'{dataset_name}_{split}_file_list.json')
    with open(output_file, 'w') as f:
        json.dump(file_list, f, indent=4)
    print(f'Wrote {output_file}')
    assert len(file_list) == 246
