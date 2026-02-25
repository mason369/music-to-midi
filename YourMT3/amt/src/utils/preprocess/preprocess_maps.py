"""preprocess_maps.py"""
import os
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


def create_note_event_and_note_from_midi(mid_file: str,
                                         id: str,
                                         ignore_pedal: bool = False) -> Tuple[Dict, Dict]:
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
        force_all_program_to=0,  # always piano
        trim_overlap=True,
        fix_offset=True,
        quantize=True,
        verbose=0,
        minimum_offset_sec=0.01,
        drum_offset_sec=0.01,
        ignore_pedal=ignore_pedal)
    return {  # notes
        'maps_id': id,
        'program': [0],
        'is_drum': [0],
        'duration_sec': dur_sec,
        'notes': notes,
    }, {  # note_events
        'maps_id': id,
        'program': [0],
        'is_drum': [0],
        'duration_sec': dur_sec,
        'note_events': note2note_event(notes),
    }


def rewrite_midi_120bpm(file: os.PathLike, note_events: List[NoteEvent]):
    """Rewrite midi file with 120 bpm."""
    note_event2midi(note_events, file)
    return


# def note_event2event_sanity_check(note_events: List[NoteEvent]):
#     """Sanity check for note events."""
#     events = note_event2event(note_events, None)
#     note_events2, _, _ = event2note_event(events)
#     assert_note_events_almost_equal(note_events, note_events2)


def preprocess_maps16k(data_home=os.PathLike,
                       dataset_name='maps',
                       ignore_pedal=False,
                       sanity_check=False) -> None:
    """
    Splits:
        - train: following the convention described in Cheuk et al. (2021),
            we filter out the songs overlapping with the MAPS test set. 
            139 pieces from MUS folder are left for training.
        - test: 60 files (MUS)
        - all: 270 files including (unfiltered) train and test. This is used 
           for the evaluation on the MusicNet test set. 


    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
        {
            index:
            {
                'maps_id': maps_id,
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
    train_mid_pattern = os.path.join(base_dir, 'train/**/MUS/*.mid')
    test_mid_pattern = os.path.join(base_dir, 'test/**/MUS/*.mid')
    all_mid_pattern = os.path.join(base_dir, '**/MUS/*.mid')

    train_mid_files = glob.glob(train_mid_pattern, recursive=True)
    test_mid_files = glob.glob(test_mid_pattern, recursive=True)
    all_mid_files = glob.glob(all_mid_pattern, recursive=True)

    # Discard duplicated songs from train and test sets (reduce train set)
    songnames_in_test_files = []
    for file in test_mid_files:
        filename = os.path.basename(file)
        match = re.search(r"MAPS_MUS-([\w-]+)_", filename)
        if match:
            songnames_in_test_files.append(match.group(1))

    filtered_train_mid_files = []
    filtered_train_wav_files = []
    for train_file in train_mid_files:
        if not any(
                songname in os.path.basename(train_file) for songname in songnames_in_test_files):
            filtered_train_mid_files.append(train_file)
            filtered_train_wav_files.append(train_file.replace('.mid', '.wav'))
    assert len(filtered_train_mid_files) == len(filtered_train_wav_files) == 139

    # Process MIDI files
    for i, mid_file in enumerate(all_mid_files):
        maps_id = os.path.basename(mid_file)[:-4]
        notes, note_events = create_note_event_and_note_from_midi(
            mid_file=mid_file, id=maps_id, ignore_pedal=ignore_pedal)

        if sanity_check:
            # sanity check
            print(f'Sanity check for {i}: {maps_id}...')
            note_event2token2note_event_sanity_check(note_events['note_events'], notes['notes'])

        notes_file = mid_file.replace('.mid', '_notes.npy')
        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        print(f'Created {notes_file}')

        note_events_file = mid_file.replace('.mid', '_note_events.npy')
        np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
        print(f'Created {note_events_file}')

        # overwrite midi file with 120 bpm
        rewrite_midi_120bpm(mid_file, note_events['note_events'])
        print(f'Overwrote {mid_file} with 120 bpm')

    # Process audio files
    pass

    # Create file_list.json
    mid_files_by_split = {
        'train': filtered_train_mid_files,
        'test': test_mid_files,
        'all': all_mid_files,
    }

    for split in ['train', 'test', 'all']:
        file_list = {}
        for i, mid_file in enumerate(mid_files_by_split[split]):
            # check if wav file exists
            wav_file = mid_file.replace('.mid', '.wav')
            if not os.path.exists(wav_file):
                raise FileNotFoundError(f'Wav file not found: {wav_file}')

            file_list[i] = {
                'maps_id': os.path.basename(mid_file)[:-4],
                'n_frames': get_audio_file_info(wav_file)[1],
                'mix_audio_file': wav_file,
                'notes_file': mid_file.replace('.mid', '_notes.npy'),
                'note_events_file': mid_file.replace('.mid', '_note_events.npy'),
                'midi_file': mid_file,
                'program': [0],
                'is_drum': [0],
            }
        output_file = os.path.join(output_index_dir, f'{dataset_name}_{split}_file_list.json')
        with open(output_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Created {output_file}')
