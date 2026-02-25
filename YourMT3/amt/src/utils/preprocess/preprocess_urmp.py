"""preprocess_mir1k.py"""
import os
import shutil
import glob
import re
import json
from typing import Dict, List, Tuple
import numpy as np
from utils.audio import get_audio_file_info, load_audio_file
from utils.midi import midi2note, note_event2midi
from utils.note2event import note2note_event, mix_notes, sort_notes, validate_notes, trim_overlapping_notes
from utils.event2note import event2note_event
from utils.note_event_dataclasses import Note, NoteEvent
from utils.utils import note_event2token2note_event_sanity_check, freq_to_midi

MT3_TEST_IDS = [1, 2, 12, 13, 24, 25, 31, 38, 39]
PROGRAM_STR2NUM = {
    'vn': 40,
    'va': 41,
    'vc': 42,
    'db': 43,
    'fl': 73,
    'ob': 68,
    'cl': 71,
    'sax': 65,  # The type of sax used in the dataset is not clear. We guess it would be alto sax.
    'bn': 70,
    'tpt': 56,
    'hn': 60,  # Just annotated as horn. We guess it would be french horn, due to the pitch range.
    'tbn': 57,
    'tba': 58,
}


def delete_hidden_files(base_dir):
    for hidden_file in glob.glob(os.path.join(base_dir, '**/.*'), recursive=True):
        os.remove(hidden_file)
        print(f"Deleted: {hidden_file}")


def convert_annotation_to_notes(id, program, ann_files):
    notes = []
    for ann_file, prog in zip(ann_files, program):
        data = np.loadtxt(ann_file)
        onset = data[:, 0]
        freq = data[:, 1]
        duration = data[:, 2]

        notes_by_instr = []
        for o, f, d in zip(onset, freq, duration):
            notes_by_instr.append(
                Note(
                    is_drum=False,
                    program=prog,
                    onset=o,
                    offset=o + d,
                    pitch=freq_to_midi(f),
                    velocity=1))
        notes = mix_notes([notes, notes_by_instr], sort=True, trim_overlap=True, fix_offset=True)
    notes = sort_notes(notes)
    note_events = note2note_event(notes, sort=True)
    duration_sec = note_events[-1].time + 0.01
    return {  # notes
            'urmp_id': id,
            'program': program,
            'is_drum': [0] * len(program),
            'duration_sec': duration_sec,
            'notes': notes,
        }, {  # note_events
            'guitarset_id': id,
            'program': program,
            'is_drum': [0] * len(program),
            'duration_sec': duration_sec,
            'note_events': note_events,
        }


def create_audio_stem(audio_tracks, id, program, n_frames):
    max_length = max([len(tr) for tr in audio_tracks])
    max_length = max(max_length, n_frames)
    n_tracks = len(audio_tracks)
    audio_array = np.zeros((n_tracks, max_length), dtype=np.float16)
    for j, audio in enumerate(audio_tracks):
        audio_array[j, :len(audio)] = audio

    return {
        'urmp_id': id,
        'program': np.array(program),
        'is_drum': np.array([0] * len(program), dtype=np.int64),
        'n_frames': n_frames,  # int
        'audio_array': audio_array  # (n_tracks, n_frames)
    }


def data_bug_fix(base_dir):
    files = glob.glob(os.path.join(base_dir, '15_Surprise_tpt_tpt_tbn', '*3_tpt*.*'))
    for file in files:
        new_file = file.replace('3_tpt', '3_tbn')
        shutil.move(file, new_file)
        print(f"Renamed: {file} -> {new_file}")


def preprocess_urmp16k(data_home=os.PathLike,
                       dataset_name='urmp',
                       delete_source_files: bool = False,
                       sanity_check=True) -> None:
    """
    URMP dataset does not have official split information. We follow the split used in MT3 paper.

    About:
    - 44 pieces of classical music
    - Duet, Trio, Quartet, Quintet of strings or winds or mixed
    - Multi-stem audio 
    - MIDI file is unaligned, it is for score
    - Annotation (10ms hop) is provided.
    - There is timing issue for annotation
    - We do not use video

    Splits:
        - train: 35 files, following MT3
        - test: 9 files, follwing MT3
        - all: 44 files

    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
        {
            index:
            {
                'urmp_id': urmp_id,
                'n_frames': (int),
                'stem_file': 'path/to/stem.npy',
                'mix_audio_file': 'path/to/mix.wav',
                'notes_file': 'path/to/notes.npy',
                'note_events_file': 'path/to/note_events.npy',
                'midi_file': 'path/to/midi.mid', # this is 120bpm converted midi file from note_events
                'program': List[int], #  
                'is_drum': List[int], # [0] or [1]
            }
        }
    """

    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # Databug fix
    data_bug_fix(base_dir)

    # Delete hidden files
    delete_hidden_files(base_dir)

    # Create file list for split==all
    file_list = dict()
    for dir_name in sorted(os.listdir(base_dir)):
        if dir_name.startswith('.'):
            continue
        if 'Supplementary' in dir_name:
            continue
        # urmp_id
        id = dir_name.split('_')[0]
        title = dir_name.split('_')[1]

        # program
        program_strings = dir_name.split('_')[2:]
        program = [PROGRAM_STR2NUM[p] for p in program_strings]

        # is_drum
        is_drum = [0] * len(program)

        # file paths
        stem_file = os.path.join(base_dir, dir_name, 'stem.npy')
        mix_audio_file = glob.glob(os.path.join(base_dir, dir_name, 'AuMix*.wav'))[0]
        notes_file = os.path.join(base_dir, dir_name, 'notes.npy')
        note_events_file = os.path.join(base_dir, dir_name, 'note_events.npy')
        midi_file = os.path.join(base_dir, dir_name, f'{str(id)}_120bpm_converted.mid')

        # n_frames
        fs, n_frames, n_channels = get_audio_file_info(mix_audio_file)
        assert fs == 16000 and n_channels == 1

        # Fill out a file list
        file_list[id] = {
            'urmp_id': id,
            'n_frames': n_frames,
            'stem_file': stem_file,
            'mix_audio_file': mix_audio_file,
            'notes_file': notes_file,
            'note_events_file': note_events_file,
            'midi_file': midi_file,
            'program': program,
            'is_drum': is_drum,
        }

        # Process Annotations
        ann_files = [
            os.path.join(base_dir, dir_name, f'Notes_{i+1}_{p}_{str(id)}_{title}.txt')
            for i, p in enumerate(program_strings)
        ]

        # Check if all files exist
        for ann_file in ann_files:
            assert os.path.exists(ann_file), f"{ann_file} does not exist."
        assert len(program) == len(ann_files)

        # Create and save notes and note_events from annotation
        notes, note_events = convert_annotation_to_notes(id, program, ann_files)
        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        print(f'Created {notes_file}')
        np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
        print(f'Created {note_events_file}')

        # Create 120bpm MIDI file from note_events
        note_event2midi(note_events['note_events'], midi_file)
        print(f'Created {midi_file}')

        # Process Audio
        audio_tracks = []
        for i, p in enumerate(program_strings):
            audio_sep_file = os.path.join(base_dir, dir_name, f'AuSep_{i+1}_{p}_{id}_{title}.wav')
            audio_track = load_audio_file(audio_sep_file, dtype=np.int16) / 2**15  # returns bytes
            audio_tracks.append(audio_track.astype(np.float16))
            if delete_source_files:
                os.remove(audio_sep_file)

        stem_content = create_audio_stem(audio_tracks, id, program, n_frames)
        np.save(stem_file, stem_content, allow_pickle=True, fix_imports=False)
        print(f'Created {stem_file}')

        # Sanity check
        if sanity_check:
            recon_notes, _ = midi2note(midi_file)
            recon_note_events = note2note_event(recon_notes)
            note_event2token2note_event_sanity_check(recon_note_events, notes['notes'])

        # File existence check
        assert os.path.exists(mix_audio_file)

    # Create index for splits
    file_list_all = {}
    for i, key in enumerate(file_list.keys()):
        file_list_all[i] = file_list[key]

    file_list_train = {}
    i = 0
    for key in file_list.keys():
        if int(key) not in MT3_TEST_IDS:
            file_list_train[i] = file_list[key]
            i += 1

    file_list_test = {}
    i = 0
    for key in file_list.keys():
        if int(key) in MT3_TEST_IDS:
            file_list_test[i] = file_list[key]
            i += 1

    all_fl = {'all': file_list_all, 'train': file_list_train, 'test': file_list_test}

    # Save index
    for split in ['all', 'train', 'test']:
        output_index_file = os.path.join(output_index_dir, f'{dataset_name}_{split}_file_list.json')
        with open(output_index_file, 'w') as f:
            json.dump(all_fl[split], f, indent=4)
        print(f'Created {output_index_file}')