"""preprocess_maestro.py"""
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
from utils.utils import assert_note_events_almost_equal


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
        'duration_sec': dur_sec + 0.01,
        'notes': notes,
    }, {  # note_events
        'maps_id': id,
        'program': [0],
        'is_drum': [0],
        'duration_sec': dur_sec + 0.01,
        'note_events': note2note_event(notes),
    }


def note_event2event_sanity_check(note_events: List[NoteEvent]):
    """Sanity check for note events."""
    events = note_event2event(note_events, None)
    note_events2, _, _ = event2note_event(events)
    assert_note_events_almost_equal(note_events, note_events2)


def preprocess_maestro16k(data_home=os.PathLike,
                          dataset_name='maestro',
                          ignore_pedal=False,
                          sanity_check=False) -> None:
    """
    Splits:
        - train: 962 files
        - validation: 137 files
        - test: 177 files
        - all: 1276 file

    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
        {
            index:
            {
                'maestro_id': maestro_id,
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

    # Get metadata
    metadata_file = os.path.join(base_dir, 'maestro-v3.0.0.json')
    with open(metadata_file, 'r') as f:
        _metadata = json.load(f)
    metadata = {}
    ids_all = list(range(len(_metadata['canonical_composer'])))
    assert len(ids_all) == 1276
    for i in ids_all:
        metadata[i] = {}
        for key in ['split', 'midi_filename', 'audio_filename', 'duration']:
            metadata[i][key] = _metadata[key][str(i)]

    # Collect ids and prepend base_dir to filenames
    ids = {'all': ids_all, 'train': [], 'validation': [], 'test': []}
    for i in ids_all:
        m = metadata[i]
        ids[m['split']].append(i)
        # Prepend base_dir
        m['midi_filename'] = os.path.join(base_dir, m['midi_filename'])
        m['audio_filename'] = os.path.join(base_dir, m['audio_filename'])

        # Rename '.midi' to '.mid'
        if '.midi' in m['midi_filename'] and not os.path.exists(m['midi_filename'].replace(
                '.midi', '.mid')):
            os.rename(m['midi_filename'], m['midi_filename'].replace('.midi', '.mid'))
        m['midi_filename'] = m['midi_filename'].replace('.midi', '.mid')

        # File sanity check
        assert os.path.exists(m['midi_filename']) and '.mid' == m['midi_filename'][-4:]
        assert os.path.exists(m['audio_filename']) and '.wav' in m['audio_filename']

    assert len(ids['train']) == 962
    assert len(ids['validation']) == 137
    assert len(ids['test']) == 177

    # Create 'all' filelist, and process MIDI
    file_list = {}
    for i in ids['all']:
        m = metadata[i]
        mix_audio_file = m['audio_filename']
        fs, n_frames, n_channels = get_audio_file_info(mix_audio_file)
        assert fs == 16000 and n_channels == 1
        n_frames = min(int(m['duration'] * 16000), n_frames)
        assert n_frames > 32001

        notes_file = m['midi_filename'].replace('.mid', '_notes.npy')
        note_events_file = m['midi_filename'].replace('.mid', '_note_events.npy')
        midi_file = m['midi_filename']

        file_list[i] = {
            'maestro_id': i,
            'n_frames': n_frames,
            'mix_audio_file': mix_audio_file,
            'notes_file': notes_file,
            'note_events_file': note_events_file,
            'midi_file': midi_file,
            'program': [0],
            'is_drum': [0],
        }

        # Process MIDI
        notes, note_events = create_note_event_and_note_from_midi(
            mid_file=midi_file, id=i, ignore_pedal=ignore_pedal)

        if sanity_check:
            # sanity check
            print(f'Sanity check for {i}: {midi_file}')
            note_event2token2note_event_sanity_check(note_events['note_events'], notes['notes'])

        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        print(f'Created {notes_file}')
        np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
        print(f'Created {note_events_file}')

    # Save index
    for split in ['all', 'train', 'validation', 'test']:
        fl = {}
        for i, maestro_id in enumerate(ids[split]):
            fl[i] = file_list[maestro_id]
        output_index_file = os.path.join(output_index_dir, f'{dataset_name}_{split}_file_list.json')
        with open(output_index_file, 'w') as f:
            json.dump(fl, f, indent=4)
        print(f'Created {output_index_file}')
