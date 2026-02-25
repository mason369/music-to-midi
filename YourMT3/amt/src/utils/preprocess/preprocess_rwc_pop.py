"""preprocess_rwc_pop.py"""
import os
import json
import csv
from typing import Dict, List, Tuple
import numpy as np
from utils.audio import get_audio_file_info, load_audio_file
from utils.midi import midi2note, note_event2midi
from utils.note2event import note2note_event, sort_notes, validate_notes, trim_overlapping_notes, extract_program_from_notes
from utils.event2note import event2note_event
from utils.note_event_dataclasses import Note, NoteEvent
from utils.utils import note_event2token2note_event_sanity_check
from mido import Message, MidiFile

ID_NO_BASS = ['071', '072', '073', '074', '075', '076', '077', '078', '079', '080']  # 10 files


def check_file_existence(file: str) -> bool:
    """Checks if file exists."""
    res = True
    if not os.path.exists(file):
        res = False
    elif get_audio_file_info(file)[1] < 10 * 16000:
        print(f'File {file} is too short.')
        res = False
    return res


def create_note_event_and_note_from_midi(mid_file: str,
                                         id: str,
                                         ignore_pedal: bool = True) -> Tuple[Dict, Dict]:
    """Extracts note or note_event and metadata from midi:

    Returns:
        notes (dict): note events and metadata.
        note_events (dict): note events and metadata.
    """
    notes, dur_sec, programs = midi2note(
        mid_file,
        binary_velocity=True,
        ch_9_as_drum=True,
        trim_overlap=True,
        fix_offset=True,
        quantize=True,
        verbose=0,
        minimum_offset_sec=0.01,
        drum_offset_sec=0.01,
        ignore_pedal=ignore_pedal,
        return_programs=True)

    # Check drum availability
    has_drum = False
    for note in notes:
        if note.is_drum:
            has_drum = True
    is_drum = [0] * len(programs)
    if has_drum:
        is_drum[9] = 1

    return {  # notes
        'rwc_pop_id': id,
        'program': programs,
        'is_drum': is_drum,
        'duration_sec': dur_sec,
        'notes': notes,
    }, {  # note_events
        'rwc_pop_id': id,
        'program': programs,
        'is_drum': is_drum,
        'duration_sec': dur_sec,
        'note_events': note2note_event(notes),
    }


def preprocess_rwc_pop16k(data_home=os.PathLike, dataset_name='rwc_pop') -> None:
    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # Load CSV: construct id to midi/wav dictionary
    csv_file = os.path.join(base_dir, 'wav_to_midi_filename_mapping.csv')
    rwc_bass = {}
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)

        for row in reader:
            id = row[2]
            # Skip unused ids
            # if id in UNUSED_IDS:
            #     continue
            # if id in MULTI_BASS_IDS:
            #     continue

            mix_audio_file = os.path.join(base_dir, headers[0] + row[0],
                                          row[1] + ' ' + headers[1] + '.wav')
            assert check_file_existence(mix_audio_file)
            # mid_file = os.path.join(base_dir, 'MIDI', id + '.mid')
            mid_file = os.path.join(base_dir, 'MIDI-Bass-Octave-fixed-v2', id + '_bass.mid')
            # assert os.path.exists(mid_file)
            if not os.path.exists(mid_file):
                print(mid_file, "does not exist")
                continue

            notes_file = mid_file.replace('.mid', '_notes.npy')
            note_events_file = mid_file.replace('.mid', '_note_events.npy')

            rwc_bass[id] = {
                'rwc_pop_id': id,
                'n_frames': get_audio_file_info(mix_audio_file)[1],
                'mix_audio_file': mix_audio_file,
                'notes_file': notes_file,
                'note_events_file': note_events_file,
                'midi_file': mid_file,
                'program': None,
                'is_drum': None,
            }
    assert len(rwc_bass) == 90

    # Create note and note_event files
    for id in rwc_bass.keys():
        midi_file = rwc_bass[id]['midi_file']
        notes_file = rwc_bass[id]['notes_file']
        note_events_file = rwc_bass[id]['note_events_file']

        # Create note and note_event files
        notes, note_events = create_note_event_and_note_from_midi(midi_file, id, ignore_pedal=True)

        # Update programs and is_drum
        rwc_bass[id]['program'] = notes['program']
        rwc_bass[id]['is_drum'] = notes['is_drum']

        # Save note and note_event files
        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        print(f'Created {notes_file}')
        np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
        print(f'Created {note_events_file}')

        # saving bpm 120 midi files
        bpm120_midi_file = midi_file.replace('.mid', '_bpm120.mid')
        note_event2midi(note_events['note_events'], bpm120_midi_file)
        print(f'Created {bpm120_midi_file}')

    # Save index file
    split = 'bass'
    output_index_file = os.path.join(output_index_dir, f'rwc_pop_{split}_file_list.json')

    file_list = {}
    for i, id in enumerate(rwc_bass.keys()):
        file_list[i] = rwc_bass[id]

    with open(output_index_file, 'w') as f:
        json.dump(file_list, f, indent=4)
    print(f'Created {output_index_file}')
