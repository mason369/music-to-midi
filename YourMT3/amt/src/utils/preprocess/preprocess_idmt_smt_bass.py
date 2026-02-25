""" preprocess_idmt_smt_bass.py """
import os
import glob
import json
import wave
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from utils.audio import get_audio_file_info, load_audio_file, write_wav_file, guess_onset_offset_by_amp_envelope
from utils.midi import midi2note, note_event2midi
from utils.note2event import note2note_event, sort_notes, validate_notes, trim_overlapping_notes
from utils.event2note import event2note_event
from utils.note_event_dataclasses import Note, NoteEvent
from utils.utils import assert_note_events_almost_equal

SPLIT_INFO_FILE = 'stratified_split_crepe_smt.json'

# Plucking style to GM program
PS2program = {
    "FS": 33,  # Fingered Elec Bass
    "MU": 33,  # Muted Elec Bass
    "PK": 34,  # Picked Elec Bass
    "SP": 36,  # Slap-Pluck Elec Bass
    "ST": 37,  # Salp-Thumb Elec Bass
}
PREPEND_SILENCE = 1.8  # seconds
APPEND_SILENCE = 1.8  # seconds


def bass_string_to_midi_pitch(string_number: int, fret: int, string_pitches=[28, 33, 38, 43, 48]):
    """ sring_number: 1, 2, 3, 4,  fret: 0, 1, 2, ..."""
    return string_pitches[string_number - 1] + fret


def regenerate_stratified_split(audio_files_dict):
    train_ids_dict = {}
    val_ids_dict = {}
    offset = 0

    for key, files in audio_files_dict.items():
        ids = np.arange(len(files)) + offset
        train_ids, val_ids = train_test_split(
            ids, test_size=0.2, random_state=42, stratify=np.zeros_like(ids))
        train_ids_dict[key] = train_ids
        val_ids_dict[key] = val_ids
        offset += len(files)

    train_ids = np.concatenate(list(train_ids_dict.values()))
    val_ids = np.concatenate(list(val_ids_dict.values()))
    assert len(train_ids) == 1872 and len(val_ids) == 470
    return train_ids, val_ids


def create_note_event_and_note_from_midi(mid_file: str,
                                         id: str,
                                         program: int = 0,
                                         ignore_pedal: bool = True) -> Tuple[Dict, Dict]:
    """Extracts note or note_event and metadata from midi:

    Returns:
        notes (dict): note events and metadata.
        note_events (dict): note events and metadata.
    """
    notes, dur_sec = midi2note(
        mid_file,
        binary_velocity=True,
        force_all_program_to=program,
        fix_offset=True,
        quantize=True,
        verbose=0,
        minimum_offset_sec=0.01,
        ignore_pedal=ignore_pedal)
    return {  # notes
        'idmt_smt_bass_id': str(id),
        'program': [program],
        'is_drum': [0],
        'duration_sec': dur_sec,
        'notes': notes,
    }, {  # note_events
        'idmt_smt_bass_id': str(id),
        'program': [0],
        'is_drum': [0],
        'duration_sec': dur_sec,
        'note_events': note2note_event(notes),
    }


def preprocess_idmt_smt_bass_16k(data_home=os.PathLike,
                                 dataset_name='idmt_smt_bass',
                                 sanity_check=True,
                                 edit_audio=True,
                                 regenerate_split=False) -> None:
    """
    Splits: stratified by plucking style
        'train': 1872
        'validation': 470
    Total: 2342

    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
        {
            index:
            {
                'idmt_smt_bass_id': idmt_smt_bass_id,
                'n_frames': (int),
                'mix_audio_file': 'path/to/mix.wav',
                'notes_file': 'path/to/notes.npy',
                'note_events_file': 'path/to/note_events.npy',
                'midi_file': 'path/to/midi.mid',
                'program': List[int], see PS2program above
                'is_drum': List[int], # always [0] for this dataset
            }
        }
    """

    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # # audio file list
    # FS_audio_pattern = os.path.join(base_dir, 'PS/FS/*.wav')
    # MU_audio_pattern = os.path.join(base_dir, 'PS/MU/*.wav')
    # PK_audio_pattern = os.path.join(base_dir, 'PS/PK/*.wav')
    # SP_audio_pattern = os.path.join(base_dir, 'PS/SP/*.wav')
    # ST_audio_pattern = os.path.join(base_dir, 'PS/ST/*.wav')
    # FS_audio_files = sorted(glob.glob(FS_audio_pattern, recursive=False))
    # MU_audio_files = sorted(glob.glob(MU_audio_pattern, recursive=False))
    # PK_audio_files = sorted(glob.glob(PK_audio_pattern, recursive=False))
    # SP_audio_files = sorted(glob.glob(SP_audio_pattern, recursive=False))
    # ST_audio_files = sorted(glob.glob(ST_audio_pattern, recursive=False))
    # assert len(FS_audio_files) == 469
    # assert len(MU_audio_files) == 468
    # assert len(PK_audio_files) == 468
    # assert len(SP_audio_files) == 469
    # assert len(ST_audio_files) == 468
    # audio_files_dict = {
    #     'FS': FS_audio_files,
    #     'MU': MU_audio_files,
    #     'PK': PK_audio_files,
    #     'SP': SP_audio_files,
    #     'ST': ST_audio_files
    # }

    # splits:
    split_info_file = os.path.join(base_dir, SPLIT_INFO_FILE)
    with open(split_info_file, 'r') as f:
        split_info = json.load(f)

    all_info_dict = {}
    id = 0
    for split in ['train', 'validation']:
        for file_path in split_info[split]:
            audio_file = os.path.join(base_dir, file_path)
            assert os.path.exists(audio_file)
            all_info_dict[id] = {
                'idmt_smt_bass_id': id,
                'n_frames': None,
                'mix_audio_file': audio_file,
                'notes_file': None,
                'note_events_file': None,
                'midi_file': None,
                'program': None,
                'is_drum': [0]
            }
            id += 1
    train_ids = np.arange(len(split_info['train']))
    val_ids = np.arange(len(split_info['validation'])) + len(train_ids)
    # if regenerate_split is True:
    #     train_ids, val_ids = regenerate_stratified_split(audio_files_dict)
    # else:
    #     val_ids = VALIDATION_IDS
    #     train_ids = [i for i in range(len(all_info_dict)) if i not in val_ids]

    # Audio processing: prepend/append 1.8s silence
    if edit_audio is True:
        for v in all_info_dict.values():
            audio_file = v['mix_audio_file']
            fs, x_len, _ = get_audio_file_info(audio_file)
            x = load_audio_file(audio_file)  # (T,)
            prefix_len = int(fs * PREPEND_SILENCE)
            suffix_len = int(fs * APPEND_SILENCE)
            x_new_len = prefix_len + x_len + suffix_len
            x_new = np.zeros(x_new_len)
            x_new[prefix_len:prefix_len + x_len] = x

            # overwrite audio file
            print(f'Overwriting {audio_file} with silence prepended/appended')
            write_wav_file(audio_file, x_new, fs)

    # Guess Program/Pitch/Onset/Offset and Generate Notes/NoteEvents/MIDI
    for id in all_info_dict.keys():
        audio_file = all_info_dict[id]['mix_audio_file']

        # Guess program/pitch from audio file name
        _, _, _, _, pluck_style, _, string_num, fret_num = os.path.basename(audio_file).split(
            '.')[0].split('_')
        program = PS2program[pluck_style]
        pitch = bass_string_to_midi_pitch(int(string_num), int(fret_num))

        # Guess onset/offset from audio signal x
        fs, n_frames, _ = get_audio_file_info(audio_file)
        x = load_audio_file(audio_file, fs=fs)
        onset, offset, _ = guess_onset_offset_by_amp_envelope(
            x, fs=fs, onset_threshold=0.05, offset_threshold=0.02, frame_size=256)
        onset = round((onset / fs) * 1000) / 1000
        offset = round((offset / fs) * 1000) / 1000

        # Notes and NoteEvents
        notes = [
            Note(
                is_drum=False,
                program=program,
                onset=onset,
                offset=offset,
                pitch=pitch,
                velocity=1,
            )
        ]
        note_events = note2note_event(notes)

        # Write MIDI
        midi_file = audio_file.replace('.wav', '.mid')
        note_event2midi(note_events, midi_file)

        # Reconvert MIDI to Notes/NoteEvents, and validate
        notes_dict, note_events_dict = create_note_event_and_note_from_midi(
            midi_file, id, program=program, ignore_pedal=True)
        if sanity_check:
            assert_note_events_almost_equal(note_events_dict['note_events'], note_events)

        # Write notes and note_events
        notes_file = audio_file.replace('.wav', '_notes.npy')
        note_events_file = audio_file.replace('.wav', '_note_events.npy')
        np.save(notes_file, notes_dict, allow_pickle=True, fix_imports=False)
        np.save(note_events_file, note_events_dict, allow_pickle=True, fix_imports=False)
        print(f'Created {notes_file}')
        print(f'Created {note_events_file}')

        # Update all_info_dict
        all_info_dict[id]['n_frames'] = n_frames
        all_info_dict[id]['notes_file'] = notes_file
        all_info_dict[id]['note_events_file'] = note_events_file
        all_info_dict[id]['midi_file'] = midi_file
        all_info_dict[id]['program'] = [program]

    # Save index
    ids = {'train': train_ids, 'validation': val_ids, 'all': list(all_info_dict.keys())}
    for split in ['train', 'validation']:
        fl = {}
        for i, id in enumerate(ids[split]):
            fl[i] = all_info_dict[id]
        output_index_file = os.path.join(output_index_dir, f'{dataset_name}_{split}_file_list.json')
        with open(output_index_file, 'w') as f:
            json.dump(fl, f, indent=4)
        print(f'Created {output_index_file}')


def test_guess_onset_offset_by_amp_envelope(all_info_dict):
    import matplotlib.pyplot as plt
    id = np.random.randint(0, 2300)
    x = load_audio_file(all_info_dict[id]['mix_audio_file'])
    onset, offset, amp_env = guess_onset_offset_by_amp_envelope(x)
    plt.plot(x)
    plt.axvline(x=onset, color='r', linestyle='--', label='onset')
    plt.axvline(x=offset, color='g', linestyle='--', label='offset')
    plt.show()