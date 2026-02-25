""" preprocess_guitarset.py """
import os
import glob
import copy
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import jams
from utils.note_event_dataclasses import Note, NoteEvent
from utils.audio import get_audio_file_info, pitch_shift_audio
from utils.midi import note_event2midi, pitch_shift_midi
from utils.note2event import note2note_event, sort_notes, validate_notes, trim_overlapping_notes


def create_note_event_and_note_from_jam(jam_file: str, id: str) -> Tuple[Dict, Dict]:
    jam = jams.load(jam_file)
    notes = []
    for ann in jam.annotations:
        for obs in ann.data:
            if isinstance(obs.value, float):
                if obs.confidence == None:
                    note = Note(is_drum=False,
                                program=24,
                                onset=obs.time,
                                offset=obs.time + obs.duration,
                                pitch=round(obs.value),
                                velocity=1)
                    notes.append(note)
    # Sort, validate, and trim notes
    notes = sort_notes(notes)
    notes = validate_notes(notes)
    notes = trim_overlapping_notes(notes)

    return {  # notes
        'guitarset_id': id,
        'program': [24],
        'is_drum': [0],
        'duration_sec': jam.file_metadata.duration,
        'notes': notes,
    }, {  # note_events
        'guitarset_id': id,
        'program': [24],
        'is_drum': [0],
        'duration_sec': jam.file_metadata.duration,
        'note_events': note2note_event(notes),
    }


def generate_pitch_shifted_wav_and_midi(file_list: Dict, min_pitch_shift: int = -5, max_pitch_shift: int = 6):
    for key in file_list.keys():
        midi_file = file_list[key]['midi_file']
        audio_file = file_list[key]['mix_audio_file']

        # Write midi, notes, and note_events with pitch shift
        pitch_shift_midi(src_midi_file=midi_file,
                         min_pitch_shift=min_pitch_shift,
                         max_pitch_shift=max_pitch_shift,
                         write_midi_file=True,
                         write_notes_file=True,
                         write_note_events_file=True)

        # Write wav with pitch shift
        pitch_shift_audio(src_audio_file=audio_file,
                          min_pitch_shift=min_pitch_shift,
                          max_pitch_shift=max_pitch_shift,
                          random_microshift_range=(-10, 11))


def preprocess_guitarset16k(data_home: os.PathLike,
                            dataset_name: str = 'guitarset',
                            pitch_shift_range: Optional[Tuple[int, int]] = (-5, 6)) -> None:
    """
    Splits:
        - progression_1, progression_2, progression_3
        - train, validation, test (by random selection [4,1,1] player)

    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
        {
            index:
            {
                'guitarset_id': guitarset_id,
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

    # Process annotations
    all_ann_files = glob.glob(os.path.join(base_dir, 'annotation/*.jams'), recursive=True)
    assert len(all_ann_files) == 360
    notes_files = {}
    note_events_files = {}
    midi_files = {}
    for ann_file in all_ann_files:
        # Convert all annotations to notes and note events
        guitarset_id = os.path.basename(ann_file).split('.')[0]
        notes, note_events = create_note_event_and_note_from_jam(ann_file, guitarset_id)

        notes_file = ann_file.replace('.jams', '_notes.npy')
        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        print(f'Created {notes_file}')

        note_events_file = ann_file.replace('.jams', '_note_events.npy')
        np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
        print(f'Created {note_events_file}')

        # Create a midi file from the note_events
        midi_file = ann_file.replace('.jams', '.mid')
        note_event2midi(note_events=note_events['note_events'], output_file=midi_file)
        print(f'Created {midi_file}')

        notes_files[guitarset_id] = notes_file
        note_events_files[guitarset_id] = note_events_file
        midi_files[guitarset_id] = midi_file

    # Process audio files
    pass

    # Create file_list.json
    guitarset_ids_by_split = {
        'progression_1': [],
        'progression_2': [],
        'progression_3': [],
        'player_0': [],
        'player_1': [],
        'player_2': [],
        'player_3': [],
        'player_4': [],
        'player_5': [],
        'train': [],  # random selection of 4 players for each style
        'validation': [],  # random selection of 1 player for each style
        'test': [],  # random selection of 1 player for each style
        'all': [],
    }
    # by progressions, players and all
    for ann_file in all_ann_files:
        guitarset_id = os.path.basename(ann_file).split('.')[0]
        progression = int(guitarset_id.split('_')[1].split('-')[0][-1])
        player = int(guitarset_id.split('_')[0])

        # all
        guitarset_ids_by_split['all'].append(guitarset_id)

        # progression
        if progression == 1:
            guitarset_ids_by_split['progression_1'].append(guitarset_id)
        elif progression == 2:
            guitarset_ids_by_split['progression_2'].append(guitarset_id)
        elif progression == 3:
            guitarset_ids_by_split['progression_3'].append(guitarset_id)
        else:
            raise ValueError(f'Invalid progression: {guitarset_id}')

        # player
        if player == 0:
            guitarset_ids_by_split['player_0'].append(guitarset_id)
        elif player == 1:
            guitarset_ids_by_split['player_1'].append(guitarset_id)
        elif player == 2:
            guitarset_ids_by_split['player_2'].append(guitarset_id)
        elif player == 3:
            guitarset_ids_by_split['player_3'].append(guitarset_id)
        elif player == 4:
            guitarset_ids_by_split['player_4'].append(guitarset_id)
        elif player == 5:
            guitarset_ids_by_split['player_5'].append(guitarset_id)
        else:
            raise ValueError(f'Invalid player: {guitarset_id}')

    # sort
    for key in guitarset_ids_by_split.keys():
        guitarset_ids_by_split[key] = sorted(guitarset_ids_by_split[key])
    for i in range(6):
        assert len(guitarset_ids_by_split[f'player_{i}']) == 60

    # train/valid/test by random player
    for i in range(60):
        rand_sel = np.random.choice(6, size=6, replace=False)
        player_train = rand_sel[:4]
        player_valid = rand_sel[4]
        player_test = rand_sel[5]
        for player in player_train:
            guitarset_ids_by_split['train'].append(guitarset_ids_by_split[f'player_{player}'][i])
        guitarset_ids_by_split['validation'].append(guitarset_ids_by_split[f'player_{player_valid}'][i])
        guitarset_ids_by_split['test'].append(guitarset_ids_by_split[f'player_{player_test}'][i])

    assert len(guitarset_ids_by_split['train']) == 240
    assert len(guitarset_ids_by_split['validation']) == 60
    assert len(guitarset_ids_by_split['test']) == 60

    # Create file_list.json
    for split in ['progression_1', 'progression_2', 'progression_3', 'train', 'validation', 'test', 'all']:
        file_list = {}
        for i, gid in enumerate(guitarset_ids_by_split[split]):
            # Check if wav files exist for the 4 versions
            wav_file = {}
            wav_file['hex'] = os.path.join(base_dir, 'audio_hex-pickup_original', gid + '_' + 'hex' + '.wav')
            wav_file['hex_cln'] = os.path.join(base_dir, 'audio_hex-pickup_debleeded', gid + '_' + 'hex_cln' + '.wav')
            wav_file['mic'] = os.path.join(base_dir, 'audio_mono-mic', gid + '_' + 'mic' + '.wav')
            wav_file['mix'] = os.path.join(base_dir, 'audio_mono-pickup_mix', gid + '_' + 'mix' + '.wav')
            for ver in wav_file:
                assert os.path.exists(wav_file[ver])

            for ver in ['mic', 'mix']:  #'hex', 'hex_cln',
                file_list[i, ver] = {
                    'guitarset_id': gid + '_' + ver,
                    'n_frames': get_audio_file_info(wav_file[ver])[1],
                    'mix_audio_file': wav_file[ver],
                    'notes_file': notes_files[gid],
                    'note_events_file': note_events_files[gid],
                    'midi_file': midi_files[gid],
                    'program': [24],
                    'is_drum': [0],
                }

        # Reindexing file_list
        _file_list = {}
        for i, v in enumerate(file_list.values()):
            _file_list[i] = v
        file_list = _file_list

        # Write json
        output_file = os.path.join(output_index_dir, f'{dataset_name}_{split}_file_list.json')
        with open(output_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Created {output_file}')

    if pitch_shift_range == None:
        return
    else:
        min_pitch_shift, max_pitch_shift = pitch_shift_range

    # Generate pitch shifted wav and MIDI
    file_list_all_path = os.path.join(output_index_dir, f'{dataset_name}_all_file_list.json')
    with open(file_list_all_path, 'r') as f:
        fl = json.load(f)
    file_list_all = {int(key): value for key, value in fl.items()}
    generate_pitch_shifted_wav_and_midi(file_list_all, min_pitch_shift=min_pitch_shift, max_pitch_shift=max_pitch_shift)

    # Create file_list.json for pitch shifted data
    for split in ['progression_1', 'progression_2', 'progression_3', 'train', 'all']:
        src_file_list_path = os.path.join(output_index_dir, f'{dataset_name}_{split}_file_list.json')
        with open(src_file_list_path, 'r') as f:
            fl = json.load(f)
        src_file_list = {int(key): value for key, value in fl.items()}

        file_list = {}
        for k, v in src_file_list.items():
            for pitch_shift in range(min_pitch_shift, max_pitch_shift):
                if pitch_shift == 0:
                    file_list[k, 0] = copy.deepcopy(v)
                else:
                    file_list[k, pitch_shift] = copy.deepcopy(v)
                    shifted_audio_file = v['mix_audio_file'].replace('.wav', f'_pshift{pitch_shift}.wav')
                    assert os.path.isfile(shifted_audio_file) == True
                    file_list[k, pitch_shift]['mix_audio_file'] = shifted_audio_file
                    file_list[k, pitch_shift]['n_frames'] = get_audio_file_info(shifted_audio_file)[1]
                    file_list[k, pitch_shift]['pitch_shift'] = pitch_shift

                    shifted_midi_file = v['midi_file'].replace('.mid', f'_pshift{pitch_shift}.mid')
                    shifted_notes_file = v['notes_file'].replace('_notes', f'_pshift{pitch_shift}_notes')
                    shifted_note_events_file = v['note_events_file'].replace('_note_events',
                                                                             f'_pshift{pitch_shift}_note_events')
                    assert os.path.isfile(shifted_midi_file) == True
                    assert os.path.isfile(shifted_notes_file) == True
                    assert os.path.isfile(shifted_note_events_file) == True
                    file_list[k, pitch_shift]['midi_file'] = shifted_midi_file
                    file_list[k, pitch_shift]['notes_file'] = shifted_notes_file
                    file_list[k, pitch_shift]['note_events_file'] = shifted_note_events_file
        assert len(file_list) == len(src_file_list) * (max_pitch_shift - min_pitch_shift)

        # Reindexing file_list
        _file_list = {}
        for i, v in enumerate(file_list.values()):
            _file_list[i] = v
        file_list = _file_list

        # Write json
        output_file = os.path.join(output_index_dir, f'{dataset_name}_{split}_pshift_file_list.json')
        with open(output_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Created {output_file}')


def create_filelist_by_style_guitarset16k(data_home: os.PathLike, dataset_name: str = 'guitarset') -> None:

    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # Load filelist, pshift_all for train
    file_list_pshift_all_path = os.path.join(output_index_dir, f'{dataset_name}_all_pshift_file_list.json')
    with open(file_list_pshift_all_path, 'r') as f:
        fl_pshift = json.load(f)
    assert len(fl_pshift) == 7920

    # Load filelist, all for test
    file_list_all_path = os.path.join(output_index_dir, f'{dataset_name}_all_file_list.json')
    with open(file_list_all_path, 'r') as f:
        fl = json.load(f)
    assert len(fl) == 720

    # Create file_list.json for training each style using pitch shifted data
    styles = ['BN', 'Funk', 'SS', 'Jazz', 'Rock']
    for style in styles:
        # Create and write pshift file list
        train_file_list = {}
        i = 0
        for v in fl_pshift.values():
            if style in v['guitarset_id']:
                train_file_list[i] = copy.deepcopy(v)
                i += 1
        output_file = os.path.join(output_index_dir, f'{dataset_name}_{style}_pshift_file_list.json')
        with open(output_file, 'w') as f:
            json.dump(train_file_list, f, indent=4)
        print(f'Created {output_file}')

        test_file_list = {}
        i = 0
        for v in fl.values():
            if style in v['guitarset_id']:
                test_file_list[i] = copy.deepcopy(v)
                i += 1
        output_file = os.path.join(output_index_dir, f'{dataset_name}_{style}_file_list.json')
        with open(output_file, 'w') as f:
            json.dump(test_file_list, f, indent=4)
        print(f'Created {output_file}')


# BASIC_PITCH_VALIDATION_IDS = [
#     "05_Funk2-108-Eb_comp", "04_BN2-131-B_comp", "04_Jazz3-150-C_solo", "05_Rock2-85-F_solo",
#     "05_Funk3-98-A_comp", "05_BN3-119-G_comp", "02_SS2-107-Ab_solo", "01_BN2-131-B_solo",
#     "00_BN2-166-Ab_comp", "04_SS1-100-C#_solo", "01_BN2-166-Ab_solo", "01_Rock1-130-A_solo",
#     "04_Funk2-119-G_solo", "01_SS2-107-Ab_comp", "05_Funk3-98-A_solo", "05_Funk1-114-Ab_comp",
#     "05_Jazz2-187-F#_solo", "05_SS1-100-C#_comp", "00_Rock3-148-C_solo", "02_Rock3-117-Bb_comp",
#     "01_BN1-147-Gb_solo", "01_Rock1-90-C#_solo", "01_SS2-107-Ab_solo", "02_Jazz3-150-C_solo",
#     "00_Funk1-97-C_solo", "05_SS3-98-C_solo", "03_Rock3-148-C_comp", "03_Rock3-117-Bb_solo",
#     "04_Jazz2-187-F#_solo", "05_Jazz2-187-F#_comp", "02_SS1-68-E_solo", "04_SS2-88-F_solo",
#     "04_BN2-131-B_solo", "04_Jazz3-137-Eb_comp", "00_SS2-107-Ab_comp", "01_Rock1-130-A_comp",
#     "00_Jazz1-130-D_comp", "04_Funk2-108-Eb_comp", "05_BN2-166-Ab_comp"
# ]

# BASIC_PITCH_TEST_IDS = [
#     "04_SS3-84-Bb_solo", "02_Funk1-114-Ab_solo", "05_Funk1-114-Ab_solo", "05_Funk1-97-C_solo",
#     "00_Rock3-148-C_comp", "00_Jazz3-137-Eb_comp", "00_Jazz1-200-B_comp", "03_SS3-98-C_solo",
#     "05_Jazz1-130-D_comp", "00_Jazz2-110-Bb_comp", "02_Funk3-98-A_comp", "04_Rock1-130-A_comp",
#     "03_BN1-129-Eb_comp", "03_Funk2-119-G_comp", "05_BN1-147-Gb_comp", "02_Rock1-90-C#_comp",
#     "00_Funk3-98-A_solo", "01_SS1-100-C#_comp", "00_Funk3-98-A_comp", "02_BN3-154-E_comp",
#     "01_Jazz3-137-Eb_comp", "00_BN2-131-B_comp", "04_SS1-68-E_solo", "05_Funk1-97-C_comp",
#     "04_Jazz3-137-Eb_solo", "05_Rock2-142-D_solo", "02_BN3-119-G_solo", "02_Rock2-142-D_solo",
#     "01_BN1-129-Eb_solo", "00_Rock2-85-F_comp", "00_Rock1-130-A_solo"
# ]

# def create_filelist_for_basic_pitch_benchmark_guitarset16k(data_home: os.PathLike,
#                                                            dataset_name: str = 'guitarset') -> None:

#     # Directory and file paths
#     base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
#     output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
#     os.makedirs(output_index_dir, exist_ok=True)

#     # Load filelist, pshift_all for train
#     file_list_pshift_all_path = os.path.join(output_index_dir,
#                                              f'{dataset_name}_all_pshift_file_list.json')
#     with open(file_list_pshift_all_path, 'r') as f:
#         fl_pshift = json.load(f)
#     assert len(fl_pshift) == 7920

#     # Load filelist, all without pshift
#     file_list_all_path = os.path.join(output_index_dir, f'{dataset_name}_all_file_list.json')
#     with open(file_list_all_path, 'r') as f:
#         fl = json.load(f)
#     assert len(fl) == 720

#     # This is abandoned, because the split is not official one.
