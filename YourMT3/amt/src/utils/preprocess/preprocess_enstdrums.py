""" preprocess_enstdrums.py """
import os
import re
import glob
import copy
import json
import numpy as np
from typing import Dict
from utils.note_event_dataclasses import Note
from utils.audio import get_audio_file_info, load_audio_file, write_wav_file
from utils.midi import note_event2midi
from utils.note2event import note2note_event, sort_notes, validate_notes, trim_overlapping_notes, mix_notes
from config.vocabulary import ENST_DRUM_NOTES

DRUM_OFFSET = 0.01


def create_enst_audio_stem(drum_audio_file, accomp_audio_file, enst_id) -> Dict:
    program = [128, 129]
    is_drum = [1, 0]

    audio_tracks = []  # multi-channel audio array (C, T)
    drum_audio = load_audio_file(drum_audio_file, dtype=np.int16) / 2**15  # returns bytes
    audio_tracks.append(drum_audio.astype(np.float16))
    accomp_audio = load_audio_file(accomp_audio_file, dtype=np.int16) / 2**15  # returns bytes
    audio_tracks.append(accomp_audio.astype(np.float16))
    max_length = max(len(drum_audio), len(accomp_audio))

    # collate all the audio tracks into a single array
    n_tracks = 2
    audio_array = np.zeros((n_tracks, max_length), dtype=np.float16)
    for j, audio in enumerate(audio_tracks):
        audio_array[j, :len(audio)] = audio

    stem_content = {
        'enstdrums_id': enst_id,
        'program': program,
        'is_drum': is_drum,
        'n_frames': max_length,  # int
        'audio_array': audio_array  # (n_tracks, n_frames)
    }
    return stem_content


def create_note_note_event_midi_from_enst_annotation(ann_file, enst_id):
    """
    Args:
        ann_file: 'path/to/annotation.txt'
        enst_id: str
    Returns:
        notes: List[Note]
        note_events: List[NoteEvent]
        midi: List[List[int]]
    """
    # Read the text file and split each line into timestamp and drum instrument name
    with open(ann_file, 'r') as f:
        lines = f.readlines()  # ignore additional annotations like rc2, sd-
    anns = [(float(line.split()[0]), re.sub('[^a-zA-Z]', '', line.split()[1])) for line in lines]

    # Convert ann to notes by ENST_DRUM_NOTES vocabulary
    notes = []
    for time, drum_name in anns:
        if drum_name not in ENST_DRUM_NOTES.keys():
            raise ValueError(f"Drum name {drum_name} is not in ENST_DRUM_NOTES")
        notes.append(
            Note(
                is_drum=True,
                program=128,
                onset=float(time),
                offset=float(time) + DRUM_OFFSET,
                pitch=ENST_DRUM_NOTES[drum_name][0],
                velocity=1))

    notes = sort_notes(notes)
    notes = validate_notes(notes)
    notes = trim_overlapping_notes(notes)
    note_events = note2note_event(notes)

    # Write midi file
    midi_file = ann_file.replace('.txt', '.mid')
    note_event2midi(note_events, midi_file)
    print(f"Created {midi_file}")

    program = [128]
    is_drum = [1]

    return {  # notes
        'enstdrums_id': enst_id,
        'program': program,
        'is_drum': is_drum,
        'duration_sec': note_events[-1].time,
        'notes': notes,
    }, {  # note_events
        'enstdrums_id': enst_id,
        'program': program,
        'is_drum': is_drum,
        'duration_sec': note_events[-1].time,
        'note_events': note_events,
    }


def preprocess_enstdrums16k(data_home: os.PathLike, dataset_name='enstdrums') -> None:
    """
    Some tracks ('minus-one' in the file name) of ENST-drums contain accompaniments.
    'stem file' will contain these accompaniments. 'mix_audio_file' will contain the
    mix of the drums and accompaniments.

    Splits:
        - drummer_1, drummer_2, drummer_3, all
        - drummer3_dtd, drummer3_dtp, drummer3_dtm_r1, drummer3_dtm_r2 (for validation/test)

        DTD means drum track only, DTP means drum track plus percussions, DTM means
        drum track plus music.

        DTM r1 and r2 are two different versions of the mixing tracks derived from listening
        test in [Gillet 2008, Paulus 2009], and used in [Wu 2018].
        r1 uses 1:3 ratio of accompaniment to drums, and r2 uses 2:3 ratio.

        O. Gillet and G. Richard, “Transcription and separation of drum signals from polyphonic music,”
          IEEE Trans. Audio, Speech, Lang. Process., vol. 16, no. 3, pp. 529–540, Mar. 2008.
        J. Paulus and A. Klapuri, “Drum sound detection in polyphonic mu- sic with hidden Markov models,”
          EURASIP J. Audio, Speech, Music Process., vol. 2009, no. 14, 2009, Art. no. 14.
        C. -W. Wu et al., "A Review of Automatic Drum Transcription," 
          IEEE/ACM TASLP, vol. 26, no. 9, pp. 1457-1483, Sept. 2018, 
        
    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
        {
            index:
            {
                'enstdrums_id': {drummer_id}_{3-digit-track-id} 
                'n_frames': (int),
                'stem_file': Dict of stem audio file with metadata,
                'mix_audio_file': 'path/to/mix.wav',
                'notes_file': 'path/to/notes.npy',
                'note_events_file': 'path/to/note_events.npy',
                'midi_file': 'path/to/midi.mid',
                'program': List[int], # 128 for drums, 129 for unannotated (accompaniment) 
                'is_drum': List[int], # 0 or 1
            }
        }
    """
    # Directory and file path
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # Gather info
    enst_ids = []
    enst_info = {}
    for i in [1, 2, 3]:
        drummer_files = sorted(
            glob.glob(os.path.join(base_dir, f'drummer_{i}', 'annotation/*.txt')))
        for file in drummer_files:
            track_id = os.path.basename(file).split('_')[0]
            enst_id = f'{i}_{track_id}'
            enst_ids.append(enst_id)

            # Create notes, note_events, and MIDI from annotation
            ann_file = file
            assert os.path.exists(ann_file), f'{ann_file} does not exist'
            notes, note_events = create_note_note_event_midi_from_enst_annotation(ann_file, enst_id)
            notes_file = ann_file.replace('.txt', '_notes.npy')
            note_events_file = ann_file.replace('.txt', '_note_events.npy')
            np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
            print(f"Created {notes_file}")
            np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
            print(f"Created {note_events_file}")

            # Create stem file from audio for accompaniment
            drum_audio_file = os.path.join(base_dir, f'drummer_{i}', 'audio', 'wet_mix',
                                           os.path.basename(file).replace('.txt', '.wav'))
            assert os.path.exists(drum_audio_file), f'{drum_audio_file} does not exist'

            if 'minus-one' in file:  # unannotated accompaniment exists
                # 129: Unannotated accompaniment exists
                accomp_audio_file = os.path.join(base_dir, f'drummer_{i}', 'audio', 'accompaniment',
                                                 os.path.basename(file).replace('.txt', '.wav'))
                assert os.path.exists(accomp_audio_file), f'{accomp_audio_file} does not exist'
                os.makedirs(os.path.join(base_dir, f'drummer_{i}', 'audio', 'stem'), exist_ok=True)
                stem_file = os.path.join(base_dir, f'drummer_{i}', 'audio', 'stem',
                                         os.path.basename(file).replace('.txt', '_stem.npy'))
                stem_content = create_enst_audio_stem(drum_audio_file, accomp_audio_file, enst_id)
                # write audio stem
                np.save(stem_file, stem_content, allow_pickle=True, fix_imports=False)
                print(f"Created {stem_file}")

                # create (drum + accompaniment) mix audio file. r1
                os.makedirs(
                    os.path.join(base_dir, f'drummer_{i}', 'audio', 'accompaniment_mix_r1'),
                    exist_ok=True)
                accomp_mix_audio_file_r1 = os.path.join(
                    base_dir, f'drummer_{i}', 'audio', 'accompaniment_mix_r1',
                    os.path.basename(file).replace('.txt', '.wav'))
                accomp_mix_audio_r1 = stem_content['audio_array'][0] / np.max(
                    np.abs(stem_content['audio_array'][0])) * 0.75 + stem_content['audio_array'][
                        1] / np.max(np.abs(stem_content['audio_array'][1])) * 0.25
                accomp_mix_audio_r1 = accomp_mix_audio_r1 / np.max(np.abs(accomp_mix_audio_r1))
                write_wav_file(accomp_mix_audio_file_r1, accomp_mix_audio_r1, 16000)
                print(f"Created {accomp_mix_audio_file_r1}")

                # create (drum + accompaniment) mix audio file. r1
                os.makedirs(
                    os.path.join(base_dir, f'drummer_{i}', 'audio', 'accompaniment_mix_r2'),
                    exist_ok=True)
                accomp_mix_audio_file_r2 = os.path.join(
                    base_dir, f'drummer_{i}', 'audio', 'accompaniment_mix_r2',
                    os.path.basename(file).replace('.txt', '.wav'))
                accomp_mix_audio_r2 = stem_content['audio_array'][0] / np.max(
                    np.abs(stem_content['audio_array'][0])) * 0.6 + stem_content['audio_array'][
                        1] / np.max(np.abs(stem_content['audio_array'][1])) * 0.4
                accomp_mix_audio_r2 = accomp_mix_audio_r2 / np.max(np.abs(accomp_mix_audio_r2))
                write_wav_file(accomp_mix_audio_file_r2, accomp_mix_audio_r2, 16000)
                print(f"Created {accomp_mix_audio_file_r2}")
                n_frames = len(accomp_mix_audio_r2)

                # use r2 for training...
                mix_audio_file = accomp_mix_audio_file_r2
            else:
                # No unannotated accompaniment
                stem_file = None
                mix_audio_file = drum_audio_file
                n_frames = get_audio_file_info(drum_audio_file)[1]

            # Create index, this is based on dtm setup
            enst_info[enst_id] = {
                'enstdrums_id': enst_id,
                'n_frames': n_frames,
                'stem_file': stem_file,
                'mix_audio_file': mix_audio_file,
                'notes_file': notes_file,
                'note_events_file': note_events_file,
                'midi_file': ann_file.replace('.txt', '.mid'),
                'program': stem_content['program'] if 'minus-one' in file else notes['program'],
                'is_drum': stem_content['is_drum'] if 'minus-one' in file else notes['is_drum'],
            }

    # Write index
    for split in [
            'drummer_1_dtm', 'drummer_2_dtm', 'all_dtm', 'drummer_1_dtp', 'drummer_2_dtp',
            'all_dtp', 'drummer_3_dtd', 'drummer_3_dtp', 'drummer_3_dtm_r1', 'drummer_3_dtm_r2'
    ]:
        # splits for training
        file_list = {}
        i = 0
        if split == 'drummer_1_dtm':
            for enst_id in enst_ids:
                if enst_id.startswith('1_'):
                    file_list[str(i)] = enst_info[enst_id]
                    i += 1
            assert len(file_list) == 97
        elif split == 'drummer_2_dtm':
            for enst_id in enst_ids:
                if enst_id.startswith('2_'):
                    file_list[str(i)] = enst_info[enst_id]
                    i += 1
            assert len(file_list) == 105
        elif split == 'all_dtm':
            for enst_id in enst_ids:
                file_list[str(i)] = enst_info[enst_id]
                i += 1
            assert len(file_list) == 318
        elif split == 'drummer_1_dtp':
            for enst_id in enst_ids:
                if enst_id.startswith('1_'):
                    file_list[str(i)] = copy.deepcopy(enst_info[enst_id])
                    file_list[str(i)]['stem_file'] = None
                    file_list[str(i)]['mix_audio_file'] = file_list[str(
                        i)]['mix_audio_file'].replace('accompaniment_mix_r2', 'wet_mix')
                    file_list[str(i)]['program'] = [128]
                    file_list[str(i)]['is_drum'] = [1]
                    i += 1
            assert len(file_list) == 97
        elif split == 'drummer_2_dtp':
            for enst_id in enst_ids:
                if enst_id.startswith('2_'):
                    file_list[str(i)] = copy.deepcopy(enst_info[enst_id])
                    file_list[str(i)]['stem_file'] = None
                    file_list[str(i)]['mix_audio_file'] = file_list[str(
                        i)]['mix_audio_file'].replace('accompaniment_mix_r2', 'wet_mix')
                    file_list[str(i)]['program'] = [128]
                    file_list[str(i)]['is_drum'] = [1]
                    i += 1
            assert len(file_list) == 105
        elif split == 'all_dtp':
            for enst_id in enst_ids:
                file_list[str(i)] = copy.deepcopy(enst_info[enst_id])
                file_list[str(i)]['stem_file'] = None
                file_list[str(i)]['mix_audio_file'] = file_list[str(i)]['mix_audio_file'].replace(
                    'accompaniment_mix_r2', 'wet_mix')
                file_list[str(i)]['program'] = [128]
                file_list[str(i)]['is_drum'] = [1]
                i += 1
            assert len(file_list) == 318
        elif split == 'drummer_3_dtd':
            for enst_id in enst_ids:
                if enst_id.startswith('3_') and len(enst_info[enst_id]['program']) == 1:
                    assert enst_info[enst_id]['stem_file'] == None
                    file_list[str(i)] = enst_info[enst_id]
                    i += 1
            assert len(file_list) == 95
        elif split == 'drummer_3_dtp':
            for enst_id in enst_ids:
                if enst_id.startswith('3_') and len(enst_info[enst_id]['program']) == 2:
                    file_list[str(i)] = copy.deepcopy(enst_info[enst_id])
                    file_list[str(i)]['stem_file'] = None
                    # For DTP, we use the drum audio file as the mix audio file
                    file_list[str(i)]['mix_audio_file'] = file_list[str(
                        i)]['mix_audio_file'].replace('accompaniment_mix_r2', 'wet_mix')
                    file_list[str(i)]['program'] = [128]
                    file_list[str(i)]['is_drum'] = [1]
                    i += 1
            assert len(file_list) == 21
        elif split == 'drummer_3_dtm_r1':
            for enst_id in enst_ids:
                if enst_id.startswith('3_') and len(enst_info[enst_id]['program']) == 2:
                    file_list[str(i)] = copy.deepcopy(enst_info[enst_id])
                    file_list[str(i)]['stem_file'] = None
                    file_list[str(i)]['mix_audio_file'] = file_list[str(
                        i)]['mix_audio_file'].replace('accompaniment_mix_r2',
                                                      'accompaniment_mix_r1')
                    i += 1
            assert len(file_list) == 21
        elif split == 'drummer_3_dtm_r2':
            for enst_id in enst_ids:
                if enst_id.startswith('3_') and len(enst_info[enst_id]['program']) == 2:
                    file_list[str(i)] = copy.deepcopy(enst_info[enst_id])
                    file_list[str(i)]['stem_file'] = None
                    i += 1
            assert len(file_list) == 21

        # final check for file existence
        for k, v in file_list.items():
            if v['stem_file'] is not None:
                assert os.path.exists(v['stem_file'])
            assert os.path.exists(v['mix_audio_file'])
            assert os.path.exists(v['notes_file'])
            assert os.path.exists(v['note_events_file'])
            assert os.path.exists(v['midi_file'])

        # write json file
        output_index_file = os.path.join(output_index_dir, f'{dataset_name}_{split}_file_list.json')
        with open(output_index_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f"Created {output_index_file}")


def create_filelist_dtm_random_enstdrums16k(data_home: os.PathLike,
                                            dataset_name: str = 'enstdrums') -> None:
    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # Load all filelist
    file_list_all_dtm_path = os.path.join(output_index_dir,
                                          f'{dataset_name}_all_dtm_file_list.json')
    file_list_all_dtp_path = os.path.join(output_index_dir,
                                          f'{dataset_name}_all_dtp_file_list.json')

    # Collect dtm tracks
    with open(file_list_all_dtm_path, 'r') as f:
        fl = json.load(f)
    fl_dtm = {}
    i = 0
    for v in fl.values():
        if 129 in v['program']:
            fl_dtm[i] = copy.deepcopy(v)
            i += 1
    # Collect dtd tracks
    fl_dtd = {}
    i = 0
    for v in fl.values():
        if 129 not in v['program']:
            fl_dtd[i] = copy.deepcopy(v)
            i += 1

    # Split: 70, 15, 15
    # rand_idx = np.random.permutation(len(fl_dtm))
    idx = {}
    idx['train_dtm'] = [
        47, 58, 14, 48, 60, 44, 34, 31, 5, 62, 46, 12, 9, 26, 57, 11, 16, 22, 33, 3, 6, 55, 50, 32,
        52, 53, 10, 28, 24, 41, 63, 51, 43, 49, 54, 15, 20, 1, 27, 2, 23, 45, 38, 37
    ]
    idx['validation_dtm'] = [39, 4, 19, 59, 61, 17, 56, 36, 29, 0]
    idx['test_dtm'] = [18, 7, 42, 25, 40, 8, 30, 21, 13, 35]
    idx['train_dtp'] = idx['train_dtm']
    idx['validation_dtp'] = idx['validation_dtm']
    idx['test_dtp'] = idx['test_dtm']

    for split in [
            'train_dtm',
            'validation_dtm',
            'test_dtm',
            'train_dtp',
            'validation_dtp',
            'test_dtp',
            'all_dtd',
    ]:
        file_list = {}
        i = 0
        if 'dtm' in split:
            for k, v in fl_dtm.items():
                if int(k) in idx[split]:
                    file_list[i] = copy.deepcopy(v)
                    i += 1
            if split == 'test_dtm' or split == 'validation_dtm':
                # add r1 mix tracks
                for k, v in fl_dtm.items():
                    if int(k) in idx[split]:
                        _v = copy.deepcopy(v)
                        _v['mix_audio_file'] = _v['mix_audio_file'].replace(
                            'accompaniment_mix_r2', 'accompaniment_mix_r1')
                        file_list[i] = _v
                        i += 1
        elif 'dtp' in split:
            for k, v in fl_dtm.items():
                if int(k) in idx[split]:
                    _v = copy.deepcopy(v)
                    _v['stem_file'] = None
                    _v['mix_audio_file'] = _v['mix_audio_file'].replace(
                        'accompaniment_mix_r2', 'wet_mix')
                    _v['program'] = [128]  # bug fixed..
                    _v['is_drum'] = [1]  # bug fixed..
                    file_list[i] = _v
                    i += 1
        elif 'dtd' in split:
            for k, v in fl_dtd.items():
                file_list[i] = copy.deepcopy(v)
                i += 1
        else:
            raise ValueError(f'Unknown split: {split}')

        output_file = os.path.join(output_index_dir, f'{dataset_name}_{split}_file_list.json')
        with open(output_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Created {output_file}')
