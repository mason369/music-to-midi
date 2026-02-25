"""preprocess_mir_st500.py"""
import os
import json
from typing import Dict
import numpy as np
from utils.audio import get_audio_file_info, load_audio_file
from utils.midi import midi2note, note_event2midi
from utils.note2event import note2note_event, sort_notes, validate_notes, trim_overlapping_notes
from utils.event2note import event2note_event
from utils.note_event_dataclasses import Note, NoteEvent
from utils.utils import note_event2token2note_event_sanity_check

SINGING_WITH_UNANNOTATED_PROGRAM = [100, 129]  # 100 for singing voice, 129 for unannotated
SINGING_ONLY_PROGRAM = [100]


def check_file_existence(file: str) -> bool:
    """Checks if file exists."""
    res = True
    if not os.path.exists(file):
        res = False
    elif get_audio_file_info(file)[1] < 10 * 16000:
        print(f'File {file} is too short.')
        res = False
    return res


def create_spleeter_audio_stem(vocal_audio_file, accomp_audio_file, mir_st500_id) -> Dict:
    program = SINGING_WITH_UNANNOTATED_PROGRAM
    is_drum = [0, 0]

    audio_tracks = []  # multi-channel audio array (C, T)
    vocal_audio = load_audio_file(vocal_audio_file, dtype=np.int16) / 2**15  # returns bytes
    audio_tracks.append(vocal_audio.astype(np.float16))
    accomp_audio = load_audio_file(accomp_audio_file, dtype=np.int16) / 2**15  # returns bytes
    audio_tracks.append(accomp_audio.astype(np.float16))
    max_length = max(len(vocal_audio), len(accomp_audio))

    # collate all the audio tracks into a single array
    n_tracks = 2
    audio_array = np.zeros((n_tracks, max_length), dtype=np.float16)
    for j, audio in enumerate(audio_tracks):
        audio_array[j, :len(audio)] = audio

    stem_content = {
        'mir_st500_id': mir_st500_id,
        'program': np.array(program, dtype=np.int64),
        'is_drum': np.array(is_drum, dtype=np.int64),
        'n_frames': max_length,  # int
        'audio_array': audio_array  # (n_tracks, n_frames)
    }
    return stem_content


def create_note_note_event_midi_from_mir_st500_annotation(ann, midi_file, mir_st500_id):
    """
    Args:
        ann: List[List[float, float, float]] # [onset, offset, pitch]
        mir_st500_id: str
    Returns:
        notes: List[Note]
        note_events: List[NoteEvent]
        midi: List[List[int]]
    """
    notes = []
    for onset, offset, pitch in ann:
        notes.append(
            Note(
                is_drum=False,
                program=100,
                onset=float(onset),
                offset=float(offset),
                pitch=int(pitch),
                velocity=1))
    notes = sort_notes(notes)
    notes = validate_notes(notes)
    notes = trim_overlapping_notes(notes)
    note_events = note2note_event(notes)

    # Write midi file
    note_event2midi(note_events, midi_file)
    print(f"Created {midi_file}")

    return {  # notes
        'mir_st500_id': mir_st500_id,
        'program': SINGING_ONLY_PROGRAM,
        'is_drum': [0, 0],
        'duration_sec': note_events[-1].time,
        'notes': notes,
    }, {  # note_events
        'mir_st500_id': mir_st500_id,
        'program': SINGING_ONLY_PROGRAM,
        'is_drum': [0, 0],
        'duration_sec': note_events[-1].time,
        'note_events': note_events,
    }


def correct_ann(ann_all: Dict, fix_offset: bool = False, max_dur: float = 0.5):
    """ correct too short notes that are actully sung in legato """
    for i in range(1, 101):
        for j, v in enumerate(ann_all[str(i)]):
            dur = v[1] - v[0]
            if dur < 0.01:
                next_onset = ann_all[str(i)][j + 1][0]
                dist_to_next_onset = next_onset - v[1]
                if fix_offset is True:
                    if dist_to_next_onset < max_dur:
                        # correct the offset
                        ann_all[str(i)][j][1] = next_onset
                        print(f'Corrected track {i}: {v} to {ann_all[str(i)][j]}')
                else:
                    print(v, ann_all[str(i)][j + 1], f'dist_to_next_onset: {dist_to_next_onset}')


def preprocess_mir_st500_16k(data_home=os.PathLike,
                             dataset_name='mir_st500',
                             apply_correction=False,
                             sanity_check=False) -> None:
    """
    Splits:
        'train',
        'train_vocal',
        'train_stem',
        'test',
        'test_vocal',
        'all',
        'all_vocal',
        'all_stem'

    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
        {
            index:
            {
                'mir_st500_id': mir_st500_id,
                'n_frames': (int),
                'mix_audio_file': 'path/to/mix.wav',
                'notes_file': 'path/to/notes.npy',
                'note_events_file': 'path/to/note_events.npy',
                'midi_file': 'path/to/midi.mid',
                'program': List[int], 100 for singing voice, and 129 for unannotated  
                'is_drum': List[int], # [0] or [1]
            }
        }
    """

    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # Load annotation json file as dictionary
    ann_file = os.path.join(base_dir, 'MIR-ST500_20210206', 'MIR-ST500_corrected.json')
    with open(ann_file, 'r') as f:
        ann_all = json.load(f)  # index "1" to "500"

    # Correction for annotation
    correct_ann(ann_all, fix_offset=apply_correction, max_dur=0.5)

    # Check missing audio files and create a dictionary
    audio_all = {}  # except for missing files
    audio_missing = {'train': [], 'test': []}
    for i in range(1, 501):
        split = 'train' if i < 401 else 'test'
        audio_file = os.path.join(base_dir, f'{split}', f'{i}', 'converted_Mixture.wav')
        audio_vocal_file = os.path.join(base_dir, f'{split}', f'{i}', 'vocals.wav')
        audio_acc_file = os.path.join(base_dir, f'{split}', f'{i}', 'accompaniment.wav')
        if check_file_existence(audio_file) and check_file_existence(
                audio_vocal_file) and check_file_existence(audio_acc_file):
            audio_all[str(i)] = audio_file
        else:
            audio_missing[split].append(i)
    print(
        f'Number of missing audio files: train = {len(audio_missing["train"])}, test = {len(audio_missing["test"])}'
    )
    assert len(audio_all.keys()) == 500

    # Track ids
    ids_all = audio_all.keys()
    ids_train = []
    ids_test = []
    for i in ids_all:
        if int(i) < 401:
            ids_train.append(i)
        else:
            ids_test.append(i)
    # assert len(ids_train) == 346 and len(ids_test) == 94
    assert len(ids_train) == 400 and len(ids_test) == 100

    # Create notes, note_events, and MIDI from annotation
    for id in ids_all:
        ann = ann_all[id]
        split = 'train' if int(id) < 401 else 'test'
        midi_file = os.path.join(base_dir, f'{split}', id, 'singing.mid')
        notes, note_events = create_note_note_event_midi_from_mir_st500_annotation(
            ann, midi_file, id)

        notes_file = midi_file.replace('.mid', '_notes.npy')
        note_events_file = midi_file.replace('.mid', '_note_events.npy')
        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        print(f"Created {notes_file}")
        np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
        print(f"Created {note_events_file}")

        if sanity_check:
            # sanity check
            print(f'Sanity check for {id}...')
            note_event2token2note_event_sanity_check(note_events['note_events'], notes['notes'])

    # Process audio files
    for id in ids_all:
        split = 'train' if int(id) < 401 else 'test'
        audio_vocal_file = os.path.join(base_dir, f'{split}', id, 'vocals.wav')
        audio_acc_file = os.path.join(base_dir, f'{split}', id, 'accompaniment.wav')
        stem_file = os.path.join(base_dir, f'{split}', id, 'stem.npy')
        stem_content = create_spleeter_audio_stem(audio_vocal_file, audio_acc_file, id)
        # write audio stem
        np.save(stem_file, stem_content, allow_pickle=True, fix_imports=False)
        print(f"Created {stem_file}")

    # Create file_list.json
    ids_by_split = {
        'train': ids_train,
        'train_vocal': ids_train,
        'train_stem': ids_train,
        'test': ids_test,
        'test_vocal': ids_test,
        'all': ids_all,
        'all_vocal': ids_all,
        'all_stem': ids_all
    }

    for split in [
            'train', 'train_vocal', 'train_stem', 'test', 'test_vocal', 'all', 'all_vocal',
            'all_stem'
    ]:
        file_list = {}
        for i, id in enumerate(ids_by_split[split]):
            wav_file = audio_all[id]
            n_frames = get_audio_file_info(wav_file)[1]
            if 'vocal' in split:
                stem_file = None
                wav_file = wav_file.replace('converted_Mixture.wav', 'vocals.wav')
                program = SINGING_ONLY_PROGRAM
                is_drum = [0]
            elif 'stem' in split:
                stem_file = wav_file.replace('converted_Mixture.wav', 'stem.npy')
                program = SINGING_WITH_UNANNOTATED_PROGRAM
                is_drum = [0, 0]
            else:
                stem_file = None
                program = SINGING_WITH_UNANNOTATED_PROGRAM
                is_drum = [0, 0]

            mid_file = os.path.join(os.path.dirname(wav_file), 'singing.mid')
            file_list[i] = {
                'mir_st500_id': id,
                'n_frames': n_frames,
                'stem_file': stem_file,
                'mix_audio_file': wav_file,
                'notes_file': mid_file.replace('.mid', '_notes.npy'),
                'note_events_file': mid_file.replace('.mid', '_note_events.npy'),
                'midi_file': mid_file,
                'program': program,
                'is_drum': is_drum,
            }
            if stem_file is None:
                del file_list[i]['stem_file']

        output_file = os.path.join(output_index_dir, f'{dataset_name}_{split}_file_list.json')
        with open(output_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'Created {output_file}')