""" preprocess_mtrack_slakh.py

"""
import os
import time
import json
from typing import Dict, List, Tuple
import numpy as np
from utils.audio import get_audio_file_info, load_audio_file
from utils.midi import midi2note
from utils.note2event import note2note_event, mix_notes
import mirdata
from utils.mirdata_dev.datasets import slakh16k


def create_audio_stem_from_mtrack(ds: mirdata.core.Dataset,
                                  mtrack_id: str,
                                  delete_source_files: bool = False) -> Dict:
    """Extracts audio stems and metadata from a multitrack."""
    mtrack = ds.multitrack(mtrack_id)
    track_ids = mtrack.track_ids
    max_length = 0
    program_numbers = []
    is_drum = []
    audio_tracks = []  # multi-channel audio array (C, T)

    # collect all the audio tracks and their metadata
    for track_id in track_ids:
        track = ds.track(track_id)
        audio_file = track.audio_path
        program_numbers.append(track.program_number)
        is_drum.append(1) if track.is_drum else is_drum.append(0)

        fs, n_frames, n_channels = get_audio_file_info(audio_file)
        assert (fs == 16000 and n_channels == 1)
        max_length = n_frames if n_frames > max_length else max_length
        audio = load_audio_file(audio_file, dtype=np.int16)  # returns bytes
        audio = audio / 2**15
        audio = audio.astype(np.float16)
        audio_tracks.append(audio)
        if delete_source_files:
            print(f'🗑️ Deleting {audio_file} ...')
            os.remove(audio_file)

    # collate all the audio tracks into a single array
    n_tracks = len(track_ids)
    audio_array = np.zeros((n_tracks, max_length), dtype=np.float16)
    for j, audio in enumerate(audio_tracks):
        audio_array[j, :len(audio)] = audio

    stem_content = {
        'mtrack_id': mtrack_id,  # str
        'program': np.array(program_numbers, dtype=np.int64),
        'is_drum': np.array(is_drum, dtype=np.int64),
        'n_frames': max_length,  # int
        'audio_array': audio_array  # (n_tracks, n_frames)
    }
    return stem_content


def create_note_event_and_note_from_mtrack_mirdata(
        ds: mirdata.core.Dataset,
        mtrack_id: str,
        fix_bass_octave: bool = True) -> Tuple[Dict, Dict]:
    """Extracts note or note_event and metadata from a multitrack:
    Args:
        ds (mirdata.core.Dataset): Slakh dataset.
        mtrack_id (str): multitrack id.
    Returns:
        notes (dict): note events and metadata.
        note_events (dict): note events and metadata.
    """
    mtrack = ds.multitrack(mtrack_id)
    track_ids = mtrack.track_ids
    program_numbers = []
    is_drum = []
    mixed_notes = []
    duration_sec = 0.

    # mix notes from all stem midi files
    for track_id in track_ids:
        track = ds.track(track_id)
        stem_midi_file = track.midi_path
        notes, dur_sec = midi2note(
            stem_midi_file,
            binary_velocity=True,
            ch_9_as_drum=False,  # checked safe to set to False in Slakh
            force_all_drum=True if track.is_drum else False,
            force_all_program_to=None,  # Slakh always has program number
            trim_overlap=True,
            fix_offset=True,
            quantize=True,
            verbose=0,
            minimum_offset_sec=0.01,
            drum_offset_sec=0.01)

        if fix_bass_octave == True and track.program_number in np.arange(32, 40):
            if track.plugin_name == 'scarbee_jay_bass_slap_both.nkm':
                pass
            else:
                for note in notes:
                    note.pitch -= 12
                print("Fixed bass octave for track", track_id)

        mixed_notes = mix_notes((mixed_notes, notes), True, True, True)
        program_numbers.append(track.program_number)
        is_drum.append(1) if track.is_drum else is_drum.append(0)
        duration_sec = max(duration_sec, dur_sec)

    # convert mixed notes to note events
    mixed_note_events = note2note_event(mixed_notes, sort=True, return_activity=True)
    return {  # notes
        'mtrack_id': mtrack_id,  # str
        'program': np.array(program_numbers, dtype=np.int64),  # (n,)
        'is_drum': np.array(is_drum, dtype=np.int64),  # (n,) with 1 is drum
        'duration_sec': duration_sec,  # float
        'notes': mixed_notes  # list of Note instances
    }, {  # note_events
        'mtrack_id': mtrack_id,  # str
        'program': np.array(program_numbers, dtype=np.int64),  # (n,)
        'is_drum': np.array(is_drum, dtype=np.int64),  # (n,) with 1 is drum
        'duration_sec': duration_sec,  # float
        'note_events': mixed_note_events  # list of NoteEvent instances
    }


def preprocess_slakh16k(data_home: str,
                        run_checksum: bool = False,
                        delete_source_files: bool = False,
                        fix_bass_octave: bool = True) -> None:
    """
    Processes the Slakh dataset and extracts stems for each multitrack.

    Args:
        data_home (str): path to the Slakh data.
        run_checksum (bool): if True, validates the dataset using its checksum. Default is False.
        delete_source_files (bool): if True, deletes original audio files. Default is False.
        fix_bass_octave (bool): if True, fixes the bass to be -1 octave. Slakh bass is annotated as +1 octave. Default is True.
        
    Writes:
        - {dataset_name}_{split}_file_list.json: a dictionary with the following keys:
            {
                'mtrack_id': mtrack_id,
                'n_frames': n of audio frames
                'stem_file': Dict of stem audio file info
                'mix_audio_file': mtrack.mix_path,
                'notes_file': available only for 'validation' and 'test'
                'note_events_file': available only for 'train' and 'validation'
                'midi_file': mtrack.midi_path
            }
    """
    start_time = time.time()

    ds = slakh16k.Dataset(data_home=data_home, version='2100-yourmt3-16k')
    if run_checksum:
        print('Checksum for slakh dataset...')
        ds.validate()
    print('Preprocessing slakh dataset...')

    mtrack_split_dict = ds.get_mtrack_splits()
    for split in ['train', 'validation', 'test']:
        file_list = {}  # write a file list for each split
        mtrack_ids = mtrack_split_dict[split]

        for i, mtrack_id in enumerate(mtrack_ids):
            print(f'🏃🏻‍♂️: processing {mtrack_id} ({i+1}/{len(mtrack_ids)} in {split})')
            mtrack = ds.multitrack(mtrack_id)
            output_dir = os.path.dirname(mtrack.mix_path)  # same as mtrack
            """Audio: get stems (as array) and metadata from the multitrack"""
            stem_content = create_audio_stem_from_mtrack(ds, mtrack_id, delete_source_files)

            # save the audio array and metadata to disk
            stem_file = os.path.join(output_dir, mtrack_id + '_stem.npy')
            np.save(stem_file, stem_content)
            print(f'💿 Created {stem_file}')

            # no preprocessing for mix audio
            """MIDI: pre-process and get metadata from the multitrack"""
            notes, note_events = create_note_event_and_note_from_mtrack_mirdata(
                ds, mtrack_id, fix_bass_octave=fix_bass_octave)
            # save the note events and metadata to disk
            notes_file = os.path.join(output_dir, mtrack_id + '_notes.npy')
            np.save(notes_file, notes, allow_pickle=True, \
                    fix_imports=False)
            print(f'🎹 Created {notes_file}')

            note_events_file = os.path.join(output_dir, mtrack_id + '_note_events.npy')
            np.save(note_events_file, note_events, allow_pickle=True, \
                    fix_imports=False)
            print(f'🎹 Created {note_events_file}')

            # add to the file list of the split
            file_list[i] = {
                'mtrack_id': mtrack_id,
                'n_frames': stem_content['n_frames'], # n of audio frames
                'stem_file': stem_file,
                'mix_audio_file': mtrack.mix_path,
                'notes_file': notes_file,
                'note_events_file': note_events_file,\
                'midi_file': mtrack.midi_path
            }
        # By split, save a file list as json
        summary_dir = os.path.join(data_home, 'yourmt3_indexes')
        os.makedirs(summary_dir, exist_ok=True)
        summary_file = os.path.join(summary_dir, f'slakh_{split}_file_list.json')
        with open(summary_file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'💾 Created {summary_file}')

        elapsed_time = time.time() - start_time
        print(
            f"⏰: {int(elapsed_time // 3600):02d}h {int(elapsed_time % 3600 // 60):02d}m {elapsed_time % 60:.2f}s"
        )
    """ end of preprocess_slakh16k """


def add_program_and_is_drum_info_to_file_list(data_home: str):

    for split in ['train', 'validation', 'test']:
        file_list_dir = os.path.join(data_home, 'yourmt3_indexes')
        file = os.path.join(file_list_dir, f'slakh_{split}_file_list.json')
        with open(file, 'r') as f:
            file_list = json.load(f)

        for v in file_list.values():
            stem_file = v['stem_file']
            stem_content = np.load(stem_file, allow_pickle=True).item()
            v['program'] = stem_content['program'].tolist()
            v['is_drum'] = stem_content['is_drum'].tolist()

        with open(file, 'w') as f:
            json.dump(file_list, f, indent=4)
        print(f'💾 Added program and drum info to {file}')


if __name__ == '__main__':
    from config.config import shared_cfg
    data_home = shared_cfg['PATH']['data_home']
    preprocess_slakh16k(data_home=data_home, delete_source_files=False)