"""preprocess_mir1k.py"""
import os
import glob
import re
import json
from typing import Dict, List, Tuple
import numpy as np
from utils.audio import get_audio_file_info, load_audio_file
from utils.midi import midi2note, note_event2midi
from utils.note2event import note2note_event, sort_notes, validate_notes, trim_overlapping_notes
from utils.event2note import event2note_event
from utils.note_event_dataclasses import Note, NoteEvent
from utils.utils import note_event2token2note_event_sanity_check

# def create_spleeter_audio_stem(vocal_audio_file, accomp_audio_file, mir_st500_id) -> Dict:
#     program = MIR_ST500_PROGRAM
#     is_drum = [0, 0]

#     audio_tracks = []  # multi-channel audio array (C, T)
#     vocal_audio = load_audio_file(vocal_audio_file, dtype=np.int16) / 2**15  # returns bytes
#     audio_tracks.append(vocal_audio.astype(np.float16))
#     accomp_audio = load_audio_file(accomp_audio_file, dtype=np.int16) / 2**15  # returns bytes
#     audio_tracks.append(accomp_audio.astype(np.float16))
#     max_length = max(len(vocal_audio), len(accomp_audio))

#     # collate all the audio tracks into a single array
#     n_tracks = 2
#     audio_array = np.zeros((n_tracks, max_length), dtype=np.float16)
#     for j, audio in enumerate(audio_tracks):
#         audio_array[j, :len(audio)] = audio

#     stem_content = {
#         'mir_st500_id': mir_st500_id,
#         'program': np.array(program, dtype=np.int64),
#         'is_drum': np.array(is_drum, dtype=np.int64),
#         'n_frames': max_length,  # int
#         'audio_array': audio_array  # (n_tracks, n_frames)
#     }
#     return stem_content

# def create_note_note_event_midi_from_mir1k_annotation(ann, midi_file, mir_st500_id):
#     """
#     Args:
#         ann: List[List[float, float, float]] # [onset, offset, pitch]
#         mir_st500_id: str
#     Returns:
#         notes: List[Note]
#         note_events: List[NoteEvent]
#         midi: List[List[int]]
#     """
#     notes = []
#     for onset, offset, pitch in ann:
#         notes.append(
#             Note(
#                 is_drum=False,
#                 program=100,
#                 onset=float(onset),
#                 offset=float(offset),
#                 pitch=int(pitch),
#                 velocity=1))
#     notes = sort_notes(notes)
#     notes = validate_notes(notes)
#     notes = trim_overlapping_notes(notes)
#     note_events = note2note_event(notes)

#     # Write midi file
#     note_event2midi(note_events, midi_file)
#     print(f"Created {midi_file}")

#     return {  # notes
#         'mir_st500_id': mir_st500_id,
#         'program': MIR_ST500_PROGRAM,
#         'is_drum': [0, 0],
#         'duration_sec': note_events[-1].time,
#         'notes': notes,
#     }, {  # note_events
#         'mir_st500_id': mir_st500_id,
#         'program': MIR_ST500_PROGRAM,
#         'is_drum': [0, 0],
#         'duration_sec': note_events[-1].time,
#         'note_events': note_events,
#     }


def preprocess_mir1k_16k(data_home=os.PathLike, dataset_name='mir1k', sanity_check=False) -> None:
    """
    Splits:
        - train: index 1 to 400, 346 files (54 files missing)
        - test: index 401 to 500, 94 files (6 files missing)
        - all: 440 files (60 files missing)

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
                'program': List[int], # [100, 129], 100 for singing voice, and 129 for unannotated  
                'is_drum': List[int], # [0] or [1]
            }
        }
    """

    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)