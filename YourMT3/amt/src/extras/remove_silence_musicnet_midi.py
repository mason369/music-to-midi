import os
import glob

from utils.midi import midi2note
from utils.note2event import note2note_event
from utils.note_event_dataclasses import Note
from utils.note_event_dataclasses import NoteEvent
from utils.midi import note_event2midi

data_home = '../../data'
dataset_name = 'musicnet'
base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
mid_pattern = os.path.join(base_dir, '*_midi', '*.mid')
mid_files = glob.glob(mid_pattern, recursive=True)

for mid_file in mid_files:
    notes, _ = midi2note(mid_file)
    first_onset_time = notes[0].onset
    fixed_notes = []
    for note in notes:
        fixed_notes.append(
            Note(
                is_drum=note.is_drum,
                program=note.program,
                onset=note.onset - first_onset_time,
                offset=note.offset - first_onset_time,
                pitch=note.pitch,
                velocity=note.velocity))
    assert len(notes) == len(fixed_notes)
    fixed_note_events = note2note_event(fixed_notes, return_activity=False)
    note_event2midi(fixed_note_events, mid_file)
    print(f'Overwriting {mid_file}')
