"""preprocess_rwc_pop.py"""
import os
import glob
import re
import json
import csv
from typing import Dict, List, Any, Tuple
import numpy as np
from utils.audio import get_audio_file_info, load_audio_file
from utils.midi import midi2note, note_event2midi
from utils.note2event import note2note_event, sort_notes, validate_notes, trim_overlapping_notes, extract_program_from_notes
from utils.event2note import event2note_event
from utils.note_event_dataclasses import Note, NoteEvent
from utils.utils import note_event2token2note_event_sanity_check
from mido import MetaMessage, Message, MidiFile, MidiTrack

# UNUSED_IDS = ["010", "071", "099", "023", "034", "036", "038", "049", "060", "062"]
# UNUSED_IDS = ["071", "099", "049", "060", "062"]
UNUSED_IDS = []

DRUM_CHANNEL = 9  # all drums are in channel 9 in geerdes dataset
DRUM_PROGRAM = 128
SINGING_VOICE_PROGRAM = 100
SINGING_VOICE_CHORUS_PROGRAM = 101
TRACK_NAME_TO_PROGRAM_MAP = { # compared by exact match of lowercase
    "Singing Voice": SINGING_VOICE_PROGRAM,
    "Singing Voice (Chorus)": SINGING_VOICE_CHORUS_PROGRAM,
    "Drums": DRUM_PROGRAM,
}

# yapf: disable
TRACK_NAME_FILTERS = {
    SINGING_VOICE_PROGRAM: {"include": ["MELO", "VOCAL"], "exclude": ["SUB", "GT"]},
    SINGING_VOICE_CHORUS_PROGRAM: {"include": ["CHORUS", "SUB VOCAL", "SUB MELO"],
                                   "exclude": ["/", "GT"]},
    DRUM_PROGRAM: {"include": ["DRUMS", "DR", "HIHAT", "BD&SD", "TOM", "KICK"],
                   "exclude": ["ATOMOS"], "exact": ["DS"]},
    0: {"include": ["P.F.", "PF", "PIANO", "A.P", "CLAV", "CEMBAL", "HARPSI"], "exclude": ["E.PIANO", "MARIMBA"]},
    2: {"include": ["E.P"], "exclude": []},
    8: {"include": ["GLOCKEN", "VIBRA", "VIBE", "MARIMBA", "BELL", "CHIME", "CHAIM", "KALIMB", "CHIMRE", "MALLET"],
        "exclude": []},
    16: {"include": ["ORG", "HAMO", "HARMONICA", "ACCORD"], "exclude": []},
    24: {"include": ["MANDORIN", "AG", "NYLON", "AC.G", "GUITAR", "A.G", "E.G", "GT", "G. SOLO", "CLEAN LEAD", "SITAR", "ATOMOS", "ATMOS",
                      "CLEAN"],
         "exclude": ["DIST", "DIS.", "D.", "E.G SOLO", "E.G.SOLO"]},
    30: {"include": ["OD L", "OD R", "DIS.", "DIST GT", "D.G", "DIST", "DIS.SOLO", "E.GUITAR (SOLO)", "E.G SOLO", "LEAD", "E.G.SOLO", "EG", "GT MELO"],
        "exclude": ["PAD","SYN.LEAD"]},
    33: {"include": ["BASS"], "exclude": []},
    48: {"include": ["OR 2", "ST", "STR", "ORCH", "PIZZ", "HIT", "TIMPANI", "VIORA", "VIOLA", "VIOLIN", "VN", "VA", "VC", "HARP", "LO FI", "CHO", "VLN", "CELLO"],
          "exclude": ["CHORUS", "HARPSI", "STEEL", "GUITAR", "PAD", "BRASS", "GT", "HORN"],
          "exact": ["OR"]},
    56: {"include": ["BRAS", "TRUMP", "TP", "TB", "TROM", "HORN", "FLUGEL"], "exclude": []},
    64: {"include": ["SAX", "OBOE", "BASS"], "exclude": ["SYNSAX"]},
    72: {"include": ["FLUTE", "PICO", "BOTTLE", "GAYA"], "exclude": []},
    80: {"include": ["S SOLO", "SYN SOLO", "SOLO SYNTH", "SYNTH SOLO", "SYN.LEAD", "SYNTH(SEQ)", "PORTASYN", "SQ", "SEQ", "VOICE"], "exclude": []},
    88: {"include": ["SYNTH", "SYN", "PAD", "FANTASIA", "BRIGHTNESS", "FANTASY"], "exclude": ["SYNBELL", "PORTA", "SOLO", "SEQ", "LEAD", "ORGAN", "BRAS", "BASS", "TROM"]},
    None: {"include": ["INTRO SE", "WOW", "PERC", "EXC", "REVERSE", "GONG", "PER.", "RAP", "REV", "S.E", "LASER",
                        "LESER", "TAMBOURINE", "KANE", "PER", "SHAKER", "RWC-MDB"],
           "exclude": [],
           "exact": ["SE", "EX", "808", "ICERAIN"]},
    "USE RWC PROGRAM MAP": {"include": ["KIRA", "KILA", "ETHNIC&GK"], "exclude": [], "exact": ["FUE", "OU-01A"]},
}
# yapf: enable
RWC_PROGRAM_MAP = {
    9: 8,
    11: 8,
    74: 72,
    94: 80,
    98: 88,
    100: 88,
}

PRG2CH = {
    0: (0, "Acoustic Piano"),
    2: (1, "Electric Piano"),
    8: (2, "Chromatic Percussion"),
    16: (3, "Organ"),
    24: (4, "Guitar (clean)"),
    30: (5, "Guitar (distortion)"),
    33: (6, "Bass"),
    48: (7, "Strings"),
    56: (8, "Brass"),
    DRUM_PROGRAM: (9, "Drums"),
    64: (10, "Reed"),
    72: (11, "Pipe"),
    80: (12, "Synth Lead"),
    88: (13, "Synth Pad"),
    SINGING_VOICE_PROGRAM: (14, "Singing Voice"),
    SINGING_VOICE_CHORUS_PROGRAM: (15, "Singing Voice (Chorus)"),
}


def find_matching_filters(input_text, filters):
    input_text = input_text.upper()

    def text_matches_filter(text, filter_dict):
        matchness = False
        if "exact" in filter_dict:
            for keyword in filter_dict["exact"]:
                if keyword == text:
                    matchness = True
                    break
        for keyword in filter_dict["include"]:
            if keyword in text:
                matchness = True
                break
        for keyword in filter_dict["exclude"]:
            if keyword in text:
                matchness = False
                break
        return matchness

    matching_filters = []
    for filter_name, filter_dict in filters.items():
        if text_matches_filter(input_text, filter_dict):
            matching_filters.append(filter_name)
    return matching_filters


def generate_corrected_midi(org_mid_file: os.PathLike,
                            new_mid_file: os.PathLike,
                            filters: Dict[Any, Dict[str, List]],
                            prg2ch: Dict[int, Tuple[int, str]]):
    # Load original MIDI file
    org_mid = MidiFile(org_mid_file)

    # Create a new MIDI file
    new_mid = MidiFile(ticks_per_beat=org_mid.ticks_per_beat)

    # Extract global messages from the first track (usually the master track)
    global_messages = [msg for msg in org_mid.tracks[0] if msg.is_meta]
    global_track = MidiTrack(global_messages)
    new_mid.tracks.append(global_track)

    # Loop over all tracks
    for track in org_mid.tracks[1:]:
        # Get track name
        track_name = None
        for msg in track:
            if msg.type == 'track_name':
                track_name = msg.name
                break
        if track_name is None:
            raise ValueError('track name not found in midi file')

        # Get program number from track name
        matching_filters = find_matching_filters(track_name, filters)
        assert (len(matching_filters) != 0)
        if isinstance(matching_filters[0], int):
            program = matching_filters[0]
        elif matching_filters[0] == "USE RWC PROGRAM MAP":
            for msg in track:
                if msg.type == 'program_change':
                    program = RWC_PROGRAM_MAP.get(msg.program, msg.program)
                    break
        elif matching_filters[0] == None:
            continue

        # Get channel and new track name
        ch, new_track_name = prg2ch[program]

        # Copy messages to new track with new program, channel, and track_name
        new_track = MidiTrack()
        new_track.append(MetaMessage('track_name', name=new_track_name,
                                     time=0))
        if program == DRUM_PROGRAM:
            new_track.append(
                Message('program_change', program=0, time=0, channel=9))
        else:
            new_track.append(
                Message('program_change', program=program, time=0, channel=ch))
        new_mid.tracks.append(new_track)

        for msg in track:
            if msg.type in ['track_name', 'instrument_name', 'program_change']:
                continue
            else:
                new_msg = msg.copy()
                if hasattr(msg, 'channel'):
                    new_msg.channel = ch
                new_track.append(new_msg)

    # Save new MIDI file
    new_mid.save(new_mid_file)
    print(f'Created {new_mid_file}')


def check_file_existence(file: str) -> bool:
    """Checks if file exists."""
    res = True
    if not os.path.exists(file):
        res = False
    elif get_audio_file_info(file)[1] < 10 * 16000:
        print(f'File {file} is too short.')
        res = False
    return res


def create_note_event_and_note_from_midi(
        mid_file: str,
        id: str,
        ch_9_as_drum: bool = False,
        track_name_to_program: Dict = None,
        ignore_pedal: bool = False) -> Tuple[Dict, Dict]:
    """Create note_events and notes from midi file."""

    # Load midi file
    notes, dur_sec, program = midi2note(
        mid_file,
        ch_9_as_drum=ch_9_as_drum,
        track_name_to_program=track_name_to_program,
        binary_velocity=True,
        ignore_pedal=ignore_pedal,
        return_programs=True)
    program = [x for x in set(program)
               if x is not None]  # remove None and duplicates
    return { # notes
        'rwc_pop_id': id,
        'program': program,
        'is_drum': [1 if p == DRUM_PROGRAM else 0 for p in program],
        'duration_sec': dur_sec,
        'notes': notes,
    }, { # note_events
        'rwc_pop_id': id,
        'program': program,
        'is_drum': [1 if p == DRUM_PROGRAM else 0 for p in program],
        'duration_sec': dur_sec,
        'note_events': note2note_event(notes),
    }


def preprocess_rwc_pop_full16k(data_home='../../data',
                               dataset_name='rwc_pop') -> None:
    # Directory and file paths
    base_dir = os.path.join(data_home, dataset_name + '_yourmt3_16k')
    output_index_dir = os.path.join(data_home, 'yourmt3_indexes')
    os.makedirs(output_index_dir, exist_ok=True)

    # Load CSV: construct id to midi/wav dictionary
    csv_file = os.path.join(base_dir, 'wav_to_midi_filename_mapping.csv')
    rwc_all = {}
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)

        for row in reader:
            id = row[2]
            mix_audio_file = os.path.join(base_dir, headers[0] + row[0],
                                          row[1] + ' ' + headers[1] + '.wav')
            assert check_file_existence(mix_audio_file)
            mid_file = os.path.join(base_dir, 'MIDI', id + '.mid')
            assert os.path.exists(mid_file)
            notes_file = mid_file.replace('.mid', '_notes.npy')
            note_events_file = mid_file.replace('.mid', '_note_events.npy')

            rwc_all[id] = {
                'rwc_pop_id': id,
                'n_frames': get_audio_file_info(mix_audio_file)[1],
                'mix_audio_file': mix_audio_file,
                'notes_file': notes_file,
                'note_events_file': note_events_file,
                'midi_file': mid_file,
                'program': None,
                'is_drum': None,
            }
    assert len(rwc_all) == 100

    # Generate corrected MIDI files by reassigning program numbers
    os.makedirs(os.path.join(base_dir, 'MIDI_full_corrected'), exist_ok=True)
    for id, info in rwc_all.items():
        org_mid_file = info['midi_file']
        new_mid_file = org_mid_file.replace('/MIDI/', '/MIDI_full_corrected/')
        generate_corrected_midi(org_mid_file,
                                new_mid_file,
                                filters=TRACK_NAME_FILTERS,
                                prg2ch=PRG2CH)
        # Update file path with corrected MIDI file
        rwc_all[id]['midi_file'] = new_mid_file
        rwc_all[id]['notes_file'] = new_mid_file.replace('.mid', '_notes.npy')
        rwc_all[id]['note_events_file'] = new_mid_file.replace(
            '.mid', '_note_events.npy')

    # Unused ids
    for id in UNUSED_IDS:
        rwc_all.pop(str(id))
    print(f'Number of used IDs: {len(rwc_all)}, Unused ids: {UNUSED_IDS}')

    # Create note and note_event files
    for id in rwc_all.keys():
        midi_file = rwc_all[id]['midi_file']
        notes_file = rwc_all[id]['notes_file']
        note_events_file = rwc_all[id]['note_events_file']

        # Create note and note_event files
        notes, note_events = create_note_event_and_note_from_midi(
            midi_file,
            id,
            ch_9_as_drum=False,  # we will use track_name_to_program instead
            track_name_to_program=TRACK_NAME_TO_PROGRAM_MAP,
            ignore_pedal=False)

        # Update programs and is_drum
        rwc_all[id]['program'] = notes['program']
        rwc_all[id]['is_drum'] = notes['is_drum']

        # Save note and note_event files
        np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
        print(f'Created {notes_file}')
        np.save(note_events_file,
                note_events,
                allow_pickle=True,
                fix_imports=False)
        print(f'Created {note_events_file}')

    # Save index file
    split = 'full'
    output_index_file = os.path.join(output_index_dir,
                                     f'rwc_pop_{split}_file_list.json')

    file_list = {}
    for i, id in enumerate(rwc_all.keys()):
        file_list[i] = rwc_all[id]

    with open(output_index_file, 'w') as f:
        json.dump(file_list, f, indent=4)
    print(f'Created {output_index_file}')
