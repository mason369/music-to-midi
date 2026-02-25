# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""midi.py

MIDI <-> Note
• midi2note: convert a MIDI file to a list of Note instances.
• note2midi: convert a list of Note instances to a MIDI file.

"""
import os
import copy
import warnings
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from mido import MetaMessage, Message, MidiFile, MidiTrack, second2tick
from utils.note_event_dataclasses import Note, NoteEvent
from utils.note2event import validate_notes, trim_overlapping_notes
from utils.note2event import note2note_event
""" midi2note:
Convert a MIDI file to a list of Note instances.

About new implementation:

 The widely used MIDI parsers (implementations from pretty_midi, 
onset_and_frames, reconvat, and mir_data) implementations used a method of 
applying the offset to the nearest previous note when note overlaps occurred.
 
 We often found issues with this lazy-processing approach, where the length of 
the overlapped notes later in the sequence would become extremely short. 

 This code has been re-implemented to address these issues by keeping note 
activations in channel-specific buffers, similar to actual DAWs,  
allowing for the application of the sustain pedal effect in multi-channel
tracks. 

Example from Slkah,'Track00805-S00' (bass stem):

(onset, offset)

<actual midi>
(8.83, 9.02*) * first note's offset is later than second note's onset, so overlap occurs.
(9.0, 9.55)

<pretty_midi & mir_data parser>
(8.83, 9.0)
(9.0, 9.02*) * second note is too short, because first note's offset is applied to second note.

<onset_and_frames & reconvat parser>
(8.83, 8.84*) * due to reverse search, first note's offset is missing, so minimum offset is applied.
(9.0, 9.55) 

<your_mt3 parser>
(8.83, 9.0) 
(9.0, 9.55)

"""
DRUM_PROGRAM = 128


def find_channel_of_track_name(midi_file: os.PathLike, track_name_keywords: List[str]) -> Optional[int]:
    mid = MidiFile(midi_file)
    found_channels = []

    for track in mid.tracks:
        track_name_found = False
        for msg in track:
            if msg.type == 'track_name':
                for k in track_name_keywords:
                    if k.lower() == msg.name.lower():  # exact match only
                        track_name_found = True
                        break

            if track_name_found and msg.type in ['note_on', 'note_off']:
                found_channels.append(msg.channel)
                break

    return list(set(found_channels))


def midi2note(file: Union[os.PathLike, str],
              binary_velocity: bool = True,
              ch_9_as_drum: bool = False,
              force_all_drum: bool = False,
              force_all_program_to: Optional[int] = None,
              track_name_to_program: Optional[Dict] = None,
              trim_overlap: bool = True,
              fix_offset: bool = True,
              quantize: bool = True,
              verbose: int = 0,
              minimum_offset_sec: float = 0.01,
              drum_offset_sec: float = 0.01,
              ignore_pedal: bool = False,
              return_programs: bool = False) -> Tuple[List[Note], float]:
    midi = MidiFile(file)
    max_time = midi.length  # in seconds

    finished_notes = []
    program_state = [None] * 16  # program_number = program_state[ch]
    sustain_state = [None] * 16  # sustain_state[ch] = True if sustain is on
    active_notes = [[] for i in range(16)]  # active notes by channel(0~15). active_notes[ch] = [Note1, Note_2,..]
    sustained_notes = [[] for i in range(16)
                      ]  # offset is passed, but sustain is applied. sustained_notes[ch] = [Note1, Note_2,..]

    # Mapping track name to program (for geerdes data)
    reserved_channels = []
    if track_name_to_program is not None:
        for key in track_name_to_program.keys():
            found_channels = find_channel_of_track_name(file, [key])
            if len(found_channels) > 0:
                for ch in found_channels:
                    program_state[ch] = track_name_to_program[key]
                    reserved_channels.append(ch)
    if ch_9_as_drum is True:
        program_state[9] = DRUM_PROGRAM
        reserved_channels.append(9)

    current_time = 0.
    for i, msg in enumerate(midi):
        current_time += msg.time
        if msg.type == 'program_change' and msg.channel not in reserved_channels:
            program_state[msg.channel] = msg.program
        elif msg.type == 'control_change' and msg.control == 64 and not ignore_pedal:
            if msg.value >= 64:
                sustain_state[msg.channel] = True
            else:
                sustain_state[msg.channel] = False
                for note in sustained_notes[msg.channel]:
                    note.offset = current_time
                    finished_notes.append(note)
                sustained_notes[msg.channel] = []
        elif msg.type == 'note_on' and msg.velocity > 0:
            if program_state[msg.channel] == None:
                if force_all_program_to == None:
                    raise ValueError(
                        '📕 midi2note: program_change message is missing. Use `force_all_program_to` option')
                else:
                    program_state[msg.channel] = force_all_program_to
            # if (ch_9_as_drum and msg.channel == 9) or force_all_drum:
            if program_state[msg.channel] == DRUM_PROGRAM or force_all_drum:
                # drum's offset, active_notes, sustained_notes are not tracked.
                new_note = Note(is_drum=True,
                                program=program_state[msg.channel],
                                onset=current_time,
                                offset=current_time + drum_offset_sec,
                                pitch=msg.note,
                                velocity=msg.velocity)
                finished_notes.append(new_note)
            else:
                new_note = Note(is_drum=False,
                                program=program_state[msg.channel],
                                onset=current_time,
                                offset=None,
                                pitch=msg.note,
                                velocity=msg.velocity)
                active_notes[msg.channel].append(new_note)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            temp_active_notes = active_notes.copy()
            offset_done_flag = False
            for note in active_notes[msg.channel]:
                if note.pitch == msg.note:
                    if sustain_state[msg.channel]:
                        sustained_notes[msg.channel].append(note)
                        temp_active_notes[msg.channel].remove(note)
                    elif offset_done_flag == False:
                        note.offset = current_time
                        finished_notes.append(note)
                        temp_active_notes[msg.channel].remove(note)
                        offset_done_flag = True
                        # fix: note_off message is only for the oldest note_on message
                    else:
                        pass
            active_notes = temp_active_notes

    # Handle any still-active notes (e.g., if the file ends without note_off messages)
    for ch_notes in active_notes:
        for note in ch_notes:
            note.offset = min(current_time, note.onset + minimum_offset_sec)
            finished_notes.append(note)
    for ch_notes in sustained_notes:
        for note in ch_notes:
            note.offset = min(current_time, note.onset + minimum_offset_sec)
            finished_notes.append(note)

    notes = finished_notes

    if binary_velocity:
        for note in notes:
            note.velocity = 1 if note.velocity > 0 else 0

    notes.sort(key=lambda note: (note.onset, note.is_drum, note.program, note.velocity, note.pitch))

    # Quantize notes to 10 ms
    if quantize:
        for note in notes:
            note.onset = round(note.onset * 100) / 100.
            note.offset = round(note.offset * 100) / 100.

    # Trim overlapping notes
    if trim_overlap:
        notes = trim_overlapping_notes(notes, sort=True)

    # fix offset >= onset the Note instances
    if fix_offset:
        notes = validate_notes(notes, fix=True)

    # Print some statistics
    has_drum = False
    for note in notes:
        if note.is_drum:
            has_drum = True
            break
    num_instr = sum([int(c is not None) for c in program_state])
    if verbose > 0:
        print(
            f'parsed {file}: midi_type={midi.type}, num_notes={len(notes)}, num_instr={num_instr}, has_drum={has_drum}')
    if return_programs:
        return notes, max_time, program_state
    else:
        return notes, max_time


def note_event2midi(note_events: List[NoteEvent],
                    output_file: Optional[os.PathLike] = None,
                    velocity: int = 100,
                    ticks_per_beat: int = 480,
                    tempo: int = 500000,
                    singing_program_mapping: int = 65,
                    singing_chorus_program_mapping: int = 53,
                    output_inverse_vocab: Optional[Dict] = None) -> None:
    """Converts a list of Note instances to a MIDI file.

    List[NoteEvent]: 
        [NoteEvent(is_drum: bool, program: int, time: Optional[float], velocity: int,
         pitch: int, activity: Optional[Set[int]] = {<factory>})
        
    Example usage:

        note_event2midi(note_events, 'output.mid')

    """
    midi = MidiFile(ticks_per_beat=ticks_per_beat, type=0)
    midi.type = 1
    track = MidiTrack()
    midi.tracks.append(track)

    # Set tempo
    # track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    # Assign channels to programs
    programs = set()
    for ne in note_events:
        if ne.program == 128 or ne.is_drum == True:
            programs.add(128)  # 128 represents drum here...
            ne.program = 128  # internally we use 128 for drum
        else:
            programs.add(ne.program)
    programs = sorted(programs)

    program_to_channel = {}
    available_channels = list(range(0, 9)) + list(range(10, 16))
    for prg in programs:
        if prg == 128:
            program_to_channel[prg] = 9
        else:
            try:
                program_to_channel[prg] = available_channels.pop(0)
            except IndexError:
                warnings.warn(f'not available channels for program {prg}, share channel 16')
                program_to_channel[prg] = 15

    # notes to note_events (this is simpler)
    drum_offset_events = []  # for drum notes, we need to add an offset event
    for ne in note_events:
        if ne.is_drum:
            drum_offset_events.append(
                NoteEvent(is_drum=True, program=ne.program, time=ne.time + 0.01, pitch=ne.pitch, velocity=0))
    note_events += drum_offset_events
    note_events.sort(key=lambda ne: (ne.time, ne.is_drum, ne.program, ne.velocity, ne.pitch))

    # Add note events to multitrack
    for program in programs:
        # Create a track for each program
        track = MidiTrack()
        midi.tracks.append(track)

        # Add track name
        if program == 128:
            program_name = 'Drums'
        elif output_inverse_vocab is not None:
            program_name = output_inverse_vocab.get(program, (program, f'Prg. {str(program)}'))[1]
        else:
            program_name = f'Prg. {str(program)}'
        track.append(MetaMessage('track_name', name=program_name, time=0))

        # Channel is determined by the program
        channel = program_to_channel[program]

        # Some special treatment for singing voice and drums
        if program == 128:  # drum
            # set 0 but it is ignored in drum channel
            track.append(Message('program_change', program=0, time=0, channel=channel))
        elif program == 100:  # singing voice --> Alto Sax
            track.append(Message('program_change', program=singing_program_mapping, time=0, channel=channel))
        elif program == 101:  # singing voice (chrous) --> Voice Oohs
            track.append(Message('program_change', program=singing_chorus_program_mapping, time=0, channel=channel))
        else:
            track.append(Message('program_change', program=program, time=0, channel=channel))

        current_tick = int(0)
        for ne in note_events:
            if ne.program == program:
                absolute_tick = round(second2tick(ne.time, ticks_per_beat, tempo))
                if absolute_tick == current_tick:
                    delta_tick = int(0)
                elif absolute_tick < current_tick:
                    # this should not happen after sorting
                    raise ValueError(
                        f'at ne.time {ne.time}, absolute_tick {absolute_tick} < current_tick {current_tick}')
                else:
                    # Convert time shift value from seconds to ticks
                    delta_tick = absolute_tick - current_tick
                    current_tick += delta_tick

                # Create a note on or note off message
                msg_note = 'note_on' if ne.velocity > 0 else 'note_off'
                msg_velocity = velocity if ne.velocity > 0 else 0
                new_msg = Message(msg_note, note=ne.pitch, velocity=msg_velocity, time=delta_tick, channel=channel)

                track.append(new_msg)

    # Save MIDI file
    if output_file != None:
        midi.save(output_file)


def get_pitch_range_from_midi(midi_file: os.PathLike) -> Tuple[int, int]:
    """Returns the pitch range of a MIDI file.

    Args:
        midi_file (os.PathLike): Path to a MIDI file.

    Returns:
        Tuple[int, int]: The lowest and highest notes in the MIDI file.
    """
    notes = midi2note(midi_file, quantize=False, trim_overlap=False)
    pitches = [n.pitch for n in notes]
    return min(pitches), max(pitches)


def pitch_shift_midi(src_midi_file: os.PathLike,
                     min_pitch_shift: int = -5,
                     max_pitch_shift: int = 6,
                     write_midi_file: bool = True,
                     write_notes_file: bool = True,
                     write_note_events_file: bool = True) -> None:
    """Pitch shifts a MIDI file and write it as MIDI.

    Args:
        src_midi_file (os.PathLike): Path to a MIDI file.
        min_pitch_shift (int): The number of semitones to shift.
        max_pitch_shift (int): The number of semitones to shift.

    Writes:
        dst_midi_file (os.PathLike): {src_midi_filename}_pshift_{i}.mid, where i can be [...,-1, 1, 2,...]
        dst_notes : List[Note]
        dst_note_events: List[NoteEvent]
    """
    # source file
    src_midi_dir = os.path.dirname(src_midi_file)
    src_midi_filename = os.path.basename(src_midi_file).split('.')[0]
    src_notes_file = os.path.join(src_midi_dir, f'{src_midi_filename}_notes.npy')
    src_note_events_file = os.path.join(src_midi_dir, f'{src_midi_filename}_note_events.npy')
    src_notes, _ = midi2note(src_midi_file)
    # src_note_events = note2note_event(src_notes)

    for pitch_shift in range(min_pitch_shift, max_pitch_shift):
        if pitch_shift == 0:
            continue

        # destination file
        dst_midi_file = os.path.join(src_midi_dir, f'{src_midi_filename}_pshift{pitch_shift}.mid')
        dst_notes_file = os.path.join(src_midi_dir, f'{src_midi_filename}_pshift{pitch_shift}_notes.npy')
        dst_note_events_file = os.path.join(src_midi_dir, f'{src_midi_filename}_pshift{pitch_shift}_note_events.npy')

        dst_notes = []
        for note in src_notes:
            dst_note = copy.deepcopy(note)
            dst_note.pitch += pitch_shift
            dst_notes.append(dst_note)

        dst_note_events = note2note_event(dst_notes)

        # write midi file
        if write_midi_file:
            note_event2midi(dst_note_events, dst_midi_file)
            print(f'Created {dst_midi_file}')

        # write notes file
        if write_notes_file:
            # get metadata for notes
            src_notes_metadata = np.load(src_notes_file, allow_pickle=True).tolist()
            dst_notes_metadata = src_notes_metadata
            dst_notes_metadata['pitch_shift'] = pitch_shift
            dst_notes_metadata['notes'] = dst_notes
            np.save(dst_notes_file, dst_notes_metadata, allow_pickle=True, fix_imports=False)
            print(f'Created {dst_notes_file}')

        # write note events file
        if write_note_events_file:
            # get metadata for note events
            src_note_events_metadata = np.load(src_note_events_file, allow_pickle=True).tolist()
            dst_note_events_metadata = src_note_events_metadata
            dst_note_events_metadata['pitch_shift'] = pitch_shift
            dst_note_events_metadata['note_events'] = dst_note_events
            np.save(dst_note_events_file, dst_note_events_metadata, allow_pickle=True, fix_imports=False)
            print(f'Created {dst_note_events_file}')
