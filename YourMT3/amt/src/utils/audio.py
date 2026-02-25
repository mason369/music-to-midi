# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""audio.py"""
import os
import subprocess
import numpy as np
import wave
import math
from typing import Tuple, List
from numpy.lib.stride_tricks import as_strided


def load_audio_file(filename: str,
                    seg_start_sec: float = 0.,
                    seg_length_sec: float = 0.,
                    fs: int = 16000,
                    dtype: np.dtype = np.float64) -> np.ndarray:
    """Load audio file and return the segment of audio."""
    start_frame_idx = int(np.floor(seg_start_sec * fs))
    seg_length_frame = int(np.floor(seg_length_sec * fs))
    end_frame_idx = start_frame_idx + seg_length_frame

    file_ext = filename[-3:]

    if file_ext == 'wav':
        with wave.open(filename, 'r') as f:
            f.setpos(start_frame_idx)
            if seg_length_sec == 0:
                x = f.readframes(f.getnframes())
            else:
                x = f.readframes(end_frame_idx - start_frame_idx)

            if dtype == np.float64:
                x = np.frombuffer(x, dtype=np.int16) / 2**15
            elif dtype == np.float32:
                x = np.frombuffer(x, dtype=np.int16) / 2**15
                x = x.astype(np.float32)
            elif dtype == np.int16:
                x = np.frombuffer(x, dtype=np.int16)
            elif dtype is None:
                pass
            else:
                raise NotImplementedError(f"Unsupported dtype: {dtype}")
    else:
        raise NotImplementedError(f"Unsupported file extension: {file_ext}")

    return x


def get_audio_file_info(filename: str) -> Tuple[int, int, int]:
    """Get audio file info.
    
    Args:
        filename: path to the audio file
    Returns:
        fs: sampling rate
        n_frames: number of frames
        n_channels: number of channels
        
    """
    file_ext = filename[-3:]

    if file_ext == 'wav':
        with wave.open(filename, 'r') as f:
            fs = f.getframerate()
            n_frames = f.getnframes()
            n_channels = f.getnchannels()
    else:
        raise NotImplementedError(f"Unsupported file extension: {file_ext}")

    return fs, n_frames, n_channels


def get_segments_from_numpy_array(arr: np.ndarray,
                                  slice_length: int,
                                  start_frame_indices: List[int],
                                  dtype: np.dtype = np.float32) -> np.ndarray:
    """Get random audio slices from numpy array.
    
    Args:
        arr: numpy array of shape (c, n_frames)
        slice_length: length of the slice
        start_frame_indices: list of m start frames
    Returns:
        slices: numpy array of shape (m, c, slice_length)
    """
    c, max_length = arr.shape
    max_length = arr.shape[1]
    m = len(start_frame_indices)

    slices = np.zeros((m, c, slice_length), dtype=dtype)
    for i, start_frame in enumerate(start_frame_indices):
        end_frame = start_frame + slice_length
        assert (end_frame <= max_length - 1)
        slices[i, :, :] = arr[:, start_frame:end_frame].astype(dtype)
    return slices


def slice_padded_array(x: np.ndarray, slice_length: int, slice_hop: int, pad: bool = True) -> np.ndarray:
    """
    Slices the input array into overlapping windows based on the given slice length and slice hop.

    Args:
        x: The input array to be sliced.
        slice_length: The length of each slice.
        slice_hop: The number of elements between the start of each slice.
        pad: If True, the last slice will be padded with zeros if necessary.

    Returns:
        A numpy array with shape (n_slices, slice_length) containing the slices.
    """
    num_slices = (x.shape[1] - slice_length) // slice_hop + 1
    remaining = (x.shape[1] - slice_length) % slice_hop

    if pad and remaining > 0:
        padding = np.zeros((x.shape[0], slice_length - remaining))
        x = np.hstack((x, padding))
        num_slices += 1

    shape: Tuple[int, int] = (num_slices, slice_length)
    strides: Tuple[int, int] = (slice_hop * x.strides[1], x.strides[1])
    sliced_x = as_strided(x, shape=shape, strides=strides)

    return sliced_x


def slice_padded_array_for_subbatch(x: np.ndarray,
                                    slice_length: int,
                                    slice_hop: int,
                                    pad: bool = True,
                                    sub_batch_size: int = 1,
                                    dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Slices the input array into overlapping windows based on the given slice length and slice hop,
    and pads it to make the output divisible by the sub_batch_size.

    NOTE: This method is currently not used.
    
    Args:
        x: The input array to be sliced, such as (1, n_frames).
        slice_length: The length of each slice.
        slice_hop: The number of elements between the start of each slice.
        pad: If True, the last slice will be padded with zeros if necessary.
        sub_batch_size: The desired number of slices to be divisible by.

    Returns:
        A numpy array with shape (n_slices, slice_length) containing the slices.
    """
    num_slices = (x.shape[1] - slice_length) // slice_hop + 1
    remaining = (x.shape[1] - slice_length) % slice_hop

    if pad and remaining > 0:
        padding = np.zeros((x.shape[0], slice_length - remaining), dtype=dtype)
        x = np.hstack((x, padding))
        num_slices += 1

    # Adjust the padding to make n_slices divisible by sub_batch_size
    if pad and num_slices % sub_batch_size != 0:
        additional_padding_needed = (sub_batch_size - (num_slices % sub_batch_size)) * slice_hop
        additional_padding = np.zeros((x.shape[0], additional_padding_needed), dtype=dtype)
        x = np.hstack((x, additional_padding))
        num_slices += (sub_batch_size - (num_slices % sub_batch_size))

    shape: Tuple[int, int] = (num_slices, slice_length)
    strides: Tuple[int, int] = (slice_hop * x.strides[1], x.strides[1])
    sliced_x = as_strided(x, shape=shape, strides=strides)

    return sliced_x


def pitch_shift_audio(src_audio_file: os.PathLike,
                      min_pitch_shift: int = -5,
                      max_pitch_shift: int = 6,
                      random_microshift_range: tuple[int, int] = (-10, 11)):
    """
    Pitch shift audio file using the Sox command-line tool.

    NOTE: This method is currently not used. Previously, we used this for 
    offline augmentation for GuitarSet.

    Args:
        src_audio_file: Path to the input audio file.
        min_pitch_shift: Minimum pitch shift in semitones.
        max_pitch_shift: Maximum pitch shift in semitones.
        random_microshift_range: Range of random microshifts to apply in tenths of a semitone.

    Returns:
        None

    Raises:
        CalledProcessError: If the Sox command fails to execute.

    """

    # files
    src_audio_dir = os.path.dirname(src_audio_file)
    src_audio_filename = os.path.basename(src_audio_file).split('.')[0]

    # load source audio
    try:
        audio = load_audio_file(src_audio_file, dtype=np.int16)
        audio = audio / 2**15
        audio = audio.astype(np.float16)
    except Exception as e:
        print(f"Failed to load audio file: {src_audio_file}. {e}")
        return

    # pitch shift audio for each semitone in the range
    for pitch_shift in range(min_pitch_shift, max_pitch_shift):
        if pitch_shift == 0:
            continue

        # pitch shift audio by sox
        dst_audio_file = os.path.join(src_audio_dir, f'{src_audio_filename}_pshift{pitch_shift}.wav')
        shift_semitone = 100 * pitch_shift + np.random.randint(*random_microshift_range)

        # build Sox command
        command = ['sox', src_audio_file, '-r', '16000', dst_audio_file, 'pitch', str(shift_semitone)]

        try:
            # execute Sox command and check for errors
            subprocess.run(command, check=True)
            print(f"Created {dst_audio_file}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to pitch shift audio file: {src_audio_file}, pitch_shift: {pitch_shift}. {e}")


def write_wav_file(filename: str, x: np.ndarray, samplerate: int = 16000) -> None:
    """
    Write a mono PCM WAV file from a NumPy array of audio samples.

    Args:
        filename (str): The name of the WAV file to be created.
        x (np.ndarray): A 1D NumPy array containing the audio samples to be written to the WAV file. 
                        The audio samples should be in the range [-1, 1].
        samplerate (int): The sample rate (in Hz) of the audio samples.

    Returns:
        None
    """
    # Set the WAV file parameters
    nchannels = 1  # Mono
    sampwidth = 2  # 16-bit
    framerate = samplerate
    nframes = len(x)

    # Scale the audio samples to the range [-32767, 32767]
    x_scaled = np.array(x * 32767, dtype=np.int16)

    # Set the buffer size for writing the WAV file
    BUFFER_SIZE = 1024

    # Open the WAV file for writing
    with wave.open(filename, "wb") as wav_file:
        # Set the WAV file parameters
        wav_file.setparams((nchannels, sampwidth, framerate, nframes, "NONE", "NONE"))

        # Write the audio samples to the file in chunks
        for i in range(0, len(x_scaled), BUFFER_SIZE):
            # Get the next chunk of audio samples
            chunk = x_scaled[i:i + BUFFER_SIZE]

            # Convert the chunk of audio samples to a byte string and write it to the WAV file
            wav_file.writeframes(chunk.tobytes())

    # Close the WAV file
    wav_file.close()


def guess_onset_offset_by_amp_envelope(x, fs=16000, onset_threshold=0.05, offset_threshold=0.02, frame_size=256):
    """ Guess onset/offset from audio signal x """
    amp_env = []
    num_frames = math.floor(len(x) / frame_size)
    for t in range(num_frames):
        lower = t * frame_size
        upper = (t + 1) * frame_size - 1
        # Find maximum of each frame and add it to our array
        amp_env.append(np.max(x[lower:upper]))
    amp_env = np.array(amp_env)
    # Find the first index where the amplitude envelope is greater than the threshold
    onset = np.where(amp_env > onset_threshold)[0][0] * frame_size
    offset = (len(amp_env) - 1 - np.where(amp_env[::-1] > offset_threshold)[0][0]) * frame_size
    return onset, offset, amp_env


# from pydub import AudioSegment
# def convert_flac_to_wav(input_path, output_path):
#     # Load FLAC file using Pydub
#     sound = AudioSegment.from_file(input_path, format="flac")

#     # Set the parameters for the output WAV file
#     channels = 1  # mono
#     sample_width = 2  # 16-bit
#     frame_rate = 16000

#     # Convert the input sound to the specified format
#     sound = sound.set_frame_rate(frame_rate)
#     sound = sound.set_channels(channels)
#     sound = sound.set_sample_width(sample_width)

#     # Save the output WAV file to the specified path
#     sound.export(output_path, format="wav")
