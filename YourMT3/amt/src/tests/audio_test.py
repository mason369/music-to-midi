# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""audio_test.py"""
import unittest
import os
import numpy as np
import wave
import tempfile
from utils.audio import load_audio_file
from utils.audio import get_audio_file_info
from utils.audio import slice_padded_array
from utils.audio import slice_padded_array_for_subbatch
from utils.audio import write_wav_file


class TestLoadAudioFile(unittest.TestCase):

    def create_temp_wav_file(self, duration: float, fs: int = 16000) -> str:
        n_samples = int(duration * fs)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filename = temp_file.name

        data = np.random.randint(-2**15, 2**15, n_samples, dtype=np.int16)

        with wave.open(temp_filename, 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(fs)
            f.writeframes(data.tobytes())

        return temp_filename

    def test_load_audio_file(self):
        duration = 3.0
        fs = 16000
        temp_filename = self.create_temp_wav_file(duration, fs)

        # Test load entire file
        audio_data = load_audio_file(temp_filename, dtype=np.int16)
        file_fs, n_frames, n_channels = get_audio_file_info(temp_filename)

        self.assertEqual(len(audio_data), n_frames)
        self.assertEqual(file_fs, fs)
        self.assertEqual(n_channels, 1)

        # Test load specific segment
        seg_start_sec = 1.0
        seg_length_sec = 1.0
        audio_data = load_audio_file(temp_filename, seg_start_sec, seg_length_sec, dtype=np.int16)

        self.assertEqual(len(audio_data), int(seg_length_sec * fs))

        # Test unsupported file extension
        with self.assertRaises(NotImplementedError):
            load_audio_file("unsupported.xyz")


class TestSliceArray(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randint(0, 10, size=(1, 10000))

    def test_without_padding(self):
        sliced_x = slice_padded_array(self.x, slice_length=100, slice_hop=50, pad=False)
        self.assertEqual(sliced_x.shape, (199, 100))

    def test_with_padding(self):
        sliced_x = slice_padded_array(self.x, slice_length=100, slice_hop=50, pad=True)
        self.assertEqual(sliced_x.shape, (199, 100))

    def test_content(self):
        sliced_x = slice_padded_array(self.x, slice_length=100, slice_hop=50, pad=True)
        for i in range(sliced_x.shape[0] - 1):
            np.testing.assert_array_equal(sliced_x[i, :], self.x[:, i * 50:i * 50 + 100].flatten())
        # Test the last slice separately to account for potential padding
        last_slice = sliced_x[-1, :]
        last_slice_no_padding = self.x[:, -100:].flatten()
        np.testing.assert_array_equal(last_slice[:len(last_slice_no_padding)], last_slice_no_padding)


class TestSlicePadForSubbatch(unittest.TestCase):

    def test_slice_padded_array_for_subbatch(self):
        input_array = np.random.randn(6, 10)
        slice_length = 4
        slice_hop = 2
        pad = True
        sub_batch_size = 4

        expected_output_shape = (4, 4)

        # Call the slice_pad_for_subbatch function
        result = slice_padded_array_for_subbatch(input_array, slice_length, slice_hop, pad, sub_batch_size)

        # Check if the output shape is correct
        self.assertEqual(result.shape, expected_output_shape)

        # Check if the number of slices is divisible by sub_batch_size
        self.assertEqual(result.shape[0] % sub_batch_size, 0)


class TestWriteWavFile(unittest.TestCase):

    def test_write_wav_file_z(self):
        # Generate some test audio data
        samplerate = 16000
        duration = 1  # 1 second
        t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
        x = np.sin(2 * np.pi * 440 * t)

        # Write the test audio data to a WAV file
        filename = "extras/test.wav"
        write_wav_file(filename, x, samplerate)

        # Read the written WAV file and check its contents
        with wave.open(filename, "rb") as wav_file:
            # Check the WAV file parameters
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), samplerate)
            self.assertEqual(wav_file.getnframes(), len(x))

            # Read the audio samples from the WAV file
            data = wav_file.readframes(len(x))

        # Convert the audio sample byte string to a NumPy array and normalize it to the range [-1, 1]
        x_read = np.frombuffer(data, dtype=np.int16) / 32767.0

        # Check that the audio samples read from the WAV file are equal to the original audio samples
        np.testing.assert_allclose(x_read, x, atol=1e-4)

        # Delete the written WAV file
        os.remove(filename)


if __name__ == '__main__':
    unittest.main()
