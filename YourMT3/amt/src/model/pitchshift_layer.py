# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""pitchshift.py"""
# import math
import numpy as np
# from scipy import special
from einops import rearrange
from typing import Optional, Literal, Dict, List, Tuple, Callable

import torch
from torch import nn
import torchaudio
from torchaudio import transforms
# from torchaudio import functional as F
# from torchaudio.functional.functional import (
#     _fix_waveform_shape,
#     _stretch_waveform,
# )
# from model.ops import adjust_b_to_gcd, check_all_elements_equal


class PitchShiftLayer(nn.Module):
    """Applying batch-wise pitch-shift to time-domain audio signals.

    Args:
        pshift_range (List[int]): Range of pitch shift in semitones. Default: ``[-2, 2]``.
        resample_source_fs (int): Default is 4000.
        stretch_n_fft (int): Default is 2048.
        window: (Optional[Literal['kaiser']]) Default is None.
        beta: (Optional[float]): Parameter for 'kaiser' filter. Default: None.
    """

    def __init__(
        self,
        pshift_range: List[int] = [-2, 2],
        resample_source_fs: int = 4000,
        strecth_n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window: Optional[Literal['kaiser']] = None,
        beta: Optional[float] = None,
        expected_input_shape: Optional[Tuple[int]] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.pshift_range = pshift_range
        self.resample_source_fs = resample_source_fs
        self.strecth_n_fft = strecth_n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        if window is None:
            self.window_fn = torch.hann_window
            self.window_kwargs = None
        elif 'kaiser' in window:

            def custom_kaiser_window(window_length, beta, **kwargs):
                return torch.kaiser_window(window_length, periodic=True, beta=beta, **kwargs)

            self.window_fn = custom_kaiser_window
            self.window_kwargs = {'beta': beta}

        # Initialize pitch shifters for every semitone
        self.pshifters = None
        self.frame_gaps = None
        self._initialize_pshifters(expected_input_shape, device=device)
        self.requires_grad_(False)

    def _initialize_pshifters(self,
                              expected_input_shape: Optional[Tuple[int]] = None,
                              device: Optional[torch.device] = None) -> None:
        # DDP requires initializing parameters with a dummy input
        if expected_input_shape is not None:
            if device is not None:
                dummy_input = torch.randn(expected_input_shape, requires_grad=False).to(device)
            else:
                dummy_input = torch.randn(expected_input_shape, requires_grad=False)
        else:
            dummy_input = None

        pshifters = nn.ModuleDict()
        for semitone in range(self.pshift_range[0], self.pshift_range[1] + 1):
            if semitone == 0:
                # No need to shift and resample
                pshifters[str(semitone)] = None
            else:
                pshifter = transforms.PitchShift(self.resample_source_fs,
                                                 n_steps=semitone,
                                                 n_fft=self.strecth_n_fft,
                                                 win_length=self.win_length,
                                                 hop_length=self.hop_length,
                                                 window_fn=self.window_fn,
                                                 wkwargs=self.window_kwargs)
                pshifters[str(semitone)] = pshifter
                # Pass dummy input to initialize parameters
                with torch.no_grad():
                    if dummy_input is not None:
                        _ = pshifter.initialize_parameters(dummy_input)
        self.pshifters = pshifters

    def calculate_frame_gaps(self) -> Dict[int, float]:
        """Calculate the expected gap between the original and the stretched audio."""
        frame_gaps = {}  # for debugging
        for semitone in range(self.pshift_range[0], self.pshift_range[1] + 1):
            if semitone == 0:
                # No need to shift and resample
                frame_gaps[semitone] = 0.
            else:
                pshifter = self.pshifters[str(semitone)]
                gap_in_ms = 1000. * (pshifter.kernel.shape[2] -
                                     pshifter.kernel.shape[0] / 2.0**(-float(semitone) / 12)) / self.resample_source_fs
                frame_gaps[semitone] = gap_in_ms
        return frame_gaps

    @torch.no_grad()
    def forward(self, x: torch.Tensor, semitone: int) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, 1, T) or (B, T)
        Returns:
            torch.Tensor: (B, 1, T) or (B, T)
        """
        if semitone == 0:
            return x
        elif semitone >= min(self.pshift_range) and semitone <= max(self.pshift_range):
            return self.pshifters[str(semitone)](x)
        else:
            raise ValueError(f"semitone must be in range {self.pshift_range}")


def test_resampler_sinewave():
    # x: {440Hz, 220Hz} sine wave at 16kHz
    t = torch.arange(0, 2, 1 / 16000)  # 2 seconds at 16kHz
    x0 = torch.sin(2 * torch.pi * 440 * t) * 0.5
    x1 = torch.sin(2 * torch.pi * 220 * t) * 0.5
    x = torch.stack((x0, x1), dim=0)  # (2, 32000)

    # Resample
    psl = PitchShiftLayer(pshift_range=[-2, 2], resample_source_fs=4000)
    y = psl(x, 2)  # (2, 24000)

    # Export to wav
    torchaudio.save("x.wav", x, 16000, bits_per_sample=16)
    torchaudio.save("y.wav", y, 12000, bits_per_sample=16)


# class Resampler(nn.Module):
#     """
#     Resampling using conv1d operations, more memory-efficient than torchaudio's resampler.

#     Based on Dan Povey's resampler.py:
#     https://github.com/danpovey/filtering/blob/master/lilfilter/resampler.py
#     """

#     def __init__(self,
#                  input_sr: int,
#                  output_sr: int,
#                  dtype: torch.dtype = torch.float32,
#                  filter_width: int = 16,
#                  cutoff_ratio: float = 0.85,
#                  filter: Literal['kaiser', 'kaiser_best', 'kaiser_fast', 'hann'] = 'kaiser_fast',
#                  beta: float = 8.555504641634386) -> None:
#         super().__init__()  # init the base class
#         """
#         Initialize the Resampler.

#         Args:
#         - input_sr (int): Input sampling rate.
#         - output_sr (int): Output sampling rate.
#         - dtype (torch.dtype): Computation data type. Default: torch.float32.
#         - filter_width (int): Number of zeros per side in the sinc function. Default: 16.
#         - cutoff_ratio (float): Filter rolloff point as a fraction of Nyquist freq. Default: 0.95.
#         - filter (str): Filter type. One of ['kaiser', 'kaiser_best', 'kaiser_fast', 'hann']. Default: 'kaiser_fast'.
#         - beta (float): Parameter for 'kaiser' filter. Default: 8.555504641634386.

#         Note: Ratio between input_sr and output_sr should be reduced to simplest form.
#         """
#         assert isinstance(input_sr, int) and isinstance(output_sr, int)
#         if input_sr == output_sr:
#             self.resample_type = 'trivial'
#             return

#         d = math.gcd(input_sr, output_sr)
#         input_sr, output_sr = input_sr // d, output_sr // d

#         assert dtype in [torch.float32, torch.float64]
#         assert filter_width > 3  # a reasonable bare minimum
#         np_dtype = np.float32 if dtype == torch.float32 else np.float64

#         assert filter in ['hann', 'kaiser', 'kaiser_best', 'kaiser_fast']

#         if filter == 'kaiser_best':
#             filter_width = 64
#             beta = 14.769656459379492
#             cutoff_ratio = 0.9475937167399596
#             filter = 'kaiser'
#         elif filter == 'kaiser_fast':
#             filter_width = 16
#             beta = 8.555504641634386
#             cutoff_ratio = 0.85
#             filter = 'kaiser'
#         """
#         - Define a sample 'block' correlating `input_sr` input samples to `output_sr` output samples.
#         - Dividing samples into these blocks allows corresponding block alignment.
#         - On average, `zeros_per_block` zeros per block are present in the sinc function.
#         """
#         zeros_per_block = min(input_sr, output_sr) * cutoff_ratio
#         """
#         - Define conv kernel size n = (blocks_per_side*2 + 1), adding blocks to each side of the center.
#         - `blocks_per_side` blocks as window radius ensures each central block sample accesses its window.
#         - `blocks_per_side` is determined, rounding up if needed, as 1 + int(filter_width / zeros_per_block).
#         """
#         blocks_per_side = int(np.ceil(filter_width / zeros_per_block))

#         kernel_width = 2 * blocks_per_side + 1

#         # Shape of conv1d weights: (out_channels, in_channels, kernel_width)
#         """ Time computations are in units of 1 block, aligning with the `canonical` time axis,
#         since each block has input_sr input samples, adhering to our time unit."""

#         window_radius_in_blocks = blocks_per_side
#         """`times` will be sinc function arguments, expanding to shape (output_sr, input_sr, kernel_width)
#         via broadcasting. Ensuring t == 0 along the central block diagonal (when input_sr == output_sr)"""
#         times = (
#             np.arange(output_sr, dtype=np_dtype).reshape(
#                 (output_sr, 1, 1)) / output_sr - np.arange(input_sr, dtype=np_dtype).reshape(
#                     (1, input_sr, 1)) / input_sr - (np.arange(kernel_width, dtype=np_dtype).reshape(
#                         (1, 1, kernel_width)) - blocks_per_side))

#         def hann_window(a):
#             """
#             returning 0.5 + 0.5 cos(a*pi) on [-1,1] and 0 outside.
#             """
#             return np.heaviside(1 - np.abs(a), 0.0) * (0.5 + 0.5 * np.cos(a * np.pi))

#         def kaiser_window(a, beta):
#             w = special.i0(beta * np.sqrt(np.clip(1 - (
#                 (a - 0.0) / 1.0)**2.0, 0.0, 1.0))) / special.i0(beta)
#             return np.heaviside(1 - np.abs(a), 0.0) * w

#         """The weights are computed as a sinc function times a Hann-window function, normalized by
#         `zeros_per_block` (sinc) and `input_sr` (input function) to maintain integral and magnitude."""
#         if filter == 'hann':
#             weights = (
#                 np.sinc(times * zeros_per_block) * hann_window(times / window_radius_in_blocks) *
#                 zeros_per_block / input_sr)
#         else:
#             weights = (
#                 np.sinc(times * zeros_per_block) *
#                 kaiser_window(times / window_radius_in_blocks, beta) * zeros_per_block / input_sr)

#         self.input_sr = input_sr
#         self.output_sr = output_sr
#         """If output_sr == 1, merge input_sr into kernel_width for weights (shape: output_sr, input_sr,
#         kernel_width) to optimize convolution speed and avoid extra reshaping."""

#         assert weights.shape == (output_sr, input_sr, kernel_width)
#         if output_sr == 1:
#             self.resample_type = 'integer_downsample'
#             self.padding = input_sr * blocks_per_side
#             weights = torch.tensor(weights, dtype=dtype, requires_grad=False)
#             weights = weights.transpose(1, 2).contiguous().view(1, 1, input_sr * kernel_width)

#         elif input_sr == 1:
#             # For conv_transpose, use weights as if input_sr and output_sr were swapped, simulating downsampling.
#             self.resample_type = 'integer_upsample'
#             self.padding = output_sr * blocks_per_side
#             weights = torch.tensor(weights, dtype=dtype, requires_grad=False)
#             weights = weights.flip(2).transpose(0,
#                                                 2).contiguous().view(1, 1, output_sr * kernel_width)
#         else:
#             self.resample_type = 'general'
#             self.reshaped = False
#             self.padding = blocks_per_side
#             weights = torch.tensor(weights, dtype=dtype, requires_grad=False)

#         self.weights = torch.nn.Parameter(weights, requires_grad=False)

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Parameters:
#         - x: torch.Tensor, with shape (minibatch_size, sequence_length), dtype should match the instance's dtype.

#         Returns:
#         - A torch.Tensor with shape (minibatch_size, (sequence_length//input_sr)*output_sr), dtype matching the input,
#           and content resampled.
#         """
#         if self.resample_type == 'trivial':
#             return x
#         elif self.resample_type == 'integer_downsample':
#             (minibatch_size, seq_len) = x.shape  # (B, in_C, L) with in_C == 1
#             x = x.unsqueeze(1)
#             x = torch.nn.functional.conv1d(
#                 x, self.weights, stride=self.input_sr, padding=self.padding)  # (B, out_C, L)
#             return x.squeeze(1)  # (B, L)

#         elif self.resample_type == 'integer_upsample':
#             x = x.unsqueeze(1)
#             x = torch.nn.functional.conv_transpose1d(
#                 x, self.weights, stride=self.output_sr, padding=self.padding)

#             return x.squeeze(1)
#         else:
#             assert self.resample_type == 'general'
#             (minibatch_size, seq_len) = x.shape
#             num_blocks = seq_len // self.input_sr
#             if num_blocks == 0:
#                 # TODO: pad with zeros.
#                 raise RuntimeError("Signal is too short to resample")
#             # Truncate input
#             x = x[:, 0:(num_blocks * self.input_sr)].view(minibatch_size, num_blocks, self.input_sr)
#         x = x.transpose(1, 2)  # (B, in_C, L)
#         x = torch.nn.functional.conv1d(
#             x, self.weights, padding=self.padding)  # (B, out_C, num_blocks)
#         return x.transpose(1, 2).contiguous().view(minibatch_size, num_blocks * self.output_sr)

# def test_resampler_sinewave():
#     import torchaudio
#     # x: {440Hz, 220Hz} sine wave at 16kHz
#     t = torch.arange(0, 2, 1 / 16000)  # 2 seconds at 16kHz
#     x0 = torch.sin(2 * torch.pi * 440 * t) * 0.5
#     x1 = torch.sin(2 * torch.pi * 220 * t) * 0.5
#     x = torch.stack((x0, x1), dim=0)  # (2, 32000)

#     # Resample
#     resampler = Resampler(input_sr=16000, output_sr=12000)
#     y = resampler(x)  # (2, 24000)

#     # Export to wav
#     torchaudio.save("x.wav", x, 16000, bits_per_sample=16)
#     torchaudio.save("y.wav", y, 12000, bits_per_sample=16)

# def test_resampler_music():
#     import torchaudio
#     # x: music at 16kHz
#     x, _ = torchaudio.load("music.wav")
#     slice_length = 32000
#     n_slices = 80
#     slices = [x[0, i * slice_length:(i + 1) * slice_length] for i in range(n_slices)]
#     x = torch.stack(slices)  # (80, 32000)

#     # Resample
#     filter_width = 32
#     resampler = Resampler(16000, 12000, filter_width=filter_width)
#     y = resampler(x)  # (80, 24000)
#     y = y.reshape(1, -1)  # (1, 1920000)
#     torchaudio.save(f"y_filter_width{filter_width}.wav", y, 12000, bits_per_sample=16)

# class PitchShiftLayer(nn.Module):
#     """Applying batch-wise pitch-shift to time-domain audio signals.

#     Args:
#         expected_input_length (int): Expected input length. Default: ``32767``.
#         pshift_range (List[int]): Range of pitch shift in semitones. Default: ``[-2, 2]``.
#         min_gcd (int): Minimum GCD of input and output sampling rates for resampling. Setting high value can save GPU memory. Default: ``16``.
#         max_timing_error (float): Maximum allowed timing error in seconds. Default: ``0.002``.
#         fs (int): Sample rate of input waveform, x. Default: 16000.
#         bins_per_octave (int, optional): The number of steps per octave (Default : ``12``).
#         n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins (Default: ``512``).
#         win_length (int or None, optional): Window size. If None, then ``n_fft`` is used. (Default: ``None``).
#         hop_length (int or None, optional): Length of hop between STFT windows. If None, then ``win_length // 4``
#             is used (Default: ``None``).
#         window (Tensor or None, optional): Window tensor that is applied/multiplied to each frame/window.
#             If None, then ``torch.hann_window(win_length)`` is used (Default: ``None``).

#     """

#     def __init__(
#         self,
#         expected_input_length: int = 32767,
#         pshift_range: List[int] = [-2, 2],
#         min_gcd: int = 16,
#         max_timing_error: float = 0.002,
#         fs: int = 16000,
#         bins_per_octave: int = 12,
#         n_fft: int = 2048,
#         win_length: Optional[int] = None,
#         hop_length: Optional[int] = None,
#         window: Optional[torch.Tensor] = None,
#         filter_width: int = 16,
#         filter: Literal['kaiser', 'kaiser_best', 'kaiser_fast', 'hann'] = 'kaiser_fast',
#         cutoff_ratio: float = 0.85,
#         beta: float = 8.555504641634386,
#         **kwargs,
#     ):
#         super().__init__()
#         self.expected_input_length = expected_input_length
#         self.pshift_range = pshift_range
#         self.min_gcd = min_gcd
#         self.max_timing_error = max_timing_error
#         self.fs = fs
#         self.bins_per_octave = bins_per_octave
#         self.n_fft = n_fft
#         self.win_length = win_length
#         self.hop_length = hop_length
#         self.window = window
#         self.resample_args = {
#             "filter_width": filter_width,
#             "filter": filter,
#             "cutoff_ratio": cutoff_ratio,
#             "beta": beta,
#         }

#         # Initialize Resamplers
#         self._initialize_resamplers()

#     def _initialize_resamplers(self):
#         resamplers = nn.ModuleDict()
#         self.frame_gaps = {}  # for debugging
#         for i in range(self.pshift_range[0], self.pshift_range[1] + 1):
#             if i == 0:
#                 # No need to shift and resample
#                 resamplers[str(i)] = None
#             else:
#                 # Find optimal reconversion frames meeting the min_gcd
#                 stretched_frames, recon_frames, gap = self._find_optimal_reconversion_frames(i)
#                 self.frame_gaps[i] = gap
#                 resamplers[str(i)] = Resampler(stretched_frames, recon_frames, **self.resample_args)
#         self.resamplers = resamplers

#     def _find_optimal_reconversion_frames(self, semitone: int):
#         """
#         Find the optimal reconversion frames for a given source sample rate, input length, and semitone for strech.

#         Parameters:
#         - sr (int): Input audio sample rate, which should be power of 2
#         - n_step (int): The number of pitch-shift steps in semi-tone.
#         - min_gcd (int): The minimum desired GCD, power of 2. Defaults to 16. 16 or 32 are good choices.
#         - max_timing_error (float): The maximum allowed timing error, in seconds. Defaults to 5 ms

#         Returns:
#         - int: The optimal target sample rate
#         """
#         stretch_rate = 1 / 2.0**(-float(semitone) / self.bins_per_octave)
#         stretched_frames = round(self.expected_input_length * stretch_rate)

#         gcd = math.gcd(self.expected_input_length, stretched_frames)
#         if gcd >= self.min_gcd:
#             return stretched_frames, self.expected_input_length, 0
#         else:
#             reconversion_frames = adjust_b_to_gcd(stretched_frames, self.expected_input_length,
#                                                   self.min_gcd)
#             gap = reconversion_frames - self.expected_input_length
#             gap_sec = gap / self.fs
#             if gap_sec > self.max_timing_error:
#                 # TODO: modifying vocoder of stretch_waveform to adjust pitch-shift rate in cents
#                 raise ValueError(
#                     gap_sec < self.max_timing_error,
#                     f"gap_sec={gap_sec} > max_timing_error={self.max_timing_error} with semitone={semitone}, stretched_frames={stretched_frames}, recon_frames={reconversion_frames}. Try adjusting input lenght or decreasing min_gcd."
#                 )
#             else:
#                 return stretched_frames, reconversion_frames, gap_sec

#     @torch.no_grad()
#     def forward(self,
#                 x: torch.Tensor,
#                 semitone: int,
#                 resample: bool = True,
#                 fix_shape: bool = True) -> torch.Tensor:
#         """
#         Args:
#             x (torch.Tensor): (B, 1, T)
#         Returns:
#             torch.Tensor: (B, 1, T)
#         """
#         if semitone == 0:
#             return x
#         elif semitone >= min(self.pshift_range) and semitone <= max(self.pshift_range):
#             x = x.squeeze(1)  # (B, T)
#             original_x_size = x.size()
#             x = _stretch_waveform(
#                 x,
#                 semitone,
#                 self.bins_per_octave,
#                 self.n_fft,
#                 self.win_length,
#                 self.hop_length,
#                 self.window,
#             )
#             if resample:
#                 x = self.resamplers[str(semitone)].forward(x)
#             # Fix waveform shape
#             if fix_shape:
#                 if x.size(1) != original_x_size[1]:
#                     # print(f"Warning: {x.size(1)} != {original_x_length}")
#                     x = _fix_waveform_shape(x, original_x_size)
#             return x.unsqueeze(1)  # (B, 1, T)
#         else:
#             raise ValueError(f"semitone must be in range {self.pshift_range}")

# def test_pitchshift_layer():
#     import torchaudio
#     # music
#     # x, _ = torchaudio.load("music.wav")
#     # slice_length = 32767
#     # n_slices = 80
#     # slices = [x[0, i * slice_length:(i + 1) * slice_length] for i in range(n_slices)]
#     # x = torch.stack(slices).unsqueeze(1)  # (80, 1, 32767)

#     # sine wave
#     t = torch.arange(0, 2.0479, 1 / 16000)  # 2.05 seconds at 16kHz
#     x = torch.sin(2 * torch.pi * 440 * t) * 0.5
#     x = x.reshape(1, 1, 32767).tile(80, 1, 1)

#     # Resample
#     pos = 0
#     ps = PitchShiftLayer(
#         pshift_range=[-3, 4],
#         expected_input_length=32767,
#         fs=16000,
#         min_gcd=16,
#         max_timing_error=0.002,
#         # filter_width=64,
#         filter='kaiser_fast',
#         n_fft=2048)
#     y = []
#     for i in range(-3, 4):
#         y.append(ps(x[[pos], :, :], i, resample=False, fix_shape=False)[0, 0, :])
#     y = torch.cat(y).unsqueeze(0)  # (1, 32767 * 7)
#     torchaudio.save("y_2048_kaiser_fast.wav", y, 16000, bits_per_sample=16)

#     # TorchAudio PitchShifter fopr comparision
#     y_ta = []
#     for i in range(-3, 4):
#         ta_transform = torchaudio.transforms.PitchShift(16000, n_steps=i)
#         y_ta.append(ta_transform(x[[pos], :, :])[0, 0, :])
#     y_ta = torch.cat(y_ta).unsqueeze(0)  # (1, 32767 * 7)
#     torchaudio.save("y_ta.wav", y_ta, 16000, bits_per_sample=16)

# def test_min_gcd_mem_usage():
#     min_gcd = 16
#     for i in range(-3, 4):
#         stretched_frames = _stretch_waveform(x, i).shape[1]
#         adjusted = adjust_b_to_gcd(stretched_frames, 32767, min_gcd)
#         gcd_val = math.gcd(adjusted, stretched_frames)
#         gap = adjusted - 32767
#         gap_ms = (gap / 16000) * 1000
#         mem_mb = (stretched_frames / gcd_val) * (adjusted / gcd_val) * 3 * 4 / 1000 / 1000
#         print(f'\033[92mmin_gcd={min_gcd}\033[0m', f'ps={i}', f'frames={stretched_frames}',
#               f'adjusted_frames={adjusted}', f'gap={gap}', f'\033[91mgap_ms={gap_ms}\033[0m',
#               f'gcd={gcd_val}', f'mem_MB={mem_mb}')
