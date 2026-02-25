""" Test the speed of the augmentation """
import torch
import torchaudio

# Device
device = torch.device("cuda")
# device = torch.device("cpu")

# Music
# x, _ = torchaudio.load("music.wav")
# slice_length = 32767
# n_slices = 80
# slices = [x[0, i * slice_length:(i + 1) * slice_length] for i in range(n_slices)]
# x = torch.stack(slices)  # (80, 32767)
# Sine wave
t = torch.arange(0, 2.0479, 1 / 16000)  # 2.05 seconds at 16kHz
x = torch.sin(2 * torch.pi * 440 * t) * 0.5
x = x.reshape(1, 1, 32767).tile(80, 1, 1)
x = x.to(device)

############################################################################################
# torch-audiomentation: https://github.com/asteroid-team/torch-audiomentation
#
# process time <CPU>: 1.18 s ± 5.35 ms
# process time <GPU>: 58 ms
# GPU memory usage: 3.8 GB per 1 semitone
############################################################################################
import torch
from torch_audiomentations import Compose, PitchShift, Gain, PolarityInversion

apply_augmentation = Compose(transforms=[
    # Gain(
    #     min_gain_in_db=-15.0,
    #     max_gain_in_db=5.0,
    #     p=0.5,
    # ),
    # PolarityInversion(p=0.5)
    PitchShift(
        min_transpose_semitones=0,
        max_transpose_semitones=2.2,
        mode="per_batch",  #"per_example",
        p=1.0,
        p_mode="per_batch",
        sample_rate=16000,
        target_rate=16000)
])
x_am = apply_augmentation(x, sample_rate=16000)

############################################################################################
# torchaudio:
#
# process time <CPU>: 4.01 s ± 19.6 ms per loop
# process time <GPU>: 25.1 ms ± 161 µs per loop
# memory usage <GPU>: 1.2 (growth to 5.49) GB per 1 semitone
############################################################################################
from torchaudio import transforms

ta_transform = transforms.PitchShift(16000, n_steps=2).to(device)
x_ta = ta_transform(x)

############################################################################################
# YourMT3 pitch_shift_layer:
#
# process time <CPU>: 389ms ± 22ms, (stretch=143 ms, resampler=245 ms)
# process time <GPU>: 7.18 ms ± 17.3 µs (stretch=6.47 ms, resampler=0.71 ms)
# memory usage: 16 MB per 1 semitone (average)
############################################################################################
from model.pitchshift_layer import PitchShiftLayer

ps_ymt3 = PitchShiftLayer(pshift_range=[2, 2], fs=16000, min_gcd=16, n_fft=2048).to(device)
x_ymt3 = ps_ymt3(x, 2)

############################################################################################
# Plot 1: Comparison of Process Time and GPU Memory Usage for 3 Pitch Shifting Methods
############################################################################################
import matplotlib.pyplot as plt

# Model names
models = ['torch-audiomentation', 'torchaudio', 'YourMT3:PitchShiftLayer']

# Process time (CPU) in seconds
cpu_time = [1.18, 4.01, 0.389]

# Process time (GPU) in milliseconds
gpu_time = [58, 25.1, 7.18]

# GPU memory usage in GB
gpu_memory = [3.8, 5.49, 0.016]

# Creating subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Creating bar charts
bar1 = axs[0].bar(models, cpu_time, color=['#FFB6C1', '#ADD8E6', '#98FB98'])
bar2 = axs[1].bar(models, gpu_time, color=['#FFB6C1', '#ADD8E6', '#98FB98'])
bar3 = axs[2].bar(models, gpu_memory, color=['#FFB6C1', '#ADD8E6', '#98FB98'])

# Adding labels and titles
axs[0].set_ylabel('Time (s)')
axs[0].set_title('Process Time (CPU) bsz=80')
axs[1].set_ylabel('Time (ms)')
axs[1].set_title('Process Time (GPU) bsz=80')
axs[2].set_ylabel('Memory (GB)')
axs[2].set_title('GPU Memory Usage per semitone')

# Adding grid for better readability of the plots
for ax in axs:
    ax.grid(axis='y')
    ax.set_yscale('log')
    ax.set_xticklabels(models, rotation=45, ha="right")

# Adding text labels above the bars
for i, rect in enumerate(bar1):
    axs[0].text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height(),
        f'{cpu_time[i]:.2f} s',
        ha='center',
        va='bottom')
for i, rect in enumerate(bar2):
    axs[1].text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height(),
        f'{gpu_time[i]:.2f} ms',
        ha='center',
        va='bottom')
for i, rect in enumerate(bar3):
    axs[2].text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height(),
        f'{gpu_memory[i]:.3f} GB',
        ha='center',
        va='bottom')
plt.tight_layout()
plt.show()

############################################################################################
# Plot 2: Stretch and Resampler Processing Time Contribution
############################################################################################
# Data
processing_type = ['Stretch (Phase Vocoder)', 'Resampler (Conv1D)']
cpu_times = [143, 245]  # [Stretch, Resampler] times for CPU in milliseconds
gpu_times = [6.47, 0.71]  # [Stretch, Resampler] times for GPU in milliseconds

# Creating subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plotting bar charts
axs[0].bar(processing_type, cpu_times, color=['#ADD8E6', '#98FB98'])
axs[1].bar(processing_type, gpu_times, color=['#ADD8E6', '#98FB98'])

# Adding labels and titles
axs[0].set_ylabel('Time (ms)')
axs[0].set_title('Contribution of CPU Processing Time: YMT3-PS (BSZ=80)')
axs[1].set_title('Contribution of GPU Processing Time: YMT3-PS (BSZ=80)')

# Adding grid for better readability of the plots
for ax in axs:
    ax.grid(axis='y')
    ax.set_yscale('log')  # Log scale to better visualize the smaller values

# Adding values on top of the bars
for ax, times in zip(axs, [cpu_times, gpu_times]):
    for idx, time in enumerate(times):
        ax.text(idx, time, f"{time:.2f} ms", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()
