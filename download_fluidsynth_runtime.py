"""Prepare the pinned FluidSynth runtime used by MuScriptor playback/export."""

from src.utils.fluidsynth_runtime import download_fluidsynth_windows

if __name__ == "__main__":
    print(download_fluidsynth_windows())
