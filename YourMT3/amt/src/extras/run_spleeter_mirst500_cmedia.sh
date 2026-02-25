#!/bin/bash
shopt -s globstar
for file in "$1"/**/*.wav; do
    output_dir="${file%/*}"
    input_file="$output_dir/converted_Mixture.wav"
    spleeter separate -p spleeter:2stems -o $output_dir $input_file -f {instrument}.{codec}
    ffmpeg -i "$output_dir/vocals.wav" -acodec pcm_s16le -ac 1 -ar 16000 -y "$output_dir/vocals_16k.wav"
    ffmpeg -i "$output_dir/accompaniment.wav" -acodec pcm_s16le -ac 1 -ar 16000 -y "$output_dir/accompaniment_16k.wav"
    rm "$output_dir/vocals.wav"
    rm "$output_dir/accompaniment.wav"
    mv "$output_dir/vocals_16k.wav" "$output_dir/vocals.wav"
    mv "$output_dir/accompaniment_16k.wav" "$output_dir/accompaniment.wav"    
done
