#!/bin/bash
shopt -s globstar
for file in "$1"/**/*.wav; do
    echo $file
    output_dir="tmp"
    spleeter separate -b 256k -B tensorflow -p spleeter:2stems -o $output_dir $file -f {instrument}.{codec}
    sox --ignore-length tmp/accompaniment.wav -r 16000 -c 1 -b 16 tmp/accompaniment_16k.wav
    sox --ignore-length tmp/vocals.wav -r 16000 -c 1 -b 16 tmp/vocals_16k.wav
    acc_file="${file//.wav/_accompaniment.wav}"
    voc_file="${file//.wav/_vocals.wav}"
    mv -f "tmp/accompaniment_16k.wav" $acc_file
    mv -f "tmp/vocals_16k.wav" $voc_file
    echo $acc_file
    echo $voc_file
    rm -rf tmp    
done
rm -rf pretrained_models