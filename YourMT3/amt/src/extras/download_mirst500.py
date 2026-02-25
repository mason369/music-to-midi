import os
import json
import numpy as np
from pytube import YouTube


def downloadMp3(yt, idx, askPath=0):
    # extract only audio
    video = yt.streams.filter(only_audio=True).first()

    destination = 'mp3File'
    # check for destination to save file
    if (askPath == 1):
        print("Enter the destination (leave blank for default dir mp3File)")
        destination = str(input(">> ")) or 'mp3File'

    # download the file
    out_file = video.download(output_path=destination)

    # save the file
    # base, ext = os.path.splitext(out_file)
    dir_path, file_base = os.path.split(out_file)

    new_file = os.path.join(dir_path, f'{idx}.mp3')
    os.rename(out_file, new_file)
    # result of success
    print(yt.title + " has been successfully downloaded.")


MISSING_FILE_IDS = [
    16, 26, 33, 38, 40, 50, 53, 55, 60, 81, 82, 98, 107, 122, 126, 127, 129, 141, 145, 150, 172,
    201, 205, 206, 215, 216, 221, 226, 232, 240, 243, 245, 255, 257, 267, 273, 278, 279, 285, 287,
    291, 304, 312, 319, 321, 325, 329, 332, 333, 336, 337, 342, 359, 375, 402, 417, 438, 445, 454,
    498
]

data_link_file = '../../../data/mir_St500_yourmt3_16k/MIR-ST500_20210206/MIR-ST500_link.json'
data_link = json.load(open(data_link_file, 'r'))
download_fail = []

for i in MISSING_FILE_IDS:
    print(f'Downloading {i}...')
    yt = YouTube(data_link[str(i)])
    try:
        downloadMp3(yt, idx=i)
    except:
        download_fail.append(i)
        print(f'Failed to download {i}.')

print(f'Failed to download {len(download_fail)} files: {download_fail}')