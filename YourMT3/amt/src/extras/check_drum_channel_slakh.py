from utils.mirdata_dev.datasets import slakh16k


def check_drum_channel_slakh(data_home: str):
    ds = slakh16k.Dataset(data_home, version='default')
    for track_id in ds.track_ids:
        is_drum = ds.track(track_id).is_drum
        midi = MidiFile(ds.track(track_id).midi_path)
        cnt = 0
        for msg in midi:
            if 'note' in msg.type:
                if is_drum and (msg.channel != 9):
                    print('found drum track with channel != 9 in track_id: ',
                          track_id)
                if not is_drum and (msg.channel == 9):
                    print(
                        'found non-drum track with channel == 9 in track_id: ',
                        track_id)
                if is_drum and (msg.channel == 9):
                    cnt += 1
        if cnt > 0:
            print(f'found {cnt} notes in drum track with ch 9 in track_id: ',
                  track_id)
    return