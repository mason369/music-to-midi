import mirdata
from utils.mirdata_dev.datasets import slakh16k

ds = slakh16k.Dataset(data_home='../../data', version='2100-yourmt3-16k')
mtrack_ids = ds.mtrack_ids

# Collect plugin names
plugin_names = set()
cnt = 0
for mtrack_id in mtrack_ids:
    mtrack = ds.multitrack(mtrack_id)
    for track_id in mtrack.track_ids:
        track = ds.track(track_id)
        if track.instrument.lower() == 'bass':
            if track.plugin_name == 'upright_bass.nkm':
                print(f'{str(cnt)}: {track_id}: {track.plugin_name}')
            # if track.plugin_name not in plugin_names:
            #     plugin_names.add(track.plugin_name)
            #     print(f'{str(cnt)}: {track_id}: {track.plugin_name}')
            #     cnt += 1
"""
0: Track00001-S03: scarbee_rickenbacker_bass_palm_muted.nkm
1: Track00002-S01: classic_bass.nkm
2: Track00004-S01: scarbee_rickenbacker_bass.nkm
3: Track00005-S04: scarbee_jay_bass_both.nkm
4: Track00006-S03: pop_bass.nkm
5: Track00008-S00: scarbee_pre_bass.nkm
6: Track00013-S00: jazz_upright.nkm
7: Track00014-S01: funk_bass.nkm
8: Track00016-S01: scarbee_mm_bass.nkm
9: Track00024-S07: upright_bass.nkm
10: Track00027-S03: scarbee_jay_bass_slap_both.nkm
11: Track00094-S08: upright_bass2.nkm
"""