for n in range(1000):
    sampled_data = ds.__getitem__(n)

    a = deepcopy(sampled_data['note_event_segments'])
    b = deepcopy(sampled_data['note_event_segments'])

    for (note_events, tie_note_events, start_time) in list(zip(*b.values())):
        note_events = pitch_shift_note_events(note_events, 2)
        tie_note_events = pitch_shift_note_events(tie_note_events, 2)

    # compare
    for i, (note_events, tie_note_events, start_time) in enumerate(list(zip(*b.values()))):
        for j, ne in enumerate(note_events):
            if ne.is_drum is False:
                if ne.pitch != a['note_events'][i][j].pitch + 2:
                    print(i, j)
                assert ne.pitch == a['note_events'][i][j].pitch + 2

        for k, tne in enumerate(tie_note_events):
            assert tne.pitch == a['tie_note_events'][i][k].pitch + 2

    print('test {} passed'.format(n))


def assert_note_events_almost_equal(actual_note_events,
                                    predicted_note_events,
                                    ignore_time=False,
                                    ignore_activity=True,
                                    delta=5.1e-3):
    """
    Asserts that the given lists of Note instances are equal up to a small
    floating-point tolerance, similar to `assertAlmostEqual` of `unittest`.
    Tolerance is 5e-3 by default, which is 5 ms for 100 ticks-per-second.

    If `ignore_time` is True, then the time field is ignored. (useful for 
    comparing tie note events, default is False)

    If `ignore_activity` is True, then the activity field is ignored (default
    is True).
    """
    assert len(actual_note_events) == len(predicted_note_events)
    for j, (actual_note_event,
            predicted_note_event) in enumerate(zip(actual_note_events, predicted_note_events)):
        if ignore_time is False:
            assert abs(actual_note_event.time - predicted_note_event.time) <= delta
        assert actual_note_event.is_drum == predicted_note_event.is_drum
        if actual_note_event.is_drum is False and predicted_note_event.is_drum is False:
            assert actual_note_event.program == predicted_note_event.program
        assert actual_note_event.pitch == predicted_note_event.pitch
        assert actual_note_event.velocity == predicted_note_event.velocity
        if ignore_activity is False:
            assert actual_note_event.activity == predicted_note_event.activity


cache_old = deepcopy(dict(ds.cache))
for n in range(500):
    sampled_data = ds.__getitem__(n)
    cache_new = ds.cache
    cnt = 0
    for k, v in cache_new.items():
        if k in cache_old:
            cnt += 1
            assert (cache_new[k]['programs'] == cache_old[k]['programs']).all()
            assert (cache_new[k]['is_drum'] == cache_old[k]['is_drum']).all()
            assert (cache_new[k]['has_stems'] == cache_old[k]['has_stems'])
            assert (cache_new[k]['has_unannotated'] == cache_old[k]['has_unannotated'])
            assert (cache_new[k]['audio_array'] == cache_old[k]['audio_array']).all()

            for nes_new, nes_old in zip(cache_new[k]['note_event_segments']['note_events'],
                                        cache_old[k]['note_event_segments']['note_events']):
                assert_note_events_almost_equal(nes_new, nes_old)

            for tnes_new, tnes_old in zip(cache_new[k]['note_event_segments']['tie_note_events'],
                                          cache_old[k]['note_event_segments']['tie_note_events']):
                assert_note_events_almost_equal(tnes_new, tnes_old, ignore_time=True)

            for s_new, s_old in zip(cache_new[k]['note_event_segments']['start_times'],
                                    cache_old[k]['note_event_segments']['start_times']):
                assert s_new == s_old
    cache_old = deepcopy(dict(ds.cache))
    print(n, cnt)
