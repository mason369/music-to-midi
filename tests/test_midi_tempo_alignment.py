from pathlib import Path

import mido
import pytest

from src.core.pipeline import MusicToMidiPipeline


def _write_semantic_midi(path: Path, *, include_tempo: bool = False) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    if include_tempo:
        conductor.append(mido.MetaMessage("set_tempo", tempo=600_000, time=0))
    conductor.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(conductor)

    track = mido.MidiTrack()
    track.extend(
        [
            mido.MetaMessage("track_name", name="rare official track", time=0),
            mido.Message("program_change", program=73, channel=0, time=0),
            mido.Message("control_change", control=11, value=87, channel=0, time=12),
            mido.Message("pitchwheel", pitch=321, channel=0, time=3),
            mido.Message("note_on", note=60, velocity=91, channel=0, time=1),
            mido.Message("note_off", note=60, velocity=0, channel=0, time=1),
            mido.Message("note_on", note=60, velocity=74, channel=0, time=0),
            mido.Message("note_off", note=60, velocity=0, channel=0, time=1),
            mido.MetaMessage("end_of_track", time=0),
        ]
    )
    midi.tracks.append(track)
    midi.save(path)


def test_telknet_tempo_alignment_preserves_every_non_tempo_message_and_seconds(tmp_path):
    midi_path = tmp_path / "backend.mid"
    _write_semantic_midi(midi_path)
    source = mido.MidiFile(midi_path)

    result = MusicToMidiPipeline._normalize_midi_tempo_metadata(
        str(midi_path),
        90.0,
    )

    assert result == str(midi_path.resolve())
    normalized = mido.MidiFile(midi_path)
    tempo_messages = [
        message
        for track in normalized.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    ]
    assert tempo_messages == [mido.MetaMessage("set_tempo", tempo=666_667, time=0)]
    assert len(normalized.tracks) == len(source.tracks)

    source_tempo = 500_000
    target_tempo = tempo_messages[0].tempo
    tolerance = mido.tick2second(1, normalized.ticks_per_beat, target_tempo)
    for source_track, target_track in zip(source.tracks, normalized.tracks):
        target_messages = [
            message
            for message in target_track
            if not (message.is_meta and message.type == "set_tempo")
        ]
        assert len(target_messages) == len(source_track)

        source_tick = 0
        target_tick = 0
        for source_message, target_message in zip(source_track, target_messages):
            source_tick += source_message.time
            target_tick += target_message.time
            assert source_message.copy(time=0) == target_message.copy(time=0)
            source_seconds = mido.tick2second(
                source_tick,
                source.ticks_per_beat,
                source_tempo,
            )
            target_seconds = mido.tick2second(
                target_tick,
                normalized.ticks_per_beat,
                target_tempo,
            )
            assert target_seconds == pytest.approx(source_seconds, abs=tolerance)


def test_telknet_tempo_alignment_leaves_existing_tempo_midi_byte_identical(tmp_path):
    midi_path = tmp_path / "already-tempo.mid"
    _write_semantic_midi(midi_path, include_tempo=True)
    original_bytes = midi_path.read_bytes()

    MusicToMidiPipeline._normalize_midi_tempo_metadata(str(midi_path), 90.0)

    assert midi_path.read_bytes() == original_bytes


@pytest.mark.parametrize("tempo", [0.0, -1.0, float("nan"), float("inf")])
def test_tempo_alignment_rejects_invalid_detected_bpm_without_overwriting(tmp_path, tempo):
    midi_path = tmp_path / "invalid-tempo.mid"
    _write_semantic_midi(midi_path)
    original_bytes = midi_path.read_bytes()

    with pytest.raises(RuntimeError, match="无效 MIDI 速度"):
        MusicToMidiPipeline._normalize_midi_tempo_metadata(str(midi_path), tempo)

    assert midi_path.read_bytes() == original_bytes


def test_tempo_alignment_force_replaces_backend_placeholder_tempo(tmp_path):
    """force=True 必须替换后端 writer 写入的占位 tempo（钢琴后端自身不检测 BPM）。"""
    midi_path = tmp_path / "placeholder-tempo.mid"
    _write_semantic_midi(midi_path, include_tempo=True)

    MusicToMidiPipeline._normalize_midi_tempo_metadata(str(midi_path), 90.0, force=True)

    normalized = mido.MidiFile(midi_path)
    tempo_messages = [
        message
        for track in normalized.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    ]
    assert tempo_messages == [mido.MetaMessage("set_tempo", tempo=666_667, time=0)]


def test_specialized_piano_writes_detected_bpm_into_backend_midi(tmp_path):
    """钢琴模式必须将检测 BPM 写入输出 MIDI，而不是留下后端占位 120 BPM。

    复现用户报告：TransKun writeMidi 固定写入 500000us (=120 BPM)，
    _process_specialized_piano 此前不做 tempo 归一化，DAW 打开显示 120。
    """
    from src.models.data_models import BeatInfo, Config

    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"wav")

    class FakePianoTranscriber:
        def is_available(self):
            return True

        def is_model_available(self):
            return True

        def transcribe(self, audio_path, output_path, progress_callback=None):
            # 模拟 TransKun writeMidi：固定占位 120 BPM，一个 0.5 秒的音符 + CC64 踏板
            midi = mido.MidiFile(type=1, ticks_per_beat=960)
            conductor = mido.MidiTrack()
            conductor.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
            conductor.append(mido.MetaMessage("end_of_track", time=0))
            midi.tracks.append(conductor)
            track = mido.MidiTrack()
            track.append(mido.Message("program_change", program=0, channel=0, time=0))
            track.append(
                mido.Message("control_change", control=64, value=127, channel=0, time=0)
            )
            track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
            track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=960))
            track.append(mido.MetaMessage("end_of_track", time=0))
            midi.tracks.append(track)
            midi.save(output_path)
            return output_path

    pipeline = MusicToMidiPipeline(Config())
    pipeline._detect_beat_or_raise = lambda *_args, **_kwargs: BeatInfo(bpm=90.0)

    result = pipeline._process_specialized_piano(
        str(audio_path),
        str(tmp_path / "out"),
        transcriber=FakePianoTranscriber(),
        mode_label="FakePiano",
        output_suffix="piano_fake",
        install_hint="install",
        model_hint="model",
    )

    normalized = mido.MidiFile(result.midi_path)
    tempo_messages = [
        message
        for track in normalized.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    ]
    assert tempo_messages == [
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(90.0), time=0)
    ]
    # CC64 踏板消息必须原样保留
    assert any(
        message.type == "control_change" and message.control == 64 and message.value == 127
        for track in normalized.tracks
        for message in track
    )
    # 音符绝对时长不变：960 ticks @120BPM/960tpb = 0.5s，归一化后仍为 0.5s
    assert normalized.length == pytest.approx(0.5, abs=1e-3)


def test_muscriptor_official_writes_detected_bpm_into_backend_midi(tmp_path):
    """MuScriptor 路径必须将检测 BPM 写入输出 MIDI。

    muscriptor.utils.midi.notes_to_midi 默认 tempo_bpm=120，官方 writer 输出
    固定 500000us 占位 tempo；此前归一化未 force，占位 tempo 被原样保留。
    """
    from src.models.data_models import BeatInfo, Config

    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"wav")

    class FakeMuScriptorTranscriber:
        last_detected_instruments = ["acoustic_piano"]

        def transcribe_to_midi(self, audio_path, output_path, progress_callback=None):
            midi = mido.MidiFile(type=1, ticks_per_beat=480)
            conductor = mido.MidiTrack()
            conductor.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
            conductor.append(mido.MetaMessage("end_of_track", time=0))
            midi.tracks.append(conductor)
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("track_name", name="acoustic piano", time=0))
            track.append(mido.Message("program_change", program=0, channel=0, time=0))
            track.append(mido.Message("note_on", note=64, velocity=77, channel=0, time=0))
            track.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=480))
            track.append(mido.MetaMessage("end_of_track", time=0))
            midi.tracks.append(track)
            midi.save(output_path)
            return output_path

        def unload_model(self):
            return None

    config = Config(multi_instrument_model="muscriptor")
    config.muscriptor_instruments = ["acoustic_piano"]
    pipeline = MusicToMidiPipeline(config)
    pipeline._require_multi_instrument_available = lambda: None
    pipeline.muscriptor_transcriber = FakeMuScriptorTranscriber()
    pipeline._detect_beat_or_raise = lambda *_args, **_kwargs: BeatInfo(bpm=90.0)

    result = pipeline._process_muscriptor_official(str(audio_path), str(tmp_path / "out"))

    normalized = mido.MidiFile(result.midi_path)
    tempo_messages = [
        message
        for track in normalized.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    ]
    assert tempo_messages == [
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(90.0), time=0)
    ]
    # 音符绝对时长不变：480 ticks @120BPM/480tpb = 0.5s，归一化后仍为 0.5s
    assert normalized.length == pytest.approx(0.5, abs=1e-3)


def _write_tempo_map_source(path: Path) -> None:
    """源 MIDI：占位 120 BPM，track 1 在 0/1/2/3 秒各有一个音符事件。"""
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    conductor.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(conductor)
    track = mido.MidiTrack()
    # 480tpb @120BPM：每秒 = 960 ticks
    track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=960))
    track.append(mido.Message("note_on", note=62, velocity=100, channel=0, time=960))
    track.append(mido.Message("note_off", note=62, velocity=0, channel=0, time=960))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    midi.save(path)


def test_tempo_map_writes_multiple_set_tempo_events(tmp_path):
    """变速 tempo map：写入多个 set_tempo，位置/数值正确且绝对秒不变。"""
    midi_path = tmp_path / "variable.mid"
    _write_tempo_map_source(midi_path)

    MusicToMidiPipeline._normalize_midi_tempo_metadata(
        str(midi_path),
        60.0,
        force=True,
        tempo_map=[(0.0, 60.0), (1.5, 120.0)],
    )

    normalized = mido.MidiFile(midi_path)
    # 收集 track 0 的 set_tempo（绝对 tick, tempo）
    absolute_tick = 0
    tempo_events = []
    for message in normalized.tracks[0]:
        absolute_tick += message.time
        if message.is_meta and message.type == "set_tempo":
            tempo_events.append((absolute_tick, message.tempo))

    # 段1: 0-1.5s @60BPM → 1.5拍 = 720 ticks @480tpb；段2 从 tick 720 起 @120BPM
    assert tempo_events == [(0, 1_000_000), (720, 500_000)]
    # mido 按 tempo map 解释时总长不变（3 秒）
    assert normalized.length == pytest.approx(3.0, abs=1e-3)
    # 非 tempo 消息逐一保留
    source_notes = [
        message
        for track in mido.MidiFile(midi_path).tracks
        for message in track
        if message.type in {"note_on", "note_off"}
    ]
    assert len(source_notes) == 4
    # 每个音符的绝对秒与源一致：note_off@1s, note_on@2s, note_off@3s
    ticks = []
    absolute_tick = 0
    for message in normalized.tracks[1]:
        absolute_tick += message.time
        ticks.append(absolute_tick)
    # @tempo map：1s→480tick（段1 @60BPM，1s=1拍）；
    # 2s→720+second2tick(0.5s)=720+480=1200；3s→720+1440=2160；eot 同刻
    assert ticks == [0, 480, 1200, 2160, 2160]


def test_tempo_map_single_section_falls_back_to_single_tempo(tmp_path):
    """单点 tempo map 退化为单 tempo 行为。"""
    midi_path = tmp_path / "single-section.mid"
    _write_semantic_midi(midi_path)

    MusicToMidiPipeline._normalize_midi_tempo_metadata(
        str(midi_path),
        90.0,
        tempo_map=[(0.0, 90.0)],
    )

    normalized = mido.MidiFile(midi_path)
    tempo_messages = [
        message
        for track in normalized.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    ]
    assert tempo_messages == [mido.MetaMessage("set_tempo", tempo=666_667, time=0)]


def test_specialized_piano_writes_tempo_map_for_variable_song(tmp_path):
    """钢琴模式集成：变速歌曲输出 MIDI 携带多个 set_tempo。"""
    from src.models.data_models import BeatInfo, Config

    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"wav")

    class FakePianoTranscriber:
        def is_available(self):
            return True

        def is_model_available(self):
            return True

        def transcribe(self, audio_path, output_path, progress_callback=None):
            midi = mido.MidiFile(type=1, ticks_per_beat=960)
            conductor = mido.MidiTrack()
            conductor.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
            conductor.append(mido.MetaMessage("end_of_track", time=0))
            midi.tracks.append(conductor)
            track = mido.MidiTrack()
            track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
            track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=960))
            track.append(mido.MetaMessage("end_of_track", time=0))
            midi.tracks.append(track)
            midi.save(output_path)
            return output_path

    pipeline = MusicToMidiPipeline(Config(enable_tempo_map=True))
    pipeline._detect_beat_or_raise = lambda *_args, **_kwargs: BeatInfo(
        bpm=90.0,
        tempo_map=[(0.0, 60.0), (0.25, 120.0)],
    )

    result = pipeline._process_specialized_piano(
        str(audio_path),
        str(tmp_path / "out"),
        transcriber=FakePianoTranscriber(),
        mode_label="FakePiano",
        output_suffix="piano_fake",
        install_hint="install",
        model_hint="model",
    )

    normalized = mido.MidiFile(result.midi_path)
    absolute_tick = 0
    tempo_events = []
    for message in normalized.tracks[0]:
        absolute_tick += message.time
        if message.is_meta and message.type == "set_tempo":
            tempo_events.append((absolute_tick, message.tempo))

    # 段1: 0-0.25s @60BPM → 0.25拍 = 240 ticks @960tpb；段2 从 tick 240 起 @120BPM
    assert tempo_events == [(0, 1_000_000), (240, 500_000)]
    # 音符绝对时长不变（0.5 秒）
    assert normalized.length == pytest.approx(0.5, abs=1e-3)


def test_report_detected_bpm_includes_variable_range():
    """变速时进度消息与 bpm_display 携带范围和变速标注。"""
    from src.models.data_models import BeatInfo, Config

    pipeline = MusicToMidiPipeline(Config())
    reports = []
    pipeline._progress_callback = reports.append

    pipeline._report_detected_bpm(
        BeatInfo(bpm=90.0, tempo_map=[(0.0, 69.8), (30.0, 128.4)]),
        1.0,
        0.1,
    )

    assert len(reports) == 1
    progress = reports[0]
    assert progress.bpm_display is not None
    assert "69.8" in progress.bpm_display
    assert "128.4" in progress.bpm_display
    assert progress.message.startswith("BPM: ")


def test_report_detected_bpm_constant_tempo():
    """恒速时 bpm_display 为单值，无变速标注。"""
    from src.models.data_models import BeatInfo, Config

    pipeline = MusicToMidiPipeline(Config())
    reports = []
    pipeline._progress_callback = reports.append

    pipeline._report_detected_bpm(BeatInfo(bpm=120.0), 1.0, 0.1)

    assert len(reports) == 1
    assert reports[0].bpm_display == "120.0"


def test_specialized_piano_ignores_tempo_map_when_disabled_by_default(tmp_path):
    """默认（enable_tempo_map=False）即使 BeatInfo 带变速点，也只写单一全局 tempo。"""
    from src.models.data_models import BeatInfo, Config

    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"wav")

    class FakePianoTranscriber:
        def is_available(self):
            return True

        def is_model_available(self):
            return True

        def transcribe(self, audio_path, output_path, progress_callback=None):
            midi = mido.MidiFile(type=1, ticks_per_beat=960)
            conductor = mido.MidiTrack()
            conductor.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
            conductor.append(mido.MetaMessage("end_of_track", time=0))
            midi.tracks.append(conductor)
            track = mido.MidiTrack()
            track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
            track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=960))
            track.append(mido.MetaMessage("end_of_track", time=0))
            midi.tracks.append(track)
            midi.save(output_path)
            return output_path

    pipeline = MusicToMidiPipeline(Config())  # enable_tempo_map 默认 False
    pipeline._detect_beat_or_raise = lambda *_args, **_kwargs: BeatInfo(
        bpm=90.0,
        tempo_map=[(0.0, 60.0), (0.25, 120.0)],
    )

    result = pipeline._process_specialized_piano(
        str(audio_path),
        str(tmp_path / "out"),
        transcriber=FakePianoTranscriber(),
        mode_label="FakePiano",
        output_suffix="piano_fake",
        install_hint="install",
        model_hint="model",
    )

    normalized = mido.MidiFile(result.midi_path)
    tempo_messages = [
        message
        for track in normalized.tracks
        for message in track
        if message.is_meta and message.type == "set_tempo"
    ]
    # 只有一个 set_tempo，且是全局 bpm（90.0）而非变速点
    assert tempo_messages == [
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(90.0), time=0)
    ]
    assert normalized.length == pytest.approx(0.5, abs=1e-3)
