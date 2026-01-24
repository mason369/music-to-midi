"""
MIDI generation module with lyrics embedding.
"""
import logging
import os
from typing import List, Dict, Optional
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from src.models.data_models import Config, NoteEvent, LyricEvent, TrackType

logger = logging.getLogger(__name__)


class MidiGenerator:
    """
    MIDI file generation with multi-track support and lyrics embedding.

    Features:
    - Multi-track MIDI creation
    - Lyrics Meta Event embedding (0xFF 0x05)
    - LRC file export
    - Configurable tempo and instruments
    """

    # MIDI channel mapping for different track types
    CHANNEL_MAP = {
        TrackType.DRUMS: 9,     # GM standard drum channel
        TrackType.BASS: 0,
        TrackType.VOCALS: 1,
        TrackType.OTHER: 2
    }

    # GM program numbers for different track types
    PROGRAM_MAP = {
        TrackType.DRUMS: 0,      # Drums don't need program change
        TrackType.BASS: 33,      # Electric Bass (finger)
        TrackType.VOCALS: 52,    # Choir Aahs
        TrackType.OTHER: 0       # Acoustic Grand Piano
    }

    # Track names
    TRACK_NAMES = {
        TrackType.DRUMS: "Drums",
        TrackType.BASS: "Bass",
        TrackType.VOCALS: "Vocals",
        TrackType.OTHER: "Other Instruments"
    }

    def __init__(self, config: Config):
        """
        Initialize MIDI generator.

        Args:
            config: Application configuration
        """
        self.config = config
        self.ticks_per_beat = config.ticks_per_beat

    def generate(
        self,
        tracks: Dict[TrackType, List[NoteEvent]],
        lyrics: List[LyricEvent],
        tempo: float,
        output_path: str,
        embed_lyrics: bool = True
    ) -> str:
        """
        Generate multi-track MIDI file with optional lyrics.

        Args:
            tracks: Dictionary mapping TrackType to note events
            lyrics: List of lyric events with timestamps
            tempo: BPM
            output_path: Output MIDI file path
            embed_lyrics: Whether to embed lyrics as meta events

        Returns:
            Path to generated MIDI file
        """
        logger.info(f"Generating MIDI: {output_path}")

        # Create MIDI file
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)

        # Track 0: Tempo and lyrics (conductor track)
        meta_track = MidiTrack()
        mid.tracks.append(meta_track)
        meta_track.name = "Conductor"

        # Set tempo
        tempo_value = mido.bpm2tempo(tempo)
        meta_track.append(MetaMessage('set_tempo', tempo=tempo_value, time=0))

        # Set time signature (4/4)
        meta_track.append(MetaMessage(
            'time_signature',
            numerator=4,
            denominator=4,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0
        ))

        # Add lyrics to conductor track
        if embed_lyrics and lyrics:
            self._add_lyrics_events(meta_track, lyrics, tempo)

        # End of track
        meta_track.append(MetaMessage('end_of_track', time=0))

        # Create tracks for each stem
        for track_type, notes in tracks.items():
            if notes:
                track = self._create_track(track_type, notes, tempo)
                mid.tracks.append(track)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Save MIDI file
        mid.save(output_path)
        logger.info(f"MIDI saved: {output_path}")

        return output_path

    def _add_lyrics_events(
        self,
        track: MidiTrack,
        lyrics: List[LyricEvent],
        tempo: float
    ) -> None:
        """
        Add lyrics as MIDI Meta Events.

        Args:
            track: MIDI track to add lyrics to
            lyrics: List of lyric events
            tempo: BPM for time conversion
        """
        logger.info(f"Embedding {len(lyrics)} lyrics events")

        current_tick = 0

        for lyric in sorted(lyrics, key=lambda l: l.start_time):
            # Convert time to ticks
            tick = self._time_to_ticks(lyric.start_time, tempo)
            delta = max(0, tick - current_tick)

            # Add lyrics meta event
            try:
                track.append(MetaMessage(
                    'lyrics',
                    text=lyric.text,
                    time=delta
                ))
                current_tick = tick
            except Exception as e:
                logger.warning(f"Could not add lyric '{lyric.text}': {e}")

    def _create_track(
        self,
        track_type: TrackType,
        notes: List[NoteEvent],
        tempo: float
    ) -> MidiTrack:
        """
        Create a MIDI track with notes.

        Args:
            track_type: Type of track
            notes: Note events
            tempo: BPM

        Returns:
            MidiTrack with notes
        """
        track = MidiTrack()
        track.name = self.TRACK_NAMES.get(track_type, "Track")

        channel = self.CHANNEL_MAP.get(track_type, 0)
        program = self.PROGRAM_MAP.get(track_type, 0)

        # Program change (not for drums)
        if track_type != TrackType.DRUMS:
            track.append(Message(
                'program_change',
                channel=channel,
                program=program,
                time=0
            ))

        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        # Create note on/off events
        events = []
        for note in sorted_notes:
            start_tick = self._time_to_ticks(note.start_time, tempo)
            end_tick = self._time_to_ticks(note.end_time, tempo)

            events.append({
                'type': 'note_on',
                'tick': start_tick,
                'note': note.pitch,
                'velocity': note.velocity,
                'channel': channel
            })
            events.append({
                'type': 'note_off',
                'tick': end_tick,
                'note': note.pitch,
                'velocity': 0,
                'channel': channel
            })

        # Sort events by tick
        events.sort(key=lambda e: (e['tick'], e['type'] == 'note_on'))

        # Add events with delta times
        current_tick = 0
        for event in events:
            delta = max(0, event['tick'] - current_tick)

            if event['type'] == 'note_on':
                track.append(Message(
                    'note_on',
                    note=event['note'],
                    velocity=event['velocity'],
                    channel=event['channel'],
                    time=delta
                ))
            else:
                track.append(Message(
                    'note_off',
                    note=event['note'],
                    velocity=0,
                    channel=event['channel'],
                    time=delta
                ))

            current_tick = event['tick']

        # End of track
        track.append(MetaMessage('end_of_track', time=0))

        logger.info(f"Created {track_type.value} track with {len(sorted_notes)} notes")

        return track

    def _time_to_ticks(self, time_seconds: float, tempo: float) -> int:
        """
        Convert time in seconds to MIDI ticks.

        Args:
            time_seconds: Time in seconds
            tempo: BPM

        Returns:
            MIDI ticks
        """
        # ticks = time * (ticks_per_beat * bpm / 60)
        ticks = int(time_seconds * self.ticks_per_beat * tempo / 60)
        return ticks

    def export_lrc(
        self,
        lyrics: List[LyricEvent],
        output_path: str,
        title: str = "",
        artist: str = ""
    ) -> str:
        """
        Export lyrics to LRC format.

        Args:
            lyrics: List of lyric events
            output_path: Output LRC file path
            title: Song title (optional)
            artist: Artist name (optional)

        Returns:
            Path to LRC file
        """
        logger.info(f"Exporting LRC: {output_path}")

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write metadata
            if title:
                f.write(f"[ti:{title}]\n")
            if artist:
                f.write(f"[ar:{artist}]\n")
            f.write("[by:Music to MIDI Converter]\n")
            f.write("\n")

            # Group lyrics by line
            lines = self._group_lyrics_by_line(lyrics)

            # Write lyrics
            for line_start, line_text in lines:
                minutes = int(line_start // 60)
                seconds = line_start % 60
                f.write(f"[{minutes:02d}:{seconds:05.2f}]{line_text}\n")

        logger.info(f"LRC saved: {output_path}")
        return output_path

    def _group_lyrics_by_line(
        self,
        lyrics: List[LyricEvent],
        gap_threshold: float = 1.5
    ) -> List[tuple]:
        """
        Group lyrics into lines based on timing gaps.

        Args:
            lyrics: List of lyric events
            gap_threshold: Time gap to start new line

        Returns:
            List of (start_time, line_text) tuples
        """
        if not lyrics:
            return []

        sorted_lyrics = sorted(lyrics, key=lambda l: l.start_time)
        lines = []
        current_line_start = sorted_lyrics[0].start_time
        current_line_words = []

        for i, lyric in enumerate(sorted_lyrics):
            # Check if we should start a new line
            if i > 0:
                gap = lyric.start_time - sorted_lyrics[i-1].end_time
                if gap > gap_threshold:
                    # Save current line
                    if current_line_words:
                        lines.append((current_line_start, " ".join(current_line_words)))
                    current_line_start = lyric.start_time
                    current_line_words = []

            current_line_words.append(lyric.text)

        # Add last line
        if current_line_words:
            lines.append((current_line_start, " ".join(current_line_words)))

        return lines
