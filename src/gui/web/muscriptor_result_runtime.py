"""Project-native browser workbench with the public MuScriptor result controls."""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Mapping

from src.core.midi_quantization import (
    DEFAULT_MIDI_QUANTIZE_GRID,
    DEFAULT_MIDI_QUANTIZE_SCOPE,
    MIDI_QUANTIZE_GRIDS,
    MIDI_QUANTIZE_SCOPES,
)
from src.core.muscriptor_result_assets import (
    DEFAULT_MIDI_AUDIO_EXPORT_PRESET,
    MIDI_AUDIO_EXPORT_PRESETS,
)
from src.gui.web.track_mixer_runtime import track_file_url
from src.models.gm_instruments import get_instrument_name
from src.models.muscriptor_instruments import (
    MUSCRIPTOR_REPRESENTATIVE_PROGRAMS,
    muscriptor_instrument_label,
)

_COLORS = (
    "#4a9eff",
    "#ff8d66",
    "#7bd88f",
    "#c89bff",
    "#ffd166",
    "#ff70a6",
    "#62d2c3",
    "#b3e35d",
)


def _result_instrument_label(instrument: str, language: str) -> str:
    if instrument in MUSCRIPTOR_REPRESENTATIVE_PROGRAMS or instrument == "drums":
        return muscriptor_instrument_label(instrument, language)
    if instrument.startswith("gm:"):
        try:
            return get_instrument_name(int(instrument.split(":", 1)[1]), language)
        except ValueError:
            pass
    return instrument.replace("_", " ")


def build_muscriptor_result_html(
    state: Mapping[str, object],
    translate: Callable[[str], str],
    language: str,
) -> str:
    detected = [str(item) for item in state.get("detected_instruments", [])]
    selected = [str(item) for item in state.get("selected_instruments", [])]
    ordered = list(selected or detected)
    for instrument in detected:
        if instrument not in ordered:
            ordered.append(instrument)
    instrument_wavs = dict(state.get("instrument_wavs", {}))
    instruments = [
        {
            "id": instrument,
            "label": _result_instrument_label(instrument, language),
            "detected": instrument in detected,
            "color": _COLORS[index % len(_COLORS)],
            "url": (
                track_file_url(instrument_wavs[instrument]) if instrument in instrument_wavs else ""
            ),
        }
        for index, instrument in enumerate(ordered)
    ]
    manifest = {
        "notes": list(state.get("notes", [])),
        "duration": float(state.get("duration", 0.0)),
        "referenceBpm": float(state.get("reference_bpm", 0.0)),
        "targetBpm": float(state.get("target_bpm", 0.0)),
        "timeSignature": [int(value) for value in state.get("time_signature", (4, 4))],
        "beatTimes": [float(value) for value in state.get("beat_times", [])],
        "downbeats": [float(value) for value in state.get("downbeats", [])],
        "repeatTempoPerNoteTrack": bool(state.get("repeat_tempo_per_note_track", False)),
        "quantizeGrids": list(MIDI_QUANTIZE_GRIDS),
        "defaultQuantizeGrid": DEFAULT_MIDI_QUANTIZE_GRID,
        "quantizeScopes": list(MIDI_QUANTIZE_SCOPES),
        "defaultQuantizeScope": DEFAULT_MIDI_QUANTIZE_SCOPE,
        "backendLabel": str(state.get("backend_label", "")),
        "sourceTrackName": str(state.get("source_track_name", "")),
        "previewApi": str(state.get("preview_api", "")),
        "previewToken": str(state.get("preview_token", "")),
        "sheetApi": str(state.get("sheet_api", "")),
        "sheetToken": str(state.get("sheet_token", "")),
        "audioExportApi": str(state.get("audio_export_api", "")),
        "audioStemExportApi": str(state.get("audio_stem_export_api", "")),
        "audioExportPresets": [
            {
                "id": preset.id,
                "bitDepth": preset.bit_depth,
                "sampleRate": preset.sample_rate,
                "subtype": preset.soundfile_subtype,
                "label": translate(f"muscriptor_result.audio_export_{preset.id}"),
            }
            for preset in MIDI_AUDIO_EXPORT_PRESETS
        ],
        "defaultAudioExportPreset": DEFAULT_MIDI_AUDIO_EXPORT_PRESET,
        "originalUrl": track_file_url(str(state["playback_audio_path"])),
        "instruments": instruments,
        "downloads": {
            "midi": track_file_url(str(state["midi_path"])),
            "transcription": track_file_url(str(state["transcription_wav"])),
            "stereo": track_file_url(str(state["stereo_mix_wav"])),
        },
        "strings": {
            key: translate(f"muscriptor_result.{key}")
            for key in (
                "play",
                "pause",
                "follow",
                "original",
                "stereo",
                "instruments",
                "not_detected",
                "solo",
                "mute",
                "download",
                "download_midi",
                "download_sheet_music",
                "download_transcription",
                "audio_export_format",
                "audio_export_start",
                "audio_export_rendering",
                "audio_export_saved",
                "audio_export_failed",
                "stem_audio_export_start",
                "stem_audio_export_rendering",
                "stem_audio_export_saved",
                "stem_audio_export_failed",
                "sheet_music_rendering",
                "sheet_music_ready",
                "sheet_music_failed",
                "download_stereo",
                "ready",
                "linked_source",
                "zoom_help",
                "editor_toggle",
                "editor_help",
                "editor_add",
                "editor_delete",
                "editor_undo",
                "editor_redo",
                "editor_reset",
                "editor_select_all",
                "editor_cut",
                "editor_copy",
                "editor_paste",
                "editor_duplicate",
                "editor_quantize",
                "editor_quantize_scope",
                "editor_quantize_scope_all_tracks",
                "editor_quantize_scope_selected_notes",
                "editor_quantize_scope_tooltip",
                "editor_quantize_grid",
                "editor_quantize_grid_tooltip",
                "editor_resize_hint",
                "editor_instrument",
                "editor_velocity",
                "editor_view_zoom",
                "editor_view_zoom_tooltip",
                "editor_summary",
                "editor_audio_notice",
                "editor_audio_rendering",
                "editor_audio_ready",
                "editor_audio_failed",
                "editor_export_failed",
                "export_edited_midi",
            )
        },
    }
    encoded = html.escape(json.dumps(manifest, ensure_ascii=False), quote=False)
    return (
        '<div class="msr-root">'
        f'<pre class="msr-manifest" hidden>{encoded}</pre>'
        '<div class="msr-host"></div>'
        "</div>"
    )


MUSCRIPTOR_RESULT_CSS = r"""
.muscriptor-instrument-selector {
  background:#17243d !important; border:1px solid #2c4f7c !important;
  border-radius:8px !important; padding:10px 12px !important;
}
.muscriptor-instrument-selector label > span:first-child {
  color:#4a9eff !important; font-size:13px !important; font-weight:700 !important;
}
.muscriptor-instrument-selector .info {
  color:#9fb3d9 !important; font-size:12px !important; line-height:1.4 !important;
}
.muscriptor-instrument-selector [data-testid="token"] {
  background:#2a3f5f !important; border:1px solid #4a6d96 !important;
  border-radius:5px !important; color:#e0e0e0 !important; font-size:12px !important;
}
.muscriptor-instrument-selector input {
  color:#e0e0e0 !important; font-size:12px !important; min-height:34px !important;
}
.msr-root { margin: 10px 0; color: #e0e0e0; }
.msr-source { color:#8fc6ff; font-weight:600; background:#122039; border:1px solid #2c4f7c; border-radius:6px; padding:8px 10px; margin-bottom:8px; }
.msr-toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:10px; padding:12px; border:1px solid #365f8d; border-radius:8px; background:#17243d; }
.msr-btn { background:#2a3f5f; color:#e0e0e0; border:1px solid #3a4a6a; border-radius:5px; padding:6px 11px; cursor:pointer; }
.msr-btn:hover { background:#3a5a7c; border-color:#4a9eff; }
.msr-btn:disabled { opacity:.4; cursor:default; }
.msr-btn.active { color:#8fc6ff; border-color:#4a9eff; background:#203f68; }
.msr-clock { font-family:monospace; color:#c8d3e6; border:1px solid #3a4a6a; border-radius:4px; background:#16213e; padding:5px 8px; }
.msr-transport { display:flex; align-items:center; gap:10px; margin-top:8px; padding:8px 12px; border:1px solid #365f8d; border-radius:7px; background:#132139; }
.msr-editor { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:8px; padding:8px 12px; border:1px solid #365f8d; border-radius:7px; background:#132139; }
.msr-editor label { display:flex; align-items:center; gap:5px; color:#9fb3d9; }
.msr-editor select, .msr-editor input[type="number"] { color:#e0e0e0; background:#16213e; border:1px solid #3a4a6a; border-radius:4px; padding:5px 7px; }
.msr-editor .msr-quantize-grid select { min-width:74px; }
.msr-editor .msr-quantize-scope select { min-width:112px; }
.msr-editor .msr-zoom-input { width:68px; }
.msr-edit-summary { color:#8da4c9; font-family:monospace; }
.msr-edit-notice { flex-basis:100%; color:#d8b56a; display:none; }
.msr-progress { flex:1; min-width:160px; accent-color:#4a9eff; cursor:pointer; }
.msr-duration { font-family:monospace; color:#8da4c9; white-space:nowrap; }
.msr-mix { margin-left:auto; display:flex; align-items:center; gap:8px; color:#9aa5ad; }
.msr-grid { display:grid; grid-template-columns:minmax(0,4fr) minmax(220px,1fr); gap:12px; margin-top:12px; }
.msr-resize-handle { height:10px; margin:8px 0 -4px; cursor:ns-resize; touch-action:none; border-top:1px solid #4a78aa; border-bottom:1px solid #10233e; background:#203657; border-radius:4px; }
.msr-resize-handle:hover, .msr-resize-handle.dragging { background:#365f8d; }
.msr-roll-scroll { overflow:auto; height:390px; min-height:240px; max-height:650px; border:1px solid #365f8d; border-radius:6px; background:#0f1a2d; scrollbar-color:#3d628e #101b2d; scrollbar-width:thin; }
.msr-roll-scroll::-webkit-scrollbar { width:12px; height:12px; }
.msr-roll-scroll::-webkit-scrollbar-track { background:#101b2d; border-radius:6px; }
.msr-roll-scroll::-webkit-scrollbar-thumb { background:#3d628e; border-radius:6px; border:2px solid #101b2d; }
.msr-roll-scroll::-webkit-scrollbar-thumb:hover { background:#4a9eff; }
.msr-roll-world { position:relative; min-height:616px; }
.msr-roll-viewport { position:sticky; left:0; height:616px; overflow:hidden; }
.msr-roll { display:block; cursor:crosshair; }
.msr-playhead { position:absolute; top:0; bottom:0; width:2px; background:#fff; pointer-events:none; will-change:transform; }
.msr-instruments { border:1px solid #365f8d; border-radius:6px; background:#16213e; padding:12px; align-self:start; }
.msr-instruments h3 { margin:0 0 10px; }
.msr-row { display:flex; align-items:center; gap:8px; padding:6px 4px; }
.msr-row .msr-btn { min-width:32px; padding:5px 8px; }
.msr-row.undetected { opacity:.38; text-decoration:line-through; }
.msr-row.active-instrument { background:#203a61; border:1px solid #4a9eff; border-radius:4px; }
.msr-swatch { width:11px; height:11px; }
.msr-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.msr-row.muted .msr-name { opacity:.2; }
.msr-downloads { display:flex; flex-wrap:wrap; gap:8px; padding:12px 0 0; }
.msr-downloads a { text-decoration:none; }
.msr-audio-export { display:flex; flex-wrap:wrap; align-items:center; gap:6px; }
.msr-audio-export label { color:#9fb3d9; font-size:12px; }
.msr-audio-export select { color:#e0e0e0; background:#16213e; border:1px solid #3a4a6a; border-radius:4px; padding:6px 7px; }
@media (max-width:760px) { .msr-grid { grid-template-columns:1fr; } .msr-mix { margin-left:0; } }
"""


MUSCRIPTOR_RESULT_JS = r"""
(function () {
  "use strict";
  var sessions = [], sharedContext = null, nextSessionId = 1;
  var bufferCache = {}, bufferPromises = {};
  var LEFT = 72, ROW = 7, HEIGHT = 616, BASE_PPS = 92, MIN_PPS = 46, MAX_PPS = 368, ZOOM_STEP = 1.15;
  var MIN_BPM = 4, MAX_BPM = 400, SNAP_SECONDS = 0.05;

  function ctx() {
    if (!sharedContext) sharedContext = new (window.AudioContext || window.webkitAudioContext)();
    return sharedContext;
  }
  function clamp(value, low, high) { return Math.min(high, Math.max(low, value)); }
  function projectPlaybackRate(referenceBpm, targetBpm) {
    var reference = Number(referenceBpm), target = Number(targetBpm);
    if (
      !Number.isFinite(reference) || reference < MIN_BPM || reference > MAX_BPM ||
      !Number.isFinite(target) || target < MIN_BPM || target > MAX_BPM
    ) {
      throw new Error(
        "Invalid result playback BPM context: reference=" +
        reference + ", target=" + target
      );
    }
    return target / reference;
  }
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function button(text, title) {
    var node = el("button", "msr-btn", text);
    node.type = "button";
    if (title) node.title = title;
    return node;
  }
  function cloneNotes(notes) {
    return notes.map(function (note) { return Object.assign({}, note); });
  }
  function notesEqual(left, right) { return JSON.stringify(left) === JSON.stringify(right); }
  function snapTime(value) { return Math.round(Math.max(0, value) / SNAP_SECONDS) * SNAP_SECONDS; }
  function load(url) {
    if (bufferCache[url]) return Promise.resolve(bufferCache[url]);
    if (bufferPromises[url]) return bufferPromises[url];
    bufferPromises[url] = fetch(url)
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status + " " + url);
        return response.arrayBuffer();
      })
      .then(function (buffer) { return ctx().decodeAudioData(buffer); })
      .then(function (buffer) {
        bufferCache[url] = buffer;
        delete bufferPromises[url];
        return buffer;
      })
      .catch(function (error) {
        delete bufferPromises[url];
        throw error;
      });
    return bufferPromises[url];
  }

  function readU16(bytes, offset) { return (bytes[offset] << 8) | bytes[offset + 1]; }
  function readU32(bytes, offset) {
    return (
      bytes[offset] * 0x1000000 +
      (bytes[offset + 1] << 16) +
      (bytes[offset + 2] << 8) +
      bytes[offset + 3]
    );
  }
  function appendBytes(target, values) {
    for (var index = 0; index < values.length; index++) target.push(values[index]);
  }
  function writeU16(value) { return [(value >>> 8) & 255, value & 255]; }
  function writeU32(value) {
    return [
      Math.floor(value / 0x1000000) & 255,
      (value >>> 16) & 255,
      (value >>> 8) & 255,
      value & 255
    ];
  }
  function readVlq(bytes, cursor, limit) {
    var value = 0;
    for (var count = 0; count < 4; count++) {
      if (cursor.pos >= limit) throw new Error("Truncated MIDI variable-length quantity");
      var octet = bytes[cursor.pos++];
      value = (value << 7) | (octet & 127);
      if (!(octet & 128)) return value;
    }
    throw new Error("Invalid MIDI variable-length quantity");
  }
  function writeVlq(value) {
    var current = Math.max(0, Math.round(value));
    var output = [current & 127];
    while ((current >>>= 7)) output.unshift((current & 127) | 128);
    return output;
  }
  function ascii(bytes, offset, length) {
    var text = "";
    for (var index = 0; index < length; index++) text += String.fromCharCode(bytes[offset + index]);
    return text;
  }
  function eventKind(status, metaType, data) {
    if (status === 255) {
      if (metaType === 47) return "end";
      if (metaType === 81) return "tempo";
      return "other";
    }
    var high = status & 240;
    if (high === 128 || high === 144) return "note";
    if (high === 192) return "program";
    return "other";
  }
  function parseSmf(arrayBuffer) {
    var bytes = new Uint8Array(arrayBuffer);
    if (bytes.length < 14 || ascii(bytes, 0, 4) !== "MThd") {
      throw new Error("Edited MIDI source is not a Standard MIDI File");
    }
    var headerLength = readU32(bytes, 4);
    if (headerLength < 6 || 8 + headerLength > bytes.length) {
      throw new Error("Edited MIDI source has an invalid header");
    }
    var format = readU16(bytes, 8);
    var trackCount = readU16(bytes, 10);
    var division = readU16(bytes, 12);
    if (!trackCount) throw new Error("Edited MIDI source has no tracks");
    if (division & 32768) throw new Error("SMPTE MIDI timing is not supported by the editor");
    var offset = 8 + headerLength, tracks = [];
    for (var trackIndex = 0; trackIndex < trackCount; trackIndex++) {
      if (offset + 8 > bytes.length || ascii(bytes, offset, 4) !== "MTrk") {
        throw new Error("Edited MIDI source has a missing track chunk");
      }
      var trackLength = readU32(bytes, offset + 4);
      var cursor = { pos: offset + 8 }, end = cursor.pos + trackLength;
      if (end > bytes.length) throw new Error("Edited MIDI source has a truncated track");
      var tick = 0, runningStatus = 0, events = [];
      while (cursor.pos < end) {
        tick += readVlq(bytes, cursor, end);
        if (cursor.pos >= end) throw new Error("Edited MIDI source ends before an event status");
        var first = bytes[cursor.pos++], status;
        if (first & 128) {
          status = first;
        } else {
          if (!runningStatus) throw new Error("Edited MIDI source uses running status before a channel event");
          status = runningStatus;
          cursor.pos--;
        }
        var raw = [], metaType = null, data = [];
        if (status === 255) {
          if (cursor.pos >= end) throw new Error("Truncated MIDI meta event");
          metaType = bytes[cursor.pos++];
          var metaLength = readVlq(bytes, cursor, end);
          if (cursor.pos + metaLength > end) throw new Error("Truncated MIDI meta payload");
          data = Array.prototype.slice.call(bytes, cursor.pos, cursor.pos + metaLength);
          cursor.pos += metaLength;
          raw = [255, metaType];
          appendBytes(raw, writeVlq(metaLength));
          appendBytes(raw, data);
        } else if (status === 240 || status === 247) {
          var sysexLength = readVlq(bytes, cursor, end);
          if (cursor.pos + sysexLength > end) throw new Error("Truncated MIDI SysEx payload");
          data = Array.prototype.slice.call(bytes, cursor.pos, cursor.pos + sysexLength);
          cursor.pos += sysexLength;
          raw = [status];
          appendBytes(raw, writeVlq(sysexLength));
          appendBytes(raw, data);
        } else {
          if (status < 128 || status > 239) {
            throw new Error("Unsupported system event in edited MIDI source: " + status);
          }
          runningStatus = status;
          var dataLength = ((status & 240) === 192 || (status & 240) === 208) ? 1 : 2;
          if (cursor.pos + dataLength > end) throw new Error("Truncated MIDI channel event");
          data = Array.prototype.slice.call(bytes, cursor.pos, cursor.pos + dataLength);
          cursor.pos += dataLength;
          raw = [status];
          appendBytes(raw, data);
        }
        events.push({
          tick: tick,
          status: status,
          metaType: metaType,
          data: data,
          bytes: raw,
          kind: eventKind(status, metaType, data)
        });
      }
      tracks.push(events);
      offset = end;
    }
    return { format: format, division: division, tracks: tracks };
  }
  function tempoMicros(event) {
    if (event.kind !== "tempo" || event.data.length !== 3) {
      throw new Error("Invalid tempo event in edited MIDI");
    }
    var value = (event.data[0] << 16) | (event.data[1] << 8) | event.data[2];
    if (!value) throw new Error("Edited MIDI contains a zero tempo value");
    return value;
  }
  function buildTempoSections(parsed) {
    function sectionsFromChanges(changes) {
      changes.sort(function (left, right) {
        return left.tick - right.tick || left.track - right.track || left.index - right.index;
      });
      var sections = [{ tick: 0, seconds: 0, tempo: 500000 }];
      var currentTick = 0, currentSeconds = 0, currentTempo = 500000;
      changes.forEach(function (change) {
        currentSeconds += (
          (change.tick - currentTick) * currentTempo / 1000000 / parsed.division
        );
        currentTick = change.tick;
        currentTempo = tempoMicros(change.event);
        if (sections[sections.length - 1].tick === currentTick) {
          sections[sections.length - 1] = {
            tick: currentTick,
            seconds: currentSeconds,
            tempo: currentTempo
          };
        } else {
          sections.push({
            tick: currentTick,
            seconds: currentSeconds,
            tempo: currentTempo
          });
        }
      });
      return sections;
    }
    if (parsed.format === 2) {
      return parsed.tracks.map(function (events, trackIndex) {
        return sectionsFromChanges(events.map(function (event, index) {
          return { event: event, tick: event.tick, track: trackIndex, index: index };
        }).filter(function (item) { return item.event.kind === "tempo"; }));
      });
    }
    var sharedChanges = [];
    parsed.tracks.forEach(function (events, trackIndex) {
      events.forEach(function (event, index) {
        if (event.kind === "tempo") {
          sharedChanges.push({ event: event, tick: event.tick, track: trackIndex, index: index });
        }
      });
    });
    var shared = sectionsFromChanges(sharedChanges);
    return parsed.tracks.map(function () { return shared; });
  }
  function tickToSeconds(tick, sections, division) {
    var active = sections[0];
    for (var index = 1; index < sections.length && sections[index].tick <= tick; index++) {
      active = sections[index];
    }
    return active.seconds + (
      (tick - active.tick) * active.tempo / 1000000 / division
    );
  }
  function retainedEventKey(trackIndex, event) {
    return trackIndex + "|" + event.bytes.join(",");
  }
  function appendRetained(retained, key, tick) {
    if (!retained[key]) retained[key] = [];
    retained[key].push(tick);
  }
  function encodeSmf(parsed, trackEvents) {
    var output = [77, 84, 104, 100];
    appendBytes(output, writeU32(6));
    appendBytes(output, writeU16(parsed.format));
    appendBytes(output, writeU16(trackEvents.length));
    appendBytes(output, writeU16(parsed.division));
    trackEvents.forEach(function (events) {
      var data = [], previousTick = 0;
      events.sort(function (left, right) {
        return left.tick - right.tick || left.order - right.order || left.sequence - right.sequence;
      });
      events.forEach(function (event) {
        appendBytes(data, writeVlq(event.tick - previousTick));
        appendBytes(data, event.bytes);
        previousTick = event.tick;
      });
      appendBytes(data, [0, 255, 47, 0]);
      appendBytes(output, [77, 84, 114, 107]);
      appendBytes(output, writeU32(data.length));
      appendBytes(output, data);
    });
    return new Uint8Array(output);
  }
  function normalizeExpectedNote(note, parsed, timelineBpm, index) {
    var pitch = Number(note.pitch), velocity = Number(note.velocity);
    var start = Number(note.start), end = Number(note.end);
    var program = Number(note.program), trackIndex = Number(note.track_index);
    var channel = Number(note.channel), isDrum = Boolean(note.is_drum);
    if (
      !note.instrument || !Number.isInteger(pitch) || pitch < 0 || pitch > 127 ||
      !Number.isInteger(velocity) || velocity < 1 || velocity > 127 ||
      !Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end - start < 0.01 - 1e-9 ||
      !Number.isInteger(program) || program < 0 || program > 127 ||
      !Number.isInteger(trackIndex) || trackIndex < 0 || trackIndex >= parsed.tracks.length ||
      !Number.isInteger(channel) || channel < 0 || channel > 15 ||
      (isDrum && channel !== 9) || (!isDrum && channel === 9)
    ) {
      throw new Error("Invalid edited MIDI note at index " + index);
    }
    var ticksPerSecond = parsed.division * timelineBpm / 60;
    var startTick = Math.max(0, Math.round(start * ticksPerSecond));
    var durationTicks = Math.max(1, Math.round((end - start) * ticksPerSecond));
    return {
      track: trackIndex,
      channel: isDrum ? 9 : channel,
      pitch: pitch,
      velocity: velocity,
      start: startTick,
      end: startTick + durationTicks,
      program: isDrum ? 0 : program,
      drum: isDrum
    };
  }
  function collectNoteFingerprint(parsed) {
    var output = [], tempoValues = [], retained = {};
    parsed.tracks.forEach(function (events, trackIndex) {
      var programs = new Array(16).fill(0), active = {};
      events.forEach(function (event) {
        if (event.kind === "tempo") {
          tempoValues.push(tempoMicros(event));
          return;
        }
        if (
          event.kind !== "note" && event.kind !== "program" &&
          event.kind !== "tempo" && event.kind !== "end"
        ) {
          appendRetained(
            retained,
            retainedEventKey(trackIndex, event),
            event.tick
          );
        }
        var high = event.status & 240, channel = event.status & 15;
        if (high === 192) {
          programs[channel] = event.data[0];
          return;
        }
        if (high !== 128 && high !== 144) return;
        var pitch = event.data[0], velocity = event.data[1] || 0;
        var key = channel + ":" + pitch;
        if (high === 144 && velocity > 0) {
          if (!active[key]) active[key] = [];
          active[key].push({
            tick: event.tick,
            velocity: velocity,
            program: channel === 9 ? 0 : programs[channel]
          });
        } else {
          if (!active[key] || !active[key].length) {
            throw new Error("Edited MIDI verification found an unmatched note-off");
          }
          var started = active[key].shift();
          output.push([
            trackIndex, channel, pitch, started.velocity, started.tick, event.tick,
            started.program, channel === 9
          ]);
        }
      });
      Object.keys(active).forEach(function (key) {
        if (active[key].length) throw new Error("Edited MIDI verification found an unterminated note");
      });
    });
    output.sort();
    return { notes: output, tempos: tempoValues, retained: retained };
  }
  function buildEditedSmf(
    arrayBuffer,
    notes,
    targetBpm,
    referenceBpm,
    repeatTempoPerNoteTrack
  ) {
    if (!Number.isFinite(targetBpm) || targetBpm < MIN_BPM || targetBpm > MAX_BPM) {
      throw new Error("Invalid target BPM for edited MIDI: " + targetBpm);
    }
    if (!Number.isFinite(referenceBpm) || referenceBpm < MIN_BPM || referenceBpm > MAX_BPM) {
      throw new Error("Invalid reference BPM for edited MIDI: " + referenceBpm);
    }
    var parsed = parseSmf(arrayBuffer), sequence = 0, retainedSource = {};
    var tempo = Math.round(60000000 / targetBpm);
    var normalizedNotes = notes.map(function (note, noteIndex) {
      return normalizeExpectedNote(note, parsed, referenceBpm, noteIndex);
    });
    var reservedChannels = { 9: true };
    parsed.tracks.forEach(function (events) {
      events.forEach(function (event) {
        if (event.status >= 128 && event.status < 240) {
          reservedChannels[event.status & 15] = true;
        }
      });
    });
    normalizedNotes.forEach(function (note) {
      reservedChannels[note.channel] = true;
      note.sourceChannel = note.channel;
    });
    var spareChannels = [];
    for (var channelIndex = 0; channelIndex < 16; channelIndex++) {
      if (!reservedChannels[channelIndex]) spareChannels.push(channelIndex);
    }
    var logicalLanes = {}, auxiliaryBySource = {};
    normalizedNotes.forEach(function (note, noteIndex) {
      var laneKey = note.track + ":" + note.sourceChannel;
      if (!logicalLanes[laneKey]) logicalLanes[laneKey] = [];
      logicalLanes[laneKey].push({ note: note, index: noteIndex });
    });
    Object.keys(logicalLanes).forEach(function (laneKey) {
      var lane = logicalLanes[laneKey];
      lane.sort(function (left, right) {
        return (
          left.note.start - right.note.start ||
          left.note.end - right.note.end ||
          left.index - right.index
        );
      });
      var voiceChannels = [lane[0].note.sourceChannel], pitchEnds = [{}];
      lane.forEach(function (item) {
        var selectedVoice = -1;
        for (var voiceIndex = 0; voiceIndex < pitchEnds.length; voiceIndex++) {
          var activeEnd = pitchEnds[voiceIndex][item.note.pitch];
          if (activeEnd === undefined || activeEnd <= item.note.start) {
            selectedVoice = voiceIndex;
            break;
          }
        }
        if (selectedVoice < 0) {
          if (item.note.drum) {
            throw new Error(
              "Edited MIDI cannot losslessly encode overlapping drum hits " +
              "with the same track/channel/pitch: " + laneKey +
              " pitch " + item.note.pitch + " at tick " + item.note.start
            );
          }
          if (!spareChannels.length) {
            throw new Error(
              "Edited MIDI needs another melodic channel to preserve overlapping " +
              "same-pitch notes, but all 15 melodic channels are already in use: " +
              laneKey + " pitch " + item.note.pitch + " at tick " + item.note.start
            );
          }
          var auxiliaryChannel = spareChannels.shift();
          voiceChannels.push(auxiliaryChannel);
          pitchEnds.push({});
          if (!auxiliaryBySource[item.note.sourceChannel]) {
            auxiliaryBySource[item.note.sourceChannel] = [];
          }
          auxiliaryBySource[item.note.sourceChannel].push(auxiliaryChannel);
          selectedVoice = voiceChannels.length - 1;
        }
        item.note.channel = voiceChannels[selectedVoice];
        pitchEnds[selectedVoice][item.note.pitch] = item.note.end;
      });
    });
    var trackEvents = parsed.tracks.map(function (events, trackIndex) {
      var output = [];
      events.forEach(function (event) {
        if (
          event.kind === "note" || event.kind === "program" ||
          event.kind === "tempo" || event.kind === "end"
        ) {
          return;
        }
        // Source MIDI has already been normalized onto the detected/reference
        // tick grid by the pipeline. Preserve the exact tick here; converting
        // through target-tempo seconds would apply the BPM ratio twice.
        var targetTick = event.tick;
        var retainedCopies = [event.bytes.slice()];
        if (event.status >= 128 && event.status < 240) {
          var sourceChannel = event.status & 15;
          (auxiliaryBySource[sourceChannel] || []).forEach(function (channel) {
            var duplicate = event.bytes.slice();
            duplicate[0] = (duplicate[0] & 240) | channel;
            retainedCopies.push(duplicate);
          });
        }
        retainedCopies.forEach(function (bytes) {
          appendRetained(
            retainedSource,
            retainedEventKey(trackIndex, { bytes: bytes }),
            targetTick
          );
          sequence++;
          output.push({
            tick: targetTick,
            order: 0,
            sequence: sequence,
            bytes: bytes
          });
        });
      });
      return output;
    });
    var tempoBytes = [255, 81, 3, (tempo >>> 16) & 255, (tempo >>> 8) & 255, tempo & 255];
    var tempoTracks = parsed.format === 2 ? trackEvents.map(function (_track, index) { return index; }) : [0];
    if (parsed.format !== 2 && Boolean(repeatTempoPerNoteTrack)) {
      normalizedNotes.forEach(function (note) {
        if (tempoTracks.indexOf(note.track) < 0) tempoTracks.push(note.track);
      });
      tempoTracks.sort(function (left, right) { return left - right; });
    }
    tempoTracks.forEach(function (trackIndex) {
      trackEvents[trackIndex].push({ tick: 0, order: 0, sequence: 0, bytes: tempoBytes.slice() });
    });
    var lanes = {};
    normalizedNotes.forEach(function (normalized, noteIndex) {
      if (normalized.drum) return;
      var laneKey = normalized.track + ":" + normalized.channel;
      if (!lanes[laneKey]) lanes[laneKey] = [];
      lanes[laneKey].push({ note: normalized, index: noteIndex });
    });
    Object.keys(lanes).forEach(function (laneKey) {
      var lane = lanes[laneKey];
      lane.sort(function (left, right) {
        return (
          left.note.start - right.note.start ||
          left.note.end - right.note.end ||
          left.index - right.index
        );
      });
      var active = [], currentProgram = null, cursor = 0;
      while (cursor < lane.length) {
        var startTick = lane[cursor].note.start;
        active = active.filter(function (item) { return item.end > startTick; });
        var groupEnd = cursor, starting = [];
        while (groupEnd < lane.length && lane[groupEnd].note.start === startTick) {
          starting.push(lane[groupEnd]);
          groupEnd++;
        }
        var programs = {};
        active.forEach(function (item) { programs[item.program] = true; });
        starting.forEach(function (item) { programs[item.note.program] = true; });
        var programValues = Object.keys(programs).map(Number);
        if (programValues.length !== 1) {
          throw new Error(
            "Edited MIDI cannot assign overlapping instruments to one track/channel: " +
            laneKey + " at tick " + startTick
          );
        }
        var desiredProgram = programValues[0];
        if (currentProgram !== desiredProgram) {
          sequence++;
          trackEvents[starting[0].note.track].push({
            tick: currentProgram === null ? 0 : startTick,
            order: 20,
            sequence: sequence,
            bytes: [192 | starting[0].note.channel, desiredProgram]
          });
          currentProgram = desiredProgram;
        }
        starting.forEach(function (item) {
          active.push({ end: item.note.end, program: item.note.program });
        });
        cursor = groupEnd;
      }
    });
    var expected = normalizedNotes.map(function (normalized, noteIndex) {
      sequence++;
      trackEvents[normalized.track].push({
        tick: normalized.end,
        order: 10,
        sequence: sequence,
        bytes: [128 | normalized.channel, normalized.pitch, 0]
      });
      sequence++;
      trackEvents[normalized.track].push({
        tick: normalized.start,
        order: 30,
        sequence: sequence,
        bytes: [144 | normalized.channel, normalized.pitch, normalized.velocity]
      });
      return [
        normalized.track, normalized.channel, normalized.pitch, normalized.velocity,
        normalized.start, normalized.end, normalized.program, normalized.drum
      ];
    });
    var encoded = encodeSmf(parsed, trackEvents);
    var verified = collectNoteFingerprint(parseSmf(encoded.buffer));
    expected.sort();
    if (JSON.stringify(expected) !== JSON.stringify(verified.notes)) {
      throw new Error("Edited MIDI note verification failed");
    }
    var expectedTempoCount = tempoTracks.length;
    if (
      verified.tempos.length !== expectedTempoCount ||
      verified.tempos.some(function (value) { return value !== tempo; })
    ) {
      throw new Error("Edited MIDI tempo verification failed");
    }
    var retainedSourceKeys = Object.keys(retainedSource).sort();
    var retainedOutputKeys = Object.keys(verified.retained).sort();
    if (
      JSON.stringify(retainedSourceKeys) !== JSON.stringify(retainedOutputKeys) ||
      retainedSourceKeys.some(function (key) {
        var sourceTicks = retainedSource[key].slice().sort(function (a, b) { return a - b; });
        var outputTicks = verified.retained[key].slice().sort(function (a, b) { return a - b; });
        return JSON.stringify(sourceTicks) !== JSON.stringify(outputTicks);
      })
    ) {
      throw new Error(
        "Edited MIDI pass-through event verification failed: source and output differ"
      );
    }
    return encoded;
  }

  function ResultSession(root) {
    this.root = root;
    this.host = root.querySelector(".msr-host");
    this.m = {};
    this.buffers = {};
    this.sources = [];
    this.gains = {};
    this.panners = {};
    this.position = 0;
    this.startedAt = 0;
    this.playing = false;
    this.muted = new Set();
    this.solo = null;
    this.mix = 0.75;
    this.stereo = false;
    this.follow = true;
    this.raf = 0;
    this.pps = BASE_PPS;
    this.drawRaf = 0;
    this.disposed = false;
    this.ownerId = "midi-result-" + (nextSessionId++);
    this.editing = false;
    this.selectedIndex = null;
    this.selectedIndices = new Set();
    this.drag = null;
    this.undoStack = [];
    this.redoStack = [];
    this.clipboard = [];
    this.originalNotes = [];
    this.targetBpm = 0;
    this.quantizeGrid = "";
    this.activeInstrument = "";
    this.rollHeight = 390;
    this.previewRevision = 0;
    this.previewTimer = 0;
    this.originalPreview = null;
    this.downloadAnchors = {};
    this.audioDownloadReady = true;
    this.audioExportInFlight = false;
    this.onExternalPlayback = this.handleExternalPlayback.bind(this);
  }
  ResultSession.prototype.init = function () {
    try {
      this.m = JSON.parse(this.root.querySelector(".msr-manifest").textContent);
      this.m.notes = this.m.notes.map(function (note) {
        return {
          instrument: String(note.instrument),
          pitch: Number(note.pitch),
          velocity: Number(note.velocity),
          start: Number(note.start),
          end: Number(note.end),
          program: Number(note.program),
          is_drum: Boolean(note.is_drum),
          track_index: Number(note.track_index),
          channel: Number(note.channel)
        };
      });
      this.m.beatTimes = (this.m.beatTimes || []).map(Number);
      this.m.downbeats = (this.m.downbeats || []).map(Number);
      this.m.quantizeGrids = (this.m.quantizeGrids || []).map(String);
      this.m.defaultQuantizeGrid = String(this.m.defaultQuantizeGrid || "");
      this.m.quantizeScopes = (this.m.quantizeScopes || []).map(String);
      this.m.defaultQuantizeScope = String(this.m.defaultQuantizeScope || "");
      this.m.audioExportPresets = (this.m.audioExportPresets || []).map(function (preset) {
        return {
          id: String(preset.id),
          bitDepth: Number(preset.bitDepth),
          sampleRate: Number(preset.sampleRate),
          subtype: String(preset.subtype),
          label: String(preset.label)
        };
      });
      this.m.defaultAudioExportPreset = String(this.m.defaultAudioExportPreset || "");
      var defaultAudioExportPreset = this.m.defaultAudioExportPreset;
      if (this.m.audioExportPresets.length !== 2
          || this.m.audioExportPresets.filter(function (preset) {
            return preset.id === defaultAudioExportPreset;
          }).length !== 1
          || this.m.audioExportPresets.some(function (preset) {
            return !preset.id || !preset.label
              || ![16, 24].includes(preset.bitDepth)
              || ![44100, 48000].includes(preset.sampleRate)
              || !["PCM_16", "PCM_24"].includes(preset.subtype);
          })) {
        throw new Error("Invalid MIDI audio export preset contract");
      }
      if (!this.m.quantizeGrids.length
          || this.m.quantizeGrids.indexOf(this.m.defaultQuantizeGrid) < 0
          || this.m.quantizeGrids.some(function (grid) {
            return !/^1\/(4|8|16|32|64)$/.test(grid);
          })) {
        throw new Error("Invalid MIDI quantization-grid contract");
      }
      this.quantizeGrid = this.m.defaultQuantizeGrid;
      if (this.m.quantizeScopes.length !== 2
          || this.m.quantizeScopes.indexOf("all_tracks") < 0
          || this.m.quantizeScopes.indexOf("selected_notes") < 0
          || this.m.defaultQuantizeScope !== "all_tracks") {
        throw new Error("Invalid MIDI quantization-scope contract");
      }
      this.quantizeScope = this.m.defaultQuantizeScope;
      this.m.timeSignature = (this.m.timeSignature || [4, 4]).map(Number);
      if (this.m.timeSignature.length !== 2
          || !Number.isInteger(this.m.timeSignature[0])
          || this.m.timeSignature[0] < 1
          || this.m.timeSignature[0] > 255
          || !Number.isInteger(this.m.timeSignature[1])
          || this.m.timeSignature[1] < 1
          || this.m.timeSignature[1] > 128
          || (this.m.timeSignature[1] & (this.m.timeSignature[1] - 1)) !== 0) {
        throw new Error("Invalid MIDI time signature in piano-roll manifest");
      }
      [this.m.beatTimes, this.m.downbeats].forEach(function (marks) {
        if (marks.some(function (value, index) {
          return !Number.isFinite(value) || value < 0 || (index > 0 && value <= marks[index - 1]);
        })) {
          throw new Error("Invalid Beat This grid in piano-roll manifest");
        }
      });
      if (this.m.beatTimes.length === 1) {
        throw new Error("Piano-roll beat grid requires at least two beats");
      }
      this.originalNotes = cloneNotes(this.m.notes);
      this.targetBpm = Number(this.m.targetBpm);
      this.activeInstrument = this.m.notes.length ? this.m.notes[0].instrument : "";
    } catch (error) {
      this.host.textContent = String(error);
      return;
    }
    this.build();
    window.addEventListener("music-to-midi-playback-start", this.onExternalPlayback);
    var self = this, initialBuffers = {};
    var jobs = [load(this.m.originalUrl).then(function (buffer) { initialBuffers.original = buffer; })];
    this.m.instruments.forEach(function (instrument) {
      if (instrument.detected && instrument.url) {
        jobs.push(load(instrument.url).then(function (buffer) { initialBuffers[instrument.id] = buffer; }));
      }
    });
    Promise.all(jobs).then(function () {
      if (self.disposed) return;
      self.originalPreview = {
        buffers: initialBuffers,
        duration: self.m.duration,
        instruments: self.m.instruments.map(function (instrument) {
          return { id: instrument.id, detected: instrument.detected, url: instrument.url };
        }),
        transcriptionUrl: self.m.downloads.transcription,
        stereoUrl: self.m.downloads.stereo
      };
      if (self.previewRevision === 0) {
        self.buffers = Object.assign({}, initialBuffers);
        self.play.disabled = false;
        self.progress.disabled = false;
        self.status.textContent = self.m.strings.ready;
      }
      self.drawStatic();
      self.layoutPlayhead();
    }).catch(function (error) {
      if (!self.disposed) self.status.textContent = String(error);
    });
  };
  ResultSession.prototype.build = function () {
    var self = this, strings = this.m.strings;
    if (this.m.sourceTrackName) {
      this.host.appendChild(el(
        "div",
        "msr-source",
        strings.linked_source
          .replace("{track}", this.m.sourceTrackName)
          .replace("{backend}", this.m.backendLabel)
      ));
    }
    var bar = el("div", "msr-toolbar");
    this.play = button(strings.play);
    this.play.disabled = true;
    this.play.onclick = function () { self.toggle(); };
    bar.appendChild(this.play);
    var follow = button(strings.follow);
    follow.classList.add("active");
    follow.onclick = function () {
      self.follow = !self.follow;
      follow.classList.toggle("active", self.follow);
    };
    bar.appendChild(follow);
    this.clock = el("span", "msr-clock", "0.0s");
    bar.appendChild(this.clock);
    this.status = el("span", "msr-clock", "");
    bar.appendChild(this.status);
    var mix = el("label", "msr-mix");
    mix.appendChild(document.createTextNode(strings.original));
    this.mixInput = el("input");
    this.mixInput.type = "range";
    this.mixInput.min = "0";
    this.mixInput.max = "1";
    this.mixInput.step = ".01";
    this.mixInput.value = String(this.mix);
    this.mixInput.oninput = function () { self.mix = parseFloat(this.value); self.applyMix(); };
    mix.appendChild(this.mixInput);
    mix.appendChild(document.createTextNode("MIDI"));
    var stereo = el("input");
    stereo.type = "checkbox";
    stereo.onchange = function () {
      self.stereo = this.checked;
      self.mixInput.disabled = self.stereo;
      self.applyMix();
    };
    mix.appendChild(stereo);
    mix.appendChild(document.createTextNode(strings.stereo));
    bar.appendChild(mix);
    this.host.appendChild(bar);

    var transport = el("div", "msr-transport");
    this.progress = el("input", "msr-progress");
    this.progress.type = "range";
    this.progress.min = "0";
    this.progress.max = String(this.m.duration);
    this.progress.step = ".01";
    this.progress.value = "0";
    this.progress.disabled = true;
    this.progress.oninput = function () { self.seek(parseFloat(this.value)); };
    transport.appendChild(this.progress);
    this.durationLabel = el("span", "msr-duration", "/ " + this.m.duration.toFixed(1) + "s");
    transport.appendChild(this.durationLabel);
    this.host.appendChild(transport);
    this.buildEditor();

    var resizeHandle = el("div", "msr-resize-handle");
    resizeHandle.title = strings.editor_resize_hint;
    resizeHandle.setAttribute("role", "separator");
    resizeHandle.setAttribute("aria-orientation", "horizontal");
    resizeHandle.onpointerdown = function (event) {
      if (event.button !== 0) return;
      self.resizeDrag = {
        pointerId: event.pointerId,
        startY: event.clientY,
        startHeight: self.rollHeight
      };
      resizeHandle.classList.add("dragging");
      resizeHandle.setPointerCapture(event.pointerId);
      event.preventDefault();
    };
    resizeHandle.onpointermove = function (event) {
      if (!self.resizeDrag || self.resizeDrag.pointerId !== event.pointerId) return;
      self.rollHeight = clamp(
        self.resizeDrag.startHeight + self.resizeDrag.startY - event.clientY,
        240,
        650
      );
      if (self.scroll) self.scroll.style.height = self.rollHeight + "px";
      event.preventDefault();
    };
    var finishResize = function (event) {
      if (!self.resizeDrag || self.resizeDrag.pointerId !== event.pointerId) return;
      self.resizeDrag = null;
      resizeHandle.classList.remove("dragging");
      if (resizeHandle.hasPointerCapture(event.pointerId)) {
        resizeHandle.releasePointerCapture(event.pointerId);
      }
      event.preventDefault();
    };
    resizeHandle.onpointerup = finishResize;
    resizeHandle.onpointercancel = finishResize;
    this.host.appendChild(resizeHandle);
    this.resizeHandle = resizeHandle;

    var grid = el("div", "msr-grid");
    var scroll = el("div", "msr-roll-scroll");
    scroll.style.height = this.rollHeight + "px";
    var world = el("div", "msr-roll-world");
    var viewport = el("div", "msr-roll-viewport");
    this.canvas = el("canvas", "msr-roll");
    this.canvas.tabIndex = 0;
    this.playhead = el("div", "msr-playhead");
    viewport.appendChild(this.canvas);
    viewport.appendChild(this.playhead);
    world.appendChild(viewport);
    scroll.appendChild(world);
    this.scroll = scroll;
    this.world = world;
    this.viewport = viewport;
    this.canvas.onpointerdown = function (event) { self.onPointerDown(event); };
    this.canvas.onpointermove = function (event) { self.onPointerMove(event); };
    this.canvas.onpointerup = function (event) { self.onPointerUp(event); };
    this.canvas.onpointercancel = function (event) { self.onPointerUp(event); };
    this.canvas.ondblclick = function (event) { self.onDoubleClick(event); };
    this.canvas.onkeydown = function (event) { self.onEditorKey(event); };
    scroll.addEventListener("scroll", function () { self.scheduleDraw(); self.layoutPlayhead(); }, { passive: true });
    scroll.addEventListener("wheel", function (event) { self.onWheel(event); }, { passive: false });
    scroll.title = strings.zoom_help;
    grid.appendChild(scroll);

    var aside = el("aside", "msr-instruments");
    this.instrumentAside = aside;
    aside.appendChild(el("h3", "", strings.instruments));
    this.m.instruments.forEach(function (instrument) {
      var row = el("div", "msr-row" + (instrument.detected ? "" : " undetected"));
      row.dataset.instrument = instrument.id;
      var swatch = el("span", "msr-swatch");
      swatch.style.background = instrument.detected ? instrument.color : "#4b5157";
      row.appendChild(swatch);
      row.appendChild(el("span", "msr-name", instrument.label));
      if (!instrument.detected) {
        row.appendChild(el("small", "", strings.not_detected));
      } else {
        var solo = button("S", strings.solo), mute = button("🔊", strings.mute);
        mute.setAttribute("aria-label", strings.mute);
        solo.onclick = function () { self.toggleSolo(instrument.id); };
        mute.onclick = function () { self.toggleMute(instrument.id); };
        row.appendChild(solo);
        row.appendChild(mute);
        instrument.row = row;
        instrument.soloButton = solo;
        instrument.muteButton = mute;
      }
      row.onclick = function (event) {
        if (instrument.detected && !event.target.closest("button")) {
          self.activateInstrument(instrument.id);
        }
      };
      aside.appendChild(row);
    });
    grid.appendChild(aside);
    this.host.appendChild(grid);

    var downloads = el("div", "msr-downloads");
    var editedMidi = button(strings.export_edited_midi);
    editedMidi.onclick = function () { self.downloadEditedMidi(); };
    downloads.appendChild(editedMidi);
    this.sheetMusicButton = button(strings.download_sheet_music);
    this.sheetMusicButton.onclick = function () { self.downloadSheetMusic(); };
    downloads.appendChild(this.sheetMusicButton);
    var audioExport = el("div", "msr-audio-export");
    var audioExportLabel = el("label", "", strings.audio_export_format);
    this.audioExportSelect = el("select");
    this.m.audioExportPresets.forEach(function (preset) {
      var option = el("option", "", preset.label);
      option.value = preset.id;
      if (preset.id === self.m.defaultAudioExportPreset) option.selected = true;
      self.audioExportSelect.appendChild(option);
    });
    audioExportLabel.appendChild(this.audioExportSelect);
    audioExport.appendChild(audioExportLabel);
    this.downloadAudioButton = button(strings.audio_export_start);
    this.downloadAudioButton.onclick = function () { self.downloadTranscriptionAudio(); };
    audioExport.appendChild(this.downloadAudioButton);
    this.downloadStemAudioButton = button(strings.stem_audio_export_start);
    this.downloadStemAudioButton.onclick = function () { self.downloadStemAudio(); };
    audioExport.appendChild(this.downloadStemAudioButton);
    downloads.appendChild(audioExport);
    var stereoAnchor = el("a", "msr-btn", strings.download_stereo);
    stereoAnchor.href = self.m.downloads.stereo;
    stereoAnchor.download = "";
    self.downloadAnchors.stereo = stereoAnchor;
    downloads.appendChild(stereoAnchor);
    this.host.appendChild(downloads);
    this.resizeObserver = new ResizeObserver(function () { self.layout(); });
    this.resizeObserver.observe(scroll);
    this.layout();
  };
  ResultSession.prototype.setDownloadAudioEnabled = function (enabled) {
    this.audioDownloadReady = Boolean(enabled);
    ["stereo"].forEach(function (key) {
      var anchor = this.downloadAnchors[key];
      if (!anchor) return;
      if (enabled) {
        anchor.href = this.m.downloads[key];
        anchor.removeAttribute("aria-disabled");
      } else {
        anchor.removeAttribute("href");
        anchor.setAttribute("aria-disabled", "true");
      }
    }, this);
    if (this.downloadAudioButton) {
      this.downloadAudioButton.disabled = !this.audioDownloadReady || this.audioExportInFlight;
    }
    if (this.downloadStemAudioButton) {
      this.downloadStemAudioButton.disabled = !this.audioDownloadReady || this.audioExportInFlight;
    }
    if (this.audioExportSelect) {
      this.audioExportSelect.disabled = !this.audioDownloadReady || this.audioExportInFlight;
    }
  };
  ResultSession.prototype.syncInstrumentAvailability = function (instrumentUrls) {
    var self = this;
    this.m.instruments.forEach(function (instrument) {
      var detected = Object.prototype.hasOwnProperty.call(instrumentUrls, instrument.id);
      instrument.detected = detected;
      instrument.url = detected ? String(instrumentUrls[instrument.id]) : "";
      if (instrument.row) {
        instrument.row.classList.toggle("undetected", !detected);
      }
      if (instrument.soloButton) instrument.soloButton.disabled = !detected;
      if (instrument.muteButton) instrument.muteButton.disabled = !detected;
      if (!detected) self.muted.delete(instrument.id);
    });
    if (this.solo && !Object.prototype.hasOwnProperty.call(instrumentUrls, this.solo)) {
      this.solo = null;
    }
    this.syncRows();
  };
  ResultSession.prototype.applyPlaybackPreview = function (preview, nextBuffers, statusText) {
    this.buffers = nextBuffers;
    this.m.duration = Number(preview.duration);
    this.progress.max = String(this.m.duration);
    this.position = clamp(this.position, 0, this.m.duration);
    this.durationLabel.textContent = "/ " + this.m.duration.toFixed(1) + "s";
    this.m.downloads.transcription = String(preview.transcriptionUrl);
    this.m.downloads.stereo = String(preview.stereoUrl);
    this.syncInstrumentAvailability(preview.instrumentUrls || {});
    this.setDownloadAudioEnabled(true);
    this.play.disabled = false;
    this.progress.disabled = false;
    this.status.textContent = statusText;
    this.drawStatic();
    this.layoutPlayhead();
  };
  ResultSession.prototype.restoreOriginalPreview = function () {
    if (!this.originalPreview) {
      throw new Error("Original SoundFont preview has not finished loading");
    }
    var instrumentUrls = {};
    this.originalPreview.instruments.forEach(function (instrument) {
      if (instrument.detected && instrument.url) instrumentUrls[instrument.id] = instrument.url;
    });
    this.m.downloads.transcription = this.originalPreview.transcriptionUrl;
    this.m.downloads.stereo = this.originalPreview.stereoUrl;
    this.applyPlaybackPreview(
      {
        duration: this.originalPreview.duration,
        instrumentUrls: instrumentUrls,
        transcriptionUrl: this.originalPreview.transcriptionUrl,
        stereoUrl: this.originalPreview.stereoUrl
      },
      Object.assign({}, this.originalPreview.buffers),
      this.m.strings.ready
    );
  };
  ResultSession.prototype.scheduleEditedPreview = function () {
    var self = this, revision = ++this.previewRevision;
    clearTimeout(this.previewTimer);
    if (this.playing) this.pause();
    this.buffers = this.buffers.original ? { original: this.buffers.original } : {};
    this.play.disabled = true;
    this.progress.disabled = true;
    this.setDownloadAudioEnabled(false);
    if (notesEqual(this.m.notes, this.originalNotes) && this.originalPreview) {
      try {
        this.restoreOriginalPreview();
        this.syncEditor();
      } catch (error) {
        this.status.textContent = this.m.strings.editor_audio_failed
          .replace("{error}", String(error));
      }
      return;
    }
    this.editNotice.textContent = this.m.strings.editor_audio_notice;
    this.status.textContent = this.m.strings.editor_audio_rendering;
    this.previewTimer = setTimeout(function () {
      self.renderEditedPreview(revision);
    }, 300);
  };
  ResultSession.prototype.renderEditedPreview = function (revision) {
    var self = this;
    if (!this.m.previewApi || !this.m.previewToken) {
      this.status.textContent = this.m.strings.editor_audio_failed
        .replace("{error}", "server render context is unavailable");
      return;
    }
    fetch(this.m.previewApi, {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data: [JSON.stringify({ token: this.m.previewToken, notes: this.m.notes })]
      })
    })
      .then(function (response) {
        return response.text().then(function (body) {
          if (!response.ok) throw new Error("HTTP " + response.status + " " + body);
          var envelope = JSON.parse(body);
          if (!envelope.data || envelope.data.length !== 1) {
            throw new Error("Edited preview endpoint returned no result");
          }
          return typeof envelope.data[0] === "string"
            ? JSON.parse(envelope.data[0])
            : envelope.data[0];
        });
      })
      .then(function (preview) {
        if (self.disposed || revision !== self.previewRevision) return null;
        var nextBuffers = {};
        var jobs = [];
        if (self.originalPreview && self.originalPreview.buffers.original) {
          nextBuffers.original = self.originalPreview.buffers.original;
        } else if (self.buffers.original) {
          nextBuffers.original = self.buffers.original;
        }
        Object.keys(preview.instrumentUrls || {}).forEach(function (instrumentId) {
          jobs.push(load(preview.instrumentUrls[instrumentId]).then(function (buffer) {
            nextBuffers[instrumentId] = buffer;
          }));
        });
        return Promise.all(jobs).then(function () {
          return { preview: preview, buffers: nextBuffers };
        });
      })
      .then(function (loaded) {
        if (!loaded || self.disposed || revision !== self.previewRevision) return;
        self.applyPlaybackPreview(
          loaded.preview,
          loaded.buffers,
          self.m.strings.editor_audio_ready
        );
        self.editNotice.textContent = self.m.strings.editor_audio_ready;
      })
      .catch(function (error) {
        if (self.disposed || revision !== self.previewRevision) return;
        self.buffers = self.buffers.original ? { original: self.buffers.original } : {};
        self.play.disabled = true;
        self.progress.disabled = true;
        self.setDownloadAudioEnabled(false);
        self.status.textContent = self.m.strings.editor_audio_failed
          .replace("{error}", String(error));
      });
  };
  ResultSession.prototype.buildEditor = function () {
    var self = this, strings = this.m.strings;
    var editor = el("div", "msr-editor");
    this.editToggle = button(strings.editor_toggle, strings.editor_help);
    this.editToggle.onclick = function () {
      self.editing = !self.editing;
      self.editToggle.classList.toggle("active", self.editing);
      self.canvas.style.cursor = self.editing ? "crosshair" : "crosshair";
      if (!self.editing) self.selectNote(null);
      self.syncEditor();
    };
    editor.appendChild(this.editToggle);
    this.addButton = button(strings.editor_add);
    this.addButton.onclick = function () { self.addNote(self.position, 60); };
    editor.appendChild(this.addButton);
    this.deleteButton = button(strings.editor_delete);
    this.deleteButton.onclick = function () { self.deleteSelected(); };
    editor.appendChild(this.deleteButton);
    this.undoButton = button(strings.editor_undo);
    this.undoButton.onclick = function () { self.undo(); };
    editor.appendChild(this.undoButton);
    this.redoButton = button(strings.editor_redo);
    this.redoButton.onclick = function () { self.redo(); };
    editor.appendChild(this.redoButton);
    this.resetButton = button(strings.editor_reset);
    this.resetButton.onclick = function () { self.resetEdits(); };
    editor.appendChild(this.resetButton);
    this.selectAllButton = button(strings.editor_select_all);
    this.selectAllButton.onclick = function () { self.selectAll(); };
    editor.appendChild(this.selectAllButton);
    this.cutButton = button(strings.editor_cut);
    this.cutButton.onclick = function () { self.cutSelected(); };
    editor.appendChild(this.cutButton);
    this.copyButton = button(strings.editor_copy);
    this.copyButton.onclick = function () { self.copySelected(); };
    editor.appendChild(this.copyButton);
    this.pasteButton = button(strings.editor_paste);
    this.pasteButton.onclick = function () { self.pasteNotes(); };
    editor.appendChild(this.pasteButton);
    this.duplicateButton = button(strings.editor_duplicate);
    this.duplicateButton.onclick = function () { self.duplicateSelected(); };
    editor.appendChild(this.duplicateButton);
    this.quantizeButton = button(strings.editor_quantize);
    this.quantizeButton.onclick = function () { self.quantizeSelected(); };
    editor.appendChild(this.quantizeButton);

    var quantizeScopeLabel = el(
      "label",
      "msr-quantize-scope",
      strings.editor_quantize_scope
    );
    quantizeScopeLabel.title = strings.editor_quantize_scope_tooltip;
    this.quantizeScopeSelect = el("select");
    this.m.quantizeScopes.forEach(function (scope) {
      var label = scope === "all_tracks"
        ? strings.editor_quantize_scope_all_tracks
        : strings.editor_quantize_scope_selected_notes;
      var option = el("option", "", label);
      option.value = scope;
      self.quantizeScopeSelect.appendChild(option);
    });
    this.quantizeScopeSelect.value = this.quantizeScope;
    this.quantizeScopeSelect.title = strings.editor_quantize_scope_tooltip;
    this.quantizeScopeSelect.onchange = function () {
      if (self.m.quantizeScopes.indexOf(this.value) < 0) {
        throw new Error("Unsupported MIDI quantization scope: " + this.value);
      }
      self.quantizeScope = this.value;
      self.syncEditor();
    };
    quantizeScopeLabel.appendChild(this.quantizeScopeSelect);
    editor.appendChild(quantizeScopeLabel);

    var quantizeGridLabel = el(
      "label",
      "msr-quantize-grid",
      strings.editor_quantize_grid
    );
    quantizeGridLabel.title = strings.editor_quantize_grid_tooltip;
    this.quantizeGridSelect = el("select");
    this.m.quantizeGrids.forEach(function (grid) {
      var option = el("option", "", grid);
      option.value = grid;
      self.quantizeGridSelect.appendChild(option);
    });
    this.quantizeGridSelect.value = this.quantizeGrid;
    this.quantizeGridSelect.title = strings.editor_quantize_grid_tooltip;
    this.quantizeGridSelect.onchange = function () {
      if (self.m.quantizeGrids.indexOf(this.value) < 0) {
        throw new Error("Unsupported MIDI quantization grid: " + this.value);
      }
      self.quantizeGrid = this.value;
      self.drawStatic();
    };
    quantizeGridLabel.appendChild(this.quantizeGridSelect);
    editor.appendChild(quantizeGridLabel);

    var instrumentLabel = el("label", "", strings.editor_instrument);
    this.instrumentSelect = el("select");
    this.m.instruments.filter(function (instrument) { return instrument.detected; }).forEach(function (instrument) {
      var option = el("option", "", instrument.label);
      option.value = instrument.id;
      self.instrumentSelect.appendChild(option);
    });
    this.instrumentSelect.value = this.activeInstrument;
    this.instrumentSelect.onchange = function () { self.changeInstrument(this.value); };
    instrumentLabel.appendChild(this.instrumentSelect);
    editor.appendChild(instrumentLabel);

    var velocityLabel = el("label", "", strings.editor_velocity);
    this.velocityInput = el("input");
    this.velocityInput.type = "number";
    this.velocityInput.min = "1";
    this.velocityInput.max = "127";
    this.velocityInput.step = "1";
    this.velocityInput.value = "100";
    this.velocityInput.onchange = function () { self.changeVelocity(Number(this.value)); };
    velocityLabel.appendChild(this.velocityInput);
    editor.appendChild(velocityLabel);

    var zoomLabel = el("label", "", strings.editor_view_zoom);
    this.zoomInput = el("input", "msr-zoom-input");
    this.zoomInput.type = "number";
    this.zoomInput.min = String(MIN_PPS / BASE_PPS);
    this.zoomInput.max = String(MAX_PPS / BASE_PPS);
    this.zoomInput.step = "0.25";
    this.zoomInput.value = "1.00";
    this.zoomInput.title = strings.editor_view_zoom_tooltip;
    this.zoomInput.onchange = function () {
      self.setZoomRatio(Number(this.value), self.scroll.clientWidth / 2);
    };
    zoomLabel.appendChild(this.zoomInput);
    zoomLabel.appendChild(document.createTextNode("×"));
    editor.appendChild(zoomLabel);

    var bpmLabel = el("label", "", "BPM");
    this.bpmInput = el("input");
    this.bpmInput.type = "number";
    this.bpmInput.min = String(MIN_BPM);
    this.bpmInput.max = String(MAX_BPM);
    this.bpmInput.step = "0.1";
    this.bpmInput.value = this.targetBpm.toFixed(1);
    this.bpmInput.onchange = function () {
      try {
        self.commitBpm();
      } catch (error) {
        self.status.textContent = strings.editor_export_failed.replace("{error}", String(error));
      }
    };
    bpmLabel.appendChild(this.bpmInput);
    editor.appendChild(bpmLabel);
    this.editSummary = el("span", "msr-edit-summary");
    editor.appendChild(this.editSummary);
    this.editNotice = el("span", "msr-edit-notice", strings.editor_audio_notice);
    editor.appendChild(this.editNotice);
    this.host.appendChild(editor);
    this.syncEditor();
  };
  ResultSession.prototype.playbackRate = function () {
    return projectPlaybackRate(this.m.referenceBpm, this.targetBpm);
  };
  ResultSession.prototype.commitBpm = function () {
    var nextBpm = Number(this.bpmInput.value);
    if (!Number.isFinite(nextBpm) || nextBpm < MIN_BPM || nextBpm > MAX_BPM) {
      throw new Error("Invalid MIDI BPM: " + this.bpmInput.value);
    }
    var wasPlaying = this.playing;
    if (wasPlaying) this.pause();
    this.targetBpm = nextBpm;
    this.drawStatic();
    if (wasPlaying) {
      var self = this;
      this.start().catch(function (error) {
        self.status.textContent = self.m.strings.editor_export_failed
          .replace("{error}", String(error));
      });
    }
  };
  ResultSession.prototype.gridSeconds = function () {
    var referenceBpm = Number(this.m.referenceBpm);
    if (!Number.isFinite(referenceBpm) || referenceBpm <= 0) {
      throw new Error("Cannot derive MIDI editor grid from reference BPM " + referenceBpm);
    }
    var match = /^1\/(4|8|16|32|64)$/.exec(this.quantizeGrid);
    if (!match || this.m.quantizeGrids.indexOf(this.quantizeGrid) < 0) {
      throw new Error("Unsupported MIDI quantization grid: " + this.quantizeGrid);
    }
    return 60 / referenceBpm * 4 / Number(match[1]);
  };
  ResultSession.prototype.snapTime = function (value) {
    var grid = this.gridSeconds();
    return Math.round(Math.max(0, value) / grid) * grid;
  };
  ResultSession.prototype.stopSources = function () {
    this.sources.forEach(function (source) {
      try { source.stop(); } catch (_error) {}
      try { source.disconnect(); } catch (_error) {}
    });
    this.sources = [];
    this.gains = {};
    this.panners = {};
  };
  ResultSession.prototype.start = function () {
    var context = ctx(), self = this, rate = this.playbackRate();
    if (this.position >= this.m.duration) this.position = 0;
    window.dispatchEvent(new CustomEvent("music-to-midi-playback-start", { detail: { owner: this.ownerId } }));
    return context.resume().then(function () {
      if (self.disposed) return;
      self.stopSources();
      var startAt = context.currentTime + 0.02;
      self.startedAt = startAt - self.position / rate;
      ["original"].concat(self.m.instruments.filter(function (instrument) {
        return instrument.detected;
      }).map(function (instrument) {
        return instrument.id;
      })).forEach(function (id) {
        var buffer = self.buffers[id];
        if (!buffer || self.position >= buffer.duration) return;
        var source = context.createBufferSource(), gain = context.createGain();
        var pan = context.createStereoPanner();
        source.buffer = buffer;
        source.playbackRate.value = rate;
        source.connect(gain);
        gain.connect(pan);
        pan.connect(context.destination);
        source.start(startAt, self.position);
        self.sources.push(source);
        self.gains[id] = gain;
        self.panners[id] = pan;
      });
      self.playing = true;
      self.applyMix();
      self.play.textContent = self.m.strings.pause;
      self.tick();
    });
  };
  ResultSession.prototype.pause = function () {
    if (this.playing) {
      this.position = Math.min(
        this.m.duration,
        (ctx().currentTime - this.startedAt) * this.playbackRate()
      );
    }
    this.playing = false;
    this.stopSources();
    this.play.textContent = this.m.strings.play;
    cancelAnimationFrame(this.raf);
    this.layoutPlayhead();
  };
  ResultSession.prototype.toggle = function () { if (this.playing) this.pause(); else this.start(); };
  ResultSession.prototype.seek = function (seconds) {
    var wasPlaying = this.playing;
    if (wasPlaying) this.pause();
    this.position = clamp(seconds, 0, this.m.duration);
    if (wasPlaying) this.start(); else this.layoutPlayhead();
  };
  ResultSession.prototype.handleExternalPlayback = function (event) {
    if (event.detail && event.detail.owner !== this.ownerId) this.pause();
  };
  ResultSession.prototype.audible = function (id) { return !this.muted.has(id); };
  ResultSession.prototype.applyMix = function () {
    var context = ctx(), time = context.currentTime, self = this;
    if (this.gains.original) {
      this.gains.original.gain.setTargetAtTime(this.stereo ? 1 : 1 - this.mix, time, 0.01);
      this.panners.original.pan.setTargetAtTime(this.stereo ? -1 : 0, time, 0.01);
    }
    this.m.instruments.forEach(function (instrument) {
      if (!self.gains[instrument.id]) return;
      self.gains[instrument.id].gain.setTargetAtTime(
        self.audible(instrument.id) ? (self.stereo ? 1 : self.mix) : 0,
        time,
        0.01
      );
      self.panners[instrument.id].pan.setTargetAtTime(self.stereo ? 1 : 0, time, 0.01);
    });
  };
  ResultSession.prototype.toggleMute = function (id) {
    this.activateInstrument(id);
    this.solo = null;
    if (this.muted.has(id)) this.muted.delete(id); else this.muted.add(id);
    this.syncRows();
  };
  ResultSession.prototype.toggleSolo = function (id) {
    this.activateInstrument(id);
    if (this.solo === id) {
      this.solo = null;
      this.muted.clear();
    } else {
      this.solo = id;
      this.muted = new Set(this.m.instruments.filter(function (instrument) {
        return instrument.detected && instrument.id !== id;
      }).map(function (instrument) {
        return instrument.id;
      }));
    }
    this.syncRows();
  };
  ResultSession.prototype.syncRows = function () {
    var self = this;
    this.m.instruments.forEach(function (instrument) {
      if (!instrument.detected) return;
      var muted = self.muted.has(instrument.id);
      instrument.row.classList.toggle("active-instrument", self.activeInstrument === instrument.id);
      instrument.row.classList.toggle("muted", muted);
      instrument.soloButton.classList.toggle("active", self.solo === instrument.id);
      instrument.muteButton.classList.toggle("active", muted);
      instrument.muteButton.textContent = "🔊";
    });
    this.applyMix();
    this.drawStatic();
  };
  ResultSession.prototype.activateInstrument = function (instrument) {
    this.activeInstrument = instrument;
    if (this.instrumentSelect) this.instrumentSelect.value = instrument;
    this.syncRows();
    this.syncEditor();
  };
  ResultSession.prototype.tick = function () {
    if (!this.playing) return;
    this.position = Math.min(
      this.m.duration,
      Math.max(this.position, (ctx().currentTime - this.startedAt) * this.playbackRate())
    );
    if(this.position>=this.m.duration){this.position=this.m.duration;this.pause();return;}
    if (this.follow) {
      var target = LEFT + this.position * this.pps - this.scroll.clientWidth / 2;
      this.scroll.scrollLeft = clamp(
        target,
        0,
        Math.max(0, this.world.clientWidth - this.scroll.clientWidth)
      );
    }
    this.layoutPlayhead();
    var self = this;
    this.raf = requestAnimationFrame(function () { self.tick(); });
  };
  ResultSession.prototype.layout = function () {
    var width = Math.max(320, this.scroll.clientWidth || 950);
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    this.viewport.style.width = width + "px";
    this.world.style.width = Math.max(width, LEFT + this.m.duration * this.pps + 80) + "px";
    this.canvas.style.width = width + "px";
    this.canvas.style.height = HEIGHT + "px";
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(HEIGHT * dpr);
    this.dpr = dpr;
    this.drawStatic();
    this.layoutPlayhead();
  };
  ResultSession.prototype.scheduleDraw = function () {
    var self = this;
    if (this.drawRaf) return;
    this.drawRaf = requestAnimationFrame(function () {
      self.drawRaf = 0;
      self.drawStatic();
    });
  };
  ResultSession.prototype.onWheel = function (e) {
    var modifier=e.ctrlKey||e.altKey;
    if (modifier) {
      e.preventDefault();
      var rect = this.scroll.getBoundingClientRect();
      var anchorX = clamp(e.clientX - rect.left, 0, this.scroll.clientWidth);
      var factor = e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
      this.setZoomPps(this.pps * factor, anchorX);
      return;
    }
    if(e.shiftKey){
      e.preventDefault();
      this.scroll.scrollLeft += e.deltaY || e.deltaX;
    }
  };
  ResultSession.prototype.setZoomRatio = function (ratio, anchorX) {
    if (!Number.isFinite(ratio) || ratio <= 0) {
      throw new Error("Piano-roll zoom ratio must be positive: " + ratio);
    }
    this.setZoomPps(BASE_PPS * ratio, anchorX);
  };
  ResultSession.prototype.setZoomPps = function (pixelsPerSecond, anchorX) {
    if (!Number.isFinite(pixelsPerSecond) || pixelsPerSecond <= 0) {
      throw new Error("Piano-roll zoom must be positive: " + pixelsPerSecond);
    }
    var resolvedAnchor = clamp(
      Number.isFinite(anchorX) ? anchorX : this.scroll.clientWidth / 2,
      0,
      this.scroll.clientWidth
    );
    var anchorTime = (this.scroll.scrollLeft + resolvedAnchor - LEFT) / this.pps;
    this.pps = clamp(pixelsPerSecond, MIN_PPS, MAX_PPS);
    this.zoomInput.value = (this.pps / BASE_PPS).toFixed(2);
    this.world.style.width = Math.max(
      this.scroll.clientWidth,
      LEFT + this.m.duration * this.pps + 80
    ) + "px";
    this.scroll.scrollLeft = Math.max(
      0,
      LEFT + anchorTime * this.pps - resolvedAnchor
    );
    this.drawStatic();
    this.layoutPlayhead();
  };
  ResultSession.prototype.drawStatic = function () {
    if (!this.canvas) return;
    var painter = this.canvas.getContext("2d"), dpr = this.dpr || 1;
    var width = this.canvas.width / dpr, scrollX = this.scroll.scrollLeft;
    var start = Math.max(0, (scrollX - LEFT) / this.pps);
    var end = Math.min(this.m.duration, (scrollX + width - LEFT) / this.pps);
    painter.setTransform(dpr, 0, 0, dpr, 0, 0);
    painter.fillStyle = "#0f1a2d";
    painter.fillRect(0, 0, width, HEIGHT);
    for (var pitch = 21; pitch <= 108; pitch++) {
      var y = (108 - pitch) * ROW;
      var black = [1, 3, 6, 8, 10].indexOf(pitch % 12) >= 0;
      painter.fillStyle = black ? "#13213a" : "#172842";
      painter.fillRect(LEFT, y, width - LEFT, ROW);
      painter.strokeStyle = "#2b3d5c";
      painter.beginPath();
      painter.moveTo(LEFT, y);
      painter.lineTo(width, y);
      painter.stroke();
      painter.fillStyle = black ? "#23282e" : "#e4e8eb";
      painter.fillRect(0, y, LEFT, ROW);
      if (pitch % 12 === 0) {
        painter.fillStyle = black ? "#ddd" : "#222";
        painter.font = "7px monospace";
        painter.fillText("C" + (Math.floor(pitch / 12) - 1), 3, y + 6);
      }
    }
    var self = this;
    var referenceBpm = Number(this.m.referenceBpm);
    var numerator = this.m.timeSignature[0], denominator = this.m.timeSignature[1];
    var quarterSeconds = 60 / referenceBpm;
    var subdivisionSeconds = this.gridSeconds();
    var beatSeconds = quarterSeconds * 4 / denominator;
    var barSeconds = beatSeconds * numerator;
    var firstBar = Math.max(0, Math.floor(start / barSeconds));
    var lastBar = Math.ceil(end / barSeconds) + 1;
    for (var barIndex = firstBar; barIndex <= lastBar; barIndex++) {
      var barTime = barIndex * barSeconds;
      var barX = LEFT + barTime * this.pps - scrollX;
      if (barIndex % 2 === 1) {
        var barRight = LEFT + (barTime + barSeconds) * this.pps - scrollX;
        painter.fillStyle = "rgba(255,255,255,0.03)";
        painter.fillRect(barX, 0, Math.max(0, barRight - barX), HEIGHT);
      }
    }
    var subdivisionStride = 1;
    while (subdivisionSeconds * this.pps * subdivisionStride < 5) subdivisionStride *= 2;
    var firstSubdivision = Math.max(0, Math.floor(start / subdivisionSeconds) - 1);
    var lastSubdivision = Math.ceil(end / subdivisionSeconds) + 1;
    painter.strokeStyle = "#263d59";
    painter.lineWidth = 1;
    for (var subdivisionIndex = firstSubdivision;
         subdivisionIndex <= lastSubdivision;
         subdivisionIndex += subdivisionStride) {
      var subdivisionX = LEFT + subdivisionIndex * subdivisionSeconds * this.pps - scrollX;
      painter.beginPath();
      painter.moveTo(subdivisionX, 0);
      painter.lineTo(subdivisionX, HEIGHT);
      painter.stroke();
    }
    var firstBeat = Math.max(0, Math.floor(start / beatSeconds) - 1);
    var lastBeat = Math.ceil(end / beatSeconds) + 1;
    var labelBeats = beatSeconds * this.pps >= 38;
    for (var beatIndex = firstBeat; beatIndex <= lastBeat; beatIndex++) {
      var beatInBar = beatIndex % numerator;
      if (beatInBar === 0) continue;
      var beatX = LEFT + beatIndex * beatSeconds * this.pps - scrollX;
      painter.strokeStyle = "#36506f";
      painter.beginPath();
      painter.moveTo(beatX, 0);
      painter.lineTo(beatX, HEIGHT);
      painter.stroke();
      if (labelBeats) {
        painter.fillStyle = "#7f9dbd";
        painter.font = "8px monospace";
        painter.fillText((Math.floor(beatIndex / numerator) + 1) + "." + (beatInBar + 1), beatX + 3, 11);
      }
    }
    for (var downbeatIndex = firstBar; downbeatIndex <= lastBar; downbeatIndex++) {
      var downbeatX = LEFT + downbeatIndex * barSeconds * this.pps - scrollX;
      painter.strokeStyle = "#78aee8";
      painter.lineWidth = 1.5;
      painter.beginPath();
      painter.moveTo(downbeatX, 0);
      painter.lineTo(downbeatX, HEIGHT);
      painter.stroke();
      painter.fillStyle = "#a9c8e8";
      painter.font = "8px monospace";
      painter.fillText((downbeatIndex + 1) + ".1", downbeatX + 3, 11);
    }
    painter.lineWidth = 1;
    this.m.notes.forEach(function (note, index) {
      if (note.pitch < 21 || note.pitch > 108 || note.end < start || note.start > end) return;
      var x = LEFT + note.start * self.pps - scrollX;
      var noteY = (108 - note.pitch) * ROW + 1;
      var noteWidth = Math.max(2, (note.end - note.start) * self.pps);
      var instrument = self.m.instruments.find(function (item) { return item.id === note.instrument; });
      painter.globalAlpha = self.muted.has(note.instrument) ? 0.12 : 1;
      painter.fillStyle = instrument ? instrument.color : "#4a9eff";
      painter.fillRect(x, noteY, noteWidth, ROW - 2);
      if (self.selectedIndices.has(index)) {
        painter.globalAlpha = 1;
        painter.strokeStyle = index === self.selectedIndex ? "#ffffff" : "#b7d9ff";
        painter.lineWidth = 1.5;
        painter.strokeRect(x, noteY, noteWidth, ROW - 2);
      }
    });
    painter.globalAlpha = 1;
    if (this.drag && this.drag.mode === "marquee" && this.drag.current) {
      var x1 = this.drag.originX - scrollX, x2 = this.drag.current.logicalX - scrollX;
      var y1 = this.drag.originY, y2 = this.drag.current.y;
      painter.fillStyle = "rgba(74,158,255,.15)";
      painter.strokeStyle = "#9fc9ff";
      painter.setLineDash([4, 3]);
      painter.fillRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
      painter.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
      painter.setLineDash([]);
    }
  };
  ResultSession.prototype.layoutPlayhead = function () {
    if (!this.playhead) return;
    var x = LEFT + this.position * this.pps - this.scroll.scrollLeft;
    this.playhead.style.transform="translate3d("+x.toFixed(2)+"px,0,0)";
    this.playhead.style.visibility = (x >= LEFT && x <= this.scroll.clientWidth) ? "visible" : "hidden";
    this.clock.textContent = this.position.toFixed(1) + "s";
    if (this.progress) this.progress.value = String(this.position);
  };
  ResultSession.prototype.pointerCoordinates = function (event) {
    var rect = this.canvas.getBoundingClientRect();
    return {
      logicalX: this.scroll.scrollLeft + event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  };
  ResultSession.prototype.noteAt = function (logicalX, y) {
    var pitch = 108 - Math.floor(y / ROW);
    for (var index = this.m.notes.length - 1; index >= 0; index--) {
      var note = this.m.notes[index];
      if (note.pitch !== pitch) continue;
      var left = LEFT + note.start * this.pps, right = LEFT + note.end * this.pps;
      if (logicalX >= left - 2 && logicalX <= Math.max(left + 2, right) + 2) return index;
    }
    return null;
  };
  ResultSession.prototype.selectNotes = function (indices, primary) {
    var self = this;
    this.selectedIndices = new Set(
      Array.from(indices || []).map(Number).filter(function (index) {
        return Number.isInteger(index) && index >= 0 && index < self.m.notes.length;
      })
    );
    this.selectedIndex = (
      primary !== null && primary !== undefined && this.selectedIndices.has(Number(primary))
    ) ? Number(primary) : (this.selectedIndices.size ? Math.min.apply(null, Array.from(this.selectedIndices)) : null);
    if (this.selectedIndex !== null) {
      var note = this.m.notes[this.selectedIndex];
      this.activateInstrument(note.instrument);
      this.velocityInput.value = String(note.velocity);
    }
    this.syncEditor();
    this.drawStatic();
  };
  ResultSession.prototype.selectNote = function (index) {
    this.selectNotes(index === null ? [] : [Number(index)], index);
  };
  ResultSession.prototype.selectAll = function () {
    if (!this.editing) return;
    var indices = this.m.notes.map(function (_note, index) { return index; });
    this.selectNotes(indices, indices.length ? indices[0] : null);
  };
  ResultSession.prototype.onPointerDown = function (event) {
    if (event.button !== 0) return;
    this.canvas.focus();
    var point = this.pointerCoordinates(event);
    if (!this.editing) {
      this.seek((point.logicalX - LEFT) / this.pps);
      return;
    }
    var index = this.noteAt(point.logicalX, point.y);
    if (index === null) {
      var base = (event.ctrlKey || event.metaKey || event.shiftKey)
        ? new Set(this.selectedIndices) : new Set();
      if (!base.size) this.selectNotes([], null);
      this.drag = {
        pointerId: event.pointerId,
        mode: "marquee",
        originX: point.logicalX,
        originY: point.y,
        current: point,
        base: base,
        before: cloneNotes(this.m.notes)
      };
      this.canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    var additive = event.ctrlKey || event.metaKey || event.shiftKey;
    if (additive) {
      var selected = new Set(this.selectedIndices);
      if (selected.has(index)) {
        selected.delete(index);
        this.selectNotes(selected, null);
        event.preventDefault();
        return;
      }
      selected.add(index);
      this.selectNotes(selected, index);
    } else if (!this.selectedIndices.has(index)) {
      this.selectNote(index);
    } else {
      this.selectNotes(this.selectedIndices, index);
    }
    var note = this.m.notes[index], left = LEFT + note.start * this.pps;
    var right = LEFT + note.end * this.pps;
    var edge = Math.min(6, Math.max(3, (right - left) / 3));
    var mode = Math.abs(point.logicalX - left) <= edge ? "start" :
      (Math.abs(point.logicalX - right) <= edge ? "end" : "move");
    this.drag = {
      pointerId: event.pointerId,
      mode: mode,
      originX: point.logicalX,
      originY: point.y,
      note: Object.assign({}, note),
      notes: Array.from(this.selectedIndices).map(function (selectedIndex) {
        return { index: selectedIndex, note: Object.assign({}, this.m.notes[selectedIndex]) };
      }, this),
      before: cloneNotes(this.m.notes)
    };
    this.canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  };
  ResultSession.prototype.onPointerMove = function (event) {
    if (!this.drag || this.drag.pointerId !== event.pointerId) return;
    if (this.drag.mode === "marquee") {
      var marqueePoint = this.pointerCoordinates(event);
      this.drag.current = marqueePoint;
      var left = Math.min(this.drag.originX, marqueePoint.logicalX);
      var right = Math.max(this.drag.originX, marqueePoint.logicalX);
      var top = Math.min(this.drag.originY, marqueePoint.y);
      var bottom = Math.max(this.drag.originY, marqueePoint.y);
      var marqueeSelected = new Set(this.drag.base);
      this.m.notes.forEach(function (note, index) {
        var noteLeft = LEFT + note.start * this.pps;
        var noteRight = LEFT + note.end * this.pps;
        var noteTop = (108 - note.pitch) * ROW + 1;
        var noteBottom = noteTop + ROW - 2;
        if (noteRight >= left && noteLeft <= right && noteBottom >= top && noteTop <= bottom) {
          marqueeSelected.add(index);
        }
      }, this);
      this.selectedIndices = marqueeSelected;
      this.selectedIndex = marqueeSelected.size ? Math.min.apply(null, Array.from(marqueeSelected)) : null;
      this.syncEditor();
      this.drawStatic();
      event.preventDefault();
      return;
    }
    var point = this.pointerCoordinates(event), origin = this.drag.note;
    var delta = (point.logicalX - this.drag.originX) / this.pps;
    if (this.drag.mode === "move") {
      var start = origin.start + delta;
      if (!event.altKey) start = this.snapTime(start);
      delta = start - origin.start;
      var minStart = Math.min.apply(null, this.drag.notes.map(function (item) { return item.note.start; }));
      var maxEnd = Math.max.apply(null, this.drag.notes.map(function (item) { return item.note.end; }));
      delta = clamp(delta, -minStart, this.m.duration - maxEnd);
      var pitchDelta = Math.round((this.drag.originY - point.y) / ROW);
      var minPitch = Math.min.apply(null, this.drag.notes.map(function (item) { return item.note.pitch; }));
      var maxPitch = Math.max.apply(null, this.drag.notes.map(function (item) { return item.note.pitch; }));
      pitchDelta = clamp(pitchDelta, 21 - minPitch, 108 - maxPitch);
      this.drag.notes.forEach(function (item) {
        this.m.notes[item.index] = Object.assign({}, item.note, {
          start: item.note.start + delta,
          end: item.note.end + delta,
          pitch: item.note.pitch + pitchDelta
        });
      }, this);
    } else if (this.drag.mode === "start") {
      var nextStart = origin.start + delta;
      if (!event.altKey) nextStart = this.snapTime(nextStart);
      delta = nextStart - origin.start;
      var earliest = Math.min.apply(null, this.drag.notes.map(function (item) { return item.note.start; }));
      var shortest = Math.min.apply(null, this.drag.notes.map(function (item) {
        return item.note.end - item.note.start - 0.01;
      }));
      delta = clamp(delta, -earliest, shortest);
      this.drag.notes.forEach(function (item) {
        this.m.notes[item.index] = Object.assign({}, item.note, {
          start: item.note.start + delta
        });
      }, this);
    } else {
      var nextEnd = origin.end + delta;
      if (!event.altKey) nextEnd = this.snapTime(nextEnd);
      delta = nextEnd - origin.end;
      var shortestDuration = Math.min.apply(null, this.drag.notes.map(function (item) {
        return item.note.end - item.note.start - 0.01;
      }));
      var latest = Math.max.apply(null, this.drag.notes.map(function (item) { return item.note.end; }));
      delta = clamp(delta, -shortestDuration, this.m.duration - latest);
      this.drag.notes.forEach(function (item) {
        this.m.notes[item.index] = Object.assign({}, item.note, {
          end: item.note.end + delta
        });
      }, this);
    }
    if (this.selectedIndex !== null) {
      this.velocityInput.value = String(this.m.notes[this.selectedIndex].velocity);
    }
    this.drawStatic();
    event.preventDefault();
  };
  ResultSession.prototype.onPointerUp = function (event) {
    if (!this.drag || this.drag.pointerId !== event.pointerId) return;
    var drag = this.drag, before = drag.before;
    this.drag = null;
    if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
    if (drag.mode === "marquee") {
      var distance = Math.abs(drag.current.logicalX - drag.originX) +
        Math.abs(drag.current.y - drag.originY);
      if (distance < 4) this.seek((drag.current.logicalX - LEFT) / this.pps);
      this.syncEditor();
      this.drawStatic();
    } else {
      this.commitEdit(before);
    }
    event.preventDefault();
  };
  ResultSession.prototype.onDoubleClick = function (event) {
    if (!this.editing || event.button !== 0) return;
    var point = this.pointerCoordinates(event);
    if (this.noteAt(point.logicalX, point.y) !== null) return;
    this.addNote(
      this.snapTime((point.logicalX - LEFT) / this.pps),
      clamp(108 - Math.floor(point.y / ROW), 21, 108)
    );
    event.preventDefault();
  };
  ResultSession.prototype.onEditorKey = function (event) {
    if (!this.editing) return;
    var command = event.ctrlKey || event.metaKey, key = event.key.toLowerCase();
    if (command && key === "z" && !event.shiftKey) {
      this.undo();
      event.preventDefault();
    } else if (
      (command && key === "y") ||
      (command && event.shiftKey && key === "z")
    ) {
      this.redo();
      event.preventDefault();
    } else if (command && key === "a") {
      this.selectAll();
      event.preventDefault();
    } else if (command && key === "x") {
      this.cutSelected();
      event.preventDefault();
    } else if (command && key === "c") {
      this.copySelected();
      event.preventDefault();
    } else if (command && key === "v") {
      this.pasteNotes();
      event.preventDefault();
    } else if (command && (key === "b" || key === "d")) {
      this.duplicateSelected();
      event.preventDefault();
    } else if ((event.altKey && key === "q") || (command && key === "u")) {
      this.quantizeSelected();
      event.preventDefault();
    } else if (["arrowleft", "arrowright", "arrowup", "arrowdown"].indexOf(key) >= 0) {
      if (key === "arrowleft" || key === "arrowright") {
        this.transformSelected(
          event.shiftKey ? "resize_time" : "move_time",
          key === "arrowleft" ? -1 : 1
        );
      } else if (command) {
        this.transformSelected("velocity", key === "arrowup" ? 1 : -1);
      } else {
        this.transformSelected(
          "pitch",
          (key === "arrowup" ? 1 : -1) * (event.shiftKey ? 12 : 1)
        );
      }
      event.preventDefault();
    } else if (event.key === "Delete" || event.key === "Backspace") {
      this.deleteSelected();
      event.preventDefault();
    } else if (event.key === "Escape") {
      this.selectNote(null);
      event.preventDefault();
    }
  };
  ResultSession.prototype.templateForInstrument = function (instrument) {
    var template = this.m.notes.concat(this.originalNotes).find(function (note) {
      return note.instrument === instrument;
    });
    if (!template) throw new Error("No MIDI channel template for instrument " + instrument);
    return template;
  };
  ResultSession.prototype.addNote = function (start, pitch) {
    if (!this.editing || !this.activeInstrument) return;
    var before = cloneNotes(this.m.notes);
    var template = this.templateForInstrument(this.activeInstrument);
    var noteStart = clamp(Number(start), 0, this.m.duration - 0.01);
    var noteEnd = Math.min(this.m.duration, noteStart + Math.min(0.5, this.m.duration));
    if (noteEnd - noteStart < 0.01) {
      noteStart = Math.max(0, this.m.duration - 0.01);
      noteEnd = this.m.duration;
    }
    this.m.notes.push(Object.assign({}, template, {
      pitch: clamp(Math.round(pitch), 21, 108),
      velocity: clamp(Math.round(Number(this.velocityInput.value) || 100), 1, 127),
      start: noteStart,
      end: noteEnd
    }));
    this.selectedIndices = new Set([this.m.notes.length - 1]);
    this.selectedIndex = this.m.notes.length - 1;
    this.commitEdit(before);
  };
  ResultSession.prototype.deleteSelected = function () {
    if (!this.editing || !this.selectedIndices.size) return;
    var before = cloneNotes(this.m.notes);
    var selected = this.selectedIndices;
    var first = Math.min.apply(null, Array.from(selected));
    this.m.notes = this.m.notes.filter(function (_note, index) { return !selected.has(index); });
    this.selectedIndex = this.m.notes.length ? Math.min(first, this.m.notes.length - 1) : null;
    this.selectedIndices = this.selectedIndex === null ? new Set() : new Set([this.selectedIndex]);
    this.commitEdit(before);
  };
  ResultSession.prototype.copySelected = function () {
    var self = this;
    if (!this.editing || !this.selectedIndices.size) return;
    this.clipboard = Array.from(this.selectedIndices).sort(function (a, b) { return a - b; })
      .map(function (index) { return Object.assign({}, self.m.notes[index]); });
    this.syncEditor();
  };
  ResultSession.prototype.cutSelected = function () {
    if (!this.selectedIndices.size) return;
    this.copySelected();
    this.deleteSelected();
  };
  ResultSession.prototype.pasteNotes = function () {
    if (!this.editing || !this.clipboard.length) return;
    var before = cloneNotes(this.m.notes);
    var sourceStart = Math.min.apply(null, this.clipboard.map(function (note) { return note.start; }));
    var sourceEnd = Math.max.apply(null, this.clipboard.map(function (note) { return note.end; }));
    var span = sourceEnd - sourceStart;
    var targetStart = clamp(this.position, 0, this.m.duration - span);
    var offset = targetStart - sourceStart, first = this.m.notes.length;
    this.clipboard.forEach(function (note) {
      this.m.notes.push(Object.assign({}, note, {
        start: note.start + offset,
        end: note.end + offset
      }));
    }, this);
    this.selectedIndices = new Set(this.clipboard.map(function (_note, index) { return first + index; }));
    this.selectedIndex = first;
    this.commitEdit(before);
  };
  ResultSession.prototype.duplicateSelected = function () {
    var self = this;
    if (!this.editing || !this.selectedIndices.size) return;
    var selected = Array.from(this.selectedIndices).sort(function (a, b) { return a - b; })
      .map(function (index) { return Object.assign({}, self.m.notes[index]); });
    var selectionStart = Math.min.apply(null, selected.map(function (note) { return note.start; }));
    var selectionEnd = Math.max.apply(null, selected.map(function (note) { return note.end; }));
    var offset = Math.max(this.gridSeconds(), selectionEnd - selectionStart);
    offset = Math.min(offset, this.m.duration - selectionEnd);
    if (offset <= 0) return;
    var before = cloneNotes(this.m.notes), first = this.m.notes.length;
    selected.forEach(function (note) {
      this.m.notes.push(Object.assign({}, note, {
        start: note.start + offset,
        end: note.end + offset
      }));
    }, this);
    this.selectedIndices = new Set(selected.map(function (_note, index) { return first + index; }));
    this.selectedIndex = first;
    this.commitEdit(before);
  };
  ResultSession.prototype.quantizeSelected = function () {
    if (!this.editing) return;
    var indices = this.quantizeScope === "all_tracks"
      ? this.m.notes.map(function (_note, index) { return index; })
      : Array.from(this.selectedIndices);
    if (!indices.length) return;
    var before = cloneNotes(this.m.notes), grid = this.gridSeconds();
    indices.forEach(function (index) {
      var note = this.m.notes[index], duration = note.end - note.start;
      var quantizedDuration = Math.max(grid, Math.round(duration / grid) * grid);
      quantizedDuration = Math.min(this.m.duration, quantizedDuration);
      var start = clamp(
        Math.round(note.start / grid) * grid,
        0,
        this.m.duration - quantizedDuration
      );
      this.m.notes[index] = Object.assign({}, note, {
        start: start,
        end: start + quantizedDuration
      });
    }, this);
    this.commitEdit(before);
  };
  ResultSession.prototype.transformSelected = function (command, amount) {
    if (!this.editing || !this.selectedIndices.size) return;
    var self = this, before = cloneNotes(this.m.notes);
    var selected = Array.from(this.selectedIndices).map(function (index) { return self.m.notes[index]; });
    if (command === "move_time") {
      var delta = this.gridSeconds() * amount;
      delta = clamp(
        delta,
        -Math.min.apply(null, selected.map(function (note) { return note.start; })),
        this.m.duration - Math.max.apply(null, selected.map(function (note) { return note.end; }))
      );
      this.selectedIndices.forEach(function (index) {
        var note = this.m.notes[index];
        this.m.notes[index] = Object.assign({}, note, {
          start: note.start + delta, end: note.end + delta
        });
      }, this);
    } else if (command === "resize_time") {
      var durationDelta = this.gridSeconds() * amount;
      durationDelta = clamp(
        durationDelta,
        -Math.min.apply(null, selected.map(function (note) { return note.end - note.start - 0.01; })),
        this.m.duration - Math.max.apply(null, selected.map(function (note) { return note.end; }))
      );
      this.selectedIndices.forEach(function (index) {
        var note = this.m.notes[index];
        this.m.notes[index] = Object.assign({}, note, { end: note.end + durationDelta });
      }, this);
    } else if (command === "pitch") {
      var pitchDelta = clamp(
        amount,
        21 - Math.min.apply(null, selected.map(function (note) { return note.pitch; })),
        108 - Math.max.apply(null, selected.map(function (note) { return note.pitch; }))
      );
      this.selectedIndices.forEach(function (index) {
        var note = this.m.notes[index];
        this.m.notes[index] = Object.assign({}, note, { pitch: note.pitch + pitchDelta });
      }, this);
    } else if (command === "velocity") {
      var velocityDelta = clamp(
        amount,
        1 - Math.min.apply(null, selected.map(function (note) { return note.velocity; })),
        127 - Math.max.apply(null, selected.map(function (note) { return note.velocity; }))
      );
      this.selectedIndices.forEach(function (index) {
        var note = this.m.notes[index];
        this.m.notes[index] = Object.assign({}, note, { velocity: note.velocity + velocityDelta });
      }, this);
    } else {
      throw new Error("Unsupported piano-roll transform: " + command);
    }
    this.commitEdit(before);
  };
  ResultSession.prototype.changeInstrument = function (instrument) {
    this.activeInstrument = instrument;
    this.syncRows();
    if (!this.editing || !this.selectedIndices.size) {
      this.syncEditor();
      return;
    }
    var before = cloneNotes(this.m.notes), template = this.templateForInstrument(instrument);
    this.selectedIndices.forEach(function (index) {
      this.m.notes[index] = Object.assign({}, this.m.notes[index], {
        instrument: instrument,
        program: template.program,
        is_drum: template.is_drum,
        track_index: template.track_index,
        channel: template.channel
      });
    }, this);
    this.commitEdit(before);
  };
  ResultSession.prototype.changeVelocity = function (velocity) {
    if (!this.editing || !this.selectedIndices.size) return;
    if (!Number.isInteger(velocity) || velocity < 1 || velocity > 127) {
      throw new Error("Invalid MIDI velocity: " + velocity);
    }
    var before = cloneNotes(this.m.notes);
    this.selectedIndices.forEach(function (index) {
      this.m.notes[index].velocity = velocity;
    }, this);
    this.commitEdit(before);
  };
  ResultSession.prototype.commitEdit = function (before) {
    if (notesEqual(before, this.m.notes)) return;
    if (this.playing) this.pause();
    this.undoStack.push(before);
    if (this.undoStack.length > 100) this.undoStack.shift();
    this.redoStack = [];
    this.syncEditor();
    this.drawStatic();
    this.scheduleEditedPreview();
  };
  ResultSession.prototype.undo = function () {
    if (!this.undoStack.length) return;
    this.redoStack.push(cloneNotes(this.m.notes));
    this.m.notes = this.undoStack.pop();
    this.selectedIndex = null;
    this.selectedIndices = new Set();
    this.syncEditor();
    this.drawStatic();
    this.scheduleEditedPreview();
  };
  ResultSession.prototype.redo = function () {
    if (!this.redoStack.length) return;
    this.undoStack.push(cloneNotes(this.m.notes));
    this.m.notes = this.redoStack.pop();
    this.selectedIndex = null;
    this.selectedIndices = new Set();
    this.syncEditor();
    this.drawStatic();
    this.scheduleEditedPreview();
  };
  ResultSession.prototype.resetEdits = function () {
    if (notesEqual(this.m.notes, this.originalNotes)) return;
    var before = cloneNotes(this.m.notes);
    this.m.notes = cloneNotes(this.originalNotes);
    this.selectedIndex = null;
    this.selectedIndices = new Set();
    this.commitEdit(before);
  };
  ResultSession.prototype.syncEditor = function () {
    if (!this.editSummary) return;
    var selected = this.selectedIndices.size > 0;
    var dirty = !notesEqual(this.m.notes, this.originalNotes);
    this.addButton.disabled = !this.editing || !this.activeInstrument;
    this.deleteButton.disabled = !this.editing || !selected;
    this.undoButton.disabled = !this.editing || !this.undoStack.length;
    this.redoButton.disabled = !this.editing || !this.redoStack.length;
    this.resetButton.disabled = !this.editing || !dirty;
    this.selectAllButton.disabled = !this.editing || !this.m.notes.length;
    this.cutButton.disabled = !this.editing || !selected;
    this.copyButton.disabled = !this.editing || !selected;
    this.pasteButton.disabled = !this.editing || !this.clipboard.length;
    this.duplicateButton.disabled = !this.editing || !selected;
    var quantizeTargetAvailable = this.quantizeScope === "all_tracks"
      ? this.m.notes.length > 0
      : selected;
    this.quantizeButton.disabled = !this.editing || !quantizeTargetAvailable;
    this.quantizeScopeSelect.disabled = !this.editing;
    this.quantizeGridSelect.disabled = !this.editing;
    this.instrumentSelect.disabled = !this.editing;
    this.velocityInput.disabled = !this.editing || !selected;
    this.editSummary.textContent = this.m.strings.editor_summary
      .replace("{count}", String(this.m.notes.length))
      .replace("{changes}", String(this.undoStack.length));
    this.editNotice.style.display = dirty ? "inline" : "none";
  };
  ResultSession.prototype.downloadTranscriptionAudio = function () {
    var self = this;
    if (!this.audioDownloadReady || this.audioExportInFlight) return;
    if (!this.m.audioExportApi || !this.m.previewToken) {
      this.status.textContent = this.m.strings.audio_export_failed
        .replace("{error}", "server export context is unavailable");
      return;
    }
    var presetId = String(this.audioExportSelect.value || "");
    var preset = this.m.audioExportPresets.find(function (candidate) {
      return candidate.id === presetId;
    });
    if (!preset) {
      this.status.textContent = this.m.strings.audio_export_failed
        .replace("{error}", "selected WAV preset is invalid");
      return;
    }
    var noteSnapshot = cloneNotes(this.m.notes);
    this.audioExportInFlight = true;
    this.setDownloadAudioEnabled(this.audioDownloadReady);
    this.status.textContent = this.m.strings.audio_export_rendering
      .replace("{bit_depth}", String(preset.bitDepth))
      .replace("{sample_rate}", preset.sampleRate.toLocaleString());
    fetch(this.m.audioExportApi, {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data: [JSON.stringify({
          token: this.m.previewToken,
          notes: noteSnapshot,
          preset: preset.id
        })]
      })
    })
      .then(function (response) {
        return response.text().then(function (body) {
          if (!response.ok) throw new Error("HTTP " + response.status + " " + body);
          var envelope = JSON.parse(body);
          if (!envelope.data || envelope.data.length !== 1) {
            throw new Error("MIDI audio export endpoint returned no result");
          }
          return typeof envelope.data[0] === "string"
            ? JSON.parse(envelope.data[0])
            : envelope.data[0];
        });
      })
      .then(function (rendered) {
        if (!rendered
            || String(rendered.presetId) !== preset.id
            || Number(rendered.bitDepth) !== preset.bitDepth
            || Number(rendered.sampleRate) !== preset.sampleRate
            || String(rendered.subtype) !== preset.subtype
            || Number(rendered.channels) !== 2
            || !Number.isFinite(Number(rendered.frames))
            || Number(rendered.frames) <= 0
            || !String(rendered.url || "")
            || !String(rendered.filename || "")) {
          throw new Error("MIDI audio export verification metadata is invalid");
        }
        var anchor = document.createElement("a");
        anchor.href = String(rendered.url);
        anchor.download = String(rendered.filename);
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        self.status.textContent = self.m.strings.audio_export_saved
          .replace("{bit_depth}", String(preset.bitDepth))
          .replace("{sample_rate}", preset.sampleRate.toLocaleString())
          .replace("{path}", String(rendered.filename));
      })
      .catch(function (error) {
        self.status.textContent = self.m.strings.audio_export_failed
          .replace("{error}", String(error));
      })
      .finally(function () {
        self.audioExportInFlight = false;
        self.setDownloadAudioEnabled(self.audioDownloadReady);
      });
  };
  ResultSession.prototype.downloadStemAudio = function () {
    var self = this;
    if (!this.audioDownloadReady || this.audioExportInFlight) return;
    if (!this.m.audioStemExportApi || !this.m.previewToken) {
      this.status.textContent = this.m.strings.stem_audio_export_failed
        .replace("{error}", "server stem export context is unavailable");
      return;
    }
    var presetId = String(this.audioExportSelect.value || "");
    var preset = this.m.audioExportPresets.find(function (candidate) {
      return candidate.id === presetId;
    });
    if (!preset) {
      this.status.textContent = this.m.strings.stem_audio_export_failed
        .replace("{error}", "selected WAV preset is invalid");
      return;
    }
    this.audioExportInFlight = true;
    this.setDownloadAudioEnabled(this.audioDownloadReady);
    this.status.textContent = this.m.strings.stem_audio_export_rendering
      .replace("{bit_depth}", String(preset.bitDepth))
      .replace("{sample_rate}", preset.sampleRate.toLocaleString());
    fetch(this.m.audioStemExportApi, {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data: [JSON.stringify({
          token: this.m.previewToken,
          notes: cloneNotes(this.m.notes),
          preset: preset.id
        })]
      })
    })
      .then(function (response) {
        return response.text().then(function (body) {
          if (!response.ok) throw new Error("HTTP " + response.status + " " + body);
          var envelope = JSON.parse(body);
          if (!envelope.data || envelope.data.length !== 1) {
            throw new Error("MIDI stem export endpoint returned no result");
          }
          return typeof envelope.data[0] === "string"
            ? JSON.parse(envelope.data[0])
            : envelope.data[0];
        });
      })
      .then(function (rendered) {
        var frames = Number(rendered && rendered.frames);
        var members = rendered && rendered.members;
        if (!rendered
            || String(rendered.presetId) !== preset.id
            || Number(rendered.bitDepth) !== preset.bitDepth
            || Number(rendered.sampleRate) !== preset.sampleRate
            || String(rendered.subtype) !== preset.subtype
            || !Number.isFinite(frames)
            || frames <= 0
            || !Array.isArray(members)
            || members.length < 1
            || Number(rendered.memberCount) !== members.length
            || members.some(function (member) {
              return Number(member.frames) !== frames
                || Number(member.channels) !== 2
                || !String(member.instrument || "")
                || !String(member.filename || "");
            })
            || !String(rendered.url || "")
            || !String(rendered.filename || "")) {
          throw new Error("MIDI stem export verification metadata is invalid");
        }
        var anchor = document.createElement("a");
        anchor.href = String(rendered.url);
        anchor.download = String(rendered.filename);
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        self.status.textContent = self.m.strings.stem_audio_export_saved
          .replace("{count}", String(members.length))
          .replace("{bit_depth}", String(preset.bitDepth))
          .replace("{sample_rate}", preset.sampleRate.toLocaleString())
          .replace("{path}", String(rendered.filename));
      })
      .catch(function (error) {
        self.status.textContent = self.m.strings.stem_audio_export_failed
          .replace("{error}", String(error));
      })
      .finally(function () {
        self.audioExportInFlight = false;
        self.setDownloadAudioEnabled(self.audioDownloadReady);
      });
  };
  ResultSession.prototype.fetchEditedMidiBytes = function () {
    var self = this;
    return fetch(this.m.downloads.midi, { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status + " " + self.m.downloads.midi);
        return response.arrayBuffer();
      })
      .then(function (arrayBuffer) {
        return buildEditedSmf(
          arrayBuffer,
          self.m.notes,
          self.targetBpm,
          Number(self.m.referenceBpm),
          Boolean(self.m.repeatTempoPerNoteTrack)
        );
      });
  };
  function bytesToBase64(bytes) {
    var binary = "";
    var chunkSize = 0x8000;
    for (var offset = 0; offset < bytes.length; offset += chunkSize) {
      binary += String.fromCharCode.apply(null, bytes.subarray(offset, offset + chunkSize));
    }
    return btoa(binary);
  }
  ResultSession.prototype.downloadEditedMidi = function () {
    var self = this;
    try {
      this.commitBpm();
    } catch (error) {
      this.status.textContent = this.m.strings.editor_export_failed.replace("{error}", String(error));
      return;
    }
    this.status.textContent = this.m.strings.ready;
    this.fetchEditedMidiBytes()
      .then(function (encoded) {
        var blobUrl = URL.createObjectURL(new Blob([encoded], { type: "audio/midi" }));
        var anchor = document.createElement("a");
        anchor.href = blobUrl;
        anchor.download = "music-to-midi-edited-" + self.targetBpm.toFixed(1) + "BPM.mid";
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        setTimeout(function () { URL.revokeObjectURL(blobUrl); }, 1000);
      })
      .catch(function (error) {
        self.status.textContent = self.m.strings.editor_export_failed.replace("{error}", String(error));
      });
  };
  ResultSession.prototype.downloadSheetMusic = function () {
    var self = this;
    if (!this.m.sheetApi || !this.m.sheetToken) {
      this.status.textContent = this.m.strings.sheet_music_failed
        .replace("{error}", "server sheet-music context is unavailable");
      return;
    }
    try {
      this.commitBpm();
    } catch (error) {
      this.status.textContent = this.m.strings.sheet_music_failed
        .replace("{error}", String(error));
      return;
    }
    this.sheetMusicButton.disabled = true;
    this.status.textContent = this.m.strings.sheet_music_rendering
      .replace("{grid}", this.quantizeGrid);
    this.fetchEditedMidiBytes()
      .then(function (editedMidi) {
        return fetch(self.m.sheetApi, {
          method: "POST",
          cache: "no-store",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            data: [JSON.stringify({
              token: self.m.sheetToken,
              midi_base64: bytesToBase64(editedMidi),
              quantize_grid: self.quantizeGrid
            })]
          })
        });
      })
      .then(function (response) {
        return response.text().then(function (body) {
          if (!response.ok) throw new Error("HTTP " + response.status + " " + body);
          var envelope = JSON.parse(body);
          if (!envelope.data || envelope.data.length !== 1) {
            throw new Error("Sheet-music endpoint returned no result");
          }
          return typeof envelope.data[0] === "string"
            ? JSON.parse(envelope.data[0])
            : envelope.data[0];
        });
      })
      .then(function (result) {
        if (!result.url || !result.filename) {
          throw new Error("Sheet-music endpoint returned an invalid download");
        }
        var anchor = document.createElement("a");
        anchor.href = String(result.url);
        anchor.download = String(result.filename);
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        self.status.textContent = self.m.strings.sheet_music_ready
          .replace("{count}", String(result.memberCount))
          .replace("{grid}", String(result.quantizeGrid))
          .replace("{version}", String(result.musescoreVersion));
      })
      .catch(function (error) {
        self.status.textContent = self.m.strings.sheet_music_failed
          .replace("{error}", String(error));
      })
      .finally(function () {
        if (!self.disposed) self.sheetMusicButton.disabled = false;
      });
  };
  ResultSession.prototype.dispose = function () {
    if (this.disposed) return;
    this.disposed = true;
    clearTimeout(this.previewTimer);
    this.previewRevision += 1;
    this.pause();
    window.removeEventListener("music-to-midi-playback-start", this.onExternalPlayback);
    if (this.resizeObserver) this.resizeObserver.disconnect();
    cancelAnimationFrame(this.drawRaf);
  };
  window.musicToMidiMidiEditorRuntime = Object.freeze({
    buildEditedSmf: buildEditedSmf,
    projectPlaybackRate: projectPlaybackRate
  });
  function scan() {
    for (var index = sessions.length - 1; index >= 0; index--) {
      if (!sessions[index].root.isConnected) {
        sessions[index].dispose();
        sessions.splice(index, 1);
      }
    }
    document.querySelectorAll(".msr-root:not([data-msr-init])").forEach(function (root) {
      root.setAttribute("data-msr-init", "1");
      var session = new ResultSession(root);
      sessions.push(session);
      session.init();
    });
  }
  var timer = 0;
  function schedule() {
    if (timer) return;
    timer = setTimeout(function () { timer = 0; scan(); }, 40);
  }
  new MutationObserver(function (changes) {
    for (var index = 0; index < changes.length; index++) {
      var added = changes[index].addedNodes, removed = changes[index].removedNodes;
      for (var addIndex = 0; addIndex < added.length; addIndex++) {
        var node = added[addIndex];
        if (node.nodeType === 1 && (node.matches(".msr-root") || node.querySelector(".msr-root"))) {
          schedule();
          return;
        }
      }
      for (var removeIndex = 0; removeIndex < removed.length; removeIndex++) {
        var removedNode = removed[removeIndex];
        if (
          removedNode.nodeType === 1 &&
          (removedNode.matches(".msr-root") || removedNode.querySelector(".msr-root"))
        ) {
          schedule();
          return;
        }
      }
    }
  }).observe(document.documentElement, { childList: true, subtree: true });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", schedule);
  else schedule();
})();
"""


def muscriptor_result_head() -> str:
    return f"<style>{MUSCRIPTOR_RESULT_CSS}</style><script>{MUSCRIPTOR_RESULT_JS}</script>"
