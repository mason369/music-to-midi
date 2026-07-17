"""Shared browser multi-track mixer runtime for the Gradio platforms.

The desktop Qt mixer (:mod:`src.gui.widgets.audio_track_mixer`) is the
reference contract. This module mirrors its controls and semantics for the
Hugging Face Space and the Colab notebook so every platform exposes the same
multi-track timeline after a separation:

- shared transport (play all / pause all / replay from start) and playhead
- per-track mute / solo
- per-track volume slider (-60 .. 0 dB, linear gain = 10^(dB/20))
- per-track time offset slider (-10 .. +10 s; local = global - offset)
- timeline zoom (1 .. 16x), fit-to-window, and track alignment (reset offsets)
- click / drag seeking on the ruler and waveforms

The JavaScript runtime is injected once through ``mixer_head()`` into the
Gradio ``Blocks`` head; ``build_track_mixer_html()`` renders one mixer root
whose hidden manifest the runtime picks up through a MutationObserver. The
mouse wheel intentionally has no handler: scrolling only scrolls the page,
exactly like the desktop widget.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.parse import quote

# Mixer semantics mirrored from src/gui/widgets/audio_track_mixer.py.
MIN_VOLUME_DB = -60.0
MAX_VOLUME_DB = 0.0
MIN_OFFSET_SECONDS = -10.0
MAX_OFFSET_SECONDS = 10.0
MIN_ZOOM = 1
MAX_ZOOM = 16
PLAYHEAD_COLOR = "#ff5d73"
ROW_HEIGHT_PX = 72
RULER_HEIGHT_PX = 28

TRACK_COLORS = (
    "#5eb1ff",
    "#ff8d66",
    "#7bd88f",
    "#c89bff",
    "#ff70a6",
    "#ffd166",
    "#62d2c3",
    "#9aa8ff",
)

KNOWN_TRACK_NAMES = frozenset(
    (
        "bass",
        "drums",
        "guitar",
        "piano",
        "vocals",
        "accompaniment",
        "other",
        "source",
    )
)

_MIXER_STRING_KEYS = (
    "play",
    "pause",
    "replay",
    "align",
    "fit",
    "zoom",
    "zoom_out",
    "zoom_in",
    "mute",
    "solo",
    "volume",
    "offset",
    "loading",
    "waveform_failed",
    "ready",
    "failed",
    "empty",
    "timeline",
)


def display_track_name(track_name: str, translate: Callable[[str], str]) -> str:
    """Resolve one raw track name through the shared i18n catalog."""
    if track_name in KNOWN_TRACK_NAMES:
        return translate(f"dialogs.complete.audio_tracks.track_names.{track_name}")
    return track_name


def track_mixer_strings(translate: Callable[[str], str]) -> dict[str, str]:
    """Collect every label the browser runtime needs from the i18n catalog."""
    return {
        key: translate(f"dialogs.complete.audio_tracks.{key}")
        for key in _MIXER_STRING_KEYS
    }


def track_file_url(path: str | Path) -> str:
    """Return the Gradio ``/file=`` URL for one absolute audio path.

    The matching output root must be listed in ``demo.launch(allowed_paths=...)``
    or Gradio refuses to serve the file. Forward slashes are enforced so the
    URL form is identical on every build platform.
    """
    posix_path = str(path).replace("\\", "/")
    return "/file=" + quote(posix_path, safe="/")


def build_track_mixer_manifest(
    tracks: Iterable[Mapping[str, object]],
    translate: Callable[[str], str],
) -> dict:
    """Build the JSON manifest consumed by the browser runtime."""
    entries = []
    for track in tracks:
        raw_name = str(track.get("name", ""))
        audio_path = Path(str(track.get("audio_path", "")))
        entries.append(
            {
                "id": str(track.get("id", raw_name)),
                "name": display_track_name(raw_name, translate),
                "color": str(track.get("color") or TRACK_COLORS[0]),
                "url": track_file_url(audio_path),
                "fileName": audio_path.name,
            }
        )
    return {"tracks": entries, "strings": track_mixer_strings(translate)}


def build_track_mixer_html(
    tracks: Iterable[Mapping[str, object]],
    translate: Callable[[str], str],
) -> str:
    """Return the mixer root markup with an embedded JSON manifest."""
    manifest = json.dumps(
        build_track_mixer_manifest(tracks, translate),
        ensure_ascii=False,
    )
    return (
        '<div class="mtm-mixer-root">'
        f'<pre class="mtm-manifest" hidden>{html.escape(manifest, quote=False)}</pre>'
        '<div class="mtm-mixer"></div>'
        "</div>"
    )


TRACK_MIXER_CSS = """
.mtm-mixer-root { margin: 8px 0 4px; }
.mtm-mixer { font-size: 13px; color: #dbe4f3; }
.mtm-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 6px; }
.mtm-btn {
  background: #1c2a47; border: 1px solid #33415e; color: #dbe4f3;
  border-radius: 6px; padding: 4px 12px; font-size: 12px; cursor: pointer;
}
.mtm-btn:hover:not(:disabled) { border-color: #5eb1ff; }
.mtm-btn:disabled { opacity: 0.45; cursor: default; }
.mtm-btn.mtm-active { background: #274a7d; border-color: #5eb1ff; color: #ffffff; }
.mtm-zoom-slider { width: 110px; }
.mtm-zoom-label, .mtm-position, .mtm-status { color: #9eacc1; font-size: 12px; white-space: nowrap; }
.mtm-scroll {
  overflow-x: auto; border: 1px solid #263653; border-radius: 8px;
  background: #0f1830;
}
.mtm-content { position: relative; min-width: 100%; }
.mtm-ruler { display: block; }
.mtm-track { border-top: 1px solid #1d2a44; }
.mtm-wave { display: block; cursor: crosshair; }
.mtm-controls {
  position: sticky; left: 0; display: flex; flex-wrap: wrap; align-items: center;
  gap: 12px; padding: 6px 10px; background: #141f38; border-top: 1px solid #1d2a44;
  box-sizing: border-box;
}
.mtm-name { font-weight: 600; }
.mtm-file-name { color: #7f8ea8; font-size: 11px; }
.mtm-ctl { display: inline-flex; align-items: center; gap: 6px; color: #9eacc1; font-size: 12px; }
.mtm-ctl input[type="range"] { width: 90px; }
.mtm-ctl-value { min-width: 52px; text-align: right; color: #dbe4f3; }
.mtm-playhead {
  position: absolute; top: 0; bottom: 0; width: 2px;
  background: #ff5d73; pointer-events: none; z-index: 3;
}
.mtm-seek-row { padding: 6px 2px 0; }
.mtm-seek { width: 100%; }
.mtm-empty { padding: 24px; text-align: center; color: #9eacc1; }
.mtm-track-failed .mtm-name::after { content: " ⚠"; }
"""


TRACK_MIXER_JS = r"""
(function () {
  "use strict";

  var MIN_VOLUME_DB = -60.0;
  var MAX_VOLUME_DB = 0.0;
  var MIN_OFFSET_SECONDS = -10.0;
  var MAX_OFFSET_SECONDS = 10.0;
  var MIN_ZOOM = 1;
  var MAX_ZOOM = 16;
  var PLAYHEAD_COLOR = "#ff5d73";
  var ROW_HEIGHT = 72;
  var RULER_HEIGHT = 28;
  var MAX_CANVAS_PIXELS = 16000;

  var audioContext = null;
  var bufferCache = {};
  var bufferPromises = {};
  var sessions = [];
  var lastSnapshot = null;

  function ensureAudioContext() {
    if (!audioContext) {
      var Ctor = window.AudioContext || window.webkitAudioContext;
      audioContext = new Ctor();
    }
    return audioContext;
  }

  function loadBuffer(url) {
    if (bufferCache[url]) {
      return Promise.resolve(bufferCache[url]);
    }
    if (bufferPromises[url]) {
      return bufferPromises[url];
    }
    var ctx = ensureAudioContext();
    bufferPromises[url] = fetch(url)
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.arrayBuffer();
      })
      .then(function (data) {
        return ctx.decodeAudioData(data);
      })
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

  function dbToLinear(db) {
    if (db <= MIN_VOLUME_DB) {
      return 0.0;
    }
    return Math.min(1.0, Math.pow(10.0, db / 20.0));
  }

  function clamp(value, lo, hi) {
    return Math.min(hi, Math.max(lo, value));
  }

  function pad2(value) {
    return value < 10 ? "0" + value : String(value);
  }

  function formatClock(seconds) {
    var total = Math.max(0, Math.floor(seconds));
    var hours = Math.floor(total / 3600);
    var minutes = Math.floor((total % 3600) / 60);
    var secs = total % 60;
    if (hours > 0) {
      return hours + ":" + pad2(minutes) + ":" + pad2(secs);
    }
    return minutes + ":" + pad2(secs);
  }

  function formatOffset(seconds) {
    return (seconds >= 0 ? "+" : "") + seconds.toFixed(2) + "s";
  }

  function formatDb(db) {
    return db.toFixed(1) + " dB";
  }

  function el(tag, className) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    return node;
  }

  function makeButton(label, className, ariaLabel) {
    var node = el("button", className);
    node.type = "button";
    node.textContent = label;
    if (ariaLabel) {
      node.setAttribute("aria-label", ariaLabel);
    }
    return node;
  }

  function makeRange(min, max, value, step, className, ariaLabel) {
    var node = el("input", className);
    node.type = "range";
    node.min = String(min);
    node.max = String(max);
    node.step = String(step);
    node.value = String(value);
    if (ariaLabel) {
      node.setAttribute("aria-label", ariaLabel);
    }
    return node;
  }

  function MixerSession(root) {
    this.root = root;
    this.host = null;
    this.strings = {};
    this.tracks = [];
    this.ready = false;
    this.playing = false;
    this.position = 0;
    this.duration = 0;
    this.zoom = MIN_ZOOM;
    this.pxPerSec = 0;
    this.dpr = 1;
    this.playCtxTime = 0;
    this.playStartPosition = 0;
    this.rafId = null;
    this.disposed = false;
    this.pointerSeeking = false;
    this.seekDragging = false;
    this.resizeObserver = null;
  }

  MixerSession.prototype.init = function () {
    var manifest = { tracks: [], strings: {} };
    var manifestNode = this.root.querySelector("pre.mtm-manifest");
    if (manifestNode) {
      try {
        manifest = JSON.parse(manifestNode.textContent);
      } catch (error) {
        manifest = { tracks: [], strings: {} };
      }
    }
    this.strings = manifest.strings || {};
    this.host = this.root.querySelector(".mtm-mixer");
    if (!this.host) {
      return;
    }
    this.host.innerHTML = "";
    var specs = manifest.tracks || [];
    if (!specs.length) {
      this.showEmpty();
      return;
    }
    this.buildShell();
    var self = this;
    this.tracks = specs.map(function (spec) {
      return {
        id: String(spec.id),
        name: String(spec.name),
        color: spec.color || "#5eb1ff",
        url: String(spec.url),
        fileName: spec.fileName || "",
        buffer: bufferCache[spec.url] || null,
        failed: false,
        gain: null,
        source: null,
        muted: false,
        solo: false,
        volumeDb: 0.0,
        offsetS: 0.0,
        els: {}
      };
    });
    this.buildRows();
    if (
      lastSnapshot &&
      lastSnapshot.urls.length === this.tracks.length &&
      lastSnapshot.urls.every(function (url, index) {
        return url === self.tracks[index].url;
      })
    ) {
      this.position = lastSnapshot.position;
    }
    this.setStatus("loading");
    this.updateToolbar();
    var pending = 0;
    var hadFailure = false;
    this.tracks.forEach(function (track) {
      if (track.buffer) {
        return;
      }
      pending += 1;
      loadBuffer(track.url)
        .then(function (buffer) {
          if (!self.disposed) {
            track.buffer = buffer;
          }
        })
        .catch(function () {
          if (!self.disposed) {
            track.failed = true;
            hadFailure = true;
          }
        })
        .then(function () {
          if (self.disposed) {
            return;
          }
          pending -= 1;
          if (pending === 0) {
            self.onAllLoaded(hadFailure);
          } else {
            self.layout();
          }
        });
    });
    if (pending === 0) {
      this.onAllLoaded(false);
    }
  };

  MixerSession.prototype.showEmpty = function () {
    var empty = el("div", "mtm-empty");
    empty.textContent = this.strings.empty || "";
    this.host.appendChild(empty);
  };

  MixerSession.prototype.buildShell = function () {
    var strings = this.strings;
    var toolbar = el("div", "mtm-toolbar");
    this.playButton = makeButton(strings.play || "", "mtm-btn mtm-play", strings.play);
    this.playButton.addEventListener("click", this.togglePlay.bind(this));
    this.replayButton = makeButton(strings.replay || "", "mtm-btn mtm-replay", strings.replay);
    this.replayButton.addEventListener("click", this.replay.bind(this));
    this.alignButton = makeButton(strings.align || "", "mtm-btn mtm-align", strings.align);
    this.alignButton.disabled = true;
    this.alignButton.addEventListener("click", this.alignTracks.bind(this));
    this.zoomOutButton = makeButton("−", "mtm-btn mtm-zoom-out", strings.zoom_out);
    this.zoomOutButton.addEventListener(
      "click",
      function () {
        this.setZoom(this.zoom - 1);
      }.bind(this)
    );
    this.zoomSlider = makeRange(MIN_ZOOM, MAX_ZOOM, this.zoom, 1, "mtm-zoom-slider", strings.zoom);
    this.zoomSlider.addEventListener(
      "input",
      function () {
        this.setZoom(parseInt(this.zoomSlider.value, 10));
      }.bind(this)
    );
    this.zoomInButton = makeButton("+", "mtm-btn mtm-zoom-in", strings.zoom_in);
    this.zoomInButton.addEventListener(
      "click",
      function () {
        this.setZoom(this.zoom + 1);
      }.bind(this)
    );
    this.fitButton = makeButton(strings.fit || "", "mtm-btn mtm-fit", strings.fit);
    this.fitButton.addEventListener("click", this.fitTimeline.bind(this));
    this.zoomLabel = el("span", "mtm-zoom-label");
    this.zoomLabel.textContent = this.zoom + "×";
    this.positionLabel = el("span", "mtm-position");
    this.statusLabel = el("span", "mtm-status");
    toolbar.appendChild(this.playButton);
    toolbar.appendChild(this.replayButton);
    toolbar.appendChild(this.alignButton);
    toolbar.appendChild(this.zoomOutButton);
    toolbar.appendChild(this.zoomSlider);
    toolbar.appendChild(this.zoomInButton);
    toolbar.appendChild(this.fitButton);
    toolbar.appendChild(this.zoomLabel);
    toolbar.appendChild(this.positionLabel);
    toolbar.appendChild(this.statusLabel);
    this.host.appendChild(toolbar);

    this.scroll = el("div", "mtm-scroll");
    this.scroll.setAttribute("aria-label", strings.timeline || "");
    this.content = el("div", "mtm-content");
    this.ruler = el("canvas", "mtm-ruler");
    this.content.appendChild(this.ruler);
    this.rowsHost = el("div", "mtm-rows");
    this.content.appendChild(this.rowsHost);
    this.playhead = el("div", "mtm-playhead");
    this.content.appendChild(this.playhead);
    this.scroll.appendChild(this.content);
    this.host.appendChild(this.scroll);
    this.bindSeekSurface(this.ruler);

    this.seekSlider = makeRange(0, 0, 0, 1, "mtm-seek", strings.timeline);
    this.seekSlider.addEventListener(
      "input",
      function () {
        this.seekTo(parseInt(this.seekSlider.value, 10) / 1000);
      }.bind(this)
    );
    this.seekSlider.addEventListener(
      "pointerdown",
      function () {
        this.seekDragging = true;
      }.bind(this)
    );
    this.seekSlider.addEventListener(
      "pointerup",
      function () {
        this.seekDragging = false;
      }.bind(this)
    );
    var seekRow = el("div", "mtm-seek-row");
    seekRow.appendChild(this.seekSlider);
    this.host.appendChild(seekRow);

    if (window.ResizeObserver) {
      var self = this;
      this.resizeObserver = new ResizeObserver(function () {
        if (!self.disposed) {
          self.layout();
        }
      });
      this.resizeObserver.observe(this.scroll);
    }
  };

  MixerSession.prototype.buildRows = function () {
    var self = this;
    this.tracks.forEach(function (track) {
      var row = el("div", "mtm-track");
      row.setAttribute("data-track-id", track.id);
      var wave = el("canvas", "mtm-wave");
      row.appendChild(wave);
      track.els.wave = wave;
      self.bindSeekSurface(wave);

      var controls = el("div", "mtm-controls");
      var name = el("span", "mtm-name");
      name.textContent = track.name;
      name.style.color = track.color;
      controls.appendChild(name);
      if (track.fileName) {
        var fileName = el("span", "mtm-file-name");
        fileName.textContent = track.fileName;
        controls.appendChild(fileName);
      }
      var muteButton = makeButton(self.strings.mute || "", "mtm-btn mtm-mute", self.strings.mute);
      muteButton.addEventListener("click", function () {
        track.muted = !track.muted;
        muteButton.classList.toggle("mtm-active", track.muted);
        self.applyMix();
      });
      controls.appendChild(muteButton);
      var soloButton = makeButton(self.strings.solo || "", "mtm-btn mtm-solo", self.strings.solo);
      soloButton.addEventListener("click", function () {
        track.solo = !track.solo;
        soloButton.classList.toggle("mtm-active", track.solo);
        self.applyMix();
      });
      controls.appendChild(soloButton);

      var volumeLabel = el("label", "mtm-ctl mtm-volume");
      volumeLabel.appendChild(document.createTextNode(self.strings.volume || ""));
      var volumeSlider = makeRange(
        MIN_VOLUME_DB,
        MAX_VOLUME_DB,
        track.volumeDb,
        1,
        "",
        self.strings.volume
      );
      var volumeValue = el("span", "mtm-ctl-value");
      volumeValue.textContent = formatDb(track.volumeDb);
      volumeSlider.addEventListener("input", function () {
        track.volumeDb = clamp(
          parseFloat(volumeSlider.value),
          MIN_VOLUME_DB,
          MAX_VOLUME_DB
        );
        volumeValue.textContent = formatDb(track.volumeDb);
        self.applyMix();
      });
      volumeLabel.appendChild(volumeSlider);
      volumeLabel.appendChild(volumeValue);
      controls.appendChild(volumeLabel);

      var offsetLabel = el("label", "mtm-ctl mtm-offset");
      offsetLabel.appendChild(document.createTextNode(self.strings.offset || ""));
      var offsetSlider = makeRange(
        MIN_OFFSET_SECONDS,
        MAX_OFFSET_SECONDS,
        track.offsetS,
        0.1,
        "",
        self.strings.offset
      );
      var offsetValue = el("span", "mtm-ctl-value");
      offsetValue.textContent = formatOffset(track.offsetS);
      offsetSlider.addEventListener("input", function () {
        track.offsetS = clamp(
          parseFloat(offsetSlider.value),
          MIN_OFFSET_SECONDS,
          MAX_OFFSET_SECONDS
        );
        offsetValue.textContent = formatOffset(track.offsetS);
        self.onOffsetChanged();
      });
      offsetLabel.appendChild(offsetSlider);
      offsetLabel.appendChild(offsetValue);
      controls.appendChild(offsetLabel);

      track.els.row = row;
      track.els.controls = controls;
      row.appendChild(controls);
      self.rowsHost.appendChild(row);
    });
  };

  MixerSession.prototype.bindSeekSurface = function (canvas) {
    var self = this;
    canvas.addEventListener("pointerdown", function (event) {
      if (!self.ready || !self.duration) {
        return;
      }
      self.pointerSeeking = true;
      if (canvas.setPointerCapture) {
        try {
          canvas.setPointerCapture(event.pointerId);
        } catch (error) {}
      }
      self.seekTo(self.eventSeconds(canvas, event));
      event.preventDefault();
    });
    canvas.addEventListener("pointermove", function (event) {
      if (self.pointerSeeking) {
        self.seekTo(self.eventSeconds(canvas, event));
      }
    });
    canvas.addEventListener("pointerup", function () {
      self.pointerSeeking = false;
    });
    canvas.addEventListener("pointercancel", function () {
      self.pointerSeeking = false;
    });
  };

  MixerSession.prototype.eventSeconds = function (canvas, event) {
    var rect = canvas.getBoundingClientRect();
    return clamp((event.clientX - rect.left) / this.pxPerSec, 0, this.duration);
  };

  MixerSession.prototype.onAllLoaded = function (hadFailure) {
    this.ready = this.tracks.some(function (track) {
      return track.buffer && !track.failed;
    });
    this.tracks.forEach(function (track) {
      if (track.failed && track.els.row) {
        track.els.row.classList.add("mtm-track-failed");
      }
    });
    this.recomputeDuration();
    this.position = clamp(this.position, 0, this.duration);
    this.layout();
    this.setStatus(hadFailure ? "failed" : "ready");
    this.updateToolbar();
  };

  MixerSession.prototype.setStatus = function (kind) {
    if (!this.statusLabel) {
      return;
    }
    this.statusLabel.textContent = this.strings[kind] || "";
  };

  MixerSession.prototype.updateToolbar = function () {
    if (!this.playButton) {
      return;
    }
    var usable = this.ready && this.duration > 0;
    this.playButton.disabled = !usable;
    this.replayButton.disabled = !usable;
    this.playButton.textContent = this.playing
      ? this.strings.pause || ""
      : this.strings.play || "";
  };

  MixerSession.prototype.trackGain = function (track) {
    var anySolo = this.tracks.some(function (item) {
      return item.solo && item.buffer && !item.failed;
    });
    var audible = anySolo ? track.solo : !track.muted;
    return audible ? dbToLinear(track.volumeDb) : 0.0;
  };

  MixerSession.prototype.applyMix = function () {
    var self = this;
    this.tracks.forEach(function (track) {
      if (track.gain) {
        track.gain.gain.value = self.trackGain(track);
      }
    });
  };

  MixerSession.prototype.stopTrackSource = function (track) {
    if (track.source) {
      try {
        track.source.stop();
      } catch (error) {}
      try {
        track.source.disconnect();
      } catch (error) {}
      track.source = null;
    }
    if (track.gain) {
      try {
        track.gain.disconnect();
      } catch (error) {}
      track.gain = null;
    }
  };

  MixerSession.prototype.stopAllSources = function () {
    var self = this;
    this.tracks.forEach(function (track) {
      self.stopTrackSource(track);
    });
  };

  MixerSession.prototype.startSources = function () {
    var ctx = ensureAudioContext();
    var self = this;
    this.tracks.forEach(function (track) {
      self.stopTrackSource(track);
      if (!track.buffer || track.failed) {
        return;
      }
      var local = self.playStartPosition - track.offsetS;
      var when = ctx.currentTime;
      var offset = local;
      if (local < 0) {
        when = ctx.currentTime + -local;
        offset = 0;
      }
      if (offset >= track.buffer.duration) {
        return;
      }
      var source = ctx.createBufferSource();
      source.buffer = track.buffer;
      var gain = ctx.createGain();
      gain.gain.value = self.trackGain(track);
      source.connect(gain);
      gain.connect(ctx.destination);
      source.start(when, offset);
      track.source = source;
      track.gain = gain;
    });
  };

  MixerSession.prototype.currentPosition = function () {
    if (!this.playing) {
      return this.position;
    }
    return clamp(
      this.playStartPosition + (ensureAudioContext().currentTime - this.playCtxTime),
      0,
      this.duration
    );
  };

  MixerSession.prototype.togglePlay = function () {
    if (this.playing) {
      this.pause();
    } else {
      this.play();
    }
  };

  MixerSession.prototype.play = function () {
    if (!this.ready || !this.duration || this.playing) {
      return;
    }
    var ctx = ensureAudioContext();
    if (ctx.state === "suspended") {
      ctx.resume();
    }
    if (this.position >= this.duration) {
      this.position = 0;
    }
    this.playing = true;
    this.playStartPosition = this.position;
    this.playCtxTime = ctx.currentTime;
    this.startSources();
    this.updateToolbar();
    this.startClock();
  };

  MixerSession.prototype.pause = function () {
    if (!this.playing) {
      return;
    }
    this.position = this.currentPosition();
    this.playing = false;
    this.stopAllSources();
    this.stopClock();
    this.updateToolbar();
    this.updatePositionUI();
    this.layoutPlayhead();
  };

  MixerSession.prototype.replay = function () {
    if (!this.ready) {
      return;
    }
    this.stopAllSources();
    this.position = 0;
    if (this.playing) {
      this.playStartPosition = 0;
      this.playCtxTime = ensureAudioContext().currentTime;
      this.startSources();
    }
    this.updatePositionUI();
    this.layoutPlayhead();
  };

  MixerSession.prototype.seekTo = function (seconds) {
    if (!this.ready || !this.duration) {
      return;
    }
    this.position = clamp(seconds, 0, this.duration);
    if (this.playing) {
      this.playStartPosition = this.position;
      this.playCtxTime = ensureAudioContext().currentTime;
      this.startSources();
    }
    this.updatePositionUI();
    this.layoutPlayhead();
  };

  MixerSession.prototype.alignTracks = function () {
    var anyOffset = this.tracks.some(function (track) {
      return track.offsetS !== 0;
    });
    if (!anyOffset) {
      return;
    }
    this.tracks.forEach(function (track) {
      track.offsetS = 0;
      if (track.els.controls) {
        var slider = track.els.controls.querySelector(".mtm-offset input");
        var value = track.els.controls.querySelector(".mtm-offset .mtm-ctl-value");
        if (slider) {
          slider.value = "0";
        }
        if (value) {
          value.textContent = formatOffset(0);
        }
      }
    });
    this.alignButton.disabled = true;
    this.onTimelineShapeChanged();
  };

  MixerSession.prototype.onOffsetChanged = function () {
    var anyOffset = this.tracks.some(function (track) {
      return track.offsetS !== 0;
    });
    this.alignButton.disabled = !anyOffset;
    this.onTimelineShapeChanged();
  };

  MixerSession.prototype.onTimelineShapeChanged = function () {
    if (this.playing) {
      this.playStartPosition = this.currentPosition();
      this.playCtxTime = ensureAudioContext().currentTime;
      this.startSources();
    }
    this.recomputeDuration();
    this.layout();
  };

  MixerSession.prototype.setZoom = function (zoom) {
    var next = clamp(Math.round(zoom), MIN_ZOOM, MAX_ZOOM);
    var anchorSeconds = this.position;
    var anchorX = anchorSeconds * this.pxPerSec - this.scroll.scrollLeft;
    this.zoom = next;
    this.layout();
    this.scroll.scrollLeft = Math.max(0, anchorSeconds * this.pxPerSec - anchorX);
    this.zoomSlider.value = String(next);
    this.zoomLabel.textContent = next + "×";
  };

  MixerSession.prototype.fitTimeline = function () {
    this.setZoom(MIN_ZOOM);
  };

  MixerSession.prototype.startClock = function () {
    var self = this;
    var tick = function () {
      if (self.disposed || !self.playing) {
        return;
      }
      self.position = self.currentPosition();
      if (self.position >= self.duration) {
        self.position = self.duration;
        self.pause();
        return;
      }
      self.updatePositionUI();
      self.layoutPlayhead();
      self.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  };

  MixerSession.prototype.stopClock = function () {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  };

  MixerSession.prototype.recomputeDuration = function () {
    var duration = 0;
    this.tracks.forEach(function (track) {
      if (track.buffer && !track.failed) {
        duration = Math.max(duration, track.offsetS + track.buffer.duration);
      }
    });
    this.duration = duration;
    if (this.position > duration) {
      this.position = duration;
    }
    if (this.seekSlider) {
      this.seekSlider.max = String(Math.max(0, Math.round(duration * 1000)));
    }
  };

  MixerSession.prototype.layout = function () {
    if (!this.scroll || !this.duration) {
      if (this.scroll) {
        this.drawEmptyLayout();
      }
      return;
    }
    var viewport = this.scroll.clientWidth || 800;
    this.pxPerSec = (viewport / this.duration) * this.zoom;
    var timelineWidth = Math.max(viewport, Math.round(this.duration * this.pxPerSec));
    this.content.style.width = timelineWidth + "px";
    this.dpr = Math.min(
      window.devicePixelRatio || 1,
      MAX_CANVAS_PIXELS / Math.max(1, timelineWidth)
    );
    this.sizeCanvas(this.ruler, timelineWidth, RULER_HEIGHT);
    this.drawRuler();
    var self = this;
    this.tracks.forEach(function (track) {
      self.sizeCanvas(track.els.wave, timelineWidth, ROW_HEIGHT);
      self.drawWave(track);
      track.els.controls.style.width = viewport + "px";
    });
    this.layoutPlayhead();
    this.updatePositionUI();
  };

  MixerSession.prototype.drawEmptyLayout = function () {
    var viewport = this.scroll.clientWidth || 800;
    this.content.style.width = viewport + "px";
    this.sizeCanvas(this.ruler, viewport, RULER_HEIGHT);
    var self = this;
    this.tracks.forEach(function (track) {
      self.sizeCanvas(track.els.wave, viewport, ROW_HEIGHT);
      track.els.controls.style.width = viewport + "px";
    });
  };

  MixerSession.prototype.sizeCanvas = function (canvas, cssWidth, cssHeight) {
    canvas.style.width = cssWidth + "px";
    canvas.style.height = cssHeight + "px";
    canvas.width = Math.max(1, Math.round(cssWidth * this.dpr));
    canvas.height = Math.max(1, Math.round(cssHeight * this.dpr));
  };

  MixerSession.prototype.chooseRulerStep = function () {
    var pxPerSec = this.pxPerSec * this.dpr;
    var steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800];
    for (var index = 0; index < steps.length; index += 1) {
      if (steps[index] * pxPerSec >= 80 * this.dpr) {
        return steps[index];
      }
    }
    return 3600;
  };

  MixerSession.prototype.drawRuler = function () {
    var ctx = this.ruler.getContext("2d");
    var width = this.ruler.width;
    var height = this.ruler.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#9eacc1";
    ctx.strokeStyle = "#3a4a63";
    ctx.font = 10 * this.dpr + "px sans-serif";
    ctx.textBaseline = "top";
    var pxPerSec = this.pxPerSec * this.dpr;
    var step = this.chooseRulerStep();
    ctx.lineWidth = 1;
    for (var seconds = 0; seconds <= this.duration + 1e-6; seconds += step) {
      var x = Math.round(seconds * pxPerSec) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, height * 0.45);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.fillText(formatClock(seconds), x + 3 * this.dpr, 2 * this.dpr);
    }
  };

  MixerSession.prototype.drawWave = function (track) {
    var canvas = track.els.wave;
    var ctx = canvas.getContext("2d");
    var width = canvas.width;
    var height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    var mid = height / 2;
    ctx.fillStyle = "#263653";
    ctx.fillRect(0, mid, width, 1);
    if (!track.buffer || track.failed) {
      return;
    }
    var pxPerSec = this.pxPerSec * this.dpr;
    var x0 = track.offsetS * pxPerSec;
    var bufferPx = track.buffer.duration * pxPerSec;
    var startCol = Math.max(0, Math.floor(x0));
    var endCol = Math.min(width, Math.ceil(x0 + bufferPx));
    var channelA = track.buffer.getChannelData(0);
    var channelB =
      track.buffer.numberOfChannels > 1 ? track.buffer.getChannelData(1) : null;
    var samplesPerCol = channelA.length / bufferPx;
    ctx.fillStyle = track.color;
    for (var col = startCol; col < endCol; col += 1) {
      var s0 = Math.floor((col - x0) * samplesPerCol);
      var s1 = Math.min(
        channelA.length,
        Math.max(s0 + 1, Math.floor((col + 1 - x0) * samplesPerCol))
      );
      var min = 1;
      var max = -1;
      var step = Math.max(1, Math.floor((s1 - s0) / 64));
      for (var index = s0; index < s1; index += step) {
        var value = channelA[index];
        if (channelB) {
          value = (value + channelB[index]) / 2;
        }
        if (value < min) {
          min = value;
        }
        if (value > max) {
          max = value;
        }
      }
      var top = mid - max * (mid - 2);
      var bottom = mid - min * (mid - 2);
      ctx.fillRect(col, top, 1, Math.max(1, bottom - top));
    }
  };

  MixerSession.prototype.layoutPlayhead = function () {
    if (!this.playhead) {
      return;
    }
    this.playhead.style.left = this.position * this.pxPerSec + "px";
  };

  MixerSession.prototype.updatePositionUI = function () {
    if (this.positionLabel) {
      this.positionLabel.textContent =
        formatClock(this.position) + " / " + formatClock(this.duration);
    }
    if (this.seekSlider && !this.seekDragging) {
      this.seekSlider.value = String(Math.round(this.position * 1000));
    }
  };

  MixerSession.prototype.dispose = function () {
    this.disposed = true;
    this.stopClock();
    this.stopAllSources();
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    lastSnapshot = {
      urls: this.tracks.map(function (track) {
        return track.url;
      }),
      position: this.position
    };
  };

  function scan() {
    for (var index = sessions.length - 1; index >= 0; index -= 1) {
      if (!sessions[index].root.isConnected) {
        sessions[index].dispose();
        sessions.splice(index, 1);
      }
    }
    var roots = document.querySelectorAll(".mtm-mixer-root:not([data-mtm-init])");
    roots.forEach(function (root) {
      root.setAttribute("data-mtm-init", "1");
      var session = new MixerSession(root);
      sessions.push(session);
      session.init();
    });
  }

  var scanTimer = null;
  function scheduleScan() {
    if (scanTimer !== null) {
      return;
    }
    scanTimer = window.setTimeout(function () {
      scanTimer = null;
      scan();
    }, 30);
  }

  if (typeof document !== "undefined") {
    var observer = new MutationObserver(scheduleScan);
    observer.observe(document.documentElement, { childList: true, subtree: true });
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", scheduleScan);
    } else {
      scheduleScan();
    }
  }
})();
"""


def mixer_head() -> str:
    """Return the ``Blocks(head=...)`` fragment with the runtime and styles."""
    return (
        f"<style>\n{TRACK_MIXER_CSS}\n</style>\n"
        f"<script>\n{TRACK_MIXER_JS}\n</script>"
    )
