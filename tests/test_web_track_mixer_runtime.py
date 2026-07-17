"""Contract tests for the shared browser multi-track mixer runtime.

The desktop Qt mixer (src/gui/widgets/audio_track_mixer.py) is the reference
implementation; the Gradio Space and the Colab notebook must expose the same
controls with the same semantics through
src/gui/web/track_mixer_runtime.py.
"""

import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.gui.web import track_mixer_runtime as mixer
from src.i18n.translator import Translator


def test_mixer_constants_match_the_desktop_widget():
    from src.gui.widgets import audio_track_mixer as desktop

    assert mixer.MIN_VOLUME_DB == desktop._MIN_VOLUME_DB
    assert mixer.MAX_VOLUME_DB == desktop._MAX_VOLUME_DB
    assert mixer.MIN_OFFSET_SECONDS * 1000 == desktop._MIN_OFFSET_MS
    assert mixer.MAX_OFFSET_SECONDS * 1000 == desktop._MAX_OFFSET_MS
    assert mixer.MIN_ZOOM == 1
    assert mixer.MAX_ZOOM == 16
    assert mixer.TRACK_COLORS == desktop._TRACK_COLORS
    assert mixer.PLAYHEAD_COLOR == "#ff5d73"


def test_mixer_javascript_uses_the_desktop_semantics():
    js = mixer.TRACK_MIXER_JS

    assert "var MIN_VOLUME_DB = -60.0;" in js
    assert "var MAX_VOLUME_DB = 0.0;" in js
    assert "var MIN_OFFSET_SECONDS = -10.0;" in js
    assert "var MAX_OFFSET_SECONDS = 10.0;" in js
    assert "var MIN_ZOOM = 1;" in js
    assert "var MAX_ZOOM = 16;" in js
    assert 'var PLAYHEAD_COLOR = "#ff5d73";' in js

    # dB -> linear gain identical to the desktop _db_to_linear helper.
    assert "Math.pow(10.0, db / 20.0)" in js
    assert "db <= MIN_VOLUME_DB" in js

    # Offset scheduling mirrors local_position = global_position - offset_ms.
    assert "var local = self.playStartPosition - track.offsetS;" in js

    # Duration is the max of (offset + track duration), like _recompute_duration.
    assert "duration = Math.max(duration, track.offsetS + track.buffer.duration);" in js

    # Web Audio primitives required for shared-transport playback.
    for marker in (
        "AudioContext",
        "decodeAudioData",
        "createBufferSource",
        "createGain",
        "requestAnimationFrame",
        "MutationObserver",
    ):
        assert marker in js

    # The mouse wheel must stay inert: scrolling only scrolls the page.
    assert "wheel" not in js


def test_mixer_javascript_is_syntactically_valid(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node runtime is unavailable for JS syntax validation")
    js_path = tmp_path / "track_mixer_runtime.js"
    js_path.write_text(mixer.TRACK_MIXER_JS, encoding="utf-8")
    result = subprocess.run(
        [node, "--check", str(js_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_track_mixer_strings_exist_in_every_catalog_language():
    for language in sorted(Translator.AVAILABLE_LANGUAGES):
        translator = Translator(language)
        strings = mixer.track_mixer_strings(translator.t)
        assert set(strings) == set(mixer._MIXER_STRING_KEYS)
        for key, value in strings.items():
            assert isinstance(value, str) and value.strip(), (language, key)
            assert not value.startswith("dialogs."), (language, key, value)


def test_build_track_mixer_html_embeds_a_parseable_manifest(tmp_path):
    translator = Translator("zh_CN")
    audio = tmp_path / "song_vocals.wav"
    audio.write_bytes(b"RIFF")
    tracks = [
        {
            "id": "vocals",
            "name": "vocals",
            "audio_path": str(audio),
            "color": "#ff70a6",
        },
        {"id": "local_1", "name": "自定义 <Track>", "audio_path": str(audio), "color": ""},
    ]

    markup = mixer.build_track_mixer_html(tracks, translator.t)

    assert markup.startswith('<div class="mtm-mixer-root">')
    assert '<pre class="mtm-manifest" hidden>' in markup
    assert markup.endswith("</div>")
    assert '<div class="mtm-mixer"></div>' in markup
    # Raw HTML metacharacters from track names must never reach the DOM.
    assert "自定义 <Track>" not in markup

    manifest_text = re.search(
        r'<pre class="mtm-manifest" hidden>(.*?)</pre>', markup, re.DOTALL
    ).group(1)
    manifest = json.loads(html.unescape(manifest_text))
    assert [entry["id"] for entry in manifest["tracks"]] == ["vocals", "local_1"]
    assert manifest["tracks"][0]["name"] == translator.t(
        "dialogs.complete.audio_tracks.track_names.vocals"
    )
    # Unknown (added) track names pass through untranslated.
    assert manifest["tracks"][1]["name"] == "自定义 <Track>"
    # Missing colors fall back to the shared palette's first color.
    assert manifest["tracks"][1]["color"] == mixer.TRACK_COLORS[0]
    for entry in manifest["tracks"]:
        assert entry["url"].startswith("/file=")
        assert entry["fileName"] == "song_vocals.wav"


def test_display_track_name_matches_the_desktop_mapping():
    from src.gui.widgets import audio_track_mixer as desktop

    # Both platforms cover exactly the same known stem identities.
    assert mixer.KNOWN_TRACK_NAMES == desktop._KNOWN_TRACK_NAMES

    # Both resolve through the same shared i18n keys in every language.
    for language in sorted(Translator.AVAILABLE_LANGUAGES):
        translator = Translator(language)
        for raw_name in sorted(mixer.KNOWN_TRACK_NAMES):
            assert mixer.display_track_name(raw_name, translator.t) == translator.t(
                f"dialogs.complete.audio_tracks.track_names.{raw_name}"
            )
        # Unknown names (added local files) pass through untranslated.
        assert mixer.display_track_name("my-local-take", translator.t) == "my-local-take"


def test_track_file_url_quotes_paths_like_gradio_file_routes():
    assert mixer.track_file_url("/tmp/request-abc/song_vocals.wav") == (
        "/file=/tmp/request-abc/song_vocals.wav"
    )
    quoted = mixer.track_file_url("/tmp/request abc/音 轨.wav")
    assert quoted.startswith("/file=/tmp/")
    assert " " not in quoted
    assert "音" not in quoted  # non-ASCII must be percent-encoded


def test_mixer_head_contains_runtime_and_styles():
    head = mixer.mixer_head()
    assert "<style>" in head and "<script>" in head
    assert mixer.TRACK_MIXER_CSS.strip() in head
    assert mixer.TRACK_MIXER_JS.strip() in head


def test_space_and_colab_share_the_single_mixer_runtime():
    space_source = Path("space/app.py").read_text(encoding="utf-8")
    notebook = json.loads(Path("colab_notebook.ipynb").read_text(encoding="utf-8"))
    colab_source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    for source in (space_source, colab_source):
        assert "from src.gui.web.track_mixer_runtime import (" in source
        assert "build_track_mixer_html" in source
        assert "mixer_head()" in source
        # Track audio is served through /file= which requires allowed_paths.
        assert "allowed_paths=" in source

    # Both platforms localize mixer labels through the shared i18n catalog.
    assert "track_mixer_strings" in space_source or "build_track_mixer_html(state[\"tracks\"], st)" in space_source
    assert "build_track_mixer_html(normalized[\"tracks\"], COLAB_TRANSLATOR.t)" in colab_source

    # Per-track removal exists on every platform now.
    assert "fn=_remove_track" in space_source
    assert "_make_track_remove_handler" in colab_source
