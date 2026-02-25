# @title HTML helper
import re
import base64
def to_data_url(midi_filename):
    """ This is crucial for Colab/WandB support. Thanks to Scott Hawley!!
        https://github.com/drscotthawley/midi-player/blob/main/midi_player/midi_player.py

    """
    with open(midi_filename, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    return 'data:audio/midi;base64,'+encoded_string.decode('utf-8')


def to_youtube_embed_url(video_url):
    regex = r"(?:https:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?(.+)"
    return re.sub(regex, r"https://www.youtube.com/embed/\1",video_url)


def create_html_from_midi(midifile):
    html_template = """
<!DOCTYPE html>
<html>
<head>
  <title>Awesome MIDI Player</title>
  <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0">
  </script>
  <style>
    /* Background color for the section */
    #proll {{background-color:transparent}}

    /* Custom player style */
    #proll midi-player {{
      display: block;
      width: inherit;
      margin: 4px;
      margin-bottom: 0;
      transform-origin: top; 
      transform: scaleY(0.8); /* Added scaleY */
    }}

    #proll midi-player::part(control-panel) {{
      background: #d8dae880;
      border-radius: 8px 8px 0 0;
      border: 1px solid #A0A0A0;
    }}

    /* Custom visualizer style */
    #proll midi-visualizer .piano-roll-visualizer {{
      background: #45507328;
      border-radius: 0 0 8px 8px;
      border: 1px solid #A0A0A0;
      margin: 4px;
      margin-top: 1;
      overflow: auto;
      transform-origin: top; 
      transform: scaleY(0.8); /* Added scaleY */
    }}

    #proll midi-visualizer svg rect.note {{
      opacity: 0.6;
      stroke-width: 2;
    }}

    #proll midi-visualizer svg rect.note[data-instrument="0"] {{
      fill: #e22;
      stroke: #055;
    }}

    #proll midi-visualizer svg rect.note[data-instrument="2"] {{
      fill: #2ee;
      stroke: #055;
    }}

    #proll midi-visualizer svg rect.note[data-is-drum="true"] {{
      fill: #888;
      stroke: #888;
    }}

    #proll midi-visualizer svg rect.note.active {{
      opacity: 0.9;
      stroke: #34384F;
    }}

    /* Media queries for responsive scaling */
    @media (max-width: 700px) {{ #proll midi-visualizer .piano-roll-visualizer {{transform-origin: top; transform: scaleY(0.75);}} }}
    @media (max-width: 500px) {{ #proll midi-visualizer .piano-roll-visualizer {{transform-origin: top; transform: scaleY(0.7);}} }}
    @media (max-width: 400px) {{ #proll midi-visualizer .piano-roll-visualizer {{transform-origin: top; transform: scaleY(0.6);}} }}
    @media (max-width: 300px) {{ #proll midi-visualizer .piano-roll-visualizer {{transform-origin: top; transform: scaleY(0.5);}} }}
  </style>
</head>
<body>
  <div>
    <a href="{midifile}" target="_blank" style="font-size: 14px;">Download MIDI</a> <br>
  </div>
  <div>
    <section id="proll">
      <midi-player src="{midifile}" sound-font="https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus" visualizer="#proll midi-visualizer">
      </midi-player>
      <midi-visualizer src="{midifile}">
      </midi-visualizer>
    </section>
  </div>

</body>
</html>
""".format(midifile=midifile)
    html = f"""<div style="display: flex; justify-content: center; align-items: center;">
                  <iframe style="width: 100%; height: 500px; overflow:hidden" srcdoc='{html_template}'></iframe>
            </div>"""
    return html

def create_html_youtube_player(youtube_url):
    youtube_url = to_youtube_embed_url(youtube_url)
    html = f"""
    <div style="display: flex; justify-content: center; align-items: center; position: relative; width: 100%; height: 100%;">
        <style>
            .responsive-iframe {{ width: 560px; height: 315px; transform-origin: top left; transition: width 0.3s ease, height 0.3s ease; }}
            @media (max-width: 560px) {{ .responsive-iframe {{ width: 100%; height: 100%; }} }}
        </style>
        <iframe class="responsive-iframe" src="{youtube_url}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>
    """
    return html

def create_html_oauth():
    html = f"""
    <div style="display: flex; justify-content: center; align-items: center; position: relative; width: 100%; height: 100%;">
        <style>
            .responsive-link {{ display: inline-block; padding: 10px 20px; text-align: center; font-size: 16px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px; transition: background-color 0.3s ease; }}
            .responsive-link:hover {{ background-color: #0056b3; }}
        </style>
        <a href="https://www.google.com/device" target="_blank" rel="noopener noreferrer" class="responsive-link">
            Open Google Device Page
        </a>
    </div>
    """
    return html