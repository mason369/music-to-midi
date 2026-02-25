
import spaces

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'amt/src')))

import subprocess
from typing import Tuple, Dict, Literal
from ctypes import ArgumentError

from html_helper import *
from model_helper import *

import torchaudio
import glob
import gradio as gr
from gradio_log import Log
from pathlib import Path

# gradio_log
log_file = 'amt/log.txt'
Path(log_file).touch()

# @title Load Checkpoint
model_name = 'YPTF.MoE+Multi (noPS)' # @param ["YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"]
precision = '16'# if torch.cuda.is_available() else '32'# @param ["32", "bf16-mixed", "16"]
project = '2024'

if model_name == "YMT3+":
    checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
    args = [checkpoint, '-p', project, '-pr', precision]
elif model_name == "YPTF+Single (noPS)":
    checkpoint = "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
    args = [checkpoint, '-p', project, '-enc', 'perceiver-tf', '-ac', 'spec',
            '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF+Multi (PS)":
    checkpoint = "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256',
            '-dec', 'multi-t5', '-nl', '26', '-enc', 'perceiver-tf',
            '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF.MoE+Multi (noPS)":
    checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
            '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
            '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
            '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF.MoE+Multi (PS)":
    checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
            '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
            '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
            '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
else:
    raise ValueError(model_name)

model = load_model_checkpoint(args=args, device="cpu")
model.to("cuda")
# @title GradIO helper


def prepare_media(source_path_or_url: os.PathLike,
                  source_type: Literal['audio_filepath', 'youtube_url'],
                  delete_video: bool = True,
                  simulate = False) -> Dict:
    """prepare media from source path or youtube, and return audio info"""
    # Get audio_file
    if source_type == 'audio_filepath':
        audio_file = source_path_or_url
    elif source_type == 'youtube_url':
        if os.path.exists('/download/yt_audio.mp3'):
            os.remove('/download/yt_audio.mp3')
        # Download from youtube
        with open(log_file, 'w') as lf:
            audio_file = './downloaded/yt_audio'
            command = ['yt-dlp', '-x', source_path_or_url, '-f', 'bestaudio',
                '-o', audio_file, '--audio-format', 'mp3', '--restrict-filenames',
                '--extractor-retries', '10',
                '--force-overwrites', '--username', 'oauth2', '--password', '', '-v']
            if simulate:
                command = command + ['-s']
            process = subprocess.Popen(command,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
            for line in iter(process.stdout.readline, ''):
                # Filter out unnecessary messages
                print(line)
                if "www.google.com/device" in line:
                    hl_text = line.replace("https://www.google.com/device", "\033[93mhttps://www.google.com/device\x1b[0m").split()
                    hl_text[-1] = "\x1b[31;1m" + hl_text[-1] + "\x1b[0m"
                    lf.write(' '.join(hl_text)); lf.flush()
                elif "Authorization successful" in line or "Video unavailable" in line:
                    lf.write(line); lf.flush()
            process.stdout.close()
            process.wait()
        
        audio_file += '.mp3'
    else:
        raise ValueError(source_type)

    # Create info
    info = torchaudio.info(audio_file)
    return {
        "filepath": audio_file,
        "track_name": os.path.basename(audio_file).split('.')[0],
        "sample_rate": int(info.sample_rate),
        "bits_per_sample": int(info.bits_per_sample),
        "num_channels": int(info.num_channels),
        "num_frames": int(info.num_frames),
        "duration": int(info.num_frames / info.sample_rate),
        "encoding": str.lower(info.encoding),
        }

@spaces.GPU
def process_audio(audio_filepath):
    if audio_filepath is None:
        return None
    audio_info = prepare_media(audio_filepath, source_type='audio_filepath')
    midifile = transcribe(model, audio_info)
    midifile = to_data_url(midifile)
    return create_html_from_midi(midifile) # html midiplayer

# This is a temporary function for using pre-transcribed midi
@spaces.GPU
def process_audio_yt_temp(youtube_url):
    if youtube_url is None:
        return None
    elif youtube_url == "https://youtu.be/5vJBhdjvVcE?si=s3NFG_SlVju0Iklg":
        midifile = "./mid/Free Jazz Intro Music - Piano Sway (Intro B - 10 seconds) - OurMusicBox.mid"
    elif youtube_url == "https://youtu.be/mw5VIEIvuMI?si=Dp9UFVw00Tl8CXe2":
        midifile = "./mid/Naomi Scott   Speechless from Aladdin Official Video Sony vevo Music.mid"
    elif youtube_url == "https://youtu.be/OXXRoa1U6xU?si=dpYMun4LjZHNydSb":
        midifile = "./mid/Mozart_Sonata_for_Piano_and_Violin_(getmp3.pro).mid"
    midifile = to_data_url(midifile)
    return create_html_from_midi(midifile) # html midiplayer


@spaces.GPU
def process_video(youtube_url):
    if 'youtu' not in youtube_url:
        return None
    audio_info = prepare_media(youtube_url, source_type='youtube_url')
    midifile = transcribe(model, audio_info)
    midifile = to_data_url(midifile)
    return create_html_from_midi(midifile) # html midiplayer

def play_video(youtube_url):
    if 'youtu' not in youtube_url:
        return None
    return create_html_youtube_player(youtube_url)

# def oauth_google():
#     return create_html_oauth()

AUDIO_EXAMPLES = glob.glob('examples/*.*', recursive=True)
YOUTUBE_EXAMPLES = ["https://youtu.be/5vJBhdjvVcE?si=s3NFG_SlVju0Iklg",
                    "https://youtu.be/mw5VIEIvuMI?si=Dp9UFVw00Tl8CXe2",
                    "https://youtu.be/OXXRoa1U6xU?si=dpYMun4LjZHNydSb"]
# YOUTUBE_EXAMPLES = ["https://youtu.be/5vJBhdjvVcE?si=s3NFG_SlVju0Iklg",
#                     "https://www.youtube.com/watch?v=vMboypSkj3c",
#                     "https://youtu.be/vRd5KEjX8vw?si=b-qw633ZjaX6Uxy5",
#                     "https://youtu.be/bnS-HK_lTHA?si=PQLVAab3QHMbv0S3https://youtu.be/zJB0nnOc7bM?si=EA1DN8nHWJcpQWp_",
#                     "https://youtu.be/7mjQooXt28o?si=qqmMxCxwqBlLPDI2",
#                     "https://youtu.be/mIWYTg55h10?si=WkbtKfL6NlNquvT8"]

theme = gr.Theme.from_hub("gradio/dracula_revamped")
theme.text_md = '10px'
theme.text_lg = '12px'

theme.body_background_fill_dark = '#060a1c' #'#372037'# '#a17ba5' #'#73d3ac'
theme.border_color_primary_dark = '#45507328'
theme.block_background_fill_dark = '#3845685c'

theme.body_text_color_dark = 'white'
theme.block_title_text_color_dark = 'black'
theme.body_text_color_subdued_dark = '#e4e9e9'

css = """
.gradio-container {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    height: 100vh;
}
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
#mylog {font-size: 12pt; line-height: 1.2; min-height: 2em; max-height: 4em;}  
"""

with gr.Blocks(theme=theme, css=css) as demo:

    with gr.Row():
        with gr.Column(scale=10):
            gr.Markdown(
            f"""
            ## 🎶YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer Architectures and Cross-dataset Stem Augmentation
            - Model name: `{model_name}`
                <details>
                <summary>▶model details◀</summary>
                     
                | **Component**            | **Details**                                      |
                |--------------------------|--------------------------------------------------|
                | Encoder backbone         | Perceiver-TF + Mixture of Experts (2/8)          |
                | Decoder backbone         | Multi-channel T5-small                           |
                | Tokenizer                | MT3 tokens with Singing extension                |
                | Dataset                  | YourMT3 dataset                                  |
                | Augmentation strategy    | Intra-/Cross dataset stem augment, No Pitch-shifting |
                | FP Precision             | BF16-mixed for training, FP16 for inference      |
                </details>
            
            ## Caution:
            - For acadmic reproduction purpose, we strongly recommend to use [Colab Demo](https://colab.research.google.com/drive/1AgOVEBfZknDkjmSRA7leoa81a2vrnhBG?usp=sharing) with multiple checkpoints.

            ## YouTube transcription (Sorry!! YouTube blocked HuggingFace IP. We display a few pre-transcribed examples in the below!):
            - Select one from the `Examples`, click `Get Audio from YouTube`, and then press `Transcribe`.
            
            <div style="display: inline-block;">
                <a href="https://arxiv.org/abs/2407.04822">
                    <img src="https://img.shields.io/badge/arXiv:2407.04822-B31B1B?logo=arxiv&logoColor=fff&style=plastic" alt="arXiv Badge"/>
                </a>
            </div>
            <div style="display: inline-block;">
                <a href="https://github.com/mimbres/YourMT3">
                    <img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=fff&style=plastic" alt="GitHub Badge"/>
                </a>
            </div>
            <div style="display: inline-block;">
                <a href="https://colab.research.google.com/drive/1AgOVEBfZknDkjmSRA7leoa81a2vrnhBG?usp=sharing">
                    <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=fff&style=plastic"/>
                </a>
            </div>
            """)

    with gr.Group():

        with gr.Tab("From YouTube"):
            with gr.Column(scale=4):
                # Input URL
                youtube_url = gr.Textbox(label="YouTube Link URL",
                        placeholder="https://youtu.be/...")
                # Display examples
                gr.Examples(examples=YOUTUBE_EXAMPLES, inputs=youtube_url)
                # Play button
                play_video_button = gr.Button("Get Audio from YouTube", variant="primary")
                # Play youtube
                youtube_player = gr.HTML(render=True)

            with gr.Column(scale=4):
                    with gr.Row():
                        # Submit button
                        transcribe_video_button = gr.Button("Transcribe", variant="primary")
                        # Oauth button
                        oauth_button = gr.Button("google.com/device", variant="primary", link="https://www.google.com/device")
                    
            with gr.Column(scale=1):
                # Transcribe
                output_tab2 = gr.HTML(render=True)
                # video_output = gr.Text(label="Video Info")
                transcribe_video_button.click(process_audio_yt_temp, inputs=youtube_url, outputs=output_tab2)
                # transcribe_video_button.click(process_video, inputs=youtube_url, outputs=output_tab2)
                # Play
                play_video_button.click(play_video, inputs=youtube_url, outputs=youtube_player)
            with gr.Column(scale=1):
                Log(log_file, dark=True, xterm_font_size=12, elem_id='mylog')

        with gr.Tab("Upload audio"):
            # Input
            audio_input = gr.Audio(label="Record Audio", type="filepath",
                                show_share_button=True, show_download_button=True)
            # Display examples
            gr.Examples(examples=AUDIO_EXAMPLES, inputs=audio_input)
            # Submit button
            transcribe_audio_button = gr.Button("Transcribe", variant="primary")
            # Transcribe
            output_tab1 = gr.HTML()
            transcribe_audio_button.click(process_audio, inputs=audio_input, outputs=output_tab1)

demo.launch(debug=True)
