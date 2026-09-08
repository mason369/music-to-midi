"""Exercise process pipes with a real child, including noisy bank diagnostics."""

import subprocess
import sys
import time

from src.core import muscriptor_result_assets as assets


def test_synthesizer_diagnostics_cannot_fill_pipe_and_block_export(tmp_path, monkeypatch):
    real_popen = subprocess.Popen

    def process(command, **options):
        output = command[command.index("-F") + 1]
        code = "import sys,wave;sys.stderr.write('diagnostic '*30000);f=wave.open(sys.argv[1],'wb');f.setparams((2,2,48000,0,'NONE',''));f.writeframes(b'\\0'*9600);f.close()"
        return real_popen([sys.executable, "-c", code, output], **options)

    monkeypatch.setattr(assets.subprocess, "Popen", process)
    monkeypatch.setattr(assets, "get_fluidsynth_subprocess_env", lambda _: None)
    started = time.monotonic()
    target = tmp_path / "render.wav"
    assets._run_fluidsynth(
        tmp_path / "synth.exe",
        tmp_path / "bank.sf2",
        tmp_path / "input.mid",
        target,
        cancel_check=lambda: time.monotonic() - started > 2,
    )
    assert target.read_bytes().startswith(b"RIFF")
