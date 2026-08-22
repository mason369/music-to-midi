"""Find which desktop inference sequence contaminates the separation runtime."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.utils.runtime_paths import activate_audio_separator_runtime

CASES = {
    "y1": {"name": "ymt3-plus", "mode": "smart", "yourmt3_model": "ymt3_plus"},
    "y2": {
        "name": "yptf-single-nops",
        "mode": "smart",
        "yourmt3_model": "yptf_single_nops",
    },
    "y3": {
        "name": "yptf-multi-ps",
        "mode": "smart",
        "yourmt3_model": "yptf_multi_ps",
    },
    "y4": {
        "name": "yptf-moe-multi-nops",
        "mode": "smart",
        "yourmt3_model": "yptf_moe_multi_nops",
    },
    "y5": {
        "name": "yptf-moe-multi-ps",
        "mode": "smart",
        "yourmt3_model": "yptf_moe_multi_ps",
    },
    "miros": {"name": "miros", "mode": "smart", "backend": "miros"},
    "ml": {
        "name": "muscriptor-large",
        "mode": "smart",
        "backend": "muscriptor",
        "muscriptor_model": "large",
    },
    "mm": {
        "name": "muscriptor-medium",
        "mode": "smart",
        "backend": "muscriptor",
        "muscriptor_model": "medium",
    },
    "ms": {
        "name": "muscriptor-small",
        "mode": "smart",
        "backend": "muscriptor",
        "muscriptor_model": "small",
    },
    "t1": {"name": "transkun", "mode": "piano_transkun", "audio": "piano"},
    "t2": {
        "name": "transkun-v2-aug",
        "mode": "piano_transkun_v2_aug",
        "audio": "piano",
    },
    "aria": {"name": "aria", "mode": "piano_aria_amt", "audio": "piano"},
    "byte": {
        "name": "bytedance",
        "mode": "piano_bytedance_pedal",
        "audio": "piano",
    },
}


def _probe() -> None:
    activate_audio_separator_runtime()
    import librosa
    import onnxruntime
    import soundfile
    import torch
    import yaml
    from audio_separator.separator import Separator
    from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import BSRoformer

    print(
        "PROBE PASS",
        f"librosa={librosa.__version__}",
        f"onnxruntime={onnxruntime.__version__}",
        f"soundfile={soundfile.__version__}",
        f"torch={torch.__version__}",
        f"yaml={yaml.__version__}",
        f"Separator={Separator.__name__}",
        f"BSRoformer={BSRoformer.__name__}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="Comma-separated probe case ids")
    parser.add_argument("--mix-audio", required=True)
    parser.add_argument("--piano-audio", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--preload-separation", action="store_true")
    args = parser.parse_args()

    selected = [item.strip() for item in args.cases.split(",") if item.strip()]
    unknown = [item for item in selected if item not in CASES]
    if unknown:
        raise ValueError(f"unknown probe cases: {unknown}")
    mix_audio = Path(args.mix_audio).resolve()
    piano_audio = Path(args.piano_audio).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.preload_separation:
        print("PRELOAD SEPARATION RUNTIME", flush=True)
        _probe()

    from PyQt6.QtWidgets import QApplication, QDialog

    from src.gui.main_window import MainWindow
    from tools.app_real_song_acceptance import (
        _click_primary,
        _close_window,
        _new_window,
    )

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)

    with (
        mock.patch.object(MainWindow, "_start_gpu_detection", return_value=None),
        mock.patch.object(QDialog, "exec", return_value=0),
    ):
        for index, case_id in enumerate(selected, start=1):
            case = CASES[case_id]
            print(f"RUN {index}/{len(selected)} {case_id} {case['name']}", flush=True)
            case_root = output_root / f"{index:02d}-{case_id}"
            case_root.mkdir(parents=True, exist_ok=True)
            window = _new_window(case, case_root)
            audio = piano_audio if case.get("audio") == "piano" else mix_audio
            _click_primary(app, window, audio, timeout_seconds=600.0)
            _close_window(app, window)
            print(f"  COMPLETE {case_id}", flush=True)

    try:
        _probe()
    except Exception:
        print("PROBE FAIL", flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
