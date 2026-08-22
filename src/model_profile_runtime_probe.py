"""Isolated native-runtime probes used by source and frozen Web backends."""

from __future__ import annotations

import json


MODEL_PROFILE_RUNTIME_PROBE_SWITCH = "--model-profile-runtime-probe"


def run_model_profile_runtime_probe(profile_id: str) -> int:
    """Import the exact native stack required by one separation profile.

    The caller deliberately runs this function in a fresh process so CUDA,
    OpenVINO and audio-separator DLL loading cannot contaminate the long-lived
    HTTP process.  Any import, device or provider failure is allowed to escape
    and produce a non-zero subprocess exit code with its original traceback.
    """

    if profile_id == "vocal_split":
        from src.core.vocal_separator import _resolve_onnx_providers
        from src.utils.gpu_utils import get_device
        from src.utils.runtime_paths import activate_audio_separator_runtime

        activate_audio_separator_runtime()
        import librosa  # noqa: F401
        import onnxruntime as ort
        import soundfile  # noqa: F401
        import torch  # noqa: F401
        import yaml  # noqa: F401
        from audio_separator.separator.uvr_lib_v5.roformer.bs_roformer import (
            BSRoformer,  # noqa: F401
        )

        device = get_device(prefer_gpu=True, gpu_index=0)
        selected_providers = _resolve_onnx_providers(device, ort)
        print(
            json.dumps(
                {
                    "device": device,
                    "available_providers": ort.get_available_providers(),
                    "selected_providers": selected_providers,
                }
            )
        )
        return 0

    if profile_id == "six_stem_split":
        from src.utils.audio_separator_compat import get_separator_cls

        get_separator_cls()
        print("audio-separator runtime ready")
        return 0

    raise ValueError(f"unsupported audio-separator profile: {profile_id}")


__all__ = [
    "MODEL_PROFILE_RUNTIME_PROBE_SWITCH",
    "run_model_profile_runtime_probe",
]
