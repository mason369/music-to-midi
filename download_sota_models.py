from pathlib import Path

from download_multistem_model import download_multistem_model
from download_vocal_harmony_model import download_chorus_model
from download_vocal_model import download_vocal_model
from src.utils.yourmt3_downloader import (
    OFFICIAL_YOURMT3_MODEL_KEYS,
    YOURMT3_MODELS,
    download_model,
    get_model_path,
)


def download_official_yourmt3_models() -> dict[str, Path]:
    """Download every official YourMT3+ mode exposed by the app UI."""
    downloaded = {}
    total = len(OFFICIAL_YOURMT3_MODEL_KEYS)

    print(f"Preparing {total} official YourMT3+ model modes...")
    for index, model_key in enumerate(OFFICIAL_YOURMT3_MODEL_KEYS, start=1):
        model_info = YOURMT3_MODELS[model_key]
        label = model_info.get("ui_label", model_key)
        checkpoint = model_info.get("checkpoint", "")
        print("")
        print(f"[{index}/{total}] {label}")
        print(f"checkpoint: {checkpoint}")

        download_model(model_key)
        model_path = get_model_path(model_key)
        if model_path is None:
            raise RuntimeError(f"Downloaded {label}, but no checkpoint can be resolved for '{model_key}'.")

        downloaded[model_key] = Path(model_path)
        print(f"ready: {model_path}")

    print("")
    print("All official YourMT3+ model modes are ready.")
    return downloaded


def download_sota_models() -> dict[str, object]:
    """Prepare the full SOTA model set used by the app's public workflows."""
    yourmt3_models = download_official_yourmt3_models()

    print("")
    print("Preparing BS-RoFormer SW Fixed six-stem separation assets...")
    six_stem_model, six_stem_config = download_multistem_model()
    print(f"ready: {six_stem_model}")
    print(f"ready: {six_stem_config}")

    print("")
    print("Preparing RoFormer vocal_rvc ensemble models...")
    vocal_model = download_vocal_model()
    print(f"ready: {vocal_model}")

    print("")
    print("Preparing RoFormer karaoke ensemble models...")
    karaoke_model = download_chorus_model()
    print(f"ready: {karaoke_model}")

    return {
        "yourmt3": yourmt3_models,
        "six_stem": {
            "model": six_stem_model,
            "config": six_stem_config,
        },
        "vocal_rvc": {
            "primary_model": vocal_model,
        },
        "karaoke": {
            "primary_model": karaoke_model,
        },
    }


if __name__ == "__main__":
    download_sota_models()
