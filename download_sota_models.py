from pathlib import Path

from download_accompaniment_model import download_accompaniment_model
from download_aria_amt_model import download_aria_model
from download_bytedance_piano_model import download_bytedance_piano_model
from download_miros_model import prepare_miros_model
from download_multistem_model import download_multistem_model
from download_muscriptor_model import download_muscriptor_large_model
from download_transkun_v2_aug_model import download_transkun_v2_aug_model
from download_vocal_model import download_vocal_model
from src.utils.fluidsynth_runtime import download_fluidsynth_windows
from src.utils.muscriptor_soundfont_downloader import download_muscriptor_soundfont
from src.core.transkun_transcriber import (
    TRANSKUN_CONF_NAME,
    TRANSKUN_CONF_SHA256,
    TRANSKUN_CONF_SIZE,
    TRANSKUN_PACKAGE_NAME,
    TRANSKUN_PACKAGE_VERSION,
    TRANSKUN_WEIGHT_NAME,
    TRANSKUN_WEIGHT_SHA256,
    TRANSKUN_WEIGHT_SIZE,
    TranskunTranscriber,
)
from src.utils.yourmt3_downloader import (
    OFFICIAL_YOURMT3_MODEL_KEYS,
    YOURMT3_MODELS,
    download_model,
    get_model_path,
)

# Historical aggregate-downloader API name.  It now resolves the PolarFormer
# accompaniment leg; no karaoke/chorus model is used by the current route.
download_chorus_model = download_accompaniment_model


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
            raise RuntimeError(
                f"Downloaded {label}, but no checkpoint can be resolved for '{model_key}'."
            )

        downloaded[model_key] = Path(model_path)
        print(f"ready: {model_path}")

    print("")
    print("All official YourMT3+ model modes are ready.")
    return downloaded


def validate_default_transkun_runtime() -> dict[str, object]:
    """Require the exact TransKun package and packaged V2 resource identities."""
    unavailable_reason = TranskunTranscriber.get_unavailable_reason()
    if unavailable_reason:
        raise RuntimeError(
            f"Default TransKun {TRANSKUN_PACKAGE_VERSION} runtime validation failed: "
            f"{unavailable_reason}"
        )

    transcriber = TranskunTranscriber()
    if not transcriber.is_model_available():
        raise RuntimeError(
            f"Default TransKun {TRANSKUN_PACKAGE_VERSION} packaged resources are "
            "missing or failed exact identity validation: "
            f"{TRANSKUN_WEIGHT_NAME} expected {TRANSKUN_WEIGHT_SIZE} bytes / "
            f"SHA-256 {TRANSKUN_WEIGHT_SHA256}; "
            f"{TRANSKUN_CONF_NAME} expected {TRANSKUN_CONF_SIZE} bytes / "
            f"SHA-256 {TRANSKUN_CONF_SHA256}."
        )

    return {
        "package": TRANSKUN_PACKAGE_NAME,
        "version": TRANSKUN_PACKAGE_VERSION,
        "weight_name": TRANSKUN_WEIGHT_NAME,
        "weight_size": TRANSKUN_WEIGHT_SIZE,
        "weight_sha256": TRANSKUN_WEIGHT_SHA256,
        "config_name": TRANSKUN_CONF_NAME,
        "config_size": TRANSKUN_CONF_SIZE,
        "config_sha256": TRANSKUN_CONF_SHA256,
    }


def download_sota_models() -> dict[str, object]:
    """Prepare the full SOTA model set used by the app's public workflows."""
    print("Preparing exact TransKun 2.0.1 package and built-in V2 resources...")
    transkun_runtime = validate_default_transkun_runtime()
    print(
        "ready: "
        f"{transkun_runtime['package']}=={transkun_runtime['version']} "
        f"({transkun_runtime['weight_name']}, {transkun_runtime['config_name']})"
    )

    print("")
    yourmt3_models = download_official_yourmt3_models()

    print("")
    print("Preparing pinned MIROS source plus finetuned and pretrained weights...")
    miros_repo = prepare_miros_model()
    print(f"ready: {miros_repo}")

    print("")
    print("Preparing BS-RoFormer SW Fixed six-stem separation assets...")
    six_stem_model, six_stem_config = download_multistem_model()
    print(f"ready: {six_stem_model}")
    print(f"ready: {six_stem_config}")

    print("")
    print("Preparing BS-RoFormer Leap XE 90-band vocals assets...")
    vocal_model = download_vocal_model()
    print(f"ready: {vocal_model}")

    print("")
    print("Preparing BS PolarFormer public ONNX accompaniment assets...")
    accompaniment_model = download_chorus_model()
    print(f"ready: {accompaniment_model}")

    print("")
    print("Preparing official TransKun V2 Aug checkpoint assets...")
    transkun_v2_aug_dir = download_transkun_v2_aug_model()
    print(f"ready: {transkun_v2_aug_dir}")

    print("")
    print("Preparing exact Aria-AMT checkpoint assets...")
    aria_checkpoint = download_aria_model()
    print(f"ready: {aria_checkpoint}")

    print("")
    print("Preparing exact ByteDance pedal-aware piano checkpoint assets...")
    bytedance_checkpoint = download_bytedance_piano_model()
    print(f"ready: {bytedance_checkpoint}")

    print("\n[10/12] Preparing gated MuScriptor-large checkpoint...")
    muscriptor_weights, muscriptor_config = download_muscriptor_large_model()
    print(f"ready: {muscriptor_weights}")

    print("\n[11/12] Preparing MuScriptor official playback SoundFont...")
    muscriptor_soundfont = download_muscriptor_soundfont()
    print(f"ready: {muscriptor_soundfont}")

    print("\n[12/12] Verifying the real FluidSynth playback runtime...")
    fluidsynth_executable = download_fluidsynth_windows()
    print(f"ready: {fluidsynth_executable}")

    return {
        "transkun": transkun_runtime,
        "yourmt3": yourmt3_models,
        "miros": {
            "repo_dir": miros_repo,
        },
        "six_stem": {
            "model": six_stem_model,
            "config": six_stem_config,
        },
        "vocal_leap_xe": {
            "primary_model": vocal_model,
        },
        "accompaniment_polarformer": {
            "primary_model": accompaniment_model,
        },
        "transkun_v2_aug": {
            "model_dir": transkun_v2_aug_dir,
        },
        "aria_amt": {
            "checkpoint": aria_checkpoint,
        },
        "bytedance_piano": {
            "checkpoint": bytedance_checkpoint,
        },
        "muscriptor": {
            "weights": muscriptor_weights,
            "config": muscriptor_config,
            "soundfont": muscriptor_soundfont,
            "fluidsynth": fluidsynth_executable,
        },
        # Deprecated compatibility keys retained for aggregate-downloader callers.
        "vocal_rvc": {
            "primary_model": vocal_model,
        },
        "karaoke": {
            "primary_model": accompaniment_model,
        },
    }


if __name__ == "__main__":
    download_sota_models()
