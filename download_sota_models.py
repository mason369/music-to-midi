from pathlib import Path

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


if __name__ == "__main__":
    download_official_yourmt3_models()
