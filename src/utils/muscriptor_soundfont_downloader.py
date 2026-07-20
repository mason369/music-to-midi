"""Exact MuseScore General SoundFont used by the official MuScriptor demo."""

from __future__ import annotations

from pathlib import Path

from src.utils.artifact_identity import validate_file_identity
from src.utils.runtime_paths import get_resource_path

MUSCRIPTOR_ASSETS_REPO_ID = "MuScriptor/assets"
MUSCRIPTOR_ASSETS_REVISION = "7755beb2da7cb1d3c663ff4a9ad0d0e99437f78f"
MUSCRIPTOR_SF2_FILENAME = "MuseScore_General.sf2"
MUSCRIPTOR_SF2_EXACT_BYTES = 215_614_036
MUSCRIPTOR_SF2_SHA256 = "ee51d2c4b1525e70f19a45909c4fd7a2e26d91d115fa89dbf5a6bc413d8b9bf3"
MUSCRIPTOR_SF3_FILENAME = "MuseScore_General.sf3"
MUSCRIPTOR_SF3_EXACT_BYTES = 39_900_972
MUSCRIPTOR_SF3_SHA256 = "5b85b6c2c61d10b2b91cddd41efcce7b25cd31c8271d511c73afafbef20b6fa3"


def download_muscriptor_soundfont(
    filename: str = MUSCRIPTOR_SF2_FILENAME,
    *,
    printer=print,
) -> Path:
    identities = {
        MUSCRIPTOR_SF2_FILENAME: (MUSCRIPTOR_SF2_EXACT_BYTES, MUSCRIPTOR_SF2_SHA256),
        MUSCRIPTOR_SF3_FILENAME: (MUSCRIPTOR_SF3_EXACT_BYTES, MUSCRIPTOR_SF3_SHA256),
    }
    try:
        expected_size, expected_sha256 = identities[filename]
    except KeyError as exc:
        raise ValueError(f"Unsupported MuScriptor soundfont: {filename!r}") from exc

    bundled = get_resource_path(f"models/muscriptor_assets/{filename}")
    if bundled.is_file():
        validate_file_identity(
            bundled,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=f"bundled MuScriptor official {filename}",
        )
        printer(f"Bundled SoundFont identity verified: {bundled}")
        return bundled

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download the SoundFont") from exc

    printer(
        f"Preparing {filename} from {MUSCRIPTOR_ASSETS_REPO_ID}@" f"{MUSCRIPTOR_ASSETS_REVISION}"
    )
    path = Path(
        hf_hub_download(
            repo_id=MUSCRIPTOR_ASSETS_REPO_ID,
            filename=filename,
            revision=MUSCRIPTOR_ASSETS_REVISION,
        )
    )
    validate_file_identity(
        path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        label=f"MuScriptor official {filename}",
    )
    printer(f"SoundFont identity verified: {path}")
    return path
