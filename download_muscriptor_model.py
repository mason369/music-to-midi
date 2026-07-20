"""Download and strictly validate the pinned gated MuScriptor-large weights."""

from src.utils.muscriptor_downloader import (
    MUSCRIPTOR_MODEL_SHA256,
    MUSCRIPTOR_REPO_REVISION,
    download_muscriptor_large_model,
)

if __name__ == "__main__":
    weights, config = download_muscriptor_large_model()
    print(f"MuScriptor-large revision: {MUSCRIPTOR_REPO_REVISION}")
    print(f"model: {weights}")
    print(f"model SHA-256: {MUSCRIPTOR_MODEL_SHA256}")
    print(f"config: {config}")
