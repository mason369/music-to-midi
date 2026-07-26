"""Download and strictly validate pinned gated MuScriptor checkpoints."""

from __future__ import annotations

import argparse

from src.models.data_models import MuscriptorModel
from src.utils.muscriptor_downloader import (
    download_muscriptor_large_model,
    download_muscriptor_model,
    get_muscriptor_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        choices=[model.value for model in MuscriptorModel] + ["all"],
        default=MuscriptorModel.LARGE.value,
        help="Checkpoint size to prepare (default: large).",
    )
    args = parser.parse_args()
    sizes = (
        [model.value for model in MuscriptorModel]
        if args.size == "all"
        else [args.size]
    )
    for model_size in sizes:
        artifact = get_muscriptor_artifact(model_size)
        weights, config = download_muscriptor_model(model_size)
        print(f"{artifact.display_name} revision: {artifact.revision}")
        print(f"model: {weights}")
        print(f"model SHA-256: {artifact.model_sha256}")
        print(f"config: {config}")
        print(f"config SHA-256: {artifact.config_sha256}")


if __name__ == "__main__":
    main()
