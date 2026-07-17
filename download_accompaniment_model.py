"""Public/compatibility entry point for the PolarFormer accompaniment assets.

The implementation remains in ``download_vocal_harmony_model`` so existing
installations that import the historical module keep working.  New callers
should use the accompaniment terminology exported here.
"""

from __future__ import annotations

import sys

from download_vocal_harmony_model import (
    CHORUS_MODEL,
    CHORUS_MODELS,
    DEFAULT_CACHE_DIR,
    POLARFORMER_CONFIG_NAME,
    POLARFORMER_ONNX_NAME,
    POLARFORMER_ONNX_SHA256,
    POLARFORMER_ONNX_SIZE,
    POLARFORMER_REPO_ID,
    POLARFORMER_REVISION,
    download_accompaniment_model,
    download_chorus_model,
    is_accompaniment_model_available,
    is_chorus_model_available,
    main,
    resolve_accompaniment_config_path,
    resolve_accompaniment_model_path,
    resolve_chorus_model_path,
    resolve_chorus_model_paths,
)

__all__ = [
    "CHORUS_MODEL",
    "CHORUS_MODELS",
    "DEFAULT_CACHE_DIR",
    "POLARFORMER_CONFIG_NAME",
    "POLARFORMER_ONNX_NAME",
    "POLARFORMER_ONNX_SHA256",
    "POLARFORMER_ONNX_SIZE",
    "POLARFORMER_REPO_ID",
    "POLARFORMER_REVISION",
    "download_accompaniment_model",
    "download_chorus_model",
    "is_accompaniment_model_available",
    "is_chorus_model_available",
    "resolve_accompaniment_config_path",
    "resolve_accompaniment_model_path",
    "resolve_chorus_model_path",
    "resolve_chorus_model_paths",
]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
