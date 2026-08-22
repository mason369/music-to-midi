# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc
FROM ${CUDA_IMAGE}

ARG VCS_REF=unknown
ARG BUILD_VERSION=dev

LABEL org.opencontainers.image.title="Music to MIDI backend" \
      org.opencontainers.image.description="NVIDIA CUDA 12.8 inference API; model assets are prepared separately" \
      org.opencontainers.image.source="https://github.com/mason369/music-to-midi" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.licenses="MIT"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/app/venv \
    PATH=/app/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MUSIC_TO_MIDI_ACCELERATOR=cuda \
    MUSIC_TO_MIDI_MIROS_DIR=/models/miros \
    HOME=/models/home \
    HF_HOME=/models/huggingface \
    HF_HUB_CACHE=/models/huggingface/hub \
    XDG_CACHE_HOME=/models/home/.cache \
    TORCH_HOME=/models/home/.cache/torch \
    MPLCONFIGDIR=/models/home/.cache/matplotlib \
    NUMBA_CACHE_DIR=/models/home/.cache/numba \
    TMPDIR=/tmp \
    MPLBACKEND=Agg \
    WANDB_MODE=disabled \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        fluidsynth \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libportaudio2 \
        libsamplerate0-dev \
        libsndfile1 \
        libsndfile1-dev \
        pkg-config \
        portaudio19-dev \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /app/venv \
    && /app/venv/bin/python -m pip install --no-cache-dir \
        pip==25.2 \
        setuptools==80.10.2 \
        wheel==0.45.1

WORKDIR /app

COPY requirements.txt docker/backend-constraints.txt /tmp/

# The CUDA trio is installed from the exact cu128 index first. The headless API
# deliberately omits the desktop-only PyQt and python-rtmidi packages.
RUN python -m pip install --no-cache-dir \
        torch==2.7.0 \
        torchaudio==2.7.0 \
        torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && grep -vE '^[[:space:]]*(torch|torchaudio|torchvision|numpy|python-rtmidi|PyQt6|PyQt6-Qt6|PyQt6-sip|onnxruntime|onnxruntime-gpu)([[:space:]=<>!~@;\[]|$)' \
        /tmp/requirements.txt > /tmp/requirements-container.txt \
    && python -m pip install --no-cache-dir numpy==1.26.4 \
    && python -m pip install --no-cache-dir \
        --constraint /tmp/backend-constraints.txt \
        -r /tmp/requirements-container.txt \
    && python -m pip install --no-cache-dir \
        --constraint /tmp/backend-constraints.txt \
        beartype==0.18.5 \
        diffq-fixed==0.2.4 \
        julius==0.2.7 \
        ml_collections==1.1.0 \
        onnx-weekly==1.21.0.dev20260223 \
        onnx2torch-py313==1.6.0 \
        onnxruntime-gpu==1.23.2 \
        pydub==0.25.1 \
        resampy==0.4.3 \
        rotary-embedding-torch==0.6.5 \
        samplerate==0.1.0 \
        six==1.17.0 \
    && python -m pip install --no-cache-dir audio-separator==0.44.1 --no-deps \
    && python -m pip install --no-cache-dir \
        "muscriptor @ https://github.com/muscriptor/muscriptor/archive/d73147e75e5b9b0c0a79ebe154587db4fd603e0c.zip" \
        --no-deps \
    && python -m pip install --no-cache-dir \
        "aria-amt @ https://github.com/EleutherAI/aria-amt/archive/a1ab73fc901d1759ec3bc173c146b3c6a3040261.zip" \
        --no-deps \
    && rm -f \
        /tmp/backend-constraints.txt \
        /tmp/requirements.txt \
        /tmp/requirements-container.txt

COPY pyproject.toml LICENSE README.md THIRD_PARTY_NOTICES.md ./
COPY download_*.py ./
COPY src ./src
COPY YourMT3 ./YourMT3
COPY tools/generate_third_party_sbom.py ./tools/generate_third_party_sbom.py
COPY docker/backend-entrypoint.sh docker/healthcheck.py ./docker/

# These gates are build-time identity checks only. GPU execution and every
# selected checkpoint are checked again by the runtime readiness endpoint.
RUN python - <<'PY'
from importlib import metadata

import torch

expected = {
    "torch": "2.7.0",
    "torchaudio": "2.7.0",
    "torchvision": "0.22.0",
    "numpy": "1.26.4",
    "onnxruntime-gpu": "1.23.2",
    "audio-separator": "0.44.1",
    "transkun": "2.0.1",
    "muscriptor": "0.3.0",
}
actual = {name: metadata.version(name).split("+", 1)[0] for name in expected}
if actual != expected:
    raise RuntimeError(f"container dependency identity mismatch: expected={expected}, actual={actual}")
if torch.version.cuda != "12.8" or getattr(torch.version, "hip", None):
    raise RuntimeError(
        f"expected CUDA 12.8 PyTorch without ROCm, got cuda={torch.version.cuda!r}, "
        f"hip={getattr(torch.version, 'hip', None)!r}"
    )
print("container dependency identities verified", actual)
PY
RUN python -m src.utils.source_runtime \
    && python - <<'PY'
from download_sota_models import validate_default_transkun_runtime
from src.utils.yourmt3_source_identity import validate_patched_yourmt3_source

print("YourMT3 source:", validate_patched_yourmt3_source("/app/YourMT3/amt/src"))
print("TransKun default runtime:", validate_default_transkun_runtime())
PY
RUN python tools/generate_third_party_sbom.py --output /app/THIRD_PARTY_SBOM.txt \
    && test -s /app/THIRD_PARTY_SBOM.txt \
    && chmod 0755 /app/docker/backend-entrypoint.sh /app/docker/healthcheck.py \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --home-dir /models/home --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p \
        /data/jobs \
        /data/tmp \
        /models/home/.cache/matplotlib \
        /models/home/.cache/numba \
        /models/home/.cache/torch \
        /models/huggingface/hub \
        /models/miros \
    && chown -R 10001:10001 /data /models

USER 10001:10001

EXPOSE 8765

ENTRYPOINT ["/app/docker/backend-entrypoint.sh"]
CMD ["server"]
