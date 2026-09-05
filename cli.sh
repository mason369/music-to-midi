#!/usr/bin/env bash
# Music to MIDI - Linux/WSL native CLI launcher

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${REPO_DIR}/venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    printf 'Music to MIDI CLI environment is missing: %s\n' "$VENV_PYTHON" >&2
    printf 'Run ./install.sh in the project directory first; global Python and CPU fallback are not used.\n' >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    set -- --help
fi

export MUSIC_TO_MIDI_ACCELERATOR=cuda
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "$VENV_PYTHON" -m src.cli "$@"
