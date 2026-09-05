"""Process options for non-interactive native tools used by every delivery surface."""

from __future__ import annotations

import os
import subprocess


def hidden_subprocess_kwargs() -> dict[str, int]:
    """Prevent a Windows console window while preserving pipes, errors and cancellation.

    Apply only to background tools. Interactive CLI process groups must keep their
    console attachment so Windows can deliver Ctrl+C/Ctrl+Break to them.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
