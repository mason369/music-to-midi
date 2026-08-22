"""Shared brand assets for the browser-based frontends (Space and Colab)."""

from __future__ import annotations

import base64
from pathlib import Path

APP_ICON_PATH = (
    Path(__file__).resolve().parents[3] / "resources" / "icons" / "app_icon.png"
)


def app_icon_data_uri() -> str:
    """Return the desktop app icon as a data URI for inline web headers.

    The icon is a tracked repository asset. A missing file means the deployed
    tree is incomplete, so this raises loudly instead of silently falling back
    to a placeholder.
    """
    payload = APP_ICON_PATH.read_bytes()
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
