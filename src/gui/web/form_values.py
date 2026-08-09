"""Shared normalization for values emitted by the Gradio platforms."""

from __future__ import annotations


def normalize_optional_project_bpm(value: object) -> float | None:
    """Map the Web UI's explicit ``0 = automatic`` sentinel to ``None``.

    Gradio 4.44 renders an empty ``gr.Number`` as zero in the browser. The
    component therefore permits exactly that sentinel while ``Config`` keeps
    validating every non-zero override against the real 4--400 BPM range.
    """

    if value is None or value == "":
        return None
    numeric = float(value)
    return None if numeric == 0.0 else numeric


__all__ = ["normalize_optional_project_bpm"]
