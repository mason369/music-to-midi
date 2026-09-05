"""Native command-line interface for Music to MIDI."""


def main(*args, **kwargs):
    """Import the CLI lazily so ``python -m`` can validate its runtime first."""

    from .app import main as run_cli

    return run_cli(*args, **kwargs)


__all__ = ["main"]
