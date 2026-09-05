"""Run the supported source CLI with the same runtime gate as the desktop app."""

from __future__ import annotations

import sys

from src.utils.runtime_paths import bootstrap_runtime_environment
from src.utils.source_runtime import require_source_runtime_identity
from src.utils.warnings_filter import setup_chinese_environment


def _main() -> int:
    require_source_runtime_identity()
    setup_chinese_environment()
    bootstrap_runtime_environment()

    from src.cli.app import main

    return main()


if __name__ == "__main__":
    sys.exit(_main())
