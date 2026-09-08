"""One Aria inference job per process; errors reach the owning project runner."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aria-AMT 单文件转写进程")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args(argv)

    from src.core.aria_amt_transcriber import AriaAmtTranscriber

    transcriber = AriaAmtTranscriber(checkpoint_path=args.checkpoint)
    # The project runner already owns the file queue. Avoid nesting the upstream
    # batch workers, whose joins do not propagate a failed GPU process.
    transcriber._run_transcription_windows_single_file(args.input, args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
