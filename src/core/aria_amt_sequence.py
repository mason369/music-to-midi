"""Official Aria window reconciliation with incomplete-output checks."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class AriaSequenceStitcher:
    def __init__(self, upstream, tokenizer, duration_seconds, translate):
        self.upstream = upstream
        self.tokenizer = tokenizer
        self.duration_ms = duration_seconds * 1000.0
        self.translate = translate
        self.sequence = [tokenizer.bos_tok]
        self.concat_sequence = [tokenizer.bos_tok]
        self.covered_until_ms = 0
        self.reconciliation_error = None

    def check_window(self, index):
        window_start = min(index * self.upstream.CHUNK_LEN_MS, self.duration_ms)
        if self.reconciliation_error is not None and window_start > self.covered_until_ms:
            self._raise_incomplete(window_start)

    def append(self, index, generated_sequence, input_sequence, *, context_exhausted=False):
        upstream = self.upstream
        tokenizer = self.tokenizer
        try:
            next_sequence = upstream._truncate_seq(
                generated_sequence,
                upstream.CHUNK_LEN_MS,
                upstream.LEN_MS - upstream.CHUNK_LEN_MS,
            )
            if context_exhausted or len(generated_sequence) >= upstream.MAX_BLOCK_LEN:
                raise RuntimeError(self.translate("aria_amt.context_exhausted"))
        except InterruptedError:
            raise
        except Exception as exc:
            logger.warning("Aria-AMT window %d reconciliation failed: %s", index, exc)
            self.reconciliation_error = exc
            try:
                self.sequence = upstream._truncate_seq(
                    input_sequence,
                    upstream.CHUNK_LEN_MS - 2,
                    upstream.CHUNK_LEN_MS,
                )
            except InterruptedError:
                raise
            except Exception:
                self.sequence = [tokenizer.bos_tok]
            return False

        if generated_sequence[-1] == tokenizer.eos_tok:
            generated_sequence = generated_sequence[:-1]
        self.concat_sequence += upstream._shift_onset(
            generated_sequence[len(input_sequence) :],
            index * upstream.CHUNK_LEN_MS,
        )
        self.sequence = [tokenizer.bos_tok] if len(next_sequence) == 1 else next_sequence
        self.covered_until_ms = min(
            self.duration_ms,
            index * upstream.CHUNK_LEN_MS + upstream.LEN_MS - upstream.CHUNK_LEN_MS,
        )
        if self.reconciliation_error is not None:
            logger.info("Aria-AMT window %d recovered the overlapping interval", index)
        self.reconciliation_error = None
        return True

    def finish(self):
        if self.reconciliation_error is not None and self.covered_until_ms < self.duration_ms:
            self._raise_incomplete(self.duration_ms)

    def _raise_incomplete(self, end_ms):
        raise RuntimeError(
            self.translate(
                "aria_amt.incomplete",
                start=f"{self.covered_until_ms / 1000.0:.3f}",
                end=f"{end_ms / 1000.0:.3f}",
                error=str(self.reconciliation_error),
            )
        ) from self.reconciliation_error
