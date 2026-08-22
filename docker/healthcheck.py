#!/usr/bin/env python3
"""Fail unless the internal API and every configured production gate are ready."""

from __future__ import annotations

import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def main() -> int:
    url = "http://127.0.0.1:8765/api/v1/ready"
    try:
        with urlopen(url, timeout=15) as response:
            body = response.read().decode("utf-8", "replace")
            if response.status != 200:
                raise RuntimeError(f"readiness returned HTTP {response.status}: {body}")
            payload = json.loads(body)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        print(f"readiness returned HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except (OSError, URLError, ValueError, RuntimeError) as exc:
        print(f"readiness probe failed: {exc}", file=sys.stderr)
        return 1
    if payload.get("status") != "ready":
        print(f"readiness payload is not ready: {payload}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
