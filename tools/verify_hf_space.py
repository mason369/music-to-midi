"""Wait for the exact pushed Space revision to serve its Gradio application."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request

# Match huggingface_hub._space_api.INTERMEDIATE_SPACE_STAGES, including
# replacement-app startup while the previous instance is still serving.
PENDING_STAGES = {"BUILDING", "RUNNING_BUILDING", "APP_STARTING", "RUNNING_APP_STARTING"}


def read_json(url: str, token: str = "") -> dict:
    headers = {"Accept": "application/json", "Cache-Control": "no-cache"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def wait_for_space(
    repository: str,
    revision: str,
    token: str,
    *,
    timeout: float = 1800,
    interval: float = 15,
    fetch=read_json,
    sleep=time.sleep,
    now=time.monotonic,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid Space repository")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected an exact Space commit SHA")
    if not token:
        raise ValueError("HF_TOKEN is required to verify the deployed Space")
    deadline = now() + timeout
    while True:
        info = fetch(f"https://huggingface.co/api/spaces/{repository}", token)
        if info.get("sha") != revision:
            raise RuntimeError("Space revision differs from the pushed deployment")
        runtime = info.get("runtime", {})
        stage = runtime.get("stage")
        print(f"Space {repository} at {revision}: {stage}", flush=True)
        if stage == "RUNNING":
            subdomain = info.get("subdomain", "")
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", subdomain):
                raise RuntimeError("Space API did not return a valid application subdomain")
            # The Hub token stays on huggingface.co; the public application does
            # not need it. A successful Git push or HTTP landing page alone is
            # insufficient: require the actual Gradio UI and event graph.
            config = fetch(f"https://{subdomain}.hf.space/config")
            expected_version = info.get("cardData", {}).get("sdk_version")
            if (
                not expected_version
                or config.get("version") != expected_version
                or not config.get("components")
                or not config.get("dependencies")
            ):
                raise RuntimeError("Space did not serve the declared Gradio application")
            print(f"Verified live Space application: {subdomain}.hf.space", flush=True)
            return
        if stage not in PENDING_STAGES:
            raise RuntimeError(
                f"Space deployment failed: {stage}: {runtime.get('errorMessage', '')}"
            )
        remaining = deadline - now()
        if remaining <= 0:
            raise TimeoutError(f"Space did not become RUNNING within {timeout:g} seconds")
        # This observes an in-flight deployment, never restarts or retries a
        # failed build, failed request, dependency install or inference job.
        sleep(min(interval, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    wait_for_space(args.repository, args.revision, os.environ.get("HF_TOKEN", ""))


if __name__ == "__main__":
    main()
