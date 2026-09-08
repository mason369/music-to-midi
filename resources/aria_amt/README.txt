Aria-AMT inference directory preparation

The pinned upstream AudioTransform scans assets/impulse, assets/noise and
assets/applause even when inference uses augment=False. This file is bundled
inside a subdirectory of each directory so read-only portable installations
already contain the required directories. Upstream only reads immediate files,
so this documentation is never treated as audio. No training audio is needed
or substituted for inference, and the upstream source remains unmodified.
