"""Historical installer imports for the current Leap Instrumental model."""

import sys

from download_accompaniment_model import (
    CHORUS_MODEL,
    CHORUS_MODELS,
    CHORUS_PRESET,
    DEFAULT_CACHE_DIR,
    download_accompaniment_model,
    download_chorus_model,
    is_accompaniment_model_available,
    is_chorus_model_available,
    main,
    resolve_accompaniment_config_path,
    resolve_accompaniment_model_path,
    resolve_chorus_model_path,
    resolve_chorus_model_paths,
)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
