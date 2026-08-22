#!/usr/bin/env bash

set -euo pipefail

cd /app

fail() {
    printf 'Music to MIDI container startup failed: %s\n' "$*" >&2
    exit 2
}

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        fail "required environment variable ${name} is empty"
    fi
}

require_positive_integer() {
    local name="$1"
    local value="${!name:-}"
    [[ "$value" =~ ^[0-9]+$ ]] || fail "${name} must be a positive integer"
    (( value > 0 )) || fail "${name} must be greater than zero"
}

require_nonnegative_integer() {
    local name="$1"
    local value="${!name:-}"
    [[ "$value" =~ ^[0-9]+$ ]] || fail "${name} must be a nonnegative integer"
}

verify_source_runtime() {
    python -m src.utils.source_runtime
}

command_name="${1:-server}"
shift || true

case "$command_name" in
    model-init)
        require_env MUSIC_TO_MIDI_ENABLED_PROFILES
        [[ "${MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES:-}" == "1" ]] \
            || fail "MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES must be 1"
        verify_source_runtime
        exec python -m src.model_profiles prepare \
            --profiles "$MUSIC_TO_MIDI_ENABLED_PROFILES" "$@"
        ;;
    verify-models)
        require_env MUSIC_TO_MIDI_ENABLED_PROFILES
        [[ "${MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES:-}" == "1" ]] \
            || fail "MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES must be 1"
        verify_source_runtime
        exec python -m src.model_profiles verify \
            --profiles "$MUSIC_TO_MIDI_ENABLED_PROFILES" \
            --require-ready "$@"
        ;;
    server)
        require_env PUBLIC_ORIGIN
        require_env MUSIC_TO_MIDI_ENABLED_PROFILES
        require_positive_integer MAX_UPLOAD_BYTES
        require_nonnegative_integer MAX_QUEUED_JOBS
        require_nonnegative_integer MIN_FREE_BYTES
        require_nonnegative_integer RETENTION_DAYS
        require_nonnegative_integer RETENTION_MAX_JOBS
        require_nonnegative_integer RETENTION_MAX_BYTES
        [[ "$PUBLIC_ORIGIN" == https://* ]] \
            || fail "PUBLIC_ORIGIN must use https:// for a public deployment"
        [[ "$PUBLIC_ORIGIN" != */ ]] \
            || fail "PUBLIC_ORIGIN must not end with a slash"
        [[ "${MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT:-}" == "1" ]] \
            || fail "MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT must be 1"
        [[ "${MUSIC_TO_MIDI_EDGE_AUTH:-}" == "basic" ]] \
            || fail "MUSIC_TO_MIDI_EDGE_AUTH must be basic"
        [[ "${MUSIC_TO_MIDI_TLS_TERMINATED_AT_EDGE:-}" == "1" ]] \
            || fail "MUSIC_TO_MIDI_TLS_TERMINATED_AT_EDGE must be 1"
        [[ "${MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES:-}" == "1" ]] \
            || fail "MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES must be 1"
        verify_source_runtime
        python -m src.model_profiles verify \
            --profiles "$MUSIC_TO_MIDI_ENABLED_PROFILES" \
            --require-ready
        exec python -m src.web_api \
            --config /tmp/music-to-midi-backend.json \
            --host 0.0.0.0 \
            --port 8765 \
            --data-dir /data/jobs \
            --cors-origin "$PUBLIC_ORIGIN" \
            --log-level "${LOG_LEVEL:-info}" \
            --max-upload-bytes "$MAX_UPLOAD_BYTES" \
            --max-queued-jobs "$MAX_QUEUED_JOBS" \
            --min-free-bytes "$MIN_FREE_BYTES" \
            --retention-days "$RETENTION_DAYS" \
            --retention-max-jobs "$RETENTION_MAX_JOBS" \
            --retention-max-bytes "$RETENTION_MAX_BYTES" \
            "$@"
        ;;
    server-selfhost)
        require_env SELF_HOST_PORT
        require_env MUSIC_TO_MIDI_ENABLED_PROFILES
        require_positive_integer SELF_HOST_PORT
        require_positive_integer MAX_UPLOAD_BYTES
        require_nonnegative_integer MAX_QUEUED_JOBS
        require_nonnegative_integer MIN_FREE_BYTES
        require_nonnegative_integer RETENTION_DAYS
        require_nonnegative_integer RETENTION_MAX_JOBS
        require_nonnegative_integer RETENTION_MAX_BYTES
        (( SELF_HOST_PORT <= 65535 )) \
            || fail "SELF_HOST_PORT must be between 1 and 65535"
        [[ "${MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT:-0}" == "0" ]] \
            || fail "server-selfhost requires MUSIC_TO_MIDI_PUBLIC_DEPLOYMENT=0"
        [[ "${MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES:-}" == "1" ]] \
            || fail "MUSIC_TO_MIDI_REQUIRE_ENABLED_PROFILES must be 1"
        [[ "${HF_HUB_OFFLINE:-}" == "1" ]] \
            || fail "server-selfhost requires HF_HUB_OFFLINE=1"
        [[ "${TRANSFORMERS_OFFLINE:-}" == "1" ]] \
            || fail "server-selfhost requires TRANSFORMERS_OFFLINE=1"
        verify_source_runtime
        python -m src.model_profiles verify \
            --profiles "$MUSIC_TO_MIDI_ENABLED_PROFILES" \
            --require-ready
        exec python -m src.web_api \
            --config /tmp/music-to-midi-backend.json \
            --host 0.0.0.0 \
            --port 8765 \
            --data-dir /data/jobs \
            --cors-origin "http://127.0.0.1:$SELF_HOST_PORT" \
            --cors-origin "http://localhost:$SELF_HOST_PORT" \
            --log-level "${LOG_LEVEL:-info}" \
            --max-upload-bytes "$MAX_UPLOAD_BYTES" \
            --max-queued-jobs "$MAX_QUEUED_JOBS" \
            --min-free-bytes "$MIN_FREE_BYTES" \
            --retention-days "$RETENTION_DAYS" \
            --retention-max-jobs "$RETENTION_MAX_JOBS" \
            --retention-max-bytes "$RETENTION_MAX_BYTES" \
            "$@"
        ;;
    *)
        fail "unsupported command '$command_name'; expected server, server-selfhost, model-init or verify-models"
        ;;
esac
