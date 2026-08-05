#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${LEXIPILOT_ENV_FILE:-.env}"
PLANNER_ARGS=()

if [[ "${LEXIPILOT_DETERMINISTIC:-0}" == "1" ]]; then
    PLANNER_ARGS+=(--deterministic)
fi

if [[ "${LEXIPILOT_DEMO:-0}" == "1" ]]; then
    PROFILE="${LEXIPILOT_PROFILE:-demo}"
    exec python3 lexipilot.py \
        --demo \
        --profile "$PROFILE" \
        --env-file "$ENV_FILE" \
        "${PLANNER_ARGS[@]}" \
        --debug
fi

PROFILE="${LEXIPILOT_PROFILE:-default}"
exec python3 lexipilot.py \
    --profile "$PROFILE" \
    --env-file "$ENV_FILE" \
    --backup-profile \
    "${PLANNER_ARGS[@]}" \
    --debug
