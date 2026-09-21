#!/usr/bin/env bash

set -u -o pipefail

CRATES=(crane-core crane-ai crane-serve)
MAX_RETRIES="${MAX_RETRIES:-12}"
RETRY_DELAY="${RETRY_DELAY:-10}"
LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT

for crate in "${CRATES[@]}"; do
  published=false

  for ((attempt = 1; attempt <= MAX_RETRIES; attempt++)); do
    command=(cargo publish --package "$crate" --registry crates-io --allow-dirty)

    echo
    echo "[$crate] attempt $attempt/$MAX_RETRIES"
    printf '$'
    printf ' %q' "${command[@]}"
    printf '\n'

    if "${command[@]}" 2>&1 | tee "$LOG_FILE"; then
      published=true
      break
    fi

    if grep -Eqi 'already exists on crates\.io index|already uploaded|already published' "$LOG_FILE"; then
      echo "[$crate] already published, skip."
      published=true
      break
    fi

    if ((attempt < MAX_RETRIES)); then
      echo "[$crate] failed, retry in ${RETRY_DELAY}s..."
      sleep "$RETRY_DELAY"
    fi
  done

  if [[ "$published" != true ]]; then
    echo "[$crate] publish failed after $MAX_RETRIES attempts." >&2
    exit 1
  fi
done

echo
echo "All three crates are published."
