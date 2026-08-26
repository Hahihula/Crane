#!/usr/bin/env bash
# Send one Qwen3-ASR request to an already-running crane-serve instance.
#
# Usage:
#   ./tests/test_asr.sh [server_url] [audio_file]
#
# Examples:
#   ./tests/test_asr.sh
#   ./tests/test_asr.sh http://127.0.0.1:8080 data/audio/kinsenka_3.wav
#
# The response is deliberately not parsed or reformatted: it prints the raw
# JSON returned by POST /v1/audio/transcriptions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_URL="${1:-${CRANE_ASR_URL:-http://127.0.0.1:8080}}"
AUDIO_FILE="${2:-$ROOT_DIR/data/audio/kinsenka_3.wav}"

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required" >&2
  exit 1
fi

if [[ ! -f "$AUDIO_FILE" ]]; then
  echo "error: audio file not found: $AUDIO_FILE" >&2
  exit 1
fi

SERVER_URL="${SERVER_URL%/}"
echo "POST $SERVER_URL/v1/audio/transcriptions"
echo "Audio: $AUDIO_FILE"
echo "Raw JSON response:"

# This is the only HTTP request made by the script. `model` is sent for
# OpenAI API compatibility; crane-serve uses the ASR model loaded at startup.
curl --fail-with-body --silent --show-error \
  --request POST "$SERVER_URL/v1/audio/transcriptions" \
  --form "file=@$AUDIO_FILE" \
  --form "model=Qwen3-ASR-0.6B"
echo
