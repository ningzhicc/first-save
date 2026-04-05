#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_qwen.sh" \
  --plm-type llama \
  --plm-size small \
  --plm-path ../downloaded_plms/llama3.2/base \
  "$@"
