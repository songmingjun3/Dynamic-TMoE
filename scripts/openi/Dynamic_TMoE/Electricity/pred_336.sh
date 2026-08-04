#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

exec python -u "$REPO_ROOT/train_openi.py" --model Dynamic_TMoE --dataset Electricity --pred-len 336 "$@"
