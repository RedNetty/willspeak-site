#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SC=${SC:-.venv/bin/python}
exec "$SC" -u scripts/asr/long_runs.py acoustic "$@"
