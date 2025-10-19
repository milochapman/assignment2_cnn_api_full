#!/usr/bin/env bash
set -euo pipefail
# Generate uv lockfile and a frozen requirements.lock.txt
# Usage: bash scripts/generate_lockfiles.sh
cd "$(dirname "$0")/.."

# Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv lock
uv export --frozen --format requirements-txt > requirements.lock.txt

echo "Generated files:"
echo " - uv.lock"
echo " - requirements.lock.txt"