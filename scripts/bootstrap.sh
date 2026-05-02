#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
python -c "import yaml; print('PyYAML OK')"
