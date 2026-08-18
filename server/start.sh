#!/usr/bin/env bash
# CrowdDirector sidecar - macOS / Linux launcher.
# Creates a local venv on first run, installs requirements, then serves on ws://localhost:8765.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip --quiet
    echo "Installing requirements (this downloads PyTorch, a few hundred MB, once)..."
    .venv/bin/python -m pip install -r requirements.txt
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    cat <<'MSG'

  ANTHROPIC_API_KEY is not set.
  The per-tick director does NOT need it - the model runs locally.
  It is required only to generate a scene from a description and to
  interpret free-text instructions.

  Set it with:  export ANTHROPIC_API_KEY=sk-ant-...

MSG
fi

exec .venv/bin/python crowd_director_server.py
