#!/usr/bin/env bash
set -euo pipefail

echo "Setting up repo..."
if [ ! -f "pyproject.toml" ]; then
  git clone https://github.com/NewGuy012/chat-shakespeare.git
  cd chat-shakespeare
fi

uv sync
.venv/bin/marimo edit main.py
