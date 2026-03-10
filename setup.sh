#!/bin/bash
# This is a comment and will be ignored by the shell.
echo "Setting up repo..."

sudo ldconfig
git clone https://github.com/NewGuy012/chat-shakespeare.git
cd chat-shakespeare/
uv sync
source .venv/bin/activate
marimo edit