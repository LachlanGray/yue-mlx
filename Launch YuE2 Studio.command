#!/bin/bash
# Finder opens executable .command files in Terminal.
cd -- "$(dirname -- "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "The local Python environment is missing. Run these commands in this folder:"
  echo "  uv venv --python 3.13 .venv"
  echo "  uv pip install --python .venv/bin/python -e ."
  read -r -p "Press Return to close. "
  exit 1
fi
echo "Starting YuE2 Studio. Your browser will open automatically."
echo "Keep this Terminal window open. Press Ctrl+C to stop the server."
exec .venv/bin/python -m yue2_mlx.web --open
