#!/bin/bash
# Finder opens executable .command files in Terminal.
cd -- "$(dirname -- "$0")" || exit 1
PYTHON="python3"
if [ -x .venv/bin/python ]; then
  PYTHON=".venv/bin/python"
fi
if ! command -v yt-dlp >/dev/null 2>&1; then
  echo "yt-dlp is missing. Install it first (for example: brew install yt-dlp)."
  read -r -p "Press Return to close. "
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is missing. Install it first (for example: brew install ffmpeg)."
  read -r -p "Press Return to close. "
  exit 1
fi
echo "Starting YouTube Audio Downloader. Your browser will open automatically."
echo "Keep this Terminal window open. Press Ctrl+C to stop the server."
exec "$PYTHON" youtube_audio_downloader.py --open
