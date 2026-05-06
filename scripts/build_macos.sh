#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" -m pip install -e ".[build]"

"$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --onefile \
  --console \
  --name responses-chat-proxy \
  --specpath build/macos \
  --workpath build/macos/work \
  --distpath dist/macos \
  --paths src \
  --hidden-import responses_chat_proxy.main \
  --exclude-module IPython \
  --exclude-module matplotlib \
  --exclude-module numpy \
  --exclude-module pandas \
  --exclude-module pygame \
  --exclude-module PyQt5 \
  --exclude-module PySide6 \
  --exclude-module pytest \
  scripts/launcher_entry.py
