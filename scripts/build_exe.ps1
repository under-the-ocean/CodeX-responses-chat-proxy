$ErrorActionPreference = "Stop"

python -m pip install -e ".[build]"

python -m PyInstaller `
  --noconfirm `
  --onefile `
  --console `
  --name responses-chat-proxy `
  --specpath build `
  --paths src `
  --hidden-import responses_chat_proxy.main `
  --exclude-module IPython `
  --exclude-module matplotlib `
  --exclude-module numpy `
  --exclude-module pandas `
  --exclude-module pygame `
  --exclude-module PyQt5 `
  --exclude-module PySide6 `
  --exclude-module pytest `
  scripts\launcher_entry.py
