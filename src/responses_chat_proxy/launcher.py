from __future__ import annotations

import json
import os
import socket
from pathlib import Path

APP_DIR_NAME = ".responses-chat-proxy"
CONFIG_FILE_NAME = "config.json"

DEFAULT_UPSTREAM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
PORT_SEARCH_LIMIT = 100
DEFAULT_LOG_LEVEL = "info"


def main() -> None:
    import uvicorn

    config_path = get_config_path()
    config = load_config(config_path)

    if should_reconfigure(config):
        config = prompt_for_config(config)
        save_config(config_path, config)
        print(f"Saved configuration to {config_path}")

    port = find_available_port(DEFAULT_HOST, DEFAULT_PORT)
    apply_runtime_defaults(config, port=port)
    print_startup_message(port)

    uvicorn.run(
        "responses_chat_proxy.main:app",
        host=DEFAULT_HOST,
        port=port,
        log_level=DEFAULT_LOG_LEVEL,
    )


def get_config_path() -> Path:
    return Path.home() / APP_DIR_NAME / CONFIG_FILE_NAME


def load_config(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        return {}

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read saved configuration: {exc}")
        return {}

    if not isinstance(raw_config, dict):
        print("Saved configuration is invalid and will be recreated.")
        return {}

    config: dict[str, str] = {}
    for key in ("upstream_base_url", "upstream_api_key"):
        value = raw_config.get(key)
        if isinstance(value, str):
            config[key] = value
    return config


def should_reconfigure(config: dict[str, str]) -> bool:
    if not is_complete_config(config):
        return True

    print("Loaded saved upstream configuration:")
    print(f"  base_url: {config['upstream_base_url']}")
    print(f"  api key : {mask_api_key(config['upstream_api_key'])}")
    try:
        answer = input("Press Enter to start, or type r to reconfigure: ").strip().lower()
    except EOFError:
        return False
    return answer in {"r", "reconfigure", "y", "yes"}


def is_complete_config(config: dict[str, str]) -> bool:
    return bool(config.get("upstream_base_url", "").strip()) and bool(
        config.get("upstream_api_key", "").strip()
    )


def prompt_for_config(existing_config: dict[str, str] | None = None) -> dict[str, str]:
    existing_config = existing_config or {}
    default_base_url = existing_config.get("upstream_base_url") or DEFAULT_UPSTREAM_BASE_URL

    upstream_base_url = prompt_required(
        f"Upstream base_url [{default_base_url}]: ",
        default=default_base_url,
    )
    upstream_api_key = prompt_secret_required("Upstream api key: ")

    return {
        "upstream_base_url": upstream_base_url.rstrip("/"),
        "upstream_api_key": upstream_api_key,
    }


def prompt_required(prompt: str, *, default: str | None = None) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        if default:
            return default
        print("This value is required.")


def prompt_secret_required(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("This value is required.")


def save_config(config_path: Path, config: dict[str, str]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "upstream_base_url": config["upstream_base_url"],
        "upstream_api_key": config["upstream_api_key"],
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_runtime_defaults(config: dict[str, str], *, port: int = DEFAULT_PORT) -> None:
    os.environ["UPSTREAM_BASE_URL"] = config["upstream_base_url"]
    os.environ["UPSTREAM_API_KEY"] = config["upstream_api_key"]
    os.environ["PROXY_API_KEY"] = ""
    os.environ["HOST"] = DEFAULT_HOST
    os.environ["PORT"] = str(port)
    os.environ.setdefault("LOG_LEVEL", DEFAULT_LOG_LEVEL)


def print_startup_message(port: int = DEFAULT_PORT) -> None:
    print()
    print("Responses Chat Proxy is starting.")
    print(f"Local Responses API base_url: http://{DEFAULT_HOST}:{port}/v1")
    print(f"Responses endpoint: http://{DEFAULT_HOST}:{port}/v1/responses")
    print("Proxy authentication is disabled for local clients.")
    print("Press Ctrl+C to stop the service.")
    print()


def find_available_port(host: str = DEFAULT_HOST, start_port: int = DEFAULT_PORT) -> int:
    for port in range(start_port, start_port + PORT_SEARCH_LIMIT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                continue

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"No available local port found from {start_port} to {start_port + PORT_SEARCH_LIMIT - 1}."
    )


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"
