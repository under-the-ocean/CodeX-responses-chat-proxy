from urllib.parse import urljoin

import httpx

from .config import settings


def build_chat_completions_url() -> str:
    base = settings.upstream_base_url.rstrip("/") + "/"
    return urljoin(base, "chat/completions")


def build_upstream_headers(incoming_headers: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if incoming_headers:
        for key, value in incoming_headers.items():
            lower = key.lower()
            if lower in {"host", "content-length", "authorization", "accept-encoding", "connection"}:
                continue
            headers[key] = value

    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
    return headers


def normal_timeout() -> httpx.Timeout:
    return httpx.Timeout(settings.request_timeout_seconds, connect=30.0)


def stream_timeout() -> httpx.Timeout:
    return httpx.Timeout(settings.stream_timeout_seconds, connect=30.0, read=settings.stream_timeout_seconds)
