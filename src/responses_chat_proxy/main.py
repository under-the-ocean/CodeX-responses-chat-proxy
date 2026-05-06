from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import StreamingConverter, convert_request, convert_response
from .config import settings
from .errors import error_payload, sse_error, wrap_upstream_error
from .upstream import build_chat_completions_url, build_upstream_headers, normal_timeout, stream_timeout

app = FastAPI(
    title="Responses Chat Proxy",
    description="Convert Responses API requests to Chat Completions upstreams and convert replies back.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/responses")
async def create_response(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | StreamingResponse:
    _authorize(authorization)

    try:
        original_request = await request.json()
    except Exception:
        return JSONResponse(
            error_payload("Request body must be valid JSON.", "invalid_request_error", "invalid_json"),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if not isinstance(original_request, dict):
        return JSONResponse(
            error_payload("Request body must be a JSON object.", "invalid_request_error", "invalid_request"),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if not original_request.get("model"):
        return JSONResponse(
            error_payload(
                "Missing required parameter: model",
                "invalid_request_error",
                "missing_required_parameter",
                param="model",
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        chat_request = convert_request(original_request)
    except Exception as exc:
        return JSONResponse(
            error_payload(f"Request conversion error: {exc}", "invalid_request_error", "conversion_error"),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if bool(original_request.get("stream")):
        return StreamingResponse(
            _stream_upstream(request, original_request, chat_request),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    return await _normal_upstream(request, original_request, chat_request)


async def _normal_upstream(
    request: Request,
    original_request: dict[str, Any],
    chat_request: dict[str, Any],
) -> JSONResponse:
    url = build_chat_completions_url()
    headers = build_upstream_headers(dict(request.headers))

    try:
        async with httpx.AsyncClient(timeout=normal_timeout(), verify=settings.verify_ssl) as client:
            response = await client.post(url, headers=headers, json=chat_request)
    except httpx.TimeoutException:
        return JSONResponse(
            error_payload("Request timeout. Please try again.", "timeout_error", "request_timeout"),
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            error_payload(f"Upstream request failed: {exc}", "upstream_error", "upstream_request_failed"),
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    response_data = _parse_response_json(response)
    if response.status_code >= 400:
        return JSONResponse(wrap_upstream_error(response_data, response.status_code), status_code=response.status_code)

    try:
        converted = convert_response(response_data, original_request)
    except Exception as exc:
        return JSONResponse(
            error_payload(f"Response conversion error: {exc}", "server_error", "conversion_error"),
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    return JSONResponse(converted, status_code=response.status_code)


async def _stream_upstream(
    request: Request,
    original_request: dict[str, Any],
    chat_request: dict[str, Any],
) -> AsyncIterator[bytes]:
    url = build_chat_completions_url()
    headers = build_upstream_headers(dict(request.headers))
    headers["Accept"] = "text/event-stream"

    body = dict(chat_request)
    body["stream"] = True

    converter = StreamingConverter()

    try:
        async with httpx.AsyncClient(timeout=stream_timeout(), verify=settings.verify_ssl) as client:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code >= 400:
                    error_data = await _read_stream_error(response)
                    message = _extract_error_message(error_data)
                    yield sse_error(message, f"http_{response.status_code}")
                    return

                async for chunk in response.aiter_raw():
                    if not chunk:
                        continue
                    for event in converter.feed(chunk):
                        yield event.encode("utf-8")

                for event in converter.finish():
                    yield event.encode("utf-8")
    except httpx.TimeoutException:
        yield sse_error("Request timeout.", "request_timeout", "timeout_error")
    except httpx.HTTPError as exc:
        yield sse_error(f"Upstream request failed: {exc}", "upstream_request_failed")
    except Exception as exc:
        yield sse_error(str(exc), "internal_error", "server_error")


def _authorize(authorization: str | None) -> None:
    if not settings.proxy_api_key:
        return
    expected = f"Bearer {settings.proxy_api_key}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_payload("Incorrect API key provided.", "authentication_error", "invalid_api_key")["error"],
        )


def _parse_response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except Exception:
        data = {"raw_response": response.text}
    if isinstance(data, dict):
        return data
    return {"raw_response": data}


async def _read_stream_error(response: httpx.Response) -> dict[str, Any]:
    raw = await response.aread()
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return {"raw": "<empty response body>"}
    try:
        data = response.json()
    except Exception:
        try:
            import json

            data = json.loads(text)
        except Exception:
            data = {"raw": text}
    return data if isinstance(data, dict) else {"raw": data}


def _extract_error_message(error_data: dict[str, Any]) -> str:
    error = error_data.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    if error:
        return str(error)
    if error_data.get("message"):
        return str(error_data["message"])
    if error_data.get("detail"):
        return str(error_data["detail"])
    if error_data.get("raw"):
        return str(error_data["raw"])[:500]
    return "Upstream error"


@app.post("/v1/chat/completions")
async def chat_completions_passthrough(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | StreamingResponse:
    """Optional convenience endpoint: forward Chat Completions unchanged."""
    _authorize(authorization)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            error_payload("Request body must be valid JSON.", "invalid_request_error", "invalid_json"),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if isinstance(body, dict) and body.get("stream"):
        return StreamingResponse(
            _passthrough_stream(request, body),
            media_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )
    return await _passthrough_normal(request, body)


async def _passthrough_normal(request: Request, body: Any) -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=normal_timeout(), verify=settings.verify_ssl) as client:
            response = await client.post(
                build_chat_completions_url(),
                headers=build_upstream_headers(dict(request.headers)),
                json=body,
            )
    except httpx.TimeoutException:
        return JSONResponse(
            error_payload("Request timeout. Please try again.", "timeout_error", "request_timeout"),
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            error_payload(f"Upstream request failed: {exc}", "upstream_error", "upstream_request_failed"),
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
    return JSONResponse(_parse_response_json(response), status_code=response.status_code)


async def _passthrough_stream(request: Request, body: Any) -> AsyncIterator[bytes]:
    try:
        async with httpx.AsyncClient(timeout=stream_timeout(), verify=settings.verify_ssl) as client:
            async with client.stream(
                "POST",
                build_chat_completions_url(),
                headers=build_upstream_headers(dict(request.headers)),
                json=body,
            ) as response:
                async for chunk in response.aiter_raw():
                    yield chunk
    except httpx.TimeoutException:
        yield b'data: {"error":{"message":"Request timeout.","type":"timeout_error","code":"request_timeout"}}\n\n'
    except httpx.HTTPError as exc:
        yield f'data: {{"error":{{"message":"Upstream request failed: {exc}","type":"upstream_error","code":"upstream_request_failed"}}}}\n\n'.encode(
            "utf-8"
        )
