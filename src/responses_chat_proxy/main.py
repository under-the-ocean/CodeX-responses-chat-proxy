import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import StreamingConverter, convert_request, convert_response
from .config import settings
from .errors import error_payload, sse_error, wrap_upstream_error
from .upstream import build_chat_completions_url, build_upstream_headers, normal_timeout, stream_timeout

logger = logging.getLogger("responses_chat_proxy")
TOOL_CALL_REASONING_CACHE: dict[str, str] = {}

app = FastAPI(
    title="Responses Chat Proxy",
    description="Convert Responses API requests to Chat Completions upstreams and convert replies back.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/responses", response_model=None)
async def create_response(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | StreamingResponse:
    logger.info("=" * 60)
    logger.info("Received POST /v1/responses request")
    logger.debug(f"Request headers: {dict(request.headers)}")
    
    _authorize(authorization)

    try:
        original_request = await request.json()
    except Exception:
        logger.error("Failed to parse request body as JSON")
        return JSONResponse(
            error_payload("Request body must be valid JSON.", "invalid_request_error", "invalid_json"),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if not isinstance(original_request, dict):
        logger.error("Request body is not a JSON object")
        return JSONResponse(
            error_payload("Request body must be a JSON object.", "invalid_request_error", "invalid_request"),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if not original_request.get("model"):
        logger.error("Missing required parameter: model")
        return JSONResponse(
            error_payload(
                "Missing required parameter: model",
                "invalid_request_error",
                "missing_required_parameter",
                param="model",
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if settings.log_requests:
        logger.info(f"Request model: {original_request.get('model')}")
        logger.debug(f"Full request: {json.dumps(original_request, ensure_ascii=False, indent=2)}")

    try:
        _log_original_codex_app_tools(original_request)
        chat_request = convert_request(original_request)
        _inject_cached_reasoning_content(chat_request)
        logger.info(f"Request converted successfully")
        _log_codex_app_tool_schema(chat_request)
        logger.debug(f"Converted Chat request: {json.dumps(chat_request, ensure_ascii=False, indent=2)}")
    except Exception as exc:
        logger.error(f"Request conversion failed: {exc}")
        return JSONResponse(
            error_payload(f"Request conversion error: {exc}", "invalid_request_error", "conversion_error"),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if bool(original_request.get("stream")):
        logger.info("Streaming request detected")
        return StreamingResponse(
            _stream_upstream(request, original_request, chat_request),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    logger.info("Non-streaming request")
    return await _normal_upstream(request, original_request, chat_request)


async def _normal_upstream(
    request: Request,
    original_request: dict[str, Any],
    chat_request: dict[str, Any],
) -> JSONResponse:
    url = build_chat_completions_url()
    headers = build_upstream_headers(dict(request.headers))
    model = chat_request.get("model", "unknown")
    
    logger.info(f"→ Sending non-streaming request to upstream: {url}")
    logger.debug(f"Upstream request model: {model}")

    try:
        async with httpx.AsyncClient(timeout=normal_timeout(), verify=settings.verify_ssl) as client:
            response = await client.post(url, headers=headers, json=chat_request)
    except httpx.TimeoutException:
        logger.error(f"Request timeout to upstream: {url}")
        return JSONResponse(
            error_payload("Request timeout. Please try again.", "timeout_error", "request_timeout"),
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        logger.error(f"Upstream request failed: {exc}")
        return JSONResponse(
            error_payload(f"Upstream request failed: {exc}", "upstream_error", "upstream_request_failed"),
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    logger.info(f"← Upstream response received: status={response.status_code}")
    
    response_data = _parse_response_json(response)
    if response.status_code >= 400:
        logger.error(f"Upstream returned error status: {response.status_code}")
        logger.error(_summarize_chat_request(chat_request))
        logger.error(f"Upstream error payload: {json.dumps(response_data, ensure_ascii=False)}")
        return JSONResponse(wrap_upstream_error(response_data, response.status_code), status_code=response.status_code)

    try:
        converted = convert_response(response_data, original_request)
        _cache_reasoning_from_chat_response(response_data)
        logger.info("Response converted successfully")
        
        if settings.log_responses:
            logger.debug(f"Converted response: {json.dumps(converted, ensure_ascii=False, indent=2)}")
    except Exception as exc:
        logger.error(f"Response conversion failed: {exc}")
        return JSONResponse(
            error_payload(f"Response conversion error: {exc}", "server_error", "conversion_error"),
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    logger.info("=" * 60)
    return JSONResponse(converted, status_code=response.status_code)


async def _stream_upstream(
    request: Request,
    original_request: dict[str, Any],
    chat_request: dict[str, Any],
) -> AsyncIterator[bytes]:
    url = build_chat_completions_url()
    headers = build_upstream_headers(dict(request.headers))
    headers["Accept"] = "text/event-stream"
    model = chat_request.get("model", "unknown")

    body = dict(chat_request)
    body["stream"] = True

    logger.info(f"→ Sending streaming request to upstream: {url}")
    logger.debug(f"Upstream request model: {model}")

    converter = StreamingConverter()
    chunk_count = 0

    try:
        async with httpx.AsyncClient(timeout=stream_timeout(), verify=settings.verify_ssl) as client:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code >= 400:
                    error_data = await _read_stream_error(response)
                    message = _extract_error_message(error_data)
                    logger.error(f"Upstream streaming error: status={response.status_code}, message={message}")
                    logger.error(_summarize_chat_request(body))
                    logger.error(f"Upstream streaming error payload: {json.dumps(error_data, ensure_ascii=False)}")
                    yield sse_error(message, f"http_{response.status_code}")
                    return

                logger.info("← Streaming started, receiving chunks from upstream")
                
                async for chunk in response.aiter_raw():
                    if not chunk:
                        continue
                    chunk_count += 1
                    if chunk_count % 100 == 0:
                        logger.debug(f"Streaming progress: {chunk_count} chunks received")
                    
                    for event in converter.feed(chunk):
                        yield event.encode("utf-8")

                logger.info(f"← Streaming completed: {chunk_count} chunks received")
                
                for event in converter.finish():
                    yield event.encode("utf-8")
                _cache_reasoning_from_streaming_converter(converter)
                    
    except httpx.TimeoutException:
        logger.error("Streaming request timeout")
        yield sse_error("Request timeout.", "request_timeout", "timeout_error")
    except httpx.HTTPError as exc:
        logger.error(f"Streaming upstream request failed: {exc}")
        yield sse_error(f"Upstream request failed: {exc}", "upstream_request_failed")
    except Exception as exc:
        logger.error(f"Streaming error: {exc}")
        yield sse_error(str(exc), "internal_error", "server_error")
    finally:
        logger.info("=" * 60)


def _authorize(authorization: str | None) -> None:
    if not settings.proxy_api_key:
        logger.debug("Proxy authentication disabled")
        return
    expected = f"Bearer {settings.proxy_api_key}"
    if authorization != expected:
        logger.warning("Authentication failed: invalid API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_payload("Incorrect API key provided.", "authentication_error", "invalid_api_key")["error"],
        )
    logger.debug("Authentication successful")


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


def _inject_cached_reasoning_content(chat_request: dict[str, Any]) -> None:
    messages = chat_request.get("messages")
    if not isinstance(messages, list):
        return

    _backfill_reasoning_from_request_messages(messages)

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        if message.get("reasoning_content"):
            logger.info(
                "Reasoning injection skipped for assistant message #%s: already has reasoning_content",
                index,
            )
            continue
        matched_reasoning: str | None = None
        matched_ids: list[str] = []
        missing_ids: list[str] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            cached_reasoning = TOOL_CALL_REASONING_CACHE.get(tool_call_id)
            if isinstance(cached_reasoning, str) and cached_reasoning:
                matched_ids.append(tool_call_id)
                if matched_reasoning is None:
                    matched_reasoning = cached_reasoning
            else:
                missing_ids.append(tool_call_id)
        if matched_reasoning is not None:
            message["reasoning_content"] = matched_reasoning
            logger.info(
                "Reasoning injection matched for assistant message #%s: matched_tool_call_ids=%s missing_tool_call_ids=%s reasoning_len=%s",
                index,
                matched_ids,
                missing_ids,
                len(matched_reasoning),
            )
        else:
            logger.warning(
                "Reasoning injection missed for assistant message #%s: missing_tool_call_ids=%s cache_size=%s",
                index,
                missing_ids,
                len(TOOL_CALL_REASONING_CACHE),
            )


def _backfill_reasoning_from_request_messages(messages: list[Any]) -> None:
    reasoning_by_tool_call_id: dict[str, str] = {}

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        reasoning_content = message.get("reasoning_content")
        if not isinstance(reasoning_content, str) or not reasoning_content:
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id")
            if isinstance(tool_call_id, str) and tool_call_id:
                reasoning_by_tool_call_id[tool_call_id] = reasoning_content

    if not reasoning_by_tool_call_id:
        return

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        if message.get("reasoning_content"):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue

        matched_ids: list[str] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                continue
            if tool_call_id in reasoning_by_tool_call_id:
                matched_ids.append(tool_call_id)

        if not matched_ids:
            continue

        restored_reasoning = reasoning_by_tool_call_id[matched_ids[0]]
        message["reasoning_content"] = restored_reasoning
        logger.info(
            "Reasoning request backfill matched for assistant message #%s: matched_tool_call_ids=%s reasoning_len=%s",
            index,
            matched_ids,
            len(restored_reasoning),
        )


def _cache_reasoning_from_chat_response(chat_response: dict[str, Any]) -> None:
    choices = chat_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    if not isinstance(choice, dict):
        return
    message = choice.get("message")
    if not isinstance(message, dict):
        return
    reasoning_content = message.get("reasoning_content")
    tool_calls = message.get("tool_calls")
    _store_reasoning_by_tool_call_ids(reasoning_content, tool_calls, source="chat_response")


def _cache_reasoning_from_streaming_converter(converter: StreamingConverter) -> None:
    reasoning_content = getattr(converter, "reasoning_content", "")
    tool_calls = []
    for _, tool_call in sorted(getattr(converter, "tool_calls", {}).items()):
        if isinstance(tool_call, dict) and tool_call.get("id"):
            tool_calls.append({"id": tool_call["id"]})
    _store_reasoning_by_tool_call_ids(reasoning_content, tool_calls, source="streaming_response")


def _store_reasoning_by_tool_call_ids(reasoning_content: Any, tool_calls: Any, source: str) -> None:
    if not isinstance(reasoning_content, str) or not reasoning_content:
        logger.info("Reasoning cache skipped for %s: empty reasoning_content", source)
        return
    if not isinstance(tool_calls, list):
        logger.info("Reasoning cache skipped for %s: tool_calls is not a list", source)
        return
    stored_ids: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_call_id = tool_call.get("id")
        if isinstance(tool_call_id, str) and tool_call_id:
            TOOL_CALL_REASONING_CACHE[tool_call_id] = reasoning_content
            stored_ids.append(tool_call_id)
    if stored_ids:
        logger.info(
            "Reasoning cache write from %s: tool_call_ids=%s reasoning_len=%s",
            source,
            stored_ids,
            len(reasoning_content),
        )
    else:
        logger.info(
            "Reasoning cache skipped for %s: no valid tool_call_ids reasoning_len=%s",
            source,
            len(reasoning_content),
        )


def _summarize_chat_request(chat_request: dict[str, Any]) -> str:
    messages = chat_request.get("messages")
    if not isinstance(messages, list):
        return "Converted request summary: messages=<invalid>"

    parts: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            parts.append(f"#{index}:<invalid>")
            continue

        role = message.get("role", "unknown")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        reasoning_content = message.get("reasoning_content")

        content_len = len(content) if isinstance(content, str) else 0
        reasoning_len = len(reasoning_content) if isinstance(reasoning_content, str) else 0
        tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
        tool_call_ids: list[str] = []
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tool_call_id = tool_call.get("id")
                    if isinstance(tool_call_id, str) and tool_call_id:
                        tool_call_ids.append(tool_call_id)

        tool_call_id = message.get("tool_call_id")
        tool_call_id_str = tool_call_id if isinstance(tool_call_id, str) else ""

        parts.append(
            f"#{index}:role={role},content_len={content_len},reasoning_len={reasoning_len},"
            f"tool_calls={tool_call_count},tool_call_ids={tool_call_ids},tool_call_id={tool_call_id_str}"
        )

    return "Converted request summary: " + " | ".join(parts)


def _log_codex_app_tool_schema(chat_request: dict[str, Any]) -> None:
    tools = chat_request.get("tools")
    if not isinstance(tools, list):
        return

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function_data = tool.get("function")
        if not isinstance(function_data, dict):
            continue
        if function_data.get("name") != "codex_app":
            continue
        logger.warning(
            "Final codex_app schema: %s",
            json.dumps(function_data, ensure_ascii=False),
        )


def _log_original_codex_app_tools(original_request: dict[str, Any]) -> None:
    tools = original_request.get("tools")
    if not isinstance(tools, list):
        return

    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue

        candidate_names: list[str] = []
        tool_name = tool.get("name")
        if isinstance(tool_name, str) and tool_name:
            candidate_names.append(tool_name)

        for key, value in tool.items():
            if isinstance(value, dict):
                nested_name = value.get("name")
                if isinstance(nested_name, str) and nested_name:
                    candidate_names.append(nested_name)
                candidate_names.append(key)

        if "codex_app" not in candidate_names:
            continue

        nested_shapes: dict[str, Any] = {}
        for key, value in tool.items():
            if key in {"type", "name", "description", "parameters", "function"}:
                continue
            if isinstance(value, dict):
                nested_shapes[key] = {
                    "keys": sorted(value.keys()),
                    "subcommand_keys": sorted(
                        sub_key
                        for sub_key, sub_value in value.items()
                        if isinstance(sub_value, dict)
                    ),
                }
            else:
                nested_shapes[key] = type(value).__name__

        logger.warning(
            "Original codex_app tool[%s]: type=%s name=%s parameter_keys=%s top_level_keys=%s nested_shapes=%s raw=%s",
            index,
            tool.get("type"),
            tool.get("name"),
            sorted(tool.get("parameters", {}).keys()) if isinstance(tool.get("parameters"), dict) else [],
            sorted(tool.keys()),
            json.dumps(nested_shapes, ensure_ascii=False),
            json.dumps(tool, ensure_ascii=False),
        )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions_passthrough(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | StreamingResponse:
    """Optional convenience endpoint: forward Chat Completions unchanged."""
    logger.info("=" * 60)
    logger.info("Received POST /v1/chat/completions request (passthrough)")
    
    _authorize(authorization)
    try:
        body = await request.json()
        logger.info(f"Passthrough request model: {body.get('model', 'unknown')}")
    except Exception:
        logger.error("Failed to parse passthrough request body as JSON")
        return JSONResponse(
            error_payload("Request body must be valid JSON.", "invalid_request_error", "invalid_json"),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if isinstance(body, dict) and body.get("stream"):
        logger.info("Passthrough streaming request")
        return StreamingResponse(
            _passthrough_stream(request, body),
            media_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )
    
    logger.info("Passthrough non-streaming request")
    return await _passthrough_normal(request, body)


async def _passthrough_normal(request: Request, body: Any) -> JSONResponse:
    url = build_chat_completions_url()
    model = body.get("model", "unknown") if isinstance(body, dict) else "unknown"
    
    logger.info(f"→ Forwarding passthrough request to upstream: {url}")
    logger.debug(f"Passthrough request model: {model}")

    try:
        async with httpx.AsyncClient(timeout=normal_timeout(), verify=settings.verify_ssl) as client:
            response = await client.post(
                build_chat_completions_url(),
                headers=build_upstream_headers(dict(request.headers)),
                json=body,
            )
    except httpx.TimeoutException:
        logger.error("Passthrough request timeout")
        return JSONResponse(
            error_payload("Request timeout. Please try again.", "timeout_error", "request_timeout"),
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        logger.error(f"Passthrough upstream request failed: {exc}")
        return JSONResponse(
            error_payload(f"Upstream request failed: {exc}", "upstream_error", "upstream_request_failed"),
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
    
    logger.info(f"← Passthrough response: status={response.status_code}")
    logger.info("=" * 60)
    return JSONResponse(_parse_response_json(response), status_code=response.status_code)


async def _passthrough_stream(request: Request, body: Any) -> AsyncIterator[bytes]:
    logger.info(f"→ Forwarding passthrough streaming request to upstream")
    
    try:
        async with httpx.AsyncClient(timeout=stream_timeout(), verify=settings.verify_ssl) as client:
            async with client.stream(
                "POST",
                build_chat_completions_url(),
                headers=build_upstream_headers(dict(request.headers)),
                json=body,
            ) as response:
                logger.info(f"← Passthrough streaming started: status={response.status_code}")
                async for chunk in response.aiter_raw():
                    yield chunk
                logger.info("← Passthrough streaming completed")
    except httpx.TimeoutException:
        logger.error("Passthrough streaming timeout")
        yield b'data: {"error":{"message":"Request timeout.","type":"timeout_error","code":"request_timeout"}}\n\n'
    except httpx.HTTPError as exc:
        logger.error(f"Passthrough streaming upstream request failed: {exc}")
        yield f'data: {{"error":{{"message":"Upstream request failed: {exc}","type":"upstream_error","code":"upstream_request_failed"}}}}\n\n'.encode(
            "utf-8"
        )
    finally:
        logger.info("=" * 60)
