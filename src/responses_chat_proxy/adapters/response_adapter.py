from typing import Any


def convert_response(chat_response: dict[str, Any], original_request: dict[str, Any]) -> dict[str, Any]:
    """Convert a non-streaming Chat Completions response to Responses API."""
    chat_id = chat_response.get("id", "")
    response_id = _convert_id(chat_id)
    created = chat_response.get("created", 0)
    model = chat_response.get("model", original_request.get("model", ""))

    choices = chat_response.get("choices", [])
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None

    output = []
    reasoning_item = _build_reasoning_item(message, response_id)
    if reasoning_item:
        output.append(reasoning_item)
    if isinstance(message, dict) and message:
        output_item = _build_output_item(message, response_id, finish_reason)
        if output_item:
            output.append(output_item)

    status, incomplete_details = _map_finish_reason(finish_reason)

    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": None,
        "incomplete_details": incomplete_details,
        "instructions": original_request.get("instructions"),
        "max_output_tokens": original_request.get("max_output_tokens"),
        "model": model,
        "output": output,
        "parallel_tool_calls": original_request.get("parallel_tool_calls", True),
        "temperature": original_request.get("temperature", 1.0),
        "tool_choice": original_request.get("tool_choice", "auto"),
        "tools": original_request.get("tools", []),
        "top_p": original_request.get("top_p", 1.0),
        "truncation": "disabled",
        "usage": _convert_usage(chat_response.get("usage", {}) or {}),
        "user": original_request.get("user"),
        "metadata": original_request.get("metadata", {}),
    }


def _convert_id(chat_id: str) -> str:
    if chat_id.startswith("chatcmpl-"):
        return chat_id.replace("chatcmpl-", "resp-", 1)
    if chat_id.startswith("chatcmpl"):
        return chat_id.replace("chatcmpl", "resp", 1)
    return f"resp_{chat_id}" if chat_id else "resp_unknown"


def _build_output_item(
    message: dict[str, Any],
    response_id: str,
    finish_reason: str | None,
) -> dict[str, Any] | None:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    tool_calls = message.get("tool_calls")
    item_id = f"msg_{response_id.replace('resp-', '', 1).replace('resp_', '')}"

    output_item: dict[str, Any] = {
        "type": "message",
        "id": item_id,
        "status": "completed" if finish_reason else "in_progress",
        "role": role,
        "content": [],
    }
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        output_item["reasoning_content"] = reasoning_content

    if content:
        if isinstance(content, str):
            output_item["content"].append({"type": "output_text", "text": content, "annotations": []})
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    converted = _convert_output_part(part)
                    if converted:
                        output_item["content"].append(converted)

    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                converted = _convert_tool_call(tool_call)
                if converted:
                    output_item["content"].append(converted)

    return output_item


def _build_reasoning_item(message: dict[str, Any], response_id: str) -> dict[str, Any] | None:
    reasoning_content = message.get("reasoning_content")
    if not isinstance(reasoning_content, str) or not reasoning_content:
        return None

    item_id = f"rs_{response_id.replace('resp-', '', 1).replace('resp_', '')}"
    return {
        "id": item_id,
        "type": "reasoning",
        "content": [],
        "summary": [{"type": "summary_text", "text": reasoning_content}],
    }


def _convert_output_part(part: dict[str, Any]) -> dict[str, Any] | None:
    if part.get("type") == "text":
        return {"type": "output_text", "text": part.get("text", ""), "annotations": []}
    return part


def _convert_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    return {
        "type": "tool_call",
        "id": tool_call.get("id"),
        "call_type": tool_call.get("type", "function"),
        "status": "completed",
        "name": function.get("name", ""),
        "arguments": function.get("arguments", "{}"),
    }


def _map_finish_reason(finish_reason: str | None) -> tuple[str, dict[str, str] | None]:
    if finish_reason in {"stop", "tool_calls"}:
        return "completed", None
    if finish_reason == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    if finish_reason == "content_filter":
        return "incomplete", {"reason": "content_filter"}
    if finish_reason is None:
        return "in_progress", None
    return "completed", None


def _convert_usage(usage: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)

    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = 0
    if isinstance(prompt_details, dict):
        cached_tokens = int(prompt_details.get("cached_tokens") or 0)
    cached_tokens = cached_tokens or int(usage.get("cache_read_input_tokens") or 0)

    completion_details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = 0
    if isinstance(completion_details, dict):
        reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)

    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }
