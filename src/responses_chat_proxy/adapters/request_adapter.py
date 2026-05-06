from typing import Any


def convert_request(data: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Responses API request body to Chat Completions."""
    chat_data: dict[str, Any] = {"model": data.get("model")}

    messages: list[dict[str, Any]] = []
    instructions = data.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    input_data = data.get("input")
    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, dict):
                message = _convert_input_item(item)
                if message is not None:
                    messages.append(message)

    chat_data["messages"] = messages

    pass_through_keys = [
        "stream",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "seed",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "n",
        "logit_bias",
        "user",
    ]
    for key in pass_through_keys:
        if key in data:
            chat_data[key] = data[key]

    if "max_output_tokens" in data:
        chat_data["max_tokens"] = data["max_output_tokens"]

    if "tools" in data and isinstance(data["tools"], list):
        chat_data["tools"] = _convert_tools(data["tools"])

    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        chat_data["reasoning_effort"] = reasoning["effort"]

    return chat_data


def _convert_input_item(item: dict[str, Any]) -> dict[str, Any] | None:
    item_type = item.get("type")
    has_role = "role" in item
    is_message = item_type == "message" or (item_type is None and has_role)

    if is_message:
        role = item.get("role", "user")
        if role == "developer":
            role = "system"

        content = item.get("content")
        if isinstance(content, list):
            converted_content: list[dict[str, Any]] = []
            text_only = True
            for part in content:
                if not isinstance(part, dict):
                    text_only = False
                    continue
                converted = _convert_content_part(part)
                if converted is None:
                    text_only = False
                    continue
                converted_content.append(converted)
                if converted.get("type") != "text":
                    text_only = False

            if text_only:
                content = "".join(part.get("text", "") for part in converted_content)
            else:
                content = converted_content

        return {"role": role, "content": content}

    if item_type == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": item.get("call_id"),
            "content": item.get("output", ""),
        }

    return None


def _convert_content_part(part: dict[str, Any]) -> dict[str, Any] | None:
    part_type = part.get("type")
    if part_type in {"input_text", "output_text"}:
        return {"type": "text", "text": part.get("text", "")}
    if part_type == "input_image":
        image_url = part.get("image_url", "")
        if isinstance(image_url, dict):
            image_url = image_url.get("url", "")
        return {"type": "image_url", "image_url": {"url": image_url}}
    return part


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if "function" in tool:
            converted.append(tool)
            continue
        converted.append(
            {
                "type": tool.get("type", "function"),
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
        )
    return converted
