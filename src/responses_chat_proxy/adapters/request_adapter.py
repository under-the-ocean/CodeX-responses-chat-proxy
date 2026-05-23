import logging
from typing import Any

from ..config import settings

logger = logging.getLogger("responses_chat_proxy")


def convert_request(data: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Responses API request body to Chat Completions."""
    original_model = data.get("model", "")
    upstream_model = settings.get_upstream_model(original_model)
    
    chat_data: dict[str, Any] = {"model": upstream_model}

    messages: list[dict[str, Any]] = []
    instructions = data.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    input_data = data.get("input")
    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        messages.extend(_convert_input_items(input_data))

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

    if original_model != upstream_model:
        logger.info(f"Model mapping applied: {original_model} -> {upstream_model}")
    
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
        tool_calls: list[dict[str, Any]] = []
        if isinstance(content, list):
            converted_content: list[dict[str, Any]] = []
            text_only = True
            for part in content:
                if not isinstance(part, dict):
                    text_only = False
                    continue
                converted_tool_call = _convert_tool_call_part(part)
                if converted_tool_call is not None:
                    tool_calls.append(converted_tool_call)
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

        message: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        reasoning_content = item.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            message["reasoning_content"] = reasoning_content
        return message

    if item_type == "function_call_output":
        call_id = item.get("call_id")
        if not call_id or not isinstance(call_id, str) or len(call_id.strip()) == 0:
            logger.warning(f"Skipping function_call_output without call_id")
            return None
        
        output = item.get("output", "")
        if not output:
            output = ""
        elif not isinstance(output, str):
            try:
                output = str(output)
            except Exception:
                output = ""
        
        return {
            "role": "tool",
            "tool_call_id": call_id.strip(),
            "content": output,
        }

    return None


def _convert_input_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_tool_calls: dict[str, dict[str, Any]] = {}
    buffered_tool_calls: list[dict[str, Any]] = []
    pending_reasoning_content = ""

    def flush_buffered_tool_calls() -> None:
        nonlocal pending_reasoning_content
        if not buffered_tool_calls:
            return
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": "",
            "tool_calls": list(buffered_tool_calls),
        }
        if pending_reasoning_content:
            assistant_message["reasoning_content"] = pending_reasoning_content
        messages.append(assistant_message)
        buffered_tool_calls.clear()
        pending_reasoning_content = ""

    for item in items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "reasoning":
            summary = item.get("summary")
            reasoning_text = _extract_reasoning_text(summary)
            if reasoning_text:
                pending_reasoning_content = reasoning_text
            continue

        if item_type == "function_call":
            tool_call = _convert_function_call_item(item)
            if tool_call is None:
                continue
            call_id = tool_call["id"]
            pending_tool_calls[call_id] = tool_call
            buffered_tool_calls.append(tool_call)
            continue

        if item_type == "function_call_output":
            flush_buffered_tool_calls()
            message = _convert_function_call_output_item(item, pending_tool_calls)
            if message is not None:
                messages.append(message)
            continue

        flush_buffered_tool_calls()
        message = _convert_input_item(item)
        if message is not None:
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    call_id = tool_call.get("id")
                    if isinstance(call_id, str) and call_id:
                        pending_tool_calls[call_id] = tool_call
            if message.get("role") == "assistant" and pending_reasoning_content and "reasoning_content" not in message:
                message["reasoning_content"] = pending_reasoning_content
            if message.get("role") == "assistant":
                pending_reasoning_content = ""
            messages.append(message)

    flush_buffered_tool_calls()
    return messages


def _convert_function_call_item(item: dict[str, Any]) -> dict[str, Any] | None:
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments", "{}")

    if not isinstance(call_id, str) or len(call_id.strip()) == 0:
        logger.warning("Skipping function_call without call_id")
        return None

    if not isinstance(name, str) or len(name.strip()) == 0:
        logger.warning("Skipping function_call without name")
        return None

    if not isinstance(arguments, str):
        try:
            arguments = str(arguments)
        except Exception:
            arguments = "{}"

    return {
        "id": call_id.strip(),
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": arguments,
        },
    }


def _convert_function_call_output_item(
    item: dict[str, Any],
    pending_tool_calls: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    call_id = item.get("call_id")
    if not call_id or not isinstance(call_id, str) or len(call_id.strip()) == 0:
        logger.warning("Skipping function_call_output without call_id")
        return None

    normalized_call_id = call_id.strip()
    if normalized_call_id not in pending_tool_calls:
        logger.warning(
            "Skipping function_call_output without matching function_call in current input"
        )
        return None

    output = item.get("output", "")
    if not output:
        output = ""
    elif not isinstance(output, str):
        try:
            output = str(output)
        except Exception:
            output = ""

    return {
        "role": "tool",
        "tool_call_id": normalized_call_id,
        "content": output,
    }


def _extract_reasoning_text(summary: Any) -> str:
    if isinstance(summary, str):
        return summary
    if not isinstance(summary, list):
        return ""

    parts: list[str] = []
    for item in summary:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


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


def _convert_tool_call_part(part: dict[str, Any]) -> dict[str, Any] | None:
    if part.get("type") != "tool_call":
        return None

    call_id = part.get("call_id") or part.get("id")
    name = part.get("name")
    arguments = part.get("arguments", "{}")

    if not isinstance(call_id, str) or len(call_id.strip()) == 0:
        return None
    if not isinstance(name, str) or len(name.strip()) == 0:
        return None
    if not isinstance(arguments, str):
        try:
            arguments = str(arguments)
        except Exception:
            arguments = "{}"

    return {
        "id": call_id.strip(),
        "type": "function",
        "function": {
            "name": name.strip(),
            "arguments": arguments,
        },
    }


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        namespace_tool = _extract_namespace_tool(tool)
        if namespace_tool:
            logger.warning(
                "Namespace tool converted: %s, commands=%s",
                namespace_tool["name"],
                namespace_tool["parameters"].get("properties", {}).get("command", {}).get("enum", []),
            )
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": namespace_tool["name"],
                        "description": namespace_tool["description"],
                        "parameters": namespace_tool["parameters"],
                    },
                }
            )
            continue

        tool_type = tool.get("type")
        tool_name = tool.get("name")

        nested_key = next((k for k in tool.keys() if k not in {"description", "parameters", "type", "name"}), None)
        if not tool_name and nested_key:
            nested_data = tool.get(nested_key, {})
            if isinstance(nested_data, dict):
                tool_name = nested_data.get("name", nested_key)
                tool = {
                    "name": tool_name,
                    "description": nested_data.get("description", ""),
                    "parameters": nested_data.get("parameters", {}),
                }
                tool_type = "function"

        if tool_type is None and tool_name:
            tool_type = "function"
        elif tool_type is None and not tool_name:
            tool_type = "function"

        if tool_type != "function":
            logger.warning(f"Converting '{tool_type}' tool type to 'function' type (DeepSeek compatibility)")
            tool_type = "function"

        function_data = tool.get("function", {})
        if isinstance(function_data, dict) and function_data:
            name = function_data.get("name", "")
            description = function_data.get("description", "")
            parameters = function_data.get("parameters", {})
        else:
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})

        if not name or not isinstance(name, str) or len(name.strip()) == 0:
            logger.warning(f"Skipping tool with empty name")
            continue

        parameters = _fix_parameters_schema(parameters)

        converted.append(
            {
                "type": tool_type,
                "function": {
                    "name": name.strip(),
                    "description": description if isinstance(description, str) else "",
                    "parameters": parameters,
                },
            }
        )
        logger.warning(
            "Flat tool converted: %s, parameter_keys=%s",
            name.strip(),
            sorted(parameters.get("properties", {}).keys()) if isinstance(parameters, dict) else [],
        )
    return converted


def _extract_namespace_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    _standard_keys = {"type", "name", "description", "parameters"}

    explicit_type = tool.get("type")
    explicit_name = tool.get("name")
    explicit_description = tool.get("description")
    explicit_parameters = tool.get("parameters")

    if explicit_type == "namespace" and isinstance(explicit_name, str) and explicit_name:
        nested_tools = tool.get("tools")
        if isinstance(nested_tools, list):
            sub_commands = _extract_namespace_sub_commands_from_tools_list(nested_tools)
            if sub_commands:
                logger.info(
                    "Namespace tool extracted from tools list: name=%s, commands=%s",
                    explicit_name,
                    list(sub_commands.keys()),
                )
                return {
                    "name": explicit_name,
                    "description": explicit_description if isinstance(explicit_description, str) else f"Tools in the {explicit_name} namespace.",
                    "parameters": _build_namespace_parameters(sub_commands),
                }
            else:
                logger.warning(
                    "Namespace tool has empty tools list: name=%s, tools=%s",
                    explicit_name,
                    nested_tools,
                )

        nested_data = tool.get(explicit_name)
        if isinstance(nested_data, dict):
            sub_commands: dict[str, dict[str, Any]] = {}
            for sub_key, sub_value in nested_data.items():
                if sub_key in _standard_keys:
                    continue
                if not isinstance(sub_value, dict):
                    continue
                sub_commands[sub_key] = sub_value

            if sub_commands:
                description = explicit_description if isinstance(explicit_description, str) and explicit_description else nested_data.get(
                    "description", f"Tools in the {explicit_name} namespace."
                )
                return {
                    "name": explicit_name,
                    "description": description if isinstance(description, str) else f"Tools in the {explicit_name} namespace.",
                    "parameters": _build_namespace_parameters(sub_commands),
                }

        if isinstance(explicit_parameters, dict) and explicit_parameters.get("properties"):
            return {
                "name": explicit_name,
                "description": explicit_description if isinstance(explicit_description, str) else f"Tools in the {explicit_name} namespace.",
                "parameters": _fix_parameters_schema(explicit_parameters),
            }

    for key, value in tool.items():
        if key in _standard_keys:
            continue
        if not isinstance(value, dict):
            continue

        nested_data = value
        parent_name = nested_data.get("name", key)
        if not isinstance(parent_name, str):
            parent_name = key

        sub_commands: dict[str, dict[str, Any]] = {}
        for sub_key, sub_value in nested_data.items():
            if sub_key in _standard_keys:
                continue
            if not isinstance(sub_value, dict):
                continue
            sub_commands[sub_key] = sub_value

        if sub_commands:
            return {
                "name": parent_name,
                "description": nested_data.get("description", f"Tools in the {parent_name} namespace.") if isinstance(nested_data.get("description"), str) else f"Tools in the {parent_name} namespace.",
                "parameters": _build_namespace_parameters(sub_commands),
            }

        return None

    return None


def _extract_namespace_sub_commands_from_tools_list(tools: list[Any]) -> dict[str, dict[str, Any]]:
    sub_commands: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        sub_commands[name] = tool
    return sub_commands


def _build_namespace_parameters(sub_commands: dict[str, dict[str, Any]]) -> dict[str, Any]:
    command_names = sorted(sub_commands.keys())
    command_descriptions = {
        name: data.get("description", "") if isinstance(data.get("description"), str) else ""
        for name, data in sub_commands.items()
    }

    return {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": command_names,
                "description": "; ".join(
                    f"{name}: {desc}" if desc else name for name, desc in command_descriptions.items()
                ),
            },
            "arguments": {
                "type": "object",
                "properties": {},
                "description": "Arguments for the selected command.",
            },
        },
        "required": ["command"],
    }


def _fix_parameters_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}, "required": []}
    
    if "type" not in parameters or parameters["type"] is None:
        parameters = dict(parameters)
        parameters["type"] = "object"
    
    if "properties" not in parameters:
        parameters = dict(parameters)
        parameters["properties"] = {}
    
    if "required" not in parameters:
        parameters = dict(parameters)
        parameters["required"] = []
    
    for prop_name, prop_schema in parameters.get("properties", {}).items():
        if isinstance(prop_schema, dict) and ("type" not in prop_schema or prop_schema["type"] is None):
            parameters["properties"][prop_name]["type"] = "string"
    
    return parameters


