import json
import os
from unittest.mock import patch

from responses_chat_proxy.adapters import StreamingConverter, convert_request, convert_response
from responses_chat_proxy.main import _inject_cached_reasoning_content


def test_convert_request_string_input() -> None:
    converted = convert_request(
        {
            "model": "test-model",
            "instructions": "Be concise.",
            "input": "Hello",
            "max_output_tokens": 20,
            "reasoning": {"effort": "low"},
        }
    )

    assert converted["model"] == "test-model"
    assert converted["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]
    assert converted["max_tokens"] == 20
    assert converted["reasoning_effort"] == "low"


def test_convert_request_with_model_mapping() -> None:
    from responses_chat_proxy.config import Settings
    
    test_settings = Settings(
        model_mapping={"gpt-4o-mini": "gpt-3.5-turbo", "gpt-4-turbo": "gpt-4"}
    )
    
    assert test_settings.get_upstream_model("gpt-4o-mini") == "gpt-3.5-turbo"
    assert test_settings.get_upstream_model("gpt-4-turbo") == "gpt-4"
    assert test_settings.get_upstream_model("unknown-model") == "unknown-model"


def test_model_mapping_applied_in_convert_request() -> None:
    import importlib
    import responses_chat_proxy.config
    importlib.reload(responses_chat_proxy.config)
    
    import responses_chat_proxy.adapters.request_adapter
    importlib.reload(responses_chat_proxy.adapters.request_adapter)
    
    from responses_chat_proxy.adapters.request_adapter import convert_request
    
    from responses_chat_proxy.config import settings
    original_mapping = settings.model_mapping
    
    try:
        settings.model_mapping = {"test-model": "target-model"}
        converted = convert_request({"model": "test-model", "input": "Hello"})
        assert converted["model"] == "target-model"
    finally:
        settings.model_mapping = original_mapping


def test_custom_tool_type_conversion() -> None:
    from responses_chat_proxy.adapters.request_adapter import _convert_tools
    
    tools = [
        {"type": "custom", "name": "test", "description": "test tool", "parameters": {}},
        {"type": "namespace", "name": "ns_test", "description": "namespace tool", "parameters": {}},
        {"type": "tool_search", "name": "search", "description": "search tool", "parameters": {}},
        {"type": "function", "name": "func", "description": "function tool", "parameters": {}}
    ]
    
    converted = _convert_tools(tools)
    
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "test"
    assert converted[1]["type"] == "function"
    assert converted[1]["function"]["name"] == "ns_test"
    assert converted[2]["type"] == "function"
    assert converted[2]["function"]["name"] == "search"
    assert converted[3]["type"] == "function"
    assert converted[3]["function"]["name"] == "func"


def test_fix_parameters_schema() -> None:
    from responses_chat_proxy.adapters.request_adapter import _fix_parameters_schema
    
    params_with_null_type = {"type": None, "properties": {"name": {"type": None}}, "required": ["name"]}
    fixed = _fix_parameters_schema(params_with_null_type)
    
    assert fixed["type"] == "object"
    assert fixed["properties"]["name"]["type"] == "string"
    assert fixed["required"] == ["name"]
    
    params_missing_type = {"properties": {"id": {"type": "integer"}}}
    fixed = _fix_parameters_schema(params_missing_type)
    
    assert fixed["type"] == "object"
    assert "required" in fixed


def test_convert_request_function_call_and_output_pair() -> None:
    converted = convert_request(
        {
            "model": "test-model",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "查天气"}]},
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"city":"Shanghai"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": '{"temp":26}',
                },
            ],
        }
    )

    assert converted["messages"] == [
        {"role": "user", "content": "查天气"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Shanghai"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": '{"temp":26}'},
    ]


def test_convert_request_skips_orphan_function_call_output() -> None:
    converted = convert_request(
        {
            "model": "test-model",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_missing",
                    "output": "result",
                }
            ],
        }
    )

    assert converted["messages"] == []


def test_convert_request_groups_multiple_tool_calls_before_outputs() -> None:
    converted = convert_request(
        {
            "model": "test-model",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "tool_one",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "tool_two",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "one",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "two",
                },
            ],
        }
    )

    assert converted["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "tool_one", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "tool_two", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "one"},
        {"role": "tool", "tool_call_id": "call_2", "content": "two"},
    ]


def test_convert_request_preserves_reasoning_content_for_assistant_tool_calls() -> None:
    converted = convert_request(
        {
            "model": "test-model",
            "input": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "先分析天气接口返回格式"}],
                    "content": [],
                },
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "ok",
                },
            ],
        }
    )

    assert converted["messages"][0]["reasoning_content"] == "先分析天气接口返回格式"
    assert converted["messages"][0]["tool_calls"][0]["id"] == "call_123"


def test_convert_request_replays_assistant_message_with_tool_call_content_parts() -> None:
    converted = convert_request(
        {
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "reasoning_content": "我先调用工具",
                    "content": [
                        {
                            "type": "tool_call",
                            "call_id": "call_hist_1",
                            "name": "get_weather",
                            "arguments": '{"city":"Shanghai"}',
                        }
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_hist_1",
                    "output": "晴天",
                },
            ],
        }
    )

    assert converted["messages"] == [
        {
            "role": "assistant",
            "content": [],
            "reasoning_content": "我先调用工具",
            "tool_calls": [
                {
                    "id": "call_hist_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Shanghai"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_hist_1", "content": "晴天"},
    ]


def test_convert_response_preserves_reasoning_content() -> None:
    converted = convert_response(
        {
            "id": "chatcmpl-abc",
            "created": 123,
            "model": "test-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "先想一下",
                        "content": "Hi",
                    },
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
        {"model": "test-model", "input": "Hello"},
    )

    assert converted["output"][0]["type"] == "reasoning"
    assert converted["output"][0]["summary"][0]["text"] == "先想一下"
    assert converted["output"][1]["reasoning_content"] == "先想一下"


def test_inject_cached_reasoning_content_backfills_from_request_messages() -> None:
    chat_request = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "先分析一下工具调用",
                "tool_calls": [
                    {
                        "id": "call_hist_1",
                        "type": "function",
                        "function": {"name": "tool_one", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_hist_1", "content": "ok"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_hist_1",
                        "type": "function",
                        "function": {"name": "tool_one", "arguments": "{}"},
                    }
                ],
            },
        ]
    }

    _inject_cached_reasoning_content(chat_request)

    assert chat_request["messages"][2]["reasoning_content"] == "先分析一下工具调用"


def test_convert_tools_builds_namespace_command_schema() -> None:
    from responses_chat_proxy.adapters.request_adapter import _convert_tools

    tools = [
        {
            "type": "custom",
            "codex_app": {
                "name": "codex_app",
                "description": "Codex app operations",
                "automation_update": {
                    "description": "Update automation",
                    "parameters": {
                        "type": "object",
                        "properties": {"automation_id": {"type": "string"}},
                        "required": ["automation_id"],
                    },
                },
                "load_workspace_dependencies": {
                    "description": "Load workspace dependencies",
                    "parameters": {
                        "type": "object",
                        "properties": {"workspace_path": {"type": "string"}},
                        "required": ["workspace_path"],
                    },
                },
            },
        }
    ]

    converted = _convert_tools(tools)

    assert len(converted) == 1
    assert converted[0]["function"]["name"] == "codex_app"
    assert converted[0]["function"]["description"] == "Codex app operations"
    parameters = converted[0]["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["command"]
    assert parameters["properties"]["command"]["enum"] == ["automation_update", "load_workspace_dependencies"]
    assert parameters["properties"]["arguments"]["type"] == "object"


def test_convert_tools_handles_flat_nested_without_sub_commands() -> None:
    from responses_chat_proxy.adapters.request_adapter import _convert_tools

    tools = [
        {
            "type": "custom",
            "my_tool": {
                "name": "my_tool",
                "description": "A flat tool",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]

    converted = _convert_tools(tools)

    assert len(converted) == 1
    assert converted[0]["function"]["name"] == "my_tool"
    assert converted[0]["function"]["description"] == "A flat tool"


def test_convert_tools_handles_string_function_field() -> None:
    from responses_chat_proxy.adapters.request_adapter import _convert_tools

    tools = [
        {
            "type": "web_search",
            "name": "web_search",
            "description": "Search the web",
            "parameters": {},
            "function": "web_search",
        }
    ]

    converted = _convert_tools(tools)

    assert len(converted) == 1
    assert converted[0]["function"]["name"] == "web_search"


def test_convert_tools_handles_string_nested_payload() -> None:
    from responses_chat_proxy.adapters.request_adapter import _convert_tools

    tools = [
        {
            "type": "namespace",
            "codex_app": "unsupported",
            "name": "codex_app",
            "description": "Namespace wrapper",
            "parameters": {},
        }
    ]

    converted = _convert_tools(tools)

    assert len(converted) == 1
    assert converted[0]["function"]["name"] == "codex_app"


def test_convert_tools_handles_namespace_name_with_nested_payload() -> None:
    from responses_chat_proxy.adapters.request_adapter import _convert_tools

    tools = [
        {
            "type": "namespace",
            "name": "codex_app",
            "description": "Tools in the codex_app namespace.",
            "parameters": {},
            "codex_app": {
                "automation_update": {
                    "description": "Update automation",
                    "parameters": {
                        "type": "object",
                        "properties": {"automation_id": {"type": "string"}},
                        "required": ["automation_id"],
                    },
                },
                "load_workspace_dependencies": {
                    "description": "Load workspace dependencies",
                    "parameters": {
                        "type": "object",
                        "properties": {"workspace_path": {"type": "string"}},
                        "required": ["workspace_path"],
                    },
                },
            },
        }
    ]

    converted = _convert_tools(tools)

    assert len(converted) == 1
    assert converted[0]["function"]["name"] == "codex_app"
    parameters = converted[0]["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["command"]
    assert parameters["properties"]["command"]["enum"] == ["automation_update", "load_workspace_dependencies"]
    assert parameters["properties"]["arguments"]["type"] == "object"


def test_convert_tools_handles_namespace_tools_list_payload() -> None:
    from responses_chat_proxy.adapters.request_adapter import _convert_tools

    tools = [
        {
            "type": "namespace",
            "name": "codex_app",
            "description": "Tools in the codex_app namespace.",
            "tools": [
                {
                    "type": "function",
                    "name": "automation_update",
                    "description": "Update automation",
                    "parameters": {
                        "type": "object",
                        "properties": {"automation_id": {"type": "string"}},
                        "required": ["automation_id"],
                    },
                },
                {
                    "type": "function",
                    "name": "load_workspace_dependencies",
                    "description": "Load workspace dependencies",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            ],
        }
    ]

    converted = _convert_tools(tools)

    assert len(converted) == 1
    assert converted[0]["function"]["name"] == "codex_app"
    parameters = converted[0]["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["command"]
    assert parameters["properties"]["command"]["enum"] == ["automation_update", "load_workspace_dependencies"]
    assert parameters["properties"]["arguments"]["type"] == "object"


def test_convert_response_non_streaming() -> None:
    converted = convert_response(
        {
            "id": "chatcmpl-abc",
            "created": 123,
            "model": "test-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hi"},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
        {"model": "test-model", "input": "Hello"},
    )

    assert converted["id"] == "resp-abc"
    assert converted["status"] == "completed"
    assert converted["output"][0]["content"][0]["text"] == "Hi"
    assert converted["usage"]["input_tokens"] == 2
    assert converted["usage"]["output_tokens"] == 3


def test_streaming_converter_text_delta() -> None:
    converter = StreamingConverter()
    chunk = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-abc",
                "created": 123,
                "model": "test-model",
                "choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}],
            }
        )
        + "\n\n"
        + "data: "
        + json.dumps(
            {
                "id": "chatcmpl-abc",
                "created": 123,
                "model": "test-model",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        + "\n\n"
    ).encode()

    events = "".join(converter.feed(chunk))

    assert "response.created" in events
    assert "response.output_text.delta" in events
    assert "response.completed" in events
    assert "Hi" in events


def test_streaming_converter_preserves_reasoning_content() -> None:
    converter = StreamingConverter()
    chunk = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-abc",
                "created": 123,
                "model": "test-model",
                "choices": [{"delta": {"role": "assistant", "reasoning_content": "先思考", "content": "答案"}, "finish_reason": None}],
            }
        )
        + "\n\n"
        + "data: "
        + json.dumps(
            {
                "id": "chatcmpl-abc",
                "created": 123,
                "model": "test-model",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        + "\n\n"
    ).encode()

    events = "".join(converter.feed(chunk))

    assert "先思考" in events
    assert '"type": "reasoning"' in events
