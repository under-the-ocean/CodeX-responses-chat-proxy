import json

from responses_chat_proxy.adapters import StreamingConverter, convert_request, convert_response


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
