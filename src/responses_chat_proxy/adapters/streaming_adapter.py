import json
import re
from typing import Any


class StreamingConverter:
    """Convert Chat Completions SSE chunks to Responses API SSE events."""

    def __init__(self) -> None:
        self.initialized = False
        self.output_item_added = False
        self.content_part_added = False
        self.text_done = False
        self.output_item_done = False
        self.completed = False

        self.response_id = ""
        self.message_id = ""
        self.model = ""
        self.created = 0
        self.message_role = "assistant"

        self.reasoning_content = ""
        self.full_text = ""
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.sse_buffer = ""
        self.usage: dict[str, Any] = {}

    def feed(self, chunk: bytes) -> list[str]:
        events: list[str] = []
        try:
            self.sse_buffer += chunk.decode("utf-8", errors="replace")
        except Exception:
            return events

        while True:
            match = re.search(r"\r?\n\r?\n", self.sse_buffer)
            if match is None:
                break

            raw_event = self.sse_buffer[: match.start()]
            self.sse_buffer = self.sse_buffer[match.end() :]

            data_lines = []
            for line in raw_event.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())

            if not data_lines:
                continue

            payload = "\n".join(data_lines).strip()
            if payload == "[DONE]":
                events.extend(self.finish())
                continue

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            events.extend(self._process_chunk(data))

        return events

    def finish(self) -> list[str]:
        events: list[str] = []
        events.extend(self._finish_output_items())
        events.extend(self._finish_response())
        return events

    def _ensure_initialized(self, data: dict[str, Any]) -> list[str]:
        if self.initialized:
            return []

        self.initialized = True
        chat_id = data.get("id", "")
        self.response_id = _convert_id(chat_id)
        suffix = self.response_id.replace("resp-", "", 1).replace("resp_", "")
        self.message_id = f"msg_{suffix or 'unknown'}"
        self.model = data.get("model", self.model)
        self.created = data.get("created", 0)

        return [
            _sse_event(
                "response.created",
                {"response": _build_response_stub(self.response_id, self.created, "in_progress", self.model)},
            ),
            _sse_event(
                "response.in_progress",
                {"response": {"id": self.response_id, "object": "response", "status": "in_progress"}},
            ),
        ]

    def _ensure_message_item(self, role: str | None = None) -> list[str]:
        if self.output_item_added:
            return []
        self.output_item_added = True
        return [
            _sse_event(
                "response.output_item.added",
                {
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": self.message_id,
                        "status": "in_progress",
                        "role": role or self.message_role,
                        "content": [],
                    },
                },
            )
        ]

    def _ensure_content_part(self) -> list[str]:
        if self.content_part_added:
            return []
        self.content_part_added = True
        events = self._ensure_message_item()
        events.append(
            _sse_event(
                "response.content_part.added",
                {
                    "item_id": self.message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
        )
        return events

    def _process_chunk(self, data: dict[str, Any]) -> list[str]:
        events = self._ensure_initialized(data)

        usage = data.get("usage")
        if isinstance(usage, dict) and usage:
            self.usage = usage

        choices = data.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return events

        choice = choices[0]
        delta = choice.get("delta", {}) or {}
        finish_reason = choice.get("finish_reason")

        if delta.get("role"):
            self.message_role = delta["role"]

        reasoning_content = delta.get("reasoning_content")
        if reasoning_content:
            self.reasoning_content += reasoning_content

        content = delta.get("content")
        if content:
            events.extend(self._ensure_content_part())
            self.full_text += content
            events.append(
                _sse_event(
                    "response.output_text.delta",
                    {
                        "item_id": self.message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": content,
                    },
                )
            )

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    events.extend(self._process_tool_call_delta(tool_call))

        if finish_reason is not None and not self.output_item_done:
            status = "incomplete" if finish_reason in {"length", "content_filter"} else "completed"
            events.extend(self._finish_output_items())
            events.extend(self._finish_response(status))

        return events

    def _process_tool_call_delta(self, tool_call: dict[str, Any]) -> list[str]:
        index = tool_call.get("index", 0)
        if not isinstance(index, int):
            index = 0

        entry = self.tool_calls.setdefault(
            index,
            {"id": None, "item_id": None, "name": "", "arguments": "", "added": False},
        )

        if tool_call.get("id"):
            entry["id"] = tool_call["id"]

        function = tool_call.get("function", {}) or {}
        if function.get("name"):
            entry["name"] = function["name"]

        if not entry["item_id"]:
            entry["item_id"] = self._tool_item_id(entry.get("id"), index)

        output_index = self._tool_output_index(index)
        events: list[str] = []

        if not entry["added"]:
            entry["added"] = True
            events.append(
                _sse_event(
                    "response.output_item.added",
                    {
                        "output_index": output_index,
                        "item": {
                            "type": "function_call",
                            "id": entry["item_id"],
                            "call_id": entry["id"],
                            "status": "in_progress",
                            "name": entry["name"],
                            "arguments": "",
                        },
                    },
                )
            )

        argument_delta = function.get("arguments")
        if argument_delta:
            entry["arguments"] += argument_delta
            events.append(
                _sse_event(
                    "response.function_call_arguments.delta",
                    {"item_id": entry["item_id"], "output_index": output_index, "delta": argument_delta},
                )
            )

        return events

    def _tool_output_index(self, tool_index: int) -> int:
        return (1 if self.output_item_added else 0) + tool_index

    def _tool_item_id(self, call_id: str | None, index: int) -> str:
        if call_id:
            return f"fc_{call_id.replace('call_', '', 1)}"
        suffix = self.response_id.replace("resp-", "", 1).replace("resp_", "") or "unknown"
        return f"fc_{suffix}_{index}"

    def _finish_text_events(self) -> list[str]:
        if not self.content_part_added or self.text_done:
            return []
        self.text_done = True
        return [
            _sse_event(
                "response.output_text.done",
                {
                    "item_id": self.message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": self.full_text,
                },
            ),
            _sse_event(
                "response.content_part.done",
                {
                    "item_id": self.message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": self.full_text, "annotations": []},
                },
            ),
        ]

    def _finish_output_items(self) -> list[str]:
        events: list[str] = []
        if not self.output_item_done:
            self.output_item_done = True
            events.extend(self._finish_text_events())
            if self.output_item_added:
                events.append(
                    _sse_event(
                        "response.output_item.done",
                        {
                            "output_index": 0,
                            "item": {
                                "type": "message",
                                "id": self.message_id,
                                "status": "completed",
                                "role": self.message_role,
                                "content": [
                                    {"type": "output_text", "text": self.full_text, "annotations": []}
                                ]
                                if self.content_part_added
                                else [],
                            "reasoning_content": self.reasoning_content,
                            },
                        },
                    )
                )

        events.extend(self._finish_tool_events())
        return events

    def _finish_tool_events(self) -> list[str]:
        events: list[str] = []
        for index, tool_call in sorted(self.tool_calls.items()):
            if not tool_call.get("added"):
                continue
            output_index = self._tool_output_index(index)
            events.append(
                _sse_event(
                    "response.function_call_arguments.done",
                    {
                        "item_id": tool_call["item_id"],
                        "output_index": output_index,
                        "arguments": tool_call["arguments"],
                    },
                )
            )
            events.append(
                _sse_event(
                    "response.output_item.done",
                    {
                        "output_index": output_index,
                        "item": {
                            "type": "function_call",
                            "id": tool_call["item_id"],
                            "call_id": tool_call["id"],
                            "status": "completed",
                            "name": tool_call["name"],
                            "arguments": tool_call["arguments"],
                        },
                    },
                )
            )
        return events

    def _finish_response(self, status: str = "completed") -> list[str]:
        if self.completed:
            return []
        self.completed = True
        return [
            _sse_event(
                "response.completed",
                {
                    "response": {
                        "id": self.response_id or "resp_unknown",
                        "object": "response",
                        "created_at": self.created,
                        "status": status,
                        "output": self._build_output(),
                        "usage": self._build_usage(),
                    }
                },
            )
        ]

    def _build_output(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        if self.reasoning_content:
            output.append(
                {
                    "id": f"rs_{self.response_id.replace('resp-', '', 1).replace('resp_', '') or 'unknown'}",
                    "type": "reasoning",
                    "content": [],
                    "summary": [{"type": "summary_text", "text": self.reasoning_content}],
                }
            )
        if self.content_part_added:
            output.append(
                {
                    "type": "message",
                    "id": self.message_id,
                    "status": "completed",
                    "role": self.message_role,
                    "content": [{"type": "output_text", "text": self.full_text, "annotations": []}],
                    "reasoning_content": self.reasoning_content,
                }
            )
        for _, tool_call in sorted(self.tool_calls.items()):
            if tool_call.get("added"):
                output.append(
                    {
                        "type": "function_call",
                        "id": tool_call["item_id"],
                        "call_id": tool_call["id"],
                        "status": "completed",
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    }
                )
        return output

    def _build_usage(self) -> dict[str, int]:
        prompt_tokens = int(self.usage.get("prompt_tokens") or 0)
        completion_tokens = int(self.usage.get("completion_tokens") or 0)
        total_tokens = int(self.usage.get("total_tokens") or 0)
        return {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


def _convert_id(chat_id: str) -> str:
    if chat_id.startswith("chatcmpl-"):
        return chat_id.replace("chatcmpl-", "resp-", 1)
    if chat_id.startswith("chatcmpl"):
        return chat_id.replace("chatcmpl", "resp", 1)
    return f"resp_{chat_id}" if chat_id else "resp_unknown"


def _build_response_stub(response_id: str, created: int, status: str, model: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": [],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "user": None,
        "metadata": {},
    }


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    event = {"type": event_type}
    event.update(data)
    return f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
