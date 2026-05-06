from typing import Any


def error_payload(message: str, error_type: str, code: str, param: str | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {
        "message": message,
        "type": error_type,
        "code": code,
    }
    if param is not None:
        error["param"] = param
    return {"error": error}


def wrap_upstream_error(data: Any, status_code: int) -> dict[str, Any]:
    original = data if isinstance(data, dict) else {"raw": str(data)}
    message = "Upstream error"
    error_type = "upstream_error"
    code = f"http_{status_code}"

    if isinstance(data, dict):
        upstream_error = data.get("error")
        if isinstance(upstream_error, dict):
            message = upstream_error.get("message") or str(upstream_error)
            error_type = upstream_error.get("type") or error_type
            code = upstream_error.get("code") or code
        elif upstream_error:
            message = str(upstream_error)
        elif data.get("detail"):
            message = str(data["detail"])
        elif data.get("message"):
            message = str(data["message"])
        elif data.get("raw_response"):
            message = str(data["raw_response"])[:500] or "Upstream returned an empty response"
    else:
        message = str(data)[:500]

    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
            "original_response": original,
        }
    }


def sse_error(message: str, code: str, error_type: str = "upstream_error") -> bytes:
    import json

    data = {
        "type": "error",
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        },
    }
    return f"event: error\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
