import json
import logging
from typing import Any

from .config import settings

logger = logging.getLogger("responses_chat_proxy")


def log_request(request_data: dict[str, Any], endpoint: str) -> None:
    logger.info(f"Incoming {endpoint} request")
    logger.debug(f"Request data: {json.dumps(request_data, ensure_ascii=False, indent=2)}")


def log_request_conversion(original: dict[str, Any], converted: dict[str, Any]) -> None:
    original_model = original.get("model", "unknown")
    converted_model = converted.get("model", "unknown")
    
    if original_model != converted_model:
        logger.info(f"Model mapping applied: {original_model} -> {converted_model}")
    
    logger.debug(f"Converted request: {json.dumps(converted, ensure_ascii=False, indent=2)}")


def log_upstream_request(url: str, model: str, is_streaming: bool) -> None:
    stream_str = " (streaming)" if is_streaming else ""
    logger.info(f"Forwarding to upstream{stream_str}: {url}")
    logger.debug(f"Upstream request model: {model}")


def log_upstream_response(status_code: int, response_size: int | None = None) -> None:
    size_str = f", size: {response_size} bytes" if response_size else ""
    logger.info(f"Upstream response: status={status_code}{size_str}")


def log_streaming_chunk(chunk_size: int) -> None:
    logger.debug(f"Streaming chunk received: {chunk_size} bytes")


def log_response_conversion(response_data: dict[str, Any]) -> None:
    logger.debug(f"Converted response: {json.dumps(response_data, ensure_ascii=False, indent=2)}")


def log_error(error_type: str, error_message: str, details: Any = None) -> None:
    logger.error(f"{error_type}: {error_message}")
    if details:
        logger.debug(f"Error details: {details}")


def log_authentication(success: bool, reason: str | None = None) -> None:
    if success:
        logger.debug("Authentication successful")
    else:
        logger.warning(f"Authentication failed: {reason}")


def mask_sensitive_data(data: dict[str, Any]) -> dict[str, Any]:
    masked = data.copy()
    sensitive_keys = {"api_key", "authorization", "api-key", "key", "token", "secret"}
    
    def mask_value(value: Any) -> Any:
        if isinstance(value, dict):
            return mask_sensitive_data(value)
        elif isinstance(value, list):
            return [mask_value(item) for item in value]
        elif isinstance(value, str):
            for key in sensitive_keys:
                if key.lower() in value.lower():
                    if len(value) > 8:
                        return f"{value[:4]}...{value[-4:]}"
                    return "***"
        return value
    
    for key in list(masked.keys()):
        if key.lower() in sensitive_keys:
            masked[key] = "***"
        elif isinstance(masked[key], (dict, list, str)):
            masked[key] = mask_value(masked[key])
    
    return masked
