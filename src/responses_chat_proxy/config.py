import json
import logging
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    upstream_base_url: str = Field(default="https://api.openai.com/v1")
    upstream_api_key: str = Field(default="")
    proxy_api_key: str = Field(default="")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    request_timeout_seconds: float = Field(default=120)
    stream_timeout_seconds: float = Field(default=300)
    verify_ssl: bool = Field(default=True)
    log_level: str = Field(default="info")
    model_mapping: dict[str, str] = Field(default_factory=dict)
    log_requests: bool = Field(default=False)
    log_responses: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("model_mapping", mode="before")
    @classmethod
    def parse_model_mapping(cls, v: Any) -> dict[str, str]:
        if isinstance(v, dict):
            return v
        if isinstance(v, str) and v:
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return {str(k): str(v) for k, v in parsed.items()}
            except json.JSONDecodeError:
                logging.warning(f"Invalid MODEL_MAPPING JSON: {v}")
        return {}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._setup_logging()
        if self.model_mapping:
            logging.info(f"Model mapping configured: {len(self.model_mapping)} mappings")

    def _setup_logging(self) -> None:
        log_level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }
        level = log_level_map.get(self.log_level.lower(), logging.INFO)
        
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    def get_upstream_model(self, client_model: str) -> str:
        mapped = self.model_mapping.get(client_model, client_model)
        if mapped != client_model:
            logging.debug(f"Model mapped: {client_model} -> {mapped}")
        return mapped


settings = Settings()
