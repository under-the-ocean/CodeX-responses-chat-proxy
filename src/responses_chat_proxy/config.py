from pydantic import Field
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
