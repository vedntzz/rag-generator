"""Runtime settings from RAG_* environment variables and an optional .env file."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-haiku-4-5-20251001"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: Path = Path("./data")
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    min_score: float = 0.3
