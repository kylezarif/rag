import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ENV = BASE_DIR / ".env"
ROOT_ENV = BASE_DIR.parent / ".env"

# Load .env early so the rest of the pipeline can rely on environment variables.
for env_file in (PROJECT_ENV, ROOT_ENV):
    if env_file.exists():
        load_dotenv(env_file)


@dataclass
class Settings:
    openai_api_key: str
    database_url: str
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    table_name: str = "renovation_docs"
    embed_dim: int = 1536
    data_dir: Path = BASE_DIR / "data"


def load_settings(
    openai_api_key: Optional[str] = None,
    database_url: Optional[str] = None,
) -> Settings:
    """
    Load settings from environment, with optional explicit overrides.
    """
    key = openai_api_key or _require_env("OPENAI_API_KEY")
    db_url = database_url or _require_env(
        "DATABASE_URL",
        fallback="postgresql://postgres:postgres@localhost:5432/postgres",
    )
    return Settings(openai_api_key=key, database_url=db_url)


def _require_env(name: str, fallback: Optional[str] = None) -> str:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if fallback:
        return fallback
    raise RuntimeError(f"Missing required environment variable: {name}")
