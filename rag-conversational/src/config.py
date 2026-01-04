import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ENV = BASE_DIR / ".env"
ROOT_ENV = BASE_DIR.parent / ".env"

for env_file in (PROJECT_ENV, ROOT_ENV):
    if env_file.exists():
        load_dotenv(env_file)


@dataclass
class Settings:
    openai_api_key: str
    database_url: str
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    table_name: str = "travel_docs_conversational"
    embed_dim: int = 1536
    data_dir: Path = BASE_DIR / "data"
    history_size: int = 5
    chunk_size: int = 400  # approximate words per chunk
    chunk_overlap: int = 80  # overlapping words between chunks


def load_settings(
    openai_api_key: Optional[str] = None,
    database_url: Optional[str] = None,
) -> Settings:
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
