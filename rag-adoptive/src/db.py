import psycopg2
from psycopg2 import OperationalError, sql
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from typing import Iterable, List, Tuple

from src.config import Settings


def get_connection(settings: Settings, register: bool = True):
    try:
        conn = psycopg2.connect(settings.database_url)
        if register:
            try:
                register_vector(conn)
            except psycopg2.ProgrammingError as err:
                raise RuntimeError(
                    "pgvector extension is missing in the target database. "
                    "Create it with: CREATE EXTENSION IF NOT EXISTS vector;"
                ) from err
        return conn
    except OperationalError as err:
        raise RuntimeError(
            f"Could not connect to PostgreSQL at DATABASE_URL={settings.database_url}. "
            "Ensure the server is running and accepting connections, or set DATABASE_URL to a reachable instance."
        ) from err


def ensure_schema(settings: Settings) -> None:
    with get_connection(settings, register=False) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    title TEXT UNIQUE,
                    content TEXT,
                    embedding vector(%s)
                )
                """
            ).format(table=sql.Identifier(settings.table_name)),
            [settings.embed_dim],
        )
        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {table}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            ).format(
                index_name=sql.Identifier(f"{settings.table_name}_embedding_idx"),
                table=sql.Identifier(settings.table_name),
            )
        )
        conn.commit()


def upsert_documents(
    settings: Settings, documents: Iterable[Tuple[str, str, List[float]]]
) -> None:
    records = list(documents)
    if not records:
        return
    with get_connection(settings) as conn, conn.cursor() as cur:
        execute_values(
            cur,
            sql.SQL(
                """
                INSERT INTO {table} (title, content, embedding)
                VALUES %s
                ON CONFLICT (title) DO UPDATE
                SET content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """
            ).format(table=sql.Identifier(settings.table_name)),
            records,
        )
        conn.commit()


def fetch_similar(
    settings: Settings, query_embedding: List[float], limit: int = 3
) -> List[Tuple[str, str, float]]:
    with get_connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT title, content, (embedding <-> %s::vector) AS distance
                FROM {table}
                ORDER BY embedding <-> %s::vector
                LIMIT %s
                """
            ).format(table=sql.Identifier(settings.table_name)),
            [query_embedding, query_embedding, limit],
        )
        return cur.fetchall()
