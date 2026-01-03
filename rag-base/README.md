# RAG Renovation Helper

Simple retrieval-augmented generation pipeline that ingests renovation guideline documents into PostgreSQL with pgvector, retrieves the most relevant passages, and answers questions using OpenAI chat models.

## Prerequisites
- Python 3.12+
- PostgreSQL with the `vector` extension (e.g., `ankane/pgvector` image)
- OpenAI API key
- `uv` installed (`pip install uv` or see https://github.com/astral-sh/uv)

## Project layout
- `rag-base.py` — CLI entrypoint
- `src/` — pipeline modules (`config.py`, `data_loader.py`, `db.py`, `embeddings.py`, `rag_pipeline.py`)
- `data/` — local text documents to ingest (synthetic renovation guidelines provided)

## Setup
1) Clone the repo and enter `rag-base/`.
2) Put secrets in `.env` (repo root or `rag-base/.env`):
   ```
   OPENAI_API_KEY=sk-...
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
   ```
3) Create/activate a virtualenv with uv:
   ```
   uv venv
   source .venv/bin/activate
   ```
4) Install dependencies:
   ```
   uv pip install openai psycopg2-binary pgvector python-dotenv
   ```
5) Ensure PostgreSQL is running and has the `vector` extension (the app will create it if allowed).

## Run
Ingest docs and ask a question (pass it directly or get prompted):
```
python rag-base.py "How do I plan a kitchen renovation?"
python rag-base.py        # will prompt for your question
```
Skip ingestion if you already loaded data:
```
python rag-base.py --skip-ingest "How do I manage dust control during renovation?"
```

## Chunking & determinism
- Documents are split into overlapping word chunks (default size 400 words, overlap 80) before embedding.
- Chat temperature is fixed at 0 for deterministic responses; the system prompt still instructs the model to admit when context is insufficient.

## Workflow (text diagram)
```
[data/*.txt]
      |
      v
 load_documents (src/data_loader.py)
      |
      v
 embed_text via OpenAI (src/embeddings.py)
      |
      v
 upsert into PostgreSQL pgvector (src/db.py)
      |
      v
 retrieve top-k embeddings for a question
      |
      v
 build prompt + context (src/rag_pipeline.py)
      |
      v
 OpenAI chat completion -> answer
```

## Notes
- Defaults: embed model `text-embedding-3-small`, chat model `gpt-4o-mini`, table `renovation_docs`, embed dim 1536.
- Update `data/` with more `.txt` files to expand the knowledge base, then rerun ingestion.
