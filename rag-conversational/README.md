# RAG Conversational Helper

Conversational retrieval-augmented generation that keeps a rolling history of the last 5 turns. Renovation guideline documents are embedded into PostgreSQL with pgvector, and OpenAI chat uses both retrieved context and prior turns.

## Prerequisites
- Python 3.12+
- PostgreSQL with the `vector` extension (e.g., `ankane/pgvector`)
- OpenAI API key
- `uv` installed

## Layout
- `rag-conversational.py` — CLI entrypoint
- `src/` — pipeline modules (config, data_loader, db, embeddings, conversation, rag_pipeline)
- `data/` — sample travel guideline docs

## Setup
1) From repo root, enter `rag-conversational/`.
2) Create `.env` (here or in `../rag/.env`):
   ```
   OPENAI_API_KEY=sk-...
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
   ```
3) Create/activate env with uv:
   ```
   uv venv
   source .venv/bin/activate
   ```
4) Install deps:
   ```
   uv pip install openai psycopg2-binary pgvector python-dotenv
   ```
5) Ensure PostgreSQL is running; the app will create the `vector` extension/table/index if allowed.

## Run
- Start ingestion + interactive chat (blank line or "exit" to quit):
  ```
  python rag-conversational.py
  ```
- Seed with an initial question, then stay in chat:
  ```
python rag-conversational.py "How should I plan a weekend in Tokyo?"
  ```
- Reuse existing data without re-ingesting:
  ```
  python rag-conversational.py --skip-ingest
  ```

## Workflow (text diagram)
```
[data/*.txt]
      |
      v
 load_documents
      |
      v
 embed_text (OpenAI)
      |
      v
 upsert into PostgreSQL pgvector
      |
      v
 retrieve top-k for question
      |
      v
 build prompt with context + last 5 turns
      |
      v
 OpenAI chat completion -> answer
```

## Notes
- Rolling memory size: 5 turns (user+assistant pairs), configured in `Settings.history_size`.
- Chunking: documents are split into overlapping word chunks (default size 400, overlap 80) before embedding.
- Chat temperature is fixed at 0 for deterministic responses; the system prompt asks the model to say when context is insufficient.
- Defaults: embed model `text-embedding-3-small`, chat model `gpt-4o-mini`, table `travel_docs_conversational`, embed dim 1536.
- Add more `.txt` docs to `data/` to expand knowledge; rerun ingestion.
