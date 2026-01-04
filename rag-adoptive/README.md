# RAG Adaptive Helper

Adaptive RAG travel assistant that routes each query to the cheapest effective path:
- **direct**: no retrieval (greetings/chit-chat/simple).
- **rag**: single-pass retrieval from pgvector + external weather tool when relevant.
- **agent**: multi-source (retrieval + external weather) for multi-city plans/comparisons.

## Prerequisites
- Python 3.12+
- PostgreSQL with the `vector` extension (e.g., `ankane/pgvector`)
- OpenAI API key
- `uv` installed

## Layout
- `rag-adoptive.py` — CLI entrypoint (ingest + router + chat loop)
- `src/`
  - `config.py` — env/defaults (table `travel_docs_adoptive`)
  - `data_loader.py` — chunked document loading
  - `db.py` — pgvector schema/upsert/query
  - `embeddings.py` — OpenAI embeddings
  - `conversation.py` — rolling history (last 5 turns)
  - `external_search.py` — tool router (LLM + keywords) + multi-city weather via Open-Meteo
  - `rag_pipeline.py` — router (direct|rag|agent), retrieval, synthesis
- `data/` — travel guideline docs (USA, Europe, Asia, packing, safety, insurance, family, nomad, winter, summer/heat, etc.)
- Optional MCP (copied from corrective):
  - `mcp_weather.py`, `mcp_client.py`

## Setup
1) From repo root, enter `rag-adoptive/`.
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
   uv pip install openai psycopg2-binary pgvector python-dotenv requests
   ```
5) Ensure PostgreSQL is running; the app will create the `vector` extension/table/index if allowed.

## Run
- Ingest + ask (prompts if no question given):
  ```
  python rag-adoptive.py "Plan a 3-day trip from Dallas to Lake Oswego with today's weather."
  python rag-adoptive.py      # will prompt
  ```
- Reuse existing data without ingestion:
  ```
  python rag-adoptive.py --skip-ingest
  ```
- After the first answer, it stays in interactive chat; blank line or `exit`/`quit` to leave.

## Adaptive routing
- **Classifier (LLM-based)** decides:
  - `direct`: greetings/chit-chat/simple known facts → no retrieval.
  - `rag`: single-pass retrieval from pgvector; pulls external weather when relevant.
  - `agent`: multi-source plan/compare; combines retrieval + external weather (multi-city) with step-by-step instructions.
- Conversation history (last 5 turns) is passed into classification and answering for follow-ups.

## External tools
- `external_search.py`: keyword + LLM tool routing. Current tool: multi-city weather (LLM location extraction/correction) via Open-Meteo (no API key). Extend with more tools (traffic/search) by adding to the registry.

## Architecture (text diagram)
```
[data/*.txt travel docs]
      |
      v
 chunk_text -> embed_text (OpenAI)
      |
      v
 upsert into PostgreSQL pgvector
      |
      v
 Router (LLM): direct | rag | agent
      |        |        |
      |        |        +--> agent: retrieve top-k + external_search (multi-city weather) -> prompt -> answer
      |        +--> rag: retrieve top-k (+ weather if relevant) -> prompt -> answer
      +--> direct: no retrieval -> answer
      |
      v
 OpenAI chat completion (temperature=0), with rolling history
```

## Notes
- Chunking: default 400-word chunks with 80-word overlap.
- Determinism: chat calls set `temperature=0`.
- Table: `travel_docs_adoptive` (separate from other projects).
- MCP weather server/client remain available but are not called automatically by the app.
