# RAG Agentic Helper

Agentic RAG travel assistant that plans, reasons, and orchestrates retrieval/tools before answering. It analyzes the query, drafts a minimal plan, executes vector search and external weather tools as needed, and synthesizes a grounded response.

## Prerequisites
- Python 3.12+
- PostgreSQL with the `vector` extension (e.g., `ankane/pgvector`)
- OpenAI API key
- `uv` installed

## Layout
- `rag-agentic.py` — CLI entrypoint (ingest + LangGraph agent + chat)
- `src/`
  - `config.py` — env/defaults
  - `data_loader.py` — chunked document loading
  - `db.py` — pgvector schema/upsert/query
  - `embeddings.py` — OpenAI embeddings
  - `conversation.py` — rolling history (last 5 turns)
  - `external_search.py` — public external search (Open-Meteo weather; LLM location correction; tool routing)
  - `tools.py` — legacy tool runner (optional); LangGraph binds tools directly
  - `rag_pipeline.py` — LangGraph agent: plan (LLM) → act (tools: vector_search, weather_lookup) → answer
- `data/` — travel guideline docs (USA, Europe, Asia, packing, safety, insurance, family, nomad, winter, summer/heat, etc.)

## Setup
1) From repo root, enter `rag-agentic/`.
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
  uv pip install openai psycopg2-binary pgvector python-dotenv requests langgraph langchain-openai langchain-core langchain
   ```
5) Ensure PostgreSQL is running; the app will create the `vector` extension/table/index if allowed.

## Run
- Ingest + ask (prompts if no question given):
  ```
  python rag-agentic.py "Plan a multi-city trip from Dallas to Lake Oswego; include today's weather."
  python rag-agentic.py      # will prompt
  ```
- Reuse existing data without ingestion:
  ```
  python rag-agentic.py --skip-ingest
  ```
- After the first answer, it stays in interactive chat; blank line or `exit`/`quit` to leave.

## Agentic loop (LangGraph)
- **Tools**: vector_search (pgvector) and weather_lookup (Open-Meteo, no key; LLM location correction/multi-city).
- **Model**: OpenAI (langchain_openai ChatOpenAI) bound to tools.
- **Graph**: llm_call → (if tool call) tool_node → llm_call ... until no tool call, then respond.
- **History**: last 5 turns passed into the graph start state.

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
 LangGraph agent:
   - llm_call (Claude + tools) -> optional tool calls
   - tool_node (vector_search, weather_lookup) -> back to llm_call if more actions
   - stop when no tool calls; respond
   - history included in initial state
```

## Notes
- Chunking: default 400-word chunks with 80-word overlap.
- Determinism: chat and planning calls set `temperature=0`.
- External: Open-Meteo weather (no key), LLM location correction/multi-city extraction; add more tools via `tools.py`.
- Memory: last 5 turns included in planning and answering.
