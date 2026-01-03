# RAG Corrective Helper (CRAG)

Travel assistant that uses Corrective RAG: a decision gate grades retrieved chunks and falls back to public APIs (stubbed weather/traffic) when internal context is weak. Adapted from the conversational RAG project with CRAG-style grading.

## Prerequisites
- Python 3.12+
- PostgreSQL with the `vector` extension (e.g., `ankane/pgvector`)
- OpenAI API key
- `uv` installed

## Layout
- `rag-corrective.py` — CLI entrypoint (ingest + single Q&A + interactive chat)
- `src/` — pipeline modules:
  - `config.py` — env loading and defaults
  - `data_loader.py` — chunked document loading
  - `db.py` — pgvector schema/upsert/query
  - `embeddings.py` — OpenAI embeddings
- `conversation.py` — rolling history (last 5 turns)
- `decision_gate.py` — grader for Correct/Ambiguous/Incorrect
- `external_search.py` — public external search (Open-Meteo geocoding + forecast; no API keys; LLM-corrected locations)
- `rag_pipeline.py` — ingestion, retrieval, grading, synthesis
- `mcp_weather.py` — MCP weather server (tools: get_forecast, get_alerts)
- `mcp_client.py` — minimal MCP client to call server tools without an LLM
- `data/` — sample travel guideline docs (USA parks, Europe rail, Asia hopping, packing, insurance, safety, family, nomad, winter, summer/heat, etc.)

## Setup
1) From repo root, enter `rag-corrective/`.
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
- Ingest + ask (prompts if no question given):
  ```
  python rag-corrective.py "What should I pack for a rainy week in Paris?"
  python rag-corrective.py      # will prompt
  ```
- Reuse existing data without ingestion:
  ```
  python rag-corrective.py --skip-ingest
  ```
- After the first answer, it stays in interactive chat; blank line or `exit`/`quit` to leave.

## Architecture (CRAG flow)
```
[data/*.txt]
      |
      v
 chunk_text -> embed_text (OpenAI)
      |
      v
 upsert into PostgreSQL pgvector
      |
      v
 retrieve top-k for question
      |
      v
 decision_gate (grader): Correct | Ambiguous | Incorrect
      |           |               |
      |           |               +--> external_search() only
      |           +--> internal + external_search()
      +--> internal only
      |
      v
build prompt with context (+ conversation history)
      |
      v
OpenAI chat completion (temperature=0) -> answer
```

### Decision Gate details
- **Evaluation**: A lightweight grader model scores each retrieved chunk as **Correct**, **Ambiguous**, or **Incorrect**.
- **Trigger Gate**:
  - **Correct** → Use internal chunks only.
  - **Ambiguous** → Combine internal chunks with external fallback (weather/traffic/web stub).
  - **Incorrect** → Discard internal chunks; rely on external fallback only.
- This reduces hallucinations by refusing to synthesize from weak or irrelevant context.

### External fallback
- The demo ships with an offline stub that returns templated weather/traffic snippets when the grader deems internal results weak.
- The system prompt instructs the model to treat any “External API” context as current and to use it directly.
- Swap `external_search` to call real providers (e.g., Open-Meteo/NOAA for weather, TomTom/Google/Mapbox for traffic, or Tavily/Google search) to get live data.

## Notes
- Chunking: default 400-word chunks with 80-word overlap.
- Determinism: chat and grader calls set `temperature=0`; grader uses `gpt-4o-mini` by default.
- External search: Open-Meteo geocoding + forecast (no key). Non-weather queries return a simple “no live data” placeholder unless you extend it.
- Rolling memory: last 5 user/assistant turns included in prompts.

## MCP weather server/client
- Install deps (in this folder): `uv pip install httpx "mcp[cli]"`
- Run server (STDIO): `uv run mcp_weather.py`
- Try client against server: `uv run mcp_client.py ./mcp_weather.py`
- Tools exposed:
  - `get_alerts(state="TX")` — US NWS alerts by state code
  - `get_forecast(latitude=32.7767, longitude=-96.7970)` — 5-period forecast via Open-Meteo (no API key)

### Adding MCP servers later
- Create a new server script (e.g., `mcp_<domain>.py`) using `FastMCP`, exposing tools with `@mcp.tool`.
- Add any deps via `uv pip install <deps>` in this folder.
- Run it with `uv run mcp_<domain>.py` (STDIO transport).
- To connect from another client (e.g., Claude Desktop), register the server command in the client’s MCP config (see client docs).

## End-to-end flow (text diagram)
```
[data/*.txt travel docs]
      |
      v
 chunk_text (overlap) -> embed_text (OpenAI)
      |
      v
 upsert into PostgreSQL pgvector
      |
      v
 retrieve top-k for question
      |
      v
 decision_gate (Correct | Ambiguous | Incorrect)
      |           |               |
      |           |               +--> external_search (Open-Meteo weather; LLM location correction)
      |           +--> internal + external_search
      +--> internal only
      |
      v
 build prompt with context (+ conversation history)
      |
      v
 OpenAI chat completion (temperature=0) -> answer

 [Optional MCP path]
      |
      v
 mcp_weather server (tools: get_alerts, get_forecast) <-- mcp_client or other MCP host
```
