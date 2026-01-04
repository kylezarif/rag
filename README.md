# RAG Projects Workspace

This workspace hosts four variants of retrieval-augmented generation (RAG) travel/demo applications, each adding more capability and different routing strategies:

- **rag-base**: Single-turn RAG. Ingests local docs into Postgres + pgvector, retrieves top-k chunks, and answers with OpenAI chat (temperature=0). No conversation history or corrective gating.
- **rag-conversational**: Adds rolling in-memory history (last 5 turns). Retrieval still goes to pgvector, answers use recent turns plus context. No external fallback.
- **rag-corrective**: Travel-focused CRAG (Corrective RAG). Adds a decision gate (Correct/Ambiguous/Incorrect), external tool routing (Open-Meteo weather with LLM location correction and multi-city extraction), richer travel docs, and optional MCP weather server/client. Ambiguous/Incorrect retrieval triggers external data; Correct stays internal.
- **rag-adoptive**: Adaptive RAG with an LLM router (direct | rag | agent). The classifier (_classify in `src/rag_pipeline.py`) uses the LLM (temperature=0) plus recent conversation history to pick:
  - direct: skip retrieval for greetings/chit-chat/simple known facts
  - rag: single-pass retrieval from pgvector (+ external weather when relevant)
  - agent: multi-source for complex/multi-city planning (retrieval + external weather, step-by-step prompt)
  Uses the expanded travel doc set and keeps a rolling 5-turn history.

## Layout (simplified)
```
rag-base/
  rag-base.py              # CLI entry
  src/                     # config, data_loader, db, embeddings, rag_pipeline
  data/                    # renovation/travel docs for base RAG

rag-conversational/
  rag-conversational.py    # CLI entry with chat loop
  src/                     # config, data_loader, db, embeddings, conversation, rag_pipeline
  data/                    # sample docs

rag-corrective/
  rag-corrective.py        # CLI entry with chat loop + CRAG gate
  src/                     # config, data_loader, db, embeddings, conversation,
                           # decision_gate (grader), external_search (tool router), rag_pipeline
  data/                    # expanded travel docs (USA, Europe, Asia, packing, safety, etc.)
  mcp_weather.py           # MCP weather server (get_forecast, get_alerts)
  mcp_client.py            # Minimal MCP client

rag-adoptive/
  rag-adoptive.py          # CLI entry with adaptive router
  src/                     # config, data_loader, db, embeddings, conversation,
                           # external_search (tool router), rag_pipeline (LLM classifier: direct|rag|agent)
  data/                    # expanded travel docs
  mcp_weather.py           # MCP weather server (get_forecast, get_alerts)
  mcp_client.py            # Minimal MCP client
```

## Key differences
- **Memory**: Base has none; Conversational keeps last 5 turns; Corrective keeps last 5 turns; Adoptive keeps last 5 turns.
- **Gating/Routing**:
  - Base/Conversational: always internal retrieval.
  - Corrective: grader (Correct/Ambiguous/Incorrect) gates external fallback.
  - Adoptive: LLM classifier in `rag-adoptive/src/rag_pipeline.py` chooses direct | rag | agent using the latest history.
- **External**:
  - Base/Conversational: no external calls.
  - Corrective: external weather tool (multi-city, LLM-corrected) + optional MCP server/client.
  - Adoptive: external weather via tool router for rag/agent paths; direct path skips retrieval and external calls.
- **Docs**: Base/Conversational ship with smaller sets; Corrective/Adoptive include broader travel guidance.

## Quick start
- Each project has its own README with setup/run steps (uv venv, dependencies, Postgres + pgvector, OpenAI key). Start in the desired folder and follow its README.
