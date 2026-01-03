# RAG Projects Workspace

This workspace hosts three variants of retrieval-augmented generation (RAG) travel/demo applications, each adding more capability:

- **rag-base**: Single-turn RAG. Ingests local docs into Postgres + pgvector, retrieves top-k chunks, and answers with OpenAI chat (temperature=0). No conversation history or corrective gating.
- **rag-conversational**: Adds rolling in-memory history (last 5 turns). Retrieval still goes to pgvector, answers use recent turns plus context. No external fallback.
- **rag-corrective**: Travel-focused CRAG (Corrective RAG). Adds a decision gate (Correct/Ambiguous/Incorrect), external tool routing (Open-Meteo weather with LLM location correction and multi-city extraction), richer travel docs, and optional MCP weather server/client. Ambiguous/Incorrect retrieval triggers external data; Correct stays internal.

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
```

## Key differences
- **Memory**: Base has none; Conversational keeps last 5 turns; Corrective keeps last 5 turns.
- **Gating**: Base/Conversational always use internal retrieval; Corrective uses a grader to gate external fallback.
- **External**: Base/Conversational have no external calls; Corrective routes to external weather tools (multi-city, LLM-corrected) and includes optional MCP server/client for tool-style use.
- **Docs**: Base/Conversational ship with smaller sets; Corrective includes broader travel guidance corpus.

## Quick start
- Each project has its own README with setup/run steps (uv venv, dependencies, Postgres + pgvector, OpenAI key). Start in the desired folder and follow its README.
