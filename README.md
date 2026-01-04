# RAG Projects Workspace

This workspace hosts five variants of retrieval-augmented generation (RAG) travel/demo applications, each adding more capability and different routing strategies:

- **rag-base**: Single-turn RAG. Ingests local docs into Postgres + pgvector, retrieves top-k chunks, and answers with OpenAI chat (temperature=0). No conversation history or corrective gating.
- **rag-conversational**: Adds rolling in-memory history (last 5 turns). Retrieval still goes to pgvector, answers use recent turns plus context. No external fallback.
- **rag-corrective**: Travel-focused CRAG (Corrective RAG). Adds a decision gate (Correct/Ambiguous/Incorrect), external tool routing (Open-Meteo weather with LLM location correction and multi-city extraction), richer travel docs, and optional MCP weather server/client. Ambiguous/Incorrect retrieval triggers external data; Correct stays internal.
- **rag-adoptive**: Adaptive RAG with an LLM router (direct | rag | agent). The classifier (_classify in `src/rag_pipeline.py`) uses the LLM (temperature=0) plus recent conversation history to pick:
  - direct: skip retrieval for greetings/chit-chat/simple known facts
  - rag: single-pass retrieval from pgvector (+ external weather when relevant)
  - agent: multi-source for complex/multi-city planning (retrieval + external weather, step-by-step prompt)
  Uses the expanded travel doc set and keeps a rolling 5-turn history.
- **rag-agentic**: Agentic RAG rebuilt with LangGraph. ChatOpenAI (via langchain_openai) is bound to tools (vector_search over pgvector, weather_lookup via Open-Meteo). The graph loops llm_call → tool_node until no tool calls, then responds using history for context. Extend tools in `src/tools.py` and bindings in `src/rag_pipeline.py`.

## Layout (simplified)
```
rag-base/
  rag-base.py              # CLI entry
  src/                     # config, data_loader, db, embeddings, rag_pipeline
  data/                    # shared travel docs (copied to other projects)

rag-conversational/
  rag-conversational.py    # CLI entry with chat loop
  src/                     # config, data_loader, db, embeddings, conversation, rag_pipeline
  data/                    # shared travel docs

rag-corrective/
  rag-corrective.py        # CLI entry with chat loop + CRAG gate
  src/                     # config, data_loader, db, embeddings, conversation,
                           # decision_gate (grader), external_search (tool router), rag_pipeline
  data/                    # shared travel docs (USA, Europe, Asia, packing, safety, etc.)
  mcp_weather.py           # MCP weather server (get_forecast, get_alerts)
  mcp_client.py            # Minimal MCP client

rag-adoptive/
  rag-adoptive.py          # CLI entry with adaptive router
  src/                     # config, data_loader, db, embeddings, conversation,
                           # external_search (tool router), rag_pipeline (LLM classifier: direct|rag|agent)
  data/                    # expanded travel docs
  mcp_weather.py           # MCP weather server (get_forecast, get_alerts)
  mcp_client.py            # Minimal MCP client

rag-agentic/
  rag-agentic.py           # CLI entry with agentic plan/act loop
  src/                     # config, data_loader, db, embeddings, conversation,
                           # external_search (tool router), tools (tool runner), rag_pipeline (agent plan→act→answer)
  data/                    # travel docs
```

## Key differences
- **Memory**: Base has none; Conversational keeps last 5 turns; Corrective keeps last 5 turns; Adoptive keeps last 5 turns.
- **Gating/Routing**:
  - Base/Conversational: always internal retrieval.
  - Corrective: grader (Correct/Ambiguous/Incorrect) gates external fallback.
  - Adoptive: LLM classifier in `rag-adoptive/src/rag_pipeline.py` chooses direct | rag | agent using the latest history.
  - Agentic: LLM plans steps, then runs vector search + external tools per plan.
- **External**:
  - Base/Conversational: no external calls.
  - Corrective: external weather tool (multi-city, LLM-corrected) + optional MCP server/client.
  - Adoptive: external weather via tool router for rag/agent paths; direct path skips retrieval and external calls.
  - Agentic: external weather/tool calls invoked explicitly during the agent’s action steps.
- **Docs**: All projects share the same travel guidance corpus (USA, Europe, Asia, packing, safety, insurance, family, nomad, winter/heat, etc.).

## Quick start
- Each project has its own README with setup/run steps (uv venv, dependencies, Postgres + pgvector, OpenAI key). Start in the desired folder and follow its README.

## Choosing and designing a pipeline
- **Base**: Use when you need the simplest, fastest single-turn RAG (no memory, no external calls). Good for small, static corpora and low latency Q&A.
- **Conversational**: Use when short conversational context (last 5 turns) matters but you still want a lean, deterministic RAG without external calls.
- **Corrective (CRAG)**: Use when you want a guardrail that refuses to answer from weak retrieval. It grades hits (Correct/Ambiguous/Incorrect) and falls back to external tools (e.g., weather). Good for reducing hallucinations when internal coverage is spotty.
- **Adoptive (Adaptive)**: Use when you want cost/latency-aware routing. An LLM classifier picks direct (no retrieval), rag, or agent paths based on complexity + history. Ideal when many queries are simple but some need retrieval or light tool use.
- **Agentic (LangGraph)**: Use when you want structured plan/act loops and tool calling (vector search + weather, extensible). Better for complex/multi-step travel planning and scenarios that benefit from iterative tool use. Higher overhead than adaptive for simple queries.
- **MCP servers**: Use when you need to expose external systems as tools to MCP-capable clients (e.g., Claude Desktop) or want a tool surface decoupled from app code. Good for integrating weather/alerts/search as external tools; optional for these apps.

Design tips:
- Favor the simplest pipeline that meets the reliability requirement; add routing/agentic behavior only when needed.
- Keep prompts deterministic (temperature=0) when grounding on retrieved/tool contexts.
- Use chunking with overlap for long docs; keep table names distinct per project to avoid collisions.
- Extend `external_search`/tool registries incrementally; start with weather, add traffic/search as needed.
