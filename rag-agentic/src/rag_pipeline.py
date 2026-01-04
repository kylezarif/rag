from textwrap import dedent
from typing import Dict, List, Optional, Tuple

from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict
import operator
from langchain_openai import ChatOpenAI

from src import db
from src.config import Settings, load_settings
from src.conversation import ConversationHistory
from src.data_loader import load_documents
from src.embeddings import embed_text
from src.external_search import external_search
from src.tools import ToolResult, run_tools


class RAGPipeline:
    def __init__(self, settings: Settings, history: Optional[ConversationHistory] = None):
        self.settings = settings
        self.history = history or ConversationHistory(max_turns=settings.history_size)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._agent = None

    def ingest(self) -> None:
        documents = load_documents(
            self.settings.data_dir,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        db.ensure_schema(self.settings)
        payload = []
        for title, content in documents:
            embedding = embed_text(self.settings, content)
            payload.append((title, content, embedding))
        db.upsert_documents(self.settings, payload)

    def retrieve(self, question: str, k: int = 3) -> List[str]:
        query_embedding = embed_text(self.settings, question)
        rows = db.fetch_similar(self.settings, query_embedding, limit=k)
        return [content for _, content, _ in rows]

    def answer(self, question: str, k: int = 3) -> str:
        agent = self._build_agent(k)
        history_msgs = self.history.to_langchain()
        messages: List[AnyMessage] = history_msgs + [HumanMessage(content=question)]
        result = agent.invoke({"messages": messages, "llm_calls": 0})
        final_messages = result["messages"]
        answer = final_messages[-1].content if final_messages else ""
        # Add last turn to history
        self.history.add_turn(question, answer)
        return answer

    def _build_agent(self, k: int):
        if self._agent:
            return self._agent

        @tool
        def vector_search(query: str) -> str:
            """Search internal travel docs."""
            rows = self.retrieve(query, k=k)
            return "\n\n".join(rows) if rows else "No internal results."

        @tool
        def weather_lookup(query: str) -> str:
            """Get weather for relevant locations."""
            results = external_search(query, self.settings)
            return "\n\n".join(results) if results else "No external weather results."

        tools = [vector_search, weather_lookup]
        tools_by_name = {t.name: t for t in tools}
        model_with_tools = self.llm.bind_tools(tools)

        class MessagesState(TypedDict):
            messages: Annotated[list[AnyMessage], operator.add]
            llm_calls: int

        def llm_call(state: dict):
            """LLM decides to call tools or answer."""
            sys_msg = SystemMessage(
                content=(
                    "You are an agentic travel assistant. Plan, reason, and use tools to gather evidence. "
                    "Cite sources from tool outputs. If insufficient info, say so."
                )
            )
            result = model_with_tools.invoke([sys_msg] + state["messages"])
            return {"messages": [result], "llm_calls": state.get("llm_calls", 0) + 1}

        def tool_node(state: dict):
            outputs = []
            for call in state["messages"][-1].tool_calls:
                tool = tools_by_name[call["name"]]
                observation = tool.invoke(call["args"])
                outputs.append(ToolMessage(content=observation, tool_call_id=call["id"]))
            return {"messages": outputs}

        def should_continue(state: dict):
            messages = state["messages"]
            last = messages[-1]
            if getattr(last, "tool_calls", None):
                return "tool_node"
            return END

        builder = StateGraph(MessagesState)
        builder.add_node("llm_call", llm_call)
        builder.add_node("tool_node", tool_node)
        builder.add_edge(START, "llm_call")
        builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
        builder.add_edge("tool_node", "llm_call")
        self._agent = builder.compile()
        return self._agent


def build_pipeline() -> RAGPipeline:
    settings = load_settings()
    return RAGPipeline(settings)
