from textwrap import dedent
from typing import List, Optional, Tuple

from openai import OpenAI

from src import db
from src.config import Settings, load_settings
from src.conversation import ConversationHistory
from src.data_loader import load_documents
from src.embeddings import embed_text
from src.external_search import external_search


class RAGPipeline:
    def __init__(self, settings: Settings, history: Optional[ConversationHistory] = None):
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.history = history or ConversationHistory(max_turns=settings.history_size)

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
        route = self._classify(question)
        if route == "direct":
            answer = self._direct_answer(question)
        elif route == "agent":
            answer = self._agent_answer(question, k=k)
        else:
            answer = self._rag_answer(question, k=k)
        self.history.add_turn(question, answer)
        return answer

    def _classify(self, question: str) -> str:
        """
        Route: direct (no retrieval), rag (single-pass), agent (multi-source).
        """
        history_text = "\n".join(
            f"User: {u}\nAssistant: {a}" for u, a in self.history._messages  # type: ignore[attr-defined]
        )
        prompt = dedent(
            f"""
            Choose the best path for this travel query. Respond with one word: direct, rag, or agent.
            - direct: greetings, chit-chat, or general knowledge likely known to the model.
            - rag: simple factual travel lookups answerable from internal travel docs or a single weather check.
            - agent: multi-step planning/analysis (multi-city trips, comparisons, multi-day itineraries).

            Recent turns:
            {history_text}

            Query: {question}
            """
        ).strip()
        try:
            resp = self.client.chat.completions.create(
                model=self.settings.chat_model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            label = (resp.choices[0].message.content or "").strip().lower()
            if any(x in label for x in ["agent"]):
                return "agent"
            if any(x in label for x in ["rag", "retrieval"]):
                return "rag"
            return "direct"
        except Exception:
            return "rag"

    def _direct_answer(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a concise travel assistant. Answer briefly without retrieval.",
            },
        ]
        messages.extend(self.history.to_messages())
        messages.append({"role": "user", "content": question})
        resp = self.client.chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            messages=messages,
        )
        return resp.choices[0].message.content

    def _rag_answer(self, question: str, k: int) -> str:
        internal = self.retrieve(question, k=k)
        external = external_search(question, self.settings)
        contexts, source = self._merge_contexts(internal, external)
        prompt = self._build_prompt(question, contexts, source, route="rag")
        return self._chat(prompt)

    def _agent_answer(self, question: str, k: int) -> str:
        internal = self.retrieve(question, k=max(k, 4))
        external = external_search(question, self.settings)
        contexts, source = self._merge_contexts(internal, external)
        prompt = self._build_prompt(
            question,
            contexts,
            source,
            route="agent",
            agent_instructions="Plan or compare step-by-step. Use all relevant contexts. If something is missing, note it.",
        )
        return self._chat(prompt)

    def _merge_contexts(self, internal: List[str], external: List[str]) -> Tuple[List[str], str]:
        if internal and external:
            return internal + external, "mixed"
        if internal:
            return internal, "internal"
        if external:
            return external, "external"
        return [], "none"

    def _chat(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a travel assistant. ALWAYS ground your answer in provided context blocks. "
                    "Treat External API context as current information. If context is insufficient, say so."
                ),
            }
        ]
        messages.extend(self.history.to_messages())
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            messages=messages,
        )
        return resp.choices[0].message.content

    @staticmethod
    def _build_prompt(
        question: str,
        contexts: List[str],
        source: str,
        route: str,
        agent_instructions: str = "",
    ) -> str:
        context_block = "\n\n".join(
            f"Context {idx+1} ({source}):\n{chunk}" for idx, chunk in enumerate(contexts)
        )
        return dedent(
            f"""
            Route selected: {route.upper()}
            {agent_instructions}

            Use the context below to answer the question.

            {context_block}

            Question: {question}
            """
        ).strip()


def build_pipeline() -> RAGPipeline:
    settings = load_settings()
    return RAGPipeline(settings)
