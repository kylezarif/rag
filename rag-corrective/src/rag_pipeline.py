from textwrap import dedent
from typing import List, Optional

from openai import OpenAI

from src import db
from src.config import Settings, load_settings
from src.conversation import ConversationHistory
from src.data_loader import load_documents
from src.decision_gate import GateDecision, grade_documents
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
        internal_contexts = self.retrieve(question, k=k)
        decision = grade_documents(self.settings, question, internal_contexts)
        want_weather = any(term in question.lower() for term in ("weather", "forecast"))

        if decision == "incorrect":
            contexts = external_search(question, self.settings)
            source = "external"
        elif decision == "ambiguous":
            contexts = internal_contexts + external_search(question, self.settings)
            source = "mixed"
        else:
            contexts = internal_contexts
            source = "internal"

        if want_weather:
            ext = external_search(question, self.settings)
            if ext:
                if source == "internal":
                    contexts = contexts + ext
                    source = "mixed"
                elif source == "external":
                    contexts = ext
                else:
                    contexts = contexts + ext

        prompt = self._build_prompt(question, contexts, source=source)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a travel assistant. ALWAYS ground your answer in the provided context blocks. "
                    "If any context is marked External API, treat it as current information and use it directly. "
                    "If context is insufficient, state that explicitly."
                ),
            }
        ]
        messages.extend(self.history.to_messages())
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            messages=messages,
        )
        answer = response.choices[0].message.content
        self.history.add_turn(question, answer)
        return answer

    @staticmethod
    def _build_prompt(question: str, contexts: List[str], source: str) -> str:
        context_block = "\n\n".join(
            f"Context {idx+1} ({source}):\n{chunk}" for idx, chunk in enumerate(contexts)
        )
        return dedent(
            f"""
            Use the context below to answer the question.

            {context_block}

            Question: {question}
            """
        ).strip()


def build_pipeline() -> RAGPipeline:
    settings = load_settings()
    return RAGPipeline(settings)
