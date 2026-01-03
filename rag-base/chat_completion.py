"""
Simple RAG helpers for chat with pgvector-backed context.

The heavy lifting lives in src/rag_pipeline.py; this module exposes thin wrappers
for ingestion and querying so other scripts can import them easily.
"""

from src.rag_pipeline import RAGPipeline, build_pipeline


def ingest_documents(pipeline: RAGPipeline | None = None) -> None:
    pipe = pipeline or build_pipeline()
    pipe.ingest()


def answer_with_context(
    question: str, pipeline: RAGPipeline | None = None, k: int = 3
) -> str:
    pipe = pipeline or build_pipeline()
    return pipe.answer(question, k=k)

if __name__ == "__main__":
    print("Chat started. Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        try:
            reply = create_chat_completion(user_input)
            print(f"Assistant: {reply}\n")
        except Exception as e:
            print(f"Error: {e}")
