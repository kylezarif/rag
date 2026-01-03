"""
Thin wrappers around the conversational RAG pipeline.
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
