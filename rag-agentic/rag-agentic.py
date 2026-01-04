#!/usr/bin/env python3
import argparse
import sys

from chat_completion import ingest_documents
from src.rag_pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic RAG demo with plan/reason/act loop."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to ask the model. If omitted, you'll be prompted.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip re-ingesting local documents (assumes already in DB).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve from pgvector for vector search tool.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = build_pipeline()
    if not args.skip_ingest:
        print("Ingesting travel guideline documents...")
        ingest_documents(pipeline)

    question = args.question or input("Enter your question: ").strip()
    if not question:
        print("No question provided. Exiting.")
        return 1

    print(f"Asking: {question}")
    answer = pipeline.answer(question, k=args.top_k)
    print("\nAnswer:\n")
    print(answer)

    print("\nInteractive chat (blank line or 'exit' to quit):")
    while True:
        user_q = input("You: ").strip()
        if not user_q or user_q.lower() in {"exit", "quit"}:
            break
        reply = pipeline.answer(user_q, k=args.top_k)
        print(f"\nAssistant:\n{reply}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
