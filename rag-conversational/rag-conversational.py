#!/usr/bin/env python3
import argparse
import sys

from chat_completion import ingest_documents
from src.rag_pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conversational RAG demo with rolling 5-turn history."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What should I check before renovating a bathroom?",
        help="Question to ask the model.",
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
        help="Number of documents to retrieve from pgvector.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = build_pipeline()
    if not args.skip_ingest:
        print("Ingesting renovation guideline documents...")
        pipeline.ingest()

    if args.question:
        print(f"Asking: {args.question}")
        answer = pipeline.answer(args.question, k=args.top_k)
        print("\nAnswer:\n")
        print(answer)

    print("\nInteractive chat (press Enter on empty line or type 'exit' to quit):")
    while True:
        user_q = input("You: ").strip()
        if not user_q or user_q.lower() in {"exit", "quit"}:
            break
        answer = pipeline.answer(user_q, k=args.top_k)
        print(f"\nAssistant:\n{answer}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
