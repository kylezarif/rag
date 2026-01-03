#!/usr/bin/env python3
import argparse
import sys

from chat_completion import answer_with_context, ingest_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple RAG flow against renovation guidelines stored in pgvector."
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
        help="Skip re-ingesting local documents (assumes they are already in the database).",
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
    if not args.skip_ingest:
        print("Ingesting renovation guideline documents...")
        ingest_documents()
    print(f"Asking: {args.question}")
    answer = answer_with_context(args.question, k=args.top_k)
    print("\nAnswer:\n")
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
