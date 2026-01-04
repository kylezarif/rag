from pathlib import Path
from typing import List, Tuple


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into word-based chunks with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_documents(data_dir: Path, chunk_size: int, overlap: int) -> List[Tuple[str, str]]:
    """Load .txt documents as chunked (title, content)."""
    documents: List[Tuple[str, str]] = []
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    for path in sorted(data_dir.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        for idx, chunk in enumerate(
            chunk_text(content, chunk_size=chunk_size, overlap=overlap), start=1
        ):
            documents.append((f"{path.stem}-chunk-{idx}", chunk))
    if not documents:
        raise ValueError(f"No .txt documents found in {data_dir}")
    return documents
