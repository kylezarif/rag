from pathlib import Path
from typing import List, Tuple


def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    """
    Load documents from the data directory.

    Returns a list of (title, content) tuples, using filenames as titles.
    """
    documents: List[Tuple[str, str]] = []
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for path in sorted(data_dir.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append((path.stem.replace("_", " ").title(), content))
    if not documents:
        raise ValueError(f"No .txt documents found in {data_dir}")
    return documents
