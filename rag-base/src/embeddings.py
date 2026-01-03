from typing import List

from openai import OpenAI

from src.config import Settings


def embed_text(settings: Settings, text: str) -> List[float]:
    """Return an embedding vector for the given text."""
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        input=text,
        model=settings.embed_model,
    )
    return response.data[0].embedding
