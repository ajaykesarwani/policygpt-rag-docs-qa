from typing import List
from chromadb.utils.embedding_functions import EmbeddingFunction
from groq import Groq
from .config import get_settings


class GroqEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str | None = None):
        settings = get_settings()
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Groq is OpenAI-compatible; we call embeddings.create with the same schema.
        response = self.client.embeddings.create(
            model=self.model,
            input=input,
        )
        return [item.embedding for item in response.data]