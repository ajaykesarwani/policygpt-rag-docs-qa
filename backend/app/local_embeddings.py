from typing import List
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer


class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Loads model once; Chroma will reuse this instance
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Returns list of embedding vectors
        embeddings = self.model.encode(input, convert_to_numpy=False)
        return [emb.tolist() for emb in embeddings]