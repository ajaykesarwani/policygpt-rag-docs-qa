import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = Field("local", env="APP_ENV")
    groq_api_key: str | None = Field(None, env="GROQ_API_KEY")
    # openai_api_key: str | None = Field(None, env="OPENAI_API_KEY")
    # embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )
    llm_model: str = Field("llama-3.3-70b-versatile", env="LLM_MODEL")
    chroma_dir: str = Field("chroma_index", env="CHROMA_DIR")
    max_context_tokens: int = Field(3000, env="MAX_CONTEXT_TOKENS")
    chunk_size: int = Field(800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(150, env="CHUNK_OVERLAP")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    def get_embedding_api_key(self) -> str:
        if self.openai_api_key:
            return self.openai_api_key
        if self.groq_api_key:
            return self.groq_api_key
        raise ValueError("OPENAI_API_KEY or GROQ_API_KEY must be set to create embeddings")


@lru_cache
def get_settings() -> Settings:
    return Settings()