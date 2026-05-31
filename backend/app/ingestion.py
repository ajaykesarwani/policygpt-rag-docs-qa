from typing import List, Dict
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from .config import get_settings
from .local_embeddings import LocalEmbeddingFunction

def get_chroma_collection():
    settings = get_settings()
    client = PersistentClient(path=settings.chroma_dir)
    # collection = client.get_or_create_collection(
    #     name="documents",
    #     embedding_function=OpenAIEmbeddingFunction(
    #         api_key=settings.openai_api_key,
    #         model_name=settings.embedding_model,
    #     ),
    # )
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=LocalEmbeddingFunction(
            model_name=settings.embedding_model
        ),
    )
    return collection


def split_texts(texts: List[str]) -> List[str]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
    )
    all_chunks = []
    for t in texts:
        all_chunks.extend(splitter.split_text(t))
    return all_chunks


def ingest_texts(source_name: str, texts: List[str], base_metadata: Dict):
    collection = get_chroma_collection()
    chunks = split_texts(texts)

    ids = []
    metadatas = []
    for idx, c in enumerate(chunks):
        ids.append(str(uuid4()))
        md = {"source": source_name, "chunk_index": idx}
        md.update(base_metadata or {})
        metadatas.append(md)

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    return {"ingested_chunks": len(chunks)}