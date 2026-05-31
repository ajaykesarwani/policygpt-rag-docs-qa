from typing import List

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb import PersistentClient
from openai import OpenAI

from .config import get_settings
from .models import DocumentChunk
from .local_embeddings import LocalEmbeddingFunction

settings = get_settings()
# Groq via OpenAI-compatible API
_llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=settings.groq_api_key,
)

def get_collection():
    client = PersistentClient(path=settings.chroma_dir)
    # embedding_fn = OpenAIEmbeddingFunction(
    #     api_key=settings.openai_api_key,
    #     model_name=settings.embedding_model,
    # )
    embedding_fn = LocalEmbeddingFunction(
        model_name=settings.embedding_model
    )
    return client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn,
    )

def retrieve(query: str, top_k: int) -> List[DocumentChunk]:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks: List[DocumentChunk] = []
    for i, text in enumerate(docs):
        chunks.append(
            DocumentChunk(
                id=str(i),
                text=text,
                score=float(distances[i]),
                metadata=metadatas[i] or {},
            )
        )
    return chunks


def build_prompt(query: str, context_chunks: List[DocumentChunk]) -> str:
    context_texts = []
    for c in context_chunks:
        context_texts.append(f"[source={c.metadata.get('source','unknown')}] {c.text}")
    context_block = "\n\n".join(context_texts)

    prompt = f"""
You are a careful assistant answering questions strictly based on the provided context.
If the answer is not in the context, say you do not know and suggest where the user might look.

Context:
{context_block}

User question:
{query}

Answer:
"""
    return prompt.strip()


def generate_answer(query: str, context_chunks: List[DocumentChunk]) -> str:
    prompt = build_prompt(query, context_chunks)
    # completion = _openai_client.chat.completions.create(
    #     model=settings.llm_model,
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.2,
    #     max_tokens=512,
    # )
    completion = _llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return completion.choices[0].message.content.strip()


def rag_query(query: str, top_k: int = 5):
    chunks = retrieve(query, top_k)
    answer = generate_answer(query, chunks)
    return answer, chunks