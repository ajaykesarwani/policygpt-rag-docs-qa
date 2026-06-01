from typing import List, Dict, Optional, Tuple

from chromadb import PersistentClient
from openai import OpenAI
from rank_bm25 import BM25Okapi  # new
from .config import get_settings
from .models import DocumentChunk
from .local_embeddings import LocalEmbeddingFunction
from nltk.tokenize import word_tokenize  # or a simple split if you prefer

settings = get_settings()

# Groq via OpenAI-compatible API
_llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=settings.groq_api_key,
)

# Global Chroma client, embedding function, and collection
_client = PersistentClient(path=settings.chroma_dir)
_embedding_fn = LocalEmbeddingFunction()
_collection = _client.get_or_create_collection(
    name="documents",
    embedding_function=_embedding_fn,
)

# ---- BM25 corpus setup ----
# We build a BM25 index over all docs currently in the collection.
# For a more advanced system, you'd keep this in sync with ingestion.

def _build_bm25_index() -> Tuple[BM25Okapi, List[Dict], List[str]]:
    """
    Build BM25 index from all docs in the Chroma collection.
    Returns:
      - BM25Okapi instance
      - metadatas list (aligned with corpus)
      - documents list (aligned with corpus)
    """
    all_docs = _collection.get(include=["documents", "metadatas"])
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    # Tokenize each document for BM25
    tokenized_corpus = []
    for doc in documents:
        # You can use a very simple tokenizer if you don't want nltk:
        # tokens = doc.lower().split()
        tokens = word_tokenize(doc.lower())
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, metadatas, documents


_bm25_index, _bm25_metadatas, _bm25_docs = _build_bm25_index()


def get_collection():
    return _collection


def _dense_retrieve(
    query: str,
    n_results: int,
    where: Optional[Dict] = None,
) -> List[DocumentChunk]:
    """
    Dense retrieval using Chroma embeddings.
    """
    collection = get_collection()

    query_kwargs = {
        "query_texts": [query],
        "n_results": n_results,
    }
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks: List[DocumentChunk] = []
    for i, text in enumerate(docs):
        chunks.append(
            DocumentChunk(
                id=str(i),
                text=text,
                # Convert distance to a similarity-like score (smaller dist => higher score)
                score=float(-distances[i]),
                metadata=metadatas[i] or {},
            )
        )
    return chunks


def _bm25_retrieve(
    query: str,
    n_results: int,
) -> List[DocumentChunk]:
    """
    BM25 lexical retrieval over the full corpus.
    """
    if not _bm25_docs:
        return []

    # Tokenize query
    tokens = word_tokenize(query.lower())
    scores = _bm25_index.get_scores(tokens)

    # Take top n_results indices by score
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = indexed_scores[:n_results]

    results: List[DocumentChunk] = []
    for idx, score in top_indices:
        results.append(
            DocumentChunk(
                id=str(idx),
                text=_bm25_docs[idx],
                score=float(score),
                metadata=_bm25_metadatas[idx] or {},
            )
        )
    return results


def retrieve_hybrid(
    query: str,
    top_k: int,
    where: Optional[Dict] = None,
    overfetch_factor: int = 3,
    w_dense: float = 0.7,
    w_bm25: float = 0.3,
) -> List[DocumentChunk]:
    """
    Hybrid retrieval: combine dense (Chroma) and BM25 search.
    - overfetch_factor controls how many candidates we get before merging.
    - w_dense / w_bm25 control the weighting.
    """
    n_candidates = max(top_k * overfetch_factor, top_k)

    dense_results = _dense_retrieve(query, n_results=n_candidates, where=where)
    bm25_results = _bm25_retrieve(query, n_results=n_candidates)

    # Merge by text + metadata key; accumulate hybrid score
    def key_for(c: DocumentChunk) -> Tuple[str, str]:
        # Use text + source (if present) as a simple key
        return c.text, str(c.metadata.get("source", ""))

    combined: Dict[Tuple[str, str], Dict] = {}

    for c in dense_results:
        k = key_for(c)
        combined.setdefault(k, {"chunk": c, "dense_score": 0.0, "bm25_score": 0.0})
        combined[k]["dense_score"] = max(combined[k]["dense_score"], c.score)

    for c in bm25_results:
        k = key_for(c)
        combined.setdefault(k, {"chunk": c, "dense_score": 0.0, "bm25_score": 0.0})
        combined[k]["bm25_score"] = max(combined[k]["bm25_score"], c.score)

    merged: List[Tuple[float, DocumentChunk]] = []
    for item in combined.values():
        dense_score = item["dense_score"]
        bm25_score = item["bm25_score"]
        hybrid_score = w_dense * dense_score + w_bm25 * bm25_score
        # update chunk.score with hybrid score
        chunk = item["chunk"]
        chunk.score = float(hybrid_score)
        merged.append((hybrid_score, chunk))

    # Sort by hybrid score descending
    merged.sort(key=lambda x: x[0], reverse=True)
    # Take top_k
    return [c for _, c in merged[:top_k * overfetch_factor]]


def estimate_tokens(text: str) -> int:
    """
    Very rough token estimator: ~4 characters per token.
    """
    return max(1, len(text) // 4)


def rerank_chunks(query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """
    Simple LLM-based reranking:
    - Ask the LLM to score each chunk 0-10 for relevance.
    - Sort chunks by that score (descending).
    """

    if not chunks:
        return chunks

    chunk_texts = []
    for idx, c in enumerate(chunks):
        source = c.metadata.get("source", "unknown")
        chunk_texts.append(f"Chunk {idx} (source={source}):\n{c.text}\n")

    chunks_block = "\n\n".join(chunk_texts)

    scoring_prompt = f"""
You are helping to rank context chunks for a retrieval-augmented QA system.

User question:
{query}

Below are some candidate chunks. For each chunk, assign a relevance score between 0 and 10
(10 = extremely relevant to answering the question, 0 = totally irrelevant).

Return your answer as a JSON object with this format:
{{"scores": [s0, s1, s2, ...]}}

Where s0 is the score for Chunk 0, s1 for Chunk 1, etc.

Chunks:
{chunks_block}
"""

    completion = _llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": scoring_prompt}],
        temperature=0.0,
        max_tokens=256,
    )

    import json

    raw = completion.choices[0].message.content.strip()
    scores = []
    try:
        data = json.loads(raw)
        scores = data.get("scores", [])
    except Exception:
        return chunks

    paired: List[Tuple[float, DocumentChunk]] = []
    for i, c in enumerate(chunks):
        if i < len(scores):
            s = float(scores[i])
        else:
            s = 0.0
        paired.append((s, c))

    paired.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in paired]


def build_prompt(
    query: str,
    context_chunks: List[DocumentChunk],
    max_context_tokens: Optional[int] = None,
) -> str:
    context_texts = []
    used_tokens = 0

    for c in context_chunks:
        source = c.metadata.get("source", "unknown")
        chunk_text = f"[source={source}] {c.text}"

        if max_context_tokens is not None:
            chunk_tokens = estimate_tokens(chunk_text)
            if used_tokens + chunk_tokens > max_context_tokens:
                break
            used_tokens += chunk_tokens

        context_texts.append(chunk_text)

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


def generate_answer(
    query: str,
    context_chunks: List[DocumentChunk],
    max_context_tokens: Optional[int] = None,
) -> str:
    prompt = build_prompt(query, context_chunks, max_context_tokens=max_context_tokens)

    completion = _llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return completion.choices[0].message.content.strip()


def rag_query(
    query: str,
    top_k: int = 5,
    where: Optional[Dict] = None,
):
    """
    Full RAG pipeline:
    - hybrid retrieve (dense + BM25)
    - rerank
    - trim to top_k
    - enforce context token budget
    - generate answer
    """
    # Step 1: hybrid retrieval
    candidates = retrieve_hybrid(
        query,
        top_k=top_k,
        where=where,
        overfetch_factor=3,
        w_dense=0.7,
        w_bm25=0.3,
    )

    # Step 2: rerank
    reranked = rerank_chunks(query, candidates)

    # Step 3: take top_k after reranking
    final_chunks = reranked[:top_k]

    # Step 4: context token budget
    max_context_tokens = getattr(settings, "max_context_tokens", None)

    # Step 5: generate answer
    answer = generate_answer(query, final_chunks, max_context_tokens=max_context_tokens)
    return answer, final_chunks

def reset_vector_store():
    """
    Delete and recreate the 'documents' collection,
    and update the global _collection reference.
    """
    global _collection

    # Delete existing collection (if it exists)
    _client.delete_collection(name="documents")

    # Recreate empty collection with the same embedding function
    _collection = _client.get_or_create_collection(
        name="documents",
        embedding_function=_embedding_fn,
    )