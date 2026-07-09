from typing import List, Dict, Optional, Tuple
from chromadb import PersistentClient
from openai import OpenAI
from rank_bm25 import BM25Okapi  # new
from .config import get_settings
from .models import DocumentChunk, ChatMessage
from .local_embeddings import LocalEmbeddingFunction

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

# def _build_bm25_index() -> Tuple[BM25Okapi, List[Dict], List[str]]:
#     """
#     Build BM25 index from all docs in the Chroma collection.
#     Returns:
#       - BM25Okapi instance
#       - metadatas list (aligned with corpus)
#       - documents list (aligned with corpus)
#     """
#     all_docs = _collection.get(include=["documents", "metadatas"])
#     documents = all_docs["documents"]
#     metadatas = all_docs["metadatas"]

#     # Tokenize each document for BM25
#     tokenized_corpus = []
#     for doc in documents:
#         # You can use a very simple tokenizer if you don't want nltk:
#         # tokens = doc.lower().split()
#         tokens = word_tokenize(doc.lower())
#         tokenized_corpus.append(tokens)

#     bm25 = BM25Okapi(tokenized_corpus)
#     return bm25, metadatas, documents


# _bm25_index, _bm25_metadatas, _bm25_docs = _build_bm25_index()

try:
    from nltk.tokenize import word_tokenize
except ImportError:
    def word_tokenize(text: str):
        return text.split()


# BM25 globals; initialized lazily
_bm25_index: BM25Okapi | None = None
_bm25_metadatas: list[dict] = []
_bm25_docs: list[str] = []

def _build_bm25_index() -> None:
    """
    Build BM25 index from all docs in the Chroma collection.
    Does nothing if there are no documents yet.
    """
    global _bm25_index, _bm25_metadatas, _bm25_docs

    all_docs = _collection.get(include=["documents", "metadatas"])
    documents = all_docs.get("documents") or []
    metadatas = all_docs.get("metadatas") or []

    if not documents:
        # No docs yet; leave BM25 index as None
        _bm25_index = None
        _bm25_metadatas = []
        _bm25_docs = []
        return

    tokenized_corpus = []
    for doc in documents:
        tokens = word_tokenize(doc.lower())
        tokenized_corpus.append(tokens)

    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_metadatas = metadatas
    _bm25_docs = documents


def ensure_bm25_index_built() -> None:
    global _bm25_index
    if _bm25_index is None:
        _build_bm25_index()

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


# def _bm25_retrieve(
#     query: str,
#     n_results: int,
# ) -> List[DocumentChunk]:
#     """
#     BM25 lexical retrieval over the full corpus.
#     """
#     if not _bm25_docs:
#         return []

#     # Tokenize query
#     tokens = word_tokenize(query.lower())
#     scores = _bm25_index.get_scores(tokens)

#     # Take top n_results indices by score
#     indexed_scores = list(enumerate(scores))
#     indexed_scores.sort(key=lambda x: x[1], reverse=True)
#     top_indices = indexed_scores[:n_results]

#     results: List[DocumentChunk] = []
#     for idx, score in top_indices:
#         results.append(
#             DocumentChunk(
#                 id=str(idx),
#                 text=_bm25_docs[idx],
#                 score=float(score),
#                 metadata=_bm25_metadatas[idx] or {},
#             )
#         )
#     return results

def _bm25_retrieve(
    query: str,
    n_results: int,
) -> List[DocumentChunk]:
    """
    BM25 lexical retrieval over the full corpus.
    Returns an empty list if there are no documents.
    """
    ensure_bm25_index_built()

    if _bm25_index is None or not _bm25_docs:
        return []

    tokens = word_tokenize(query.lower())
    scores = _bm25_index.get_scores(tokens)

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
):
    """
    Build a prompt from the context and query, call the LLM, and return
    both the answer text and the raw completion object (for usage).
    """
    prompt = build_prompt(query, context_chunks, max_context_tokens=max_context_tokens)

    completion = _llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    answer_text = completion.choices[0].message.content.strip()
    return answer_text, completion


# def rag_query(
#     query: str,
#     top_k: int = 5,
#     where: Optional[Dict] = None,
#     retrieval_strategy: Optional[str] = None,
# ):
#     """
#     Full RAG pipeline:
#     - choose retrieval strategy (dense vs hybrid)
#     - rerank
#     - trim to top_k
#     - enforce context token budget
#     - generate answer
#     """

#     # Decide retrieval mode; default to hybrid to match your current behavior
#     strategy = retrieval_strategy or "hybrid"

#     if strategy == "dense":
#         # use only dense retrieval (no BM25)
#         candidates = _dense_retrieve(
#             query, n_results=max(top_k * 3, top_k), where=where
#         )
#     else:
#         # default: hybrid retrieval
#         candidates = retrieve_hybrid(
#             query,
#             top_k=top_k,
#             where=where,
#             overfetch_factor=3,
#             w_dense=0.7,
#             w_bm25=0.3,
#         )

#     # Rerank
#     reranked = rerank_chunks(query, candidates)

#     # Take top_k
#     final_chunks = reranked[:top_k]

#     # Context token budget
#     max_context_tokens = getattr(settings, "max_context_tokens", None)

#     # Generate answer
#     answer = generate_answer(query, final_chunks, max_context_tokens=max_context_tokens)
#     return answer, final_chunks


def rag_query(
    query: str,
    top_k: int = 5,
    where: Optional[Dict] = None,
    retrieval_strategy: Optional[str] = None,
):
    """
    Full RAG pipeline:
    - choose retrieval strategy (dense vs hybrid)
    - rerank
    - trim to top_k
    - enforce context token budget
    - generate answer
    - compute token usage and cost
    """

    # Decide retrieval mode; default to hybrid
    strategy = retrieval_strategy or "hybrid"

    if strategy == "dense":
        # use only dense retrieval (no BM25)
        candidates = _dense_retrieve(
            query, n_results=max(top_k * 3, top_k), where=where
        )
    else:
        # default: hybrid retrieval
        candidates = retrieve_hybrid(
            query,
            top_k=top_k,
            where=where,
            overfetch_factor=3,
            w_dense=0.7,
            w_bm25=0.3,
        )

    # Rerank
    reranked = rerank_chunks(query, candidates)

    # Take top_k
    final_chunks = reranked[:top_k]

    # Context token budget
    max_context_tokens = getattr(settings, "max_context_tokens", None)

    # Generate answer (get both text and completion for usage)
    answer, completion = generate_answer(
        query, final_chunks, max_context_tokens=max_context_tokens
    )

    # ---- Token usage and cost ----
    usage = getattr(completion, "usage", None)
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = prompt_tokens + completion_tokens

    # Example pricing: replace with your actual price per 1k tokens
    PRICE_PER_1K = 0.59  # USD per 1k tokens (example only)
    cost_usd = (total_tokens / 1000.0) * PRICE_PER_1K

    usage_info = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }

    return answer, final_chunks, usage_info

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


def decontextualize_query(query: str, history: List[ChatMessage]) -> str:
    """
    Given a user's latest query and the chat history, rewrite the query to be a standalone,
    self-contained question suitable for RAG retrieval.
    """
    if not history:
        return query

    history_lines = []
    for msg in history:
        role_label = "User" if msg.role == "user" else "Assistant"
        history_lines.append(f"{role_label}: {msg.content}")
    history_block = "\n".join(history_lines)

    rewrite_prompt = f"""
You are an AI assistant. Given the following conversation history and the user's latest follow-up question, your task is to rewrite the follow-up question into a standalone, fully self-contained question (in English). 

The standalone question should include all necessary context from the conversation history so that it can be answered correctly via a document search without needing the conversation history.

Rules:
1. Do NOT answer the question.
2. Return ONLY the rewritten standalone question.
3. If the user's query is already a standalone question or does not rely on history, return the original query exactly.

Conversation History:
{history_block}

User's Follow-up Question:
{query}

Standalone Question:
"""

    try:
        completion = _llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        rewritten = completion.choices[0].message.content.strip()
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        return rewritten if rewritten else query
    except Exception as e:
        print(f"Error rewriting query: {e}")
        return query


def rag_query_stream(
    query: str,
    history: Optional[List[ChatMessage]] = None,
    top_k: int = 5,
    where: Optional[Dict] = None,
    retrieval_strategy: Optional[str] = None,
):
    """
    Generator for the RAG pipeline.
    Yields stages:
    1. {"type": "context", "context": [...]}
    2. {"type": "token", "content": "..."}
    3. {"type": "usage", "prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ..., "cost_usd": ...}
    """
    history = history or []
    search_query = decontextualize_query(query, history)

    strategy = retrieval_strategy or "hybrid"
    if strategy == "dense":
        candidates = _dense_retrieve(
            search_query, n_results=max(top_k * 3, top_k), where=where
        )
    else:
        candidates = retrieve_hybrid(
            search_query,
            top_k=top_k,
            where=where,
            overfetch_factor=3,
            w_dense=0.7,
            w_bm25=0.3,
        )

    reranked = rerank_chunks(search_query, candidates)
    final_chunks = reranked[:top_k]

    yield {
        "type": "context",
        "context": [c.dict() for c in final_chunks]
    }

    max_context_tokens = getattr(settings, "max_context_tokens", None)
    prompt = build_prompt(search_query, final_chunks, max_context_tokens=max_context_tokens)

    stream = _llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
        stream=True,
        stream_options={"include_usage": True}
    )

    full_answer = []
    usage_info = None

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        token = delta.content if delta else None
        if token:
            full_answer.append(token)
            yield {
                "type": "token",
                "content": token
            }
        if hasattr(chunk, "usage") and chunk.usage is not None:
            usage_info = chunk.usage

    if usage_info:
        prompt_tokens = usage_info.prompt_tokens
        completion_tokens = usage_info.completion_tokens
    else:
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens("".join(full_answer))

    total_tokens = prompt_tokens + completion_tokens
    PRICE_PER_1K = 0.59
    cost_usd = (total_tokens / 1000.0) * PRICE_PER_1K

    yield {
        "type": "usage",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd
    }