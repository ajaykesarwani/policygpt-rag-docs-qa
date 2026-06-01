# PolicyGPT – RAG Backend

FastAPI backend implementing a production-style Retrieval-Augmented Generation (RAG) pipeline over local documents, with hybrid retrieval, reranking, and evaluation.

- **Framework**: FastAPI for the HTTP API
- **Retrieval**: ChromaDB for vector storage
- **Hybrid search**: Dense vector search + BM25 lexical search (rank-bm25) combined into a hybrid score
- **Embeddings**: Local SentenceTransformers model (e.g. `all-MiniLM-L6-v2`) – no paid embedding API required
- **LLM**: Groq API (OpenAI-compatible endpoint) for fast chat completions
- **Extras**: LLM-based reranking, context window control, metadata filters, and an eval script for recall/faithfulness

---

## Endpoints

- `GET /health`  
  Health check for monitoring and deployment probes.

- `POST /upload`  
  Upload a PDF or TXT file, extract text, chunk it, and ingest into the Chroma vector index under a given `source_name`.

- `POST /ingest`  
  Ingest raw texts under a `source_name`. Useful for scripted or bulk ingestion.

- `POST /query`  
  Run the full RAG pipeline:
  - Hybrid retrieval (BM25 + dense vectors)
  - LLM-based reranking
  - Context-window controlled prompt construction
  - Answer generation with retrieved chunks returned as context.

- `POST /admin/reset`  
  Admin endpoint to delete and recreate the Chroma collection, effectively “forgetting” all uploaded documents.

---

## RAG pipeline details

The core RAG logic lives in `app/rag_pipeline.py`:

- **Hybrid retrieval**

  ```python
  dense_results = _dense_retrieve(...)
  bm25_results = _bm25_retrieve(...)
  # scores combined with weights w_dense and w_bm25
  candidates = retrieve_hybrid(...)
  ```

  Dense retrieval uses Chroma’s vector search; BM25 uses a local corpus index built via `rank-bm25`. The two scores are merged into a single hybrid score before reranking.

- **Metadata filters**

  `QueryRequest` allows optional `source_name` and `filename`. These are mapped to a Chroma `where` filter, so you can restrict queries to particular sources or files.

- **Reranking**

  `rerank_chunks` asks the LLM to assign each candidate chunk a relevance score (0–10), then sorts chunks by this LLM-derived score before building the context.

- **Context window control**

  `build_prompt` accepts a `max_context_tokens` setting and stops adding chunks once the budget is reached, approximating token count to avoid overfilling the LLM context.

---

## Local setup

```bash
cd backend
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env  # fill in GROQ_API_KEY and any overrides
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Typical `.env`:

```text
APP_ENV=local
GROQ_API_KEY=your_groq_key_here

# Embeddings & LLM
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile

# Chroma
CHROMA_DIR=chroma_index

# RAG controls
MAX_CONTEXT_TOKENS=3000
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

---

## Example usage

### Upload a PDF or TXT file

```bash
curl -X POST "http://localhost:8000/upload?source_name=company-policies" \
  -F "file=@/absolute/path/to/policy.pdf"
```

Response:

```json
{
  "status": "ok",
  "ingested_chunks": 42
}
```

### Ingest raw text directly

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source_name": "sample-docs",
    "texts": ["This is a sample policy text about vacation days."],
    "metadata": {"domain": "hr"}
  }'
```

### Query over ingested documents

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the policy say about vacation carry-over?",
    "top_k": 5
  }'
```

Simplified response:

```json
{
  "answer": "The policy allows unused vacation days to be carried over for one year, subject to manager approval.",
  "context": [
    {
      "id": "0",
      "text": "Employees may carry over unused vacation days into the following calendar year with written manager approval...",
      "score": 0.87,
      "metadata": {
        "source": "company-policies",
        "filename": "policy.pdf",
        "chunk_index": 3
      }
    }
  ]
}
```

---

## Resetting the knowledge base

To clear all previously uploaded documents and start fresh:

```bash
curl -X POST http://localhost:8000/admin/reset
```

This deletes the `documents` collection in Chroma and recreates an empty one.

---

## Evaluation: faithfulness and retrieval recall

The script `app/eval_rag.py` runs a simple evaluation over a labeled JSONL dataset:

- **Retrieval recall** – whether any retrieved chunk contains a gold context snippet.  
- **Faithfulness** – LLM-judged score between 0 and 1 indicating how well the answer is grounded in the retrieved context.

Run:

```bash
cd backend
python -m app.eval_rag ./data/eval_set.jsonl
```

The eval set is a JSONL file with records like:

```jsonl
{"id": "ex1", "question": "What is the company policy on remote work?", "gold_answer": "Employees may work remotely up to three days per week with manager approval.", "gold_context_snippet": "may work remotely up to three days per week", "where": {"source": "employee_handbook"}}
{"id": "ex2", "question": "How many days of paid vacation does a new full-time employee receive?", "gold_answer": "New full-time employees receive 20 days of paid vacation per year.", "gold_context_snippet": "20 days of paid vacation per year", "where": {"source": "benefits_policy"}}
```

The script prints per-example metrics and a summary (average recall and faithfulness).

---

This backend is stateless apart from the local Chroma index directory, making it straightforward to run locally or on a single VM/container.