# PolicyGPT – RAG Backend

FastAPI backend implementing a production-style Retrieval-Augmented Generation (RAG) pipeline over local documents.

- **Framework**: FastAPI for the HTTP API
- **Retrieval**: ChromaDB for vector storage
- **Embeddings**: Local SentenceTransformers model (e.g. `all-MiniLM-L6-v2`) – no paid API required
- **LLM**: Groq API (OpenAI-compatible endpoint) for fast chat completions

## Endpoints

- `GET /health` – Health check.
- `POST /upload` – Upload a PDF or TXT file, extract text, and ingest it into the vector index.
- `POST /ingest` – Ingest raw text under a given `source_name` (used internally by `/upload` and for bulk ingestion).
- `POST /query` – Ask questions over the ingested documents using the RAG pipeline.

## Local setup

```bash
cd backend
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # fill in GROQ_API_KEY and any overrides
uvicorn app.main:app --reload --port 8000
```

The `.env` file typically includes:

```text
APP_ENV=local
GROQ_API_KEY=your_groq_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile
CHROMA_DIR=chroma_index
MAX_CONTEXT_TOKENS=3000
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

## Example usage

### Upload a PDF or TXT file

```bash
curl -X POST "http://localhost:8000/upload?source_name=company-policies" \
  -F "file=@/absolute/path/to/policy.pdf"
```

Successful response:

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

Example response (simplified):

```json
{
  "answer": "The policy allows unused vacation days to be carried over for one year, subject to manager approval.",
  "context": [
    {
      "id": "0",
      "text": "Employees may carry over unused vacation days into the following calendar year with written manager approval...",
      "score": 0.12,
      "metadata": {
        "source": "company-policies",
        "filename": "policy.pdf",
        "chunk_index": 3
      }
    }
  ]
}
```

This backend is stateless apart from the local Chroma index directory, making it easy to run locally or host on a simple VM/container.