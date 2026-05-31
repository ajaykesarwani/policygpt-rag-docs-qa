# PolicyGPT – Production-Style RAG Document Q&A

End-to-end Retrieval-Augmented Generation (RAG) system for asking grounded questions over large document sets (e.g., policies, manuals, internal docs).

- **Backend**: FastAPI, ChromaDB, local SentenceTransformers embeddings, Groq LLMs.
- **Frontend**: Next.js (React) with a modern UI, designed for Vercel.
- **RAG pipeline**: Text ingestion, chunking, vector search, prompt construction, answer generation with visible context.

## Features

- Upload PDFs or TXT files directly from the UI.
- Ingest arbitrary text under logical `source_name` values.
- Chunking with configurable size/overlap and local embedding model.
- Vector search over ChromaDB.
- Answers restricted to retrieved context; shows context snippets, scores, and metadata.
- Clean separation of backend and frontend, ready for deployment (Vercel + backend host).
- Basic tests (health + prompt-building) and structured logging.

## Local setup

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env    # fill in GROQ_API_KEY and optional overrides
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # set BACKEND_URL=http://localhost:8000
npm run dev
```

Open: http://localhost:3000

## Example workflow

1. Upload a policy PDF under a source name:

   - `Upload documents` panel → choose `company-policies` as source → select `policy.pdf`.

2. Ask questions in the UI:

   - e.g. “What does our vacation policy say about carry-over days?”

3. Inspect the context:

   - The answer is shown with the retrieved chunks, including source name, filename, and similarity score.

## Why this project

This project is designed to demonstrate production-style AI engineer skills:

- **RAG architecture**: retrieval, embeddings, chunking, prompt construction, and LLM orchestration.
- **Backend engineering**: FastAPI, file uploads, environment-based configuration, local embeddings, vector DB integration.
- **Frontend engineering**: Next.js, API routes as a proxy layer, a polished UI that non-technical users can understand.
- **Practical deployment**: Frontend on Vercel with a separately deployable backend.

See:

- [`backend/README_BACKEND.md`](backend/README_BACKEND.md) for backend details.
- [`frontend/README_FRONTEND.md`](frontend/README_FRONTEND.md) for frontend details.