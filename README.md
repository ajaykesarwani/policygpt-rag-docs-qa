# PolicyGPT – Production-Style RAG Document Q&A

End-to-end Retrieval-Augmented Generation (RAG) system for asking grounded questions over large document sets (e.g., policies, manuals, internal docs).

- **Backend**: FastAPI, ChromaDB, local SentenceTransformers embeddings, Groq LLMs
- **Frontend**: Next.js (App Router) with a modern UI, designed for Vercel or local dev
- **RAG pipeline**:
  - Text ingestion and chunking
  - Hybrid retrieval (BM25 + dense vectors)
  - Metadata filters
  - LLM-based reranking
  - Context-window-aware prompt construction
  - Answer generation with visible context

---

## Features

- Upload PDFs or TXT files directly from the UI
- Ingest arbitrary text under logical `source_name` values
- Hybrid retrieval: combine semantic vectors with BM25 keyword search for stronger recall
- Metadata filters by `source_name` / filename
- LLM-based reranking of retrieved chunks
- Context window control via a configurable token budget
- Simple “Reset knowledge” admin button to clear the vector store
- Evaluation script that measures retrieval recall and answer faithfulness on a labeled test set

---

## Repository layout

- `backend/` – FastAPI backend, RAG pipeline, Chroma integration, evaluation script
- `frontend/` – Next.js frontend with upload + chat UI and API proxy routes
- `.env.example`, `backend/.env.example`, `frontend/.env.local.example` – env variable templates

---

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
cp .env.example .env  # fill in GROQ_API_KEY and optional overrides
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local  # set BACKEND_URL=http://localhost:8000
npm run dev
```

Open: http://localhost:3000

---

## Example workflow

1. **Upload a document**

   - In the frontend, open the “Upload documents” panel.
   - Choose a `source_name` (e.g. `company-policies`) and upload a PDF or TXT.
   - The backend extracts text, chunks it, embeds it, and stores it in Chroma with metadata.

2. **Ask a question**

   - In the chat input, ask something like:
     - “What does our vacation policy say about carry-over days?”
   - The backend:
     - Runs hybrid retrieval (BM25 + dense vectors)
     - Applies metadata filters if provided
     - Reranks candidates with the LLM
     - Builds a prompt within the context token budget
     - Calls Groq’s LLM to generate an answer

3. **Inspect context**

   - The UI shows:
     - The answer
     - The retrieved chunks with `source`, `filename`, `score`, and text

4. **Reset knowledge (for dev)**

   - Click “Reset knowledge” in the UI to clear all stored documents and start fresh.

---

## Evaluation

The backend includes `app/eval_rag.py`, which runs an evaluation over a labeled JSONL dataset:

- **Retrieval recall** – checks if retrieved chunks contain a gold context snippet
- **Answer faithfulness** – LLM-judged grounding between answer and context

Run:

```bash
cd backend
python -m app.eval_rag ./data/eval_set.jsonl
```

Use this to compare different retrieval configs (e.g., pure dense vs hybrid) and log tradeoffs between latency and accuracy in your own notes or README.

---

## Why this project

This project is designed to demonstrate real-world AI engineering skills:

- **RAG architecture**: hybrid retrieval, chunking, embeddings, prompt construction, and LLM orchestration
- **Backend engineering**: FastAPI, file uploads, environment-based configuration, local models, vector DB integration
- **Frontend engineering**: Next.js, API proxies, and a user-friendly chat UI
- **Evaluation mindset**: tracking recall and faithfulness, and providing hooks to explore different retrieval configurations

For more details:

- Backend: [`backend/README.md`](backend/README.md)  
- Frontend: [`frontend/README.md`](frontend/README.md)