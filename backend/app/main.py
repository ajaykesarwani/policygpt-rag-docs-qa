from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pypdf import PdfReader
import io
import json

from .config import Settings, get_settings
from .models import IngestRequest, QueryRequest, QueryResponse, DocumentChunk
from .ingestion import ingest_texts
from .rag_pipeline import rag_query, rag_query_stream
from .logging_config import configure_logging
from .rag_pipeline import reset_vector_store

configure_logging()
app = FastAPI(title="RAG Document QA", version="1.0.0")


@app.on_event("startup")
def startup_event():
    # Could warm up vector store / LLM here if needed
    pass


# Allow local dev + Vercel frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "env": settings.app_env}


@app.post("/ingest")
def ingest(req: IngestRequest, settings: Settings = Depends(get_settings)):
    """
    Ingest raw texts under a source name.
    In real life you’d parse PDFs/HTML first and call this on the extracted text.
    """
    result = ingest_texts(req.source_name, req.texts, req.metadata or {})
    return {"status": "ok", **result}


# @app.post("/query", response_model=QueryResponse)
# def query_rag(req: QueryRequest):
#     where = None

#     if req.source_name and req.filename:
#         where = {
#             "$and": [
#                 {"source": req.source_name},
#                 {"filename": req.filename},
#             ]
#         }
#     elif req.source_name:
#         where = {"source": req.source_name}
#     elif req.filename:
#         where = {"filename": req.filename}
#     else:
#         where = None

#     answer, chunks = rag_query(
#         req.query,
#         top_k=req.top_k,
#         where=where,
#         retrieval_strategy=req.retrieval_strategy,  # NEW
#     )
#     return QueryResponse(answer=answer, context=chunks)

@app.post("/query")
def query_rag(req: QueryRequest):
    where = None

    if req.source_name and req.filename:
        where = {
            "$and": [
                {"source": req.source_name},
                {"filename": req.filename},
            ]
        }
    elif req.source_name:
        where = {"source": req.source_name}
    elif req.filename:
        where = {"filename": req.filename}
    else:
        where = None

    def stream_generator():
        try:
            for step in rag_query_stream(
                query=req.query,
                history=req.history,
                top_k=req.top_k,
                where=where,
                retrieval_strategy=req.retrieval_strategy,
            ):
                yield json.dumps(step) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

# ... existing imports and app ...

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n\n".join(texts)


@app.post("/upload")
async def upload_file(source_name: str, file: UploadFile = File(...)):
    """
    Upload a single PDF or TXT file and ingest its content.
    """
    if not (file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

    content = await file.read()
    if file.filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf_bytes(content)
    else:
        text = content.decode("utf-8", errors="ignore")

    from .ingestion import ingest_texts

    ingest_result = ingest_texts(
        source_name=source_name,
        texts=[text],
        base_metadata={"filename": file.filename},
    )

    return {"status": "ok", "ingested_chunks": ingest_result["ingested_chunks"]}

@app.post("/admin/reset")
def reset_knowledge():
    """
    Danger: clears all stored documents/embeddings.
    Intended for dev/admin use.
    """
    reset_vector_store()
    return {"status": "ok", "message": "Vector store reset; all documents deleted."}