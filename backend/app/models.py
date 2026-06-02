from pydantic import BaseModel, Field
from typing import List, Optional


class IngestRequest(BaseModel):
    source_name: str = Field(..., description="Logical name of document source")
    texts: List[str] = Field(..., description="List of raw text chunks or pages")
    metadata: Optional[dict] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    source_name: Optional[str] = None
    filename: Optional[str] = None
    # for future extensibility, e.g. to support different retrieval strategies
    retrieval_strategy: Optional[str] = None  # e.g. "dense" or "hybrid"

class DocumentChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    context: List[DocumentChunk]