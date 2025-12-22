"""Data models for DocIntel."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class Document(BaseModel):
    """Document metadata model."""

    id: UUID = Field(default_factory=uuid4)
    filename: str
    content: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    chunk_count: int = 0


class DocumentChunk(BaseModel):
    """Document chunk model for vector storage."""

    id: str
    document_id: UUID
    content: str
    chunk_index: int
    metadata: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Query request model."""

    query: str = Field(..., min_length=1, description="The question to ask about the documents")
    document_ids: Optional[list[UUID]] = Field(
        None, description="Optional list of document IDs to search within"
    )
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of chunks to retrieve")


class QueryResponse(BaseModel):
    """Query response model."""

    answer: str
    sources: list[dict]
    query: str
    processing_time: float


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""

    document_id: UUID
    filename: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    vector_db_connected: bool
    llm_provider: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
