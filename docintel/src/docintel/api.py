"""FastAPI REST API for DocIntel."""

import logging
from contextlib import asynccontextmanager
from typing import BinaryIO

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from docintel.config import settings
from docintel.models import (
    QueryRequest,
    QueryResponse,
    DocumentUploadResponse,
    HealthResponse,
)
from docintel.document_processor import DocumentProcessor
from docintel.vector_store import VectorStore
from docintel.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances (initialized in lifespan)
vector_store: VectorStore = None
rag_engine: RAGEngine = None
document_processor: DocumentProcessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global vector_store, rag_engine, document_processor

    # Startup
    logger.info("Initializing DocIntel API...")
    vector_store = VectorStore()
    rag_engine = RAGEngine(vector_store=vector_store)
    document_processor = DocumentProcessor()
    logger.info("DocIntel API initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down DocIntel API...")


app = FastAPI(
    title="DocIntel API",
    description="Production-ready document intelligence platform with RAG capabilities",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    vector_db_connected = vector_store.health_check() if vector_store else False

    return HealthResponse(
        status="healthy" if vector_db_connected else "degraded",
        vector_db_connected=vector_db_connected,
        llm_provider=settings.llm_provider,
    )


@app.post("/documents/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.

    Supported formats: PDF, DOCX, TXT, MD
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    logger.info(f"Processing upload: {file.filename}")

    try:
        # Process document
        content = await file.read()
        document, chunks = document_processor.process_document(
            file=BinaryIO(content), filename=file.filename
        )

        # Index chunks
        indexed_count = vector_store.index_chunks(chunks)

        logger.info(f"Successfully processed and indexed {file.filename}")

        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            chunks_created=indexed_count,
            message=f"Document processed successfully. {indexed_count} chunks indexed.",
        )

    except ValueError as e:
        logger.error(f"Validation error processing {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error processing document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query indexed documents using natural language.

    The system will retrieve relevant document chunks and generate an answer using RAG.
    """
    logger.info(f"Received query: {request.query[:100]}...")

    try:
        response = await rag_engine.aquery(request)
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        chunk_count = vector_store.count_documents()
        return {
            "total_chunks": chunk_count,
            "collection_name": settings.qdrant_collection_name,
            "embedding_model": settings.embedding_model,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    try:
        from uuid import UUID

        doc_uuid = UUID(document_id)
        vector_store.delete_document(doc_uuid)

        return {"message": f"Document {document_id} deleted successfully"}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/reset")
async def reset_collection():
    """
    Reset the entire collection (delete all documents).

    WARNING: This operation cannot be undone!
    """
    logger.warning("Collection reset requested")
    try:
        vector_store.reset_collection()
        return {"message": "Collection reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def start_server():
    """Start the API server."""
    import uvicorn

    uvicorn.run(
        "docintel.api:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False,
    )


if __name__ == "__main__":
    start_server()
