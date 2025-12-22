# DocIntel Architecture

## Overview

DocIntel is a Retrieval Augmented Generation (RAG) system that enables semantic search and question-answering over documents. This document describes the system architecture, component interactions, and key design decisions.

## System Components

### 1. Document Processor (`document_processor.py`)

**Responsibility**: Extract text from various document formats and split into chunks.

**Key Features**:
- Multi-format support (PDF, DOCX, TXT, Markdown)
- Intelligent chunking with overlap
- Boundary-aware splitting (sentences, paragraphs)

**Design Decisions**:
- Character-based chunking with configurable overlap ensures no information loss
- Chunk size (512) balances context preservation with search precision
- Overlap (50 chars) maintains continuity across chunk boundaries

### 2. Embedding Service (`embeddings.py`)

**Responsibility**: Convert text into dense vector representations.

**Key Features**:
- Uses Sentence-BERT models for semantic embeddings
- Batch processing for efficiency
- Cached singleton pattern to avoid model reloading

**Design Decisions**:
- `all-MiniLM-L6-v2` default: fast, reasonable quality, 384 dimensions
- Sentence-transformers: optimized for semantic similarity
- Cosine similarity for relevance scoring

### 3. Vector Store (`vector_store.py`)

**Responsibility**: Store and retrieve document chunks using vector similarity.

**Key Features**:
- Qdrant vector database integration
- Semantic search with score thresholding
- Document-level filtering
- Collection management

**Design Decisions**:
- Qdrant chosen for: production-ready, fast, good Python SDK
- Cosine distance: optimal for sentence embeddings
- Score threshold (0.3): filters low-relevance results

### 4. RAG Engine (`rag_engine.py`)

**Responsibility**: Orchestrate retrieval and generation for question-answering.

**Key Features**:
- Two-stage pipeline: retrieve → generate
- Support for multiple LLM providers
- Context window optimization
- Source attribution

**Design Decisions**:
- Provider abstraction: easy to swap between Anthropic/OpenAI
- Temperature 0.7: balance between creativity and accuracy
- Max tokens 2048: sufficient for detailed answers
- Explicit source attribution for transparency

### 5. FastAPI Application (`api.py`)

**Responsibility**: REST API for remote access.

**Key Features**:
- Async request handling
- Automatic OpenAPI documentation
- Health checks and monitoring
- CORS support

**Design Decisions**:
- FastAPI: modern, fast, automatic validation
- Lifespan events: proper resource initialization/cleanup
- Structured error handling with appropriate HTTP codes

### 6. CLI Interface (`cli.py`)

**Responsibility**: Command-line interface for local operations.

**Key Features**:
- Rich formatting for better UX
- Progress indicators
- Interactive confirmations for destructive operations

**Design Decisions**:
- Typer: type-safe, easy CLI creation
- Rich: professional terminal output
- Separate from API for different use cases

## Data Flow

### Document Upload Flow

```
User
  │
  ▼
[1] Upload document (PDF, DOCX, etc.)
  │
  ▼
[2] DocumentProcessor.extract_text()
  │
  ▼
[3] DocumentProcessor.create_chunks()
  │
  ▼
[4] EmbeddingService.embed_batch()
  │
  ▼
[5] VectorStore.index_chunks()
  │
  ▼
[6] Qdrant (persisted)
```

### Query Flow

```
User Query
  │
  ▼
[1] RAGEngine.query()
  │
  ├─▶ [2] EmbeddingService.embed_text(query)
  │     │
  │     ▼
  │   [3] VectorStore.search()
  │     │
  │     ▼
  │   [4] Qdrant → Top-K chunks
  │
  ├─▶ [5] Build context from chunks
  │
  ├─▶ [6] Build prompt with context
  │
  ├─▶ [7] Query LLM (Claude/GPT)
  │
  ▼
[8] Return answer + sources
```

## Key Design Patterns

### 1. Dependency Injection
- Components accept dependencies (e.g., `RAGEngine` accepts `VectorStore`)
- Easier testing and flexibility

### 2. Configuration Management
- Centralized in `config.py` using Pydantic
- Environment variables for deployment
- Type-safe with validation

### 3. Error Handling
- Structured exceptions at each layer
- Logging for debugging
- User-friendly error messages

### 4. Separation of Concerns
- Each module has single responsibility
- Clear boundaries between layers
- API and CLI as separate interfaces

## Scalability Considerations

### Current Limitations
- Single-instance deployment
- In-memory model loading
- Synchronous LLM calls

### Scaling Strategies

**Horizontal Scaling**:
- Qdrant clustering for large datasets
- Load balancer for API instances
- Shared embedding service

**Vertical Scaling**:
- GPU acceleration for embeddings
- Larger embedding models
- Batch processing optimizations

**Caching**:
- Query result caching
- Embedding caching for repeated texts
- LLM response caching

## Security Considerations

1. **API Keys**: Stored in environment, never in code
2. **Input Validation**: Pydantic models at API boundary
3. **File Upload**: Size limits, type validation
4. **SQL Injection**: N/A (no SQL, vector DB only)
5. **XSS**: N/A (no HTML rendering)
6. **Authentication**: Not implemented (add middleware for production)

## Performance Metrics

| Operation | Time (avg) | Notes |
|-----------|-----------|-------|
| Upload PDF (10 pages) | ~2-3s | Depends on OCR complexity |
| Generate embeddings (50 chunks) | ~1-2s | CPU-dependent |
| Vector search | <100ms | Qdrant optimized |
| LLM query | ~2-5s | API latency |
| Full RAG query | ~3-7s | End-to-end |

## Technology Choices

| Component | Choice | Alternatives | Rationale |
|-----------|--------|--------------|-----------|
| Vector DB | Qdrant | Pinecone, Weaviate, Milvus | Self-hosted, production-ready, Python SDK |
| Embeddings | Sentence-BERT | OpenAI, Cohere | Free, runs locally, good quality |
| LLM | Claude/GPT | Llama, Mistral | Best quality, API simplicity |
| API Framework | FastAPI | Flask, Django | Modern, async, auto docs |
| CLI Framework | Typer | Click, argparse | Type-safe, easy to use |

## Future Enhancements

1. **Async Embeddings**: Speed up batch processing
2. **Hybrid Search**: Combine semantic + keyword search
3. **Document Metadata**: Filter by date, author, tags
4. **Conversation Memory**: Multi-turn dialogues
5. **Fine-tuned Embeddings**: Domain-specific models
6. **Query Optimization**: Caching, pre-fetching
7. **Multi-tenancy**: User isolation, quotas
8. **Monitoring**: Metrics, tracing, alerting

## Testing Strategy

- **Unit Tests**: Individual components
- **Integration Tests**: Component interactions
- **E2E Tests**: Full workflows
- **Load Tests**: Performance under load
- **Mock LLMs**: Avoid API costs in tests

## Deployment Options

1. **Docker Compose**: Quick local deployment
2. **Kubernetes**: Production, scalable
3. **AWS ECS**: Managed containers
4. **Serverless**: Lambda + API Gateway (stateless)
5. **VM**: Traditional deployment

---

This architecture balances simplicity, performance, and maintainability while remaining extensible for future requirements.
