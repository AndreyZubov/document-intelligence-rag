# DocIntel

Intelligence platform that enables semantic search and question-answering over documents using Retrieval Augmented Generation (RAG).

## Features

- **Multiple Document Formats**: Support for PDF, DOCX, TXT, and Markdown files
- **Semantic Search**: Vector-based document retrieval using state-of-the-art embeddings
- **RAG Integration**: Question answering using Claude (Anthropic) or GPT (OpenAI)
- **REST API**: FastAPI-based HTTP API with automatic documentation
- **CLI Interface**: Rich command-line interface for local operations
- **Vector Database**: Qdrant for efficient similarity search
- **Production Ready**: Docker support, comprehensive logging, error handling

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Documents  │─────▶│   Processor  │─────▶│  Embeddings │
│ (PDF, DOCX) │      │   & Chunker  │      │   Service   │
└─────────────┘      └──────────────┘      └─────────────┘
                                                    │
                                                    ▼
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│     User    │─────▶│  RAG Engine  │◀─────│   Qdrant    │
│    Query    │      │ (Claude/GPT) │      │  Vector DB  │
└─────────────┘      └──────────────┘      └─────────────┘
```

## Quick Start

### Using Docker (Recommended)

1. **Clone and setup**:
```bash
git clone <your-repo>
cd docintel
cp .env.example .env
```

2. **Configure environment**:
Edit `.env` and add your API key:
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_api_key_here
```

3. **Start services**:
```bash
docker-compose up -d
```

4. **Upload a document**:
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@your_document.pdf"
```

5. **Ask questions**:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the document?"}'
```

### Local Installation

1. **Prerequisites**:
- Python 3.10 or higher
- Qdrant running locally or via Docker

2. **Install dependencies**:
```bash
pip install -e .
```

3. **Start Qdrant**:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Use the CLI**:
```bash
# Check health
docintel health

# Upload a document
docintel upload path/to/document.pdf

# Query documents
docintel query "What are the key findings?"

# Start API server
docintel serve
```

## API Documentation

Once the server is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Key Endpoints

#### Upload Document
```bash
POST /documents/upload
Content-Type: multipart/form-data

Response:
{
  "document_id": "uuid",
  "filename": "document.pdf",
  "chunks_created": 42,
  "message": "Document processed successfully"
}
```

#### Query Documents
```bash
POST /query
Content-Type: application/json

Body:
{
  "query": "What is the main conclusion?",
  "max_results": 5,
  "document_ids": ["uuid"] // optional
}

Response:
{
  "answer": "The main conclusion is...",
  "sources": [...],
  "query": "What is the main conclusion?",
  "processing_time": 1.23
}
```

#### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "vector_db_connected": true,
  "llm_provider": "anthropic",
  "timestamp": "2024-01-01T00:00:00"
}
```

#### System Stats
```bash
GET /stats

Response:
{
  "total_chunks": 150,
  "collection_name": "docintel_documents",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_provider": "anthropic",
  "llm_model": "claude-3-5-sonnet-20241022"
}
```

## Configuration

All configuration is done via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | LLM provider: `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | - | Your Anthropic API key |
| `OPENAI_API_KEY` | - | Your OpenAI API key |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Claude model to use |
| `OPENAI_MODEL` | `gpt-4-turbo-preview` | GPT model to use |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION_NAME` | `docintel_documents` | Collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | `512` | Document chunk size |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `MAX_CONTEXT_CHUNKS` | `5` | Max chunks for RAG context |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `LOG_LEVEL` | `INFO` | Logging level |

## CLI Commands

```bash
# Upload documents
docintel upload document.pdf
docintel upload report.docx

# Query documents
docintel query "What are the main findings?"
docintel query "Summarize the methodology" --max-results 10

# System management
docintel health         # Check system health
docintel stats          # Show statistics
docintel reset --yes    # Reset database (delete all)

# Start API server
docintel serve
docintel serve --host 0.0.0.0 --port 8080
docintel serve --reload  # Development mode
```

## Development

### Project Structure

```
docintel/
├── src/docintel/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── models.py              # Pydantic data models
│   ├── document_processor.py  # Document parsing & chunking
│   ├── embeddings.py          # Embedding generation
│   ├── vector_store.py        # Qdrant integration
│   ├── rag_engine.py          # RAG implementation
│   ├── api.py                 # FastAPI application
│   └── cli.py                 # CLI interface
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=docintel --cov-report=html

# Type checking
mypy src/docintel

# Linting
ruff check src/
black --check src/
```

### Adding New Features

1. **New document format**: Extend `DocumentProcessor._extract_*` methods
2. **Different embeddings**: Modify `EmbeddingService` class
3. **Alternative LLM**: Add provider in `RAGEngine._query_*` methods
4. **Custom chunking**: Implement new strategy in `DocumentProcessor.create_chunks`

## Use Cases

### Customer Support Knowledge Base
Index support documentation, FAQs, and product manuals. Enable support agents to quickly find relevant information for customer queries.

### Research & Analysis
Upload research papers, reports, and articles. Query across documents to find patterns, compare findings, or extract insights.

### Legal Document Review
Process contracts, agreements, and legal documents. Search for specific clauses, obligations, or terms across large document sets.

### Technical Documentation
Index API docs, technical specs, and runbooks. Help developers find relevant code examples and implementation details.

## Performance Considerations

- **Chunk Size**: Balance between context and precision. Smaller chunks (256-512) for precise answers, larger (1024+) for broader context
- **Embedding Model**: `all-MiniLM-L6-v2` is fast but less accurate. Consider `all-mpnet-base-v2` for better quality
- **Vector DB**: Qdrant is optimized for production. For millions of documents, consider clustering or scaling horizontally
- **LLM Choice**: Claude Sonnet offers best price/performance. GPT-4 for maximum accuracy. Consider caching for repeated queries

## Troubleshooting

### Connection Issues
```bash
# Check Qdrant is running
curl http://localhost:6333/health

# Check API health
curl http://localhost:8000/health
```

### Empty Results
- Ensure documents are properly indexed: `docintel stats`
- Lower the score threshold in vector search
- Try different embedding models
- Increase `max_results` in queries

### Memory Issues
- Reduce `chunk_size` and batch processing size
- Use smaller embedding models
- Limit concurrent document processing

## Security

For production deployments:

1. **API Keys**: Never commit `.env` files. Use secrets management
2. **CORS**: Configure appropriate origins in `api.py`
3. **Authentication**: Add API authentication middleware
4. **Rate Limiting**: Implement rate limiting for public APIs
5. **Input Validation**: Already included via Pydantic models
6. **HTTPS**: Use reverse proxy (nginx, Traefik) with SSL

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

Built with Python, FastAPI, Qdrant, and Anthropic Claude / OpenAI GPT
