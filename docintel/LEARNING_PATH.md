# DocIntel Learning Path

This document provides a structured learning path through the DocIntel codebase. Follow this guide to progressively understand Python, AI, RAG, and vector databases.

## Learning Modules

### Module 1: Python Fundamentals (Week 1)

#### Day 1-2: Project Structure & Configuration

**Files to Study**:
- `pyproject.toml` - Modern Python packaging
- `src/docintel/config.py` - Configuration management
- `src/docintel/models.py` - Data models with Pydantic

**Concepts**:
- Python packaging with pyproject.toml
- Type hints and type checking
- Pydantic for data validation
- Environment variable management

**Exercises**:
```python
# 1. Create a new Pydantic model
from pydantic import BaseModel, Field

class MyDocument(BaseModel):
    title: str
    content: str = Field(..., min_length=10)
    tags: list[str] = []

# Try creating invalid instances and see validation errors

# 2. Explore configuration
from docintel.config import settings
print(f"Chunk size: {settings.chunk_size}")
print(f"LLM Provider: {settings.llm_provider}")

# 3. Modify settings and see effects
```

**Questions to Answer**:
- What is the difference between `BaseModel` and `BaseSettings`?
- Why use type hints?
- How does environment variable loading work?

#### Day 3-4: Document Processing

**Files to Study**:
- `src/docintel/document_processor.py`

**Concepts**:
- File I/O in Python
- Text extraction from different formats
- String manipulation
- Algorithm: text chunking with overlap

**Exercises**:
```python
from docintel.document_processor import DocumentProcessor
from uuid import uuid4

processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

# Exercise 1: Understand chunking
text = "This is a sample text. " * 100
chunks = processor.create_chunks(text, uuid4())

print(f"Input length: {len(text)}")
print(f"Number of chunks: {len(chunks)}")
print(f"Total chunk content: {sum(len(c.content) for c in chunks)}")

# Exercise 2: Experiment with parameters
processor2 = DocumentProcessor(chunk_size=200, chunk_overlap=50)
chunks2 = processor2.create_chunks(text, uuid4())
print(f"With larger chunks: {len(chunks2)} chunks")

# Exercise 3: Handle edge cases
empty_chunks = processor.create_chunks("", uuid4())
print(f"Empty text: {len(empty_chunks)} chunks")
```

**Questions to Answer**:
- Why do we need overlap between chunks?
- What happens at sentence boundaries?
- How does chunk size affect the system?

#### Day 5-7: Testing & Best Practices

**Files to Study**:
- `tests/test_document_processor.py`
- `Makefile`

**Concepts**:
- Unit testing with pytest
- Test fixtures
- Test organization
- Development workflow

**Exercises**:
```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=docintel --cov-report=term

# Run specific test
pytest tests/test_document_processor.py::TestDocumentProcessor::test_create_chunks -v

# Add your own test
```

**Create Your Own Test**:
```python
# In tests/test_document_processor.py

def test_chunk_size_validation(self):
    """Test that chunks respect max size."""
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    text = "A" * 500
    chunks = processor.create_chunks(text, uuid4())

    for chunk in chunks:
        assert len(chunk.content) <= 50, "Chunk exceeds max size"
```

---

### Module 2: AI & Embeddings (Week 2)

#### Day 1-2: Understanding Embeddings

**Files to Study**:
- `src/docintel/embeddings.py`

**Concepts**:
- Vector embeddings
- Sentence-BERT models
- Cosine similarity
- Batch processing
- Model caching

**Exercises**:
```python
from docintel.embeddings import get_embedding_service

service = get_embedding_service()

# Exercise 1: Generate embeddings
text1 = "Python is a programming language"
emb1 = service.embed_text(text1)

print(f"Embedding dimension: {len(emb1)}")
print(f"First 5 values: {emb1[:5]}")

# Exercise 2: Compare similar and dissimilar texts
sentences = [
    "Machine learning is a subset of AI",
    "Artificial intelligence includes machine learning",
    "I enjoy eating pizza",
    "The weather is nice today"
]

embeddings = service.embed_batch(sentences)

# Compare first two (similar) vs first and third (dissimilar)
sim_similar = service.compute_similarity(embeddings[0], embeddings[1])
sim_dissimilar = service.compute_similarity(embeddings[0], embeddings[2])

print(f"Similar texts similarity: {sim_similar:.3f}")
print(f"Dissimilar texts similarity: {sim_dissimilar:.3f}")

# Exercise 3: Visualize embedding space (requires matplotlib)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Generate embeddings for various texts
texts = [
    "machine learning", "deep learning", "neural networks",
    "pizza", "pasta", "italian food",
    "car", "truck", "vehicle"
]

embs = service.embed_batch(texts)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
embs_2d = pca.fit_transform(np.array(embs))

plt.figure(figsize=(10, 6))
plt.scatter(embs_2d[:, 0], embs_2d[:, 1])
for i, txt in enumerate(texts):
    plt.annotate(txt, (embs_2d[i, 0], embs_2d[i, 1]))
plt.title("Embedding Space Visualization")
plt.show()
```

**Questions to Answer**:
- What do the numbers in an embedding represent?
- Why 384 dimensions?
- What is cosine similarity and why is it used?
- How does batch processing improve performance?

#### Day 3-4: Different Embedding Models

**Experiment**: Try different models and compare

```python
# Create a test script: test_embeddings.py

from sentence_transformers import SentenceTransformer
import time

models = [
    "sentence-transformers/all-MiniLM-L6-v2",      # Fast, small
    "sentence-transformers/all-mpnet-base-v2",     # Better quality
    "sentence-transformers/paraphrase-MiniLM-L3-v2" # Tiny, fastest
]

test_texts = ["This is a test sentence"] * 100

for model_name in models:
    print(f"\nTesting: {model_name}")

    model = SentenceTransformer(model_name)

    # Measure time
    start = time.time()
    embeddings = model.encode(test_texts)
    duration = time.time() - start

    print(f"  Dimension: {embeddings.shape[1]}")
    print(f"  Time: {duration:.2f}s for 100 texts")
    print(f"  Speed: {100/duration:.1f} texts/sec")
```

**Questions to Answer**:
- What's the tradeoff between speed and quality?
- When would you choose each model?
- How does dimension size affect storage and search?

---

### Module 3: Vector Databases (Week 2-3)

#### Day 5-7: Qdrant Vector Store

**Files to Study**:
- `src/docintel/vector_store.py`

**Concepts**:
- Vector databases
- Nearest neighbor search
- Collections and indexes
- Filtering and metadata
- Distance metrics

**Exercises**:
```python
from docintel.vector_store import VectorStore
from docintel.document_processor import DocumentProcessor

store = VectorStore()

# Exercise 1: Understand indexing
processor = DocumentProcessor()
with open("sample.txt", "rb") as f:
    doc, chunks = processor.process_document(f, "sample.txt")

print(f"Indexing {len(chunks)} chunks...")
count = store.index_chunks(chunks)
print(f"Indexed: {count}")

# Exercise 2: Search behavior
queries = [
    "machine learning",
    "python programming",
    "completely unrelated topic xyz123"
]

for query in queries:
    results = store.search(query, limit=3)
    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
    if results:
        print(f"Top score: {results[0]['score']:.3f}")

# Exercise 3: Score threshold effect
print("\n--- Testing score thresholds ---")
for threshold in [0.1, 0.3, 0.5, 0.7]:
    results = store.search(
        "machine learning",
        limit=10,
        score_threshold=threshold
    )
    print(f"Threshold {threshold}: {len(results)} results")

# Exercise 4: Database statistics
total = store.count_documents()
print(f"\nTotal chunks in database: {total}")
```

**Questions to Answer**:
- How does vector search differ from keyword search?
- What is a "good" similarity score?
- Why do we need a score threshold?
- How does Qdrant organize vectors internally?

#### Deep Dive: Vector Search Algorithm

**Read and Understand**:
```python
# Simplified version of what happens during search:

def conceptual_vector_search(query_vector, all_vectors, top_k=5):
    """
    This is what happens inside Qdrant (simplified).
    """
    similarities = []

    # Compare query to every stored vector
    for stored_vector, metadata in all_vectors:
        similarity = cosine_similarity(query_vector, stored_vector)
        similarities.append((similarity, metadata))

    # Sort by similarity (descending)
    similarities.sort(reverse=True, key=lambda x: x[0])

    # Return top K results
    return similarities[:top_k]

# In practice, Qdrant uses:
# - HNSW (Hierarchical Navigable Small World) graphs
# - Approximate nearest neighbor search
# - Much faster than naive comparison
```

---

### Module 4: RAG & LLM Integration (Week 3)

#### Day 1-3: RAG Engine

**Files to Study**:
- `src/docintel/rag_engine.py`

**Concepts**:
- Retrieval Augmented Generation
- Prompt engineering
- LLM API integration (Anthropic/OpenAI)
- Context window management
- Source attribution

**Exercises**:
```python
from docintel.rag_engine import RAGEngine
from docintel.models import QueryRequest

engine = RAGEngine()

# Exercise 1: Basic RAG query
request = QueryRequest(
    query="What is machine learning?",
    max_results=5
)

response = engine.query(request)
print(f"Answer: {response.answer}")
print(f"\nSources used: {len(response.sources)}")
print(f"Processing time: {response.processing_time:.2f}s")

# Exercise 2: Effect of max_results
for n in [1, 3, 5, 10]:
    request = QueryRequest(query="Explain Python", max_results=n)
    response = engine.query(request)
    print(f"\n{n} chunks: {len(response.answer)} chars, {response.processing_time:.2f}s")

# Exercise 3: Analyze sources
request = QueryRequest(query="What are the key concepts?", max_results=5)
response = engine.query(request)

print("\n--- Source Analysis ---")
for i, source in enumerate(response.sources, 1):
    print(f"{i}. Score: {source['relevance_score']:.3f}")
    print(f"   Preview: {source['content_preview'][:80]}...")
```

**Modify the Prompt**:

Open `src/docintel/rag_engine.py` and experiment with `_build_prompt()`:

```python
# Try different prompt styles:

# Style 1: Concise answers
def _build_prompt(self, query: str, context: str) -> str:
    return f"""Answer this question in 2-3 sentences based on the context.

Context: {context}

Question: {query}

Answer (2-3 sentences):"""

# Style 2: Bullet points
def _build_prompt(self, query: str, context: str) -> str:
    return f"""Answer using bullet points based on the context.

Context: {context}

Question: {query}

Answer (bullet points):"""

# Style 3: With confidence
def _build_prompt(self, query: str, context: str) -> str:
    return f"""Answer the question and rate your confidence (High/Medium/Low).

Context: {context}

Question: {query}

Answer:
Confidence:"""
```

**Questions to Answer**:
- Why retrieve first, then generate?
- What happens if context doesn't contain the answer?
- How does prompt engineering affect output quality?
- What's the tradeoff in max_results?

#### Day 4-5: LLM APIs

**Concepts**:
- API authentication
- Rate limiting
- Token counting
- Cost optimization
- Streaming responses

**Exercises**:
```python
# Exercise 1: Compare providers
from docintel.config import settings

# Test with Anthropic
settings.llm_provider = "anthropic"
engine_claude = RAGEngine()

# Test with OpenAI (if you have key)
settings.llm_provider = "openai"
engine_gpt = RAGEngine()

query = QueryRequest(query="What is Python?", max_results=3)

# Compare responses
response_claude = engine_claude.query(query)
response_gpt = engine_gpt.query(query)

print("Claude:", response_claude.answer[:200])
print("\nGPT:", response_gpt.answer[:200])
```

**Cost Analysis**:
```python
# Track API usage
import tiktoken

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Estimate cost per query
request = QueryRequest(query="Your question", max_results=5)
# After building prompt in RAG:
prompt_tokens = estimate_tokens(prompt)
response_tokens = estimate_tokens(response.answer)

print(f"Input tokens: {prompt_tokens}")
print(f"Output tokens: {response_tokens}")
print(f"Estimated cost: ${(prompt_tokens * 0.00003 + response_tokens * 0.00006):.4f}")
```

---

### Module 5: Web APIs & Production (Week 4)

#### Day 1-2: FastAPI Application

**Files to Study**:
- `src/docintel/api.py`

**Concepts**:
- REST API design
- Async Python (asyncio)
- Request validation
- Error handling
- API documentation
- CORS

**Exercises**:
```bash
# Start the server
docintel serve

# Visit http://localhost:8000/docs
# Try each endpoint in the interactive docs
```

**Add a New Endpoint**:
```python
# In api.py, add:

@app.get("/documents/{document_id}/info")
async def get_document_info(document_id: str):
    """Get information about a specific document."""
    try:
        from uuid import UUID
        doc_uuid = UUID(document_id)

        # Search for chunks from this document
        results = vector_store.search(
            query="",  # Empty query gets any chunks
            document_ids=[doc_uuid],
            limit=1
        )

        if not results:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "document_id": document_id,
            "chunk_count": len(results),
            "sample_content": results[0]['content'][:200]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Questions to Answer**:
- What is async/await and why use it?
- How does FastAPI validate requests?
- What are HTTP status codes?
- How does API documentation work?

#### Day 3-4: CLI Development

**Files to Study**:
- `src/docintel/cli.py`

**Concepts**:
- Command-line interfaces
- Rich terminal output
- Progress indicators
- User interactions

**Add a New Command**:
```python
# In cli.py, add:

@app.command()
def compare(
    query: str = typer.Argument(..., help="Query to compare"),
    num_results: list[int] = typer.Option([3, 5, 10], help="Number of results to compare"),
):
    """Compare query results with different result counts."""
    console.print(f"\n[cyan]Comparing results for:[/cyan] {query}\n")

    vector_store = VectorStore()
    rag_engine = RAGEngine(vector_store=vector_store)

    table = Table(title="Comparison Results")
    table.add_column("Results", style="cyan")
    table.add_column("Answer Length", style="green")
    table.add_column("Time (s)", style="yellow")

    for n in num_results:
        request = QueryRequest(query=query, max_results=n)
        response = rag_engine.query(request)

        table.add_row(
            str(n),
            str(len(response.answer)),
            f"{response.processing_time:.2f}"
        )

    console.print(table)
```

**Test Your Command**:
```bash
docintel compare "What is Python?" --num-results 3 --num-results 5 --num-results 10
```

---

### Module 6: Advanced Topics (Ongoing)

#### Docker & Deployment

**Files to Study**:
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

**Exercises**:
```bash
# Build image
docker build -t docintel:custom .

# Run full stack
docker-compose up -d

# View logs
docker-compose logs -f api

# Scale API
docker-compose up -d --scale api=3
```

#### Testing Strategy

**Create Integration Tests**:
```python
# tests/test_integration.py

import pytest
from docintel.document_processor import DocumentProcessor
from docintel.vector_store import VectorStore
from docintel.rag_engine import RAGEngine
from docintel.models import QueryRequest

@pytest.fixture
def setup_system():
    """Setup complete system for testing."""
    processor = DocumentProcessor()
    vector_store = VectorStore()
    rag_engine = RAGEngine(vector_store=vector_store)
    return processor, vector_store, rag_engine

def test_end_to_end_workflow(setup_system):
    """Test complete document upload and query workflow."""
    processor, vector_store, rag_engine = setup_system

    # Create and index document
    test_doc = b"Python is a programming language. It is used for AI."
    import io
    doc, chunks = processor.process_document(
        io.BytesIO(test_doc),
        "test.txt"
    )

    vector_store.index_chunks(chunks)

    # Query
    request = QueryRequest(query="What is Python?", max_results=3)
    response = rag_engine.query(request)

    # Assertions
    assert len(response.answer) > 0
    assert len(response.sources) > 0
    assert "python" in response.answer.lower()
```

#### Performance Optimization

**Benchmark Your System**:
```python
# performance_test.py

import time
from docintel.document_processor import DocumentProcessor
from docintel.embeddings import get_embedding_service
from docintel.vector_store import VectorStore

def benchmark_embeddings(num_texts=1000):
    service = get_embedding_service()
    texts = [f"This is test document number {i}" for i in range(num_texts)]

    start = time.time()
    embeddings = service.embed_batch(texts, batch_size=32)
    duration = time.time() - start

    print(f"Embedded {num_texts} texts in {duration:.2f}s")
    print(f"Speed: {num_texts/duration:.1f} texts/sec")

def benchmark_search(num_queries=100):
    store = VectorStore()
    queries = [f"query number {i}" for i in range(num_queries)]

    start = time.time()
    for query in queries:
        results = store.search(query, limit=5)
    duration = time.time() - start

    print(f"Performed {num_queries} searches in {duration:.2f}s")
    print(f"Speed: {num_queries/duration:.1f} searches/sec")

if __name__ == "__main__":
    benchmark_embeddings()
    benchmark_search()
```

---

## Project Ideas

Once you understand the system, try building:

### Project 1: Document Comparison Tool
Compare two documents to find similarities and differences

### Project 2: Topic Clustering
Group documents by topic using embeddings

### Project 3: Question Generator
Automatically generate questions from documents

### Project 4: Summarization Pipeline
Create summaries of long documents

### Project 5: Multi-language Support
Add support for documents in different languages

### Project 6: Web Interface
Build a React/Vue frontend for the API

### Project 7: Slack/Discord Bot
Create a bot that answers questions about your docs

### Project 8: PDF Highlighter
Highlight source text in original PDFs

---

## Mastery Checklist

Mark off as you complete each:

### Python
- [ ] Understand type hints and Pydantic
- [ ] Can write and run tests with pytest
- [ ] Understand async/await
- [ ] Can create CLI tools with Typer
- [ ] Can build APIs with FastAPI

### AI & ML
- [ ] Understand what embeddings are
- [ ] Can calculate cosine similarity
- [ ] Know when to use different models
- [ ] Understand batch processing

### RAG
- [ ] Can explain the RAG pipeline
- [ ] Understand chunking strategies
- [ ] Can engineer effective prompts
- [ ] Know how to optimize context

### Vector Databases
- [ ] Understand vector similarity search
- [ ] Can use Qdrant effectively
- [ ] Understand indexing and filtering
- [ ] Know performance tradeoffs

### Production
- [ ] Can deploy with Docker
- [ ] Understand API design
- [ ] Can write integration tests
- [ ] Know how to optimize performance

---

**Next**: Choose a project from the list above and build it! That's the best way to solidify your learning.
