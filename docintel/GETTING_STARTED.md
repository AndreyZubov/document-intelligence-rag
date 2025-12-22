# Getting Started with DocIntel

Welcome to DocIntel! This guide will help you set up and start learning Python, AI, RAG, and vector databases through this production-ready project.

## What You'll Learn

### Python
- Modern Python 3.10+ features (type hints, async/await)
- Project structure and packaging (pyproject.toml)
- Virtual environments and dependency management
- Error handling and logging
- CLI development with Typer
- API development with FastAPI

### AI & Embeddings
- **Sentence Transformers**: Convert text to vector embeddings
- **Semantic Similarity**: Understanding cosine similarity
- **Batch Processing**: Efficient embedding generation
- **Model Selection**: Choosing the right embedding model

### RAG (Retrieval Augmented Generation)
- **Document Chunking**: Breaking text into meaningful pieces
- **Vector Search**: Finding relevant context
- **Prompt Engineering**: Building effective prompts
- **Context Management**: Optimizing LLM inputs
- **Source Attribution**: Tracking where answers come from

### Vector Databases
- **Qdrant**: Production-grade vector database
- **Indexing**: Storing and organizing vectors
- **Similarity Search**: Finding nearest neighbors
- **Filtering**: Combining vector and metadata search
- **Collections**: Managing different document sets

## Prerequisites

- **Python 3.10 or higher**: Check with `python --version`
- **Docker**: For running Qdrant (optional but recommended)
- **API Key**: Either Anthropic (Claude) or OpenAI (GPT)
- **8GB RAM**: Minimum for embedding models

## Step-by-Step Setup

### 1. Initial Setup (Windows)

```bash
# Navigate to project
cd C:\GitHub\01-Python\docintel

# Run setup script
scripts\setup.bat

# Or manual setup:
python -m venv venv
venv\Scripts\activate
pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

Edit `.env` and add your API key:
```env
# Choose your LLM provider
LLM_PROVIDER=anthropic

# Add your API key
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxx

# Or use OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
```

### 3. Start Qdrant Vector Database

**Option A - Using Docker Compose (Recommended)**:
```bash
docker-compose up -d qdrant
```

**Option B - Using Docker directly**:
```bash
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
```

**Option C - Install locally**: Visit https://qdrant.tech/documentation/quick-start/

### 4. Verify Installation

```bash
# Activate virtual environment
venv\Scripts\activate

# Check system health
docintel health

# Should show:
# ✓ System is healthy
# Vector DB: Connected (localhost:6333)
# LLM Provider: anthropic
```

## Your First Document Query

### Step 1: Create a Sample Document

Create `my_first_doc.txt`:
```text
Python Machine Learning Guide

Python is a popular programming language for machine learning because of its
simple syntax and powerful libraries like NumPy, Pandas, and Scikit-learn.

Key Libraries:
- NumPy: Numerical computing
- Pandas: Data manipulation
- Scikit-learn: Machine learning algorithms
- TensorFlow: Deep learning
- PyTorch: Deep learning research

Machine learning workflows typically involve:
1. Data collection and preprocessing
2. Feature engineering
3. Model selection and training
4. Evaluation and tuning
5. Deployment to production
```

### Step 2: Upload the Document

```bash
docintel upload my_first_doc.txt
```

You should see:
```
Processing document: my_first_doc.txt
Success!
Document ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
Chunks indexed: 3
```

### Step 3: Ask Questions

```bash
docintel query "What libraries are used for machine learning in Python?"
```

You'll get an AI-generated answer based on your document!

Try more queries:
```bash
docintel query "What are the steps in a machine learning workflow?"
docintel query "Why is Python popular for machine learning?"
```

### Step 4: View Statistics

```bash
docintel stats
```

## Understanding How It Works

### The RAG Pipeline

```
Your Question
    ↓
[1] Convert question to vector (embedding)
    ↓
[2] Search vector database for similar chunks
    ↓
[3] Retrieve top 5 most relevant chunks
    ↓
[4] Build prompt with question + context
    ↓
[5] Send to LLM (Claude/GPT)
    ↓
[6] Get answer with sources
```

### Key Components

1. **Document Processor** (`document_processor.py`)
   - Extracts text from PDF, DOCX, TXT, MD
   - Splits into chunks (default: 512 characters)
   - Preserves context with overlap

2. **Embedding Service** (`embeddings.py`)
   - Converts text to 384-dimensional vectors
   - Uses Sentence-BERT models
   - Batch processing for efficiency

3. **Vector Store** (`vector_store.py`)
   - Stores embeddings in Qdrant
   - Performs similarity search
   - Returns most relevant chunks

4. **RAG Engine** (`rag_engine.py`)
   - Orchestrates retrieval + generation
   - Builds prompts with context
   - Queries LLM for answers

## Learning Exercises

### Exercise 1: Explore Document Processing

```bash
# Open Python REPL
python

# Try this:
from docintel.document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

text = "Your sample text here..." * 50
from uuid import uuid4
chunks = processor.create_chunks(text, uuid4())

print(f"Created {len(chunks)} chunks")
for chunk in chunks[:3]:
    print(f"\nChunk {chunk.chunk_index}:")
    print(chunk.content[:100] + "...")
```

**Learn**: How does chunk size affect the number of chunks? What happens with different overlap values?

### Exercise 2: Understanding Embeddings

```bash
python

from docintel.embeddings import get_embedding_service

service = get_embedding_service()

# Embed similar sentences
emb1 = service.embed_text("Machine learning is fascinating")
emb2 = service.embed_text("AI and ML are interesting")
emb3 = service.embed_text("I like pizza")

# Calculate similarity
sim_12 = service.compute_similarity(emb1, emb2)
sim_13 = service.compute_similarity(emb1, emb3)

print(f"ML vs AI similarity: {sim_12:.3f}")
print(f"ML vs Pizza similarity: {sim_13:.3f}")
```

**Learn**: Similar concepts have higher cosine similarity scores!

### Exercise 3: Vector Search

```bash
python

from docintel.vector_store import VectorStore

store = VectorStore()

# Search for concepts
results = store.search("Python libraries", limit=3)

for result in results:
    print(f"\nScore: {result['score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
```

**Learn**: How does the search find relevant content without exact keyword matches?

### Exercise 4: Running the API

```bash
# Start the server
docintel serve
```

Visit http://localhost:8000/docs to see interactive API documentation.

Try API calls:
```bash
# Upload via API
curl -X POST "http://localhost:8000/documents/upload" -F "file=@my_first_doc.txt"

# Query via API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is Python?\"}"

# Get stats
curl http://localhost:8000/stats
```

**Learn**: RESTful API design, FastAPI documentation, async request handling

### Exercise 5: Modify the RAG Prompt

Open `src/docintel/rag_engine.py` and find `_build_prompt()`.

Try modifying the prompt to:
- Make answers more concise
- Add a specific tone (friendly, technical, etc.)
- Request bullet-point format
- Ask for citations in a specific format

**Learn**: Prompt engineering directly affects answer quality!

## Common Experiments

### Experiment 1: Change Chunk Size

Edit `.env`:
```env
CHUNK_SIZE=256
CHUNK_OVERLAP=30
```

Restart and re-upload documents. How does this affect answers?

### Experiment 2: Try Different Embedding Models

Edit `src/docintel/embeddings.py`:
```python
# Change in EmbeddingService.__init__:
# Try: "sentence-transformers/all-mpnet-base-v2" (better quality, slower)
# Try: "sentence-transformers/paraphrase-MiniLM-L3-v2" (faster, smaller)
```

### Experiment 3: Adjust Search Parameters

In `src/docintel/vector_store.py`, modify `search()`:
```python
# Try different score thresholds
score_threshold=0.5  # Higher = stricter relevance
limit=10  # More context chunks
```

### Experiment 4: Multiple Documents

Upload different documents and see how queries work across them:
```bash
docintel upload python_guide.pdf
docintel upload machine_learning.pdf
docintel upload data_science.txt

docintel query "Compare approaches to data preprocessing"
```

## Programmatic Usage

See `example_usage.py` for a complete example, or try this:

```python
import asyncio
from docintel.vector_store import VectorStore
from docintel.rag_engine import RAGEngine
from docintel.models import QueryRequest

async def main():
    # Initialize
    vector_store = VectorStore()
    rag_engine = RAGEngine(vector_store=vector_store)

    # Query
    request = QueryRequest(
        query="What is machine learning?",
        max_results=5
    )

    response = await rag_engine.aquery(request)

    print(f"Answer: {response.answer}")
    print(f"\nSources: {len(response.sources)}")
    print(f"Time: {response.processing_time:.2f}s")

asyncio.run(main())
```

## Troubleshooting

### Issue: "Vector DB not connected"
```bash
# Check if Qdrant is running
docker ps | findstr qdrant

# Or visit: http://localhost:6333/dashboard

# Restart if needed
docker restart qdrant
```

### Issue: "API key not set"
```bash
# Verify .env file exists
dir .env

# Check environment variable is loaded
python -c "from docintel.config import settings; print(settings.anthropic_api_key[:10])"
```

### Issue: "No results found"
- Ensure documents are uploaded: `docintel stats`
- Try broader queries
- Lower score threshold in `vector_store.py`

### Issue: "Out of memory"
- Use smaller embedding model
- Reduce batch size in `embeddings.py`
- Increase system RAM or use cloud instance

## Next Steps

### Beginner Path
1. ✅ Complete setup and run first query
2. 📖 Read `ARCHITECTURE.md` to understand system design
3. 🔬 Try all exercises in this guide
4. 📝 Upload your own documents and experiment
5. 🎨 Modify prompts and parameters

### Intermediate Path
1. 📚 Read all source code files
2. ✏️ Add new document format support (HTML, CSV)
3. 🔄 Implement query result caching
4. 📊 Add document metadata (tags, dates)
5. 🧪 Write more unit tests

### Advanced Path
1. 🚀 Add streaming responses for LLM
2. 💾 Implement conversation memory
3. 🔍 Add hybrid search (semantic + keyword)
4. 🎯 Fine-tune embeddings for your domain
5. ☁️ Deploy to production (AWS/GCP/Azure)

## Learning Resources

### Python
- Official Docs: https://docs.python.org/3/
- FastAPI: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/

### Machine Learning
- Sentence Transformers: https://www.sbert.net/
- Embeddings Guide: https://platform.openai.com/docs/guides/embeddings

### RAG
- LangChain Docs: https://python.langchain.com/docs/
- RAG Overview: https://www.anthropic.com/index/contextual-retrieval

### Vector Databases
- Qdrant Docs: https://qdrant.tech/documentation/
- Vector Search Basics: https://www.pinecone.io/learn/vector-database/

## Community & Support

- Read issues and discussions on GitHub
- Check `ARCHITECTURE.md` for design decisions
- Review test files for usage examples
- Experiment and break things - that's how you learn!

## Daily Learning Plan

### Week 1: Basics
- Day 1-2: Setup, first queries, understand workflow
- Day 3-4: Read and understand each module
- Day 5-7: Complete all exercises, experiment with parameters

### Week 2: Deep Dive
- Day 1-2: Study embeddings, try different models
- Day 3-4: Explore vector search, understand similarity
- Day 5-7: Modify RAG prompts, optimize performance

### Week 3: Extension
- Day 1-3: Add new features (metadata, caching, etc.)
- Day 4-5: Write tests for your new features
- Day 6-7: Document your changes, create a demo

### Week 4: Production
- Day 1-2: Deploy with Docker Compose
- Day 3-4: Add monitoring and logging
- Day 5-7: Build a simple web UI or integrate with existing app

---

**Remember**: This is a real production system. Every pattern, every design choice has a reason. As you learn, ask yourself "why was it built this way?" - the answers will teach you professional software engineering.

Good luck and happy learning! 🚀
