"""Example usage of DocIntel programmatically."""

import asyncio
from pathlib import Path

from docintel.document_processor import DocumentProcessor
from docintel.vector_store import VectorStore
from docintel.rag_engine import RAGEngine
from docintel.models import QueryRequest


async def main():
    """Example workflow."""
    print("DocIntel - Example Usage\n" + "=" * 50)

    # Initialize components
    print("\n1. Initializing components...")
    processor = DocumentProcessor()
    vector_store = VectorStore()
    rag_engine = RAGEngine(vector_store=vector_store)

    # Process and index a document
    print("\n2. Processing document...")
    doc_path = Path("sample_document.txt")

    if not doc_path.exists():
        # Create a sample document
        sample_text = """
        Artificial Intelligence and Machine Learning

        Machine learning is a subset of artificial intelligence that focuses on enabling
        systems to learn and improve from experience without being explicitly programmed.

        Key Concepts:
        - Supervised Learning: Training with labeled data
        - Unsupervised Learning: Finding patterns in unlabeled data
        - Reinforcement Learning: Learning through rewards and penalties

        Deep learning, a subset of machine learning, uses neural networks with multiple
        layers to progressively extract higher-level features from raw input.

        Applications include computer vision, natural language processing, speech
        recognition, and autonomous vehicles.
        """
        doc_path.write_text(sample_text)
        print("   Created sample document: sample_document.txt")

    with open(doc_path, "rb") as f:
        document, chunks = processor.process_document(f, doc_path.name)

    print(f"   Document ID: {document.id}")
    print(f"   Chunks created: {len(chunks)}")

    # Index chunks
    print("\n3. Indexing chunks...")
    indexed_count = vector_store.index_chunks(chunks)
    print(f"   Indexed {indexed_count} chunks")

    # Query the document
    print("\n4. Querying documents...")
    queries = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "What are some applications of deep learning?",
    ]

    for query_text in queries:
        print(f"\n   Q: {query_text}")

        request = QueryRequest(query=query_text, max_results=3)
        response = await rag_engine.aquery(request)

        print(f"   A: {response.answer}")
        print(f"   ({len(response.sources)} sources, {response.processing_time:.2f}s)")

    # Show stats
    print("\n5. System Statistics")
    chunk_count = vector_store.count_documents()
    print(f"   Total chunks in database: {chunk_count}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
