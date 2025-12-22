"""Vector database operations using Qdrant."""

import logging
from typing import Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from docintel.config import settings
from docintel.models import DocumentChunk
from docintel.embeddings import get_embedding_service

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using Qdrant."""

    def __init__(
        self,
        host: str = settings.qdrant_host,
        port: int = settings.qdrant_port,
        collection_name: str = settings.qdrant_collection_name,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_service = get_embedding_service()

        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_service.dimension, distance=Distance.COSINE
                ),
            )
            logger.info(f"Collection {self.collection_name} created successfully")
        else:
            logger.info(f"Collection {self.collection_name} already exists")

    def health_check(self) -> bool:
        """Check if vector database is accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Vector database health check failed: {e}")
            return False

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Index document chunks into the vector store."""
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return 0

        logger.info(f"Indexing {len(chunks)} chunks")

        # Generate embeddings in batch
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(texts)

        # Create points for Qdrant
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "document_id": str(chunk.document_id),
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                },
            )
            points.append(point)

        # Upload to Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"Successfully indexed {len(points)} chunks")
        return len(points)

    def search(
        self,
        query: str,
        limit: int = settings.max_context_chunks,
        document_ids: Optional[list[UUID]] = None,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        """Search for relevant chunks using semantic similarity."""
        logger.info(f"Searching for query: {query[:100]}...")

        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Prepare filter if document_ids specified
        query_filter = None
        if document_ids:
            doc_id_strings = [str(doc_id) for doc_id in document_ids]
            query_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id_strings[0]))]
            )

        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "id": result.id,
                    "content": result.payload.get("content", ""),
                    "score": result.score,
                    "document_id": result.payload.get("document_id"),
                    "chunk_index": result.payload.get("chunk_index"),
                    "metadata": result.payload.get("metadata", {}),
                }
            )

        logger.info(f"Found {len(formatted_results)} relevant chunks")
        return formatted_results

    def delete_document(self, document_id: UUID) -> int:
        """Delete all chunks associated with a document."""
        logger.info(f"Deleting chunks for document: {document_id}")

        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=str(document_id)))]
            ),
        )

        logger.info(f"Deleted chunks for document {document_id}")
        return 1

    def count_documents(self) -> int:
        """Get total number of chunks in the collection."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return info.points_count

    def reset_collection(self) -> None:
        """Delete and recreate the collection (use with caution)."""
        logger.warning(f"Resetting collection: {self.collection_name}")
        self.client.delete_collection(collection_name=self.collection_name)
        self._initialize_collection()
        logger.info("Collection reset complete")
