"""RAG (Retrieval Augmented Generation) engine with LLM integration."""

import logging
import time
from typing import Optional
from uuid import UUID

from anthropic import Anthropic
from openai import OpenAI

from docintel.config import settings
from docintel.vector_store import VectorStore
from docintel.models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


class RAGEngine:
    """Handles retrieval augmented generation using vector search and LLMs."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.llm_provider = settings.llm_provider

        # Initialize LLM clients
        if self.llm_provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
        elif self.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

        logger.info(f"RAG Engine initialized with provider: {self.llm_provider}")

    def _build_context(self, chunks: list[dict]) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Document {i}] (relevance: {chunk['score']:.2f})\n{chunk['content']}\n"
            )

        return "\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the LLM."""
        return f"""You are a helpful assistant answering questions based on provided documents.
Use the context below to answer the user's question. If the context doesn't contain enough information to answer the question, say so clearly.

Context from documents:
{context}

User question: {query}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite specific information from the documents when relevant
- If the context doesn't contain the answer, state that clearly
- Be concise but thorough
- Do not make up information not present in the context

Answer:"""

    def _query_anthropic(self, prompt: str) -> str:
        """Query Anthropic's Claude API."""
        logger.info(f"Querying Anthropic Claude ({settings.anthropic_model})")

        response = self.anthropic_client.messages.create(
            model=settings.anthropic_model,
            max_tokens=2048,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI's GPT API."""
        logger.info(f"Querying OpenAI GPT ({settings.openai_model})")

        response = self.openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )

        return response.choices[0].message.content

    def query(self, request: QueryRequest) -> QueryResponse:
        """Execute a RAG query: retrieve relevant chunks and generate answer."""
        start_time = time.time()

        logger.info(f"Processing query: {request.query[:100]}...")

        # Step 1: Retrieve relevant chunks
        chunks = self.vector_store.search(
            query=request.query, limit=request.max_results, document_ids=request.document_ids
        )

        if not chunks:
            logger.warning("No relevant chunks found for query")
            return QueryResponse(
                answer="I couldn't find any relevant information in the indexed documents to answer your question.",
                sources=[],
                query=request.query,
                processing_time=time.time() - start_time,
            )

        # Step 2: Build context and prompt
        context = self._build_context(chunks)
        prompt = self._build_prompt(request.query, context)

        # Step 3: Query LLM
        try:
            if self.llm_provider == "anthropic":
                answer = self._query_anthropic(prompt)
            else:
                answer = self._query_openai(prompt)
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            raise

        # Step 4: Format response
        processing_time = time.time() - start_time

        sources = [
            {
                "document_id": chunk["document_id"],
                "content_preview": chunk["content"][:200] + "..."
                if len(chunk["content"]) > 200
                else chunk["content"],
                "relevance_score": chunk["score"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ]

        logger.info(f"Query processed in {processing_time:.2f}s")

        return QueryResponse(
            answer=answer, sources=sources, query=request.query, processing_time=processing_time
        )

    async def aquery(self, request: QueryRequest) -> QueryResponse:
        """Async version of query (for FastAPI compatibility)."""
        # For now, just call sync version
        # In production, you'd use async clients for both Anthropic and OpenAI
        return self.query(request)
