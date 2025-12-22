"""Tests for document processor."""

import io
from uuid import uuid4

import pytest

from docintel.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test document processing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    def test_extract_text_from_txt(self):
        """Test text extraction from plain text file."""
        content = b"This is a test document.\nIt has multiple lines."
        file = io.BytesIO(content)

        text = self.processor.extract_text(file, "test.txt")

        assert text == "This is a test document.\nIt has multiple lines."

    def test_extract_text_unsupported_format(self):
        """Test that unsupported formats raise ValueError."""
        file = io.BytesIO(b"content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            self.processor.extract_text(file, "test.xyz")

    def test_create_chunks(self):
        """Test document chunking."""
        text = "A" * 250  # Text longer than chunk size
        document_id = uuid4()

        chunks = self.processor.create_chunks(text, document_id)

        assert len(chunks) > 1
        assert all(chunk.document_id == document_id for chunk in chunks)
        assert all(len(chunk.content) <= self.processor.chunk_size for chunk in chunks)

    def test_create_chunks_empty_text(self):
        """Test chunking with empty text."""
        document_id = uuid4()

        chunks = self.processor.create_chunks("", document_id)

        assert len(chunks) == 0

    def test_create_chunks_preserves_content(self):
        """Test that chunking preserves all content."""
        text = "This is a test document. " * 50
        document_id = uuid4()

        chunks = self.processor.create_chunks(text, document_id)

        # Reconstruct text from chunks (without overlap)
        reconstructed_parts = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                reconstructed_parts.append(chunk.content)
            else:
                # Remove overlap from subsequent chunks
                start_pos = chunk.metadata["start_pos"]
                prev_end = chunks[i - 1].metadata["end_pos"]
                if start_pos < prev_end:
                    overlap_size = prev_end - start_pos
                    reconstructed_parts.append(chunk.content[overlap_size:])
                else:
                    reconstructed_parts.append(chunk.content)

        reconstructed = "".join(reconstructed_parts)
        assert text.strip() in reconstructed or reconstructed in text.strip()

    def test_process_document_too_short(self):
        """Test that very short documents raise ValueError."""
        content = b"Hi"
        file = io.BytesIO(content)

        with pytest.raises(ValueError, match="appears to be empty or too short"):
            self.processor.process_document(file, "test.txt")

    def test_process_document_success(self):
        """Test successful document processing."""
        content = b"This is a test document. " * 50
        file = io.BytesIO(content)

        document, chunks = self.processor.process_document(file, "test.txt")

        assert document.filename == "test.txt"
        assert document.chunk_count == len(chunks)
        assert len(chunks) > 0
        assert all(chunk.document_id == document.id for chunk in chunks)
