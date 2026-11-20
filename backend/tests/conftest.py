"""
Shared pytest configuration and fixtures for all tests.
"""
import pytest
import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def sample_query():
    """Sample user query."""
    return "What is photosynthesis?"

@pytest.fixture
def sample_chunk():
    """Sample document chunk."""
    return {
        "chunkId": "chunk-123",
        "text": "Photosynthesis is the process by which plants convert light energy into chemical energy. "
                "It occurs mainly in the leaves of plants and involves the use of chlorophyll to "
                "absorb light energy from the sun.",
        "documentId": "doc-001",
        "fileName": "biology_notes.pdf",
        "sourceFormat": "pdf",
        "subjectId": "subj-001",
        "locationRef": "Page 42",
        "chunkIndex": 5
    }

@pytest.fixture
def sample_chunks(sample_chunk):
    """Multiple sample chunks."""
    chunks = [sample_chunk.copy()]
    for i in range(1, 5):
        chunk = sample_chunk.copy()
        chunk["chunkId"] = f"chunk-{i+123}"
        chunk["chunkIndex"] = i + 5
        chunks.append(chunk)
    return chunks

@pytest.fixture
def sample_metadata():
    """Sample document metadata."""
    return {
        "documentId": "doc-001",
        "fileName": "biology_notes.pdf",
        "sourceFormat": "pdf",
        "subjectId": "subj-001"
    }
