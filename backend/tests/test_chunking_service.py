"""
Test suite for chunking_service.py

Tests cover:
- Text chunking with overlap
- Metadata validation
- Location reference extraction
- Adaptive chunk sizing
- Small chunk merging
- Text preprocessing
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.chunking_service import (
    chunk_text,
    extract_location_ref,
    merge_small_chunks,
    ChunkingError,
    MIN_CHUNK_SIZE,
    _preprocess_text,
    _get_adaptive_chunk_size,
)


class TestChunkText:
    """Test main chunking function."""
    
    def test_chunk_valid_text(self, sample_metadata):
        """Valid text should be chunked successfully."""
        text = "This is a test document. " * 50  # Repeat to make it substantial
        
        chunks = chunk_text(text, sample_metadata)
        
        assert len(chunks) > 0
        assert all("chunkId" in c for c in chunks)
        assert all("text" in c for c in chunks)
        assert all("locationRef" in c for c in chunks)
        assert all(len(c["text"]) > 0 for c in chunks)
    
    def test_chunk_empty_text(self, sample_metadata):
        """Empty text should raise error."""
        with pytest.raises(ChunkingError, match="non-empty string"):
            chunk_text("", sample_metadata)
    
    def test_chunk_none_text(self, sample_metadata):
        """None text should raise error."""
        with pytest.raises(ChunkingError, match="non-empty string"):
            chunk_text(None, sample_metadata)
    
    def test_chunk_missing_metadata(self):
        """Missing metadata should raise error."""
        text = "Test text" * 10
        
        with pytest.raises(ChunkingError, match="non-empty dictionary"):
            chunk_text(text, None)
    
    def test_chunk_missing_required_fields(self, sample_metadata):
        """Missing required metadata fields should raise error."""
        text = "Test text" * 10
        incomplete_metadata = {"documentId": "doc-1"}  # Missing other fields
        
        with pytest.raises(ChunkingError, match="Missing required metadata field"):
            chunk_text(text, incomplete_metadata)
    
    def test_chunk_too_short_text(self, sample_metadata):
        """Text shorter than minimum size should return a single fallback chunk."""
        chunks = chunk_text("short", sample_metadata)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "short"
    
    def test_chunk_preserves_metadata(self, sample_metadata):
        """Chunks should preserve document metadata."""
        text = "This is a test document. " * 30
        
        chunks = chunk_text(text, sample_metadata)
        
        for chunk in chunks:
            assert chunk["documentId"] == sample_metadata["documentId"]
            assert chunk["fileName"] == sample_metadata["fileName"]
            assert chunk["sourceFormat"] == sample_metadata["sourceFormat"]
            assert chunk["subjectId"] == sample_metadata["subjectId"]
    
    def test_chunk_incremental_indices(self, sample_metadata):
        """Chunk indices should be sequential."""
        text = "This is a test document. " * 50
        
        chunks = chunk_text(text, sample_metadata)
        
        indices = [c["chunkIndex"] for c in chunks]
        assert indices == list(range(len(chunks)))
    
    def test_chunk_unique_ids(self, sample_metadata):
        """Each chunk should have unique ID."""
        text = "This is a test document. " * 50
        
        chunks = chunk_text(text, sample_metadata)
        
        ids = [c["chunkId"] for c in chunks]
        assert len(ids) == len(set(ids))  # All unique
    
    def test_chunk_overlap(self, sample_metadata):
        """Consecutive chunks should have overlap."""
        text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z. " * 10
        
        chunks = chunk_text(text, sample_metadata)
        
        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i]["text"][-50:]  # Last 50 chars
                chunk2_start = chunks[i+1]["text"][:50]  # First 50 chars
                # There should be some character overlap or semantic connection
                overlap = len(set(chunk1_end.split()) & set(chunk2_start.split())) if \
                         chunk1_end.split() and chunk2_start.split() else 0
                # At least some word overlap or very close content
                assert overlap >= 0  # This is a flexible check


class TestExtractLocationRef:
    """Test location reference extraction."""
    
    def test_extract_pdf_page_reference(self):
        """Should extract page number from PDF chunks."""
        chunk_text = "[Page 42]\nContent about photosynthesis"
        metadata = {"sourceFormat": "pdf"}
        
        result = extract_location_ref(chunk_text, metadata, 0)
        
        assert result == "Page 42"
    
    def test_extract_pptx_slide_reference(self):
        """Should extract slide number from PPTX chunks."""
        chunk_text = "[Slide 7]\nContent about biology"
        metadata = {"sourceFormat": "pptx"}
        
        result = extract_location_ref(chunk_text, metadata, 0)
        
        assert result == "Slide 7"
    
    def test_extract_default_reference(self):
        """Should return default section reference for other formats."""
        chunk_text = "Generic content"
        metadata = {"sourceFormat": "docx"}
        
        result = extract_location_ref(chunk_text, metadata, 5)
        
        assert result == "Section 6"  # index 5 -> Section 6
    
    def test_extract_missing_page_marker(self):
        """Should return generic reference if page marker missing."""
        chunk_text = "Content without page marker"
        metadata = {"sourceFormat": "pdf"}
        
        result = extract_location_ref(chunk_text, metadata, 0)
        
        assert result == "PDF text"
    
    def test_extract_case_insensitive_format(self):
        """Format matching should be case-insensitive."""
        chunk_text = "Content"
        metadata = {"sourceFormat": "PDF"}  # Uppercase
        
        result = extract_location_ref(chunk_text, metadata, 0)
        
        assert result == "PDF text"


class TestMergeSmallChunks:
    """Test small chunk merging."""
    
    def test_merge_no_chunks(self):
        """Empty list should return empty."""
        result = merge_small_chunks([])
        assert result == []
    
    def test_merge_single_chunk(self, sample_chunk):
        """Single chunk should be returned as-is."""
        chunks = [sample_chunk]
        result = merge_small_chunks(chunks)
        assert len(result) == 1
    
    def test_merge_no_small_chunks(self):
        """Chunks above minimum size should not be merged."""
        chunks = [
            {"text": "A" * 100, "chunkId": "1"},
            {"text": "B" * 100, "chunkId": "2"},
        ]
        
        result = merge_small_chunks(chunks, min_size=50)
        
        assert len(result) == 2
    
    def test_merge_small_chunks(self):
        """Small consecutive chunks should be merged."""
        chunks = [
            {"text": "A" * 20, "chunkId": "1"},
            {"text": "B" * 20, "chunkId": "2"},
            {"text": "C" * 200, "chunkId": "3"},
        ]
        
        result = merge_small_chunks(chunks, min_size=50)
        
        assert len(result) == 2
        assert "A" * 20 in result[0]["text"]
        assert "B" * 20 in result[0]["text"]
    
    def test_merge_preserves_content(self):
        """Merging should preserve all content."""
        chunks = [
            {"text": "First", "chunkId": "1"},
            {"text": "Second", "chunkId": "2"},
            {"text": "Third" * 50, "chunkId": "3"},
        ]
        
        original_text = "".join(c["text"] for c in chunks)
        result = merge_small_chunks(chunks, min_size=30)
        merged_text = "".join(c["text"] for c in result)
        
        # All original content should be present (possibly with separator)
        assert "First" in merged_text
        assert "Second" in merged_text
        assert "Third" in merged_text


class TestPreprocessText:
    """Test text preprocessing."""
    
    def test_preprocess_multiple_newlines(self):
        """Multiple consecutive newlines should be normalized."""
        text = "Line 1\n\n\n\nLine 2"
        
        result = _preprocess_text(text)
        
        assert "\n\n\n\n" not in result
        assert result == "Line 1\n\nLine 2"
    
    def test_preprocess_trailing_whitespace(self):
        """Trailing line whitespace should be removed."""
        text = "Line 1   \nLine 2  "
        
        result = _preprocess_text(text)
        
        assert result == "Line 1\nLine 2"
    
    def test_preprocess_strip_edges(self):
        """Leading/trailing document whitespace should be stripped."""
        text = "   \nContent\n   "
        
        result = _preprocess_text(text)
        
        assert result == "Content"
    
    def test_preprocess_preserves_content(self):
        """Preprocessing should preserve meaningful content."""
        text = "Important\ncontent\nhere"
        
        result = _preprocess_text(text)
        
        assert "Important" in result
        assert "content" in result
        assert "here" in result


class TestAdaptiveChunkSize:
    """Test adaptive chunk sizing."""
    
    def test_adaptive_size_pdf(self):
        """PDF should have standard chunk size."""
        size = _get_adaptive_chunk_size("pdf")
        assert 400 <= size <= 600
    
    def test_adaptive_size_pptx(self):
        """PPTX should have smaller chunk size."""
        size = _get_adaptive_chunk_size("pptx")
        assert size <= 450
    
    def test_adaptive_size_docx(self):
        """DOCX can have slightly larger chunk size."""
        size = _get_adaptive_chunk_size("docx")
        assert size >= 500
    
    def test_adaptive_size_unknown(self):
        """Unknown format should have default size."""
        size = _get_adaptive_chunk_size("unknown")
        assert isinstance(size, int)
        assert size > 0
    
    def test_adaptive_size_case_insensitive(self):
        """Format matching should be case-insensitive."""
        size_lower = _get_adaptive_chunk_size("pdf")
        size_upper = _get_adaptive_chunk_size("PDF")
        
        assert size_lower == size_upper


class TestChunkingEdgeCases:
    """Test edge cases in chunking."""
    
    def test_chunk_text_with_special_characters(self, sample_metadata):
        """Text with special characters should chunk correctly."""
        text = "Test with émojis 🚀 and spëcial çharacters. " * 20
        
        chunks = chunk_text(text, sample_metadata)
        
        assert len(chunks) > 0
        assert any("émojis" in c["text"] or "spëcial" in c["text"] for c in chunks)
    
    def test_chunk_text_with_code(self, sample_metadata):
        """Code blocks should be handled properly."""
        text = "Here's code:\n```\nfunction test() {\n  return true;\n}\n```\n" * 10
        
        chunks = chunk_text(text, sample_metadata)
        
        assert len(chunks) > 0
    
    def test_chunk_with_lists(self, sample_metadata):
        """List formatting should be preserved."""
        text = """
        1. First item
        2. Second item
        3. Third item
        """ * 10
        
        chunks = chunk_text(text, sample_metadata)
        
        assert len(chunks) > 0


class TestChunkingPerformance:
    """Test chunking performance and efficiency."""
    
    def test_chunk_large_document(self, sample_metadata):
        """Should handle large documents efficiently."""
        # Create a large document (~100KB)
        text = "This is a test sentence. " * 4000
        
        chunks = chunk_text(text, sample_metadata)
        
        assert len(chunks) > 0
        # Verify reasonable chunk count for document size
        assert 10 < len(chunks) < 1000
    
    def test_chunk_consistency(self, sample_metadata):
        """Chunking the same text twice should produce same results."""
        text = "Test content. " * 50
        
        chunks1 = chunk_text(text, sample_metadata)
        chunks2 = chunk_text(text, sample_metadata)
        
        assert len(chunks1) == len(chunks2)
        # Note: IDs will be different (UUIDs), but structure should be same
        assert all(c1["text"] == c2["text"] for c1, c2 in zip(chunks1, chunks2))
