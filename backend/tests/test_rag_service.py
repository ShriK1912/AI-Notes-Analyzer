"""
Test suite for rag_service.py

Tests cover:
- Query preprocessing
- Confidence scoring with enhanced logic
- Citation extraction and validation
- Source deduplication
- End-to-end RAG pipeline
- Error handling and edge cases
"""
import pytest
import sys
import os
from unittest.mock import patch, AsyncMock, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.rag_service import (
    ask_question,
    compute_confidence,
    extract_citations,
    build_sources_block,
    _preprocess_query,
    _deduplicate_chunks,
    RAGError,
    CONFIDENCE_THRESHOLDS,
)


class TestPreprocessQuery:
    """Test query preprocessing."""
    
    def test_preprocess_normal_query(self):
        """Normal query should be cleaned."""
        query = "What   is   photosynthesis?"
        
        cleaned, keywords = _preprocess_query(query)
        
        assert "  " not in cleaned  # No double spaces
        assert len(keywords) > 0
    
    def test_preprocess_removes_leading_punctuation(self):
        """Leading punctuation should be removed."""
        query = "???What is photosynthesis"
        
        cleaned, keywords = _preprocess_query(query)
        
        assert not cleaned.startswith("?")
    
    def test_preprocess_extracts_keywords(self):
        """Should extract meaningful keywords."""
        query = "What is the process of photosynthesis?"
        
        cleaned, keywords = _preprocess_query(query)
        
        assert "photosynthesis" in keywords
        assert "process" in keywords
        assert "is" not in keywords  # Stop word
        assert "the" not in keywords  # Stop word
    
    def test_preprocess_empty_query(self):
        """Empty query should raise error."""
        with pytest.raises(RAGError, match="non-empty string"):
            _preprocess_query("")
    
    def test_preprocess_whitespace_only(self):
        """Whitespace-only query should raise error."""
        with pytest.raises(RAGError):
            _preprocess_query("   ")
    
    def test_preprocess_special_characters(self):
        """Query with special characters should be handled."""
        query = "What's the difference? (explain clearly!)"
        
        cleaned, keywords = _preprocess_query(query)
        
        assert isinstance(cleaned, str)
        assert len(keywords) > 0
    
    def test_preprocess_case_handling(self):
        """Keyword extraction should be case-insensitive."""
        query = "What is PHOTOSYNTHESIS?"
        
        cleaned, keywords = _preprocess_query(query)
        
        assert "photosynthesis" in keywords  # Lowercased


class TestComputeConfidence:
    """Test confidence scoring."""
    
    def test_confidence_empty_similarities(self):
        """Empty similarities should return NOT_FOUND."""
        result = compute_confidence([])
        
        assert result["tier"] == "NOT_FOUND"
        assert result["score"] == 0.0
    
    def test_confidence_high_similarities(self):
        """High similarities should give HIGH confidence."""
        similarities = [0.95, 0.93, 0.91, 0.92]
        
        result = compute_confidence(similarities)
        
        assert result["tier"] == "HIGH"
        assert result["score"] >= 0.85 # New formula is slightly more conservative
    
    def test_confidence_low_similarities(self):
        """Low similarities should give LOW or NOT_FOUND."""
        similarities = [0.55, 0.50, 0.52]
        
        result = compute_confidence(similarities)
        
        assert result["tier"] in ["LOW", "NOT_FOUND"]
        assert result["score"] < CONFIDENCE_THRESHOLDS["LOW"]
    
    def test_confidence_medium_similarities(self):
        """Medium similarities should give MEDIUM confidence."""
        similarities = [0.82, 0.80, 0.78]
        
        result = compute_confidence(similarities)
        
        assert result["tier"] in ["MEDIUM", "HIGH"]
        assert 0.75 <= result["score"] <= 0.95
    
    def test_confidence_single_high_outlier(self):
        """Single high match shouldn't over-inflate confidence if others low."""
        similarities = [0.95, 0.40, 0.35]
        
        result = compute_confidence(similarities)
        
        # Should be demoted due to low average
        assert result["score"] < 0.90
    
    def test_confidence_keyword_bonus(self):
        """Keyword matching should provide confidence bonus."""
        similarities = [0.75, 0.73, 0.74]
        chunk_texts = [
            "This discusses photosynthesis extensively",
            "Photosynthesis is the process",
            "More on photosynthesis"
        ]
        keywords = ["photosynthesis"]
        
        result = compute_confidence(similarities, keywords, chunk_texts)
        
        assert result["keywordBonus"] > 0
    
    def test_confidence_variance_penalty(self):
        """High variance in similarities should reduce confidence."""
        similarities_consistent = [0.80, 0.81, 0.79]
        similarities_inconsistent = [0.90, 0.50, 0.30]
        
        result1 = compute_confidence(similarities_consistent)
        result2 = compute_confidence(similarities_inconsistent)
        
        assert result1["score"] > result2["score"]
    
    def test_confidence_returns_all_diagnostics(self):
        """Should return full diagnostic info."""
        similarities = [0.85, 0.83]
        
        result = compute_confidence(similarities)
        
        assert "tier" in result
        assert "score" in result
        assert "maxScore" in result
        assert "avgScore" in result
        assert "minScore" in result
        assert "variance" in result


class TestExtractCitations:
    """Test citation extraction."""
    
    def test_extract_valid_citations(self, sample_chunks):
        """Valid citations should be extracted."""
        answer = """
        Based on the sources, [SOURCE: biology_notes.pdf, Page 42] contains information 
        about photosynthesis. Additional details are in [SOURCE: biology_notes.pdf, Page 43].
        """
        
        citations = extract_citations(answer, sample_chunks)
        
        assert len(citations) > 0
        assert all("fileName" in c for c in citations)
        assert all("locationRef" in c for c in citations)
    
    def test_extract_no_citations(self, sample_chunks):
        """Text without citations should return empty list."""
        answer = "This is an answer without any citations."
        
        citations = extract_citations(answer, sample_chunks)
        
        assert citations == []
    
    def test_extract_duplicate_citations(self, sample_chunks):
        """Duplicate citations should be deduplicated."""
        answer = """
        [SOURCE: biology_notes.pdf, Page 42] says one thing.
        [SOURCE: biology_notes.pdf, Page 42] also says another thing.
        """
        
        citations = extract_citations(answer, sample_chunks)
        
        # Should only have one citation (duplicate removed)
        assert len(citations) <= 1
    
    def test_extract_whitespace_in_citations(self, sample_chunks):
        """Extra whitespace in citations should be handled."""
        answer = "[SOURCE:   biology_notes.pdf  ,  Page 42  ]"
        
        citations = extract_citations(answer, sample_chunks)
        
        # Should still find and normalize the citation
        assert len(citations) > 0
    
    def test_extract_partial_filename_match(self):
        """Should match citations with partial filenames."""
        chunks = [{
            "fileName": "complete_path/biology_notes.pdf",
            "locationRef": "Page 42",
            "chunkId": "chunk-1",
            "sourceFormat": "pdf"
        }]
        
        answer = "[SOURCE: biology_notes.pdf, Page 42]"
        
        citations = extract_citations(answer, chunks)
        
        # Should still find match despite partial path
        assert len(citations) > 0


class TestBuildSourcesBlock:
    """Test sources block building."""
    
    def test_build_sources_block(self, sample_chunks):
        """Sources block should be properly formatted."""
        sources = build_sources_block(sample_chunks)
        
        assert "[SOURCE" in sources
        assert "File:" in sources
        assert "Location:" in sources
        assert "Format:" in sources
    
    def test_sources_block_includes_all_chunks(self, sample_chunks):
        """All chunks should be included in sources block."""
        sources = build_sources_block(sample_chunks)
        
        for chunk in sample_chunks:
            assert chunk["text"] in sources
    
    def test_sources_block_empty_chunks(self):
        """Empty chunk list should produce some output."""
        sources = build_sources_block([])
        
        # Should return something (even if just separators)
        assert isinstance(sources, str)


class TestDeduplicateChunks:
    """Test chunk deduplication."""
    
    def test_deduplicate_no_duplicates(self):
        """Unique chunks should not be deduplicated."""
        chunks = [
            {"text": "Unique content A", "chunkId": "1"},
            {"text": "Unique content B", "chunkId": "2"},
        ]
        result = _deduplicate_chunks(chunks)
        
        assert len(result) == 2
    
    def test_deduplicate_removes_duplicates(self):
        """Duplicate chunks should be removed."""
        chunks = [
            {"text": "Same content", "chunkId": "1"},
            {"text": "Same content", "chunkId": "2"},
            {"text": "Different content", "chunkId": "3"},
        ]
        
        result = _deduplicate_chunks(chunks)
        
        assert len(result) < len(chunks)
    
    def test_deduplicate_preserves_first(self):
        """Should preserve first occurrence of duplicate."""
        chunks = [
            {"text": "Content A", "chunkId": "1"},
            {"text": "Content A", "chunkId": "2"},
        ]
        
        result = _deduplicate_chunks(chunks)
        
        assert len(result) == 1
        assert result[0]["chunkId"] == "1"
    
    def test_deduplicate_empty_list(self):
        """Empty list should be handled."""
        result = _deduplicate_chunks([])
        assert result == []


class TestRAGPipeline:
    """Integration tests for RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_ask_question_invalid_query(self):
        """Invalid query should raise error."""
        with pytest.raises(RAGError):
            await ask_question("", "subj-1", "Biology", "user-1")
    
    @pytest.mark.asyncio
    async def test_ask_question_no_results(self):
        """Query with no results should return NOT_FOUND response."""
        with patch('app.services.rag_service.embed_queries', new_callable=AsyncMock) as mock_embed:
            with patch('app.vectorstore.firestore_client.find_similar_chunks') as mock_find:
                mock_embed.return_value = [[0.1] * 768]
                mock_find.return_value = []
                
                result = await ask_question("random question", "subj-1", "Biology", "user-1")
            
            assert result["confidenceTier"] == "NOT_FOUND"
            assert "Not found" in result["answer"]

    @pytest.mark.asyncio
    async def test_ask_question_embedding_failure(self):
        """Embedding failure should raise RAGError."""
        with patch('app.services.rag_service.embed_queries', new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = Exception("Embedding failed")
            
            with pytest.raises(RAGError):
                await ask_question("test question", "subj-1", "Biology", "user-1")

    @pytest.mark.asyncio
    async def test_ask_question_with_results(self):
        """Valid query with results should return answer."""
        with patch('app.services.rag_service.embed_queries', new_callable=AsyncMock) as mock_embed:
            with patch('app.vectorstore.firestore_client.find_similar_chunks') as mock_find:
                with patch('app.services.rag_service._generate_with_retry', new_callable=AsyncMock) as mock_gen:
                    mock_embed.return_value = [[0.1] * 768]
                    mock_find.return_value = [{
                        "id": "chunk-1",
                        "chunkId": "chunk-1",
                        "text": "Photosynthesis is the process...",
                        "fileName": "notes.pdf",
                        "locationRef": "Page 1",
                        "sourceFormat": "pdf",
                        "vector_distance": 0.1
                    }]
                    mock_gen.return_value = "[SOURCE: notes.pdf, Page 1] Photosynthesis is the chemical process..."
                    
                    result = await ask_question("What is photosynthesis?", "subj-1", "Biology", "user-1")
            
            assert result["answer"]
            assert "confidenceTier" in result
            assert result["confidenceTier"] != "NOT_FOUND"
            assert len(result["citations"]) > 0


class TestRAGEdgeCases:
    """Test edge cases in RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_ask_question_long_query(self):
        """Very long query should be handled."""
        query = "What is photosynthesis in detail? " * 10
        with patch('app.services.rag_service.embed_queries', new_callable=AsyncMock) as mock_embed:
            with patch('app.vectorstore.firestore_client.find_similar_chunks') as mock_find:
                with patch('app.services.rag_service._generate_with_retry', new_callable=AsyncMock) as mock_gen:
                    mock_embed.return_value = [[0.1] * 768]
                    mock_find.return_value = []
                    mock_gen.return_value = "Answer"
                    result = await ask_question(query, "subj-1", "Biology", "user-1")
                    assert result["confidenceTier"] == "NOT_FOUND"
    
    def test_confidence_extreme_values(self):
        """Extreme similarity values should be handled."""
        # Perfect match
        result = compute_confidence([1.0, 1.0, 1.0])
        assert result["score"] >= 0.95
        
        # No match
        result = compute_confidence([0.0, 0.0, 0.0])
        assert result["score"] <= 0.1
    
    def test_sources_block_special_characters(self):
        """Sources with special characters should be handled."""
        chunks = [{
            "text": "Content with émojis 🚀 and spëcial çharacters",
            "fileName": "special_file.pdf",
            "locationRef": "Page 1",
            "sourceFormat": "pdf"
        }]
        
        sources = build_sources_block(chunks)
        
        assert "special_file.pdf" in sources
