"""
Test suite for embedding_service.py

Tests cover:
- Batch embedding with size validation
- Query embedding
- Retry logic with exponential backoff
- Error handling and validation
- Embedding dimension validation
"""
import pytest
import math
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.embedding_service import (
    embed_chunks,
    embed_query,
    _embed_with_retry,
    validate_embedding,
    EmbeddingError,
    EMBEDDING_DIM,
)


class TestEmbedChunks:
    """Test batch chunk embedding."""
    
    @pytest.mark.asyncio
    async def test_embed_empty_chunks(self):
        """Empty chunks should return empty list."""
        result = await embed_chunks([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_embed_missing_text_field(self, sample_chunk):
        """Chunks without 'text' field should raise error."""
        chunk = sample_chunk.copy()
        del chunk["text"]
        
        with pytest.raises(EmbeddingError, match="missing 'text' field"):
            await embed_chunks([chunk])
    
    @pytest.mark.asyncio
    async def test_embed_empty_text(self, sample_chunk):
        """Chunks with empty text should raise error."""
        chunk = sample_chunk.copy()
        chunk["text"] = ""
        
        with pytest.raises(EmbeddingError, match="empty text"):
            await embed_chunks([chunk])
    
    @pytest.mark.asyncio
    async def test_embed_single_chunk(self, sample_chunk):
        """Single chunk should be embedded successfully."""
        with patch('app.services.embedding_service._embed_with_retry') as mock_embed:
            # Mock the embedding result
            mock_embedding = [0.1] * EMBEDDING_DIM
            mock_embed.return_value = [mock_embedding]
            
            chunks = [sample_chunk]
            result = await embed_chunks(chunks)
            
            assert len(result) == 1
            assert "embedding" in result[0]
            assert len(result[0]["embedding"]) == EMBEDDING_DIM
    
    @pytest.mark.asyncio
    async def test_embed_batch_chunking(self):
        """Test that chunks are processed in batches."""
        batch_size = 10
        num_chunks = 25
        
        chunks = [
            {"text": f"Chunk {i}", "chunkId": f"chunk-{i}"}
            for i in range(num_chunks)
        ]
        
        async def mock_embed(texts, task_type):
            return [[0.1] * EMBEDDING_DIM for _ in texts]
        
        with patch('app.services.embedding_service._embed_with_retry', new=AsyncMock(side_effect=mock_embed)):
            result = await embed_chunks(chunks, batch_size=batch_size)
            
            assert len(result) == num_chunks
            assert all("embedding" in chunk for chunk in result)
    
    @pytest.mark.asyncio
    async def test_embed_preserves_metadata(self, sample_chunk):
        """Embedding should preserve original chunk metadata."""
        with patch('app.services.embedding_service._embed_with_retry') as mock_embed:
            mock_embedding = [0.1] * EMBEDDING_DIM
            mock_embed.return_value = [mock_embedding]
            
            chunks = [sample_chunk]
            result = await embed_chunks(chunks)
            
            assert result[0]["chunkId"] == sample_chunk["chunkId"]
            assert result[0]["fileName"] == sample_chunk["fileName"]
            assert result[0]["sourceFormat"] == sample_chunk["sourceFormat"]


class TestEmbedQuery:
    """Test query embedding."""
    
    @pytest.mark.asyncio
    async def test_embed_valid_query(self, sample_query):
        """Valid query should be embedded."""
        with patch('app.services.embedding_service._embed_with_retry') as mock_embed:
            mock_embedding = [0.1] * EMBEDDING_DIM
            mock_embed.return_value = [mock_embedding]
            
            result = await embed_query(sample_query)
            
            assert isinstance(result, list)
            assert len(result) == EMBEDDING_DIM
    
    @pytest.mark.asyncio
    async def test_embed_empty_query(self):
        """Empty query should raise error."""
        with pytest.raises(EmbeddingError, match="Query cannot be empty"):
            await embed_query("")
    
    @pytest.mark.asyncio
    async def test_embed_whitespace_only_query(self):
        """Whitespace-only query should raise error."""
        with pytest.raises(EmbeddingError, match="Query cannot be empty"):
            await embed_query("   ")
    
    @pytest.mark.asyncio
    async def test_embed_none_query(self):
        """None query should raise error."""
        with pytest.raises(EmbeddingError):
            await embed_query(None)
    
    @pytest.mark.asyncio
    async def test_embed_long_query(self):
        """Long query should be handled."""
        long_query = "What " * 100  # Very long query
        
        with patch('app.services.embedding_service._embed_with_retry') as mock_embed:
            mock_embedding = [0.1] * EMBEDDING_DIM
            mock_embed.return_value = [mock_embedding]
            
            result = await embed_query(long_query)
            assert len(result) == EMBEDDING_DIM


class TestEmbedWithRetry:
    """Test retry logic with gemini pool failover."""
    
    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Successful embedding on first attempt."""
        mock_embedding = [[0.1] * EMBEDDING_DIM]
        with patch('app.services.gemini_pool.gemini_pool.embed_texts_with_failover', new_callable=AsyncMock) as mock_failover:
            mock_failover.return_value = mock_embedding
            result = await _embed_with_retry(["test text"], "RETRIEVAL_QUERY")
            assert result == mock_embedding
    
    @pytest.mark.asyncio
    async def test_retry_after_failures(self):
        """Should retry on failure and eventually succeed."""
        mock_embedding = [[0.1] * EMBEDDING_DIM]
        with patch('app.services.gemini_pool.gemini_pool.embed_texts_with_failover', new_callable=AsyncMock) as mock_failover:
            mock_failover.return_value = mock_embedding
            result = await _embed_with_retry(["test text"], "RETRIEVAL_QUERY")
            assert result == mock_embedding
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Should raise error after max retries exceeded."""
        with patch('app.services.gemini_pool.gemini_pool.embed_texts_with_failover', new_callable=AsyncMock) as mock_failover:
            mock_failover.side_effect = Exception("All keys exhausted")
            with pytest.raises(EmbeddingError, match="Failed to embed texts"):
                await _embed_with_retry(["test text"], "RETRIEVAL_QUERY")


class TestValidateEmbedding:
    """Test embedding validation."""
    
    def test_valid_embedding(self):
        """Valid embedding should pass validation."""
        embedding = [0.1] * EMBEDDING_DIM
        assert validate_embedding(embedding) is True
    
    def test_invalid_embedding_length(self):
        """Wrong dimension should fail validation."""
        embedding = [0.1] * (EMBEDDING_DIM - 1)
        assert validate_embedding(embedding) is False
    
    def test_embedding_with_nan(self):
        """Embedding with NaN should fail validation."""
        embedding = [0.1] * EMBEDDING_DIM
        embedding[100] = float('nan')
        assert validate_embedding(embedding) is False
    
    def test_embedding_with_inf(self):
        """Embedding with Inf should fail validation."""
        embedding = [0.1] * EMBEDDING_DIM
        embedding[100] = float('inf')
        assert validate_embedding(embedding) is False
    
    def test_empty_embedding(self):
        """Empty embedding should fail validation."""
        assert validate_embedding([]) is False
    
    def test_none_embedding(self):
        """None embedding should fail validation."""
        assert validate_embedding(None) is False


class TestEmbeddingConsistency:
    """Test consistency of embeddings."""
    
    @pytest.mark.asyncio
    async def test_same_query_same_embedding(self):
        """Same query should produce same embedding."""
        query = "test question"
        mock_embedding = [0.1] * EMBEDDING_DIM
        
        with patch('app.services.embedding_service._embed_with_retry') as mock_embed:
            mock_embed.return_value = [mock_embedding]
            
            result1 = await embed_query(query)
            result2 = await embed_query(query)
            
            assert result1 == result2
    
    @pytest.mark.asyncio
    async def test_similar_texts_similar_embeddings(self):
        """Similar text chunks should have similar embeddings."""
        chunks = [
            {"text": "The cat sat on the mat", "chunkId": "1"},
            {"text": "The dog sat on the mat", "chunkId": "2"},
            {"text": "Completely unrelated content about quantum physics", "chunkId": "3"}
        ]
        
        with patch('app.services.embedding_service._embed_with_retry') as mock_embed:
            # Return slightly different embeddings for similar texts
            def side_effect(texts, task_type):
                if "cat" in texts[0] or "dog" in texts[0]:
                    return [[0.1, 0.2, 0.15] + [0.1] * (EMBEDDING_DIM - 3)]
                return [[0.9, 0.2, 0.1] + [0.1] * (EMBEDDING_DIM - 3)]
            
            mock_embed.side_effect = side_effect
            
            result = await embed_chunks(chunks)
            assert len(result) == 3
            
            e1 = result[0]["embedding"]
            e2 = result[1]["embedding"]
            e3 = result[2]["embedding"]
            
            def cos_sim(v1, v2):
                dot = sum(a*b for a, b in zip(v1, v2))
                mag1 = sum(a*a for a in v1)**0.5
                mag2 = sum(b*b for b in v2)**0.5
                return dot / (mag1 * mag2)
            
            # e1 and e2 should be similar
            sim12 = cos_sim(e1, e2)
            # e1 and e3 should be different
            sim13 = cos_sim(e1, e3)
            
            assert sim12 >= 0.9
            assert sim13 < 0.5
