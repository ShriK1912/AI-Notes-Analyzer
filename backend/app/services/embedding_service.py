from google import genai
from google.genai import types as genai_types
from app.core.config import settings
import asyncio
import math
import logging

logger = logging.getLogger(__name__)

print(f"DEBUG: embedding_service initializing with key: {settings.GEMINI_API_KEY[:10]}...")
_client = genai.Client(api_key=settings.GEMINI_API_KEY)

# Only model available in google.genai SDK (v1beta API)
EMBEDDING_MODELS = ["models/gemini-embedding-001"]
EMBEDDING_DIM = 768  # text-embedding-004 outputs 768-dim vectors
_active_embedding_dim = 0  # Will auto-detect from first successful embedding call
TASK_TYPE_DOC = "RETRIEVAL_DOCUMENT"
TASK_TYPE_QUERY = "RETRIEVAL_QUERY"
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

class EmbeddingError(Exception):
    """Custom exception for embedding operations."""
    pass

def _stabilize_fallback_embedding(embedding: list[float]) -> list[float]:
    """
    Stabilize fallback embeddings by mean-centering and L2 normalization.
    This is only used in rare mismatch fallback mode.
    """
    if not embedding:
        return embedding

    mean_val = sum(embedding) / len(embedding)
    centered = [x - mean_val for x in embedding]
    norm = math.sqrt(sum(x * x for x in centered))
    if norm == 0:
        return embedding
    return [x / norm for x in centered]

async def embed_chunks(chunks: list[dict], batch_size: int = 5) -> list[dict]:
    """
    Embed document chunks with RETRIEVAL_DOCUMENT task type for asymmetric retrieval.
    
    Args:
        chunks: List of chunk dictionaries with 'text' key
        batch_size: Number of chunks to embed per API call (default 5)
        
    Returns:
        List of chunks with 'embedding' field added
        
    Raises:
        EmbeddingError: If embedding fails after retries
    """
    if not chunks:
        return []
    
    logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}")
    
    # Validate chunks have required fields
    for chunk in chunks:
        if "text" not in chunk:
            raise EmbeddingError("Chunk missing 'text' field")
        if not chunk["text"].strip():
            raise EmbeddingError("Chunk contains empty text")
    
    # Process in batches to handle rate limits
    embedded_chunks = []
    texts = [c["text"] for c in chunks]
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_chunks = chunks[i:i + batch_size]
        
        embeddings = await _embed_with_retry(batch_texts, TASK_TYPE_DOC)

        # Defensive handling: some providers/mocks may return mismatched counts.
        if len(embeddings) != len(batch_chunks):
            logger.warning(
                "Embedding count mismatch (expected %d, got %d). Falling back to per-chunk embedding.",
                len(batch_chunks),
                len(embeddings),
            )
            embeddings = []
            for chunk in batch_chunks:
                single = await _embed_with_retry([chunk["text"]], TASK_TYPE_DOC)
                if not single:
                    raise EmbeddingError("Failed to embed chunk during fallback mode")
                embeddings.append(_stabilize_fallback_embedding(single[0]))
                await asyncio.sleep(1.0)  # Rate limiting
        
        for chunk, embedding in zip(batch_chunks, embeddings):
            if not validate_embedding(embedding):
                logger.warning(f"Invalid embedding detected for chunk {chunk.get('chunkId', 'unknown')}")
                raise EmbeddingError(f"Invalid embedding for chunk {chunk.get('chunkId', 'unknown')}")
            chunk["embedding"] = embedding
            embedded_chunks.append(chunk)

        # Sleep between batches strictly to respect Gemini Free Tier 15 RPM
        await asyncio.sleep(5.0)
            
    logger.info(f"Successfully embedded {len(embedded_chunks)} chunks")
    return embedded_chunks

async def embed_query(query: str) -> list[float]:
    """
    Embed a user query with RETRIEVAL_QUERY task type.
    
    Args:
        query: The user's question/query string
        
    Returns:
        Query embedding vector (list of floats)
        
    Raises:
        EmbeddingError: If embedding fails after retries
    """
    if not query or not query.strip():
        raise EmbeddingError("Query cannot be empty")
    
    logger.info(f"Embedding query: {query[:50]}...")
    
    embeddings = await _embed_with_retry([query], TASK_TYPE_QUERY)
    return embeddings[0]

async def embed_queries(queries: list[str]) -> list[list[float]]:
    """
    Embed multiple user queries with RETRIEVAL_QUERY task type in a single API call.
    
    Args:
        queries: List of query strings
        
    Returns:
        List of query embedding vectors
    """
    if not queries:
        return []
    
    logger.info(f"Embedding {len(queries)} queries at once...")
    return await _embed_with_retry(queries, TASK_TYPE_QUERY)

async def _embed_with_retry(texts: list[str], task_type: str) -> list[list[float]]:
    """
    Helper function to embed texts with exponential backoff retry logic.
    
    Args:
        texts: List of text strings to embed
        task_type: Either TASK_TYPE_DOC or TASK_TYPE_QUERY
        
    Returns:
        List of embedding vectors
        
    Raises:
        EmbeddingError: If all retries fail
    """
    last_error: Exception | None = None

    for model_name in EMBEDDING_MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda m=model_name: _client.models.embed_content(
                        model=m,
                        contents=texts,
                        config=genai_types.EmbedContentConfig(task_type=task_type),
                    )
                )
                embeddings = result.embeddings  # list of ContentEmbedding objects
                
                # Firestore has a hard limit of 2048 dimensions.
                # If a model like gemini-embedding-001 outputs 3072 dimensions, we MUST truncate it.
                vecs = []
                for e in embeddings:
                    val_list = list(e.values)
                    if len(val_list) > EMBEDDING_DIM:
                        val_list = val_list[:EMBEDDING_DIM]
                    vecs.append(val_list)
                    
                _sync_embedding_dim(vecs)
                return vecs

            except Exception as e:
                last_error = e
                error_text = str(e)
                logger.warning(
                    "Embedding attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    MAX_RETRIES,
                    model_name,
                    error_text,
                )

                # If model is not available for this API key/version, fail over quickly.
                if "not found" in error_text.lower() or "404" in error_text:
                    break

                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    
                    # Try to parse exact requested delay from Gemini 429 errors (e.g. "'retryDelay': '44s'")
                    import re
                    retry_match = re.search(r"'retryDelay':\s*'(\d+)s'", error_text)
                    if retry_match:
                        requested_delay = int(retry_match.group(1))
                        logger.warning(f"Rate limited! Gemini requested a delay of {requested_delay}s. Sleeping...")
                        wait_time = requested_delay + 2  # Add 2s buffer
                        
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Embedding failed after %d retries for model %s", MAX_RETRIES, model_name)

    raise EmbeddingError(f"Failed to embed texts: {last_error}")

def validate_embedding(embedding: list[float]) -> bool:
    """Validate embedding vector quality."""
    if not embedding:
        return False

    expected_dim = _active_embedding_dim if _active_embedding_dim > 0 else EMBEDDING_DIM
    if len(embedding) != expected_dim:
        return False

    # Check for NaN or Inf values
    import math
    return all(not math.isnan(x) and not math.isinf(x) for x in embedding)


def _sync_embedding_dim(embeddings: list[list[float]]) -> None:
    global _active_embedding_dim
    if not embeddings or not embeddings[0]:
        return
    new_dim = len(embeddings[0])
    if _active_embedding_dim != new_dim:
        logger.info("Embedding dimension updated from %d to %d", _active_embedding_dim, new_dim)
        _active_embedding_dim = new_dim
