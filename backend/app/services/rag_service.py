import logging
import asyncio
import re
import random
from google import genai
import os
import numpy as np
from app.core.config import settings

from firebase_admin import firestore

from app.services.embedding_service import embed_query, embed_queries

logger = logging.getLogger(__name__)

NOT_FOUND_TEMPLATE = "Not found in your notes for {subject_name}"
CONFIDENCE_THRESHOLDS = {"NOT_FOUND": 0.35, "LOW": 0.55, "MEDIUM": 0.75, "HIGH": 0.85}
GENERATION_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash-lite"]
GENERATION_MODEL = GENERATION_MODELS[0]

from app.services.gemini_pool import gemini_pool

class RAGError(Exception):
    pass

async def _generate_with_retry(prompt: str) -> str:
    """Call Gemini generation using resilient key-pool and multi-model failover."""
    try:
        return await gemini_pool.generate_content_with_failover(prompt)
    except Exception as e:
        raise RAGError(str(e))

async def _generate_multi_queries(query: str, subject_name: str) -> list[str]:
    # To conserve the 5 requests-per-minute Free Tier quota, avoid burning an extra LLM call
    # on query rewriting. Semantic retrieval using gemini-embedding-001 accurately matches natural queries.
    return [query]

def _preprocess_query(query: str):
    if not query or not isinstance(query, str) or not query.strip():
        raise RAGError("Query must be a non-empty string")
    query = re.sub(r'\s+', ' ', query).strip()
    query = re.sub(r'^[?!]+\s*', '', query)
    stop_words = {'what', 'is', 'the', 'a', 'an', 'are', 'and', 'or'}
    keywords = [w.strip('.,?!:;()[]{}') for w in query.lower().split() if w.strip('.,?!:;()[]{}') not in stop_words and len(w)>2]
    return query, keywords

def compute_confidence(similarities, query_keywords=None, chunk_texts=None):
    if not similarities:
        return {
            "tier": "NOT_FOUND",
            "score": 0.0,
            "maxScore": 0.0,
            "avgScore": 0.0,
            "minScore": 0.0,
            "variance": 0.0,
            "keywordBonus": 0.0,
        }
    max_s = max(similarities)
    min_s = min(similarities)
    avg_s = sum(similarities) / len(similarities)
    var = sum((s - avg_s) ** 2 for s in similarities) / len(similarities)
    bonus = 0.0
    if query_keywords and chunk_texts:
        matches = sum(1 for text in chunk_texts for kw in query_keywords if kw.lower() in text.lower())
        bonus = min(0.05, (matches / len(chunk_texts)) * 0.10)
    w = max(0.0, min(1.0, 0.75 * max_s + 0.25 * avg_s + bonus - (0.15 * var)))
    tier = "NOT_FOUND"
    if w >= CONFIDENCE_THRESHOLDS["HIGH"]:
        tier = "HIGH"
    elif w >= CONFIDENCE_THRESHOLDS["MEDIUM"]:
        tier = "MEDIUM"
    elif w >= CONFIDENCE_THRESHOLDS["LOW"]:
        tier = "LOW"
    return {
        "tier": tier,
        "score": round(w, 4),
        "maxScore": round(max_s, 4),
        "avgScore": round(avg_s, 4),
        "minScore": round(min_s, 4),
        "variance": round(var, 4),
        "keywordBonus": round(bonus, 4),
    }

def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Extract and validate [SOURCE: filename, location] citations from LLM answer."""
    citations = []
    seen_cites = set()
    for fn, loc in re.findall(r"\[SOURCE:\s*(.*?),\s*(.*?)\]", answer):
        fn, loc = fn.strip(), loc.strip()
        if (fn, loc) in seen_cites:
            continue
        for c in chunks:
            c_file = c.get("fileName", "")
            if (c_file == fn or c_file.endswith(fn) or fn.endswith(c_file)) and c.get("locationRef") == loc:
                citations.append({
                    "fileName": c["fileName"],
                    "locationRef": loc,
                    "chunkId": c.get("chunkId", ""),
                    "sourceFormat": c.get("sourceFormat"),
                })
                seen_cites.add((fn, loc))
                break
    return citations

def build_sources_block(chunks: list[dict]) -> str:
    """Format document chunks into prompt sources block."""
    blocks = []
    for i, c in enumerate(chunks):
        blocks.append(
            f"[SOURCE {i+1}]\nFile: {c.get('fileName')}\nLocation: {c.get('locationRef')}\nFormat: {c.get('sourceFormat', 'unknown')}\nContent:\n{c.get('text')}"
        )
    return "\n---\n".join(blocks)

def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Deduplicate chunks by text content or chunkId preserving order."""
    seen = set()
    deduped = []
    for c in chunks:
        key = c.get("text") or c.get("chunkId")
        if key not in seen:
            deduped.append(c)
            seen.add(key)
    return deduped

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def ask_question(query: str, subject_id: str, subject_name: str, user_id: str, history=None):
    cleaned_query, keywords = _preprocess_query(query)
    multi_queries = await _generate_multi_queries(cleaned_query, subject_name)
    
    from app.vectorstore.firestore_client import find_similar_chunks
    
    try:
        scored_chunks = []
        # Batch embed all queries to save API calls
        from app.services.embedding_service import embed_queries
        query_embeddings = await embed_queries(multi_queries)
        
        for q, q_emb in zip(multi_queries, query_embeddings):
            results = find_similar_chunks(userId=user_id, subjectId=subject_id, query_embedding=q_emb, limit=5)
            
            logger.info(f"Results for query: {q} -> count: {len(results)}")
            for c in results:
                # find_nearest returns a distance (0 is exact match, higher is less similar). Sim = 1.0 - distance
                dist = c.get("vector_distance")
                sim = max(0.0, 1.0 - dist) if dist is not None else 0.0
                scored_chunks.append({**c, "similarity": float(sim), "chunkId": c.get("id", c.get("chunkId"))})
                
        if not scored_chunks:
            return {"answer": NOT_FOUND_TEMPLATE.format(subject_name=subject_name), "confidenceTier": "NOT_FOUND", "confidenceScore": 0.0, "citations": [], "evidenceSnippets": [], "topChunkIds": []}

        # Deduplicate
        deduped = _deduplicate_chunks(scored_chunks)

        # Sort by similarity and take top 5
        deduped.sort(key=lambda x: x["similarity"], reverse=True)
        chunk_dicts = deduped[:5]
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise RAGError(str(e))

    logger.info(f"RAG: Deduplicated chunks -> {len(chunk_dicts)}")

    conf = compute_confidence([c["similarity"] for c in chunk_dicts], keywords, [c.get("text","") for c in chunk_dicts])
    if conf["tier"] in ("NOT_FOUND", "LOW"):
        # We can still try to have the LLM answer, but with low confidence
        pass

    # Build Prompt
    sources = build_sources_block(chunk_dicts)
    hist = "\n".join([f"{'User' if t.get('role')=='user' else 'Teacher'}: {t.get('content')}" for t in (history or [])[-5:]])
    prompt = f"RULES:\n1. ONLY use sources below. If not found, output EXACTLY: {NOT_FOUND_TEMPLATE.format(subject_name=subject_name)}\n2. Cite sources using [SOURCE: filename, location].\n3. Confidence is {conf['tier']}.\nHISTORY:\n{hist}\nSOURCES:\n{sources}\nQUERY: {query}"

    try:
        ans = await _generate_with_retry(prompt)
    except Exception as e:
        raise RAGError(f"Gen failed: {e}")

    # Extract citations
    citations = extract_citations(ans, chunk_dicts)

    return {
        "answer": ans, "confidenceTier": conf["tier"], "confidenceScore": conf["score"],
        "citations": citations, "evidenceSnippets": [],
        "topChunkIds": [c["chunkId"] for c in chunk_dicts],
    }
