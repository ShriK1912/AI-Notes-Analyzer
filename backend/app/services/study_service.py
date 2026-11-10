import json
import asyncio
import random
import logging
from typing import List
from google import genai
from google.genai import types as genai_types
from firebase_admin import firestore
from app.core.database import get_db
import os

from app.core.config import settings

logger = logging.getLogger(__name__)
GENERATION_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash-lite"]
GENERATION_MODEL = GENERATION_MODELS[0]
MAX_GEN_RETRIES = 4
GEN_BASE_DELAY = 2.0

from app.services.gemini_pool import gemini_pool

async def _generate_with_retry(prompt: str, json_mode: bool = False) -> str:
    """Call Gemini with key pooling, multi-model fallback, and backoff on 503/429 errors."""
    return await gemini_pool.generate_content_with_failover(prompt, json_mode=json_mode)

async def _identify_key_concepts(subject_id: str, user_id: str) -> List[str]:
    db = get_db()
    chunks_ref = db.collection("users").document(user_id).collection("subjects").document(subject_id).collection("chunks")
    docs = chunks_ref.limit(20).get()
    chunks = [{"chunkId": doc.id, **doc.to_dict()} for doc in docs]
    if not chunks:
        return []
    
    chunks_summary = "\n".join([f"- {c.get('text', '')[:150]}..." for c in chunks])
    prompt = f"Analyze these snippets and identify the 5 most important distinct concepts. Output ONLY a JSON list of strings.\nNOTES:\n{chunks_summary}"
    try:
        text = await _generate_with_retry(prompt, json_mode=True)
        return json.loads(text)
    except Exception:
        return []

async def generate_quiz(subject_id: str, user_id: str) -> dict:
    db = get_db()
    chunks_ref = db.collection("users").document(user_id).collection("subjects").document(subject_id).collection("chunks")
    concepts = await _identify_key_concepts(subject_id, user_id)
    selected = []
    
    all_chunks = [{"chunkId": doc.id, **doc.to_dict()} for doc in chunks_ref.limit(50).get()]
    if not all_chunks:
        raise ValueError("Not enough notes uploaded for this subject to generate a quiz.")
    
    for c in concepts:
        for chunk in all_chunks:
            if chunk not in selected and c.lower() in chunk.get("text", "").lower():
                selected.append(chunk)
                break
                
    if len(selected) < 8:
        rem = [c for c in all_chunks if c not in selected]
        selected.extend(random.sample(rem, min(8 - len(selected), len(rem))))
        
    chunks_text = "\n".join([f"[CHUNK ID: {c['chunkId']}]\n{c.get('text', '')}" for c in selected])
    prompt = f"""Generate a study quiz based ONLY on the following text chunks. Follow exact JSON structure.
CHUNKS:
{chunks_text}
JSON SCHEMA:
{{ "mcqs": [ {{"question":"", "options":["A","B","C","D"], "correctOptionIndex":0, "explanation":"", "sourceChunkId":""}} ],
  "saqs": [ {{"question":"", "modelAnswer":"", "sourceChunkId":""}} ] }}
Must have exactly 5 MCQs and 3 SAQs."""

    try:
        text = await _generate_with_retry(prompt, json_mode=True)
        return json.loads(text)
    except Exception:
        return {"error": "Failed to generate quiz format"}
