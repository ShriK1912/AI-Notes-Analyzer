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

def _get_client():
    return genai.Client(api_key=settings.GEMINI_API_KEY)

async def _generate_with_retry(prompt: str, json_mode: bool = False) -> str:
    """Call Gemini with multi-model fallback and backoff on 503/429 errors."""
    client = _get_client()
    last_error = None
    config = genai_types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    for attempt in range(MAX_GEN_RETRIES):
        for model in GENERATION_MODELS:
            try:
                loop = asyncio.get_running_loop()
                resp = await loop.run_in_executor(
                    None, lambda m=model: client.models.generate_content(
                        model=m, contents=prompt, config=config
                    )
                )
                text = (resp.text or "").strip()
                if text:
                    return text
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    logger.warning(f"Model {model} hit transient error: {error_str[:80]}. Trying fallback...")
                    continue
                else:
                    raise
        wait = GEN_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
        retry_match = re.search(r"'retryDelay':\s*'(\d+)s'", str(last_error))
        if retry_match:
            wait = min(int(retry_match.group(1)) + 1, 60)
        logger.warning(f"Gemini API attempt {attempt+1}/{MAX_GEN_RETRIES} failed. Retrying in {wait:.1f}s...")
        await asyncio.sleep(wait)
    raise Exception(f"Gemini API failed after {MAX_GEN_RETRIES} retries: {last_error}")

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
