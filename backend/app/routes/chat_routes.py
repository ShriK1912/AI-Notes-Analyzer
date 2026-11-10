import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.services.rag_service import ask_question, RAGError
from app.utils.user_context import resolve_user_id

import firebase_admin
from firebase_admin import firestore
from app.core.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatTurn(BaseModel):
    role: str
    content: str
            
class ChatRequest(BaseModel):
    subjectId: str
    question: str
    history: list[ChatTurn] | None = None

@router.post("/ask")
async def chat_with_docs(req: ChatRequest, request: Request):
    user_id = resolve_user_id(request)
    
    # Verify Subject
    db = get_db()
    subj_ref = db.collection("users").document(user_id).collection("subjects").document(req.subjectId).get()
    if not subj_ref.exists:
        raise HTTPException(status_code=404, detail="Subject not found")
        
    subject_name = subj_ref.to_dict().get("name", "Unknown Subject")

    try:
        history_dicts = [t.dict() for t in (req.history or [])]
        response = await ask_question(
            query=req.question,
            subject_id=req.subjectId,
            subject_name=subject_name,
            user_id=user_id,
            history=history_dicts
        )
        return response
    except RAGError as e:
        logger.error(f"RAG Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected QA error")
        raise HTTPException(status_code=500, detail="Internal server error")
