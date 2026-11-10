import logging
import uuid
from datetime import datetime
from urllib.parse import urlparse

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, BackgroundTasks

from app.core.database import get_db
from firebase_admin import firestore, storage
from app.services.chunking_service import chunk_text
from app.services.embedding_service import embed_chunks
from app.services.extraction_service import extract_text_from_file, get_page_count
from app.services.background_task_service import run_ocr_pipeline
from app.services.scraping_service import ScrapingError, scrape_url
from app.utils.user_context import resolve_user_id

import asyncio

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "pptx", "png", "jpg", "jpeg"}

FIXED_SUBJECTS = [
    {"id": "maths", "name": "Maths"},
    {"id": "chemistry", "name": "Chemistry"},
    {"id": "physics", "name": "Physics"},
]

async def _ensure_subject(db, subject_id: str, user_id: str) -> None:
    for s in FIXED_SUBJECTS:
        subj_ref = db.collection("users").document(user_id).collection("subjects").document(s["id"])
        if not subj_ref.get().exists:
            subj_ref.set({"id": s["id"], "name": s["name"], "userId": user_id, "createdAt": firestore.SERVER_TIMESTAMP})

def _url_to_filename(url: str, title: str | None) -> str:
    parsed = urlparse(url)
    raw_name = (title or parsed.path.strip("/").split("/")[-1] or parsed.netloc or "web_source").strip()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw_name)
    return f"{safe[:70]}.url.txt"

async def _index_document(*, db, user_id: str, subject_id: str, filename: str, source_format: str, extraction: dict, source_url: str | None = None) -> dict:
    doc_id = str(uuid.uuid4())
    
    # Save to Firestore
    doc_ref = db.collection("users").document(user_id).collection("subjects").document(subject_id).collection("documents").document(doc_id)
    doc_data = {
        "id": doc_id,
        "userId": user_id,
        "subjectId": subject_id,
        "fileName": filename,
        "sourceFormat": source_format,
        "uploadDate": firestore.SERVER_TIMESTAMP,
        "status": "Indexed",
        "ocrConfidence": extraction.get("confidence", 1.0)
    }
    if source_url:
        doc_data["sourceUrl"] = source_url
    doc_ref.set(doc_data)

    metadata = {
        "subjectId": subject_id,
        "documentId": doc_id,
        "fileName": filename,
        "sourceFormat": source_format,
    }
    chunks = chunk_text(extraction["text"], metadata)
    for chunk in chunks: chunk["userId"] = user_id

    embedded_chunks = await embed_chunks(chunks)

    from google.cloud.firestore_v1.vector import Vector
    chunks_coll = db.collection("users").document(user_id).collection("subjects").document(subject_id).collection("chunks")
    batch = db.batch()
    for c in embedded_chunks:
        c_ref = chunks_coll.document(c["chunkId"])
        c_copy = dict(c)
        if "embedding" in c_copy:
            c_copy["embedding"] = Vector(c_copy["embedding"])
        batch.set(c_ref, c_copy)
    batch.commit()

    return {"message": "Upload successful", "documentId": doc_id, "chunkCount": len(embedded_chunks)}

@router.post("/")
async def upload_document(
    request: Request, 
    background_tasks: BackgroundTasks,
    subjectId: str = Form(...), 
    file: UploadFile = File(...)
):
    db = get_db()
    user_id = resolve_user_id(request)
    await _ensure_subject(db, subjectId, user_id)

    filename = file.filename or "unknown"
    source_format = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if source_format not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type")

    # Read bytes and metadata in the request thread
    file_bytes = await file.read()
    total_pages = get_page_count(file_bytes)
    job_id = str(uuid.uuid4())

    # Initialize job document in Firestore
    job_data = {
        "job_id": job_id,
        "user_id": user_id,
        "status": "pending",
        "file_name": filename,
        "pages_total": total_pages,
        "pages_done": 0,
        "pages_failed": [],
        "result": [],
        "created_at": firestore.SERVER_TIMESTAMP,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "error": None
    }
    db.collection("jobs").document(job_id).set(job_data)

    # Trigger background processing
    background_tasks.add_task(
        run_ocr_pipeline, 
        job_id, 
        file_bytes, 
        user_id, 
        subjectId, 
        filename, 
        source_format
    )

    return {"job_id": job_id}

@router.get("/status/{job_id}")
async def get_job_status(job_id: str, request: Request):
    db = get_db()
    user_id = resolve_user_id(request)
    # Note: Status check via GET is less used now with onSnapshot but kept for compatibility
    # However we'd need subjectId here to find it. For now, since it's just a GET, 
    # and we want real-time, onSnapshot is the primary path.
    raise HTTPException(status_code=501, detail="Synchronous status polling deprecated. Use Firestore onSnapshot.")
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Job not found")
    
    data = doc.to_dict()
    if data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
        
    return data

@router.post("/url")
async def upload_url(request: Request, subjectId: str = Form(...), url: str = Form(...)):
    db = get_db()
    user_id = resolve_user_id(request)
    await _ensure_subject(db, subjectId, user_id)

    extraction = await scrape_url(url)
    filename = _url_to_filename(url, extraction.get("title"))
    result = await _index_document(db=db, user_id=user_id, subject_id=subjectId, filename=filename, source_format="url", extraction=extraction, source_url=extraction.get("sourceUrl"))
    result["sourceUrl"] = extraction.get("sourceUrl")
    result["title"] = extraction.get("title", "")
    return result

@router.get("/documents/{subject_id}")
async def list_documents(subject_id: str, request: Request):
    db = get_db()
    user_id = resolve_user_id(request)
    docs_ref = db.collection("users").document(user_id).collection("subjects").document(subject_id).collection("documents")
    docs = [doc.to_dict() for doc in docs_ref.get()]
    return {"documents": docs}

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request, subjectId: str):
    db = get_db()
    user_id = resolve_user_id(request)
    
    from google.cloud.firestore_v1.base_query import FieldFilter
    chunks_ref = db.collection("users").document(user_id).collection("subjects").document(subjectId).collection("chunks")
    chunks = chunks_ref.where(filter=FieldFilter("documentId", "==", doc_id)).get()
    batch = db.batch()
    for c in chunks: batch.delete(c.reference)
    batch.delete(db.collection("users").document(user_id).collection("subjects").document(subjectId).collection("documents").document(doc_id))
    batch.commit()

    return {"message": "Document deleted", "documentId": doc_id}
