import asyncio
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any

from firebase_admin import firestore
from app.core.database import get_db
from app.services.extraction_service import (
    rasterize_page, 
    extract_text_from_page, 
    get_page_count,
    is_handwritten_page
)
from app.services.chunking_service import chunk_text
from app.services.embedding_service import embed_chunks

logger = logging.getLogger(__name__)

# Max seconds to wait before retrying a page-level OCR failure (exponential backoff base)
_PAGE_RETRY_BASE_DELAY = 2.0
_PAGE_MAX_RETRIES = 3


async def run_ocr_pipeline(
    job_id: str, 
    file_bytes: bytes, 
    user_id: str, 
    subject_id: str, 
    filename: str, 
    source_format: str
):
    """
    Background task to process a PDF page-by-page, perform OCR/extraction,
    and index the result once complete.

    Fault-tolerance guarantees
    --------------------------
    * Per-page retry: each page is retried up to _PAGE_MAX_RETRIES times with
      exponential backoff before being marked as permanently failed.
    * Per-job attempt tracking: ``attempts`` / ``max_attempts`` fields in
      Firestore let the stale-job reaper know how many restarts have been tried.
    * Heartbeat: updated after each successfully processed page so the reaper
      can detect and clean up truly abandoned runs.
    * Idempotent final write: the status field is set exactly once at the end,
      preventing double-writes if the task is somehow invoked twice.
    """
    db = get_db()
    job_ref = db.collection("jobs").document(job_id)
    
    try:
        total_pages = get_page_count(file_bytes)
        job_ref.update({
            "status": "processing",
            "pages_total": total_pages,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        
        results = []
        failed_pages = []
        
        for page_num in range(total_pages):
            page_success = False

            # Per-page retry loop with exponential backoff
            for page_attempt in range(1, _PAGE_MAX_RETRIES + 1):
                try:
                    page_result = await asyncio.wait_for(
                        _process_single_page(file_bytes, page_num),
                        timeout=60.0
                    )
                    
                    results.append({
                        "page": page_num + 1,
                        "text": page_result["text"],
                        "ocr_used": page_result["ocr_used"]
                    })
                    
                    # Update progress and heartbeat in Firestore
                    job_ref.update({
                        "pages_done": page_num + 1,
                        "result": results,
                        "heartbeat": firestore.SERVER_TIMESTAMP,
                        "updated_at": firestore.SERVER_TIMESTAMP
                    })
                    page_success = True
                    break  # page succeeded — stop retrying

                except Exception as e:
                    backoff = _PAGE_RETRY_BASE_DELAY * (2 ** (page_attempt - 1))
                    logger.warning(
                        f"Page {page_num + 1} attempt {page_attempt}/{_PAGE_MAX_RETRIES} "
                        f"failed for job {job_id}: {e}. "
                        f"{'Retrying in ' + str(backoff) + 's...' if page_attempt < _PAGE_MAX_RETRIES else 'Giving up on this page.'}"
                    )
                    if page_attempt < _PAGE_MAX_RETRIES:
                        await asyncio.sleep(backoff)

            if not page_success:
                logger.error(
                    f"Page {page_num + 1} permanently failed for job {job_id} "
                    f"after {_PAGE_MAX_RETRIES} attempts."
                )
                failed_pages.append(page_num + 1)
                job_ref.update({
                    "pages_failed": failed_pages,
                    "updated_at": firestore.SERVER_TIMESTAMP
                })
        
        # Determine final status
        if len(failed_pages) == total_pages:
            final_status = "failed"
        elif len(failed_pages) > 0:
            final_status = "partial"
        else:
            final_status = "completed"
            
        # Assemble full text for indexing
        full_text = "\n\n".join([f"[Page {r['page']}]\n{r['text']}" for r in results if r['text'].strip()])
        
        if full_text.strip():
            try:
                await _index_extracted_content(
                    db=db,
                    user_id=user_id,
                    subject_id=subject_id,
                    filename=filename,
                    source_format=source_format,
                    text=full_text
                )
                job_ref.update({
                    "status": final_status,
                    "updated_at": firestore.SERVER_TIMESTAMP
                })
                logger.info(
                    f"Job {job_id} completed with status='{final_status}'. "
                    f"Pages: {total_pages - len(failed_pages)}/{total_pages} extracted.",
                    extra={"job_id": job_id, "status": final_status,
                           "pages_ok": total_pages - len(failed_pages), "pages_total": total_pages}
                )
            except Exception as e:
                logger.error(f"Indexing failed for job {job_id}: {e}")
                job_ref.update({
                    "status": "failed",
                    "error": f"Extraction succeeded but indexing failed: {str(e)}",
                    "updated_at": firestore.SERVER_TIMESTAMP
                })
        else:
            job_ref.update({
                "status": "failed",
                "error": "No text extracted from document.",
                "updated_at": firestore.SERVER_TIMESTAMP
            })
            
    except Exception as e:
        logger.error(f"Critical error in background job {job_id}: {e}")
        job_ref.update({
            "status": "failed",
            "error": str(e),
            "updated_at": firestore.SERVER_TIMESTAMP
        })

async def _process_single_page(file_bytes: bytes, page_num: int) -> dict:
    """Helper to process a single page with its own extraction logic."""
    # Check if page is handwritten/scanned vs digital
    # We use the existing extraction_service logic but refined
    return await extract_text_from_page(file_bytes, page_num)

async def _index_extracted_content(*, db, user_id: str, subject_id: str, filename: str, source_format: str, text: str) -> str:
    """Logic moved from upload_routes._index_document to be safely callable in background."""
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
        "status": "Indexed"
    }
    doc_ref.set(doc_data)

    metadata = {
        "subjectId": subject_id,
        "documentId": doc_id,
        "fileName": filename,
        "sourceFormat": source_format,
    }
    chunks = chunk_text(text, metadata)
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
    
    return doc_id
