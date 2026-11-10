from firebase_admin import firestore
from app.core.database import get_db
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

def get_firestore_client():
    """Retrieve the initialized Firestore client (named 'neww' database)."""
    return get_db()

def store_chunk(userId: str, subjectId: str, documentId: str, chunkId: str, chunk_data: dict, embedding: list[float]):
    """
    Store a document chunk with its vector embedding in Firestore.
    Path: /users/{userId}/subjects/{subjectId}/documents/{documentId}/chunks/{chunk_id}
    """
    db = get_firestore_client()
    
    # Store embedding as a Firestore Vector
    chunk_data["embedding"] = Vector(embedding)
    
    doc_ref = db.collection("users").document(userId) \
                .collection("subjects").document(subjectId) \
                .collection("documents").document(documentId) \
                .collection("chunks").document(chunkId)
                
    doc_ref.set(chunk_data)
    return chunk_id

import numpy as np

def find_similar_chunks(userId: str, subjectId: str, query_embedding: list[float], limit: int = 5):
    """
    Perform a vector search (KNN) on chunks belonging to a specific subject.
    Since creating a Firestore vector index requires permissions we don't have,
    this fetches all chunks for the subject and computes cosine similarity in-memory.
    """
    db = get_firestore_client()
    
    chunks_ref = db.collection("users").document(userId).collection("subjects").document(subjectId).collection("chunks")
    chunks = chunks_ref.stream()
    
    all_chunks = []
    for chunk in chunks:
        data = chunk.to_dict()
        if "embedding" in data:
            all_chunks.append((chunk.id, data))
                
    if not all_chunks:
        return []
        
    q_vec = np.array(query_embedding)
    q_norm = np.linalg.norm(q_vec)
    
    if q_norm == 0:
        return []
        
    results = []
    for chunk_id, data in all_chunks:
        emb = data.get("embedding")
        # Handle Firestore Vector object or plain list
        emb_list = list(emb) if hasattr(emb, "__iter__") else emb
        
        c_vec = np.array(emb_list)
        c_norm = np.linalg.norm(c_vec)
        
        if c_norm > 0:
            sim = np.dot(q_vec, c_vec) / (q_norm * c_norm)
        else:
            sim = 0.0
            
        del data["embedding"]
        data["id"] = chunk_id
        # distance is 1 - similarity (cosine distance)
        data["vector_distance"] = 1.0 - sim 
        results.append((sim, data))
        
    # Sort by descending similarity
    results.sort(key=lambda x: x[0], reverse=True)
    
    return [r[1] for r in results[:limit]]
