import logging
from datetime import datetime, timezone, timedelta
from app.core.database import get_db
from firebase_admin import firestore

logger = logging.getLogger(__name__)

# Jobs whose heartbeat has not updated in 10 minutes are considered abandoned
STALE_THRESHOLD_MINUTES = 10

async def recover_stale_jobs():
    """
    Startup reaper task: scans Firestore for abandoned/orphaned jobs that were
    interrupted by a server crash, restart, or Render container idle spin-down.
    Marks them with an actionable error state so client listeners don't hang indefinitely.
    """
    try:
        db = get_db()
        jobs_ref = db.collection("jobs")
        
        # Query active jobs
        active_jobs = jobs_ref.where("status", "in", ["pending", "processing"]).stream()
        
        now = datetime.now(timezone.utc)
        stale_count = 0
        
        for job_doc in active_jobs:
            job_data = job_doc.to_dict()
            updated_at = job_data.get("updated_at") or job_data.get("created_at")
            
            # Check elapsed time
            is_stale = False
            if updated_at:
                if isinstance(updated_at, datetime):
                    elapsed = now - (updated_at if updated_at.tzinfo else updated_at.replace(tzinfo=timezone.utc))
                    if elapsed > timedelta(minutes=STALE_THRESHOLD_MINUTES):
                        is_stale = True
                else:
                    # If timestamp format is unknown, mark stale if old
                    is_stale = True
            
            if is_stale:
                stale_count += 1
                job_id = job_doc.id
                attempts = job_data.get("attempts", 1)
                max_attempts = job_data.get("max_attempts", 3)
                
                logger.warning(
                    f"Recovering stale job {job_id}: status was '{job_data.get('status')}', "
                    f"attempt {attempts}/{max_attempts}. Marking failed due to container restart."
                )
                
                job_doc.reference.update({
                    "status": "failed",
                    "error": "Processing interrupted by cloud server restart/timeout. Please re-upload.",
                    "recovery_reason": "stale_lease_detected",
                    "recovered_at": firestore.SERVER_TIMESTAMP,
                    "updated_at": firestore.SERVER_TIMESTAMP
                })
                
        if stale_count > 0:
            logger.info(f"Stale job recovery complete: cleaned up {stale_count} orphaned jobs.")
        else:
            logger.info("Stale job recovery: 0 orphaned jobs found. System healthy.")
            
    except Exception as e:
        logger.error(f"Error during stale job recovery sweep: {e}")

