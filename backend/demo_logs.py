"""
Live Demonstration Script for Cloud Structured JSON Logging
============================================================
Run this script to demonstrate how AskMyNotes formats all server,
task, security, and error telemetry as single-line JSON events for
cloud ingestion (Render, Datadog, AWS CloudWatch, GCP Cloud Logging).

Usage:
    cd backend
    python demo_logs.py
"""

import sys
import os
import json
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.structured_logger import setup_structured_logging

def main():
    print("=" * 70, flush=True)
    print("  AskMyNotes: Cloud Structured JSON Logging Demonstration", flush=True)
    print("=" * 70, flush=True)
    print("\n[INFO] Initializing structured JSON logging via JSONFormatter...\n", flush=True)

    # Activate production JSON logger
    setup_structured_logging(level=logging.INFO)

    logger = logging.getLogger("app.services.demo")

    # 1. Standard Info Log with Cloud Metadata
    logger.info(
        "Container boot completed successfully",
        extra={
            "cloud_provider": "Render",
            "region": "oregon-us-west",
            "instance_type": "free-tier-512mb",
            "boot_duration_ms": 321.4
        }
    )
    time.sleep(0.3)

    # 2. Asynchronous Background Task Log (with job_id and page metrics)
    logger.info(
        "OCR background task progress updated",
        extra={
            "job_id": "c7a8b9e1-4567-89ab-cdef-0123456789ab",
            "user_id": "firebase_usr_771829",
            "pages_done": 5,
            "pages_total": 12,
            "ocr_engine": "gemini-vision-fallback",
            "heartbeat": "2026-11-21T08:30:15Z"
        }
    )
    time.sleep(0.3)

    # 3. Distributed Resilience Warning (Key Pool Rate-Limit Cooldown)
    logger.warning(
        "Gemini API rate limit 429 encountered, failing over to alternate key",
        extra={
            "model": "gemini-2.5-flash",
            "failed_key_index": 0,
            "cooldown_seconds": 60.0,
            "failover_target_model": "gemini-2.0-flash"
        }
    )
    time.sleep(0.3)

    # 4. Self-Healing Reaper Event
    logger.warning(
        "Startup reaper detected abandoned job lease from previous container restart",
        extra={
            "job_id": "f1d2c3b4-9876-5432-10fe-dcba98765432",
            "last_heartbeat_age_minutes": 14.2,
            "action": "marked_failed_stale_lease",
            "recovered": True
        }
    )
    time.sleep(0.3)

    # 5. Error with Serialized Cloud Exception Stack Trace
    try:
        raise ConnectionResetError("Remote cloud connection dropped by peer load balancer")
    except Exception:
        logger.error(
            "Cloud network connection reset during Firestore vector batch commit",
            exc_info=True,
            extra={
                "retry_attempt": 2,
                "max_retries": 3,
                "database": "neww",
                "backoff_delay_sec": 4.0
            }
        )

    print("\n" + "=" * 70)
    print("  Notice how every event:")
    print("  1. Is a single, valid JSON string on stdout.")
    print("  2. Includes UTC timestamp, log level, and logger module name.")
    print("  3. Contains domain-specific metadata (job_id, user_id, latencies).")
    print("  4. Allows cloud aggregators (Render, Datadog) to filter & alert instantly.")
    print("=" * 70)

if __name__ == "__main__":
    main()
