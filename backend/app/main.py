import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.routes import upload_routes, chat_routes, study_routes
from app.utils.structured_logger import setup_structured_logging
from app.services.job_recovery_service import recover_stale_jobs

import firebase_admin
from firebase_admin import credentials

# Initialize production-grade structured JSON logging
setup_structured_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Initialize Firebase Admin SDK
    # Supports three credential sources (in priority order):
    #   1. FIREBASE_SERVICE_ACCOUNT_JSON env var — JSON string (ideal for Render/Railway)
    #   2. File path via GOOGLE_APPLICATION_CREDENTIALS / FIREBASE_SERVICE_ACCOUNT_PATH
    #   3. Application Default Credentials (GCP, gcloud auth)
    try:
        if not firebase_admin._apps:
            import os, json

            sa_json_str = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
            service_account_path = settings.GOOGLE_APPLICATION_CREDENTIALS or settings.FIREBASE_SERVICE_ACCOUNT_PATH

            FIREBASE_OPTIONS = {
                'databaseURL': 'https://ai-notes-analyser-default-rtdb.firebaseio.com',
                'projectId': 'ai-notes-analyser',
            }

            if sa_json_str:
                # Option 1: JSON string env var — used on Render, Railway, etc.
                try:
                    sa_dict = json.loads(sa_json_str)
                    cred = credentials.Certificate(sa_dict)
                    firebase_admin.initialize_app(cred, FIREBASE_OPTIONS)
                    logger.info("Firebase Admin initialized from FIREBASE_SERVICE_ACCOUNT_JSON env var")
                except Exception as e:
                    logger.error(f"Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: {e}")
                    raise

            elif service_account_path and os.path.exists(service_account_path):
                # Option 2: JSON file on disk — used locally
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(service_account_path)
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred, FIREBASE_OPTIONS)
                logger.info(f"Firebase Admin initialized with service account file: {service_account_path}")

            else:
                # Option 3: Application Default Credentials (GCP environments)
                try:
                    firebase_admin.initialize_app()
                    logger.info("Firebase Admin initialized via Application Default Credentials")
                except Exception as inner_e:
                    logger.warning(f"No Firebase credentials found ({inner_e}). Booting in demo mode — Firestore writes will fail.")
                    firebase_admin.initialize_app(options={'projectId': 'demo-project'})

        # Cloud maturity: launch stale-job recovery sweep on container startup/spin-up
        asyncio.create_task(recover_stale_jobs())

    except Exception as e:
        logger.warning(f"Firebase Admin initialization failed: {e}")
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AskMyNotes API - Firebase Spark Plan Version",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local development
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        # Firebase Hosting (production)
        "https://ai-notes-analyser.web.app",
        "https://ai-notes-analyser.firebaseapp.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_routes.router, prefix=f"{settings.API_V1_STR}/upload", tags=["Upload"])
app.include_router(chat_routes.router, prefix=f"{settings.API_V1_STR}/qa", tags=["Q&A"])
app.include_router(study_routes.router, prefix=f"{settings.API_V1_STR}/study", tags=["Study Mode"])

@app.get("/")
async def root() -> dict:
    return {"service": settings.PROJECT_NAME, "status": "running"}

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
