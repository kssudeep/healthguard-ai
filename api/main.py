"""
api/main.py

FastAPI backend for HealthGuard AI.
- Async file upload + pipeline trigger
- Background task execution (Celery-ready)
- JWT-based auth middleware
- Rate limiting
- Prometheus metrics endpoint
- Swagger docs at /docs
"""

from __future__ import annotations
import uuid
import time
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas.request_schemas import AnalysisRequest, AnalysisResponse, ReportStatus
from api.middleware.auth import verify_token
from api.middleware.rate_limiter import RateLimiter
from agents.orchestrator.graph import run_pipeline
from config.settings import settings

logger = logging.getLogger(__name__)

# ── App init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HealthGuard AI API",
    description=(
        "Multimodal Clinical Intelligence Platform — "
        "Chest X-ray analysis + NLP symptom extraction + RAG-based clinical decision support"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# In-memory job store (replace with Redis in production)
_job_store: dict[str, dict] = {}
security = HTTPBearer(auto_error=False)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models": {
            "vision": "densenet121-chestxray14-v2",
            "nlp": "biobert-ner",
            "rag": "faiss-hybrid",
            "llm": "gemini-2.0-flash + groq-llama-3.3-70b",
        },
    }


@app.get("/", tags=["System"])
async def root():
    return {"message": "HealthGuard AI API", "docs": "/docs"}


# ── Main analysis endpoint ────────────────────────────────────────────────────

@app.post(
    "/api/v1/analyse",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Submit chest X-ray + symptoms for clinical analysis",
    description=(
        "Accepts a chest X-ray image (JPEG/PNG/DICOM) and free-text symptom description. "
        "Triggers the 5-agent LangGraph pipeline asynchronously. "
        "Returns a job_id to poll for results."
    ),
)
async def analyse(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Chest X-ray image (JPEG, PNG, or DICOM)"),
    symptoms: str = Form(..., description="Patient symptom description"),
    patient_age: Optional[int] = Form(None, description="Patient age in years"),
    patient_sex: Optional[str] = Form(None, description="Patient sex: M, F, or unknown"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    # Validate file type
    allowed_types = {".jpg", ".jpeg", ".png", ".dcm"}
    suffix = Path(image.filename).suffix.lower()
    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_types}",
        )

    # Save uploaded file
    job_id = str(uuid.uuid4())[:8]
    upload_dir = Path(settings.UPLOAD_DIR) / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = str(upload_dir / image.filename)

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    logger.info(f"[API] Job {job_id} created for {image.filename}")

    # Store job status
    _job_store[job_id] = {
        "status": "queued",
        "created_at": time.time(),
        "image_path": image_path,
        "symptoms": symptoms,
    }

    # Run pipeline in background
    background_tasks.add_task(
        _run_pipeline_task,
        job_id=job_id,
        image_path=image_path,
        symptoms=symptoms,
        patient_age=patient_age,
        patient_sex=patient_sex or "unknown",
    )

    return AnalysisResponse(
        job_id=job_id,
        status="queued",
        message="Analysis pipeline started. Poll /api/v1/results/{job_id} for results.",
        estimated_time_seconds=25,
    )


async def _run_pipeline_task(
    job_id: str,
    image_path: str,
    symptoms: str,
    patient_age: Optional[int],
    patient_sex: str,
):
    """Background task: runs the full LangGraph pipeline."""
    _job_store[job_id]["status"] = "processing"
    try:
        result = run_pipeline(
            image_path=image_path,
            symptom_text=symptoms,
            patient_age=patient_age,
            patient_sex=patient_sex,
        )
        _job_store[job_id].update({
            "status": "complete",
            "result": result,
            "completed_at": time.time(),
        })
        logger.info(f"[API] Job {job_id} complete")
    except Exception as e:
        logger.error(f"[API] Job {job_id} failed: {e}", exc_info=True)
        _job_store[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": time.time(),
        })


# ── Results polling ───────────────────────────────────────────────────────────

@app.get(
    "/api/v1/results/{job_id}",
    tags=["Analysis"],
    summary="Poll for analysis results",
)
async def get_results(job_id: str):
    if job_id not in _job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _job_store[job_id]

    if job["status"] == "queued":
        return {"job_id": job_id, "status": "queued", "message": "Job is queued"}
    elif job["status"] == "processing":
        elapsed = time.time() - job["created_at"]
        return {"job_id": job_id, "status": "processing", "elapsed_seconds": elapsed}
    elif job["status"] == "failed":
        return {"job_id": job_id, "status": "failed", "error": job.get("error")}
    else:
        return {"job_id": job_id, "status": "complete", "result": job["result"]}


# ── GradCAM image endpoint ────────────────────────────────────────────────────

@app.get(
    "/api/v1/gradcam/{session_id}",
    tags=["Analysis"],
    summary="Retrieve Grad-CAM heatmap for a session",
)
async def get_gradcam(session_id: str):
    path = Path(settings.GRADCAM_OUTPUT_DIR) / f"{session_id}_gradcam.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="GradCAM not found for this session")
    return FileResponse(str(path), media_type="image/png")


# ── MLflow run metadata ────────────────────────────────────────────────────────

@app.get(
    "/api/v1/runs/{mlflow_run_id}",
    tags=["Monitoring"],
    summary="Get MLflow run metrics for a pipeline execution",
)
async def get_mlflow_run(mlflow_run_id: str):
    from monitoring.mlflow_tracking.tracker import MLflowTracker
    tracker = MLflowTracker()
    try:
        run_data = tracker.get_run(mlflow_run_id)
        return run_data
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── System stats ──────────────────────────────────────────────────────────────

@app.get("/api/v1/stats", tags=["System"])
async def get_stats():
    total = len(_job_store)
    completed = sum(1 for j in _job_store.values() if j["status"] == "complete")
    failed = sum(1 for j in _job_store.values() if j["status"] == "failed")
    return {
        "total_jobs": total,
        "completed": completed,
        "failed": failed,
        "success_rate": f"{(completed / total * 100):.1f}%" if total > 0 else "N/A",
    }
