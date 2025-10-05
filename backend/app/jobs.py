import json
import hashlib
from uuid import UUID as UUID_t
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from redis.asyncio import from_url
from sqlalchemy import func

from .settings import settings
from .db import SessionLocal
from .models import Job, User, Case
from .auth import get_current_user
from .logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_redis():
    r = from_url(settings.redis_url, decode_responses=True)
    try:
        await r.ping()
    except Exception as e:
        logger.error("Redis ping failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Redis unavailable: {e}")
    logger.debug("Connected to Redis at %s", settings.redis_url)
    return r

def get_current_user_obj(username: str = Depends(get_current_user), db: Session = Depends(get_db)) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

def _validate_csv_content_type(ct: Optional[str]) -> None:
    # different broswers can have different content types
    allowed = {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    }
    if ct is None or ct.lower() not in allowed:
        logger.debug("Unrecognized CSV content-type: %s", ct)

def _parse_case_id(case_id: Optional[str]) -> Optional[UUID_t]:
    if not case_id:
        return None
    try:
        return UUID_t(case_id)
    except Exception:
        raise HTTPException(status_code=400, detail="case_id must be a UUID")

@router.post("/upload")
async def upload_and_enqueue(
    file: UploadFile = File(...),
    case_id: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    r=Depends(get_redis),
):
    """
    Uploads a CSV, creates a Job linked to an optional Case, saves to disk,
    and enqueues a work item to Redis.
    """
    _validate_csv_content_type(file.content_type)
    base_dir = Path(settings.storage_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc)
    case_uuid: Optional[UUID_t] = _parse_case_id(case_id)

    # If case supplied, verify it exists
    if case_uuid:
        case = db.get(Case, case_uuid)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

    # Create job row first to get its UUID
    new_job = Job(
        case_id=case_uuid,
        file_path="",
        sha256="",
        status="queued",
        created_at=ts,
        user_id=current_user.id,
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # Save file to disk: {storage_dir}/{case_id or uncategorized}/{job_id}/sample.csv
    case_folder = str(case_uuid) if case_uuid else "uncategorized"
    job_folder = base_dir / case_folder / str(new_job.id)
    job_folder.mkdir(parents=True, exist_ok=True)
    csv_path = job_folder / "sample.csv"

    # hash file while saving
    hasher = hashlib.sha256()
    total_bytes = 0
    try:
        with csv_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                hasher.update(chunk)
                total_bytes += len(chunk)
    except Exception as e:
        logger.error("Error saving uploaded file for job %s: %s", new_job.id, e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file") from e

    # Update job with file path and sha256
    sha256 = hasher.hexdigest()
    logger.info(
        "Saved uploaded file for job %s: path=%s bytes=%d sha256=%s",
        new_job.id, str(csv_path), total_bytes, sha256
    )

    new_job.file_path = str(csv_path)
    new_job.sha256 = sha256
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # Enqueue
    payload = {
        "job_id": str(new_job.id),
        "case_id": str(case_uuid) if case_uuid else None,
        "filepath": new_job.file_path,
        "sha256": new_job.sha256,
        "user_id": new_job.user_id,
        "created_at": new_job.created_at.isoformat(),
    }

    try:
        await r.lpush(settings.redis_queue, json.dumps(payload))
        logger.info("Enqueued job %s to Redis queue %s", new_job.id, settings.redis_queue)

    except Exception as e:
        logger.error("Failed to enqueue job %s to Redis: %s", new_job.id, e)
        raise HTTPException(status_code=500, detail="Failed to enqueue job") from e

    return JSONResponse({"status": "queued", "queue": settings.redis_queue, "job": new_job.to_dict()})


@router.get("/{job_id}")
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj)
):
    # validate job
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # non admin/researcher users can see only their jobs
    if current_user.role.value not in ("admin", "researcher") and job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return job.to_dict()


@router.get("")
def list_jobs(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    case_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
):
    """List jobs globally or within a case. Non-admin users see only their jobs."""
    q = db.query(Job)

    # find by case id
    if case_id:
        q = q.filter(Job.case_id == _parse_case_id(case_id))

    # find by user role
    if current_user.role.value not in ("admin", "researcher"):
        q = q.filter(Job.user_id == current_user.id)

    # pagination
    total = q.with_entities(func.count()).scalar() or 0
    jobs = q.order_by(Job.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()

    return {"page": page, "per_page": per_page, "total": total, "jobs": [j.to_dict() for j in jobs]}
