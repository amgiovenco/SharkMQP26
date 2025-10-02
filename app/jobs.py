import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from redis.asyncio import from_url as redis_from_url

from .settings import settings
from .db import SessionLocal
from .models import Job, User
from .auth import get_current_user
from .logger import get_logger
from sqlalchemy import func

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_redis():
    r = redis_from_url(settings.redis_url, decode_responses=True)
    try:
        await r.ping()
    except Exception as e:
        logger.error("Redis ping failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Redis unavailable: {e}")
    logger.debug("Connected to Redis at %s", settings.redis_url)
    return r

def _validate_csv_content_type(ct: Optional[str]) -> None:
    # browser CSV types may vary
    allowed = {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    }
    if ct is None or ct.lower() not in allowed:
        # Don't hard fail
        logger.debug("Unrecognized CSV content-type: %s", ct)


@router.post("/upload")
async def upload_and_enqueue(
    file: UploadFile = File(...),
    case_id: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    r=Depends(get_redis),
):
    logger.info(
        "Upload request received: filename=%s content_type=%s case_id=%s user_id=%s",
        file.filename,
        file.content_type,
        case_id,
        getattr(current_user, "id", None),
    )

    _validate_csv_content_type(file.content_type)

    base_dir = Path(settings.storage_dir)
    ts = datetime.now(timezone.utc)

    new_job = Job(
        case_id=case_id,
        file_path="", # fill after we know where we wrote it
        sha256="", # fill after hashing
        status="queued",
        created_at=ts,
        user_id=current_user.id,
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job) # to obtain UUID primary key

    logger.debug("Created job record: id=%s user_id=%s status=%s", new_job.id, new_job.user_id, new_job.status)

    # save file to disk
    case_folder = case_id if case_id else "uncategorized"
    job_folder = base_dir / case_folder / str(new_job.id)
    job_folder.mkdir(parents=True, exist_ok=True)
    csv_path = job_folder / "sample.csv"

    # hash file while writing
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

    sha256 = hasher.hexdigest()
    logger.info("Saved uploaded file for job %s: path=%s bytes=%d sha256=%s", new_job.id, str(csv_path), total_bytes, sha256)

    # update job with path + sha256
    new_job.file_path = str(csv_path)
    new_job.sha256 = sha256
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # enqueue to redis
    payload = {
        "job_id": str(new_job.id), # UUID as string for portability
        "case_id": new_job.case_id, # may be None
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

    return JSONResponse(
        {
            "status": "queued",
            "queue": settings.redis_queue,
            "job": new_job.to_dict(),
        }
    )


@router.get("/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    logger.debug("Fetching job %s for user_id=%s", job_id, getattr(current_user, "id", None))
    job = db.get(Job, job_id)

    # verify job exists
    if not job:
        logger.warning("Job %s not found", job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    
    # verify ownership or admin/researcher
    if current_user.role.value not in ("admin", "researcher") and job.user_id != current_user.id:
        logger.warning("Forbidden access to job %s by user %s", job_id, current_user.id)
        raise HTTPException(status_code=403, detail="Forbidden")
    
    logger.debug("Job %s returned to user %s", job_id, current_user.id)
    return job.to_dict()

@router.get("/recent")
def get_recent_jobs(
    page: int = 1,
    per_page: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # basic validation / limits
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    if per_page < 1 or per_page > 100:
        raise HTTPException(status_code=400, detail="per_page must be between 1 and 100")

    q = db.query(Job)
    # non-admin/researcher users only see their own jobs
    if current_user.role.value not in ("admin", "researcher"):
        q = q.filter(Job.user_id == current_user.id)

    total = q.with_entities(func.count()).scalar() or 0

    jobs = (
        q.order_by(Job.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return JSONResponse(
        {
            "page": page,
            "per_page": per_page,
            "total": total,
            "jobs": [j.to_dict() for j in jobs],
        }
    )