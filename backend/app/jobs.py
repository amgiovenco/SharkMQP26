import json
import hashlib
import uuid
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
from .socket_io import sio
from worker.extract_melt_block import process_file as convert_raw_csv

logger = get_logger(__name__)

router = APIRouter(tags=["jobs"])

# Helper to emit a job update to the frontend
async def emit_job_status(job_id: str, status: str):
    await sio.emit('job_status', {'job_id': job_id, 'status': status})

# Helper to get DB connection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper to get Redis connection
async def get_redis():
    r = from_url(settings.redis_url, decode_responses=True)
    try:
        await r.ping()
    except Exception as e:
        logger.error("Redis ping failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Redis unavailable: {e}")
    logger.debug("Connected to Redis at %s", settings.redis_url)
    return r

# Get the current user based on their jwt
def get_current_user_obj(username: str = Depends(get_current_user), db: Session = Depends(get_db)) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

# Helper to validate file upload type
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

# Parse case_id from string to UUID
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

    # Create batch ID to group all samples from this upload
    batch_id = uuid.uuid4()

    # Save file to disk: {storage_dir}/{case_id or uncategorized}/{batch_id}/sample.csv
    case_folder = str(case_uuid) if case_uuid else "uncategorized"
    batch_folder = base_dir / case_folder / str(batch_id)
    batch_folder.mkdir(parents=True, exist_ok=True)
    csv_path = batch_folder / "sample.csv"

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
        logger.error("Error saving uploaded file for batch %s: %s", batch_id, e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file") from e

    # Get file hash
    sha256 = hasher.hexdigest()
    logger.info(
        "Saved uploaded file for batch %s: path=%s bytes=%d sha256=%s",
        batch_id, str(csv_path), total_bytes, sha256
    )

    # Convert raw CSV to inference format
    converted_csv_path = batch_folder / "sample_converted.csv"
    try:
        convert_raw_csv(str(csv_path), str(converted_csv_path))
        logger.info("Converted CSV for batch %s: path=%s", batch_id, str(converted_csv_path))
    except Exception as e:
        logger.error("Failed to convert CSV for batch %s: %s", batch_id, e)
        raise HTTPException(status_code=500, detail="Failed to convert uploaded CSV") from e

    # Read converted CSV to determine number of samples
    import pandas as pd
    try:
        df_converted = pd.read_csv(converted_csv_path)
        num_samples = len(df_converted)
        logger.info("Batch %s contains %d samples", batch_id, num_samples)
    except Exception as e:
        logger.error("Failed to read converted CSV for batch %s: %s", batch_id, e)
        raise HTTPException(status_code=500, detail="Failed to read converted CSV") from e

    # Create individual jobs and queue entries for each sample
    job_ids = []
    try:
        for sample_idx in range(num_samples):
            # Create individual job for this sample
            new_job = Job(
                id=uuid.uuid4(),
                batch_id=batch_id,
                sample_index=sample_idx,
                case_id=case_uuid,
                file_path=str(converted_csv_path),
                sha256=sha256,
                status="queued",
                created_at=ts,
                user_id=current_user.id,
            )
            db.add(new_job)
            db.commit()
            db.refresh(new_job)
            job_ids.append(str(new_job.id))

            # Create job payload for this sample
            payload = {
                "job_id": str(new_job.id),
                "batch_id": str(batch_id),
                "sample_index": sample_idx,
                "case_id": str(case_uuid) if case_uuid else None,
                "filepath": str(converted_csv_path),
                "converted_filepath": str(converted_csv_path),
                "sha256": sha256,
                "user_id": new_job.user_id,
                "created_at": new_job.created_at.isoformat(),
            }

            # Enqueue to Redis
            await r.lpush(settings.redis_queue, json.dumps(payload))
            logger.info("Enqueued job %s (sample %d/%d) from batch %s", new_job.id, sample_idx, num_samples, batch_id)

        logger.info("Successfully created and queued %d jobs for batch %s", num_samples, batch_id)

    except Exception as e:
        logger.error("Failed to create/enqueue jobs for batch %s: %s", batch_id, e)
        raise HTTPException(status_code=500, detail="Failed to enqueue jobs") from e

    return JSONResponse({
        "status": "queued",
        "batch_id": str(batch_id),
        "num_samples": num_samples,
        "job_ids": job_ids,
        "queue": settings.redis_queue
    })


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
    batch_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
):
    """List jobs globally or within a case. Non-admin users see only their jobs."""
    q = db.query(Job)

    # find by batch id (all samples from same upload)
    if batch_id:
        try:
            batch_uuid = UUID_t(batch_id)
            q = q.filter(Job.batch_id == batch_uuid)
        except Exception:
            raise HTTPException(status_code=400, detail="batch_id must be a valid UUID")

    # find by case id
    if case_id:
        q = q.filter(Job.case_id == _parse_case_id(case_id))

    # find by user role
    if current_user.role.value not in ("admin", "researcher"):
        q = q.filter(Job.user_id == current_user.id)

    # pagination
    total = q.with_entities(func.count()).scalar() or 0
    # Order by batch_id and sample_index to keep batch samples together
    jobs = q.order_by(Job.batch_id.desc(), Job.sample_index.asc()).offset((page - 1) * per_page).limit(per_page).all()

    return {"page": page, "per_page": per_page, "total": total, "jobs": [j.to_dict() for j in jobs]}

@router.delete("/{job_id}")
def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
):
    """Delete a specific job. Only admins can delete."""
    if current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Only admins can delete jobs")

    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    db.delete(job)
    db.commit()

    return {"detail": "Job deleted successfully"}

@router.delete("")
def delete_all_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
):
    """Delete all jobs. Only admins can delete."""
    if current_user.role.value != "admin":
        raise HTTPException(status_code=403, detail="Only admins can delete jobs")

    db.query(Job).delete()
    db.commit()

    return {"detail": "All jobs deleted successfully"}
