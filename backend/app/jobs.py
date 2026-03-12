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
from .models import Job, User, Case, Organization, OrganizationMembership, JobResult
from .auth import get_current_user, get_current_user_obj, get_current_organization, get_current_membership
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
    current_org: Organization = Depends(get_current_organization),
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

    # Store the original filename
    original_filename = file.filename or "unknown.csv"

    # If case supplied, verify it exists and belongs to the organization
    if case_uuid:
        case = db.query(Case).filter(
            Case.id == case_uuid,
            Case.organization_id == current_org.id
        ).first()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

    # Create batch ID to group all samples from this upload
    batch_id = uuid.uuid4()

    # Save file to disk: {storage_dir}/{org_id}/{case_id or uncategorized}/{batch_id}/sample.csv
    org_folder = str(current_org.id)
    case_folder = str(case_uuid) if case_uuid else "uncategorized"
    batch_folder = base_dir / org_folder / case_folder / str(batch_id)
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
                original_filename=original_filename,
                sha256=sha256,
                status="queued",
                created_at=ts,
                user_id=current_user.id,
                organization_id=current_org.id,
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
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    # Verify job belongs to organization
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.organization_id == current_org.id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Members can only see their own jobs
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value == "member" and job.user_id != current_user.id:
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
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    """List jobs within the current organization. Members see only their jobs."""
    # Filter by organization
    q = db.query(Job).filter(Job.organization_id == current_org.id)

    # Find by batch id (all samples from same upload)
    if batch_id:
        try:
            batch_uuid = UUID_t(batch_id)
            q = q.filter(Job.batch_id == batch_uuid)
        except Exception:
            raise HTTPException(status_code=400, detail="batch_id must be a valid UUID")

    # Find by case id
    if case_id:
        case_uuid = _parse_case_id(case_id)
        # Verify case belongs to organization
        case = db.query(Case).filter(
            Case.id == case_uuid,
            Case.organization_id == current_org.id
        ).first()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        q = q.filter(Job.case_id == case_uuid)

    # Members only see their own jobs
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value == "member":
        q = q.filter(Job.user_id == current_user.id)

    # Pagination
    total = q.with_entities(func.count()).scalar() or 0
    # Order by batch_id and sample_index to keep batch samples together
    jobs = q.order_by(Job.batch_id.desc(), Job.sample_index.asc()).offset((page - 1) * per_page).limit(per_page).all()

    return {"page": page, "per_page": per_page, "total": total, "jobs": [j.to_dict(slim=True) for j in jobs]}

@router.delete("/{job_id}")
def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    """Delete a specific job. Only owners and admins can delete."""
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can delete jobs")

    # Verify job belongs to organization
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.organization_id == current_org.id
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    db.delete(job)
    db.commit()

    logger.info("Job deleted id=%s by user=%s org=%s", job_id, current_user.id, current_org.id)
    return {"detail": "Job deleted successfully"}

@router.delete("")
def delete_all_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
):
    """Delete all jobs in the organization. Only owners and admins can delete."""
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can delete jobs")

    # Delete all jobs in this organization
    deleted_count = db.query(Job).filter(
        Job.organization_id == current_org.id
    ).delete()

    db.commit()

    logger.warning(
        "All jobs deleted: org=%s user=%s deleted_count=%d",
        current_org.id,
        current_user.id,
        deleted_count
    )
    return {"detail": f"All jobs deleted successfully ({deleted_count} jobs)"}

@router.post("/{job_id}/rerun")
async def rerun_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
    r=Depends(get_redis),
):
    """
    Rerun an existing job by creating a new job with the same parameters.
    The new job will be enqueued for processing, creating a fresh analysis.
    """
    # Verify job belongs to organization
    original_job = db.query(Job).filter(
        Job.id == job_id,
        Job.organization_id == current_org.id
    ).first()

    if not original_job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Members can only rerun their own jobs
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value == "member" and original_job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Verify the file still exists
    file_path = Path(original_job.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Original file no longer exists, cannot rerun analysis"
        )

    try:
        # Create new job with same parameters but new ID
        new_job = Job(
            id=uuid.uuid4(),
            batch_id=original_job.batch_id,
            sample_index=original_job.sample_index,
            case_id=original_job.case_id,
            file_path=original_job.file_path,
            original_filename=original_job.original_filename,
            sha256=original_job.sha256,
            status="queued",
            created_at=datetime.now(timezone.utc),
            user_id=current_user.id,
            organization_id=current_org.id,
        )
        db.add(new_job)
        db.commit()
        db.refresh(new_job)

        # Create job payload
        payload = {
            "job_id": str(new_job.id),
            "batch_id": str(new_job.batch_id),
            "sample_index": new_job.sample_index,
            "case_id": str(new_job.case_id) if new_job.case_id else None,
            "filepath": str(new_job.file_path),
            "converted_filepath": str(new_job.file_path),
            "sha256": new_job.sha256,
            "user_id": new_job.user_id,
            "created_at": new_job.created_at.isoformat(),
        }

        # Enqueue to Redis
        await r.lpush(settings.redis_queue, json.dumps(payload))
        logger.info(
            "Rerun created job=%s from original=%s by user=%s org=%s",
            new_job.id,
            job_id,
            current_user.id,
            current_org.id
        )

        return {
            "status": "queued",
            "original_job_id": str(original_job.id),
            "new_job_id": str(new_job.id),
            "message": "Analysis queued for rerun"
        }

    except Exception as e:
        logger.error("Failed to rerun job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to rerun analysis: {str(e)}")

@router.post("/batch/{batch_id}/rerun")
async def rerun_batch(
    batch_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_obj),
    current_org: Organization = Depends(get_current_organization),
    membership: OrganizationMembership = Depends(get_current_membership),
    r=Depends(get_redis),
):
    """
    Rerun all jobs in a batch by creating new jobs for each sample.
    """
    try:
        batch_uuid = UUID_t(batch_id)
    except Exception:
        raise HTTPException(status_code=400, detail="batch_id must be a valid UUID")

    # Get all jobs in this batch
    jobs_query = db.query(Job).filter(
        Job.batch_id == batch_uuid,
        Job.organization_id == current_org.id
    )

    # Members can only rerun their own jobs
    role_value = membership.role.value if hasattr(membership.role, 'value') else membership.role
    if role_value == "member":
        jobs_query = jobs_query.filter(Job.user_id == current_user.id)

    original_jobs = jobs_query.order_by(Job.sample_index.asc()).all()

    if not original_jobs:
        raise HTTPException(status_code=404, detail="No jobs found in this batch")

    # Verify files still exist (check first job's file path)
    file_path = Path(original_jobs[0].file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Original files no longer exist, cannot rerun analysis"
        )

    new_job_ids = []
    try:
        for original_job in original_jobs:
            # Create new job
            new_job = Job(
                id=uuid.uuid4(),
                batch_id=original_job.batch_id,
                sample_index=original_job.sample_index,
                case_id=original_job.case_id,
                file_path=original_job.file_path,
                original_filename=original_job.original_filename,
                sha256=original_job.sha256,
                status="queued",
                created_at=datetime.now(timezone.utc),
                user_id=current_user.id,
                organization_id=current_org.id,
            )
            db.add(new_job)
            db.commit()
            db.refresh(new_job)
            new_job_ids.append(str(new_job.id))

            # Create job payload
            payload = {
                "job_id": str(new_job.id),
                "batch_id": str(new_job.batch_id),
                "sample_index": new_job.sample_index,
                "case_id": str(new_job.case_id) if new_job.case_id else None,
                "filepath": str(new_job.file_path),
                "converted_filepath": str(new_job.file_path),
                "sha256": new_job.sha256,
                "user_id": new_job.user_id,
                "created_at": new_job.created_at.isoformat(),
            }

            # Enqueue to Redis
            await r.lpush(settings.redis_queue, json.dumps(payload))

        logger.info(
            "Batch rerun created %d jobs from batch=%s by user=%s org=%s",
            len(new_job_ids),
            batch_id,
            current_user.id,
            current_org.id
        )

        return {
            "status": "queued",
            "batch_id": batch_id,
            "num_jobs": len(new_job_ids),
            "new_job_ids": new_job_ids,
            "message": f"Batch of {len(new_job_ids)} analyses queued for rerun"
        }

    except Exception as e:
        logger.error("Failed to rerun batch %s: %s", batch_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to rerun batch: {str(e)}")
