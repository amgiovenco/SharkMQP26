import csv
import io
import json
import hashlib
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from redis.asyncio import from_url

from .settings import settings
from .db import SessionLocal
from .models import Job
from .auth import get_current_user
from .logger import get_logger
from .socket_io import sio
from worker.extract_melt_block import process_file as convert_raw_csv

logger = get_logger(__name__)

router = APIRouter(tags=["jobs"])


async def emit_job_status(job_id: str, status: str):
    await sio.emit('job_status', {'job_id': job_id, 'status': status})


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
    return r


def _validate_csv_content_type(ct: Optional[str]) -> None:
    allowed = {
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    }
    if ct is None or ct.lower() not in allowed:
        logger.debug("Unrecognized CSV content-type: %s", ct)


@router.post("/upload")
async def upload_and_enqueue(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _sub: str = Depends(get_current_user),
    r=Depends(get_redis),
):
    """
    Upload a CSV, create Jobs for each sample, and enqueue to Redis.
    Stateless: no case, no user, no organization context.
    """
    _validate_csv_content_type(file.content_type)
    base_dir = Path(settings.storage_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc)
    original_filename = file.filename or "unknown.csv"
    batch_id = uuid.uuid4()

    # Save file: {storage_dir}/{batch_id}/sample.csv
    batch_folder = base_dir / str(batch_id)
    batch_folder.mkdir(parents=True, exist_ok=True)
    csv_path = batch_folder / "sample.csv"

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

    sha256 = hasher.hexdigest()
    logger.info("Saved uploaded file for batch %s: path=%s bytes=%d", batch_id, csv_path, total_bytes)

    # Convert raw CSV to inference format
    converted_csv_path = batch_folder / "sample_converted.csv"
    try:
        convert_raw_csv(str(csv_path), str(converted_csv_path))
        logger.info("Converted CSV for batch %s", batch_id)
    except Exception as e:
        logger.error("Failed to convert CSV for batch %s: %s", batch_id, e)
        raise HTTPException(status_code=500, detail="Failed to convert uploaded CSV") from e

    # Count samples
    import pandas as pd
    try:
        df_converted = pd.read_csv(converted_csv_path)
        num_samples = len(df_converted)
        logger.info("Batch %s contains %d samples", batch_id, num_samples)
    except Exception as e:
        logger.error("Failed to read converted CSV for batch %s: %s", batch_id, e)
        raise HTTPException(status_code=500, detail="Failed to read converted CSV") from e

    # Create jobs and enqueue
    job_ids = []
    try:
        for sample_idx in range(num_samples):
            new_job = Job(
                id=uuid.uuid4(),
                batch_id=batch_id,
                sample_index=sample_idx,
                file_path=str(converted_csv_path),
                original_filename=original_filename,
                sha256=sha256,
                status="queued",
                created_at=ts,
            )
            db.add(new_job)
            db.commit()
            db.refresh(new_job)
            job_ids.append(str(new_job.id))

            payload = {
                "job_id": str(new_job.id),
                "batch_id": str(batch_id),
                "sample_index": sample_idx,
                "filepath": str(converted_csv_path),
                "converted_filepath": str(converted_csv_path),
                "sha256": sha256,
                "created_at": new_job.created_at.isoformat(),
            }

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
    })


@router.get("/batch/{batch_id}/results/csv")
def download_batch_csv(
    batch_id: str,
    db: Session = Depends(get_db),
    _sub: str = Depends(get_current_user),
):
    """
    Download results for a batch as a CSV file.
    """
    try:
        batch_uuid = uuid.UUID(batch_id)
    except Exception:
        raise HTTPException(status_code=400, detail="batch_id must be a valid UUID")

    jobs = db.query(Job).filter(
        Job.batch_id == batch_uuid
    ).order_by(Job.sample_index.asc()).all()

    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found for this batch")

    # Build CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Header: sample_index, top_species, confidence, species_2, confidence_2, ...
    header = ["sample_index"]
    for i in range(1, 6):
        suffix = "" if i == 1 else f"_{i}"
        header.extend([f"species{suffix}", f"confidence{suffix}"])
    writer.writerow(header)

    for job in jobs:
        row = [job.sample_index]
        predictions = []
        if job.result_json and isinstance(job.result_json, dict):
            predictions = job.result_json.get("predictions", [])

        for i in range(5):
            if i < len(predictions):
                row.append(predictions[i].get("species", ""))
                conf = predictions[i].get("confidence", 0)
                row.append(f"{conf:.4f}" if isinstance(conf, (int, float)) else str(conf))
            else:
                row.extend(["", ""])

        writer.writerow(row)

    output.seek(0)

    filename = f"results_{batch_id[:8]}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
