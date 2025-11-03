import json
import time
from datetime import datetime, timezone

from redis.asyncio import from_url
from sqlalchemy.orm import Session
import asyncio
import torch

from app.db import SessionLocal
from app.models import Job, JobResult
from app.settings import settings
from app.logger import get_logger
from worker.inference import run_inference as ml_inference

logger = get_logger(__name__)

# Determine device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_inference(filepath: str):
    """Run ML inference on the uploaded file"""
    try:
        result = ml_inference(filepath, device=DEVICE)
        return result
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise

def _db_session() -> Session:
    return SessionLocal()

def _mark_running(db: Session, job_id: str):
    now = datetime.now(timezone.utc)
    job = db.get(Job, job_id)
    if not job:
        return
    
    # job is now running
    job.status = "running"
    job.started_at = now

    db.add(job)
    db.commit()

def _mark_done(db: Session, job_id: str, result: dict):
    now = datetime.now(timezone.utc)
    job = db.get(Job, job_id)
    if not job:
        return

    # job is now done
    job.status = "completed"
    job.finished_at = now
    job.result_json = result
    db.add(job)

    # insert a JobResult
    jr = JobResult(job_id=job.id, result=result)

    db.add(jr)
    db.commit()

def _mark_error(db: Session, job_id: str, error_msg: str):
    now = datetime.now(timezone.utc)
    job = db.get(Job, job_id)
    if not job:
        return
    
    # job is now errored
    job.status = "error"
    job.finished_at = now
    job.result_json = {"error": error_msg}

    db.add(job)
    db.commit()

async def main():
    r = from_url(settings.redis_url, decode_responses=True)
    print(f"[worker] Connected to {settings.redis_url}; listening on '{settings.redis_queue}'")

    while True:
        job_id = None
        try:
            item = await r.blpop(settings.redis_queue, timeout=0)  # block forever
            if not item:
                continue

            # got a job
            _qname, payload = item
            job_msg = json.loads(payload)
            job_id = job_msg.get("job_id")
            filepath = job_msg.get("filepath")

            # update DB -> running
            db = _db_session()
            try:
                _mark_running(db, job_id)
            finally:
                db.close()

            # Run model
            result = run_inference(filepath)

            # update DB -> done
            db = _db_session()
            try:
                _mark_done(db, job_id, result)
            finally:
                db.close()

            # Publish completion status
            await r.publish('job_status_updates', json.dumps({
                'job_id': job_id,
                'status': 'completed',
                'result': result,
            }))
            
            logger.info(f"[worker] completed job {job_id}")

            print(f"[worker] completed job {job_id}")

        except Exception as e:
            if job_id:
                db = _db_session()
                try:
                    _mark_error(db, job_id, str(e))
                    
                    # Publish error status
                    await r.publish('job_status_updates', json.dumps({
                        'job_id': job_id,
                        'status': 'error',
                        'error': str(e),
                    }))
                finally:
                    db.close()
            
            logger.error(f"[worker] error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())